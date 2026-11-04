// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Standalone GGML_OP_SET_ROWS for HRX (engine #679, P2) -> hip_set_rows (hip/kernels/hip_set_rows.hip).
//
// The two Loom matchers that own this op today are the f16 element-wise scatter (i64 indices, f16 cache,
// hidden == 1) and the vectorised 2-D writer (i64 indices, hidden >= 4, <= 2048 tokens). Qwen3.8-Flash-Next's
// QSA mask rebuild is neither: build_attn_qsa fills [1, n_kv] with -inf and scatters the 2051 selected cells
// with **i32** indices into an **f32** cache, so the node was declined by both and ran on the CPU backend,
// splitting every full-attention layer's graph around it (measured in ~/wt/hrx-probe/qsa_layer.cpp: "unsupported
// HRX node: SET_ROWS output=f32[1,18816]" while the rest of the QSA decode path is on device).
//
// This matcher is registered at priority 50, i.e. BELOW common.set_rows_scatter (200) and common.set_rows (100),
// and fused registrations are tried before every single-op one: it only ever sees a node the existing kernels
// declined, so no shape that runs today changes backend. It also declines a quantized cache, which is the
// add-on's hook (hip_set_rows_hook in hip-dispatches.h).
//
// Dims 2 and 3 are handled by slicing rather than by the kernel: ggml_set_rows batches over them with the indices
// broadcast as dst_dim % indices_dim (ggml-cpu/ops.cpp set_rows), so the matcher emits one dispatch per slice with
// its own byte offsets into the rows, indices and output bindings. The model's QSA mask is one slice; the
// test-backend-ops cases that this kernel can take are 3 to 6.
//
// GGML_HRX_DISABLE_DISPATCH=hip.set_rows turns the matcher off, which makes the node fall back to the CPU
// backend through the same path -- that is the A/B switch for the model row.

#include "hip/hip-dispatches.h"
#include "hip/hip-kernel-registry.h"

#include "ggml.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

constexpr KernelCatalogRef kHipSetRowsKernel = hip_kernel_ref("hip_set_rows");

// Bounds the kernel's 32-bit launch parameters allow. Kept in one place so the matcher and the launch function
// cannot drift apart.
constexpr int64_t kSetRowsMaxTokens      = 1 << 20;
constexpr int64_t kSetRowsMaxHidden      = 1 << 15;
constexpr int64_t kSetRowsMaxCacheRows    = 1 << 27;
constexpr int64_t kSetRowsMaxWorkItems    = std::numeric_limits<uint32_t>::max();
constexpr int64_t kSetRowsMaxSlices       = 64;
constexpr uint32_t kSetRowsWorkgroupSize  = 256;
constexpr uint32_t kSetRowsVecWidth       = 4;

constexpr int64_t kSetRowsFormatF32 = 0;
constexpr int64_t kSetRowsFormatF16 = 1;
constexpr int64_t kSetRowsIndexI32  = 0;
constexpr int64_t kSetRowsIndexI64  = 1;

bool set_rows_element_format(ggml_type type, int64_t & format) {
    if (type == GGML_TYPE_F32) {
        format = kSetRowsFormatF32;
        return true;
    }
    if (type == GGML_TYPE_F16) {
        format = kSetRowsFormatF16;
        return true;
    }
    return false;
}

bool set_rows_index_format(ggml_type type, int64_t & format) {
    if (type == GGML_TYPE_I32) {
        format = kSetRowsIndexI32;
        return true;
    }
    if (type == GGML_TYPE_I64) {
        format = kSetRowsIndexI64;
        return true;
    }
    return false;
}

// The kernel indexes rows as `element * row_stride` with no per-dimension strides, so a packed row layout is
// part of the contract rather than an assumption; dims 2 and 3 are handled by slicing (see the matcher), not by
// the kernel, so their strides are free.
bool packed_row_stride(const Value & value, int64_t element_size, int64_t width) {
    return value.nb[0] == static_cast<size_t>(element_size) &&
           value.nb[1] == static_cast<size_t>(width) * static_cast<size_t>(element_size);
}

// 16-byte aligned rows let a thread write a whole 4-element packet in one instruction; anything else falls back
// to one element per thread per packet, which is also the element-wise (hidden == 1) shape.
uint32_t set_rows_packet_width(const Value & rows, const Value & cache, int64_t hidden_size) {
    const size_t rows_stride  = rows.nb[1];
    const size_t cache_stride = cache.nb[1];
    const bool   aligned = hidden_size % static_cast<int64_t>(kSetRowsVecWidth) == 0 &&
                         rows.storage_offset % 16 == 0 && cache.storage_offset % 16 == 0 &&
                         rows_stride % 16 == 0 && cache_stride % 16 == 0;
    return aligned ? kSetRowsVecWidth : 1;
}

bool launch_hip_set_rows(const std::map<std::string, int64_t> & parameters, HipLaunchGeometry & geometry) {
    const auto tokens  = parameters.find("token_count");
    const auto packets = parameters.find("packets");
    if (tokens == parameters.end() || packets == parameters.end() || tokens->second <= 0 || packets->second <= 0) {
        return false;
    }
    if (tokens->second > std::numeric_limits<int64_t>::max() / packets->second) {
        return false;
    }
    const int64_t work   = tokens->second * packets->second;
    const int64_t groups = (work + static_cast<int64_t>(kSetRowsWorkgroupSize) - 1) / static_cast<int64_t>(kSetRowsWorkgroupSize);
    if (groups <= 0 || groups > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    geometry.workgroup_count = { static_cast<uint32_t>(groups), 1, 1 };
    geometry.workgroup_size  = { kSetRowsWorkgroupSize, 1, 1 };
    return true;
}

bool match_hip_set_rows(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SET_ROWS || node->inputs.size() != 3) {
        return false;
    }
    const Value * rows    = context.graph.values().find(node->inputs[0]);
    const Value * indices = context.graph.values().find(node->inputs[1]);
    const Value * cache   = context.graph.values().find(node->inputs[2]);
    const Value * output  = context.graph.values().find(node->output);
    if (rows == nullptr || indices == nullptr || cache == nullptr || output == nullptr) {
        return false;
    }

    int64_t row_format   = 0;
    int64_t cache_format = 0;
    int64_t index_format = 0;
    if (!set_rows_element_format(rows->type, row_format) || !set_rows_element_format(cache->type, cache_format) ||
        !set_rows_index_format(indices->type, index_format)) {
        return false;
    }

    // the in-place contract: the output is a view of the cache it writes into, same type and shape
    if (output->type != cache->type || output->ne != cache->ne || !context.graph.values().same_storage(cache->id, output->id)) {
        return false;
    }
    if (!indices->contiguous || indices->nb[0] != static_cast<size_t>(ggml_type_size(indices->type))) {
        return false;
    }

    const int64_t hidden_size     = rows->ne[0];
    const int64_t token_count     = rows->ne[1];
    const int64_t cache_row_count = cache->ne[1];
    if (hidden_size != cache->ne[0] || token_count != indices->ne[0] || hidden_size < 1 ||
        hidden_size > kSetRowsMaxHidden || token_count < 1 || token_count > kSetRowsMaxTokens || cache_row_count < 1 ||
        cache_row_count > kSetRowsMaxCacheRows) {
        return false;
    }
    if (!packed_row_stride(*rows, ggml_type_size(rows->type), hidden_size) ||
        !packed_row_stride(*cache, ggml_type_size(cache->type), hidden_size)) {
        return false;
    }

    // ggml_set_rows batches over dims 2 and 3 with the indices broadcast as dst_dim % indices_dim (the CPU
    // reference, ggml-cpu/ops.cpp set_rows): one dispatch per (dim2, dim3) slice, each with its own byte offsets.
    // The model's QSA mask is a single slice; test-backend-ops' cases are 3-6 of them.
    const int64_t slices2 = cache->ne[2] % indices->ne[1];
    const int64_t slices3 = cache->ne[3] % indices->ne[2];
    if (cache->ne[2] < 1 || cache->ne[3] < 1 || slices2 != 0 || slices3 != 0 ||
        cache->ne[2] > kSetRowsMaxSlices / cache->ne[3]) {
        return false;
    }

    const uint32_t packet_width = set_rows_packet_width(*rows, *cache, hidden_size);
    const int64_t  packets      = hidden_size / static_cast<int64_t>(packet_width);
    if (packets <= 0 || token_count > static_cast<int64_t>(kSetRowsMaxWorkItems) / packets) {
        return false;
    }
    const size_t rows_slice_bytes = static_cast<size_t>(token_count) * static_cast<size_t>(hidden_size) *
                                    static_cast<size_t>(ggml_type_size(rows->type));
    const size_t cache_slice_bytes = static_cast<size_t>(cache_row_count) * static_cast<size_t>(hidden_size) *
                                     static_cast<size_t>(ggml_type_size(cache->type));
    const size_t index_slice_span = static_cast<size_t>(token_count) *
                                    static_cast<size_t>(ggml_type_size(indices->type));

    for (int64_t i03 = 0; i03 < cache->ne[3]; ++i03) {
        for (int64_t i02 = 0; i02 < cache->ne[2]; ++i02) {
            const size_t rows_offset = rows->storage_offset + i02 * rows->nb[2] + i03 * rows->nb[3];
            const size_t index_offset = (i02 % indices->ne[1]) * indices->nb[1] + (i03 % indices->ne[2]) * indices->nb[2];
            const size_t output_offset = i02 * cache->nb[2] + i03 * cache->nb[3];

            Dispatch dispatch;
            dispatch.kernel = make_kernel_specialization(kHipSetRowsKernel);
            dispatch.kernel.integer_parameters.emplace("token_count", token_count);
            dispatch.kernel.integer_parameters.emplace("packets", packets);
            dispatch.kernel.integer_parameters.emplace("packet_width", static_cast<int64_t>(packet_width));
            dispatch.kernel.integer_parameters.emplace("hidden_size", hidden_size);
            dispatch.kernel.integer_parameters.emplace("cache_row_count", cache_row_count);
            dispatch.kernel.integer_parameters.emplace("row_format", row_format);
            dispatch.kernel.integer_parameters.emplace("index_format", index_format);
            dispatch.kernel.integer_parameters.emplace("cache_format", cache_format);
            dispatch.bindings.push_back({ rows->storage_root, rows_offset, rows_slice_bytes });
            dispatch.bindings.push_back({ indices->id, index_offset, index_slice_span });
            dispatch.bindings.push_back({ output->id, output_offset, cache_slice_bytes });
            match.dispatches.push_back(std::move(dispatch));
        }
    }
    match.covered_nodes.push_back(context.root_index);
    return true;
}

}  // namespace

void register_hip_set_rows_dispatch(DispatchRegistryBuilder & registry) {
    if (!hip_code_object_embedded("hip_set_rows")) {
        return;
    }
    ggml::hrx::register_hip_kernel({
        "hip_set_rows",
        "hip_set_rows",
        { { "rows", ResourceAccess::Read }, { "indices", ResourceAccess::Read }, { "cache", ResourceAccess::Write } },
        { "token_count", "packets", "packet_width", "hidden_size", "cache_row_count", "row_format", "index_format",
          "cache_format" },  // by-value kernel arguments, in order
        { "token_count", "packets" },  // read by launch_hip_set_rows to size the grid
        launch_hip_set_rows,
    });
    // 50: below common.set_rows_scatter (200) and common.set_rows (100) on purpose -- see the header comment.
    registry.add({ "hip.set_rows", GGML_OP_SET_ROWS, DispatchMatchKind::SingleOp, 50, DispatchSource::Common,
                   match_hip_set_rows });
}

}  // namespace ggml::hrx
