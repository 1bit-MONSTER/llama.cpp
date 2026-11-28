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
#include "dispatch-pad.h"
#include "dispatch-layout-utils.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kPadTailKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_pad_tail_f32");

// the kernel's index.assume bounds both counts to range(1, 2^30), exclusive
static constexpr int64_t kPadTailMaxElements = int64_t{ 1 } << 30;

static bool packed_f32(const Value & value) {
    size_t stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] != stride) {
            return false;
        }
        stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

// engine#3577: an f32 PAD with a packed output is "the source's elements in order, then zeros" exactly when every left
// pad is 0, no pad is circular, the axes below some axis k are unchanged, axis k grows at its end, and every axis above
// k is 1 in both. The no-KV-cache attention path pads K and V that way (along the token axis, k = 2); V is a strided
// view of the fused QKV output there, so the source may have any f32 element strides (read through them by the kernel).
static bool pad_is_tail(const PadParams & p, const Value & source, const Value & output) {
    if (p.circular != 0 || p.lp0 != 0 || p.lp1 != 0 || p.lp2 != 0 || p.lp3 != 0) {
        return false;
    }
    const int32_t right[GGML_MAX_DIMS] = { p.rp0, p.rp1, p.rp2, p.rp3 };
    int k = -1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (right[i] < 0 || output.ne[i] != source.ne[i] + right[i]) {
            return false;
        }
        if (right[i] > 0) {
            if (k >= 0) {
                return false;   // a second padded axis interleaves zeros with the source
            }
            k = i;
        }
    }
    if (k < 0) {
        return true;            // no padding at all: a copy
    }
    for (int i = k + 1; i < GGML_MAX_DIMS; ++i) {
        if (source.ne[i] != 1 || output.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool match_pad_tail_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_PAD || node->inputs.size() != 1) {
        return false;
    }
    const PadParams * params = op_params_as<PadParams>(node->params);
    const Value *     source = context.graph.values().find(node->inputs[0]);
    const Value *     output = context.graph.values().find(node->output);
    if (params == nullptr || source == nullptr || output == nullptr || source->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !packed_f32(*output) ||
        output->alias_source.value >= 0 || source->storage == output->storage ||
        source->element_count <= 0 || output->element_count < source->element_count ||
        output->element_count >= kPadTailMaxElements || !pad_is_tail(*params, *source, *output)) {
        return false;
    }
    size_t source_span = 0;
    if (!strided_f32_storage_span_bytes(*source, source_span) ||
        static_cast<int64_t>(source_span / sizeof(float)) >= kPadTailMaxElements) {
        return false;
    }
    // the kernel declares every stride as range(1, 2^30): a zero stride (a broadcast view) or a larger one would fail the
    // kernel's compile instead of leaving the node to another route (review cl-flm)
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const size_t stride = source->nb[i] / sizeof(float);
        if (stride < 1 || static_cast<int64_t>(stride) >= kPadTailMaxElements) {
            return false;
        }
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kPadTailKernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_count", source->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", static_cast<int64_t>(source_span / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.pad_tail_f32.ne0", std::to_string(source->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.pad_tail_f32.ne1", std::to_string(source->ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.pad_tail_f32.ne2", std::to_string(source->ne[2]));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.compile_parameters.emplace("ggml.pad_tail_f32.stride" + std::to_string(i),
                                                   std::to_string(source->nb[i] / sizeof(float)));
    }
    // the source's whole strided span, from its storage root (a view's own id would bind only its byte_count)
    dispatch.bindings.push_back({ source->storage_root, source->storage_offset, source_span });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_pad_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({ "common.pad_tail_f32", GGML_OP_PAD, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_pad_tail_f32 });
}

}  // namespace ggml::hrx
