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

// IM2COL, 2-D, f32 image -> f16 columns (ops/im2col_f32_f16.loom). engine#3549: a vision projector's patch embedding is
// ggml_conv_2d with an f16 kernel, whose IM2COL ran on the CPU, so strict placement aborted the image encode.

#include "dispatch-im2col.h"

#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kIm2colF32F16Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_im2col_f32_f16");
static constexpr uint64_t         kMaximumElements    = uint64_t{ 1 } << 30;

bool match_im2col_f32_f16(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 2) {
        return false;
    }
    const Im2colParams * params = op_params_as<Im2colParams>(node->params);
    if (params == nullptr || params->is_2d != 1 || params->s0 < 1 || params->s1 < 1 || params->d0 < 1 ||
        params->d1 < 1 || params->p0 < 0 || params->p1 < 0) {
        return false;
    }
    const Value * kernel = context.graph.values().find(node->inputs[0]);  // its shape only: KW, KH
    const Value * image  = context.graph.values().find(node->inputs[1]);  // [IW, IH, IC, N]
    const Value * output = context.graph.values().find(node->output);     // [IC*KH*KW, OW, OH, N]
    if (kernel == nullptr || image == nullptr || output == nullptr || image->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F16 || !output->contiguous || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > kMaximumElements || image->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t kw = kernel->ne[0], kh = kernel->ne[1];
    const int64_t iw = image->ne[0], ih = image->ne[1], ic = image->ne[2], n = image->ne[3];
    const int64_t ow = output->ne[1], oh = output->ne[2];
    if (kw <= 0 || kh <= 0 || output->ne[0] != ic * kh * kw || output->ne[3] != n || ow <= 0 || oh <= 0) {
        return false;
    }
    uint64_t span = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (image->ne[i] <= 0 || image->nb[i] == 0 || image->nb[i] % sizeof(float) != 0) {
            return false;
        }
        span += static_cast<uint64_t>(image->ne[i] - 1) * (image->nb[i] / sizeof(float));
    }
    if (span > kMaximumElements) {
        return false;
    }
    Dispatch          dispatch;
    const std::string k = "ggml.im2col_f32_f16.";
    dispatch.kernel     = make_kernel_specialization(kIm2colF32F16Kernel);
    const auto set      = [&](const char * name, int64_t value) {
        dispatch.kernel.compile_parameters.emplace(k + name, std::to_string(value));
    };
    set("kw", kw);
    set("kh", kh);
    set("ic", ic);
    set("ow", ow);
    set("oh", oh);
    set("iw", iw);
    set("ih", ih);
    set("s0", params->s0);
    set("s1", params->s1);
    set("p0", params->p0);
    set("p1", params->p1);
    set("d0", params->d0);
    set("d1", params->d1);
    set("stride_h", static_cast<int64_t>(image->nb[1] / sizeof(float)));
    set("stride_c", static_cast<int64_t>(image->nb[2] / sizeof(float)));
    set("stride_n", static_cast<int64_t>(image->nb[3] / sizeof(float)));
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", static_cast<int64_t>(span));
    dispatch.bindings.push_back({ image->storage_root, image->storage_offset, static_cast<size_t>(span) * sizeof(float) });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_im2col_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.im2col_f32_f16",
        GGML_OP_IM2COL,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_im2col_f32_f16,
    });
}

}  // namespace ggml::hrx
