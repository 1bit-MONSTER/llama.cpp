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

// UPSCALE, GGML_SCALE_MODE_BILINEAR without antialias, f32, over dims 0 and 1 (ops/upscale_bilinear_f32.loom).
// engine#3549: a vision projector resizes its position embeddings with BILINEAR | ALIGN_CORNERS; with no HRX
// UPSCALE the node ran on the CPU and strict placement aborted the image encode.

#include "dispatch-upscale.h"

#include "dispatch-mul-mat-common.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kUpscaleBilinearF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_upscale_bilinear_f32");
static constexpr uint64_t         kMaximumElements          = uint64_t{ 1 } << 30;

bool match_upscale_bilinear_f32(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }
    const UpscaleParams * params = op_params_as<UpscaleParams>(node->params);
    if (params == nullptr) {
        return false;
    }
    const int32_t mode  = params->mode_flags & 0xFF;
    const int32_t flags = params->mode_flags & ~0xFF;
    if (mode != GGML_SCALE_MODE_BILINEAR || (flags & ~static_cast<int32_t>(GGML_SCALE_FLAG_ALIGN_CORNERS)) != 0) {
        return false;  // nearest, bicubic and antialias stay on the CPU
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !output->contiguous || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > kMaximumElements || input->ne[2] != output->ne[2] ||
        input->ne[3] != output->ne[3]) {
        return false;
    }
    // the source in elements: positive whole-element strides and the span the largest index reaches
    uint64_t span = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (input->ne[i] <= 0 || input->nb[i] == 0 || input->nb[i] % sizeof(float) != 0) {
            return false;
        }
        span += static_cast<uint64_t>(input->ne[i] - 1) * (input->nb[i] / sizeof(float));
    }
    if (span > kMaximumElements) {
        return false;
    }
    // the scale factors and pixel offset exactly as ggml_compute_forward_upscale_f32 computes them
    const int64_t ne0 = output->ne[0], ne1 = output->ne[1], ne00 = input->ne[0], ne01 = input->ne[1];
    float         sf0 = static_cast<float>(ne0) / static_cast<float>(ne00);
    float         sf1 = static_cast<float>(ne1) / static_cast<float>(ne01);
    float         po  = 0.5f;
    if (flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        po  = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? static_cast<float>(ne0 - 1) / static_cast<float>(ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? static_cast<float>(ne1 - 1) / static_cast<float>(ne01 - 1) : sf1;
    }
    Dispatch          dispatch;
    const std::string k = "ggml.upscale_bilinear_f32.";
    dispatch.kernel     = make_kernel_specialization(kUpscaleBilinearF32Kernel);
    dispatch.kernel.compile_parameters.emplace(k + "ne0", std::to_string(ne0));
    dispatch.kernel.compile_parameters.emplace(k + "ne1", std::to_string(ne1));
    dispatch.kernel.compile_parameters.emplace(k + "ne2", std::to_string(output->ne[2]));
    dispatch.kernel.compile_parameters.emplace(k + "src_ne0", std::to_string(ne00));
    dispatch.kernel.compile_parameters.emplace(k + "src_ne1", std::to_string(ne01));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.compile_parameters.emplace(k + "stride" + std::to_string(i),
                                                   std::to_string(input->nb[i] / sizeof(float)));
    }
    dispatch.kernel.compile_parameters.emplace(k + "sf0", common_to_config_value(sf0));
    dispatch.kernel.compile_parameters.emplace(k + "sf1", common_to_config_value(sf1));
    dispatch.kernel.compile_parameters.emplace(k + "pixel_offset", common_to_config_value(po));
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", static_cast<int64_t>(span));
    dispatch.bindings.push_back({ input->storage_root, input->storage_offset, static_cast<size_t>(span) * sizeof(float) });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_upscale_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.upscale_bilinear_f32",
        GGML_OP_UPSCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_upscale_bilinear_f32,
    });
}

}  // namespace ggml::hrx
