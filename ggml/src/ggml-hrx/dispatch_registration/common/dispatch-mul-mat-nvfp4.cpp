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

// NVFP4 and Q2_0 prompt matmuls (256..2048 tokens in multiples of 256, input sizes in multiples of 256) on AMD's
// q8_1 x4 int8 WMMA kernel (ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256) at weight format 43 / 42:
// activations are quantized to q8_1 x4 once; NVFP4 blocks are staged as signed E2M1 codes with one scale per 16
// values, Q2_0 blocks as signed -1..2 under their block scale (motifs/nvfp4_q8_1_x4.loom). Without this, they take
// the generic f16 WMMA kernels (common.mul_mat.f32_f32_wmma, common.mul_mat_swiglu.f32_f32_wmma), which dequantize
// every weight to f16.

#include "dispatch-mul-mat-nvfp4.h"

#include "dispatch-mul-mat-common.h"

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kNvfp4Q8_1X4Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256");

static bool match_own_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context,
                                               DispatchMatch &              dispatch_match,
                                               ggml_type                    type,
                                               CommonMulMatWeightFormat     format) {
    if (context.root_node == nullptr || context.root_node->inputs.empty()) {
        return false;
    }
    const Value * weight = common_graph_value(context.graph, context.root_node->inputs[0]);
    if (weight == nullptr || weight->type != type) {
        return false;
    }
    const CommonMulMatMatch match =
        common_match_mul_mat_any_format(context.graph, context.root_node, kNvfp4Q8_1X4Kernel, false);
    if (!match.matched() || match.weight_format != format || match.token_count < 256 ||
        match.token_count > 2048 || match.token_count % 256 != 0 || match.input_size % 256 != 0 ||
        match.output_size % 64 != 0) {
        return false;
    }

    DispatchBinding activation;
    if (!common_prepare_q8_1_x4_input(context, *match.input, match.input_size, match.token_count, dispatch_match,
                                      activation, CommonQ8ActivationPolicy::AllowStandaloneQuantize)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kNvfp4Q8_1X4Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q8_1_x4.input_size",
                                               common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q8_1_x4.output_size",
                                               common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q8_1_x4.token_capacity",
                                               common_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_q8_1_x4.weight_format",
                                               common_to_config_value(common_mul_mat_format_config_value(format)));
    dispatch.bindings.push_back(activation);
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_nvfp4_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context,
                                                 DispatchMatch &              dispatch_match) {
    return match_own_q8_1_x4_prefill_dispatch(context, dispatch_match, GGML_TYPE_NVFP4,
                                              CommonMulMatWeightFormat::NVFP4);
}

static bool match_q2_0_q8_1_x4_prefill_dispatch(const DispatchMatchContext & context,
                                                DispatchMatch &              dispatch_match) {
    return match_own_q8_1_x4_prefill_dispatch(context, dispatch_match, GGML_TYPE_Q2_0,
                                              CommonMulMatWeightFormat::Q2_0);
}

}  // namespace

void register_nvfp4_prefill_dispatches(DispatchRegistryBuilder & registry) {
    // Ahead of the generic f16 WMMA matmuls (80) and their fused post-op forms (180, 290).
    registry.add({
        "common.mul_mat.nvfp4_q8_1_x4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_nvfp4_q8_1_x4_prefill_dispatch,
    });
    registry.add({
        "common.mul_mat.q2_0_q8_1_x4_prefill",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_q2_0_q8_1_x4_prefill_dispatch,
    });
}

}  // namespace ggml::hrx
