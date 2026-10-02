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

#include "dispatch-mul-mat-tail.h"

#include "graph/op-params.h"

#include <string>
#include <utility>

namespace ggml::hrx {

namespace {

constexpr KernelCatalogRef kMulMatTailKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_f32_f32_wmma");

}  // namespace

bool common_append_mul_mat_token_tail(const CommonMulMatMatch & match,
                                      int64_t                   head_tokens,
                                      DispatchMatch &           dispatch_match) {
    const int64_t tail_tokens = match.token_count - head_tokens;
    if (head_tokens <= 0 || tail_tokens < 2) {
        return false;
    }
    const size_t input_row  = static_cast<size_t>(match.input_size) * sizeof(float);
    const size_t output_row = static_cast<size_t>(match.output_size) * sizeof(float);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatTailKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", tail_tokens);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity", common_to_config_value(tail_tokens));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.input_size", common_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_size", common_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_accumulation", "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.output_unary_op",
                                               std::to_string(unary_kind_config_value(UnaryKind::Identity)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat.weight_format",
        common_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, static_cast<size_t>(head_tokens) * input_row,
                                  static_cast<size_t>(tail_tokens) * input_row });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, static_cast<size_t>(head_tokens) * output_row,
                                  static_cast<size_t>(tail_tokens) * output_row });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace ggml::hrx
