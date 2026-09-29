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

// MUL_MAT_ID for a few tokens (decode) as a GEMV over the selected experts' rows
// (ops/mul_mat_id_decode_f32.loom), ahead of the WMMA mul_mat_id kernel, which is built for
// batches and reads a single token's expert at ~45 GB/s. Larger batches still take the WMMA path.

#include "dispatch-mul-mat-id-decode.h"

#include "dispatch-mul-mat-id-common.h"

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kMulMatIdDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_decode_f32_wave64");

// The GEMV pays off while token * slot rows are few; past this the WMMA kernel's reuse wins.
constexpr int64_t kMaximumDecodeTokens = 4;
constexpr int64_t kMaximumDecodeRows   = 64;

bool match_mul_mat_id_decode_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatIdMatch match = common_match_mul_mat_id_any_format(context.graph, context.root_node, context.plan);
    if (!match.matched() || match.token_count < 1 || match.token_count > kMaximumDecodeTokens ||
        match.token_count * match.route_count > kMaximumDecodeRows || match.route_stride < match.route_count ||
        match.input_size < 256 || match.input_size > 32768 || match.input_size % 32 != 0 ||
        match.output_size < 1 || match.expert_count < 1 ||
        (match.input_route_count != 1 && match.input_route_count != match.route_count)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kMulMatIdDecodeKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("slot_count", match.route_count);
    dispatch.kernel.integer_parameters.emplace("input_rows", match.input_route_count);
    dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
    dispatch.kernel.integer_parameters.emplace("expert_count", match.expert_count);
    dispatch.kernel.integer_parameters.emplace("route_stride", match.route_stride);
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_decode.row_capacity",
                                               common_mul_mat_id_to_config_value(match.token_count * match.route_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_decode.output_capacity",
                                               common_mul_mat_id_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_id_decode.weight_format",
        common_mul_mat_id_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.route_ids->id, 0, match.route_ids->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_mul_mat_id_decode_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat_id.decode_f32_wave64",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::SingleOp,
        150,
        DispatchSource::Common,
        match_mul_mat_id_decode_dispatch,
    });
}

}  // namespace ggml::hrx
