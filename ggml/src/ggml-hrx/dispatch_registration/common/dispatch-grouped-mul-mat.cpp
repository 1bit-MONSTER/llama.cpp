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
#include "dispatch-grouped-mul-mat.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kGroupedMulMatF16F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_grouped_mul_mat_f16_f32");

// ne[0..2] as given, ne[3] == 1, and densely packed with the given element size
static bool packed_3d(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, size_t element_size) {
    if (value.ne[0] != ne0 || value.ne[1] != ne1 || value.ne[2] != ne2 || value.ne[3] != 1) {
        return false;
    }
    return value.nb[0] == element_size && value.nb[1] == element_size * static_cast<size_t>(ne0) &&
           value.nb[2] == value.nb[1] * static_cast<size_t>(ne1);
}

static bool match_grouped_mul_mat_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return false;
    }
    const Value * weight = context.graph.values().find(node->inputs[0]);
    const Value * input  = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return false;
    }
    if (weight->type != GGML_TYPE_F16 || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = weight->ne[0];
    const int64_t n = weight->ne[1];
    const int64_t g = weight->ne[2];
    const int64_t m = input->ne[1];
    // only the batched case: 2-D weights belong to the regular MUL_MAT kernels
    if (g < 2 || g > 4096 || k < 1 || k > 65536 || n < 1 || n > 65536 || m < 1 || m > 65535) {
        return false;
    }
    if (!packed_3d(*weight, k, n, g, sizeof(uint16_t)) || !packed_3d(*input, k, m, g, sizeof(float)) ||
        !packed_3d(*output, n, m, g, sizeof(float)) || output->alias_source.value >= 0 ||
        input->storage == output->storage || weight->storage == output->storage) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGroupedMulMatF16F32Kernel);
    dispatch.kernel.integer_parameters.emplace("input_size", k);
    dispatch.kernel.integer_parameters.emplace("output_size", n);
    dispatch.kernel.integer_parameters.emplace("token_count", m);
    dispatch.kernel.integer_parameters.emplace("group_count", g);
    dispatch.bindings.push_back({ weight->id, 0, weight->byte_count });
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_grouped_mul_mat_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.grouped_mul_mat_f16_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_grouped_mul_mat_dispatch,
    });
}

}  // namespace ggml::hrx
