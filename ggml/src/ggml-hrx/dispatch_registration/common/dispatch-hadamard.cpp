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

// A MUL_MAT that llama marks GGML_HINT_SRC0_IS_HADAMARD (src0 is the normalized n x n Sylvester
// matrix) as one Walsh-Hadamard transform per row (ops/hadamard_f32.loom), as the CPU, Vulkan,
// CUDA and Metal backends do for that hint. The dense matmul matchers stop at 2048 rows, so
// Bonsai's prompt-time rotations (512 tokens x 17 blocks of 1024) used to run on the CPU in every
// layer: pp512 13 tok/s. The matrix is not read; only its size picks the transform.

#include "dispatch-hadamard.h"

#include "dispatch-mul-mat-common.h"
#include "graph/graph-matcher.h"

#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kHadamardKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_hadamard_f32");
constexpr int32_t kHintSrc0IsHadamard = 1;  // GGML_HINT_SRC0_IS_HADAMARD

bool packed_f32_rows(const Value * value, int64_t n) {
    return value != nullptr && value->type == GGML_TYPE_F32 && value->contiguous && value->ne[0] == n &&
           value->ne[2] == 1 && value->ne[3] == 1 && value->element_count == value->ne[0] * value->ne[1];
}

int log2_exact(int64_t n) {
    int k = 0;
    while ((int64_t{ 1 } << k) < n) {
        ++k;
    }
    return (int64_t{ 1 } << k) == n ? k : -1;
}

// Each workgroup reads its whole row before it writes, so the output may be the input itself,
// but not a shifted overlap of it.
bool in_place_or_disjoint(const Value & input, const Value & output) {
    if (input.storage != output.storage || input.storage_offset == output.storage_offset) {
        return true;
    }
    return input.storage_offset + input.byte_count <= output.storage_offset ||
           output.storage_offset + output.byte_count <= input.storage_offset;
}

bool match_hadamard(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return false;
    }
    const MulMatParams * params = op_params_as<MulMatParams>(node->params);
    if (params == nullptr || params->hint != kHintSrc0IsHadamard) {
        return false;
    }
    const Value * rotation = common_graph_value(context.graph, node->inputs[0]);
    const Value * input    = common_graph_value(context.graph, node->inputs[1]);
    const Value * output   = common_graph_value(context.graph, node->output);
    if (rotation == nullptr || rotation->ne[0] != rotation->ne[1] || rotation->ne[2] != 1 || rotation->ne[3] != 1) {
        return false;
    }
    const int64_t n = rotation->ne[0];
    const int     k = log2_exact(n);
    if (k < 6 || k > 12 || !packed_f32_rows(input, n) || !packed_f32_rows(output, n) ||
        output->ne[1] != input->ne[1] || output->ne[1] < 1 || !in_place_or_disjoint(*input, *output)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kHadamardKernel);
    dispatch.kernel.integer_parameters.emplace("row_count", input->ne[1]);
    dispatch.kernel.compile_parameters.emplace("ggml.hadamard_f32.log2_block", std::to_string(k));
    dispatch.bindings.push_back({ input->storage_root, input->storage_offset, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_hadamard_dispatches(DispatchRegistryBuilder & registry) {
    // Fused, not SingleOp: the registry tries every Fused matcher of an op before any SingleOp
    // one, and the dense MUL_MAT matchers are Fused, so as SingleOp a hinted MUL_MAT of up to
    // 2048 rows (Bonsai decode: 17) went to a dense WMMA kernel (1e-2 off the exact product).
    // Priority 400 is above every dense MUL_MAT matcher (the highest is 315).
    registry.add({
        "common.hadamard_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        400,
        DispatchSource::Common,
        match_hadamard,
    });
}

}  // namespace ggml::hrx
