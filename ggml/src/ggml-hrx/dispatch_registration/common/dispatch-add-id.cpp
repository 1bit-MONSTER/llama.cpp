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

// GGML_OP_ADD_ID on F32 (ops/add_id_f32.loom): gpt-oss adds a per-expert bias to each MUL_MAT_ID output row,
// output[t][r] = input[t][r] + bias[ids[t][r]]. Without this every MoE layer leaves HRX three times.

#include "dispatch-add-id.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kAddIdF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_add_id_f32");

bool flat_3d(const Value & value) {
    return value.ne[3] == 1;
}

bool match_add_id_f32(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 3) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * bias   = context.graph.values().find(node->inputs[1]);
    const Value * ids    = context.graph.values().find(node->inputs[2]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || bias == nullptr || ids == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        bias->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || output->type != GGML_TYPE_F32 ||
        !input->contiguous || !bias->contiguous || !output->contiguous || !flat_3d(*input) || !flat_3d(*output)) {
        return false;
    }
    const int64_t width        = input->ne[0];
    const int64_t rows         = input->ne[1];
    const int64_t tokens       = input->ne[2];
    const int64_t expert_count = bias->ne[1];
    if (output->ne[0] != width || output->ne[1] != rows || output->ne[2] != tokens || bias->ne[0] != width ||
        bias->ne[2] != 1 || bias->ne[3] != 1 || ids->ne[0] != rows || ids->ne[1] != tokens || ids->ne[2] != 1 ||
        ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0) {
        return false;
    }
    const int64_t ids_stride = tokens > 1 ? static_cast<int64_t>(ids->nb[1] / sizeof(int32_t)) : rows;
    // the kernel's launch ranges
    if (width < 1 || width > 1048576 || rows < 1 || rows > 4096 || tokens < 1 || tokens > 65536 ||
        ids_stride < rows || ids_stride > 4096 || expert_count < 1 || expert_count > 4096) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddIdF32Kernel);
    dispatch.kernel.integer_parameters.emplace("width", width);
    dispatch.kernel.integer_parameters.emplace("rows", rows);
    dispatch.kernel.integer_parameters.emplace("tokens", tokens);
    dispatch.kernel.integer_parameters.emplace("ids_stride", ids_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", expert_count);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ bias->id, 0, bias->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_add_id_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "extra.add_id_f32",
        GGML_OP_ADD_ID,
        DispatchMatchKind::SingleOp,
        1,
        DispatchSource::Common,
        match_add_id_f32,
    });
}

}  // namespace ggml::hrx
