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

// GGML_GLU_OP_SWIGLU_OAI on F32 with gate and up as two contiguous tensors of the same shape
// (ops/swiglu_oai_f32.loom): gpt-oss's clamped SwiGLU between the expert up/gate and down projections.

#include "dispatch-swiglu-oai.h"

#include "dispatch-mul-mat-common.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kSwigluOaiF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_swiglu_oai_f32");

bool same_shape(const Value & a, const Value & b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a.ne[i] != b.ne[i]) {
            return false;
        }
    }
    return true;
}

bool match_swiglu_oai_f32(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 2) {
        return false;
    }
    const GluParams * params = op_params_as<GluParams>(node->params);
    if (params == nullptr || params->op != GGML_GLU_OP_SWIGLU_OAI) {
        return false;
    }
    const Value * gate   = context.graph.values().find(node->inputs[0]);
    const Value * up     = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    if (gate == nullptr || up == nullptr || output == nullptr || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !gate->contiguous || !up->contiguous ||
        !output->contiguous || !same_shape(*gate, *up) || !same_shape(*gate, *output) || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > (uint64_t{ 1 } << 27)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSwigluOaiF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.swiglu_oai.alpha", common_to_config_value(params->alpha));
    dispatch.kernel.compile_parameters.emplace("ggml.swiglu_oai.limit", common_to_config_value(params->limit));
    dispatch.bindings.push_back({ gate->id, 0, gate->byte_count });
    dispatch.bindings.push_back({ up->id, 0, up->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_swiglu_oai_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "extra.swiglu_oai_f32",
        GGML_OP_GLU,
        DispatchMatchKind::SingleOp,
        1,
        DispatchSource::Common,
        match_swiglu_oai_f32,
    });
}

}  // namespace ggml::hrx
