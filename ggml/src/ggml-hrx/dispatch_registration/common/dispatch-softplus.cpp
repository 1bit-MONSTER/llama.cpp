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

// UNARY softplus on contiguous F32 values (ops/softplus_f32.loom). common.unary_f32 covers the
// exact-math unary kinds only; without this, a standalone softplus (Qwen3.5/3.8 delta-net gates in
// multi-sequence batches) is claimed by HRX and then fails as an unsupported node.

#include "dispatch-softplus.h"

#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kSoftplusF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_softplus_f32");

bool match_softplus_f32(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }
    const UnaryParams * params = op_params_as<UnaryParams>(node->params);
    if (params == nullptr || params->op != UnaryKind::SoftPlus) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input->contiguous || !output->contiguous || input->element_count != output->element_count ||
        output->element_count <= 0 || static_cast<uint64_t>(output->element_count) > (uint64_t{ 1 } << 27)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSoftplusF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_softplus_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "extra.softplus_f32",
        GGML_OP_UNARY,
        DispatchMatchKind::SingleOp,
        1,
        DispatchSource::Common,
        match_softplus_f32,
    });
}

}  // namespace ggml::hrx
