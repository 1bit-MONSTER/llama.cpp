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

// Projections for one token (decode) on Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_NL, IQ4_XS and Q8_0
// weights, read in their GGUF block layout (ops/kquant_decode_f32.loom): FFN gate/up pairs fused
// with SwiGLU, and plain projections with an optional following residual ADD. Mixed-quant models
// (Unsloth UD-Q4_K_XL and similar) pair these types freely per layer; without this they take the
// generic dequantize-4-values-at-a-time kernels.

#include "dispatch-kquant-decode.h"

#include "dispatch-mul-mat-common.h"
#include "graph/graph-matcher.h"

#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kKQuantSwiGLUDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_swiglu_decode_f32");
static constexpr KernelCatalogRef kKQuantMulMatDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_mul_mat_decode_f32");

bool kquant_format(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::Q3K:
        case CommonMulMatWeightFormat::Q4K:
        case CommonMulMatWeightFormat::Q5K:
        case CommonMulMatWeightFormat::Q6K:
        case CommonMulMatWeightFormat::IQ3_S:
        case CommonMulMatWeightFormat::IQ4_NL:
        case CommonMulMatWeightFormat::IQ4_XS:
        case CommonMulMatWeightFormat::Q8_0:
            return true;
        default:
            return false;
    }
}

// The kernels' config ranges (ops/kquant_decode_f32.loom).
constexpr int64_t kMaxInputSize  = 65536;
constexpr int64_t kMaxOutputSize = 1048576;

bool uncovered(const DispatchMatchContext & context, const GraphNode * node) {
    size_t index = 0;
    return context.graph.index().node_index(node, index) && index < context.covered_nodes.size() &&
           !context.covered_nodes[index];
}

bool match_kquant_swiglu_decode(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const Graph & graph = context.graph;
    if (!graph.has_index()) {
        return false;
    }
    const CommonMulMatMatch root =
        common_match_mul_mat_any_format(graph, context.root_node, kKQuantSwiGLUDecodeKernel, true);
    if (!root.matched() || root.token_count != 1 || !kquant_format(root.weight_format) ||
        root.input_size % 256 != 0 || root.input_size > kMaxInputSize || root.output_size > kMaxOutputSize) {
        return false;
    }

    // The root's only consumer is a SwiGLU whose other operand is a matching MUL_MAT of the same input.
    const std::vector<const GraphNode *> & root_consumers = graph.index().consumers(context.root_node->output);
    const GraphNode * glu = root_consumers.size() == 1 ? root_consumers.front() : nullptr;
    BinaryKind        op;
    if (glu == nullptr || glu->inputs.size() != 2 || !common_fused_binary_kind_from_params(glu->params, op) ||
        op != BinaryKind::SwiGLU || !uncovered(context, glu)) {
        return false;
    }
    const bool root_is_gate = glu->inputs[0] == context.root_node->output;
    if (!root_is_gate && glu->inputs[1] != context.root_node->output) {
        return false;
    }
    const ValueId     peer_id = root_is_gate ? glu->inputs[1] : glu->inputs[0];
    const GraphNode * peer    = graph.index().producer(peer_id);
    if (peer == nullptr || peer == context.root_node || !uncovered(context, peer) ||
        graph.index().consumers(peer_id).size() != 1) {
        return false;
    }
    const CommonMulMatMatch other = common_match_mul_mat_any_format(graph, peer, kKQuantSwiGLUDecodeKernel, true);
    if (!other.matched() || !kquant_format(other.weight_format) || other.input->id != root.input->id ||
        other.input_size != root.input_size || other.output_size != root.output_size || other.token_count != 1) {
        return false;
    }
    const Value * output = common_graph_value(graph, glu->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !common_same_shape(*output, *root.output)) {
        return false;
    }

    const CommonMulMatMatch & gate = root_is_gate ? root : other;
    const CommonMulMatMatch & up   = root_is_gate ? other : root;
    for (const GraphNode * node : { context.root_node, peer, glu }) {
        if (!append_covered_node_index_once(graph, context.covered_nodes, node, dispatch_match.covered_nodes)) {
            dispatch_match.covered_nodes.clear();
            return false;
        }
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kKQuantSwiGLUDecodeKernel);
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_swiglu_decode.input_size",
                                               std::to_string(root.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_swiglu_decode.output_size",
                                               std::to_string(root.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.kquant_swiglu_decode.gate_weight_format",
        std::to_string(common_mul_mat_format_config_value(gate.weight_format)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.kquant_swiglu_decode.up_weight_format",
        std::to_string(common_mul_mat_format_config_value(up.weight_format)));
    dispatch.bindings.push_back({ root.input->id, 0, root.input->byte_count });
    dispatch.bindings.push_back({ gate.weight->id, 0, gate.weight->byte_count });
    dispatch.bindings.push_back({ up.weight->id, 0, up.weight->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}


bool match_kquant_mul_mat_decode(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const Graph & graph = context.graph;
    if (!graph.has_index()) {
        return false;
    }
    const CommonMulMatMatch root =
        common_match_mul_mat_any_format(graph, context.root_node, kKQuantMulMatDecodeKernel, true);
    if (!root.matched() || root.token_count != 1 || !kquant_format(root.weight_format) ||
        root.input_size % 256 != 0 || root.input_size > kMaxInputSize || root.output_size > kMaxOutputSize) {
        return false;
    }

    // Fold a following residual ADD (the projection's only consumer) into the store.
    const Value *     addend = nullptr;
    const Value *     output = root.output;
    const GraphNode * add    = common_find_only_consumer_with_op(graph, root.output->id, GGML_OP_ADD);
    if (add != nullptr && common_binary_node_is_add(*add) && uncovered(context, add)) {
        const bool    root_is_lhs = add->inputs[0] == root.output->id;
        const Value * other       = common_graph_value(graph, root_is_lhs ? add->inputs[1] : add->inputs[0]);
        const Value * sum         = common_graph_value(graph, add->output);
        if ((root_is_lhs || add->inputs[1] == root.output->id) && other != nullptr && sum != nullptr &&
            other->type == GGML_TYPE_F32 && sum->type == GGML_TYPE_F32 && other->contiguous && sum->contiguous &&
            common_same_shape(*root.output, *other) && common_same_shape(*root.output, *sum)) {
            addend = other;
            output = sum;
        } else {
            add = nullptr;
        }
    } else {
        add = nullptr;
    }

    if (!append_covered_node_index_once(graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        (add != nullptr &&
         !append_covered_node_index_once(graph, context.covered_nodes, add, dispatch_match.covered_nodes))) {
        dispatch_match.covered_nodes.clear();
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kKQuantMulMatDecodeKernel);
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.input_size",
                                               std::to_string(root.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.output_size",
                                               std::to_string(root.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.kquant_mul_mat_decode.weight_format",
        std::to_string(common_mul_mat_format_config_value(root.weight_format)));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.add", addend != nullptr ? "1" : "0");
    dispatch.bindings.push_back({ root.input->id, 0, root.input->byte_count });
    dispatch.bindings.push_back({ root.weight->id, 0, root.weight->byte_count });
    // Without an ADD the addend is never read; bind the input in its place.
    const Value * addend_binding = addend != nullptr ? addend : root.input;
    dispatch.bindings.push_back({ addend_binding->id, 0, addend_binding->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_kquant_decode_dispatches(DispatchRegistryBuilder & registry) {
    // Above common.mul_mat_swiglu.symmetric_i4_lowrow_adjacent_dual (310): that path repacks the
    // weights to int4 at first use and quantizes activations to int4, and under llama-server it
    // made 27B decode 10.7 tok/s against 11.9 with this kernel.
    registry.add({
        "kquant.swiglu.decode_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        315,
        DispatchSource::Common,
        match_kquant_swiglu_decode,
    });
    // Above the generic common.mul_mat f32 matchers (80/70/60), below every specialized one.
    registry.add({
        "kquant.mul_mat.decode_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        85,
        DispatchSource::Common,
        match_kquant_mul_mat_decode,
    });
}

}  // namespace ggml::hrx
