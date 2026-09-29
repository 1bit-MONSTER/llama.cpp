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

// ZAYA's residual scale, residual = (x + bx) * sx + (r + br) * sr, in two dispatches
// (ops/res_scale_pair_f32.loom) instead of up to five ADD / MUL dispatches, twice per layer. At
// decode each of those is ~1.3 us of work plus a ~1.8 us gap. The graph computes the r side
// long before x exists, so one dispatch cannot cover both: each side becomes (a + b) * s where
// it starts, and the side that comes last also takes the final ADD, reading the other side's
// result as an addend. A bias ADD an earlier fusion already took (the output projection's
// bias) stays there. Every removed intermediate has exactly one consumer.

#include "dispatch-res-scale-pair.h"

#include "dispatch-mul-mat-common.h"
#include "graph/graph-matcher.h"

#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kResScalePairKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_res_scale_pair_f32");

// A full activation: packed f32, [row_size, rows] with nothing beyond dim 1.
bool is_activation(const Graph & graph, const Value * value) {
    if (value == nullptr || value->type != GGML_TYPE_F32 || !value->contiguous || value->ne[0] < 1 ||
        value->ne[2] != 1 || value->ne[3] != 1 || value->element_count != value->ne[0] * value->ne[1]) {
        return false;
    }
    if (value->alias_source.value < 0) {
        return true;
    }
    const GraphNode * producer = graph.index().producer(value->id);
    return value->storage_offset == 0 && producer != nullptr && producer->op == GGML_OP_RESHAPE;
}

// The output may reuse x's or r's storage (each element is read and written by one thread at
// one index), but only exactly in place: a shifted overlap would race.
bool in_place_or_disjoint(const Value & source, const Value & output) {
    if (source.storage != output.storage || source.storage_offset == output.storage_offset) {
        return true;
    }
    return source.storage_offset + source.byte_count <= output.storage_offset ||
           output.storage_offset + output.byte_count <= source.storage_offset;
}

// A per-channel parameter: packed f32 [row_size], broadcast over rows.
bool is_channel_parameter(const Value * value, int64_t row_size) {
    return value != nullptr && value->type == GGML_TYPE_F32 && value->contiguous && value->ne[0] == row_size &&
           value->element_count == row_size && value->alias_source.value < 0;
}

// node = BINARY(activation, parameter) with the activation as input 0 (the order ZAYA builds).
bool split_scaled(const Graph & graph, const GraphNode & node, int64_t row_size, const Value *& activation,
                  const Value *& parameter) {
    activation = common_graph_value(graph, node.inputs[0]);
    parameter  = common_graph_value(graph, node.inputs[1]);
    return is_activation(graph, activation) && activation->ne[0] == row_size &&
           is_channel_parameter(parameter, row_size);
}

bool covered(const DispatchMatchContext & context, const GraphNode * node) {
    size_t index = 0;
    return node != nullptr && context.graph.index().node_index(node, index) && index < context.covered_nodes.size() &&
           context.covered_nodes[index];
}

// Rooted at a bias ADD (a + b, feeding only a MUL) or at the MUL (a * s) itself.
bool match_res_scale_pair(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const Graph &     graph = context.graph;
    const GraphNode * root  = context.root_node;
    if (root == nullptr || !graph.has_index()) {
        return false;
    }

    std::vector<const GraphNode *> nodes;
    const Value *                  a    = nullptr;
    const Value *                  bias = nullptr;
    const Value *                  s    = nullptr;
    const GraphNode *              mul  = root;
    if (common_binary_node_is_add(*root)) {
        const Value * out = common_graph_value(graph, root->output);
        mul               = common_find_only_consumer_with_op(graph, root->output, GGML_OP_MUL);
        if (out == nullptr || mul == nullptr || mul->inputs[0] != root->output ||
            !split_scaled(graph, *root, out->ne[0], a, bias)) {
            return false;
        }
        nodes.push_back(root);
    }
    const Value * mul_out = common_graph_value(graph, mul->output);
    const Value * scaled  = nullptr;
    if (mul_out == nullptr || !common_binary_node_is_mul(*mul) || !split_scaled(graph, *mul, mul_out->ne[0], scaled, s)) {
        return false;
    }
    if (a == nullptr) {
        a = scaled;
    }
    nodes.push_back(mul);

    // Take the final ADD when the other side is already computed (its producer covered, or a leaf).
    const Value *     output = mul_out;
    const Value *     addend = nullptr;
    const GraphNode * sum    = common_find_only_consumer_with_op(graph, mul->output, GGML_OP_ADD);
    if (sum != nullptr && common_binary_node_is_add(*sum)) {
        const ValueId other_id = sum->inputs[0] == mul->output ? sum->inputs[1] : sum->inputs[0];
        const Value * other    = common_graph_value(graph, other_id);
        const Value * sum_out  = common_graph_value(graph, sum->output);
        const GraphNode * other_producer = graph.index().producer(other_id);
        if (other != nullptr && sum_out != nullptr && other_id != mul->output &&
            (other_producer == nullptr || covered(context, other_producer)) && is_activation(graph, other) &&
            is_activation(graph, sum_out) && sum_out->alias_source.value < 0 && common_same_shape(*other, *sum_out) &&
            common_same_shape(*sum_out, *a) && in_place_or_disjoint(*other, *sum_out)) {
            addend = other;
            output = sum_out;
            nodes.push_back(sum);
        }
    }
    if (!is_activation(graph, output) || output->alias_source.value >= 0 || !common_same_shape(*output, *a) ||
        !in_place_or_disjoint(*a, *output)) {
        return false;
    }
    // Nothing to gain from a lone MUL.
    if (nodes.size() < 2) {
        return false;
    }
    for (const GraphNode * node : nodes) {
        if (!append_covered_node_index_once(graph, context.covered_nodes, node, dispatch_match.covered_nodes)) {
            dispatch_match.covered_nodes.clear();
            return false;
        }
    }

    const auto source = [](const Value * value) {
        return DispatchBinding{ value->storage_root, value->storage_offset, value->byte_count };
    };
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kResScalePairKernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("row_size", output->ne[0]);
    dispatch.kernel.compile_parameters.emplace("ggml.res_scale_pair_f32.has_bias", bias != nullptr ? "1" : "0");
    dispatch.kernel.compile_parameters.emplace("ggml.res_scale_pair_f32.has_addend", addend != nullptr ? "1" : "0");
    dispatch.bindings.push_back(source(a));
    dispatch.bindings.push_back(source(bias != nullptr ? bias : s));  // unread without has_bias
    dispatch.bindings.push_back(source(s));
    dispatch.bindings.push_back(source(addend != nullptr ? addend : a));  // unread without has_addend
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_res_scale_pair_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.res_scale_pair.add_root_f32",
        GGML_OP_ADD,
        DispatchMatchKind::Fused,
        250,
        DispatchSource::Common,
        match_res_scale_pair,
    });
    registry.add({
        "common.res_scale_pair.mul_root_f32",
        GGML_OP_MUL,
        DispatchMatchKind::Fused,
        250,
        DispatchSource::Common,
        match_res_scale_pair,
    });
}

}  // namespace ggml::hrx
