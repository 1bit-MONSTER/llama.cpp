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

// ZAYA's CCA convolution at decode (one token, one sequence) as one dispatch
// (ops/zaya_cca_conv_decode_f32.loom) instead of the 13 that its ~35 graph nodes take. Matched from
// the Q/K concat, following the exact decode structure of src/models/zaya.cpp:
//
//   QKraw = CONCAT(Qraw, Kraw) -> reshape(s) -> conv_input = CONCAT(conv_state, .)       [3, C]
//   conv_input -> VIEW (steps 1..2) -> CONT -> RESHAPE -> CPY into the conv-state cache
//   conv_input -> SSM_CONV(dw) -> ADD(dw bias) = QK_dw                                     [C, 2]
//   per tap t: QK_dw -> VIEW(step t) -> PERMUTE -> CONT -> RESHAPE -> MUL_MAT(W_t, .)      [128, 1, G]
//   ADD(tap 0, tap 1) -> RESHAPE -> PERMUTE -> CONT -> RESHAPE -> ADD(grp bias) = QK_grp  [C]
//
// Anything else (prefill shapes, several sequences, other layouts) is left to the generic path.

#include "dispatch-zaya-cca-conv.h"

#include "dispatch-mul-mat-common.h"
#include "graph/graph-matcher.h"

#include <utility>
#include <vector>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kZayaCcaConvKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_zaya_cca_conv_decode_f32");

constexpr int64_t kGroupSize = 128;

struct Walk {
    const Graph &                  graph;
    std::vector<const GraphNode *> nodes;

    const Value * value(ValueId id) const { return common_graph_value(graph, id); }

    // The only consumer of |id|, when there is exactly one.
    const GraphNode * only_consumer(ValueId id) const {
        const std::vector<const GraphNode *> & consumers = graph.index().consumers(id);
        return consumers.size() == 1 ? consumers.front() : nullptr;
    }

    // Follow single-consumer layout aliases (reshape/view/permute) from |id|; returns the last value.
    ValueId skip_aliases(ValueId id) {
        for (;;) {
            const GraphNode * next = only_consumer(id);
            if (next == nullptr || !is_layout_alias_node(graph, *next)) {
                return id;
            }
            nodes.push_back(next);
            id = next->output;
        }
    }

    // |id|'s only consumer, which must be |op| with |id| as input |slot|.
    const GraphNode * expect(ValueId id, ggml_op op, size_t slot) {
        const GraphNode * next = only_consumer(id);
        if (next == nullptr || next->op != op || next->inputs.size() <= slot || next->inputs[slot] != id) {
            return nullptr;
        }
        nodes.push_back(next);
        return next;
    }
};

bool is_f32(const Value * v, int64_t ne0, int64_t ne1) {
    return v != nullptr && v->type == GGML_TYPE_F32 && v->contiguous && v->ne[0] == ne0 && v->ne[1] == ne1 &&
           v->ne[2] == 1 && v->ne[3] == 1;
}

// One tap: QK_dw -> VIEW -> PERMUTE -> CONT -> RESHAPE -> MUL_MAT(W view, .). Returns the MUL_MAT
// and its weight view, and the step the view reads (its byte offset over QK_dw's row stride).
const GraphNode * match_tap(Walk & walk, const GraphNode * view, const Value & qk_dw, int64_t channels,
                            int64_t & step, const Value *& weight_view) {
    const Value * v = walk.value(view->output);
    if (v == nullptr || !is_layout_alias_node(walk.graph, *view) || v->ne[0] != kGroupSize ||
        v->ne[1] != channels / kGroupSize || v->storage_offset < qk_dw.storage_offset ||
        (v->storage_offset - qk_dw.storage_offset) % qk_dw.nb[1] != 0) {
        return nullptr;
    }
    step = static_cast<int64_t>((v->storage_offset - qk_dw.storage_offset) / qk_dw.nb[1]);
    walk.nodes.push_back(view);
    const GraphNode * permute = walk.expect(view->output, GGML_OP_PERMUTE, 0);
    const GraphNode * cont    = permute != nullptr ? walk.expect(permute->output, GGML_OP_CONT, 0) : nullptr;
    if (cont == nullptr) {
        return nullptr;
    }
    const ValueId     x   = walk.skip_aliases(cont->output);
    const GraphNode * mul = walk.expect(x, GGML_OP_MUL_MAT, 1);
    if (mul == nullptr) {
        return nullptr;
    }
    weight_view = walk.value(mul->inputs[0]);
    return mul;
}

bool match_zaya_cca_conv(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const Graph &     graph = context.graph;
    const GraphNode * root  = context.root_node;
    if (root == nullptr || !graph.has_index() || root->op != GGML_OP_CONCAT || root->inputs.size() != 2) {
        return false;
    }
    Walk          walk{ graph, { root } };
    const Value * q  = walk.value(root->inputs[0]);
    const Value * k  = walk.value(root->inputs[1]);
    const Value * qk = walk.value(root->output);
    if (q == nullptr || k == nullptr || !is_f32(q, q->ne[0], 1) || !is_f32(k, k->ne[0], 1)) {
        return false;
    }
    const int64_t channels = q->ne[0] + k->ne[0];
    if (!is_f32(qk, channels, 1) || channels % kGroupSize != 0 || channels < kGroupSize || channels > 8192) {
        return false;
    }

    // conv_input = CONCAT(conv_state, QKraw reshaped to [1, C]): [3, C].
    const ValueId     qk_col = walk.skip_aliases(root->output);
    const GraphNode * cat    = walk.expect(qk_col, GGML_OP_CONCAT, 1);
    if (cat == nullptr) {
        return false;
    }
    const Value * state = walk.value(cat->inputs[0]);
    const Value * input = walk.value(cat->output);
    if (state == nullptr || state->type != GGML_TYPE_F32 || !state->contiguous || state->ne[0] != 2 ||
        state->ne[1] != channels || state->element_count != 2 * channels || input == nullptr ||
        input->ne[0] != 3 || input->ne[1] != channels || input->ne[2] != 1 || input->ne[3] != 1) {
        return false;
    }

    // Its two consumers: the state-update view and the depthwise conv.
    const std::vector<const GraphNode *> & input_users = graph.index().consumers(cat->output);
    if (input_users.size() != 2) {
        return false;
    }
    const GraphNode * state_view = nullptr;
    const GraphNode * conv       = nullptr;
    for (const GraphNode * user : input_users) {
        if (user->op == GGML_OP_SSM_CONV && user->inputs.size() == 2 && user->inputs[0] == cat->output) {
            conv = user;
        } else if (is_layout_alias_node(graph, *user)) {
            state_view = user;
        }
    }
    if (conv == nullptr || state_view == nullptr) {
        return false;
    }

    // State update: VIEW (steps 1..2) -> CONT -> RESHAPE -> CPY into the cache.
    const Value * last_states = walk.value(state_view->output);
    if (last_states == nullptr || last_states->ne[0] != 2 || last_states->ne[1] != channels ||
        last_states->storage_offset != input->storage_offset + input->nb[0]) {
        return false;
    }
    walk.nodes.push_back(state_view);
    const GraphNode * state_cont = walk.expect(state_view->output, GGML_OP_CONT, 0);
    if (state_cont == nullptr) {
        return false;
    }
    const GraphNode * state_copy = walk.expect(walk.skip_aliases(state_cont->output), GGML_OP_CPY, 0);
    const Value *     new_state  = state_copy != nullptr ? walk.value(state_copy->output) : nullptr;
    if (new_state == nullptr || new_state->type != GGML_TYPE_F32 || !new_state->contiguous ||
        new_state->element_count != 2 * channels) {
        return false;
    }

    // Depthwise conv + bias.
    walk.nodes.push_back(conv);
    const Value *     dw       = walk.value(conv->inputs[1]);
    const GraphNode * dw_add   = walk.expect(conv->output, GGML_OP_ADD, 0);
    if (dw == nullptr || dw->type != GGML_TYPE_F32 || !dw->contiguous || dw->ne[0] != 2 || dw->ne[1] != channels ||
        dw_add == nullptr || !common_binary_node_is_add(*dw_add)) {
        return false;
    }
    const Value * dw_bias = walk.value(dw_add->inputs[1]);
    const Value * qk_dw   = walk.value(dw_add->output);
    if (dw_bias == nullptr || dw_bias->type != GGML_TYPE_F32 || !dw_bias->contiguous ||
        dw_bias->element_count != channels || !is_f32(qk_dw, channels, 2)) {
        return false;
    }

    // Two taps, one per step, each into one grouped matmul.
    const std::vector<const GraphNode *> & tap_views = graph.index().consumers(dw_add->output);
    if (tap_views.size() != 2) {
        return false;
    }
    const GraphNode * tap_mul[2]    = { nullptr, nullptr };
    const Value *     tap_weight[2] = { nullptr, nullptr };
    for (const GraphNode * view : tap_views) {
        int64_t           step        = -1;
        const Value *     weight_view = nullptr;
        const GraphNode * mul         = match_tap(walk, view, *qk_dw, channels, step, weight_view);
        if (mul == nullptr || step < 0 || step > 1 || tap_mul[step] != nullptr) {
            return false;
        }
        tap_mul[step]    = mul;
        tap_weight[step] = weight_view;
    }

    // Both weight views are slices of one F16 [128, C, 2] tensor: tap t at t * nb[2].
    const Value * weight = tap_weight[0] != nullptr ? walk.value(tap_weight[0]->storage_root) : nullptr;
    if (weight == nullptr || tap_weight[1] == nullptr || weight->type != GGML_TYPE_F16 || !weight->contiguous ||
        weight->ne[0] != kGroupSize || weight->ne[1] != channels || weight->ne[2] != 2 || weight->ne[3] != 1 ||
        tap_weight[1]->storage_root != tap_weight[0]->storage_root ||
        tap_weight[0]->storage_offset != weight->storage_offset ||
        tap_weight[1]->storage_offset != weight->storage_offset + weight->nb[2] ||
        tap_weight[0]->ne[0] != kGroupSize || tap_weight[0]->ne[1] != kGroupSize ||
        tap_weight[0]->ne[2] != channels / kGroupSize) {
        return false;
    }

    // ADD(tap 0, tap 1) -> RESHAPE -> PERMUTE -> CONT -> RESHAPE -> ADD(grp bias).
    const GraphNode * sum = walk.only_consumer(tap_mul[0]->output);
    if (sum == nullptr || !common_binary_node_is_add(*sum) || sum->inputs[0] != tap_mul[0]->output ||
        sum->inputs[1] != tap_mul[1]->output || walk.only_consumer(tap_mul[1]->output) != sum) {
        return false;
    }
    walk.nodes.push_back(sum);
    const ValueId     sum_view = walk.skip_aliases(sum->output);
    const GraphNode * sum_cont = walk.expect(sum_view, GGML_OP_CONT, 0);
    if (sum_cont == nullptr) {
        return false;
    }
    const GraphNode * bias_add = walk.expect(walk.skip_aliases(sum_cont->output), GGML_OP_ADD, 0);
    if (bias_add == nullptr || !common_binary_node_is_add(*bias_add)) {
        return false;
    }
    const Value * grp_bias = walk.value(bias_add->inputs[1]);
    const Value * output   = walk.value(bias_add->output);
    if (grp_bias == nullptr || grp_bias->type != GGML_TYPE_F32 || !grp_bias->contiguous ||
        grp_bias->element_count != channels || !is_f32(output, channels, 1) || output->alias_source.value >= 0) {
        return false;
    }

    for (const GraphNode * node : walk.nodes) {
        if (!append_covered_node_index_once(graph, context.covered_nodes, node, dispatch_match.covered_nodes)) {
            dispatch_match.covered_nodes.clear();
            return false;
        }
    }

    const auto source = [](const Value * v) {
        return DispatchBinding{ v->storage_root, v->storage_offset, v->byte_count };
    };
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kZayaCcaConvKernel);
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_conv.channels", std::to_string(channels));
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_conv.q_size", std::to_string(q->ne[0]));
    dispatch.bindings.push_back(source(q));
    dispatch.bindings.push_back(source(k));
    dispatch.bindings.push_back(source(state));
    dispatch.bindings.push_back(source(dw));
    dispatch.bindings.push_back(source(dw_bias));
    dispatch.bindings.push_back({ weight->storage_root, weight->storage_offset, weight->byte_count });
    dispatch.bindings.push_back(source(grp_bias));
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch.bindings.push_back({ new_state->id, 0, new_state->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_zaya_cca_conv_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "zaya.cca_conv.decode_f32",
        GGML_OP_CONCAT,
        DispatchMatchKind::Fused,
        500,
        DispatchSource::Common,
        match_zaya_cca_conv,
    });
}

}  // namespace ggml::hrx
