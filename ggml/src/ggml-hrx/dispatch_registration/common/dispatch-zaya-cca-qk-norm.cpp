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

// ZAYA's CCA query/key mixing and normalization at decode (one token) as one dispatch
// (ops/zaya_cca_qk_norm_decode_f32.loom), between the CCA convolution and RoPE. Matched from the
// copy of the convolution output's query slice, following src/models/zaya.cpp:
//
//   Qcur_pre_rope = RMS_NORM(C_q + SCALE(Qpre + REPEAT(Kpre), 0.5))
//   Kcur_pre_rope = RMS_NORM(C_k + SCALE(SCALE(SUM_ROWS(CONT(PERMUTE(Qgroup))), 1/gqa) + Kpre, 0.5))
//                   * k_scale
//
// with C_q, C_k the query and key slices of the convolution output and Qpre, Kpre the plain Q/K
// projections. Every node of both chains comes after the convolution in graph order, so the
// dispatch sits at its first node with all inputs ready and both outputs still ahead of RoPE.

#include "dispatch-zaya-cca-qk-norm.h"

#include "dispatch-mul-mat-common.h"
#include "graph/graph-matcher.h"
#include "graph/op-params.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kZayaCcaQkNormKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_zaya_cca_qk_norm_decode_f32");

struct Chain {
    const Graph &                  graph;
    std::vector<const GraphNode *> nodes;

    const Value * value(ValueId id) const { return common_graph_value(graph, id); }

    const GraphNode * only_consumer(ValueId id) const {
        const std::vector<const GraphNode *> & consumers = graph.index().consumers(id);
        return consumers.size() == 1 ? consumers.front() : nullptr;
    }

    // Forward through single-consumer layout aliases.
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

    // Backward through layout aliases to the value they view.
    ValueId source(ValueId id) {
        for (;;) {
            const GraphNode * producer = graph.index().producer(id);
            if (producer == nullptr || !is_layout_alias_node(graph, *producer)) {
                return id;
            }
            nodes.push_back(producer);
            id = producer->inputs[0];
        }
    }

    // The non-alias producer of |id| (after skipping aliases), which must be |op| and consumed only
    // along this chain.
    const GraphNode * producer(ValueId id, ggml_op op) {
        const ValueId     base = source(id);
        const GraphNode * p    = graph.index().producer(base);
        if (p == nullptr || p->op != op || only_consumer(base) == nullptr) {
            return nullptr;
        }
        nodes.push_back(p);
        return p;
    }
};

bool scale_is(const GraphNode * node, float value) {
    const ScaleParams * params = node != nullptr ? op_params_as<ScaleParams>(node->params) : nullptr;
    return params != nullptr && params->bias == 0.0f && std::fabs(params->scale - value) <= 1e-7f * std::fabs(value);
}

bool is_rows(const Value * v, int64_t ne0, int64_t ne1) {
    return v != nullptr && v->type == GGML_TYPE_F32 && v->contiguous && v->ne[0] == ne0 && v->ne[1] == ne1 &&
           v->ne[2] == 1 && v->ne[3] == 1;
}

// |add| = ADD(conv slice, SCALE(mean, 0.5)); returns the SCALE's input (the mean's sum).
const GraphNode * mean_sum(Chain & chain, const GraphNode * add, ValueId conv_slice) {
    if (add == nullptr || !common_binary_node_is_add(*add) || add->inputs[0] != conv_slice) {
        return nullptr;
    }
    const GraphNode * half = chain.producer(add->inputs[1], GGML_OP_SCALE);
    if (!scale_is(half, 0.5f)) {
        return nullptr;
    }
    const GraphNode * sum = chain.producer(half->inputs[0], GGML_OP_ADD);
    return sum != nullptr && common_binary_node_is_add(*sum) ? sum : nullptr;
}

bool match_zaya_cca_qk_norm(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const Graph &     graph = context.graph;
    const GraphNode * root  = context.root_node;
    if (root == nullptr || !graph.has_index() || root->op != GGML_OP_CONT || root->inputs.size() != 1) {
        return false;
    }
    Chain chain{ graph, { root } };

    // The convolution output C and its two slices: query [0, q_size), key [q_size, q_size + k_size).
    const GraphNode * q_slice = graph.index().producer(root->inputs[0]);
    if (q_slice == nullptr || !is_layout_alias_node(graph, *q_slice)) {
        return false;
    }
    const ValueId conv_id = q_slice->inputs[0];
    const Value * conv    = chain.value(conv_id);
    const Value * q_view  = chain.value(root->inputs[0]);
    if (conv == nullptr || q_view == nullptr || conv->type != GGML_TYPE_F32 || !conv->contiguous ||
        conv->ne[1] != 1 || conv->ne[2] != 1 || conv->ne[3] != 1 || q_view->storage_offset != conv->storage_offset) {
        return false;
    }
    const std::vector<const GraphNode *> & slices = graph.index().consumers(conv_id);
    if (slices.size() != 2) {
        return false;
    }
    const GraphNode * k_slice = slices[0] == q_slice ? slices[1] : slices[0];
    const Value *     k_view  = chain.value(k_slice->output);
    if (!is_layout_alias_node(graph, *k_slice) || k_view == nullptr ||
        k_view->storage_offset != conv->storage_offset + q_view->byte_count ||
        q_view->element_count + k_view->element_count != conv->element_count) {
        return false;
    }

    // Query side: CONT -> reshape [D, n_head] -> ADD(., mean) -> RMS_NORM.
    const ValueId     cq     = chain.skip_aliases(root->output);
    const Value *     cq_val = chain.value(cq);
    const GraphNode * add_q  = chain.only_consumer(cq);
    if (cq_val == nullptr || cq_val->ne[2] != 1 || cq_val->ne[3] != 1 || add_q == nullptr) {
        return false;
    }
    const int64_t head_dim = cq_val->ne[0];
    const int64_t n_head   = cq_val->ne[1];
    chain.nodes.push_back(add_q);
    const GraphNode * q_sum = mean_sum(chain, add_q, cq);
    const GraphNode * norm_q = chain.only_consumer(add_q->output);
    if (q_sum == nullptr || norm_q == nullptr || norm_q->op != GGML_OP_RMS_NORM) {
        return false;
    }
    chain.nodes.push_back(norm_q);
    // Qpre + reshape(REPEAT(Kpre)).
    const ValueId     qraw_id  = chain.source(q_sum->inputs[0]);
    const GraphNode * repeat   = chain.producer(q_sum->inputs[1], GGML_OP_REPEAT);
    const ValueId     kraw_id  = repeat != nullptr ? chain.source(repeat->inputs[0]) : ValueId{};
    const Value *     qraw     = chain.value(qraw_id);
    const Value *     kraw     = chain.value(kraw_id);
    if (repeat == nullptr || !is_rows(qraw, head_dim * n_head, 1) || kraw == nullptr || kraw->ne[0] % head_dim != 0 ||
        !is_rows(kraw, kraw->ne[0], 1)) {
        return false;
    }
    const int64_t n_head_kv = kraw->ne[0] / head_dim;
    if (n_head_kv < 1 || n_head % n_head_kv != 0 || k_view->element_count != kraw->ne[0]) {
        return false;
    }
    const int64_t gqa = n_head / n_head_kv;

    // Key side: slice -> CONT -> reshape [D, n_head_kv] -> ADD(., mean) -> RMS_NORM -> MUL(k_scale).
    chain.nodes.push_back(k_slice);
    const GraphNode * cont_k = chain.only_consumer(k_slice->output);
    if (cont_k == nullptr || cont_k->op != GGML_OP_CONT) {
        return false;
    }
    chain.nodes.push_back(cont_k);
    const ValueId     ck    = chain.skip_aliases(cont_k->output);
    const GraphNode * add_k = chain.only_consumer(ck);
    if (add_k == nullptr) {
        return false;
    }
    chain.nodes.push_back(add_k);
    const GraphNode * k_sum  = mean_sum(chain, add_k, ck);
    const GraphNode * norm_k = chain.only_consumer(add_k->output);
    const GraphNode * mul_k  = norm_k != nullptr ? chain.only_consumer(norm_k->output) : nullptr;
    if (k_sum == nullptr || norm_k == nullptr || norm_k->op != GGML_OP_RMS_NORM || mul_k == nullptr ||
        !common_binary_node_is_mul(*mul_k) || mul_k->inputs[0] != norm_k->output ||
        chain.source(k_sum->inputs[1]) != kraw_id) {
        return false;
    }
    chain.nodes.push_back(norm_k);
    chain.nodes.push_back(mul_k);
    // SCALE(SUM_ROWS(CONT(PERMUTE(reshape(Qpre)))), 1/gqa).
    const GraphNode * inv_gqa  = chain.producer(k_sum->inputs[0], GGML_OP_SCALE);
    const GraphNode * sum_rows = inv_gqa != nullptr ? chain.producer(inv_gqa->inputs[0], GGML_OP_SUM_ROWS) : nullptr;
    const GraphNode * cont_q   = sum_rows != nullptr ? chain.producer(sum_rows->inputs[0], GGML_OP_CONT) : nullptr;
    if (!scale_is(inv_gqa, 1.0f / static_cast<float>(gqa)) || cont_q == nullptr ||
        chain.source(cont_q->inputs[0]) != qraw_id) {
        return false;
    }

    const RmsNormParams * eps_q = op_params_as<RmsNormParams>(norm_q->params);
    const RmsNormParams * eps_k = op_params_as<RmsNormParams>(norm_k->params);
    const Value *         scale = chain.value(chain.source(mul_k->inputs[1]));
    const Value *         q_out = chain.value(norm_q->output);
    const Value *         k_out = chain.value(mul_k->output);
    if (eps_q == nullptr || eps_k == nullptr || eps_q->eps != eps_k->eps || scale == nullptr ||
        scale->type != GGML_TYPE_F32 || !scale->contiguous || scale->element_count != n_head_kv ||
        q_out == nullptr || k_out == nullptr || q_out->element_count != head_dim * n_head ||
        k_out->element_count != head_dim * n_head_kv || !q_out->contiguous || !k_out->contiguous ||
        q_out->alias_source.value >= 0 || k_out->alias_source.value >= 0 || head_dim % 64 != 0 || head_dim > 256 ||
        n_head > 128 || n_head_kv > 128) {
        return false;
    }

    for (const GraphNode * node : chain.nodes) {
        if (!append_covered_node_index_once(graph, context.covered_nodes, node, dispatch_match.covered_nodes)) {
            dispatch_match.covered_nodes.clear();
            return false;
        }
    }

    const auto input = [](const Value * v) {
        return DispatchBinding{ v->storage_root, v->storage_offset, v->byte_count };
    };
    char eps[32];
    std::snprintf(eps, sizeof(eps), "%.9g", static_cast<double>(eps_q->eps));
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kZayaCcaQkNormKernel);
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_qk_norm.head_dim", std::to_string(head_dim));
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_qk_norm.n_head", std::to_string(n_head));
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_qk_norm.n_head_kv", std::to_string(n_head_kv));
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_qk_norm.gqa", std::to_string(gqa));
    dispatch.kernel.compile_parameters.emplace("ggml.zaya_cca_qk_norm.rms_epsilon", eps);
    dispatch.bindings.push_back(input(conv));
    dispatch.bindings.push_back(input(qraw));
    dispatch.bindings.push_back(input(kraw));
    dispatch.bindings.push_back(input(scale));
    dispatch.bindings.push_back({ q_out->id, 0, q_out->byte_count });
    dispatch.bindings.push_back({ k_out->id, 0, k_out->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_zaya_cca_qk_norm_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "zaya.cca_qk_norm.decode_f32",
        GGML_OP_CONT,
        DispatchMatchKind::Fused,
        500,
        DispatchSource::Common,
        match_zaya_cca_qk_norm,
    });
}

}  // namespace ggml::hrx
