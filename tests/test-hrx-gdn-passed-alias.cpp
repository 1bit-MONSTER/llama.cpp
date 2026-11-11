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

// The fused GATED_DELTA_NET pattern in a graph the scheduler walks out of order (engine #1925).
//
// Qwen3.8-Flash-Next, the first 512-token batch past 2,048 tokens of context, with the f16 mask kernels on:
//
//   unsupported HRX node 160: L2_NORM output=f32[128,16,512,1]<-L2_NORM inputs=[f32[128,16,512,1]<-VIEW]
//       consumers=[174:GATED_DELTA_NET]
//
// The cause is the walk. The scheduler visits nodes in dependency order with MUL_MAT and the other root ops first,
// which is what puts the gate and beta projections of a layer ahead of the L2_NORM its fused pattern is rooted at. An
// in-place CPY (the one ggml_cast builds) lists its own output as an input; counted as a dependency, it never became
// ready, the walk stopped at it, and every node downstream was appended in index order. The sparse path's mask chain
// has such casts, so every later layer was walked in index order, where the L2_NORM comes before the projections:
//
//   fusion order: llm.gated_delta_net.f32_wmma_head128 at node 160 (L2_NORM) refused: ... alpha_raw reads value 206
//       produced by node 164 (MUL_MAT) that is not scheduled yet
//
// Nodes that depend on nothing in the stalled part were still walked early, among them the RESHAPE that gives the
// gathered recurrent state its [128,128,heads,seqs] shape. It is a layout alias, so it was elided, and the pattern
// then could not cover a node that was already covered. That was the first refusal the model showed.
//
// The cases build the layer's graph at that model's head counts (16 q/k heads, 48 v heads, width 128, so one token's
// q|k|v row is 10,240 floats) and schedule it for gfx1151 on the host. check_walk puts the layer behind a ggml_cast;
// check puts the state RESHAPE ahead of the L2_NORM. Each plan must exist and must carry the same gated-delta-net
// dispatches (kernel, parameters, binding ranges) as the same graph without the disturbance.

#include "dispatch/dispatch-scheduler.h"
#include "ggml.h"
#include "graph/graph-traversal.h"
#include "graph/graph.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

namespace {

using namespace ggml::hrx;

std::string kernel_name(uint64_t id) {
    const KernelResolveResult resolved = resolve_kernel_definition(get_qwen_kernel_corpus(), "gfx1151", id);
    REQUIRE(resolved.found());
    return kernel_definition_name(*resolved.definition);
}

struct Planned {
    bool                     ok = false;
    std::string              error;
    std::vector<std::string> gdn;  // one line per gated-delta-net dispatch: kernel, parameters, binding ranges
};

// state_leaf: the state gather ran in another split, so the RESHAPE's input has no producer in this graph.
Planned plan_layer(int64_t tokens, bool state_first, bool state_leaf, bool model_order = false, bool cast_upstream = false) {
    ggml_init_params params = {};
    params.mem_size         = 8 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const int64_t width = 128, qheads = 16, heads = 48, inputs = 5120;
    const int64_t hidden         = width * (2 * qheads + heads);
    const int64_t state_elements = width * width * heads;
    const size_t  state_bytes    = state_elements * sizeof(float);

    // cast_upstream: the layer's input comes through a ggml_cast, whose CPY writes into its own output value (the
    // sparse path's mask chain has such casts ahead of every later layer).
    ggml_tensor * source   = cast_upstream ?
                                 ggml_cast(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inputs, tokens), GGML_TYPE_F32) :
                                 ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inputs, tokens);
    ggml_tensor * input    = ggml_scale(ctx, source, 0.5f);
    ggml_tensor * w_alpha  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, inputs, heads);
    ggml_tensor * w_beta   = ggml_dup_tensor(ctx, w_alpha);
    ggml_tensor * alpha    = ggml_mul_mat(ctx, w_alpha, input);
    ggml_tensor * beta_raw = ggml_mul_mat(ctx, w_beta, input);
    ggml_tensor * bias     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    ggml_tensor * scale    = ggml_dup_tensor(ctx, bias);
    ggml_tensor * gate     = ggml_mul(
        ctx, ggml_softplus(ctx, ggml_add(ctx, ggml_reshape_3d(ctx, alpha, heads, tokens, 1), bias)), scale);
    gate               = ggml_reshape_4d(ctx, gate, 1, heads, tokens, 1);
    ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, beta_raw, 1, heads, tokens, 1));
    // the q|k|v row: with model_order a projection of the layer input, as in the model (so the L2_NORM depends on a
    // MUL_MAT of the same input the gate and beta projections read)
    ggml_tensor * raw = model_order ?
                            ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, inputs, hidden), input) :
                            ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens), 0.5f);
    const auto    view = [&](int64_t h, size_t offset) {
        return ggml_view_4d(ctx, raw, width, h, tokens, 1, width * sizeof(float), hidden * sizeof(float),
                            hidden * tokens * sizeof(float), offset);
    };
    ggml_tensor * q        = ggml_l2_norm(ctx, view(qheads, 0), 1.e-6f);
    ggml_tensor * k        = ggml_l2_norm(ctx, view(qheads, width * qheads * sizeof(float)), 1.e-6f);
    ggml_tensor * v        = view(heads, 2 * width * qheads * sizeof(float));
    ggml_tensor * cache    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_elements, 20);
    ggml_tensor * ids      = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * gathered = ggml_get_rows(ctx, cache, ids);
    ggml_tensor * state    = ggml_reshape_4d(
        ctx, state_leaf ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_elements, 1) : gathered, width, width, heads, 1);
    ggml_tensor * gdn      = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, 5);
    const int64_t count    = std::min<int64_t>(tokens, 5);
    const size_t  attention_bytes = width * heads * tokens * sizeof(float);
    ggml_tensor * attention = ggml_view_4d(ctx, gdn, width, heads, tokens, 1, width * sizeof(float),
                                           width * heads * sizeof(float), attention_bytes, 0);
    ggml_tensor * snapshots =
        ggml_view_3d(ctx, gdn, state_elements, 1, count, state_bytes, state_bytes, attention_bytes);
    ggml_tensor * target = ggml_view_3d(ctx, cache, state_elements, 1, count, state_bytes, 4 * state_bytes, 0);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (!model_order) {
        ggml_build_forward_expand(graph, raw);
        ggml_build_forward_expand(graph, alpha);
        ggml_build_forward_expand(graph, beta_raw);
    }
    ggml_build_forward_expand(graph, gathered);
    if (state_first) {
        ggml_build_forward_expand(graph, state);
    }
    ggml_build_forward_expand(graph, ggml_cpy(ctx, snapshots, target));
    ggml_build_forward_expand(graph, ggml_scale(ctx, attention, 0.5f));

    Planned result;
    auto    imported = import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    DispatchScheduler scheduler;
    result.ok = scheduler.schedule_graph(imported.graph, DispatchTarget{ "gfx1151" });
    if (!result.ok) {
        for (const std::string & e : scheduler.plan().status.errors()) {
            result.error += e + "\n";
        }
    } else {
        for (const Dispatch & d : scheduler.plan().dispatches) {
            const std::string name = kernel_name(d.kernel.kernel_id);
            if (name.find("gated_delta_net") == std::string::npos) {
                continue;
            }
            std::string line = name;
            for (const auto & p : d.kernel.compile_parameters) {
                line += " " + p.first + "=" + p.second;
            }
            for (const auto & p : d.kernel.integer_parameters) {
                line += " " + p.first + "=" + std::to_string(p.second);
            }
            for (const DispatchBinding & b : d.bindings) {
                line += " [" + std::to_string(b.offset) + "+" + std::to_string(b.length) + "]";
            }
            result.gdn.push_back(std::move(line));
        }
    }
    ggml_free(ctx);
    return result;
}

void check(const char * what, int64_t tokens, bool state_leaf) {
    const Planned ordinary = plan_layer(tokens, false, false);
    REQUIRE(ordinary.ok);
    REQUIRE(!ordinary.gdn.empty());
    const Planned passed = plan_layer(tokens, true, state_leaf);
    std::printf("%-58s tokens=%-3lld : %s, %zu gated-delta-net dispatches (ordinary order: %zu)\n", what,
                (long long) tokens, passed.ok ? "plans" : "REFUSED", passed.gdn.size(), ordinary.gdn.size());
    if (!passed.ok) {
        std::printf("  %s", passed.error.c_str());
    }
    REQUIRE(passed.ok);
    REQUIRE(passed.gdn == ordinary.gdn);
}

// The walk itself (engine #1925, the cause). With model_order the graph is expanded from its outputs only, as llama.cpp
// builds it, so the L2_NORM has a lower index than the gate and beta projections the pattern reads. The scheduler's
// dependency walk still reaches the projections first (MUL_MAT is a root op). With cast_upstream the layer's inputs
// come through a ggml_cast: its CPY lists its own output as an input, the walk used to stop at it and append the rest
// in index order, and the pattern at the L2_NORM was refused because the projection was "not scheduled yet".
void check_walk(int64_t tokens) {
    const Planned ordinary = plan_layer(tokens, false, false, true, false);
    if (!ordinary.ok) {
        std::printf("model order, no cast, tokens=%lld REFUSED\n  %s", (long long) tokens, ordinary.error.c_str());
    }
    REQUIRE(ordinary.ok);
    REQUIRE(!ordinary.gdn.empty());
    const Planned behind_cast = plan_layer(tokens, false, false, true, true);
    std::printf("%-58s tokens=%-3lld : %s, %zu gated-delta-net dispatches (no cast: %zu)\n",
                "layer behind an in-place CPY (ggml_cast)", (long long) tokens, behind_cast.ok ? "plans" : "REFUSED",
                behind_cast.gdn.size(), ordinary.gdn.size());
    if (!behind_cast.ok) {
        std::printf("  %s", behind_cast.error.c_str());
    }
    REQUIRE(behind_cast.ok);
    REQUIRE(behind_cast.gdn == ordinary.gdn);
}

// A walk that does stall says so. No graph llama.cpp builds does, so the graph is made by hand: two SCALE nodes that
// read each other's output. Neither can be reached, both are taken in index order, the first is refused, and the
// refusal carries the line that tells the reader the order was not the dependency order.
void check_stall_report() {
    ggml_init_params params = {};
    params.mem_size         = 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * a = ggml_scale(ctx, x, 0.5f);
    ggml_tensor * b = ggml_scale(ctx, a, 0.5f);

    Graph         graph;
    const ValueId va = graph.values().get_or_add_tensor_value(a, ValueKind::Transient);
    const ValueId vb = graph.values().get_or_add_tensor_value(b, ValueKind::Transient);
    GraphNode &   na = graph.add_node(GGML_OP_SCALE, va, { vb });
    na.params        = import_op_params(*a);
    GraphNode &   nb = graph.add_node(GGML_OP_SCALE, vb, { va });
    nb.params        = import_op_params(*b);
    REQUIRE(graph.build_index().success());

    const GraphTraversalOrder order = GraphTraversalOrder::build(graph);
    REQUIRE(order.nodes().size() == 2);
    REQUIRE(order.stalled_count() == 2);
    REQUIRE(order.first_stalled_node() == 0);

    DispatchScheduler scheduler;
    REQUIRE(!scheduler.schedule_graph(graph, DispatchTarget{ "gfx1151" }));
    std::string text;
    for (const std::string & e : scheduler.plan().status.errors()) {
        text += e + "\n";
    }
    std::printf("two nodes that read each other: refused, and the refusal says:\n  %s", text.c_str());
    REQUIRE(text.find("dependency walk stalled: 2 of 2 nodes were not reached") != std::string::npos);
    REQUIRE(text.find("from node 0 (SCALE)") != std::string::npos);
    ggml_free(ctx);

    // and the cast that used to stall the walk does not: nothing is left unreached behind a ggml_cast
    ggml_context * ctx2 = ggml_init(params);
    REQUIRE(ctx2 != nullptr);
    ggml_tensor * cast  = ggml_cast(ctx2, ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 64), GGML_TYPE_F32);
    ggml_tensor * after = ggml_scale(ctx2, ggml_scale(ctx2, cast, 0.5f), 0.5f);
    ggml_cgraph * cgraph = ggml_new_graph(ctx2);
    ggml_build_forward_expand(cgraph, after);
    auto imported = import_ggml_graph(*cgraph);
    REQUIRE(imported.valid());
    const GraphTraversalOrder cast_order = GraphTraversalOrder::build(imported.graph);
    std::printf("a ggml_cast and two nodes behind it: %zu nodes, %zu not reached\n", cast_order.nodes().size(),
                cast_order.stalled_count());
    REQUIRE(cast_order.stalled_count() == 0);
    ggml_free(ctx2);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    check_stall_report();
    check_walk(512);
    check_walk(64);
    check_walk(2);
    check("state RESHAPE ahead of the L2_NORM", 512, false);
    check("state RESHAPE ahead of the L2_NORM, gather in another split", 512, true);
    check("state RESHAPE ahead of the L2_NORM", 64, false);
    check("state RESHAPE ahead of the L2_NORM", 2, false);
    return 0;
}
