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

// The fused GATED_DELTA_NET pattern and a node the scheduler has already passed (engine #1925).
//
// The prefill pattern is rooted at the L2_NORM of q and covers fifteen nodes, four of which only rename storage: the
// RESHAPEs of the gate and beta chains, the RESHAPE that gives the gathered recurrent state its [128,128,heads,seqs]
// shape, and the VIEW of the cache the new state is copied into. The scheduler visits nodes in dependency order and
// elides a layout alias as soon as its input exists. The state RESHAPE's input is the state gather, which depends on
// nothing else in the layer, so the RESHAPE can be visited before the L2_NORM. The matcher then found its pattern,
// could not cover a node that was already covered, and declined; the standalone L2_NORM leaves this consumer to the
// fused matcher, so the node had no dispatch and the graph failed:
//
//   unsupported HRX node 160: L2_NORM output=f32[128,16,512,1]<-L2_NORM inputs=[f32[128,16,512,1]<-VIEW]
//       consumers=[174:GATED_DELTA_NET]
//
// (Qwen3.8-Flash-Next, the first 512-token batch past 2,048 tokens of context, with the f16 mask kernels on.)
//
// The cases build the layer's graph at that model's head counts (16 q/k heads, 48 v heads, width 128, so one token's
// q|k|v row is 10,240 floats) and schedule it for gfx1151 on the host. `state_first` puts the state RESHAPE ahead of
// the L2_NORM in the graph, which is the order the scheduler reached in the model. Each such plan must exist and must
// carry the same gated-delta-net dispatches (kernel, parameters, binding ranges) as the ordinary order.

#include "dispatch/dispatch-scheduler.h"
#include "ggml.h"
#include "graph/graph.h"
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
Planned plan_layer(int64_t tokens, bool state_first, bool state_leaf) {
    ggml_init_params params = {};
    params.mem_size         = 8 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);

    const int64_t width = 128, qheads = 16, heads = 48, inputs = 5120;
    const int64_t hidden         = width * (2 * qheads + heads);
    const int64_t state_elements = width * width * heads;
    const size_t  state_bytes    = state_elements * sizeof(float);

    ggml_tensor * input    = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inputs, tokens), 0.5f);
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
    ggml_tensor * raw  = ggml_scale(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens), 0.5f);
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
    ggml_build_forward_expand(graph, raw);
    ggml_build_forward_expand(graph, alpha);
    ggml_build_forward_expand(graph, beta_raw);
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

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    check("state RESHAPE ahead of the L2_NORM", 512, false);
    check("state RESHAPE ahead of the L2_NORM, gather in another split", 512, true);
    check("state RESHAPE ahead of the L2_NORM", 64, false);
    check("state RESHAPE ahead of the L2_NORM", 2, false);
    return 0;
}
