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

// The attention-sink dispatch rewrites FlashAttention's output in place, so it must refuse a node whose output
// shares storage with one of its inputs, including a different value (view) of the same allocation.

#include "dispatch_registration/common/dispatch-attention-sink.h"
#include "ggml.h"
#include "graph/graph.h"
#include "graph/op-params.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static bool append_for(bool alias_output_onto_query) {
    ggml_init_params params = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    const int64_t    d = 64, tokens = 8, heads = 8, kv_heads = 2, keys = 32;  // tokens == heads: output and query share a layout
    ggml_tensor *    q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, tokens, heads);
    ggml_tensor *    k     = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, keys, kv_heads);
    ggml_tensor *    v     = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, keys, kv_heads);
    ggml_tensor *    mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, tokens);
    ggml_tensor *    sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    ggml_tensor *    fa    = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(fa, sinks);

    ggml::hrx::Graph            graph;
    std::vector<ggml::hrx::ValueId> inputs;
    for (ggml_tensor * source : fa->src) {
        if (source != nullptr) {
            inputs.push_back(graph.values().get_or_add_tensor_value(source, ggml::hrx::ValueKind::External));
        }
    }
    const ggml::hrx::ValueId output = graph.values().get_or_add_tensor_value(fa, ggml::hrx::ValueKind::Transient);
    if (alias_output_onto_query) {
        REQUIRE(graph.values().alias_storage(output, inputs[0]).success());
        REQUIRE(graph.values().same_storage(output, inputs[0]));
        REQUIRE(output.value != inputs[0].value);
    }
    ggml::hrx::GraphNode & node = graph.add_node(fa->op, output, inputs);
    node.params                 = ggml::hrx::import_op_params(*fa);
    REQUIRE(ggml::hrx::attention_sinks_supported(graph, node));
    ggml::hrx::DispatchMatch match;
    const bool               appended = ggml::hrx::append_attention_sink_dispatch(graph, node, match);
    ggml_free(ctx);
    return appended;
}

int main() {
    REQUIRE(append_for(false));
    REQUIRE(!append_for(true));
    std::printf("test-hrx-attention-sink: disjoint output accepted, output aliasing the query refused\n");
    return 0;
}
