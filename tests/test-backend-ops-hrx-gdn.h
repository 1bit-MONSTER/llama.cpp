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

// GATED_DELTA_NET layer cases for test-backend-ops: one whole Qwen3.5 / Qwen3.8 linear-attention layer
// (hrx-gdn-layer.h) with 1..4 sequences per ubatch, compared on the layer output and on both cache writes.
//
// The standalone GATED_DELTA_NET cases feed the op from leaves. Backends that run the op only inside the model's
// fused pattern (ggml-hrx: L2_NORM -> GATED_DELTA_NET -> CPY into the recurrent cache) report those as not
// supported; this case gives them the producers and consumers of the model graph.
//
// Included by test-backend-ops.cpp after test_case; `-o GATED_DELTA_NET` selects these cases too.

#include "hrx-gdn-layer.h"

#include <cstring>
#include <vector>

struct test_gated_delta_net_layer : public test_case {
    const hrx_gdn_layer::Case c;
    hrx_gdn_layer::Layer      layer;

    std::string vars() override {
        const std::string shape    = c.shape.name;
        const int64_t     tokens   = c.tokens;
        const int64_t     seqs     = c.seqs;
        const int64_t     K        = c.snapshots;
        const int64_t     kv_head  = c.kv_head;
        const int64_t     mem_size = c.mem_size;
        return VARS_TO_STR6(shape, tokens, seqs, K, kv_head, mem_size);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GATED_DELTA_NET";
    }

    // HRX keeps the 128x128 state products in F16 (WMMA) and the projections are Q4_K: compare like the
    // HRX layer test (test-hrx-gdn-multiseq), not bit-for-bit
    double max_nmse_err() override { return 5e-4; }

    bool run_whole_graph() override { return true; }

    std::vector<ggml_tensor *> fusion_test_nodes() override { return { layer.out, layer.conv_write, layer.ssm_write }; }

    explicit test_gated_delta_net_layer(hrx_gdn_layer::Case c) : c(c) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        layer = hrx_gdn_layer::build_layer(ctx, gf, c);
        for (ggml_tensor * w : { layer.w_qkv, layer.w_z, layer.w_alpha, layer.w_beta, layer.w_out }) {
            ggml_set_name(w, "weight");
        }
        return layer.out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op) || t->op != GGML_OP_NONE) {
                continue;
            }
            if (t == layer.s_copy) {
                std::vector<int32_t> ids(static_cast<size_t>(c.seqs));
                for (int64_t s = 0; s < c.seqs; ++s) {
                    ids[static_cast<size_t>(s)] = hrx_gdn_layer::s_copy_value(c, s);
                }
                ggml_backend_tensor_set(t, ids.data(), 0, ids.size() * sizeof(int32_t));
            } else if (t == layer.a) {
                std::vector<float> a(static_cast<size_t>(c.shape.v_heads));
                for (size_t i = 0; i < a.size(); ++i) {
                    a[i] = hrx_gdn_layer::a_value(static_cast<int64_t>(i));
                }
                ggml_backend_tensor_set(t, a.data(), 0, a.size() * sizeof(float));
            } else if (std::strcmp(t->name, "weight") == 0) {
                init_tensor_uniform(t, -0.05f, 0.05f);
            } else if (t == layer.ssm_cache) {
                init_tensor_uniform(t, -0.05f, 0.05f);
            } else if (t == layer.conv_w) {
                init_tensor_uniform(t, -0.5f, 0.5f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

static void add_hrx_gdn_layer_cases(std::vector<std::unique_ptr<test_case>> & test_cases) {
    for (const hrx_gdn_layer::Shape & shape : { hrx_gdn_layer::kQwen38_27B, hrx_gdn_layer::kTiny }) {
        for (int64_t seqs : { 1, 2, 3, 4 }) {
            for (int64_t tokens : { 1, 4, 32 }) {
                test_cases.emplace_back(new test_gated_delta_net_layer({ shape, tokens, seqs, 4, 0, 1 }));
            }
        }
        test_cases.emplace_back(new test_gated_delta_net_layer({ shape, 1, 2, 8, 5, 1 }));
        test_cases.emplace_back(new test_gated_delta_net_layer({ shape, 3, 2, 4, 0, 4 }));
    }
}
