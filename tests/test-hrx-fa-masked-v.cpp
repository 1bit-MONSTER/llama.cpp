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

// FLASH_ATTN_EXT on HRX must not depend on K/V rows that the mask hides. A KV cache keeps whatever an earlier request
// (or a rejected draft) wrote past the current sequence end, so a dependence there makes identical requests give
// different logits. Each case runs the same attention several times on the HRX device, changing only the masked K/V
// rows (random values, +x, -x, +0, -0), and requires bitwise-identical outputs; one run is also checked against the
// CPU backend (normalized MSE). Cases cover the decode-split kernel (1..15 query rows) and the prefill kernel
// (16+ rows), with the sequence end inside a 64-key block and in the final partial block.

#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static constexpr int64_t kHeadSize           = 128;
static constexpr int64_t kQueryHeadCount     = 16;
static constexpr int64_t kKeyValueHeadCount  = 8;
static constexpr double  kMaxNmse            = 5e-4;

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

// The HRX kernels planned for the graph, or "" when HRX does not take it.
static std::string hrx_plan(ggml_cgraph * graph) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics) || !scheduler.plan().valid()) {
        return "";
    }
    std::string names;
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        const std::string name = kernel_name_for_id(dispatch.kernel.kernel_id);
        if (names.find(name) == std::string::npos) {
            names += (names.empty() ? "" : "+") + name;
        }
    }
    return names;
}

struct Case {
    int64_t queries;   // query rows in the batch
    int64_t visible;   // cells 0 .. visible-1 hold the sequence; the last query row sees all of them
    int64_t cells;     // KV cells in the graph (the mask width)
};

// Graph like llama.cpp builds it: K/V cache views [head, cell, kv head], Q permuted to [head, token, head].
struct Graph {
    ggml_context *        ctx    = nullptr;
    ggml_cgraph *         graph  = nullptr;
    ggml_tensor *         q      = nullptr;
    ggml_tensor *         k      = nullptr;
    ggml_tensor *         v      = nullptr;
    ggml_tensor *         mask   = nullptr;
    ggml_tensor *         out    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static Graph build(ggml_backend_t backend, const Case & c) {
    Graph            g;
    ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx                   = ggml_init(params);
    REQUIRE(g.ctx != nullptr);
    g.q                      = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, kHeadSize, kQueryHeadCount, c.queries);
    g.k                      = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, kHeadSize, kKeyValueHeadCount, c.cells);
    g.v                      = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, kHeadSize, kKeyValueHeadCount, c.cells);
    g.mask                   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, c.cells, c.queries);
    ggml_tensor * k_view     = ggml_view_3d(g.ctx, g.k, kHeadSize, c.cells, kKeyValueHeadCount, g.k->nb[2], g.k->nb[1], 0);
    ggml_tensor * v_view     = ggml_view_3d(g.ctx, g.v, kHeadSize, c.cells, kKeyValueHeadCount, g.v->nb[2], g.v->nb[1], 0);
    ggml_tensor * q_permuted = ggml_permute(g.ctx, g.q, 0, 2, 1, 3);
    ggml_tensor * attention  = ggml_flash_attn_ext(g.ctx, q_permuted, k_view, v_view, g.mask,
                                                   1.0f / std::sqrt(static_cast<float>(kHeadSize)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
    g.out   = ggml_reshape_2d(g.ctx, attention, kHeadSize * kQueryHeadCount, c.queries);
    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.out);
    g.buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    REQUIRE(g.buffer != nullptr);
    return g;
}

static void release(Graph & g) {
    ggml_backend_buffer_free(g.buffer);
    ggml_free(g.ctx);
}

static std::vector<float> compute(ggml_backend_t backend, Graph & g, const std::vector<float> & q,
                                  const std::vector<ggml_fp16_t> & k, const std::vector<ggml_fp16_t> & v,
                                  const std::vector<ggml_fp16_t> & mask) {
    ggml_backend_tensor_set(g.q, q.data(), 0, ggml_nbytes(g.q));
    ggml_backend_tensor_set(g.k, k.data(), 0, ggml_nbytes(g.k));
    ggml_backend_tensor_set(g.v, v.data(), 0, ggml_nbytes(g.v));
    ggml_backend_tensor_set(g.mask, mask.data(), 0, ggml_nbytes(g.mask));
    REQUIRE(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(static_cast<size_t>(ggml_nelements(g.out)));
    ggml_backend_tensor_get(g.out, result.data(), 0, result.size() * sizeof(float));
    return result;
}

static double nmse(const std::vector<float> & got, const std::vector<float> & expected) {
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = static_cast<double>(got[i]) - expected[i];
        err += diff * diff;
        ref += static_cast<double>(expected[i]) * expected[i];
    }
    return ref > 0.0 ? err / ref : err;
}

int main() {
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-fa-masked-v: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);

    std::vector<Case> cases;
    for (int64_t seed_case = 0; seed_case < 8; ++seed_case) {
        cases.push_back({ 1 + seed_case % 15, 6 + 7 * seed_case, 256 });          // decode split, end in block 0
        cases.push_back({ 1 + (3 * seed_case) % 15, 70 + 11 * seed_case, 256 });  // decode split, end in block 1
        cases.push_back({ 16 + 9 * seed_case, 30 + 13 * seed_case, 256 });        // prefill, end inside a block
    }
    cases.push_back({ 1, 150, 200 });   // decode split, partial last block
    cases.push_back({ 24, 150, 200 });  // prefill, partial last block (tail path)
    cases.push_back({ 40, 40, 512 });   // prefill, fresh prompt in a long cache

    const size_t kv_row = static_cast<size_t>(kHeadSize * kKeyValueHeadCount);
    int          failures = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case & c = cases[ci];
        REQUIRE(c.visible >= c.queries && c.visible <= c.cells);
        std::mt19937                    rng(static_cast<uint32_t>(1000 + ci));
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float>              q(static_cast<size_t>(kHeadSize * kQueryHeadCount * c.queries));
        for (float & x : q) {
            x = 2.0f * normal(rng);
        }
        std::vector<ggml_fp16_t> k(kv_row * c.cells);
        std::vector<ggml_fp16_t> v(kv_row * c.cells);
        for (size_t i = 0; i < k.size(); ++i) {
            k[i] = ggml_fp32_to_fp16(normal(rng));
            v[i] = ggml_fp32_to_fp16(normal(rng));
        }
        // Causal mask: query row r sits at position visible - queries + r.
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(c.cells * c.queries));
        for (int64_t r = 0; r < c.queries; ++r) {
            const int64_t position = c.visible - c.queries + r;
            for (int64_t cell = 0; cell < c.cells; ++cell) {
                mask[r * c.cells + cell] = ggml_fp32_to_fp16(cell <= position ? 0.0f : -INFINITY);
            }
        }

        Graph             g    = build(hrx, c);
        const std::string plan = ggml_backend_supports_op(hrx, g.out->src[0]) ? hrx_plan(g.graph) : "";
        if (plan.empty()) {
            std::printf("FAIL case %zu (q=%lld visible=%lld cells=%lld): not planned on HRX\n", ci,
                        static_cast<long long>(c.queries), static_cast<long long>(c.visible),
                        static_cast<long long>(c.cells));
            ++failures;
            release(g);
            continue;
        }
        const std::vector<float> reference = compute(hrx, g, q, k, v, mask);

        // Rewrite only the rows past the sequence end, five ways.
        int mismatches = 0;
        for (int variant = 0; variant < 5; ++variant) {
            std::vector<ggml_fp16_t> k2 = k;
            std::vector<ggml_fp16_t> v2 = v;
            std::mt19937             stale_rng(static_cast<uint32_t>(77 + variant));
            for (size_t i = kv_row * c.visible; i < k2.size(); ++i) {
                float kv = 0.0f;
                float vv = 0.0f;
                switch (variant) {
                    case 0: kv = 3.0f * normal(stale_rng); vv = 3.0f * normal(stale_rng); break;
                    case 1: kv = 30.0f; vv = 1.0f; break;
                    case 2: kv = -30.0f; vv = -1.0f; break;
                    case 3: kv = 0.0f; vv = 0.0f; break;
                    default: kv = -0.0f; vv = -0.0f; break;
                }
                k2[i] = ggml_fp32_to_fp16(kv);
                v2[i] = ggml_fp32_to_fp16(vv);
            }
            const std::vector<float> got = compute(hrx, g, q, k2, v2, mask);
            if (std::memcmp(got.data(), reference.data(), got.size() * sizeof(float)) != 0) {
                ++mismatches;
            }
        }
        release(g);

        Graph                    gc       = build(cpu, c);
        const std::vector<float> expected = compute(cpu, gc, q, k, v, mask);
        release(gc);
        const double error = nmse(reference, expected);
        const bool   ok    = mismatches == 0 && error <= kMaxNmse;
        failures += ok ? 0 : 1;
        std::printf("%s case %zu q=%lld visible=%lld cells=%lld (%s): masked-row variants differing %d/5, nmse %.3g\n",
                    ok ? "ok  " : "FAIL", ci, static_cast<long long>(c.queries), static_cast<long long>(c.visible),
                    static_cast<long long>(c.cells), plan.c_str(), mismatches, error);
    }
    ggml_backend_free(hrx);
    ggml_backend_free(cpu);
    if (failures != 0) {
        std::printf("test-hrx-fa-masked-v: %d of %zu cases failed\n", failures, cases.size());
        return 1;
    }
    std::printf("test-hrx-fa-masked-v: %zu cases passed\n", cases.size());
    return 0;
}
