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

// Prefill FLASH_ATTN_EXT with the transposed V at 256 and 512 query tokens (engine #1989). From 256 query tokens the
// dispatch copies V into a transposed buffer and the attention kernel walks it across channels. That buffer now has a
// pitch of keys + 32 when the key count is a multiple of 64, and from 4,096 keys up 512 query tokens run as two launches
// of 256 over the one buffer. Both are layout and scheduling changes, so the values must not move: each case is compared with the CPU
// backend (normalized MSE, as test-backend-ops), with a causal mask so masked keys and a padded tail are exercised, and
// the plan must be one transpose plus the expected number of attention launches. Key counts cover a padded pitch
// (1,024, 2,048, 4,096), a packed one (1,056, 4,128) and both sides of the 4,096-key launch rule.

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
#include <limits>
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

static constexpr int64_t kHeadSize   = 128;
static constexpr int64_t kKvHeads    = 4;
static constexpr int64_t kQueryHeads = 8;
static constexpr double  kMaxNmse    = 5e-4;

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

// Counts the planned dispatches whose kernel name contains `needle`; -1 when HRX does not take the graph.
static int planned(ggml_cgraph * graph, const char * needle) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }) || !scheduler.plan().valid()) {
        return -1;
    }
    int count = 0;
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        count += kernel_name_for_id(dispatch.kernel.kernel_id).find(needle) != std::string::npos ? 1 : 0;
    }
    return count;
}

struct Case {
    int64_t queries;
    int64_t keys;
};

struct Data {
    std::vector<float>       q;
    std::vector<ggml_fp16_t> k;
    std::vector<ggml_fp16_t> v;
    std::vector<ggml_fp16_t> mask;
};

static Data make_data(const Case & c, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    Data d;
    d.q.resize(static_cast<size_t>(kHeadSize * kQueryHeads * c.queries));
    for (float & x : d.q) {
        x = uniform(rng);
    }
    d.k.resize(static_cast<size_t>(kHeadSize * kKvHeads * c.keys));
    d.v.resize(d.k.size());
    for (ggml_fp16_t & x : d.k) {
        x = ggml_fp32_to_fp16(uniform(rng));
    }
    for (ggml_fp16_t & x : d.v) {
        x = ggml_fp32_to_fp16(uniform(rng));
    }
    // causal over the last `queries` keys, like a prefill batch appended to earlier context; nothing is fully masked
    const ggml_fp16_t hidden  = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const ggml_fp16_t visible = ggml_fp32_to_fp16(0.0f);
    d.mask.resize(static_cast<size_t>(c.keys * c.queries));
    const int64_t first = c.keys - c.queries;
    for (int64_t query = 0; query < c.queries; ++query) {
        for (int64_t key = 0; key < c.keys; ++key) {
            d.mask[static_cast<size_t>(query * c.keys + key)] = key <= first + query ? visible : hidden;
        }
    }
    return d;
}

struct Plan {
    int transposes = 0;
    int attention  = 0;
};

static bool run(ggml_backend_t backend, const Case & c, const Data & d, std::vector<float> & result, Plan * plan) {
    ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    // the model's layout: token-major rows [head_size, heads, tokens], permuted to [head_size, tokens, heads]
    ggml_tensor * q0   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadSize, kQueryHeads, c.queries, 1);
    ggml_tensor * k0   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kHeadSize, kKvHeads, c.keys, 1);
    ggml_tensor * v0   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kHeadSize, kKvHeads, c.keys, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.keys, c.queries, 1, 1);
    ggml_tensor * out  = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q0, 0, 2, 1, 3), ggml_permute(ctx, k0, 0, 2, 1, 3),
                                             ggml_permute(ctx, v0, 0, 2, 1, 3), mask,
                                             1.0f / std::sqrt(static_cast<float>(kHeadSize)), 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (plan != nullptr) {
        plan->transposes = ggml_backend_supports_op(backend, out) ? planned(graph, "copy_transpose_f16") : -1;
        plan->attention  = plan->transposes < 0 ? -1 : planned(graph, "flash_attention_f32_f16_wmma");
        if (plan->transposes < 0) {
            ggml_free(ctx);
            return false;
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    ggml_backend_tensor_set(q0, d.q.data(), 0, ggml_nbytes(q0));
    ggml_backend_tensor_set(k0, d.k.data(), 0, ggml_nbytes(k0));
    ggml_backend_tensor_set(v0, d.v.data(), 0, ggml_nbytes(v0));
    ggml_backend_tensor_set(mask, d.mask.data(), 0, ggml_nbytes(mask));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    result.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-fa-transposed-value: no HRX0 device, skipped\n");
        return 77;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);

    // With GGML_HRX_FA_QUERY_CHUNK set by hand the launch count is whatever was asked for; only the values are checked.
    const bool default_chunk = std::getenv("GGML_HRX_FA_QUERY_CHUNK") == nullptr &&
                               std::getenv("GGML_HRX_QUERY_CHUNK_TOKENS") == nullptr;
    const std::vector<Case> cases = { { 256, 1024 }, { 256, 1056 }, { 512, 2048 }, { 256, 4096 },
                                      { 512, 4096 }, { 512, 4128 }, { 768, 4096 }, { 512, 4160 } };
    int      failures = 0;
    uint32_t seed     = 1;
    for (const Case & c : cases) {
        const Data         d = make_data(c, seed++);
        Plan               plan;
        std::vector<float> got;
        std::vector<float> expected;
        if (!run(hrx, c, d, got, &plan)) {
            std::printf("queries=%4lld keys=%5lld  not admitted by HRX\n", (long long) c.queries, (long long) c.keys);
            ++failures;
            continue;
        }
        REQUIRE(run(cpu, c, d, expected, nullptr));
        double err = 0.0;
        double ref = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double diff = static_cast<double>(got[i]) - expected[i];
            err += diff * diff;
            ref += static_cast<double>(expected[i]) * expected[i];
        }
        const double nmse          = ref > 0.0 ? err / ref : err;
        const int    want_launches = c.keys >= 4096 ? static_cast<int>((c.queries + 255) / 256) : 1;
        const bool   plan_ok       = plan.transposes == 1 && (!default_chunk || plan.attention == want_launches);
        const bool   ok            = nmse <= kMaxNmse && plan_ok;
        std::printf("queries=%4lld keys=%5lld  transposes=%d attention launches=%d (want %d)  nmse=%.3g %s\n",
                    (long long) c.queries, (long long) c.keys, plan.transposes, plan.attention, want_launches, nmse,
                    ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::printf("test-hrx-fa-transposed-value: %zu cases, %d failures\n", cases.size(), failures);
    REQUIRE(failures == 0);
    return 0;
}
