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

// Decode-kernel input rows for input sizes that are a multiple of 32 but not of 256 (gpt-oss 2880, BlackMamba 1152;
// 2048 as the control): every token / input row after the first must be read input_size values after the previous
// one. MUL_MAT with 2..8 tokens and MUL_MAT_ID with 1 or 4 input rows per token (the down projection reads one row
// per route) on MXFP4, Q8_0 and Q4_0 weights, and on NVFP4 also at 320 and with one token, against the CPU backend
// (normalized MSE, as test-backend-ops). Each graph runs on the HRX device itself (no scheduler) and must be planned
// on HRX.

#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

static constexpr int64_t kOutputSize  = 96;
static constexpr int64_t kExpertCount = 32;
static constexpr int64_t kRouteCount  = 4;
static constexpr double  kMaxNmse     = 5e-4;

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
        names += (names.empty() ? "" : "+") + kernel_name_for_id(dispatch.kernel.kernel_id);
    }
    return names;
}

struct Case {
    ggml_type type;
    int64_t   input_size;
    int64_t   tokens;
    int64_t   input_rows;  // 0: MUL_MAT; 1 or kRouteCount: MUL_MAT_ID
};

struct Data {
    std::vector<uint8_t> weights;
    std::vector<float>   x;
    std::vector<int32_t> ids;
};

static Data make_data(const Case & c, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    const int64_t                         experts = c.input_rows == 0 ? 1 : kExpertCount;
    const int64_t                         rows    = kOutputSize * experts;
    std::vector<float>                    w(static_cast<size_t>(rows * c.input_size));
    for (float & v : w) {
        v = uniform(rng);
    }
    Data d;
    d.weights.resize(ggml_row_size(c.type, c.input_size) * rows);
    ggml_quantize_chunk(c.type, w.data(), d.weights.data(), 0, rows, c.input_size, nullptr);
    const int64_t x_rows = c.input_rows == 0 ? 1 : c.input_rows;
    d.x.resize(static_cast<size_t>(c.input_size * x_rows * c.tokens));
    for (float & v : d.x) {
        v = uniform(rng);
    }
    if (c.input_rows != 0) {
        d.ids.resize(static_cast<size_t>(kExpertCount * c.tokens));
        for (int64_t t = 0; t < c.tokens; ++t) {
            std::vector<int32_t> order(kExpertCount);
            for (int64_t e = 0; e < kExpertCount; ++e) {
                order[e] = static_cast<int32_t>(e);
            }
            std::shuffle(order.begin(), order.end(), rng);
            std::copy(order.begin(), order.end(), d.ids.begin() + t * kExpertCount);
        }
    }
    return d;
}

// Runs the case on a backend; returns false when HRX (plan != nullptr) does not take the graph.
static bool run(ggml_backend_t backend, const Case & c, const Data & d, std::vector<float> & result, std::string * plan) {
    ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weights = nullptr;
    ggml_tensor * x       = nullptr;
    ggml_tensor * ids_all = nullptr;
    ggml_tensor * out     = nullptr;
    if (c.input_rows == 0) {
        weights = ggml_new_tensor_2d(ctx, c.type, c.input_size, kOutputSize);
        x       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.input_size, c.tokens);
        out     = ggml_mul_mat(ctx, weights, x);
    } else {
        weights           = ggml_new_tensor_3d(ctx, c.type, c.input_size, kOutputSize, kExpertCount);
        x                 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.input_size, c.input_rows, c.tokens);
        ids_all           = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kExpertCount, c.tokens);
        ggml_tensor * ids = ggml_view_2d(ctx, ids_all, kRouteCount, c.tokens, ids_all->nb[1], 0);
        out               = ggml_mul_mat_id(ctx, weights, x, ids);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (plan != nullptr) {
        *plan = ggml_backend_supports_op(backend, out) ? hrx_plan(graph) : "";
        if (plan->empty()) {
            ggml_free(ctx);
            return false;
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    ggml_backend_tensor_set(weights, d.weights.data(), 0, d.weights.size());
    ggml_backend_tensor_set(x, d.x.data(), 0, d.x.size() * sizeof(float));
    if (ids_all != nullptr) {
        ggml_backend_tensor_set(ids_all, d.ids.data(), 0, d.ids.size() * sizeof(int32_t));
    }
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    result.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
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
        std::printf("test-hrx-decode-stride: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);

    std::vector<Case> cases;
    for (ggml_type type : { GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 }) {
        for (int64_t input_size : { 2880, 1152, 2048 }) {
            for (int64_t tokens : { 2, 3, 4, 8 }) {
                cases.push_back({ type, input_size, tokens, 0 });
            }
            for (int64_t input_rows : { int64_t(1), kRouteCount }) {
                for (int64_t tokens : { 1, 2 }) {
                    cases.push_back({ type, input_size, tokens, input_rows });
                }
            }
        }
    }
    // NVFP4 (64-value blocks): the format admits input sizes that are a multiple of 64, the decode lanes only
    // multiples of 256; 320 and 2880 must take a path that reads them right, from one token up.
    for (int64_t input_size : { 320, 2880, 1152, 2048 }) {
        for (int64_t tokens : { 1, 2, 3, 4, 8 }) {
            cases.push_back({ GGML_TYPE_NVFP4, input_size, tokens, 0 });
        }
        for (int64_t input_rows : { int64_t(1), kRouteCount }) {
            for (int64_t tokens : { 1, 2 }) {
                cases.push_back({ GGML_TYPE_NVFP4, input_size, tokens, input_rows });
            }
        }
    }
    int      failures = 0;
    uint32_t seed     = 1;
    for (const Case & c : cases) {
        const Data         d = make_data(c, seed++);
        std::string        plan;
        std::vector<float> got;
        std::vector<float> expected;
        const char *       op = c.input_rows == 0 ? "MUL_MAT" : "MUL_MAT_ID";
        if (!run(hrx, c, d, got, &plan)) {
            std::printf("%-10s %-5s k=%4lld tokens=%lld rows=%lld  not admitted by HRX\n", op, ggml_type_name(c.type),
                        (long long) c.input_size, (long long) c.tokens, (long long) c.input_rows);
            ++failures;
            continue;
        }
        REQUIRE(run(cpu, c, d, expected, nullptr));
        const double e  = nmse(got, expected);
        const bool   ok = e <= kMaxNmse;
        std::printf("%-10s %-5s k=%4lld tokens=%lld rows=%lld  %-44s nmse=%.3g %s\n", op, ggml_type_name(c.type),
                    (long long) c.input_size, (long long) c.tokens, (long long) c.input_rows, plan.c_str(), e,
                    ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::printf("test-hrx-decode-stride: %zu cases, %d failures\n", cases.size(), failures);
    REQUIRE(failures == 0);
    return 0;
}
