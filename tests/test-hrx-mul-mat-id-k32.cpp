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

// MUL_MAT_ID on HRX with input sizes that are a multiple of 32 but not of 256: gpt-oss-20b's experts (2880, the last
// 256-value tile 64 values long) and BlackMamba's (1152, last tile 128 long), with 2816 (no tail) as the control.
// The 32-value block formats MXFP4, Q8_0 and Q4_0; 1 token (decode kernel) and 7 / 40 tokens (WMMA kernels). Each
// graph runs on the HRX device itself (no scheduler, no CPU fallback), its HRX plan must contain a mul_mat_id
// kernel, and the result is compared with the CPU backend (normalized MSE, as test-backend-ops).

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

static constexpr int64_t kOutputSize  = 96;  // one full 64-row tile and a partial one
static constexpr int64_t kExpertCount = 8;
static constexpr int64_t kRouteCount  = 4;
static constexpr double  kMaxNmse     = 5e-4;

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

static std::string hrx_mul_mat_id_kernel(ggml_cgraph * graph) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics)) {
        std::fprintf(stderr, "unsupported: %s\n", diagnostics.unsupported_message.c_str());
        std::abort();
    }
    REQUIRE(scheduler.plan().valid());
    std::string found;
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        const std::string name = kernel_name_for_id(dispatch.kernel.kernel_id);
        if (name.find("mul_mat_id") != std::string::npos) {
            found = name;
        }
    }
    return found;
}

struct Inputs {
    std::vector<uint8_t> weights;
    std::vector<float>   activations;
    std::vector<int32_t> ids;
};

static Inputs make_inputs(ggml_type type, int64_t input_size, int64_t token_count, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    Inputs                                in;
    const int64_t                         rows = kOutputSize * kExpertCount;
    std::vector<float>                    w(static_cast<size_t>(rows * input_size));
    for (float & v : w) {
        v = uniform(rng);
    }
    in.weights.resize(ggml_row_size(type, input_size) * rows);
    ggml_quantize_chunk(type, w.data(), in.weights.data(), 0, rows, input_size, nullptr);
    in.activations.resize(static_cast<size_t>(input_size * token_count));
    for (float & v : in.activations) {
        v = uniform(rng);
    }
    // ids [kExpertCount, token_count] (an argsort-like layout); MUL_MAT_ID reads the first kRouteCount of each column
    in.ids.resize(static_cast<size_t>(kExpertCount * token_count));
    for (int64_t t = 0; t < token_count; ++t) {
        std::vector<int32_t> order(kExpertCount);
        for (int64_t e = 0; e < kExpertCount; ++e) {
            order[e] = static_cast<int32_t>(e);
        }
        std::shuffle(order.begin(), order.end(), rng);
        for (int64_t e = 0; e < kExpertCount; ++e) {
            in.ids[t * kExpertCount + e] = order[e];
        }
    }
    return in;
}

static std::vector<float> run(ggml_backend_t backend, ggml_type type, int64_t input_size, int64_t token_count,
                              const Inputs & in, std::string * hrx_kernel) {
    ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, type, input_size, kOutputSize, kExpertCount);
    ggml_tensor * x       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_size, 1, token_count);
    ggml_tensor * ids_all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kExpertCount, token_count);
    ggml_tensor * ids     = ggml_view_2d(ctx, ids_all, kRouteCount, token_count, ids_all->nb[1], 0);
    ggml_tensor * out     = ggml_mul_mat_id(ctx, weights, x, ids);
    ggml_cgraph * graph   = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    REQUIRE(ggml_backend_supports_op(backend, out));
    if (hrx_kernel != nullptr) {
        *hrx_kernel = hrx_mul_mat_id_kernel(graph);
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    ggml_backend_tensor_set(weights, in.weights.data(), 0, in.weights.size());
    ggml_backend_tensor_set(x, in.activations.data(), 0, in.activations.size() * sizeof(float));
    ggml_backend_tensor_set(ids_all, in.ids.data(), 0, in.ids.size() * sizeof(int32_t));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static double nmse(const std::vector<float> & got, const std::vector<float> & expected) {
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - expected[i];
        err += d * d;
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
        std::printf("test-hrx-mul-mat-id-k32: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);

    const ggml_type types[]        = { GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0 };
    const int64_t   input_sizes[]  = { 2880, 1152, 2816 };
    const int64_t   token_counts[] = { 1, 7, 40 };
    int             failures       = 0;
    uint32_t        seed           = 1;
    for (ggml_type type : types) {
        for (int64_t input_size : input_sizes) {
            for (int64_t token_count : token_counts) {
                const Inputs       in = make_inputs(type, input_size, token_count, seed++);
                std::string        kernel;
                std::vector<float> got      = run(hrx, type, input_size, token_count, in, &kernel);
                std::vector<float> expected = run(cpu, type, input_size, token_count, in, nullptr);
                const double       e        = nmse(got, expected);
                const bool         ok       = !kernel.empty() && e <= kMaxNmse;
                std::printf("%-5s k=%5lld tokens=%2lld  %-48s nmse=%.3g  %s\n", ggml_type_name(type),
                            static_cast<long long>(input_size), static_cast<long long>(token_count),
                            kernel.empty() ? "(no HRX mul_mat_id kernel)" : kernel.c_str(), e, ok ? "OK" : "FAIL");
                failures += ok ? 0 : 1;
            }
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    REQUIRE(failures == 0);
    std::printf("test-hrx-mul-mat-id-k32: all cases OK\n");
    return 0;
}
