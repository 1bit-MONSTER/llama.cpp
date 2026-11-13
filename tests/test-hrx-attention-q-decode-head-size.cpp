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

// The fused Q projection at one token (MUL_MAT, the head layout, ROPE) on head sizes that are not a power of two:
// 48 (stories15M), 96 and 160 (Sarashina2.2-3B), with 64 and 128 as the controls, on BF16, F16 and Q8_0 weights -
// the formats that reach llm_attention_q_matmul_rope_decode_f32_f32 at one token. The kernel wrote each rotated
// pair at `channel rem head_size`, which the Loom compiler could bound only for a power of two, so the other head
// sizes failed to compile and the decode returned an error (engine #1926). Each graph runs on the HRX device
// itself (no scheduler), must be planned on that kernel, and is compared with the CPU backend (normalized MSE, as
// test-backend-ops).

#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

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

static constexpr double  kMaxNmse  = 5e-4;
static constexpr int32_t kPosition = 37;
static const char *      kKernel   = "loom_libs:llm_attention_q_matmul_rope_decode_f32_f32";

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
    int64_t   head_size;
    int64_t   head_count;
};

struct Data {
    std::vector<uint8_t> weights;
    std::vector<float>   x;
};

static Data make_data(const Case & c, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    const int64_t                         size = c.head_size * c.head_count;
    std::vector<float>                    w(static_cast<size_t>(size * size));
    for (float & v : w) {
        v = uniform(rng);
    }
    Data d;
    d.weights.resize(ggml_row_size(c.type, size) * size);
    ggml_quantize_chunk(c.type, w.data(), d.weights.data(), 0, size, size, nullptr);
    d.x.resize(static_cast<size_t>(size));
    for (float & v : d.x) {
        v = uniform(rng);
    }
    return d;
}

enum class Run { Done, NotAdmitted, ComputeFailed };

static Run run(ggml_backend_t backend, const Case & c, const Data & d, std::vector<float> & result, std::string * plan) {
    ggml_init_params params = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    const int64_t size      = c.head_size * c.head_count;
    ggml_tensor * weights   = ggml_new_tensor_2d(ctx, c.type, size, size);
    ggml_tensor * x         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, size, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * q         = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, weights, x), c.head_size, c.head_count, 1);
    ggml_tensor * out       = ggml_rope_ext(ctx, q, positions, nullptr, static_cast<int>(c.head_size),
                                            GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * graph     = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (plan != nullptr) {
        *plan = ggml_backend_supports_op(backend, out) ? hrx_plan(graph) : "";
        if (plan->empty()) {
            ggml_free(ctx);
            return Run::NotAdmitted;
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    ggml_backend_tensor_set(weights, d.weights.data(), 0, d.weights.size());
    ggml_backend_tensor_set(x, d.x.data(), 0, d.x.size() * sizeof(float));
    ggml_backend_tensor_set(positions, &kPosition, 0, sizeof(kPosition));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (computed) {
        result.resize(static_cast<size_t>(ggml_nelements(out)));
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return computed ? Run::Done : Run::ComputeFailed;
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
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // the per-case lines must survive the abort in REQUIRE
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-attention-q-decode-head-size: no HRX0 device, skipped\n");
        return 77;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);

    std::vector<Case> cases;
    for (ggml_type type : { GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        cases.push_back({ type, 48, 6 });    // stories15M
        cases.push_back({ type, 96, 8 });
        cases.push_back({ type, 160, 16 });  // Sarashina2.2-3B
        cases.push_back({ type, 64, 16 });   // control
        cases.push_back({ type, 128, 8 });   // control
    }
    int      failures = 0;
    uint32_t seed     = 1;
    for (const Case & c : cases) {
        const Data         d = make_data(c, seed++);
        std::string        plan;
        std::vector<float> got;
        std::vector<float> expected;
        const Run          on_hrx = run(hrx, c, d, got, &plan);
        if (on_hrx != Run::Done || plan != kKernel) {
            std::printf("%-5s head_size=%3lld heads=%2lld  %s\n", ggml_type_name(c.type), (long long) c.head_size,
                        (long long) c.head_count,
                        on_hrx == Run::NotAdmitted   ? "not admitted by HRX" :
                        on_hrx == Run::ComputeFailed ? "planned, but the compute failed" :
                                                       ("planned on another kernel: " + plan).c_str());
            ++failures;
            continue;
        }
        REQUIRE(run(cpu, c, d, expected, nullptr) == Run::Done);
        const double e  = nmse(got, expected);
        const bool   ok = e <= kMaxNmse;
        std::printf("%-5s head_size=%3lld heads=%2lld  %s nmse=%.3g %s\n", ggml_type_name(c.type),
                    (long long) c.head_size, (long long) c.head_count, plan.c_str(), e, ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::printf("test-hrx-attention-q-decode-head-size: %zu cases, %d failures\n", cases.size(), failures);
    REQUIRE(failures == 0);
    return 0;
}
