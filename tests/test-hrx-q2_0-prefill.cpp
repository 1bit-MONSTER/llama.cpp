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

// Q2_0 prompt matmuls on HRX's q8_1 x4 int8 WMMA kernel (weight format 42).
// Known answer: activations are integers in -127..127 with |max| 127 in every 32-value block, so their q8_1
// quantization is exact (d = 1) and their block sums are nonzero; the weights are raw Q2_0 blocks (every code byte,
// positive, negative, small and large f16 scales). Q2_0 codes 0..3 mean -1..2: a kernel that treated them as
// unsigned, or dropped an offset term, is off by d x (block sum of the activations) and fails the 1e-5 bound.
// HRX must match an f64 reference from ggml's dequantize_row_q2_0 to 1e-5 of sum |w x| per output. Then random
// weights (ggml_quantize_chunk) and activations against the CPU backend (normalized MSE). Each graph runs on the HRX
// device itself (no scheduler) and must be planned on the q8_1 x4 kernel.

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

static constexpr double kMaxNmse = 5e-4;

struct Case {
    int64_t input_size;
    int64_t output_size;
    int64_t tokens;
};

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

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

static void run(ggml_backend_t backend, const Case & c, const std::vector<uint8_t> & weights_host,
                const std::vector<float> & x_host, std::vector<float> & result, std::string * plan) {
    ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, c.input_size, c.output_size);
    ggml_tensor * x       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.input_size, c.tokens);
    ggml_tensor * out     = ggml_mul_mat(ctx, weights, x);
    ggml_cgraph * graph   = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (plan != nullptr) {
        REQUIRE(ggml_backend_supports_op(backend, out));
        *plan = hrx_plan(graph);
        REQUIRE(!plan->empty());
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    ggml_backend_tensor_set(weights, weights_host.data(), 0, weights_host.size());
    ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    result.resize(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-q2_0-prefill: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q2_0);
    REQUIRE(traits->to_float != nullptr);

    const Case cases[] = { { 256, 64, 256 }, { 1024, 128, 256 }, { 5120, 192, 512 }, { 2048, 64, 1024 } };
    int        failures = 0;
    uint32_t   seed     = 1;
    for (const Case & c : cases) {
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q2_0, c.input_size);
        std::mt19937 rng(seed++);

        // Known answer: raw blocks, integer activations with |max| 127 per 32 values.
        std::vector<uint8_t> raw(row_bytes * c.output_size);
        std::uniform_int_distribution<int> byte(0, 255);
        for (size_t i = 0; i < raw.size(); ++i) {
            raw[i] = static_cast<uint8_t>(byte(rng));
        }
        for (int64_t r = 0; r < c.output_size; ++r) {
            for (int64_t b = 0; b < c.input_size / 64; ++b) {
                uint8_t *    block = raw.data() + r * row_bytes + b * 18;
                const float  sign  = (r + b) % 2 ? -1.0f : 1.0f;
                const float  d     = sign * std::ldexp(1.0f + static_cast<float>((r * 7 + b) % 8) / 8.0f,
                                                       static_cast<int>((r + 3 * b) % 12) - 8);
                const ggml_fp16_t h = ggml_fp32_to_fp16(d);
                std::memcpy(block, &h, sizeof(h));
            }
        }
        std::vector<float> x(static_cast<size_t>(c.input_size * c.tokens));
        std::uniform_int_distribution<int> q(-127, 127);
        for (int64_t t = 0; t < c.tokens; ++t) {
            for (int64_t k = 0; k < c.input_size; ++k) {
                const bool peak         = k % 32 == (t + k / 32) % 32;
                const int  value        = peak ? ((t + k) % 2 ? 127 : -127) : q(rng);
                x[t * c.input_size + k] = static_cast<float>(value);
            }
        }
        std::string        plan;
        std::vector<float> got;
        run(hrx, c, raw, x, got, &plan);
        const bool on_x4 = plan.find("q8_1_x4") != std::string::npos;
        std::vector<float> w(static_cast<size_t>(c.input_size));
        double             worst = 0.0;
        for (int64_t r = 0; r < c.output_size; ++r) {
            traits->to_float(raw.data() + r * row_bytes, w.data(), c.input_size);
            for (int64_t t = 0; t < c.tokens; ++t) {
                double ref = 0.0, mag = 0.0;
                for (int64_t k = 0; k < c.input_size; ++k) {
                    const double p = static_cast<double>(w[k]) * x[t * c.input_size + k];
                    ref += p;
                    mag += std::fabs(p);
                }
                const double err = std::fabs(got[t * c.output_size + r] - ref) / (mag > 0.0 ? mag : 1.0);
                worst            = err > worst ? err : worst;
            }
        }
        const bool kat_ok = on_x4 && worst <= 1e-5;
        std::printf("known answer k=%5lld n=%4lld tokens=%4lld  %-60s worst |err| / sum|w x| = %.3g %s\n",
                    (long long) c.input_size, (long long) c.output_size, (long long) c.tokens, plan.c_str(), worst,
                    kat_ok ? "OK" : "FAIL");
        failures += kat_ok ? 0 : 1;

        // Random weights and activations against the CPU backend.
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float>              wf(static_cast<size_t>(c.input_size * c.output_size));
        for (float & v : wf) {
            v = normal(rng);
        }
        std::vector<uint8_t> quantized(row_bytes * c.output_size);
        ggml_quantize_chunk(GGML_TYPE_Q2_0, wf.data(), quantized.data(), 0, c.output_size, c.input_size, nullptr);
        for (float & v : x) {
            v = normal(rng);
        }
        std::vector<float> expected;
        run(hrx, c, quantized, x, got, nullptr);
        run(cpu, c, quantized, x, expected, nullptr);
        double err = 0.0, ref = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = static_cast<double>(got[i]) - expected[i];
            err += d * d;
            ref += static_cast<double>(expected[i]) * expected[i];
        }
        const double e  = err / ref;
        const bool   ok = e <= kMaxNmse;
        std::printf("vs CPU       k=%5lld n=%4lld tokens=%4lld  nmse=%.3g %s\n", (long long) c.input_size,
                    (long long) c.output_size, (long long) c.tokens, e, ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::printf("test-hrx-q2_0-prefill: %d failures\n", failures);
    REQUIRE(failures == 0);
    return 0;
}
