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

// Known-answer test for a MUL_MAT hinted GGML_HINT_SRC0_IS_HADAMARD on HRX0 (ops/hadamard_f32.loom
// through dispatch-hadamard.cpp). Each shape is computed on HRX0 directly (no scheduler, so it
// cannot fall back to the CPU) and on the CPU backend (its fwht path for the same hint), and for
// small blocks also against a double-precision dense product. Every shape runs four times and the
// HRX results must be bitwise equal across runs (races hide in single runs).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Shape {
    int64_t n;
    int64_t rows;
};

std::vector<float> sylvester(int64_t n) {
    std::vector<float> h(n * n);
    const float        scale = 1.0f / std::sqrt((float) n);
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            h[r * n + c] = __builtin_parityll(r & c) ? -scale : scale;
        }
    }
    return h;
}

bool run(ggml_backend_t backend, const Shape & s, const std::vector<float> & rot, const std::vector<float> & x,
         std::vector<float> & out) {
    ggml_init_params params = { 3 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    a      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.n, s.n);
    ggml_tensor *    b      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.n, s.rows);
    ggml_tensor *    y      = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_hint(y, GGML_HINT_SRC0_IS_HADAMARD);
    ggml_cgraph *         graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool                  ok     = buffer != nullptr;
    if (ok) {
        ggml_backend_tensor_set(a, rot.data(), 0, rot.size() * sizeof(float));
        ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            out.resize(s.n * s.rows);
            ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
        }
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return ok;
}

double max_rel_error(const std::vector<float> & got, const std::vector<double> & want) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs(got[i] - want[i]) / std::max(1.0, std::fabs(want[i])));
    }
    return worst;
}

}  // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HRX0");
    if (dev == nullptr) {
        printf("test-hrx-hadamard: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    // decode-sized and prompt-sized rows, including Bonsai's 1024-block rotation at 512 tokens
    const Shape shapes[] = { { 64, 1 }, { 64, 7 }, { 1024, 1 }, { 1024, 17 }, { 1024, 8704 }, { 4096, 3 } };
    std::mt19937 rng(20261002);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int failures = 0;
    for (const Shape & s : shapes) {
        const std::vector<float> rot = sylvester(s.n);
        std::vector<float>       x(s.n * s.rows);
        for (float & v : x) {
            v = dist(rng);
        }
        std::vector<float> want_cpu;
        if (!run(cpu, s, rot, x, want_cpu)) {
            printf("n=%lld rows=%lld: CPU compute failed\n", (long long) s.n, (long long) s.rows);
            ++failures;
            continue;
        }
        std::vector<double> want(want_cpu.begin(), want_cpu.end());
        if (s.n <= 1024 && s.rows <= 17) {  // dense double-precision product
            for (int64_t r = 0; r < s.rows; ++r) {
                for (int64_t j = 0; j < s.n; ++j) {
                    double sum = 0.0;
                    for (int64_t i = 0; i < s.n; ++i) {
                        sum += (double) rot[j * s.n + i] * x[r * s.n + i];
                    }
                    want[r * s.n + j] = sum;
                }
            }
        }
        std::vector<float> first;
        for (int rep = 0; rep < 4; ++rep) {
            std::vector<float> got;
            if (!run(hrx, s, rot, x, got)) {
                printf("n=%lld rows=%lld rep %d: HRX0 compute failed\n", (long long) s.n, (long long) s.rows, rep);
                ++failures;
                break;
            }
            const double err = max_rel_error(got, want);
            const bool   same = rep == 0 || std::memcmp(got.data(), first.data(), got.size() * sizeof(float)) == 0;
            if (rep == 0) {
                first = got;
            }
            const bool ok = err <= 1e-5 && same;
            printf("n=%-5lld rows=%-5lld rep %d: max rel error %.2e%s %s\n", (long long) s.n, (long long) s.rows, rep,
                   err, same ? "" : " (differs from rep 0)", ok ? "ok" : "FAIL");
            failures += ok ? 0 : 1;
        }
    }
    ggml_backend_free(hrx);
    ggml_backend_free(cpu);
    printf(failures == 0 ? "test-hrx-hadamard: PASS\n" : "test-hrx-hadamard: FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
