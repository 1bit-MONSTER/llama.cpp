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

// engine#3531: the CPU repack path (Q4_0 / IQ4_NL weights, re-laid out for the AVX2 / NEON GEMM) quantized its src1
// four rows at a time assuming packed rows. Mamba's ssm_dt multiplies a view - the dt slice of the x_proj output, row
// stride = the whole x_proj row - so rows 1..3 of every group were read from the wrong columns: a Q4_0 Mamba-370M
// scored KLD ~21 against its f16, and ~0.28 with --no-repack. This compares MUL_MAT with the weight in the repack
// buffer against the same quantized bytes in a plain CPU buffer, for packed and strided src1.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_buffer_type_t repack_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra == nullptr) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * b = get_extra(dev); b != nullptr && *b != nullptr; ++b) {
        if (std::string(ggml_backend_buft_name(*b)).find("REPACK") != std::string::npos) {
            return *b;
        }
    }
    return nullptr;
}

// max |a - b| / max |b| over the output
static int run_case(ggml_backend_t cpu, ggml_backend_buffer_type_t repack, ggml_type type, int64_t k, int64_t n,
                    int64_t m, int64_t row_pad) {
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
    ggml_context * cw = ggml_init(ip);   // the weight in the repack buffer
    ggml_context * cr = ggml_init(ip);   // the same weight in a plain CPU buffer, and everything else
    ggml_tensor * w_repack = ggml_new_tensor_2d(cw, type, k, n);
    ggml_tensor * w_plain  = ggml_new_tensor_2d(cr, type, k, n);
    ggml_tensor * x        = ggml_new_tensor_2d(cr, GGML_TYPE_F32, k + row_pad, m);
    ggml_tensor * xv       = ggml_view_2d(cr, x, k, m, x->nb[1], 0);  // row stride k + row_pad when row_pad > 0
    ggml_tensor * y_repack = ggml_mul_mat(cr, w_repack, xv);
    ggml_tensor * y_plain  = ggml_mul_mat(cr, w_plain, xv);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, repack);
    ggml_backend_buffer_t br = ggml_backend_alloc_ctx_tensors_from_buft(cr, ggml_backend_cpu_buffer_type());
    if (bw == nullptr || br == nullptr) {
        std::printf("FAIL cannot allocate the buffers\n");
        return 1;
    }

    std::vector<float> wf(k * n), xf((k + row_pad) * m);
    for (size_t i = 0; i < wf.size(); i++) {
        wf[i] = std::sin(0.37f * (float) i) * (0.5f + 0.5f * std::cos(0.011f * (float) i));
    }
    for (size_t i = 0; i < xf.size(); i++) {
        // the padding columns get values far from the real ones, so reading them shows
        xf[i] = (int64_t) (i % (k + row_pad)) < k ? std::cos(0.23f * (float) i) : 100.0f + (float) (i % 7);
    }
    std::vector<uint8_t> wq(ggml_row_size(type, k) * n);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, n, k, nullptr);
    ggml_backend_tensor_set(w_repack, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(w_plain, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_cgraph * g = ggml_new_graph(cr);
    ggml_build_forward_expand(g, y_repack);
    ggml_build_forward_expand(g, y_plain);
    int rc = 0;
    if (ggml_backend_graph_compute(cpu, g) != GGML_STATUS_SUCCESS) {
        std::printf("FAIL graph compute\n");
        rc = 1;
    } else {
        std::vector<float> a(n * m), b(n * m);
        ggml_backend_tensor_get(y_repack, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(y_plain, b.data(), 0, b.size() * sizeof(float));
        double diff = 0.0, scale = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            diff  = std::fmax(diff, std::fabs((double) a[i] - (double) b[i]));
            scale = std::fmax(scale, std::fabs((double) b[i]));
        }
        const double rel = scale > 0.0 ? diff / scale : diff;
        // both arms quantize src1 to q8_0 the same way; the repacked GEMM reorders the sums, nothing more
        const bool ok = rel < 1e-3;
        std::printf("%s %s k=%lld n=%lld m=%lld src1 row stride %lld floats (%s): max rel diff %.3g\n", ok ? "ok  " : "FAIL",
                    ggml_type_name(type), (long long) k, (long long) n, (long long) m, (long long) (k + row_pad),
                    row_pad ? "strided" : "packed", rel);
        rc = ok ? 0 : 1;
    }
    ggml_backend_buffer_free(bw);
    ggml_backend_buffer_free(br);
    ggml_free(cw);
    ggml_free(cr);
    return rc;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_buffer_type_t repack = repack_buft();
    if (repack == nullptr) {
        std::printf("SKIP no CPU repack buffer type in this build\n");
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        std::printf("FAIL no CPU backend\n");
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, 4);
    int fails = 0;
    for (ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL }) {
        // Mamba-370M's ssm_dt: K = dt_rank 64, the view's rows are 64 + 2 * d_state(16) = 96 floats apart
        fails += run_case(cpu, repack, type, 64, 256, 8, 0);    // packed: the control
        fails += run_case(cpu, repack, type, 64, 256, 8, 32);   // strided, 2 full groups of 4 rows
        fails += run_case(cpu, repack, type, 64, 256, 6, 32);   // strided, a group of 4 + a 2-row tail
        fails += run_case(cpu, repack, type, 256, 256, 1, 64);  // strided, one row (the tail path only)
    }
    ggml_backend_free(cpu);
    std::printf("test-cpu-repack-strided: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
