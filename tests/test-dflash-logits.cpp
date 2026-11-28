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

// engine#3627: llama-context's logits copy (src/llama-logits-rows.h) when a DFlash drafter's logits are narrower than
// the context's n_vocab (the ufrik Inkling-2-Mini drafter lists 201024 tokens, its logits are cropped to the target's
// 200058). Each row must land at the n_vocab stride with -INF beyond the real columns; equal widths stay a plain copy.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama-logits-rows.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int fails = 0;
static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += ok ? 0 : 1;
}

static std::vector<float> randn(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> v(n);
    for (auto & x : v) x = nd(rng);
    return v;
}

// rows of ne0 at stride n_vocab, then -INF: the property every case checks
static bool widened_ok(const std::vector<float> & src, int64_t ne0, const std::vector<float> & dst, int64_t n_vocab, int64_t n_rows) {
    for (int64_t r = 0; r < n_rows; r++) for (int64_t c = 0; c < n_vocab; c++) {
        const float v = dst[size_t(r * n_vocab + c)];
        if (c < ne0 ? v != src[size_t(r * ne0 + c)] : !(std::isinf(v) && v < 0)) return false;
    }
    return true;
}

// the backend path: a CPU-backend tensor [ne0, n_rows] read with llama_get_logit_rows into rows of n_vocab
static std::vector<float> via_backend(const std::vector<float> & src, int64_t ne0, int64_t n_rows, int64_t n_vocab) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, n_rows);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    ggml_backend_tensor_set(t, src.data(), 0, src.size() * sizeof(float));
    std::vector<float> dst(size_t(n_rows * n_vocab), 12345.0f);   // poison: every slot must be written
    llama_get_logit_rows(cpu, t, dst.data(), n_rows, n_vocab);
    ggml_backend_synchronize(cpu);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return dst;
}

int main() {
    {   // 1. the pure host widening, small
        const int64_t ne0 = 5, n_vocab = 8, n_rows = 3;
        auto src = randn(size_t(ne0 * n_rows), 1);
        std::vector<float> dst(size_t(n_vocab * n_rows), 12345.0f);
        llama_widen_logit_rows(src.data(), ne0, dst.data(), n_vocab, n_rows);
        check(widened_ok(src, ne0, dst, n_vocab, n_rows), "5 -> 8, 3 rows: each row at the n_vocab stride, the rest -INF");
    }
    {   // 2. the ufrik shape through the backend copy, as llama-context runs it
        const int64_t ne0 = 200058, n_vocab = 201024, n_rows = 3;
        auto src = randn(size_t(ne0 * n_rows), 2);
        auto dst = via_backend(src, ne0, n_rows, n_vocab);
        check(widened_ok(src, ne0, dst, n_vocab, n_rows), "ufrik 200058 -> 201024 via the CPU backend: rows aligned, 966 -INF each");
        bool argmax_real = true;
        for (int64_t r = 0; r < n_rows; r++) {
            int64_t best = 0;
            for (int64_t c = 1; c < n_vocab; c++) if (dst[size_t(r * n_vocab + c)] > dst[size_t(r * n_vocab + best)]) best = c;
            argmax_real &= best < ne0;
        }
        check(argmax_real, "ufrik: the argmax of every row is a real token");
    }
    {   // 3. equal widths (every model whose tokenizers agree): the plain copy, bit-identical
        const int64_t n = 64, n_rows = 4;
        auto src = randn(size_t(n * n_rows), 3);
        auto dst = via_backend(src, n, n_rows, n);
        check(std::memcmp(dst.data(), src.data(), src.size() * sizeof(float)) == 0, "equal widths: the plain copy, bit-identical");
    }
    {   // 4. the old behaviour, for contrast: a flat copy of n_rows*ne0 floats puts row 1 at column ne0 of row 0
        const int64_t ne0 = 5, n_vocab = 8, n_rows = 2;
        auto src = randn(size_t(ne0 * n_rows), 4);
        std::vector<float> flat(size_t(n_vocab * n_rows), 0.0f);
        std::memcpy(flat.data(), src.data(), src.size() * sizeof(float));
        check(!widened_ok(src, ne0, flat, n_vocab, n_rows), "a flat copy (the old shape) misaligns the rows: what the fix prevents");
    }
    std::printf("test-dflash-logits: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
