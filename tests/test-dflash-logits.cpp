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

// engine#3627: dflash_pad_logits_to_vocab (src/models/dflash-logits.h) on the CPU backend. A drafter whose tokenizer
// lists more tokens than the target's (ufrik Inkling-2-Mini: 201024 vs 200058) must still hand llama-context rows of
// the drafter's width, with the ids the target does not know at -INF.
#include "ggml.h"
#include "ggml-cpu.h"
#include "models/dflash-logits.h"

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

// pad an [n, m] tensor of known values to n_row; optionally apply a softcap AFTER (as dflash.cpp) or BEFORE the pad
static std::vector<float> run(int64_t n, int64_t m, int64_t n_row, float cap, bool cap_before_pad, std::vector<float> & in,
                              int64_t * ne0_out, bool * same_tensor) {
    ggml_init_params ip = { size_t(n_row) * size_t(m) * 4 * 8 + 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, m);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    in.resize(size_t(n * m));
    for (auto & v : in) v = nd(rng);
    std::memcpy(x->data, in.data(), in.size() * sizeof(float));
    auto softcap = [&](ggml_tensor * t) { return ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, t, 1.0f / cap)), cap); };
    ggml_tensor * cur = x;
    if (cap > 0 && !cap_before_pad) cur = softcap(cur);
    ggml_tensor * padded = dflash_pad_logits_to_vocab(ctx, cur, n_row);
    *same_tensor = padded == cur;
    cur = padded;
    if (cap > 0 && cap_before_pad) cur = softcap(cur);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    *ne0_out = cur->ne[0];
    std::vector<float> out(size_t(cur->ne[0] * cur->ne[1]));
    std::memcpy(out.data(), cur->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

int main() {
    std::vector<float> in;
    int64_t ne0 = 0;
    bool same = false;

    {   // 1. pad 5 -> 8: the real columns unchanged, the added ones -INF
        auto out = run(5, 3, 8, 0.0f, false, in, &ne0, &same);
        bool head = true, tail = true;
        for (int r = 0; r < 3; r++) for (int c = 0; c < 8; c++) {
            const float v = out[size_t(r * 8 + c)];
            if (c < 5) head &= v == in[size_t(r * 5 + c)];
            else       tail &= std::isinf(v) && v < 0;
        }
        check(ne0 == 8, "pad 5 -> 8: width is the drafter's n_row");
        check(head, "pad 5 -> 8: the real columns are bit-identical");
        check(tail, "pad 5 -> 8: the added columns are -INF");
    }
    {   // 2. tokenizers agree: nothing changes (the same tensor comes back)
        run(8, 2, 8, 0.0f, false, in, &ne0, &same);
        check(same && ne0 == 8, "width already n_row: the tensor is returned unchanged");
    }
    {   // 3. the ufrik shape: 200058 real tokens in a 201024-wide row; argmax stays in the real vocabulary
        const int64_t n = 200058, n_row = 201024, m = 2;
        auto out = run(n, m, n_row, 0.0f, false, in, &ne0, &same);
        bool tail = true, argmax_real = true;
        for (int64_t r = 0; r < m; r++) {
            int64_t best = 0;
            for (int64_t c = 0; c < n_row; c++) {
                const float v = out[size_t(r * n_row + c)];
                if (c >= n) tail &= std::isinf(v) && v < 0;
                if (v > out[size_t(r * n_row + best)]) best = c;
            }
            argmax_real &= best < n;
        }
        check(ne0 == n_row, "ufrik 200058 -> 201024: width is the drafter's n_row");
        check(tail, "ufrik: all 966 padded ids are -INF");
        check(argmax_real, "ufrik: the argmax of every row is a real token");
    }
    {   // 4. why dflash.cpp pads AFTER the softcap: before it, -INF becomes the finite -cap
        auto after = run(5, 2, 8, 30.0f, false, in, &ne0, &same);
        auto before = run(5, 2, 8, 30.0f, true, in, &ne0, &same);
        check(std::isinf(after[7]) && after[7] < 0, "softcap then pad (dflash.cpp's order): masked ids stay -INF");
        check(std::isfinite(before[7]) && std::fabs(before[7] + 30.0f) < 1e-3f,
              "pad then softcap would give -cap (finite): the order matters");
    }
    std::printf("test-dflash-logits: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
