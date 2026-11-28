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

#pragma once
// engine#3627: llama-context copies n_vocab floats per output row out of t_logits. A DFlash drafter crops its logits to
// the ids both tokenizers know, so when the drafter lists more tokens than the target (ufrik) the tensor is narrower.
// Re-stride on the host, not in the graph: no new op, so strict HRX placement is unaffected. Same width = the plain copy.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// n_rows rows of ne0 floats (src) -> n_rows rows of n_vocab floats (dst): the first min(ne0, n_vocab) columns copied,
// columns [ne0, n_vocab) = -INF when narrower, columns past n_vocab dropped when wider
static inline void llama_restride_logit_rows(const float * src, int64_t ne0, float * dst, int64_t n_vocab, int64_t n_rows) {
    const int64_t n = std::min(ne0, n_vocab);
    for (int64_t r = 0; r < n_rows; r++) {
        std::memcpy(dst + r*n_vocab, src + r*ne0, size_t(n)*sizeof(float));
        std::fill(dst + r*n_vocab + n, dst + (r + 1)*n_vocab, -INFINITY);
    }
}

// the copy llama-context does, for logits whose row width may differ from n_vocab
static inline void llama_get_logit_rows(ggml_backend_t backend, ggml_tensor * t, float * dst, int64_t n_rows, int64_t n_vocab) {
    const int64_t ne0 = t->ne[0];
    if (ne0 == n_vocab) {
        ggml_backend_tensor_get_async(backend, t, dst, 0, size_t(n_rows*n_vocab)*sizeof(float));
        return;
    }
    // the rows are read at stride ne0: a narrower producer must make its logits contiguous (dflash.cpp's crop is a cont)
    GGML_ASSERT(ggml_is_contiguous(t));
    if (ne0 > n_vocab) {
        // no known GGUF does this; drop the extra columns rather than abort in decode()
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "%s: logits rows are %lld wide, the vocabulary %lld: keeping the first %lld columns\n",
                    __func__, (long long) ne0, (long long) n_vocab, (long long) n_vocab);
        }
    }
    static thread_local std::vector<float> rows;   // reused across draft steps
    rows.resize(size_t(n_rows*ne0));
    ggml_backend_tensor_get_async(backend, t, rows.data(), 0, rows.size()*sizeof(float));
    ggml_backend_synchronize(backend);   // `rows` is read right below
    llama_restride_logit_rows(rows.data(), ne0, dst, n_vocab, n_rows);
}
