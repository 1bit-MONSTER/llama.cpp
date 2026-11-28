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
// Widen on the host, not in the graph: no new op, so strict HRX placement is unaffected. Same width = the plain copy.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// n_rows rows of ne0 floats (src) -> n_rows rows of n_vocab floats (dst), columns [ne0, n_vocab) = -INF
static inline void llama_widen_logit_rows(const float * src, int64_t ne0, float * dst, int64_t n_vocab, int64_t n_rows) {
    for (int64_t r = 0; r < n_rows; r++) {
        std::memcpy(dst + r*n_vocab, src + r*ne0, size_t(ne0)*sizeof(float));
        std::fill(dst + r*n_vocab + ne0, dst + (r + 1)*n_vocab, -INFINITY);
    }
}

// the copy llama-context does, for logits that may be narrower than n_vocab
static inline void llama_get_logit_rows(ggml_backend_t backend, ggml_tensor * t, float * dst, int64_t n_rows, int64_t n_vocab) {
    const int64_t ne0 = t->ne[0];
    if (ne0 == n_vocab) {
        ggml_backend_tensor_get_async(backend, t, dst, 0, size_t(n_rows*n_vocab)*sizeof(float));
        return;
    }
    GGML_ASSERT(ne0 < n_vocab && "logits wider than the vocabulary");
    std::vector<float> rows(size_t(n_rows*ne0));
    ggml_backend_tensor_get_async(backend, t, rows.data(), 0, rows.size()*sizeof(float));
    ggml_backend_synchronize(backend);   // `rows` is local: it must be filled before it goes out of scope
    llama_widen_logit_rows(rows.data(), ne0, dst, n_vocab, n_rows);
}
