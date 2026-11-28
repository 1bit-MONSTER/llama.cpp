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
// engine#3627: llama-context copies model.vocab.n_tokens() floats per output row (the drafter's count). dflash.cpp
// crops the draft logits to the ids both tokenizers know; when the drafter lists more tokens than the target (ufrik),
// that left fewer columns than the copy reads. Widen back with -INF columns: same drafting, the width the copy expects.
// Call it after every logit transform: a softcap maps -INF to the finite -cap.
#include "ggml.h"

#include <cmath>
#include <cstdint>

// [n, n_outputs] (n <= n_row) -> [n_row, n_outputs], columns [n, n_row) = -INF; unchanged when already n_row wide
static inline ggml_tensor * dflash_pad_logits_to_vocab(ggml_context * ctx, ggml_tensor * cur, int64_t n_row) {
    if (cur->ne[0] >= n_row) {
        return cur;
    }
    ggml_tensor * tail = ggml_fill(ctx, ggml_new_tensor_2d(ctx, cur->type, n_row - cur->ne[0], cur->ne[1]), -INFINITY);
    return ggml_concat(ctx, cur, tail, 0);
}
