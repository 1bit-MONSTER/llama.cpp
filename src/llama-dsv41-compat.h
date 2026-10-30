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
//
// Port shims for the DeepSeek-V4.1 (deepseek41) backport of upstream #31086.
//
// The upstream patch is written against ggml/llama APIs that landed upstream later than this tree
// (mixed token/embd ubatches, precision hints, rope offset dims, row prefetch). Each shim below is
// the equivalent construction on our side: no new ggml ops, so every backend (HRX included) keeps
// seeing the ops it already implements.

#pragma once

#include "ggml.h"
#include "llama-batch.h"

#include <cstdint>

// upstream ASSERT_EMBD_OR_TOKEN(ubatch): a ubatch carries tokens or embeddings, never both.
// This tree has no mixed ubatches (llama_ubatch has no per-token type array), so the check is the
// whole story here.
#define DSV41_ASSERT_EMBD_OR_TOKEN(ubatch) \
    GGML_ASSERT(!(ubatch).token == !(ubatch).embd || (ubatch).n_tokens == 0)

// upstream: `ubatch.is_mixed() && ubatch.type[i]` - is token i an embedding rather than a token id?
// Without mixed ubatches the answer is uniform for the whole batch.
static inline bool dsv41_ubatch_i_is_embd(const llama_ubatch & ubatch, int i) {
    GGML_UNUSED(i);
    return !ubatch.token;
}

// upstream ggml_prec_set_acc(): minimum accumulator precision the implementation may use.
// Our ggml only carries the hint for MUL_MAT (op_params), where upstream's deprecated
// ggml_mul_mat_set_prec() sets exactly the same flag.
static inline void dsv41_prec_acc(ggml_tensor * t, ggml_prec prec) {
    if (t != nullptr && t->op == GGML_OP_MUL_MAT) {
        ggml_mul_mat_set_prec(t, prec);
    }
}

// upstream ggml_prec_set_src(): an *allowance* to store src[idx] in a lower rank (e.g. quantize an
// F32 operand down to Q8). This tree has no such hint; not taking the allowance only means the
// implementation keeps reading the operand at full precision.
static inline void dsv41_prec_src(ggml_tensor * t, ggml_prec prec, int idx) {
    GGML_UNUSED(t);
    GGML_UNUSED(prec);
    GGML_UNUSED(idx);
}

// one dim-0 slice of an n-dimensional tensor, as a view
static inline ggml_tensor * dsv41_view_dim0(ggml_context * ctx, ggml_tensor * x, int64_t off, int64_t ne0) {
    switch (ggml_n_dims(x)) {
        case 1:  return ggml_view_1d(ctx, x, ne0, off*x->nb[0]);
        case 2:  return ggml_view_2d(ctx, x, ne0, x->ne[1], x->nb[1], off*x->nb[0]);
        case 3:  return ggml_view_3d(ctx, x, ne0, x->ne[1], x->ne[2], x->nb[1], x->nb[2], off*x->nb[0]);
        default: return ggml_view_4d(ctx, x, ne0, x->ne[1], x->ne[2], x->ne[3],
                                     x->nb[1], x->nb[2], x->nb[3], off*x->nb[0]);
    }
}

// upstream ggml_rope_set_offset(a, n_offs): rotate the dims [n_offs, n_offs + n_dims) instead of the
// first n_dims dims. Built as concat(unrotated | rope(view) | unrotated) - the rope op itself stays
// a plain contiguous rope on a dim-0 view, which is the shape every backend already implements
// (deepseek4.cpp rotates the trailing dims of K the same way).
static inline ggml_tensor * dsv41_rope_off(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * pos,
        int            n_dims,
        int            n_offs,
        int            mode,
        int            n_ctx_orig,
        float          freq_base,
        float          freq_scale,
        float          ext_factor,
        float          attn_factor,
        float          beta_fast,
        float          beta_slow,
        bool           back = false) {
    const int64_t ne0 = x->ne[0];

    GGML_ASSERT(n_offs >= 0 && n_offs + n_dims <= ne0);

    ggml_tensor * head = n_offs > 0 ? dsv41_view_dim0(ctx, x, 0, n_offs) : nullptr;
    ggml_tensor * tail = n_offs + n_dims < ne0 ? dsv41_view_dim0(ctx, x, n_offs + n_dims, ne0 - n_offs - n_dims) : nullptr;

    ggml_tensor * pe = dsv41_view_dim0(ctx, x, n_offs, n_dims);
    pe = back
        ? ggml_rope_ext_back(ctx, pe, pos, nullptr, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext     (ctx, pe, pos, nullptr, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

    if (head) {
        pe = ggml_concat(ctx, head, pe, 0);
    }
    if (tail) {
        pe = ggml_concat(ctx, pe, tail, 0);
    }

    return pe;
}
