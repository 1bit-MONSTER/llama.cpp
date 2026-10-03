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

// One Qwen3.5 / Qwen3.8 gated-delta-net (linear attention) layer, built the way llama.cpp builds it
// (src/models/qwen35.cpp build_layer_attn_linear, delta-net-base.cpp build_conv_state / build_recurrent_attn,
// llm_graph_context::build_rs), for n_seqs sequences of n_seq_tokens tokens each:
//
// - conv and recurrent state gathered per sequence from the recurrent cache through s_copy;
// - the conv window (CONCAT of the conv state and the transposed qkv projection), SSM_CONV + SiLU;
// - L2-normalised q/k views, the alpha/beta gate chain, GATED_DELTA_NET with K snapshot slots
//   (K = 1: final state only; K > 1: rollback snapshots, as with --mtp or a drafter, n_rs_seq = K - 1);
// - the new conv / recurrent states copied into cache rows [kv_head, kv_head + n_seqs) of each slot;
// - the gated RMS norm and the output projection, sequences folded into tokens.
//
// Used by test-hrx-gdn-multiseq and by the GATED_DELTA_NET layer cases of test-backend-ops.

#include "ggml.h"

#include <algorithm>
#include <cstdint>

namespace hrx_gdn_layer {

inline constexpr int64_t kConvKernel = 4;

struct Shape {
    const char * name;
    int64_t      n_embd;
    int64_t      v_heads;  // ssm_dt_rank
    int64_t      k_heads;  // ssm_n_group
    int64_t      head;     // ssm_d_state == head_v_dim
    ggml_type    weight_type;
};

// Qwen3.8-27B (and Qwen3.5-27B): n_embd 5120, 48 value heads, 16 key heads, head 128.
inline constexpr Shape kQwen38_27B = { "qwen38-27b", 5120, 48, 16, 128, GGML_TYPE_Q4_K };
// Qwen3.5-35B-A3B / Qwen3.6-35B-A3B linear-attention layers: n_embd 2048, 32 value heads, 16 key heads.
inline constexpr Shape kQwen35_A3B = { "qwen35-a3b", 2048, 32, 16, 128, GGML_TYPE_Q4_K };
// Small shape for fast CPU references.
inline constexpr Shape kTiny = { "tiny", 1024, 8, 4, 128, GGML_TYPE_Q4_K };

struct Case {
    Shape   shape;
    int64_t tokens;         // n_seq_tokens
    int64_t seqs;           // n_seqs
    int64_t mem_size;       // recurrent cache rows per snapshot slot (n_seq_max)
    int64_t kv_head;        // first cache row written
    int64_t snapshots = 1;  // K = n_rs_seq + 1
};

struct Layer {
    ggml_tensor * x          = nullptr;
    ggml_tensor * w_qkv      = nullptr;
    ggml_tensor * w_z        = nullptr;
    ggml_tensor * w_alpha    = nullptr;
    ggml_tensor * w_beta     = nullptr;
    ggml_tensor * w_out      = nullptr;
    ggml_tensor * dt         = nullptr;
    ggml_tensor * a          = nullptr;
    ggml_tensor * conv_w     = nullptr;
    ggml_tensor * norm_w     = nullptr;
    ggml_tensor * conv_cache = nullptr;  // [n_embd_r, mem_size * K]
    ggml_tensor * ssm_cache  = nullptr;  // [n_embd_s, mem_size * K]
    ggml_tensor * s_copy     = nullptr;  // I32 [n_seqs]
    ggml_tensor * out        = nullptr;  // [n_embd, n_seq_tokens * n_seqs]
    ggml_tensor * conv_write = nullptr;  // CPY of the final conv state into slot 0
    ggml_tensor * ssm_write  = nullptr;  // CPY of the recurrent state(s) into the cache
};

inline int64_t graph_size() {
    return 512;
}

// llm_graph_context::build_rs with rs_zero = -1 (no cleared state) and n_rs == n_seqs; states gathered with ggml_get_rows.
inline ggml_tensor * build_rs(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * s, ggml_tensor * s_copy,
                              int64_t state_size, int64_t n_seqs, int64_t rs_head) {
    const int64_t n_rs        = n_seqs;
    ggml_tensor * states      = ggml_reshape_2d(ctx, s, state_size, s->ne[1]);
    ggml_tensor * state_zero  = ggml_view_1d(ctx, states, 0, 0);
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, state_zero, 0));
    ggml_tensor * s_copy_main  = ggml_view_1d(ctx, s_copy, n_seqs, 0);
    ggml_tensor * s_copy_extra = ggml_view_1d(ctx, s_copy, n_rs - n_seqs, n_seqs * s_copy->nb[0]);
    ggml_tensor * output       = ggml_get_rows(ctx, states, s_copy_main);
    ggml_build_forward_expand(gf, output);
    ggml_tensor * extra = ggml_get_rows(ctx, states, s_copy_extra);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, extra, ggml_view_2d(ctx, s, state_size, n_rs - n_seqs, s->nb[1],
                                                                     (rs_head + n_seqs) * s->nb[1])));
    return output;
}

// Builds the layer into gf (leaves allocated in ctx, no data).
inline Layer build_layer(ggml_context * ctx, ggml_cgraph * gf, const Case & c) {
    const Shape & sh       = c.shape;
    const int64_t T        = c.tokens;
    const int64_t S        = c.seqs;
    const int64_t K        = c.snapshots;
    const int64_t d_inner  = sh.head * sh.v_heads;
    const int64_t qkv_dim  = sh.head * sh.k_heads * 2 + d_inner;
    const int64_t n_embd_r = (kConvKernel - 1) * qkv_dim;
    const int64_t n_embd_s = sh.head * sh.head * sh.v_heads;
    Layer         l;

    l.x          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.n_embd, T * S);
    l.w_qkv      = ggml_new_tensor_2d(ctx, sh.weight_type, sh.n_embd, qkv_dim);
    l.w_z        = ggml_new_tensor_2d(ctx, sh.weight_type, sh.n_embd, d_inner);
    l.w_alpha    = ggml_new_tensor_2d(ctx, sh.weight_type, sh.n_embd, sh.v_heads);
    l.w_beta     = ggml_new_tensor_2d(ctx, sh.weight_type, sh.n_embd, sh.v_heads);
    l.w_out      = ggml_new_tensor_2d(ctx, sh.weight_type, d_inner, sh.n_embd);
    l.dt         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sh.v_heads);
    l.a          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sh.v_heads);
    l.conv_w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kConvKernel, qkv_dim);
    l.norm_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sh.head);
    l.conv_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_r, c.mem_size * K);
    l.ssm_cache  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_s, c.mem_size * K);
    l.s_copy     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_name(l.x, "x");
    ggml_set_name(l.a, "ssm_a");
    ggml_set_name(l.s_copy, "s_copy");
    ggml_set_name(l.conv_cache, "conv_cache");
    ggml_set_name(l.ssm_cache, "ssm_cache");
    ggml_set_input(l.x);
    ggml_set_input(l.s_copy);

    // the residual-stream input comes out of an op in a model graph
    ggml_tensor * cur = ggml_scale(ctx, l.x, 1.0f);

    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, l.w_qkv, cur);
    qkv_mixed               = ggml_reshape_3d(ctx, qkv_mixed, qkv_mixed->ne[0], T, S);
    ggml_tensor * z         = ggml_mul_mat(ctx, l.w_z, cur);

    ggml_tensor * beta = ggml_mul_mat(ctx, l.w_beta, cur);
    beta               = ggml_reshape_4d(ctx, beta, 1, sh.v_heads, T, S);
    beta               = ggml_sigmoid(ctx, beta);

    ggml_tensor * alpha = ggml_mul_mat(ctx, l.w_alpha, cur);
    alpha               = ggml_reshape_3d(ctx, alpha, sh.v_heads, T, S);
    ggml_tensor * gate  = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, l.dt)), l.a);
    gate                = ggml_reshape_4d(ctx, gate, 1, sh.v_heads, T, S);

    // build_conv_state
    const size_t  conv_row    = ggml_row_size(l.conv_cache->type, n_embd_r);
    ggml_tensor * conv_states = build_rs(ctx, gf, l.conv_cache, l.s_copy, n_embd_r, S, c.kv_head);
    conv_states               = ggml_reshape_3d(ctx, conv_states, kConvKernel - 1, qkv_dim, S);
    ggml_tensor * conv_input  = ggml_concat(ctx, conv_states, ggml_transpose(ctx, qkv_mixed), 0);
    for (int64_t t = 1; t <= K; ++t) {
        const int64_t s_idx  = std::max<int64_t>(0, conv_input->ne[0] - (kConvKernel - 1) - K + t);
        const int64_t s_slot = K - t;
        ggml_tensor * last   = ggml_view_3d(ctx, conv_input, kConvKernel - 1, qkv_dim, S, conv_input->nb[1],
                                            conv_input->nb[2], ggml_row_size(conv_input->type, s_idx));
        ggml_tensor * write  = ggml_cpy(ctx, last, ggml_view_2d(ctx, l.conv_cache, n_embd_r, S, l.conv_cache->nb[1],
                                                                (s_slot * c.mem_size + c.kv_head) * conv_row));
        ggml_build_forward_expand(gf, write);
        if (s_slot == 0) {
            l.conv_write = write;
        }
    }

    ggml_tensor * state = build_rs(ctx, gf, l.ssm_cache, l.s_copy, n_embd_s, S, c.kv_head);
    state               = ggml_reshape_4d(ctx, state, sh.head, sh.head, sh.v_heads, S);

    ggml_tensor * conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, l.conv_w));
    const size_t  nb1_qkv  = ggml_row_size(conv_out->type, qkv_dim);
    ggml_tensor * q = ggml_view_4d(ctx, conv_out, sh.head, sh.k_heads, T, S, ggml_row_size(conv_out->type, sh.head),
                                   nb1_qkv, nb1_qkv * T, 0);
    ggml_tensor * k = ggml_view_4d(ctx, conv_out, sh.head, sh.k_heads, T, S, ggml_row_size(conv_out->type, sh.head),
                                   nb1_qkv, nb1_qkv * T, sh.head * sh.k_heads * ggml_element_size(conv_out));
    ggml_tensor * v = ggml_view_4d(ctx, conv_out, sh.head, sh.v_heads, T, S, ggml_row_size(conv_out->type, sh.head),
                                   nb1_qkv, nb1_qkv * T, ggml_row_size(conv_out->type, 2 * sh.head * sh.k_heads));
    q = ggml_l2_norm(ctx, q, 1e-6f);
    k = ggml_l2_norm(ctx, k, 1e-6f);

    // build_recurrent_attn
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, K);
    ggml_tensor * output = ggml_view_4d(ctx, result, sh.head, sh.v_heads, T, S, ggml_row_size(result->type, sh.head),
                                        ggml_row_size(result->type, sh.head * sh.v_heads),
                                        ggml_row_size(result->type, sh.head * sh.v_heads * T), 0);
    const size_t ssm_row = ggml_row_size(l.ssm_cache->type, n_embd_s);
    if (K == 1) {
        ggml_tensor * new_state = ggml_view_4d(ctx, result, sh.head, sh.head, sh.v_heads, S,
                                               ggml_row_size(result->type, sh.head),
                                               ggml_row_size(result->type, sh.head * sh.head),
                                               ggml_row_size(result->type, n_embd_s),
                                               ggml_row_size(result->type, sh.head * sh.v_heads * T * S));
        l.ssm_write = ggml_cpy(ctx, new_state, ggml_view_2d(ctx, l.ssm_cache, n_embd_s, S, l.ssm_cache->nb[1],
                                                            c.kv_head * ssm_row));
    } else {
        const int64_t n_written = std::min<int64_t>(T, K);
        ggml_tensor * src       = ggml_view_3d(ctx, result, n_embd_s, S, n_written, ggml_row_size(result->type, n_embd_s),
                                               ggml_row_size(result->type, n_embd_s * S),
                                               ggml_row_size(result->type, sh.head * sh.v_heads * T * S));
        ggml_tensor * dst = ggml_view_3d(ctx, l.ssm_cache, n_embd_s, S, n_written, l.ssm_cache->nb[1],
                                         static_cast<size_t>(c.mem_size) * ssm_row, c.kv_head * ssm_row);
        l.ssm_write = ggml_cpy(ctx, src, dst);
    }
    ggml_build_forward_expand(gf, l.ssm_write);

    // build_norm_gated, then the output projection with sequences folded into tokens
    ggml_tensor * z_4d  = ggml_reshape_4d(ctx, z, sh.head, sh.v_heads, T, S);
    ggml_tensor * norm  = ggml_mul(ctx, ggml_rms_norm(ctx, output, 1e-6f), l.norm_w);
    ggml_tensor * gated = ggml_mul(ctx, norm, ggml_silu(ctx, z_4d));
    ggml_tensor * flat  = ggml_reshape_2d(ctx, gated, d_inner, T * S);
    l.out               = ggml_mul_mat(ctx, l.w_out, flat);
    ggml_set_name(l.out, "linear_attn_out");
    ggml_set_output(l.out);
    ggml_build_forward_expand(gf, l.out);
    return l;
}

// s_copy for a case: sequence s reads cache row kv_head + (n_seqs - 1 - s), the reverse of the rows it writes, so a
// kernel that indexes the state by cache row instead of by sequence reads another sequence's state.
inline int32_t s_copy_value(const Case & c, int64_t s) {
    return static_cast<int32_t>(c.kv_head + c.seqs - 1 - s);
}

// ssm_a = -exp(A_log) in the model: negative, so the gate (softplus(alpha + dt) * a) is a decay.
inline float a_value(int64_t head) {
    return -0.05f - 0.9f * static_cast<float>(head % 7) / 7.0f;
}

}  // namespace hrx_gdn_layer
