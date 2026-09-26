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

#include "models.h"

#include "llama-memory-recurrent.h"

#include <cmath>

// Zamba v1 (Zyphra/Zamba-7B-v1). Every layer is a Mamba-1 block; every attn_layer_period-th one
// (the "hybrid" layers) first runs a transformer block shared by all of them:
//   t = shared_ffn(shared_attn(norm(concat(h, embeddings))))    (no residual inside, no RoPE)
//   h = h + mamba(norm(h + linear_il(t)))
// The Mamba-1 block is split into n_mamba_heads heads with their own x_proj / dt_proj; the scan
// takes their B/C as groups. The shared block is stored once and every hybrid layer points to it.
// Ref: transformers/models/zamba/modeling_zamba.py.

void llama_model_zamba::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    // the heads are Mamba-1 heads, not Mamba-2 groups: keep ssm_n_group at 0 so the conv state
    // stays (d_conv - 1) * d_inner
    ml.get_key(LLM_KV_SSM_GROUP_COUNT, n_mamba_heads, false);
    GGML_ASSERT(n_mamba_heads > 0 && hparams.ssm_d_inner % n_mamba_heads == 0);

    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), true);

    // attention runs on concat(h, embeddings): the scale is (head_dim / 2)^-0.5
    hparams.f_attention_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k()) / 2.0f);

    switch (hparams.n_layer()) {
        case 76: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zamba::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t dt_rank  = hparams.ssm_dt_rank;
    const int64_t n_heads  = n_mamba_heads;
    const int64_t head_dim = d_inner / n_heads;
    const int64_t n_attn   = 2 * n_embd;  // attention input: concat(h, embeddings)

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    int shared = -1;  // the hybrid layer whose tensors hold the shared transformer block
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm    = create_tensor(tn(LLM_TENSOR_ATTN_NORM,  "weight", i), {n_embd}, 0);
        layer.ssm_in       = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), {n_embd, 2*d_inner}, 0);
        layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), {d_inner}, 0);
        layer.ssm_x        = create_tensor(tn(LLM_TENSOR_SSM_X,      "weight", i), {head_dim, dt_rank + 2*d_state, n_heads}, 0);
        layer.ssm_dt       = create_tensor(tn(LLM_TENSOR_SSM_DT,     "weight", i), {dt_rank, head_dim, n_heads}, 0);
        layer.ssm_dt_b     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), {d_inner}, 0);
        layer.ssm_a        = create_tensor(tn(LLM_TENSOR_SSM_A,                i), {d_state, d_inner}, 0);
        layer.ssm_d        = create_tensor(tn(LLM_TENSOR_SSM_D,                i), {d_inner}, 0);
        layer.ssm_out      = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", i), {d_inner, n_embd}, 0);

        if (hparams.n_head_kv(i) == 0) {
            continue;
        }
        layer.ssm_mix = create_tensor(tn(LLM_TENSOR_SSM_MIX, "weight", i), {n_embd, n_embd}, 0);
        if (shared < 0) {
            shared = i;
            const int64_t n_q  = n_head * n_embd_head_k;
            const int64_t n_kv = hparams.n_head_kv(i) * n_embd_head_k;
            layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_attn}, 0);
            layer.wq       = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_attn, n_q}, 0);
            layer.wk       = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_attn, n_kv}, 0);
            layer.wv       = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_attn, n_kv}, 0);
            layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_q, n_embd}, 0);
            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        } else {
            const auto & s = layers[shared];
            layer.attn_post_norm = s.attn_post_norm;
            layer.wq = s.wq; layer.wk = s.wk; layer.wv = s.wv; layer.wo = s.wo;
            layer.ffn_norm = s.ffn_norm;
            layer.ffn_gate = s.ffn_gate; layer.ffn_up = s.ffn_up; layer.ffn_down = s.ffn_down;
        }
    }
    GGML_ASSERT(shared >= 0 && "zamba: no hybrid layer carries the shared transformer block");
}

std::unique_ptr<llm_graph_context> llama_model_zamba::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Mamba-1 with per-head x_proj / dt_proj: as llm_build_mamba_base::build_mamba_layer, except
// x_db and dt come from batched matmuls over the heads, and B / C enter the scan as groups.
ggml_tensor * llama_model_zamba::graph::build_zamba_mamba(llm_graph_input_rs * inp, ggml_tensor * cur,
                                                           const llama_model & model, int il) {
    const auto * mctx_cur = inp->mctx;
    const auto   kv_head  = mctx_cur->get_head();
    const auto & layer    = model.layers[il];

    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t d_state      = hparams.ssm_d_state;
    const int64_t dt_rank      = hparams.ssm_dt_rank;
    const int64_t n_heads      = layer.ssm_x->ne[2];
    const int64_t head_dim     = d_inner / n_heads;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_tok        = n_seq_tokens * n_seqs;

    GGML_ASSERT(n_seqs != 0 && ubatch.equal_seqs() && ubatch.n_tokens == n_tok);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    conv = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * xz = build_lora_mm(layer.ssm_in, cur);
    ggml_tensor * x  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    ggml_tensor * z  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));

    {
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2],
                                               n_seq_tokens*(conv_x->nb[0]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv,
                ggml_view_1d(ctx0, conv_states_all, (d_conv - 1)*d_inner*n_seqs,
                             kv_head*(d_conv - 1)*d_inner*ggml_element_size(conv_states_all))));
        x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);
        x = ggml_add(ctx0, x, layer.ssm_conv1d_b);
        x = ggml_silu(ctx0, x);
    }

    // per-head x_proj: {head_dim, n_tok, n_heads} => {dt_rank + 2*d_state, n_tok, n_heads}
    ggml_tensor * xh   = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, x, head_dim, n_heads, n_tok), 0, 2, 1, 3));
    ggml_tensor * x_db = ggml_mul_mat(ctx0, layer.ssm_x, xh);
    cb(x_db, "ssm_x_db", il);

    // per-head dt_proj: {dt_rank, n_tok, n_heads} => {head_dim, n_tok, n_heads} => {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * dt = ggml_view_3d(ctx0, x_db, dt_rank, n_tok, n_heads, x_db->nb[1], x_db->nb[2], 0);
    dt = ggml_mul_mat(ctx0, layer.ssm_dt, ggml_cont(ctx0, dt));
    dt = ggml_cont(ctx0, ggml_permute(ctx0, dt, 0, 2, 1, 3));
    dt = ggml_reshape_3d(ctx0, dt, d_inner, n_seq_tokens, n_seqs);
    dt = ggml_add(ctx0, dt, layer.ssm_dt_b);

    // B, C: {d_state, n_tok, n_heads} => {d_state, n_heads (groups), n_seq_tokens, n_seqs}
    auto group_view = [&](size_t offset) {
        ggml_tensor * t = ggml_view_3d(ctx0, x_db, d_state, n_tok, n_heads, x_db->nb[1], x_db->nb[2],
                                       offset*ggml_element_size(x_db));
        t = ggml_cont(ctx0, ggml_permute(ctx0, t, 0, 2, 1, 3));
        return ggml_reshape_4d(ctx0, t, d_state, n_heads, n_seq_tokens, n_seqs);
    };
    ggml_tensor * B = group_view(dt_rank);
    ggml_tensor * C = group_view(dt_rank + d_state);

    ggml_tensor * x_in = x;
    x = ggml_reshape_4d(ctx0, x, 1, d_inner, n_seq_tokens, n_seqs);

    auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
        ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, 1, d_inner, mctx_cur->get_size());
        return ggml_ssm_scan(ctx, ssm, x, dt, layer.ssm_a, B, C, ids, /*K=*/1);
    };
    ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs, get_ssm_rows);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0,
            ggml_view_1d(ctx0, y_ssm, d_state*d_inner*n_seqs, x->nb[3]*x->ne[3]),
            ggml_view_1d(ctx0, ssm_states_all, d_state*d_inner*n_seqs,
                         kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

    ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[2], x->nb[3], 0);
    y = ggml_add(ctx0, y, ggml_mul(ctx0, x_in, layer.ssm_d));
    y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

    cur = build_lora_mm(layer.ssm_out, y);
    return ggml_reshape_2d(ctx0, cur, cur->ne[0], n_tok);
}

llama_model_zamba::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * emb  = ggml_cast(ctx0, inpL, GGML_TYPE_F32);  // the embeddings every hybrid layer re-reads
    cb(emb, "inp_emb", -1);

    auto * inp = build_inp_mem_hybrid();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * mamba_in = inpL;

        if (hparams.n_head_kv(il) > 0) {
            // shared transformer on concat(h, embeddings)
            ggml_tensor * cur = ggml_concat(ctx0, inpL, emb, 0);
            cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm_concat", il);

            ggml_tensor * Qcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq, cur), n_embd_head, n_head,              n_tokens);
            ggml_tensor * Kcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wk, cur), n_embd_head, hparams.n_head_kv(il), n_tokens);
            ggml_tensor * Vcur = ggml_reshape_3d(ctx0, build_lora_mm(layer.wv, cur), n_embd_head, hparams.n_head_kv(il), n_tokens);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp->get_attn(), layer.wo, NULL, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
            cb(cur, "attn_out", il);

            // gated MLP, exact (erf) GELU as the reference's ACT2FN["gelu"]
            cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
            ggml_tensor * gate = ggml_gelu_erf(ctx0, build_lora_mm(layer.ffn_gate, cur));
            ggml_tensor * up   = build_lora_mm(layer.ffn_up, cur);
            cur = build_lora_mm(layer.ffn_down, ggml_mul(ctx0, gate, up));
            cb(cur, "ffn_out", il);

            cur = build_lora_mm(layer.ssm_mix, cur);
            cb(cur, "linear", il);
            mamba_in = ggml_add(ctx0, inpL, cur);
        }

        ggml_tensor * cur = build_norm(mamba_in, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cur = build_zamba_mamba(inp->get_recr(), cur, model, il);
        cb(cur, "mamba_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur,      inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }
        cur = ggml_add(ctx0, residual, cur);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
