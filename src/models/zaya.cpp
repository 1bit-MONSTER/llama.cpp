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
// The CCA graph started from Juste-Leo2's ZAYA1 draft for llama.cpp
// (ggml-org/llama.cpp PR #23112, MIT).

#include "models.h"

#include "ggml.h"
#include "llama-memory-recurrent.h"

#include <cmath>

// ZAYA1 (Zyphra). Every layer runs CCA attention and then a MoE, each followed by a
// learned residual scale:
//   - CCA: q and k go through a 2-tap depthwise conv (ssm_conv1d) and a grouped conv
//     (cca_conv_grp) over time, so each sequence carries two recurrent rows: the
//     conv state of q|k (n_embd_r = 2*n_qk) and the previous hidden state
//     (n_embd_s = n_embd); v is two projections, of the current and of the
//     previous hidden state.
//   - MoE router: down_proj -> EDA (adds the previous layer's router state) -> RMSNorm
//     -> MLP (GELU) x2 -> softmax -> top-1 over n_expert + 1 slots, the last being a
//     skip expert with zero output. Experts are pre-stacked (ffn_gate_up_exps).
//   - input_hidden_states_scale/bias apply to the embeddings.

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    if (hparams.f_norm_rms_eps == 0.0f) {
        hparams.f_norm_rms_eps = 1e-5f;
    }
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
    GGML_ASSERT(hparams.ssm_d_conv == 2 && "the CCA graph is written for 2-tap convs");
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
    hparams.n_rot_full = hparams.n_embd_head_k() / 2;
    hparams.n_rot_swa  = hparams.n_embd_head_k() / 2;
    // two recurrent rows per sequence, written whole (as the gated-delta-net models do):
    // r = the 2-tap conv state of q|k, 2*n_qk; s = the previous hidden state, n_embd.
    // With d_conv = 2 and d_state = 1, n_embd_r() = d_inner + 2*n_group and n_embd_s() = d_inner.
    const uint32_t n_qk = (hparams.n_head() + hparams.n_head_kv()) * hparams.n_embd_head_k();
    GGML_ASSERT(2*n_qk > hparams.n_embd && (2*n_qk - hparams.n_embd) % 2 == 0);
    hparams.ssm_d_inner = hparams.n_embd;
    hparams.ssm_d_state = 1;
    hparams.ssm_n_group = (2*n_qk - hparams.n_embd) / 2;
    GGML_ASSERT(hparams.n_embd_r() == 2*n_qk && hparams.n_embd_s() == hparams.n_embd);
    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), true);

    // ZAYA1-74B: sliding-window attention on the layers the pattern marks, with their own rope base
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        // the converter writes a per-layer array; a scalar is a period, as the other SWA archs read it.
        // Without the key (llama_model_saver does not write it) default to ZAYA1-74B's: even layers slide.
        // Read the scalar first: the array read would take a scalar too, as the raw value in every layer.
        uint32_t swa_period = 2;
        if (ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false) ||
            !ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer(), false)) {
            hparams.set_swa_pattern(swa_period);
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zaya::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_ff_exp    = hparams.n_ff_exp;
    const int64_t d_conv      = hparams.ssm_d_conv;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    zaya_input_hs_scale = create_tensor(tn(LLM_TENSOR_INPUT_HIDDEN_STATES_SCALE, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_input_hs_bias  = create_tensor(tn(LLM_TENSOR_INPUT_HIDDEN_STATES_SCALE, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    zaya_res_scale_hs    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,  "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_hs_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,  "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_l    = hparams.n_head(i);
        const int64_t n_head_kv_l = hparams.n_head_kv(i);
        const int64_t n_embd_q    = n_head_l    * n_embd_head;
        const int64_t n_embd_k    = n_head_kv_l * n_embd_head;
        const int64_t n_qk        = n_embd_q + n_embd_k;
        const int64_t n_groups    = n_head_l + n_head_kv_l;
        const int64_t n_ff_l      = hparams.n_ff(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        // every layer has both CCA attention and the MoE
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, 0);
        layer.cca_val_proj1 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ1, "weight", i), {n_embd, n_embd_k / 2}, 0);
        layer.cca_val_proj2 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ2, "weight", i), {n_embd, n_embd_k / 2}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);
        layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "weight", i), {d_conv, n_qk}, 0);
        layer.ssm_conv1d_b   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,   "bias",   i), {n_qk}, TENSOR_NOT_REQUIRED);
        layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i), {n_qk / n_groups, n_qk, d_conv}, 0);  // tap-major
        layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias",   i), {n_qk}, 0);
        layer.cca_k_scale    = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE,  "weight", i), {n_head_kv_l}, 0);

        layer.ffn_gate_inp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_ff_exp}, 0);
        layer.ffn_gate_inp_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "bias",   i), {n_ff_exp}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,     "weight", i), {n_ff_exp}, 0);
        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE,     "weight", i), {n_ff_exp, n_ff_exp}, 0);
        layer.ffn_gate_b     = create_tensor(tn(LLM_TENSOR_FFN_GATE,     "bias",   i), {n_ff_exp}, TENSOR_NOT_REQUIRED);
        layer.zaya_router_mlp2      = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2,      "weight", i), {n_ff_exp, n_ff_exp}, 0);
        layer.zaya_router_mlp2_b    = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2,      "bias",   i), {n_ff_exp}, TENSOR_NOT_REQUIRED);
        layer.zaya_router_mlp4      = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP4,      "weight", i), {n_ff_exp, n_expert + 1}, 0);
        layer.zaya_router_biases    = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_BIASES,    "weight", i), {n_expert + 1}, TENSOR_NOT_REQUIRED);
        layer.zaya_router_eda_scale = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_EDA_SCALE, "weight", i), {n_ff_exp}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, n_ff_l * 2, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_l, n_embd, n_expert}, 0);

        layer.res_scale_hs        = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS,      "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_hs_b      = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS,      "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res       = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES,     "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_b     = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES,     "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_hs_mlp    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_MLP,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_hs_mlp_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_MLP,  "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_mlp   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_MLP, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_mlp_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_MLP, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_zaya::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    }
    return std::make_unique<graph<false>>(*this, params);
}

template <bool iswa>
llama_model_zaya::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_expert    = hparams.n_expert;
    const int64_t n_seqs      = ubatch.n_seqs;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_tokens % n_seqs == 0);

    const int64_t n_seq_tokens = n_tokens / n_seqs;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    if (model.zaya_input_hs_scale != nullptr) {
        if (model.zaya_input_hs_bias != nullptr) {
            inpL = ggml_add(ctx0, inpL, model.zaya_input_hs_bias);
        }
        inpL = ggml_mul(ctx0, inpL, model.zaya_input_hs_scale);
        cb(inpL, "input_hs_scaled", -1);
    }

    auto * inp = [&] {
        if constexpr (iswa) {
            return build_inp_mem_hybrid_iswa();
        } else {
            return build_inp_mem_hybrid();
        }
    }();
    auto * inp_recr = inp->get_recr();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * prev_router = nullptr;

    const auto apply_res_scale = [&](ggml_tensor * x, ggml_tensor * scale, ggml_tensor * bias, const char * name, int il) {
        if (scale == nullptr) {
            return x;
        }
        if (bias != nullptr) {
            x = ggml_add(ctx0, x, bias);
        }
        x = ggml_mul(ctx0, x, scale);
        cb(x, name, il);
        return x;
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_gqa     = n_head / n_head_kv;

        // Zaya 8B (HF ZayaDecoderLayer): EVERY layer runs BOTH blocks.
        //   residual = h (layer input, fp32 stream)
        //   cur = input_layernorm(residual)                       (attn_norm)
        //   attn = CCA(cur)
        //   residual = (attn+pa_hsb)*pa_hss + (residual+pa_rsb)*pa_rss   (res_scale_hs/res)
        //   cur = post_attention_layernorm(residual)              (post_attn_norm)
        //   moe = MoE(cur, prev_router)
        //   h   = (moe+pm_hsb)*pm_hss + (residual+pm_rsb)*pm_rss         (res_scale_hs_mlp/res_mlp)
        ggml_tensor * residual = inpL;

        cur = build_norm(residual, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "input_norm", il);

        // ===== CCA attention (every layer) =====

        const int64_t conv_state_size = 2*n_qk;

        ggml_tensor * conv_states_all = inp_recr->mctx->get_r_l(il);
        ggml_tensor * conv_state = build_rs(inp_recr, conv_states_all, hparams.n_embd_r(), n_seqs);
        conv_state = ggml_reshape_3d(ctx0, conv_state, 2, n_qk, n_seqs);
        cb(conv_state, "cca_conv_state", il);

        ggml_tensor * hs_states_all = inp_recr->mctx->get_s_l(il);
        ggml_tensor * prev_hs = build_rs(inp_recr, hs_states_all, hparams.n_embd_s(), n_seqs);
        cb(prev_hs, "cca_prev_hs", il);

        ggml_tensor * Qraw = ggml_mul_mat(ctx0, layer.wq, cur);
        cb(Qraw, "Qraw", il);
        ggml_tensor * Kraw = ggml_mul_mat(ctx0, layer.wk, cur);
        cb(Kraw, "Kraw", il);

        ggml_tensor * cur_state_src = ggml_cont(ctx0, cur);
        ggml_tensor * cur_seq = ggml_reshape_3d(ctx0, cur_state_src, n_embd, n_seq_tokens, n_seqs);

        ggml_tensor * hs_d = ggml_reshape_3d(ctx0, ggml_cont(ctx0, prev_hs), n_embd, 1, n_seqs);
        if (n_seq_tokens > 1) {
            ggml_tensor * cur_shift = ggml_view_3d(ctx0, cur_seq, n_embd, n_seq_tokens - 1, n_seqs,
                    cur_seq->nb[1],
                    cur_seq->nb[2],
                    0);
            hs_d = ggml_concat(ctx0, hs_d, cur_shift, 1);
        }
        hs_d = ggml_reshape_2d(ctx0, ggml_cont(ctx0, hs_d), n_embd, n_tokens);
        cb(hs_d, "cca_hs_d", il);

        ggml_tensor * V1 = ggml_mul_mat(ctx0, layer.cca_val_proj1, cur);
        cb(V1, "V1", il);
        ggml_tensor * V2 = ggml_mul_mat(ctx0, layer.cca_val_proj2, hs_d);
        cb(V2, "V2", il);
        ggml_tensor * Vcur = ggml_concat(ctx0, V1, V2, 0);
        cb(Vcur, "Vcur", il);

        ggml_tensor * QKraw = ggml_concat(ctx0, Qraw, Kraw, 0);
        cb(QKraw, "QKraw", il);

        ggml_tensor * Qpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Qraw), n_embd_head, n_head, n_tokens);
        ggml_tensor * Kpre = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Kraw), n_embd_head, n_head_kv, n_tokens);

        ggml_tensor * Kpre_grouped = ggml_reshape_4d(ctx0, Kpre, n_embd_head, 1, n_head_kv, n_tokens);
        Kpre_grouped = ggml_repeat_4d(ctx0, Kpre_grouped, n_embd_head, n_gqa, n_head_kv, n_tokens);
        ggml_tensor * Kpre_rep = ggml_reshape_3d(ctx0, Kpre_grouped, n_embd_head, n_head, n_tokens);
        ggml_tensor * qk_mean_q = ggml_scale(ctx0, ggml_add(ctx0, Qpre, Kpre_rep), 0.5f);
        cb(qk_mean_q, "qk_mean_q", il);

        ggml_tensor * Qgroup = ggml_reshape_4d(ctx0, Qpre, n_embd_head, n_gqa, n_head_kv, n_tokens);
        Qgroup = ggml_permute(ctx0, Qgroup, 1, 0, 2, 3);
        Qgroup = ggml_cont(ctx0, Qgroup);
        ggml_tensor * Qmean = ggml_mean(ctx0, Qgroup);
        Qmean = ggml_reshape_3d(ctx0, Qmean, n_embd_head, n_head_kv, n_tokens);
        ggml_tensor * qk_mean_k = ggml_scale(ctx0, ggml_add(ctx0, Qmean, Kpre), 0.5f);
        cb(qk_mean_k, "qk_mean_k", il);

        // [n_qk, T, S] -> [T, n_qk, S]: split the sequences before transposing, or with more than
        // one sequence in the ubatch a channel's row would run across all of them
        ggml_tensor * QKraw_t = ggml_reshape_3d(ctx0, QKraw, n_qk, n_seq_tokens, n_seqs);
        QKraw_t = ggml_cont(ctx0, ggml_transpose(ctx0, QKraw_t));

        ggml_tensor * conv_input = ggml_concat(ctx0, conv_state, QKraw_t, 0);
        cb(conv_input, "cca_conv_input", il);

        ggml_tensor * last_conv_states = ggml_view_3d(ctx0, conv_input, 2, n_qk, n_seqs,
                conv_input->nb[1],
                conv_input->nb[2],
                n_seq_tokens*conv_input->nb[0]);
        cb(last_conv_states, "cca_last_conv_states", il);

        const auto kv_head = inp_recr->mctx->get_head();
        ggml_tensor * conv_state_update_target = ggml_view_2d(ctx0, conv_states_all, conv_state_size, n_seqs,
                conv_states_all->nb[1],
                kv_head*conv_states_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                ggml_reshape_2d(ctx0, ggml_cont(ctx0, last_conv_states), conv_state_size, n_seqs),
                conv_state_update_target));

        ggml_tensor * last_hs = ggml_view_2d(ctx0, cur_seq, n_embd, n_seqs,
                cur_seq->nb[2],
                (n_seq_tokens - 1)*cur_seq->nb[1]);
        ggml_tensor * prev_hs_update_target = ggml_view_2d(ctx0, hs_states_all, n_embd, n_seqs,
                hs_states_all->nb[1],
                kv_head*hs_states_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, last_hs), prev_hs_update_target));

        ggml_tensor * conv_dw = layer.ssm_conv1d;
        if (conv_dw->type != GGML_TYPE_F32) {
            conv_dw = ggml_cont(ctx0, ggml_cast(ctx0, conv_dw, GGML_TYPE_F32));
        }
        ggml_tensor * QK = ggml_ssm_conv(ctx0, conv_input, conv_dw);
        // Grouped conv (2 taps, no padding) as one batched matmul per tap: the weights are
        // stored tap-major, {IC_G, n_qk, 2}, so tap k is the contiguous block
        // [IC_G, OC_G, groups], applied to the depthwise output shifted by k steps.
        // QK is the depthwise output, [n_qk, T + 1, S].
        if (layer.ssm_conv1d_b) {
            QK = ggml_add(ctx0, QK, ggml_reshape_2d(ctx0, layer.ssm_conv1d_b, n_qk, 1));
        }
        cb(QK, "QK_dw", il);
        const int64_t ic_g = n_qk / n_groups;
        ggml_tensor * w_grp = layer.cca_conv_grp;
        ggml_tensor * grp = nullptr;
        for (int tap = 0; tap < 2; ++tap) {
            ggml_tensor * x = ggml_view_4d(ctx0, QK, ic_g, n_groups, n_seq_tokens, n_seqs,
                    ic_g*ggml_element_size(QK), QK->nb[1], QK->nb[2], tap*QK->nb[1]);
            x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));              // [IC_G, T, G, S]
            ggml_tensor * w = ggml_view_3d(ctx0, w_grp, ic_g, ic_g, n_groups,
                    w_grp->nb[1], ic_g*w_grp->nb[1], tap*w_grp->nb[2]);         // [IC_G, OC_G, G]
            ggml_tensor * y = ggml_mul_mat(ctx0, w, x);                          // [OC_G, T, G, S]
            grp = grp ? ggml_add(ctx0, grp, y) : y;
        }
        QK = ggml_cont(ctx0, ggml_permute(ctx0, grp, 0, 2, 1, 3));               // [OC_G, G, T, S]
        QK = ggml_reshape_2d(ctx0, QK, n_qk, n_tokens);
        QK = ggml_add(ctx0, QK, layer.cca_conv_grp_b);
        cb(QK, "QK_grp", il);

        ggml_tensor * Q_conv = ggml_view_2d(ctx0, QK, n_embd_q, n_tokens, QK->nb[1], 0);
        ggml_tensor * K_conv = ggml_view_2d(ctx0, QK, n_embd_k, n_tokens, QK->nb[1], n_embd_q*ggml_element_size(QK));

        ggml_tensor * Qcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Q_conv), n_embd_head, n_head, n_tokens);
        ggml_tensor * Kcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, K_conv), n_embd_head, n_head_kv, n_tokens);

        Qcur = ggml_add(ctx0, Qcur, qk_mean_q);
        Kcur = ggml_add(ctx0, Kcur, qk_mean_k);

        // l2_norm(x) * sqrt(head_dim) is rms_norm(x): one op, and one every backend has.
        // eps 1e-24 matches the reference clamping |x| at 1e-12.
        Qcur = ggml_rms_norm(ctx0, Qcur, 1e-24f);
        Kcur = ggml_rms_norm(ctx0, Kcur, 1e-24f);
        Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.cca_k_scale, 1, n_head_kv, 1));
        cb(Qcur, "Qcur_pre_rope", il);
        cb(Kcur, "Kcur_pre_rope", il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
        // sliding layers (ZAYA1-74B) have their own base; for other models this is freq_base
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        Vcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, Vcur), n_embd_head, n_head_kv, n_tokens);

        cur = build_attn(inp->get_attn(), layer.wo, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
            1.0f / sqrtf((float) n_embd_head), il);
        cb(cur, "attn_out", il);

        // ---- post-attention residual scale ----
        ggml_tensor * hs_scaled = apply_res_scale(cur, layer.res_scale_hs, layer.res_scale_hs_b, "res_scale_hs", il);
        ggml_tensor * res_scaled = apply_res_scale(residual, layer.res_scale_res, layer.res_scale_res_b, "res_scale_res", il);
        residual = ggml_add(ctx0, hs_scaled, res_scaled);
        cb(residual, "residual_post_attn", il);

        // ---- post-attention layernorm ----
        cur = build_norm(residual, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "post_attn_norm", il);

        // ===== MoE (every layer) =====

        // EDA: the previous layer's router state, scaled. Built before the down projection so
        // it is written before HRX's fused matmul + bias + add kernel reads it.
        ggml_tensor * eda = nullptr;
        if (prev_router != nullptr && layer.zaya_router_eda_scale != nullptr) {
            eda = ggml_mul(ctx0, prev_router, layer.zaya_router_eda_scale);
            ggml_build_forward_expand(gf, eda);
        }

        ggml_tensor * router_h = ggml_mul_mat(ctx0, layer.ffn_gate_inp, cur);
        if (layer.ffn_gate_inp_b) {
            router_h = ggml_add(ctx0, router_h, layer.ffn_gate_inp_b);
        }
        cb(router_h, "router_down", il);

        if (eda != nullptr) {
            router_h = ggml_add(ctx0, router_h, eda);
            cb(router_h, "router_eda", il);
        }

        prev_router = router_h;

        router_h = build_norm(router_h, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(router_h, "router_norm", il);

        router_h = ggml_mul_mat(ctx0, layer.ffn_gate, router_h);
        if (layer.ffn_gate_b) {
            router_h = ggml_add(ctx0, router_h, layer.ffn_gate_b);
        }
        router_h = ggml_gelu(ctx0, router_h);
        cb(router_h, "router_mlp0", il);

        router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp2, router_h);
        if (layer.zaya_router_mlp2_b) {
            router_h = ggml_add(ctx0, router_h, layer.zaya_router_mlp2_b);
        }
        router_h = ggml_gelu(ctx0, router_h);
        cb(router_h, "router_mlp2", il);

        router_h = ggml_mul_mat(ctx0, layer.zaya_router_mlp4, router_h);
        cb(router_h, "router_logits", il);

        router_h = ggml_soft_max(ctx0, router_h);
        cb(router_h, "router_probs", il);

        ggml_tensor * gate_probs = ggml_cont(ctx0,
                ggml_view_2d(ctx0, router_h, n_expert, n_tokens, router_h->nb[1], 0));
        cb(gate_probs, "gate_probs", il);

        ggml_tensor * expert_biases = nullptr;
        if (layer.zaya_router_biases != nullptr) {
            expert_biases = ggml_view_1d(ctx0, layer.zaya_router_biases, n_expert, 0);
        }

        cur = build_moe_ffn(cur,
            /* gate_inp */        nullptr,
            /* gate_inp_b */      nullptr,
            /* up_exps */         nullptr,
            /* up_exps_b */       nullptr,
            /* gate_exps */       nullptr,
            /* gate_exps_b */     nullptr,
            /* down_exps */       layer.ffn_down_exps,
            /* down_exps_b */     nullptr,
            /* exp_probs_b */     expert_biases,
            /* n_expert */        n_expert,
            /* n_expert_used */   hparams.n_expert_used,
            /* type_op */         LLM_FFN_SILU,
            /* norm_w */          false,
            /* w_scale */         1.0f,
            /* gating_op */       LLAMA_EXPERT_GATING_FUNC_TYPE_NONE,
            /* il */              il,
            /* probs_in */        gate_probs,
            /* gate_up_exps */    layer.ffn_gate_up_exps,
            /* gate_up_exps_b */  nullptr,
            /* up_exps_s */       nullptr,
            /* gate_exps_s */     nullptr,
            /* down_exps_s */     nullptr);
        cb(cur, "moe_out", il);

        // The router picks top-1 over n_expert + 1 slots; the last is a skip expert whose
        // output is zero (HF ZayaRouter masks it). build_moe_ffn chose the best of the real
        // experts; keep its output only where that expert's biased score beats the skip slot's.
        if (layer.zaya_router_biases != nullptr) {
            ggml_tensor * biased = ggml_add(ctx0, router_h, layer.zaya_router_biases);      // [n_expert + 1, T]
            ggml_tensor * biased_e = ggml_cont(ctx0, ggml_view_2d(ctx0, biased, n_expert, n_tokens, biased->nb[1], 0));
            ggml_tensor * best = ggml_argsort_top_k(ctx0, biased_e, 1);                    // [1, T]
            ggml_tensor * best_v = ggml_get_rows(ctx0,
                    ggml_reshape_3d(ctx0, biased_e, 1, n_expert, n_tokens), best);  // [1, 1, T]
            ggml_tensor * skip_v = ggml_view_2d(ctx0, biased, 1, n_tokens, biased->nb[1],
                    n_expert*ggml_element_size(biased));
            ggml_tensor * keep = ggml_step(ctx0, ggml_sub(ctx0,
                    ggml_reshape_2d(ctx0, best_v, 1, n_tokens), ggml_cont(ctx0, skip_v)));   // [1, T]
            cb(keep, "moe_keep", il);
            cur = ggml_mul(ctx0, cur, keep);
        }

        // ---- post-MLP residual scale ----
        hs_scaled = apply_res_scale(cur, layer.res_scale_hs_mlp, layer.res_scale_hs_mlp_b, "res_scale_hs_mlp", il);
        res_scaled = apply_res_scale(residual, layer.res_scale_res_mlp, layer.res_scale_res_mlp_b, "res_scale_res_mlp", il);
        inpL = ggml_add(ctx0, hs_scaled, res_scaled);
        cb(inpL, "layer_out", il);
    }

    cur = inpL;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    cur = ggml_cont(ctx0, ggml_cast(ctx0, cur, GGML_TYPE_F32));
    cb(cur, "result_output_fp32", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_zaya::graph<false>;
template struct llama_model_zaya::graph<true>;
