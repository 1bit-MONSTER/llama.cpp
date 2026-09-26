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

// BlackMamba (Zyphra/BlackMamba-1.5B, -2.8B). A pre-norm residual stack whose layers alternate a
// Mamba-1 mixer and a Switch MoE: router (with bias) -> sigmoid -> top-1 expert, the expert output
// scaled by that probability; experts are gated GELU (erf) MLPs. RMSNorm per layer, a LayerNorm
// (with bias) at the end, output tied to the embeddings.
// Ref: github.com/Zyphra/BlackMamba (mamba_block.py, switch_mlp.py, mlp.py).

void llama_model_blackmamba::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);

    switch (hparams.n_embd) {
        case 1152: type = LLM_TYPE_1_5B; break;
        case 1472: type = LLM_TYPE_3B;   break;
        default:   type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_blackmamba::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t dt_rank  = hparams.ssm_dt_rank;
    const int64_t n_ff_exp = hparams.n_ff_exp();

    tok_embd      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.n_ff(i) == 0) {  // Mamba-1 mixer
            layer.ssm_in       = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), {n_embd, 2*d_inner}, 0);
            layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), {d_inner}, 0);
            layer.ssm_x        = create_tensor(tn(LLM_TENSOR_SSM_X,      "weight", i), {d_inner, dt_rank + 2*d_state}, 0);
            layer.ssm_dt       = create_tensor(tn(LLM_TENSOR_SSM_DT,     "weight", i), {dt_rank, d_inner}, 0);
            layer.ssm_dt_b     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), {d_inner}, 0);
            layer.ssm_a        = create_tensor(tn(LLM_TENSOR_SSM_A,                i), {d_state, d_inner}, 0);
            layer.ssm_d        = create_tensor(tn(LLM_TENSOR_SSM_D,                i), {d_inner}, 0);
            layer.ssm_out      = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", i), {d_inner, n_embd}, 0);
        } else {                     // Switch MoE
            layer.ffn_gate_inp   = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_inp_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "bias",   i), {n_expert}, 0);
            layer.ffn_gate_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_up_exps    = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_blackmamba::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_blackmamba::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    auto * rs_inp = build_rs_inp();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.n_ff(il) == 0) {
            cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
        } else {
            const int64_t n_expert = hparams.n_expert;
            // router: sigmoid(W x + b), top-1; the expert's output is scaled by its probability
            ggml_tensor * probs = ggml_sigmoid(ctx0, ggml_add(ctx0, build_lora_mm(layer.ffn_gate_inp, cur), layer.ffn_gate_inp_b));
            cb(probs, "ffn_moe_probs", il);
            ggml_tensor * selected = ggml_argsort_top_k(ctx0, probs, 1);                      // [1, n_tokens]
            ggml_tensor * weights  = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected); // [1, 1, n_tokens]

            ggml_tensor * x3   = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
            ggml_tensor * gate = ggml_mul_mat_id(ctx0, layer.ffn_gate_exps, x3, selected);  // [n_ff_exp, 1, n_tokens]
            ggml_tensor * up   = ggml_mul_mat_id(ctx0, layer.ffn_up_exps,   x3, selected);
            ggml_tensor * act  = ggml_geglu_erf_split(ctx0, gate, up);
            ggml_tensor * down = ggml_mul_mat_id(ctx0, layer.ffn_down_exps, act, selected); // [n_embd, 1, n_tokens]
            cur = ggml_reshape_2d(ctx0, ggml_mul(ctx0, down, weights), n_embd, n_tokens);
            cb(cur, "ffn_moe_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = ggml_add(ctx0, cur, inpL);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
