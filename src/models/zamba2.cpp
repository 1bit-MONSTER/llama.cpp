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

void llama_model_zamba2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    // absent in files written before the MLP adapter was stored separately (engine #1036 cause 2)
    ml.get_key(LLM_KV_ZAMBA2_FFN_GATE_UP_ADAPTER_RANK, hparams.zamba2_ffn_gate_up_adapter_rank, false);

    // All layers have Mamba-2 (recurrent state needed for all)
    std::fill(hparams.is_recr_impl.begin(), hparams.is_recr_impl.end(), true);

    switch (hparams.n_layer()) {
        case 38: type = LLM_TYPE_1B; break;
        case 54: type = LLM_TYPE_3B; break;
        case 76: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // Zamba2 uses (head_dim/2)^(-0.5) attention scaling
    // Ref: transformers/models/zamba2/modeling_zamba2.py Zamba2Attention.__init__
    hparams.f_attention_scale = 1.0f / sqrtf(float(hparams.n_embd_head_v_full) / 2.0f);
}

void llama_model_zamba2::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t d_state    = hparams.ssm_d_state;
    const int64_t n_head_ssm = hparams.ssm_dt_rank;
    const int64_t n_group    = hparams.ssm_n_group;
    const int64_t d_conv_dim = d_inner + 2*n_group*d_state;
    const int64_t d_proj     = d_inner + d_conv_dim + n_head_ssm;

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // Two forms of the file, told apart by one key. Before this key existed the per-layer MLP adapter was baked
    // into ffn_up at conversion time, so every hybrid layer carries its own full [n_embd, 2*n_ff] copy of a
    // matrix that is otherwise shared per block (engine #1036 cause 2: 1.13 B params on the 7B). With the key
    // set, ffn_up is the shared base -- stored once per block, like the rest of the shared transformer block --
    // and each hybrid layer carries the low-rank pair that makes its FFN different, applied in the graph as
    // base + B*(A*x). Old files keep loading: rank 0 means ffn_up is per layer and no adapter is read.
    uint32_t gate_up_adapter_rank = hparams.zamba2_ffn_gate_up_adapter_rank;
    const bool ffn_up_adapter = gate_up_adapter_rank > 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // The hybrid layers in visit order, and the ones that carry the shared block weights. A
    // block is stored once, at the first hybrid layers that use it; the rest are aliased below.
    std::vector<int> shared_use;
    std::vector<int> shared_src;

    for (int i = 0; i < n_layer; ++i) {
        const int64_t n_head_kv_i = hparams.n_head_kv(i);
        auto & layer = layers[i];

        // Pre-mamba norm (all layers)
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // SSM tensors (all layers have Mamba-2)
        layer.ssm_in      = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), {n_embd, d_proj}, 0);
        layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_conv_dim}, 0);
        layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), {d_conv_dim}, 0);
        layer.ssm_dt_b     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), {n_head_ssm}, 0);
        layer.ssm_a        = create_tensor(tn(LLM_TENSOR_SSM_A,               i), {1, n_head_ssm}, 0);
        layer.ssm_d        = create_tensor(tn(LLM_TENSOR_SSM_D,               i), {1, n_head_ssm}, 0);
        layer.ssm_norm     = create_tensor(tn(LLM_TENSOR_SSM_NORM,  "weight", i), {d_inner/n_group, n_group}, TENSOR_NOT_REQUIRED);
        layer.ssm_out      = create_tensor(tn(LLM_TENSOR_SSM_OUT,   "weight", i), {d_inner, n_embd}, 0);

        if (n_head_kv_i > 0) {
            // Hybrid layer: attention + FFN + linear mixing
            const int64_t attn_hidden = 2 * n_embd;

            // Everything belonging to the shared block is optional per layer: a layer that carries
            // no copy is aliased to its block's source after this loop, and a file with no copy at
            // all is refused there rather than once per tensor. ffn_up is part of the shared block
            // only in the adapter form, where the per-layer difference is the low-rank pair below.
            const int shared = TENSOR_NOT_REQUIRED;

            // Pre-attention norm (4096-dim, operates on concat)
            layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {attn_hidden}, shared);

            // Attention weights (input is 4096-dim concat)
            create_tensor_qkv(layer, i, attn_hidden, attn_hidden, attn_hidden, attn_hidden, shared);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {attn_hidden, n_embd}, shared);

            // FFN (fused gate+up, split internally by GEGLU)
            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, shared);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, 2*n_ff},
                    ffn_up_adapter ? shared : 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, shared);

            if (ffn_up_adapter) {
                // GGUF stores matmul weights as [in, out]: A is hidden -> rank, B is rank -> 2*n_ff
                layer.ffn_gate_up_adapter_a = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_ADAPTER_A, "weight", i),
                        {n_embd, gate_up_adapter_rank}, 0);
                layer.ffn_gate_up_adapter_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_ADAPTER_B, "weight", i),
                        {gate_up_adapter_rank, 2*n_ff}, 0);
            }

            // Linear mixing (transformer output -> mamba input adjustment)
            layer.ssm_mix = create_tensor(tn(LLM_TENSOR_SSM_MIX, "weight", i), {n_embd, n_embd}, 0);

            // wo, not wq: with attention adapters the converter writes q/k/v (adapter merged) in every hybrid layer
            if (layer.wo != nullptr) {
                shared_src.push_back(i);
            }
            shared_use.push_back(i);
        }
    }

    // Hybrid visit N runs block N % num_mem_blocks, which the converter stores at the first
    // num_mem_blocks hybrid layers; so those are the sources and every other hybrid layer points
    // at sources[N % num_mem_blocks]. A file written before the block was shared has a copy in
    // every layer, making every layer a source, so each alias below is the identity.
    GGML_ASSERT(!shared_use.empty() && "zamba2: this GGUF declares no hybrid layer");
    GGML_ASSERT(!shared_src.empty() && "zamba2: this GGUF stores no shared attention/FFN block");
    for (size_t k = 0; k < shared_src.size(); ++k) {
        GGML_ASSERT(shared_src[k] == shared_use[k] &&
                "zamba2: the shared block must be stored at the first hybrid layers of the file");
    }
    for (size_t v = 0; v < shared_use.size(); ++v) {
        auto & layer = layers[shared_use[v]];
        if (layer.wo != nullptr) {
            continue;
        }
        const auto & src = layers[shared_src[v % shared_src.size()]];
        layer.attn_post_norm = src.attn_post_norm;
        if (layer.wq == nullptr) {
            layer.wq = src.wq;
            layer.wk = src.wk;
            layer.wv = src.wv;
        }
        layer.wo              = src.wo;
        layer.ffn_norm        = src.ffn_norm;
        layer.ffn_down        = src.ffn_down;
        // only when the file does not carry one: the adapter form stores ffn_up once per block, while a
        // loader that fabricates tensors (no weights on disk, e.g. the arch test) gives every layer its own
        if (layer.ffn_up == nullptr) {
            layer.ffn_up = src.ffn_up;
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_zamba2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_zamba2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Embed tokens
    inpL = build_inp_embd(model.tok_embd);

    // Store original embeddings for concat at hybrid layers
    // Cast to F32 to avoid type mismatch with evolved hidden states
    ggml_tensor * inpOrig = ggml_cast(ctx0, inpL, GGML_TYPE_F32);
    cb(inpOrig, "inpOrig", -1);

    // Position embeddings for RoPE (skip if n_rot=0, i.e. use_mem_rope=false)
    ggml_tensor * inp_pos = (n_rot > 0) ? build_inp_pos() : nullptr;

    // Build hybrid memory (KV cache + recurrent state)
    auto * inp = build_inp_mem_hybrid();

    const float kq_scale = hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_head_kv_il = hparams.n_head_kv(il);
        const bool is_hybrid = (n_head_kv_il > 0);

        ggml_tensor * residual = inpL;

        if (is_hybrid) {
            // ============ HYBRID LAYER ============
            // Step 1: Shared transformer (attention + FFN)
            // Concat current hidden states with original embeddings
            cur = ggml_concat(ctx0, inpL, inpOrig, 0);
            cb(cur, "attn_concat", il);

            // Pre-attention norm (operates on 4096-dim concat)
            cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm_concat", il);

            // Self-attention on concat input
            const int64_t n_head_il = hparams.n_head(il);
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head_il, n_head_kv_il, il);

            // RoPE (conditional: Zamba2 models with use_mem_rope=false have n_rot=0)
            if (n_rot > 0) {
                Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot, hparams.rope_type, n_ctx_orig,
                    freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot, hparams.rope_type, n_ctx_orig,
                    freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Build attention (O projects from n_heads*head_dim to n_embd)
            cur = build_attn(inp->get_attn(),
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // Step 2: FFN (operates on n_embd-dim output from attention)
            cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // The per-layer MLP adapter, applied to the same normed input as the fused gate+up matmul and added to
            // its output (checkpoint: gate_up_proj(x) + adapter(x), no LoRA alpha/rank scaling). Absent in files
            // written before the adapter form existed, where the adapter is baked into ffn_up instead.
            ggml_tensor * gate_up_delta = NULL;
            if (model.layers[il].ffn_gate_up_adapter_a != NULL) {
                ggml_tensor * adapter = build_lora_mm(model.layers[il].ffn_gate_up_adapter_a, cur);
                cb(adapter, "ffn_gate_up_adapter_a", il);
                gate_up_delta = build_lora_mm(model.layers[il].ffn_gate_up_adapter_b, adapter);
                cb(gate_up_delta, "ffn_gate_up_adapter_b", il);
            }

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    NULL,                      NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_GEGLU, LLM_FFN_SEQ, il, gate_up_delta);
            cb(cur, "ffn_out", il);

            // Step 3: Linear mixing (transformer output projection)
            cur = build_lora_mm(model.layers[il].ssm_mix, cur);
            cb(cur, "ssm_mix", il);

            // Step 4: Mamba-2 with transformer residual
            ggml_tensor * mamba_input = ggml_add(ctx0, inpL, cur);
            cb(mamba_input, "mamba_input", il);

            // Pre-mamba norm
            cur = build_norm(mamba_input, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "mamba_norm", il);

            // Mamba-2 layer
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            cb(cur, "mamba_out", il);

            // Residual: original hidden + mamba output
            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, residual, cur);

        } else {
            // ============ PURE MAMBA LAYER ============
            // Pre-mamba norm
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            // Mamba-2 layer
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
            cb(cur, "mamba_out", il);

            // Residual
            if (il == n_layer - 1 && inp_out_ids) {
                cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, residual, cur);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    // Final norm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
