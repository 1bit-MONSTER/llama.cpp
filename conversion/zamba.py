# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Zamba v1 (Zyphra/Zamba-7B-v1): Mamba-1 layers split into n_mamba_heads heads, plus one shared
transformer block that every hybrid layer runs (src/models/zamba.cpp)."""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("ZambaForCausalLM")
class ZambaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ZAMBA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self.d_inner = hp["mamba_expand"] * hp["hidden_size"]
        self.n_mamba_heads = hp.get("n_mamba_heads", 1)
        self.block_types = hp["layers_block_type"]
        # the checkpoint stores the shared block once, under the first hybrid layer
        self.shared_bid = self.block_types.index("hybrid")

    def set_vocab(self):
        # LlamaTokenizer from tokenizer.json. tokenizer_config.json names a pad token, [PAD], that
        # is not in the vocab, so the HF vocab gains a 32001st entry the embeddings have no row
        # for: keep the model's vocab_size tokens.
        vocab = gguf.LlamaHfVocab(self.dir_model)
        n_vocab = self.hparams["vocab_size"]
        tokens, scores, toktypes = [], [], []
        for text, score, toktype in vocab.all_tokens():
            if len(tokens) == n_vocab:
                break
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)
        assert len(tokens) == n_vocab, (len(tokens), n_vocab)
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        gguf.SpecialVocab(self.dir_model, n_vocab=n_vocab).add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        n_head = hp["num_attention_heads"]
        head_dim = hp.get("attention_head_dim") or 2 * hp["hidden_size"] // n_head
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hp.get("max_position_embeddings", 4096))
        self.gguf_writer.add_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_head_count(n_head)
        # KV only on the hybrid layers
        self.gguf_writer.add_head_count_kv([hp["num_key_value_heads"] if t == "hybrid" else 0 for t in self.block_types])
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_rope_dimension_count(0)  # no rotary embedding in Zamba v1's attention
        self.gguf_writer.add_layer_norm_rms_eps(hp.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        self.gguf_writer.add_ssm_conv_kernel(hp["mamba_d_conv"])
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        self.gguf_writer.add_ssm_state_size(hp["mamba_d_state"])
        self.gguf_writer.add_ssm_time_step_rank(hp["mamba_dt_rank"])
        self.gguf_writer.add_ssm_group_count(self.n_mamba_heads)  # read back as the number of Mamba-1 heads
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        if name == "model.embed_tokens.weight":
            yield self.format_tensor_name(T.TOKEN_EMBD), data_torch
            return
        if name == "model.final_layernorm.weight":
            yield self.format_tensor_name(T.OUTPUT_NORM), data_torch
            return
        assert bid is not None, name

        if ".shared_transf." in name:
            tail = name.partition(".shared_transf.")[2]
            shared = {
                "input_layernorm.weight":          T.ATTN_POST_NORM,  # norm of concat(h, embeddings)
                "self_attn.q_proj.weight":         T.ATTN_Q,
                "self_attn.k_proj.weight":         T.ATTN_K,
                "self_attn.v_proj.weight":         T.ATTN_V,
                "self_attn.o_proj.weight":         T.ATTN_OUT,
                "pre_ff_layernorm.weight":         T.FFN_NORM,
                "feed_forward.gate_proj.weight":   T.FFN_GATE,
                "feed_forward.up_proj.weight":     T.FFN_UP,
                "feed_forward.down_proj.weight":   T.FFN_DOWN,
            }
            yield self.format_tensor_name(shared[tail], self.shared_bid), data_torch
            return
        if name.endswith(".linear.weight"):
            yield self.format_tensor_name(T.SSM_MIX, bid), data_torch
            return

        # Mamba layer tensors live under .mamba. (mamba layers) or .mamba_decoder.mamba. (hybrid layers)
        name = name.replace(".mamba_decoder.", ".")
        if name.endswith(".input_layernorm.weight"):
            yield self.format_tensor_name(T.ATTN_NORM, bid), data_torch
            return
        tail = name.partition(".mamba.")[2]
        n_heads = self.n_mamba_heads
        if tail == "in_proj.weight":
            # rows interleave [x, gate] per channel (view(d_inner, 2)); store x rows, then gate rows
            w = data_torch.reshape(self.d_inner, 2, data_torch.shape[-1])
            yield self.format_tensor_name(T.SSM_IN, bid), torch.cat([w[:, 0], w[:, 1]], dim=0)
        elif tail == "conv1d.weight":
            yield self.format_tensor_name(T.SSM_CONV1D, bid), data_torch.squeeze(1)
        elif tail == "conv1d.bias":
            yield self.format_tensor_name(T.SSM_CONV1D, bid, suffix=".bias"), data_torch
        elif tail == "x_proj_weight":   # {n_heads, dt_rank + 2*d_state, head_dim}
            yield self.format_tensor_name(T.SSM_X, bid), data_torch
        elif tail == "dt_proj_weight":  # {n_heads, head_dim, dt_rank}
            yield self.format_tensor_name(T.SSM_DT, bid), data_torch
        elif tail == "dt_proj_bias":    # {n_heads, head_dim} -> {d_inner}
            yield self.format_tensor_name(T.SSM_DT, bid, suffix=".bias"), data_torch.reshape(self.d_inner)
        elif tail == "A_log":           # {n_heads, head_dim, d_state} -> A = -exp(A_log), {d_inner, d_state}
            yield self.format_tensor_name(T.SSM_A, bid, suffix=""), -torch.exp(data_torch.float()).reshape(self.d_inner, -1)
        elif tail == "D":               # {n_heads, head_dim} -> {d_inner}
            yield self.format_tensor_name(T.SSM_D, bid, suffix=""), data_torch.reshape(self.d_inner)
        elif tail == "out_proj.weight":
            yield self.format_tensor_name(T.SSM_OUT, bid), data_torch
        else:
            raise ValueError(f"zamba: unexpected tensor {name}")
        assert n_heads > 0
