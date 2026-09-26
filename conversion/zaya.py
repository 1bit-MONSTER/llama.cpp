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
# limitations under the License.

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("ZayaForCausalLM")
class ZayaModel(TextModel):
    """Zyphra ZAYA1 (transformers naming): every layer is CCA attention + a top-1 MoE."""
    model_arch = gguf.MODEL_ARCH.ZAYA

    # checkpoint suffix -> (tensor, name suffix, squeeze dim 1)
    _LAYER_MAP: dict[str, tuple[gguf.MODEL_TENSOR, str, bool]] = {
        "input_layernorm.weight":                                (gguf.MODEL_TENSOR.ATTN_NORM,             ".weight", False),
        "post_attention_layernorm.weight":                       (gguf.MODEL_TENSOR.ATTN_POST_NORM,        ".weight", False),
        "self_attn.qkv_proj.q_proj.weight":                      (gguf.MODEL_TENSOR.ATTN_Q,                ".weight", False),
        "self_attn.qkv_proj.k_proj.weight":                      (gguf.MODEL_TENSOR.ATTN_K,                ".weight", False),
        "self_attn.qkv_proj.v_proj_current.weight":              (gguf.MODEL_TENSOR.CCA_VAL_PROJ1,         ".weight", False),
        "self_attn.qkv_proj.v_proj_delayed.weight":              (gguf.MODEL_TENSOR.CCA_VAL_PROJ2,         ".weight", False),
        "self_attn.qkv_proj.conv_qk_depthwise.weight":           (gguf.MODEL_TENSOR.SSM_CONV1D,            ".weight", True),
        "self_attn.qkv_proj.conv_qk_depthwise.bias":             (gguf.MODEL_TENSOR.SSM_CONV1D,            ".bias",   False),
        "self_attn.qkv_proj.conv_qk_grouped.weight":             (gguf.MODEL_TENSOR.CCA_CONV_GRP,          ".weight", False),
        "self_attn.qkv_proj.conv_qk_grouped.bias":               (gguf.MODEL_TENSOR.CCA_CONV_GRP,          ".bias",   False),
        "self_attn.qk_norm.temp":                                (gguf.MODEL_TENSOR.CCA_K_SCALE,           ".weight", False),
        "self_attn.o_proj.weight":                               (gguf.MODEL_TENSOR.ATTN_OUT,              ".weight", False),
        "mlp.gate.down_proj.weight":                             (gguf.MODEL_TENSOR.FFN_GATE_INP,          ".weight", False),
        "mlp.gate.down_proj.bias":                               (gguf.MODEL_TENSOR.FFN_GATE_INP,          ".bias",   False),
        "mlp.gate.router_states_scale":                          (gguf.MODEL_TENSOR.ZAYA_ROUTER_EDA_SCALE, ".weight", False),
        "mlp.gate.router_mlp.norm.weight":                       (gguf.MODEL_TENSOR.FFN_NORM,              ".weight", False),
        "mlp.gate.router_mlp.fc1.weight":                        (gguf.MODEL_TENSOR.FFN_GATE,              ".weight", False),
        "mlp.gate.router_mlp.fc1.bias":                          (gguf.MODEL_TENSOR.FFN_GATE,              ".bias",   False),
        "mlp.gate.router_mlp.fc2.weight":                        (gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2,      ".weight", False),
        "mlp.gate.router_mlp.fc2.bias":                          (gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP2,      ".bias",   False),
        "mlp.gate.router_mlp.out_proj.weight":                   (gguf.MODEL_TENSOR.ZAYA_ROUTER_MLP4,      ".weight", False),
        "mlp.gate.balancing_biases":                             (gguf.MODEL_TENSOR.ZAYA_ROUTER_BIASES,    ".weight", False),
        "mlp.experts.gate_up_proj":                              (gguf.MODEL_TENSOR.FFN_GATE_UP_EXP,       ".weight", False),
        "mlp.experts.down_proj":                                 (gguf.MODEL_TENSOR.FFN_DOWN_EXP,          ".weight", False),
        "post_attention_residual_scale.hidden_states_scale":     (gguf.MODEL_TENSOR.RES_SCALE_HS,          ".weight", False),
        "post_attention_residual_scale.hidden_states_bias":      (gguf.MODEL_TENSOR.RES_SCALE_HS,          ".bias",   False),
        "post_attention_residual_scale.residual_scale":          (gguf.MODEL_TENSOR.RES_SCALE_RES,         ".weight", False),
        "post_attention_residual_scale.residual_bias":           (gguf.MODEL_TENSOR.RES_SCALE_RES,         ".bias",   False),
        "post_mlp_residual_scale.hidden_states_scale":           (gguf.MODEL_TENSOR.RES_SCALE_HS_MLP,      ".weight", False),
        "post_mlp_residual_scale.hidden_states_bias":            (gguf.MODEL_TENSOR.RES_SCALE_HS_MLP,      ".bias",   False),
        "post_mlp_residual_scale.residual_scale":                (gguf.MODEL_TENSOR.RES_SCALE_RES_MLP,     ".weight", False),
        "post_mlp_residual_scale.residual_bias":                 (gguf.MODEL_TENSOR.RES_SCALE_RES_MLP,     ".bias",   False),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tensors are prepared before set_vocab(), and the embedding is trimmed to this
        self._n_vocab = gguf.LlamaHfVocab(self.dir_model).vocab_size

    def set_vocab(self):
        # the Gemma 3/4 BPE tokenizer; added tokens that are not special (such as "\n") stay
        # USER_DEFINED so they are matched in prompts and rendered in output
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens, scores, toktypes = [], [], []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        self.gguf_writer.add_tokenizer_model("gemma4")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)

    def set_gguf_parameters(self):
        hp = self.hparams
        head_dim = hp["head_dim"]
        n_qk = (hp["num_attention_heads"] + hp["num_key_value_heads"]) * head_dim
        rope = hp.get("rope_parameters", {}).get("hybrid", hp.get("rope_parameters", {}))
        rotary = rope.get("partial_rotary_factor", hp.get("partial_rotary_factor", 0.5))

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hp["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_head_count(hp["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hp["num_key_value_heads"])
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_layer_norm_rms_eps(hp.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_rope_dimension_count(int(rotary * head_dim))
        self.gguf_writer.add_rope_freq_base(float(rope.get("rope_theta", hp.get("rope_theta", 1e6))))
        self.gguf_writer.add_expert_count(hp["num_experts"])
        self.gguf_writer.add_expert_used_count(hp.get("num_experts_per_tok", 1))
        self.gguf_writer.add_expert_feed_forward_length(hp["router_hidden_size"])
        self.gguf_writer.add_ssm_conv_kernel(hp.get("cca_time0", 2))
        self.gguf_writer.add_ssm_state_size(2 * n_qk + hp["hidden_size"])
        self.gguf_writer.add_ssm_inner_size(1)
        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"zaya: {self.block_count} layers, rope theta {rope.get('rope_theta')}, rotary {rotary}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight":
            # the embedding has padding rows past the tokenizer's vocab
            yield self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch[:self._n_vocab]
            return
        if name == "model.norm.weight":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch
            return
        if name == "model.input_hidden_states_scale":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.INPUT_HIDDEN_STATES_SCALE), data_torch
            return
        if name == "model.input_hidden_states_bias":
            yield self.format_tensor_name(gguf.MODEL_TENSOR.INPUT_HIDDEN_STATES_SCALE, suffix=".bias"), data_torch
            return
        if name == "lm_head.weight":
            return  # tied to the embedding
        if bid is not None:
            suffix = name.split(f"model.layers.{bid}.", 1)[-1]
            if suffix in self._LAYER_MAP:
                tensor, tsuffix, squeeze = self._LAYER_MAP[suffix]
                if squeeze:
                    data_torch = data_torch.squeeze(1)
                if tensor == gguf.MODEL_TENSOR.CCA_CONV_GRP and tsuffix == ".weight":
                    # (OC, IC_G, taps) -> tap-major (taps, OC, IC_G): each tap's weights are one
                    # contiguous [IC_G, OC] block, which the graph applies as one batched matmul
                    data_torch = data_torch.permute(2, 0, 1).contiguous()
                yield self.format_tensor_name(tensor, bid, suffix=tsuffix), data_torch
                return
        raise ValueError(f"zaya: unmapped tensor {name}")
