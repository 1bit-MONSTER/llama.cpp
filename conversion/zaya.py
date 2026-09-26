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

import re
import shutil
import tempfile

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import torch

from .base import ModelBase, TextModel, gguf, logger
from .qwenvl import Qwen2VLVisionModel


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
        # ZAYA1-VL's vision-only LoRA (A: down to the rank, B: back up; experts stacked)
        "vlora.q_a":             (gguf.MODEL_TENSOR.ZAYA_VLORA_Q_A,         ".weight", False),
        "vlora.q_b":             (gguf.MODEL_TENSOR.ZAYA_VLORA_Q_B,         ".weight", False),
        "vlora.k_a":             (gguf.MODEL_TENSOR.ZAYA_VLORA_K_A,         ".weight", False),
        "vlora.k_b":             (gguf.MODEL_TENSOR.ZAYA_VLORA_K_B,         ".weight", False),
        "vlora.v1_a":            (gguf.MODEL_TENSOR.ZAYA_VLORA_V1_A,        ".weight", False),
        "vlora.v1_b":            (gguf.MODEL_TENSOR.ZAYA_VLORA_V1_B,        ".weight", False),
        "vlora.v2_a":            (gguf.MODEL_TENSOR.ZAYA_VLORA_V2_A,        ".weight", False),
        "vlora.v2_b":            (gguf.MODEL_TENSOR.ZAYA_VLORA_V2_B,        ".weight", False),
        "vlora.o_a":             (gguf.MODEL_TENSOR.ZAYA_VLORA_O_A,         ".weight", False),
        "vlora.o_b":             (gguf.MODEL_TENSOR.ZAYA_VLORA_O_B,         ".weight", False),
        "vlora.gate_up_exps_a":  (gguf.MODEL_TENSOR.ZAYA_VLORA_UP_EXPS_A,   ".weight", False),
        "vlora.gate_up_exps_b":  (gguf.MODEL_TENSOR.ZAYA_VLORA_UP_EXPS_B,   ".weight", False),
        "vlora.down_exps_a":     (gguf.MODEL_TENSOR.ZAYA_VLORA_DOWN_EXPS_A, ".weight", False),
        "vlora.down_exps_b":     (gguf.MODEL_TENSOR.ZAYA_VLORA_DOWN_EXPS_B, ".weight", False),
    }

    def __init__(self, *args, **kwargs):
        # ZAYA1-base, ZAYA1-reasoning-base and the *-legacy repos keep Zyphra's Megatron-style
        # checkpoint: attention and MoE are separate "layers", with per-layer config lists.
        # Normalize the config here, and the tensors in index_tensors, to the transformers layout.
        hparams = kwargs.get("hparams") or ModelBase.load_hparams(args[0], False)
        self._legacy = "zaya_layers" in hparams or "cca" in hparams
        if self._legacy:
            hparams = self._legacy_hparams(hparams)
        kwargs["hparams"] = hparams
        super().__init__(*args, **kwargs)
        # tensors are prepared before set_vocab(), and the embedding is trimmed to this
        self._n_vocab = gguf.LlamaHfVocab(self._tokenizer_dir()).vocab_size

    def _tokenizer_dir(self) -> Path:
        # transformers' AutoTokenizer reads config.json, and its ZayaConfig rejects the legacy
        # config (rope_scaling: false); load the tokenizer from its files alone
        if not self._legacy:
            return self.dir_model
        if not hasattr(self, "_tok_tmp"):
            self._tok_tmp = tempfile.TemporaryDirectory(prefix="zaya-tok-")
            for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
                if (self.dir_model / f).is_file():
                    shutil.copy(self.dir_model / f, self._tok_tmp.name)
        return Path(self._tok_tmp.name)

    @staticmethod
    def _legacy_hparams(hp: dict) -> dict:
        def first(v):  # per-layer lists hold 0 on the layers the value does not apply to
            return max(v) if isinstance(v, list) else v
        for flag in ("zaya_use_eda", "zaya_use_mod", "scale_residual_merge", "cca", "gated_linear_unit"):
            if hp.get(flag, True) is not True:
                raise ValueError(f"zaya: legacy checkpoint with {flag}={hp[flag]} is not supported")
        zl = hp.get("zaya_layers")
        n_half = len(zl) if zl else hp["num_hidden_layers"]
        if "vision_config" in hp:
            n_half = 2 * hp["num_hidden_layers"]  # ZAYA1-VL: layers.{i}.attn / layers.{i}.mlp
        n_block = n_half // 2
        n_head = first(hp["cca_num_q_heads"]) if "cca_num_q_heads" in hp else hp["num_attention_heads"]
        n_head_kv = first(hp["num_query_groups_list"]) if "num_query_groups_list" in hp else hp["num_query_groups"]
        n_expert = max(x for x in zl if isinstance(x, int)) if zl else hp["num_experts"]
        ffn = first(hp["ffn_hidden_size_list"]) if "ffn_hidden_size_list" in hp else hp["ffn_hidden_size"]
        theta = hp.get("rope_theta", hp.get("rotary_base", 1e6))
        rotary = hp.get("partial_rotary_factor", hp.get("rope_pct", 0.5))
        out = dict(hp)
        out.update({
            "num_hidden_layers": n_block,
            "num_attention_heads": n_head,
            "num_key_value_heads": n_head_kv,
            "head_dim": hp.get("head_dim") or hp["kv_channels"],
            "num_experts": n_expert,
            "num_experts_per_tok": hp.get("moe_router_topk", 1),
            "moe_intermediate_size": ffn // 2,  # fc1 holds gate and up
            "router_hidden_size": first(hp["zaya_mlp_expansion"]),
            "rms_norm_eps": hp.get("norm_epsilon", 1e-5),
            "layer_types": ["hybrid"] * n_block,
            "rope_parameters": {"hybrid": {"rope_theta": theta, "partial_rotary_factor": rotary}},
        })
        swa = hp.get("swa_layers")
        if swa and any(swa):
            # per half-layer window on the attention layers; Megatron's window excludes the query
            # position, transformers' sliding_window includes it (ZAYA1-74B-preview: 4096 -> 4097)
            out["layer_types"] = ["hybrid_sliding" if swa[2 * i] else "hybrid" for i in range(n_block)]
            out["sliding_window"] = max(swa) + 1
            out["rope_parameters"]["hybrid_sliding"] = {"rope_theta": hp.get("swa_rotary_base", theta), "partial_rotary_factor": rotary}
        return out

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
        if not self._legacy:
            return tensors
        n_block, n_expert = self.hparams["num_hidden_layers"], self.hparams["num_experts"]
        vl: dict[str, Callable[[], Tensor]] = {}
        if "vision_config" in self.hparams:
            # ZAYA1-VL: blocks hold attn. and mlp. sublayers, which are the half-layers 2i and 2i+1;
            # the vision tower goes to the mmproj
            renamed: dict[str, Callable[[], Tensor]] = {}
            for name, gen in tensors.items():
                if name.startswith("vision_tower."):
                    continue
                m = re.match(r"model\.layers\.(\d+)\.(attn|mlp)\.(.*)", name)
                if m is None:
                    renamed[name] = gen
                    continue
                i, part, rest = int(m.group(1)), m.group(2), m.group(3)
                lora = re.match(r"self_attn\.(?:qkv\.)?(lora_linear_q|lora_linear_k|lora_val_proj1|lora_val_proj2|lora_linear_o)\.([01])\.weight", rest)
                if lora:
                    short = {"lora_linear_q": "q", "lora_linear_k": "k", "lora_val_proj1": "v1",
                             "lora_val_proj2": "v2", "lora_linear_o": "o"}[lora.group(1)]
                    vl[f"model.layers.{i}.vlora.{short}_{'ab'[int(lora.group(2))]}"] = gen
                    continue
                if ".lora_fc" in rest:
                    continue  # stacked below
                renamed[f"model.layers.{2 * i + (part == 'mlp')}.{rest}"] = gen
            for i in range(n_block):
                pre = f"model.layers.{i}.mlp.zaya_block.experts.local_experts."
                for src, dst in (("lora_fc1", "gate_up_exps"), ("lora_fc2", "down_exps")):
                    for seq, ab in (("0", "a"), ("1", "b")):
                        gens = [tensors[f"{pre}{e}.{src}.{seq}.weight"] for e in range(n_expert)]
                        vl[f"model.layers.{i}.vlora.{dst}_{ab}"] = lambda gens=gens: torch.stack([g() for g in gens])
            tensors = renamed
        out: dict[str, Callable[[], Tensor]] = {}
        rename = {"model.embed_tokens.weight": "model.embed_tokens.weight", "model.final_norm.weight": "model.norm.weight"}
        for k in ("hidden_states_scale", "hidden_states_bias"):
            rename[f"model.layers.0.res_scale.{k}"] = f"model.input_{k}"
        for k in ("hidden_states_scale", "hidden_states_bias", "residual_scale", "residual_bias"):
            rename[f"model.res_scale.{k}"] = f"model.layers.{n_block - 1}.post_mlp_residual_scale.{k}"
        attn = {
            "input_norm.weight": "input_layernorm.weight",
            "self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "self_attn.qkv.temp": "self_attn.qk_norm.temp",
            "self_attn.qkv.linear_q.weight": "self_attn.qkv_proj.q_proj.weight",
            "self_attn.qkv.linear_k.weight": "self_attn.qkv_proj.k_proj.weight",
            "self_attn.qkv.val_proj1.weight": "self_attn.qkv_proj.v_proj_current.weight",
            "self_attn.qkv.val_proj2.weight": "self_attn.qkv_proj.v_proj_delayed.weight",
            "self_attn.qkv.conv_qk.0.weight": "self_attn.qkv_proj.conv_qk_depthwise.weight",
            "self_attn.qkv.conv_qk.0.bias": "self_attn.qkv_proj.conv_qk_depthwise.bias",
            "self_attn.qkv.conv_qk.1.weight": "self_attn.qkv_proj.conv_qk_grouped.weight",
            "self_attn.qkv.conv_qk.1.bias": "self_attn.qkv_proj.conv_qk_grouped.bias",
        }
        moe = {
            "input_norm.weight": "post_attention_layernorm.weight",
            "zaya_block.router.balancing_biases": "mlp.gate.balancing_biases",
            "zaya_block.router.down_proj.weight": "mlp.gate.down_proj.weight",
            "zaya_block.router.down_proj.bias": "mlp.gate.down_proj.bias",
            "zaya_block.router.router_states_scale": "mlp.gate.router_states_scale",
            "zaya_block.router.rmsnorm_eda.weight": "mlp.gate.router_mlp.norm.weight",
            "zaya_block.router.router_mlp.0.weight": "mlp.gate.router_mlp.fc1.weight",
            "zaya_block.router.router_mlp.0.bias": "mlp.gate.router_mlp.fc1.bias",
            "zaya_block.router.router_mlp.2.weight": "mlp.gate.router_mlp.fc2.weight",
            "zaya_block.router.router_mlp.2.bias": "mlp.gate.router_mlp.fc2.bias",
            "zaya_block.router.router_mlp.4.weight": "mlp.gate.router_mlp.out_proj.weight",
        }
        res = ("hidden_states_scale", "hidden_states_bias", "residual_scale", "residual_bias")
        for i in range(n_block):
            a, m = f"model.layers.{2 * i}.", f"model.layers.{2 * i + 1}."
            b = f"model.layers.{i}."
            rename.update({a + k: b + v for k, v in attn.items()})
            rename.update({m + k: b + v for k, v in moe.items()})
            # a half-layer's res_scale merges the residual in front of it: the MoE half-layer's is
            # the block's post-attention scale, the next attention half-layer's its post-MLP scale
            rename.update({f"{m}res_scale.{k}": f"{b}post_attention_residual_scale.{k}" for k in res})
            if i + 1 < n_block:
                rename.update({f"model.layers.{2 * i + 2}.res_scale.{k}": f"{b}post_mlp_residual_scale.{k}" for k in res})
        for name, gen in tensors.items():
            if ".local_experts." in name:
                continue
            if name == "lm_head.weight":
                out[name] = gen
                continue
            if name not in rename:
                raise ValueError(f"zaya: unmapped legacy tensor {name}")
            out[rename[name]] = gen
        for i in range(n_block):
            pre = f"model.layers.{2 * i + 1}.zaya_block.experts.local_experts."
            for src, dst in (("linear_fc1", "gate_up_proj"), ("linear_fc2", "down_proj")):
                gens = [tensors[f"{pre}{e}.{src}.weight"] for e in range(n_expert)]
                out[f"model.layers.{i}.mlp.experts.{dst}"] = lambda gens=gens: torch.stack([g() for g in gens])
        out.update(vl)
        return out

    def set_vocab(self):
        # the Gemma 3/4 BPE tokenizer; added tokens that are not special (such as "\n") stay
        # USER_DEFINED so they are matched in prompts and rendered in output
        vocab = gguf.LlamaHfVocab(self._tokenizer_dir())
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
        # ZAYA1-74B: layer_types alternates hybrid_sliding / hybrid; sliding layers attend to the
        # last sliding_window positions and use their own rope theta (rope_parameters.hybrid_sliding)
        layer_types = hp.get("layer_types") or []
        if hp.get("sliding_window") and "hybrid_sliding" in layer_types:
            self.gguf_writer.add_sliding_window(int(hp["sliding_window"]))
            self.gguf_writer.add_sliding_window_pattern([t == "hybrid_sliding" for t in layer_types])
            swa_rope = hp.get("rope_parameters", {}).get("hybrid_sliding", {})
            if "rope_theta" in swa_rope:
                self.gguf_writer.add_rope_freq_base_swa(float(swa_rope["rope_theta"]))
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


@ModelBase.register("Zaya1VLForConditionalGeneration")
class ZayaVLModel(ZayaModel):
    """Zyphra ZAYA1-VL: the ZAYA1 language model (Megatron-style checkpoint) with vision-only LoRA
    on CCA and on every expert, used on image tokens; the vision tower goes to the mmproj."""
    model_arch = gguf.MODEL_ARCH.ZAYA

    # Zyphra's template only takes content as a list of image and text parts. llama.cpp passes a
    # message with media as a string holding a marker, <__media_<id>__> with an id random per server,
    # where each image was; this one takes both, and puts the images (the markers) in front of the
    # user turn, as Zyphra's does (mtmd wraps each image in <|vision_start|> ... <|vision_end|>).
    _CHAT_TEMPLATE = (
        '{%- for message in messages -%}'
        "{%- if message['content'] is string -%}"
        "{%- set ns = namespace(text=message['content'], media='') -%}"
        '{%- else -%}'
        "{%- set ns = namespace(text=message['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | join(''), media='') -%}"
        "{%- for c in message['content'] | selectattr('type', 'equalto', 'image') -%}"
        "{%- set ns.media = ns.media ~ '<|vision_start|><image><|vision_end|>\\n' -%}"
        '{%- endfor -%}'
        '{%- endif -%}'
        "{%- set segs = ns.text.split('<__media_') -%}"
        '{%- if segs | length > 1 -%}'
        '{%- set ns.text = segs[0] -%}'
        '{%- for seg in segs[1:] -%}'
        "{%- set bits = seg.split('>') -%}"
        "{%- set ns.media = ns.media ~ '<__media_' ~ bits[0] ~ '>\\n' -%}"
        "{%- set ns.text = ns.text ~ (bits[1:] | join('>')) -%}"
        '{%- endfor -%}'
        '{%- set ns.text = ns.text | trim -%}'
        '{%- endif -%}'
        "{%- if message['role'] == 'user' -%}"
        "{{ ns.media ~ '<|im_start|>user\\n' ~ ns.text ~ '<|im_end|>\\n' }}"
        '{%- else -%}'
        "{{ '<|im_start|>' ~ message['role'] ~ '\\n' ~ ns.text ~ '<|im_end|>' }}"
        '{%- endif -%}'
        '{%- endfor -%}'
        '{%- if add_generation_prompt -%}'
        "{{ '<|im_start|>assistant\\n' }}"
        '{%- endif -%}'
    )

    def set_vocab(self):
        vocab = gguf.LlamaHfVocab(self._tokenizer_dir())
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
        special_vocab.chat_template = self._CHAT_TEMPLATE
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_space_prefix(False)
        self.gguf_writer.add_add_bos_token(True)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.hparams.get("vision_lora"):
            arch = self.gguf_writer.arch
            self.gguf_writer.add_uint32(f"{arch}.vision_lora.attention_rank", int(self.hparams["vision_lora_rank_attn"]))
            self.gguf_writer.add_uint32(f"{arch}.vision_lora.ffn_rank", int(self.hparams["vision_lora_rank_mlp"]))


@ModelBase.register("Zaya1VLForConditionalGeneration")
class ZayaVLVisionModel(Qwen2VLVisionModel):
    """ZAYA1-VL's vision tower: Qwen2.5-VL's, under vision_tower."""

    def __init__(self, *args, **kwargs):
        # the checkpoint's vision_config lists only what differs from Qwen2.5-VL's defaults
        # (no depth, heads, window pattern); fill it as transformers does
        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        hparams = kwargs.get("hparams") or ModelBase.load_hparams(args[0], False)
        vc = Qwen2_5_VLVisionConfig(**hparams["vision_config"]).to_dict()
        kwargs["hparams"] = {**hparams, "vision_config": {**vc, **hparams["vision_config"]}}
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # Qwen2VLVisionModel picks the projector from the top-level model_type (zaya1_vl here)
        self.global_config["model_type"] = self.hparams_vision["model_type"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        # ZAYA1-VL attends to each image bidirectionally (the image tokens are not causal)
        self.gguf_writer.add_vision_decode_non_causal(True)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("vision_tower."):
            return None
        return super().filter_tensors(("visual." + name.removeprefix("vision_tower."), gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "patch_embed.proj.weight" in name and data_torch.shape[2] == 1:
            # temporal_patch_size 1: clip's Qwen2-VL graph sums two patch convs over the same
            # still image (Qwen feeds the frame twice), so the second one is all zeros
            w = data_torch[:, :, 0, ...]
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight", w)
            yield (gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.V_ENC_EMBD_PATCH] + ".weight.1", torch.zeros_like(w))
            return
        yield from super().modify_tensors(data_torch, name, bid)
