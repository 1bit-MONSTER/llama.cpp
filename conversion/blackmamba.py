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
"""BlackMamba (Zyphra/BlackMamba-1.5B, -2.8B): Mamba-1 layers alternating with Switch MoE layers
(src/models/blackmamba.cpp). The checkpoint is Megatron-style: config.json without architectures
(conversion/base.py routes on mamba_moe_layers) and pytorch_model.bin holding {"model": state_dict}.
No tokenizer ships with it; it uses GPT-NeoX's (models/ggml-vocab-gpt-neox.gguf)."""

from __future__ import annotations

import math

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BlackMambaForCausalLM")
class BlackMambaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BLACKMAMBA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hp = self.hparams
        self.layer_types = hp["mamba_moe_layers"]  # "r": Mamba-1 layer, "8": MoE with 8 experts
        self.n_expert = max(int(t) for t in self.layer_types if t != "r")
        self.d_inner = hp["expansion_factor"] * hp["hidden_size"]
        self.dt_rank = math.ceil(hp["hidden_size"] / 16)
        self._experts: dict[int, dict[str, Tensor]] = {}

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        sd = torch.load(self.dir_model / "pytorch_model.bin", map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        return {name: (lambda t=t: t) for name, t in sd.items()}

    def set_vocab(self):
        self._set_vocab_builtin("gpt-neox", self.hparams["vocab_size"])

    def set_gguf_parameters(self):
        hp = self.hparams
        n_ff = hp["ffn_hidden_size"]
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hp.get("max_sequence_length", 2048))
        self.gguf_writer.add_embedding_length(hp["hidden_size"])
        self.gguf_writer.add_feed_forward_length([0 if t == "r" else n_ff for t in self.layer_types])
        self.gguf_writer.add_head_count(0)
        self.gguf_writer.add_layer_norm_rms_eps(1e-5)
        self.gguf_writer.add_layer_norm_eps(1e-5)
        self.gguf_writer.add_expert_count(self.n_expert)
        self.gguf_writer.add_expert_used_count(1)
        self.gguf_writer.add_expert_feed_forward_length(n_ff)
        self.gguf_writer.add_ssm_conv_kernel(hp["conv_dimension"])
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        self.gguf_writer.add_ssm_state_size(hp["state_size"])
        self.gguf_writer.add_ssm_time_step_rank(self.dt_rank)
        self.gguf_writer.add_file_type(self.ftype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        if name == "embedding.word_embeddings.weight":
            yield self.format_tensor_name(T.TOKEN_EMBD), data_torch
            return
        if name.startswith("decoder.final_layernorm."):
            yield self.format_tensor_name(T.OUTPUT_NORM, suffix="." + name.rsplit(".", 1)[1]), data_torch
            return
        assert name.startswith("decoder.layers."), name
        bid = int(name.split(".")[2])
        tail = name.split(".", 3)[3]
        if tail == "norm.weight":
            yield self.format_tensor_name(T.ATTN_NORM, bid), data_torch
            return
        mamba = {
            "mixer.in_proj.weight":  (T.SSM_IN, ".weight"),
            "mixer.x_proj.weight":   (T.SSM_X, ".weight"),
            "mixer.dt_proj.weight":  (T.SSM_DT, ".weight"),
            "mixer.dt_proj.bias":    (T.SSM_DT, ".bias"),
            "mixer.conv1d.bias":     (T.SSM_CONV1D, ".bias"),
            "mixer.out_proj.weight": (T.SSM_OUT, ".weight"),
            "mixer.router.weight":   (T.FFN_GATE_INP, ".weight"),
            "mixer.router.bias":     (T.FFN_GATE_INP, ".bias"),
        }
        if tail in mamba:
            key, suffix = mamba[tail]
            yield self.format_tensor_name(key, bid, suffix=suffix), data_torch
        elif tail == "mixer.conv1d.weight":
            yield self.format_tensor_name(T.SSM_CONV1D, bid), data_torch.squeeze(1)
        elif tail == "mixer.A_log":
            yield self.format_tensor_name(T.SSM_A, bid, suffix=""), -torch.exp(data_torch.float())
        elif tail == "mixer.D":
            yield self.format_tensor_name(T.SSM_D, bid, suffix=""), data_torch
        elif tail.startswith("mixer.local_experts."):
            # mixer.local_experts.{e}.linear_fc{1,2}.weight; fc1 rows are [gate | up] (chunk(2) in mlp.py)
            _, _, e, fc, _ = tail.split(".")
            self._experts.setdefault(bid, {})[f"{e}.{fc}"] = data_torch
            got = self._experts[bid]
            if len(got) < 2 * self.n_expert:
                return
            fc1 = [got[f"{i}.linear_fc1"] for i in range(self.n_expert)]
            fc2 = [got[f"{i}.linear_fc2"] for i in range(self.n_expert)]
            n_ff = fc1[0].shape[0] // 2
            yield self.format_tensor_name(T.FFN_GATE_EXP, bid), torch.stack([w[:n_ff] for w in fc1])
            yield self.format_tensor_name(T.FFN_UP_EXP,   bid), torch.stack([w[n_ff:] for w in fc1])
            yield self.format_tensor_name(T.FFN_DOWN_EXP, bid), torch.stack(fc2)
            del self._experts[bid]
        else:
            raise ValueError(f"blackmamba: unexpected tensor {name}")
