from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("CodeGenForCausalLM")
class CodeGenModel(TextModel):
    model_arch = gguf.MODEL_ARCH.CODEGEN

    def set_gguf_parameters(self):
        hparams = self.hparams
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hparams["n_positions"])
        self.gguf_writer.add_embedding_length(hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * hparams["n_embd"])
        self.gguf_writer.add_head_count(hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])

        n_embd_head = hparams["n_embd"] // hparams["n_head"]
        n_rot = int(hparams.get("rotary_dim", n_embd_head))
        if n_rot < n_embd_head:
            self.gguf_writer.add_rope_dimension_count(n_rot)

        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith((".attn.bias", ".attn.masked_bias", ".attn.causal_mask")):  # mask buffers
            return
        if name.endswith(".attn.qkv_proj.weight"):
            # HF splits qkv_proj into mp_num = 4 blocks, each ordered (query, value, key)
            # (CodeGenAttention.forward); the fused-QKV graph wants all of q, then k, then v
            n_embd = self.hparams["n_embd"]
            w = data_torch.reshape(4, 3, n_embd // 4, n_embd)
            q, v, k = w[:, 0], w[:, 1], w[:, 2]
            data_torch = torch.cat([t.reshape(n_embd, n_embd) for t in (q, k, v)], dim=0)
        yield from super().modify_tensors(data_torch, name, bid)
