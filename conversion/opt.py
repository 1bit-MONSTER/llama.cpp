from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("OPTForCausalLM")
class OPTModel(TextModel):
    model_arch = gguf.MODEL_ARCH.OPT

    def set_gguf_parameters(self):
        hparams = self.hparams
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(hparams["ffn_dim"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_layer_norm_eps(hparams.get("layer_norm_eps", 1e-5))
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # OPT's learned positions carry two padding rows (offset=2); drop them so the
        # runtime's 0-based positions line up with the GGUF rows.
        if name.endswith("embed_positions.weight"):
            data_torch = data_torch[2:]
        yield from super().modify_tensors(data_torch, name, bid)
