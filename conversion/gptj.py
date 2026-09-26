from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GPTJForCausalLM")
class GPTJModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GPTJ

    def set_gguf_parameters(self):
        hparams = self.hparams
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hparams.get("n_positions", hparams.get("n_ctx", 2048)))
        self.gguf_writer.add_embedding_length(hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * hparams["n_embd"])
        self.gguf_writer.add_head_count(hparams["n_head"])
        self.gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])

        # GPT-J applies rotary to the first `rotary_dim` of each head (partial RoPE).
        n_embd_head = hparams["n_embd"] // hparams["n_head"]
        n_rot = int(hparams.get("rotary_dim", n_embd_head))
        if n_rot < n_embd_head:
            self.gguf_writer.add_rope_dimension_count(n_rot)

        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_vocab_base_pre(self, tokenizer) -> str:
        # GPT-J's tokenizer is GPT-2's byte-level BPE; its checksum is just not in the list
        return "gpt-2"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # HF keeps the projections as nn.Linear (no Conv1D permute); the tensor map
        # already knows the GPT-J names, and biases ride the usual suffix path.
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            return  # causal-mask buffers older checkpoints carry, not weights

        yield from super().modify_tensors(data_torch, self.map_tensor_name(name), bid)
