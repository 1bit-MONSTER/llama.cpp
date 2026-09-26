from __future__ import annotations

from typing import Iterable, TYPE_CHECKING


if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("GPTNeoForCausalLM")
class GPTNeoModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GPTNEO

    def set_gguf_parameters(self):
        hparams = self.hparams
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_feed_forward_length(4 * hparams["hidden_size"])
        self.gguf_writer.add_head_count(hparams["num_heads"])
        self.gguf_writer.add_layer_norm_eps(hparams["layer_norm_epsilon"])
        self.gguf_writer.add_sliding_window(hparams["window_size"])
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # attention bias buffers are rebuilt at runtime
        if name.endswith((".attn.bias", ".attn.masked_bias", ".attn.attention.bias", ".attn.attention.masked_bias")):
            return
        yield from super().modify_tensors(data_torch, name, bid)
