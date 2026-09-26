from __future__ import annotations

from typing import Iterable, TYPE_CHECKING


if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("PicoDecoderHF")
class PicoDecoderModel(TextModel):
    """pico-lm's decoder: an ordinary Llama (RMSNorm, GQA, RoPE, SwiGLU) with its own
    module names. It runs on the llama architecture; only the converter differs."""

    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_gguf_parameters(self):
        c = self.hparams
        self.gguf_writer.add_block_count(c["n_layers"])
        self.gguf_writer.add_context_length(c["max_seq_len"])
        self.gguf_writer.add_embedding_length(c["d_model"])
        self.gguf_writer.add_feed_forward_length(c["activation_hidden_dim"])
        self.gguf_writer.add_head_count(c["attention_n_heads"])
        self.gguf_writer.add_head_count_kv(c["attention_n_kv_heads"])
        self.gguf_writer.add_layer_norm_rms_eps(c["norm_eps"])
        self.gguf_writer.add_rope_freq_base(c.get("position_emb_theta", 10000.0))
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the HF wrapper nests the real module under `pico_decoder.`
        if name.startswith("pico_decoder."):
            name = name[len("pico_decoder."):]
        yield from super().modify_tensors(data_torch, name, bid)
