from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BartForCausalLM", "MBartForCausalLM", "MarianForCausalLM",
                    "BioGptForCausalLM", "XGLMForCausalLM")
class BartsModel(TextModel):
    """Post-norm decoder: BART/MBart/Marian/BioGpt decoders and XGLM.

    All of these are a LayerNorm+GELU Transformer decoder with the norm applied *after*
    the residual (`x = norm(x + sublayer(x))`), separate Q/K/V projections and an
    fc1/fc2 MLP.  BART/MBart/Marian additionally normalise the summed embeddings
    (`layernorm_embedding`, GGUF `token_embd_norm`); XGLM does not.  Used standalone as
    a causal decoder there is no cross-attention, so any `encoder_attn*` tensors are
    dropped.
    """

    model_arch = gguf.MODEL_ARCH.BARTS

    def set_gguf_parameters(self):
        h = self.hparams
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.find_hparam(["max_position_embeddings", "n_positions"]))
        self.gguf_writer.add_embedding_length(self.find_hparam(["d_model", "hidden_size"]))
        self.gguf_writer.add_feed_forward_length(self.find_hparam(
            ["decoder_ffn_dim", "ffn_dim", "intermediate_size"]))
        self.gguf_writer.add_head_count(self.find_hparam(
            ["decoder_attention_heads", "num_attention_heads", "attention_heads"]))
        self.gguf_writer.add_layer_norm_eps(self.find_hparam(
            ["layer_norm_eps", "layer_norm_epsilon", "norm_eps"]))
        self.gguf_writer.add_file_type(self.ftype)

    def set_vocab(self):
        self._set_vocab_gpt2()

    _MAP = [
        ("embed_tokens", "token_embd"),
        ("embed_positions", "position_embd"),
        ("layernorm_embedding", "token_embd_norm"),
        ("self_attn_layer_norm", "attn_norm"),
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.out_proj", "attn_output"),
        ("final_layer_norm", "ffn_norm"),
        ("fc1", "ffn_up"),
        ("fc2", "ffn_down"),
        ("layer_norm", "output_norm"),
        ("lm_head", "output"),
    ]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # unused when the decoder runs causally
        if "encoder_attn" in name:
            return
        for src, dst in self._MAP:
            if name.endswith(src + ".weight") or name.endswith(src + ".bias"):
                suffix = name[name.rfind(src) + len(src):]
                name = name[: name.rfind(src)] + dst + suffix
                break
        yield from super().modify_tensors(data_torch, name, bid)
