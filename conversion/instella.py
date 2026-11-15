"""Instella-MoE (amd/Instella-MoE-16B-A3B-Think) — gguf conversion.

Read from the reference implementation in the checkpoint itself
(`inference/transformers/models/instella_moe/modeling_instella_moe.py`), not from the paper:

* **Gated MLA** — `MLAGatedAttention.forward` does
  `attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))` on the *flattened* per-head output
  (num_heads * v_head_dim = 16 * 128 = 2048), and only then `o_proj`. `gate_proj` is `nn.Linear(hidden, heads*v_dim,
  bias=False)`, one per layer. `q_lora_rank` is `None`, so there is a direct `q_proj` and no q_a/q_b pair.
* **FarSkip-Collective** — `FarSkipDecoderLayer.forward` carries a *pair* residual `(residual, residual_no_routed)`;
  the MoE returns `(routed+shared, shared)` so the routed-free stream can be built, and `input_to_mlp` is bound
  BEFORE `residual = residual + attn_output`, i.e. the MoE reads the pre-attention residual.

Everything else is DeepSeek-V2/V3 MLA + MoE, so this class extends `DeepseekV2Model` and only adds the gate tensor
and the four header keys.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

from .base import TextModel, ModelBase, gguf
from .deepseek import DeepseekV2Model

logger = logging.getLogger(__name__)


@ModelBase.register("InstellaMoEForCausalLM", "InstellaMoEModel")
class InstellaMoEModel(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.INSTELLA_MOE

    def set_gguf_parameters(self):
        # DeepSeek-V2's parameters are all correct for this model (MLA -> MQA with num_key_value_heads = 1,
        # kv_lora_rank / key_length_mla / value_length_mla, leading_dense_block_count from first_k_dense_replace,
        # expert counts and routed_scaling_factor).
        super().set_gguf_parameters()

        hparams = self.hparams
        # Gated MLA is not optional for these checkpoints: the gate tensor is present in all 27 layers and the graph
        # multiplies by it before o_proj, so the value is written even when the config predates the flag.
        self.gguf_writer.add_instella_gated_attention(bool(hparams.get("gated_attention", False)))
        self.gguf_writer.add_instella_farskip(bool(hparams.get("farskip", False)))
        # The range decides WHICH layers switch the dataflow. Defaults come from configuration_instella_moe.py
        # (0 .. 1e4), i.e. every layer; end_idx is clamped to num_hidden_layers - 1 by the reference.
        n_layer = int(hparams["num_hidden_layers"])
        start = max(0, int(hparams.get("farskip_start_idx", 0)))
        end = min(int(hparams.get("farskip_end_idx", 1e4)), n_layer - 1)
        self.gguf_writer.add_instella_farskip_start_idx(start)
        self.gguf_writer.add_instella_farskip_end_idx(end)
        logger.info("instella-moe: gated_attention=%s farskip=%s over layers %d..%d of %d",
                    hparams.get("gated_attention"), hparams.get("farskip"), start, end, n_layer)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        # The attention gate is the one tensor stock deepseek_v3 does not have: map it to blk.N.attn_gate.weight and
        # let the base class handle every other name (MLA projections, experts merging, shared experts, norms).
        if bid is not None and name.endswith("self_attn.gate_proj.weight"):
            # Same shape as conversion/qwen.py's gate: the GGUF name comes from the tensor type, not from the HF
            # spelling (ATTN_GATE maps to blk.{bid}.attn_gate, which no HF suffix maps onto by itself).
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"), data_torch)
            return
        yield from super().modify_tensors(data_torch, name, bid)
