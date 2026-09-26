from __future__ import annotations

from .base import ModelBase
from .qwen import Qwen2Model


@ModelBase.register("CambrianQwenForCausalLM")
class CambrianQwenModel(Qwen2Model):
    """Cambrian's Qwen text tower: config model_type is `qwen2`, tensors and shapes are
    Qwen2's. It runs on the qwen2 architecture; only the class name differs."""
