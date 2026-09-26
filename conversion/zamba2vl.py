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

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .qwenvl import Qwen2VLVisionModel
from .zamba2 import Zamba2Model


# Zyphra Zamba2-VL (1.2B, 2.7B, 7B): a Zamba2 language model under `language_model.` and
# Qwen2.5-VL's vision tower and merger under `vision_tower.`. The language model's tensors are
# named as in Zamba2ForCausalLM, and images enter it as embeddings between <|vision_start|> and
# <|vision_end|>, the tokens mtmd wraps Qwen2.5-VL images in.


@ModelBase.register("Zamba2_VLForConditionalGeneration")
class Zamba2VLTextModel(Zamba2Model):
    model_arch = gguf.MODEL_ARCH.ZAMBA2

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("language_model."):
            return None
        return super().filter_tensors((name.removeprefix("language_model."), gen))

    def set_vocab(self):
        # tokenizer.json holds 32005 tokens (Mistral's 32000, <unk> and the image/video markers);
        # the embedding has vocab_size (32064) rows, so pad the list as the sentencepiece path does
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens, scores, toktypes = [], [], []
        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)
        for i in range(len(tokens), self.hparams["vocab_size"]):
            tokens.append(f"[PAD{i}]".encode("utf-8"))
            scores.append(-1000.0)
            toktypes.append(gguf.TokenType.UNUSED)

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.chat_template = self._CHAT_TEMPLATE
        special_vocab.add_to_gguf(self.gguf_writer)

    # Zyphra's template only takes content as a list of parts. llama.cpp passes a string, with a
    # media marker where each image was; this one takes both, and puts the images in front of the
    # user turn as Zyphra's does (mtmd wraps each in <|vision_start|> ... <|vision_end|>).
    _CHAT_TEMPLATE = (
        "{%- for message in messages -%}"
        "{%- if message['content'] is string -%}"
        "{%- set text = message['content'] -%}"
        "{%- set n_img = (text.split('<__media__>') | length) - 1 -%}"
        "{%- set text = text.replace('<__media__>', '') | trim -%}"
        "{%- else -%}"
        "{%- set text = message['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | join('') -%}"
        "{%- set n_img = message['content'] | selectattr('type', 'equalto', 'image') | list | length -%}"
        "{%- endif -%}"
        "{%- if message['role'] == 'user' -%}"
        "{%- for i in range(n_img) -%}{{ '<__media__>\\n' }}{%- endfor -%}"
        "{{ '<s>user\\n' ~ text ~ '</s>\\n' }}"
        "{%- else -%}"
        "{{ '<s>' ~ message['role'] ~ '\\n' ~ text ~ '</s>' }}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{ '<s>assistant\\n' }}{%- endif -%}"
    )


@ModelBase.register("Zamba2_VLForConditionalGeneration")
class Zamba2VLVisionModel(Qwen2VLVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # Qwen2VLVisionModel picks the projector from the top-level model_type (zamba2_vl here);
        # the tower is Qwen2.5-VL's
        self.global_config["model_type"] = self.hparams_vision["model_type"]

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
