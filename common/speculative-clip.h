// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
// engine#3613 / #3627: a DFlash drafter keeps logits as wide as its own tokenizer (the row width llama_context copies
// out with). When that tokenizer lists ids the target lacks (an Inkling-2-Mini export that lists the target's padded
// vocab rows: 201,024 vs 200,058), a draft can propose one; cut the draft at the first such id, and its per-token
// candidate list with it.
#include "llama.h"

#include <cstddef>
#include <vector>

static inline void common_speculative_clip_to_vocab(std::vector<llama_token> & draft,
                                                    std::vector<std::vector<llama_token_data>> * draft_q, int32_t n_vocab_tgt) {
    for (size_t i = 0; i < draft.size(); ++i) {
        if (draft[i] < 0 || draft[i] >= n_vocab_tgt) {
            draft.resize(i);
            if (draft_q && draft_q->size() > i) {
                draft_q->resize(i);
            }
            return;
        }
    }
}
