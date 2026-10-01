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

// Reduced draft vocabulary for the qwen4exp (Qwen3.8-Flash-Next) MTP head (1bit engine,
// tools/mtp_draft_vocab.py). A draft head GGUF may carry
//   blk.<L>.nextn.shared_head_head.weight  [n_embd, K]  the target's output rows for K token ids
//   d2t                                    I64 [K]      draft row -> target token id
// The draft pass then reads a K-row head instead of the full one (65,536 of 248,320 rows is
// 170 MiB instead of 644 MiB at Q8_0, three times per decode step) and its logits are scattered
// into a full-vocabulary row that is -inf elsewhere. The target verifies every drafted token,
// so only the acceptance rate can change, never the verified output. The d2t values are
// checked by the tool that writes them (sorted, unique, < n_vocab); set_rows trusts them.

#include "llama-model-loader.h"

#include <cstdint>
#include <string>

struct ggml_context;
struct ggml_tensor;

// Rows of the reduced draft head: 0 when the file has no d2t (full head), else d2t's length.
// Throws if d2t is not I64 or is longer than the vocabulary.
int64_t qwen4exp_draft_vocab_rows(const llama_model_loader & ml, const std::string & d2t_name, int64_t n_vocab);

// logits [K, n_outputs] -> [n_vocab, n_outputs], -inf outside the d2t rows.
ggml_tensor * qwen4exp_draft_vocab_scatter(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * d2t, int64_t n_vocab);
