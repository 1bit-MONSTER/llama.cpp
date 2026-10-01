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
#include "qwen4exp-draft-vocab.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cmath>
#include <stdexcept>

int64_t qwen4exp_draft_vocab_rows(const llama_model_loader & ml, const std::string & d2t_name, int64_t n_vocab) {
    const auto * w = ml.get_weight(d2t_name.c_str());
    if (w == nullptr) {
        return 0;
    }
    const ggml_tensor * d2t = w->tensor;
    if (d2t->type != GGML_TYPE_I64 || ggml_n_dims(d2t) != 1) {
        throw std::runtime_error(format("QWEN4EXP MTP: d2t must be a 1-D I64 tensor, got %s", ggml_type_name(d2t->type)));
    }
    if (d2t->ne[0] <= 0 || d2t->ne[0] > n_vocab) {
        throw std::runtime_error(format("QWEN4EXP MTP: d2t has %lld rows for a %lld-token vocabulary",
                                        (long long) d2t->ne[0], (long long) n_vocab));
    }
    return d2t->ne[0];
}

ggml_tensor * qwen4exp_draft_vocab_scatter(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * d2t, int64_t n_vocab) {
    const int64_t n_rows    = logits->ne[0];
    const int64_t n_outputs = logits->ne[1];
    GGML_ASSERT(d2t->type == GGML_TYPE_I64 && d2t->ne[0] == n_rows);

    ggml_tensor * full = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
    full = ggml_set_rows(ctx, full,
            ggml_reshape_3d(ctx, logits, 1,      n_rows, n_outputs),
            ggml_reshape_3d(ctx, d2t,    n_rows, 1,      1));
    return ggml_reshape_2d(ctx, full, n_vocab, n_outputs);
}
