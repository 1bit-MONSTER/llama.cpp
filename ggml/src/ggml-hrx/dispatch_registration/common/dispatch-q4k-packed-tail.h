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

// engine #2137: one device copy of a Q4_K weight for every batch size.
//
// Decode and the 256-aligned prefill kernels read a Q4_K weight in HRX's packed row layout. The generic tiled kernels
// (plain matmul, the post-op chains, the fused gate/up pair) take every other batch size, a prompt's last, unaligned
// batch among them, and read the stored layout. Since llama.cpp #304 the weights stay in the loader's file mapping and
// each layout HRX reads is a device copy of its own, so a prompt with a remainder left two copies of each Q4_K weight
// (Qwen3-4B Q4_K_M, 1,700-token prompt: 4,032.5 MiB of copies against a 2,375.9 MiB model).
//
// With GGML_HRX_Q4K_PACKED_TAIL=1 the tiled kernels read the packed layout too (weight format 44). Opt-in until PPL
// and tok/s are measured against the stored path. It only saves memory when the stored layout costs a copy, that is
// with #304's mapping; with GGML_HRX_MODEL_BUFFER_COPY=1 it adds packed copies instead.

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/value-map.h"

#include <cstdlib>

namespace ggml::hrx {

// Says which path it took, once, at ERROR: the CLI's log sink drops WARN and below, and this selection changes a
// compile parameter inside existing registrations, so GGML_HRX_LOG_DISPATCH cannot show it.
inline bool q4k_packed_tail_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_Q4K_PACKED_TAIL");
        const bool   on  = env != nullptr && env[0] == '1';
        GGML_LOG_ERROR("%s: engine #2137 q4_k batches off the 256 grid read the %s layout (GGML_HRX_Q4K_PACKED_TAIL=%s)\n",
                       __func__, on ? "PACKED" : "stored", on ? "1" : "unset");
        return on;
    }();
    return enabled;
}

// The layout's own preconditions, and a weight that may be packed at all. A packed copy is made of a weight once; a
// tensor in an ordinary buffer can be rewritten between computes, and the executor refuses to read one in a packed
// layout ("its buffer is not flagged as weights"), which would fail a graph the stored layout runs (measured on
// test-backend-ops' q4_K large-activation cases at 22 tokens).
inline bool q4k_packed_tail_weight_ok(const Value & weight, int64_t input_size, int64_t output_size) {
    if (weight.type != GGML_TYPE_Q4_K || input_size % 256 != 0 || output_size % 64 != 0 ||
        weight.alias_source.value >= 0) {
        return false;
    }
    const ggml_tensor *   tensor = weight.tensor;
    ggml_backend_buffer_t buffer =
        tensor == nullptr ? nullptr : tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    return buffer != nullptr && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

}  // namespace ggml::hrx
