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
// The tiled kernels can read the packed layout too (weight format 44). That saves a copy exactly when the stored
// layout costs one, which is when the weight is in #304's file mapping; for a weight that was uploaded
// (GGML_HRX_MODEL_BUFFER_COPY=1, a host-weight limit, or a load without mmap) the stored layout is already on the
// device and the packed one would be the extra copy. So the default follows the weight: packed for a weight in the
// file mapping, stored otherwise. GGML_HRX_Q4K_PACKED_TAIL=1 forces packed for every weight, =0 forces stored.

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "graph/value-map.h"
#include "model-buffer-from-mmap.h"

#include <cstdlib>

namespace ggml::hrx {

// Says which path it took, once, at ERROR: the CLI's log sink drops WARN and below, and this selection changes a
// compile parameter inside existing registrations, so GGML_HRX_LOG_DISPATCH cannot show it.
enum class PackedLayoutSwitch { FollowWeight, Stored, Packed };

inline PackedLayoutSwitch packed_layout_switch(const char * name, const char * what) {
    const char *             env   = std::getenv(name);
    const PackedLayoutSwitch value = env == nullptr || env[0] == '\0' ? PackedLayoutSwitch::FollowWeight :
                                     env[0] == '0'                    ? PackedLayoutSwitch::Stored :
                                                                        PackedLayoutSwitch::Packed;
    GGML_LOG_ERROR("%s: engine #2137 %s: %s (%s=%s)\n", __func__, what,
                   value == PackedLayoutSwitch::Packed ? "PACKED layout for every weight" :
                   value == PackedLayoutSwitch::Stored ? "stored layout for every weight" :
                                                         "PACKED layout for a weight in the file mapping, stored otherwise",
                   name, env == nullptr || env[0] == '\0' ? "unset" : env);
    return value;
}

// The weight's own buffer decides the default (see the header comment).
inline bool packed_layout_wanted(PackedLayoutSwitch value, const Value & weight) {
    if (value != PackedLayoutSwitch::FollowWeight) {
        return value == PackedLayoutSwitch::Packed;
    }
    const ggml_tensor * tensor = weight.tensor;
    return tensor != nullptr &&
           buffer_is_file_mapped(tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer);
}

inline bool q4k_packed_tail_enabled(const Value & weight) {
    static const PackedLayoutSwitch value =
        packed_layout_switch("GGML_HRX_Q4K_PACKED_TAIL", "q4_k batches off the 256 grid");
    return packed_layout_wanted(value, weight);
}

// The layout's own preconditions, and a weight that may be packed at all. A packed copy is made of a weight once; a
// tensor in an ordinary buffer can be rewritten between computes, and the executor refuses to read one in a packed
// layout ("its buffer is not flagged as weights"), which would fail a graph the stored layout runs (measured on
// test-backend-ops' q4_K large-activation cases at 22 tokens).
inline bool packed_tail_weight_ok(const Value & weight, ggml_type type, int64_t input_size, int64_t output_size) {
    if (weight.type != type || input_size % 256 != 0 || output_size % 64 != 0 || weight.alias_source.value >= 0) {
        return false;
    }
    const ggml_tensor *   tensor = weight.tensor;
    ggml_backend_buffer_t buffer =
        tensor == nullptr ? nullptr : tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    return buffer != nullptr && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

inline bool q4k_packed_tail_weight_ok(const Value & weight, int64_t input_size, int64_t output_size) {
    return packed_tail_weight_ok(weight, GGML_TYPE_Q4_K, input_size, output_size);
}

// Q6_K: prefill reads the packed row layout decode reads, at every batch size. The tiled kernels take it as weight
// format 46 and the 128-aligned f16 prefill kernel as weight_layout 2. Same default as Q4_K above: packed for a weight
// in the file mapping, stored otherwise; GGML_HRX_Q6K_PACKED_ROWS=1 forces packed, =0 forces stored.
// (GGML_HRX_Q6K_PACKED_PREFILL is an older switch on the aligned kernel alone; it binds the stored layout and saves no
// copy.)
inline bool q6k_packed_rows_enabled(const Value & weight) {
    static const PackedLayoutSwitch value = packed_layout_switch("GGML_HRX_Q6K_PACKED_ROWS", "q6_k prefill");
    return packed_layout_wanted(value, weight);
}

inline bool q6k_packed_rows_weight_ok(const Value & weight, int64_t input_size, int64_t output_size) {
    return packed_tail_weight_ok(weight, GGML_TYPE_Q6_K, input_size, output_size);
}

}  // namespace ggml::hrx
