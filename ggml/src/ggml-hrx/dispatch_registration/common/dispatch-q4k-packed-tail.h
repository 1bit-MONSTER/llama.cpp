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

// engine #2137: a weight that already has a device copy in the stored layout. On a mixed-quant dense model
// (Qwen3.8-27B UD-Q4_K_XL) the one-token decode goes through the kquant family, which reads the GGUF block layout, so
// a prefill kernel that reads the same weight in the packed rows makes a second device copy of it. Where the prefill
// kernel has a stored form, it keeps the stored layout for such a weight. The copy is recorded when it is made
// (model-buffer-from-mmap.h), so this follows what ran, not what a matcher was asked about. A weight is known once a
// decode has run, which the tools' warm-up does before the first prompt.
// GGML_HRX_PREFILL_FOLLOWS_DECODE=0 turns the lookup off.
inline bool prefill_follows_decode_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_PREFILL_FOLLOWS_DECODE");
        return env == nullptr || env[0] != '0';
    }();
    return enabled;
}

inline bool decode_reads_stored(const Value & weight) {
    const ggml_tensor * tensor = weight.tensor;
    if (tensor == nullptr || !prefill_follows_decode_enabled()) {
        return false;
    }
    return has_stored_device_copy(tensor->view_src != nullptr ? tensor->view_src->data : tensor->data);
}

inline bool weight_in_file_mapping(const Value & weight) {
    const ggml_tensor * tensor = weight.tensor;
    return tensor != nullptr &&
           buffer_is_file_mapped(tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer);
}

// The weight's own buffer decides the default (see the header comment), unless decode reads the weight stored.
inline bool packed_layout_wanted(PackedLayoutSwitch value, const Value & weight) {
    if (value != PackedLayoutSwitch::FollowWeight) {
        return value == PackedLayoutSwitch::Packed;
    }
    return weight_in_file_mapping(weight) && !decode_reads_stored(weight);
}

inline PackedLayoutSwitch q4k_packed_tail_switch() {
    static const PackedLayoutSwitch value =
        packed_layout_switch("GGML_HRX_Q4K_PACKED_TAIL", "q4_k batches off the 256 grid");
    return value;
}

inline bool q4k_packed_tail_enabled(const Value & weight) {
    return packed_layout_wanted(q4k_packed_tail_switch(), weight);
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

// engine #2137: the 256-aligned Q4_K prefill. Its f16 kernel reads only the packed rows; the 8-bit-block kernel
// (ggml_mul_mat_q5_k_iq4_xs_q8_1_x4_wmma_token256) reads Q4_K stored as well (weight format 4). For a weight that
// already has a stored device copy, that kernel shares it instead of adding a packed one. GGML_HRX_Q4K_PACKED_TAIL=0
// (stored for every weight) selects it for every Q4_K weight, which is also how the path is reached by a test.
inline bool q4k_aligned_prefill_reads_stored(const Value & weight) {
    if (weight.type != GGML_TYPE_Q4_K || weight.alias_source.value >= 0) {
        return false;
    }
    const PackedLayoutSwitch value = q4k_packed_tail_switch();
    if (value == PackedLayoutSwitch::Stored) {
        return true;
    }
    return value == PackedLayoutSwitch::FollowWeight && weight_in_file_mapping(weight) && decode_reads_stored(weight);
}

// The 6-to-16-token Q6_K kernel reads an i8 layout of its own. That is an extra device copy of a weight in the file
// mapping whichever of the other two layouts the weight is read in elsewhere, so the kernel is left out for such a
// weight unless the stored layout is forced for every weight (the previous behaviour).
inline bool q6k_i8_layout_is_extra_copy(const Value & weight, int64_t input_size, int64_t output_size) {
    if (!q6k_packed_rows_weight_ok(weight, input_size, output_size)) {
        return false;
    }
    return q6k_packed_rows_enabled(weight) || (weight_in_file_mapping(weight) && decode_reads_stored(weight));
}

}  // namespace ggml::hrx
