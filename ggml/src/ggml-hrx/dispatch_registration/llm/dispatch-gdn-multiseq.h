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

// Gated delta net ubatches of several sequences (llama-server --parallel on Qwen3.5 / Qwen3.6 / Qwen3.8).
//
// With n_seqs > 1 llama.cpp gathers each sequence's recurrent state from the cache with GET_ROWS(s_copy) into a
// contiguous [128, 128, H, n_seqs] state, runs one GATED_DELTA_NET over q/k/v [128, H, n_seq_tokens, n_seqs], and
// copies the new states back into cache rows [kv_head, kv_head + n_seqs). AMD's matcher and scan kernel take a
// sequence count (one workgroup row per sequence, state slice (seq * H + head) * 128 * 128); the matcher capped it
// at 3. Two hooks in that matcher use this header:
//
// - ggml_hrx_gdn_max_sequences(): the sequence cap, kFusedContextGdnMaxSequences (fused-context-claim.h), so the
//   matcher and the per-op claim agree;
// - ggml_hrx_gdn_per_sequence(): with GGML_HRX_GDN_PER_SEQUENCE=1, a single-snapshot ubatch of n_seqs > 1
//   sequences and at most 16 tokens per sequence runs as n_seqs single-sequence scans instead of one n_seqs-wide
//   scan. Sequence s copies its gathered state into its cache row kv_head + s, then the in-place scan
//   (the kernel the matcher already uses for one-token rollback steps) updates that row and writes the attention
//   rows of sequence s. The state bytes moved are the same as the default path (scan into the GDN output, then one
//   copy into the cache); the n_seqs-wide output state plane is not written. Off by default: the switch exists to
//   bisect a multi-sequence fault between the n_seqs-wide scan and the rest of the graph.

#include "../../dispatch/dispatch.h"
#include "../../fused-context-claim.h"
#include "../dispatch-registry.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace ggml::hrx::gdn_multiseq {

inline constexpr KernelCatalogRef kCopyF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32");
inline constexpr KernelCatalogRef kInplaceKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_inplace");
inline constexpr KernelCatalogRef kInplaceProjectionEpilogueKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_gated_delta_net_f32_wmma_head128_inplace_projection_epilogue");

inline constexpr int64_t kMaxPerSequenceTokens = 16;

inline bool per_sequence_enabled() {
    const char * value = std::getenv("GGML_HRX_GDN_PER_SEQUENCE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

}  // namespace ggml::hrx::gdn_multiseq

namespace ggml::hrx {

inline constexpr int64_t ggml_hrx_gdn_max_sequences() {
    return kFusedContextGdnMaxSequences;
}

// Appends the per-sequence dispatches and returns true for a single-snapshot multi-sequence ubatch when
// GGML_HRX_GDN_PER_SEQUENCE is set; returns false (appending nothing) otherwise. Call after the matcher has
// claimed its nodes and emitted any separate projection epilogue. `configure(dispatch, match, token_count)` sets
// the scan's shape parameters from a match (it reads match.sequence_count).
template <typename Match, typename Configure>
bool ggml_hrx_gdn_per_sequence(const Match &   match,
                               bool            projection_epilogue,
                               bool            rmsnorm_gate_fused,
                               Configure &&    configure,
                               DispatchMatch & dispatch_match) {
    using namespace gdn_multiseq;
    if (match.sequence_count < 2 || match.snapshot_count != 1 || rmsnorm_gate_fused ||
        match.token_count > kMaxPerSequenceTokens || !per_sequence_enabled()) {
        return false;
    }
    const size_t state_bytes = static_cast<size_t>(match.width * match.width * match.head_count) * sizeof(float);
    const size_t attention_sequence_bytes =
        static_cast<size_t>(match.width * match.head_count * match.token_count) * sizeof(float);
    const size_t gate_sequence_bytes = static_cast<size_t>(match.head_count * match.token_count) * sizeof(float);
    if (match.cache->nb[1] != state_bytes || match.state->byte_count < state_bytes * match.sequence_count ||
        match.cache->byte_count < state_bytes * match.sequence_count) {
        return false;
    }

    // every scan below sees exactly one sequence
    Match single          = match;
    single.sequence_count = 1;

    for (int64_t s = 0; s < match.sequence_count; ++s) {
        const size_t seq         = static_cast<size_t>(s);
        const size_t state_at    = seq * state_bytes;
        const size_t q_at        = seq * match.raw_q->nb[3];
        const size_t k_at        = seq * match.raw_k->nb[3];
        const size_t v_at        = seq * match.v->nb[3];
        const size_t q_span      = match.raw_q->byte_count - q_at;
        const size_t k_span      = match.raw_k->byte_count - k_at;
        const size_t v_span      = match.v->byte_count - v_at;

        Dispatch copy;
        copy.kernel = make_kernel_specialization(kCopyF32Kernel);
        copy.kernel.integer_parameters.emplace("element_count", static_cast<int64_t>(state_bytes / sizeof(float)));
        copy.bindings.push_back({ match.state->id, state_at, state_bytes });
        copy.bindings.push_back({ match.cache->id, state_at, state_bytes });
        dispatch_match.dispatches.push_back(std::move(copy));

        Dispatch gdn;
        gdn.kernel = make_kernel_specialization(projection_epilogue ? kInplaceProjectionEpilogueKernel : kInplaceKernel);
        configure(gdn, single, match.token_count);
        gdn.bindings.push_back({ match.raw_q->id, q_at, q_span });
        gdn.bindings.push_back({ match.raw_k->id, k_at, k_span });
        gdn.bindings.push_back({ match.v->id, v_at, v_span });
        if (projection_epilogue) {
            gdn.bindings.push_back({ match.alpha_raw->id, seq * gate_sequence_bytes, gate_sequence_bytes });
            gdn.bindings.push_back({ match.beta_raw->id, seq * gate_sequence_bytes, gate_sequence_bytes });
            gdn.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
            gdn.bindings.push_back({ match.a_scale->id, 0, match.a_scale->byte_count });
        } else {
            gdn.bindings.push_back({ match.gate->id, seq * match.gate->nb[3], gate_sequence_bytes });
            gdn.bindings.push_back({ match.beta->id, seq * match.beta->nb[3], gate_sequence_bytes });
        }
        gdn.bindings.push_back({ match.cache->id, state_at, state_bytes });
        gdn.bindings.push_back({ match.gdn_output->id, seq * attention_sequence_bytes, attention_sequence_bytes });
        dispatch_match.dispatches.push_back(std::move(gdn));
    }
    return true;
}

}  // namespace ggml::hrx
