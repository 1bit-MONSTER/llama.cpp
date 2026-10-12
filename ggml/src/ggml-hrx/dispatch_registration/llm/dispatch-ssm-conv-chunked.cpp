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

#include "dispatch-ssm-conv-chunked.h"

#include "dispatch/dispatch.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"
#include "ubatch-token-limits.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

constexpr KernelCatalogRef kSsmConvRollbackChunkKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_rollback_f32");
constexpr KernelCatalogRef kSsmConvSnapshotChunkKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_snapshot_window_tail_f32");
constexpr KernelCatalogRef kSsmConvPrefill512ChunkKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_ssm_conv_dconv4_silu_prefill_512_wg1024");

// the tuned 512-token pair (snapshot + prefill_512) takes one sequence, one cache slot and these widths
constexpr int64_t kPrefill512Tokens    = 512;
constexpr int64_t kPrefill512MinHidden = 8192;
constexpr int64_t kPrefill512MaxHidden = 10240;
// rows of the snapshot window the prefill_512 kernel reads: 3 carried rows + 61 x rows
constexpr int64_t kPrefill512SnapshotRows = 64;

// llm.ssm_conv.rollback.cache_count range
constexpr size_t kRollbackCacheSlots = 5;

void set_parameter(KernelSpecialization & kernel, const char * name, int64_t value) {
    kernel.compile_parameters.emplace(name, std::to_string(value));
}

// Every 512-token chunk runs the tuned pair AMD uses for a 512-token ubatch: the snapshot kernel copies the
// carried state and the first 61 x rows to a window snapshot and writes the chunk's last 3 x rows as the new
// conv history, then prefill_512 convolves. The history ping-pongs between two transients and the last chunk
// writes the real cache.
bool append_prefill512_chunks(const DispatchMatchContext & context,
                              const SsmConvChunkedRequest & request,
                              DispatchMatch &               dispatch_match) {
    const int64_t hidden       = request.hidden_size;
    const int64_t chunk_count  = request.token_count / kPrefill512Tokens;
    const size_t  row_bytes    = static_cast<size_t>(hidden) * sizeof(float);
    const size_t  state_bytes  = 3 * row_bytes;
    const size_t  chunk_bytes  = static_cast<size_t>(kPrefill512Tokens) * row_bytes;
    const size_t  window_bytes = static_cast<size_t>(kPrefill512SnapshotRows) * row_bytes;

    const int32_t first_value = context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size());
    const ValueId snapshot(first_value);
    const ValueId history[2] = { ValueId(first_value + 1), ValueId(first_value + 2) };
    dispatch_match.transients.push_back({ snapshot, "llm.ssm_conv.chunk_window_snapshot", window_bytes, 256 });
    dispatch_match.transients.push_back({ history[0], "llm.ssm_conv.chunk_history_a", state_bytes, 256 });
    dispatch_match.transients.push_back({ history[1], "llm.ssm_conv.chunk_history_b", state_bytes, 256 });

    const Value * cache = request.caches.front();
    for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const bool   first    = chunk == 0;
        const bool   last     = chunk + 1 == chunk_count;
        const size_t x_offset = static_cast<size_t>(chunk) * chunk_bytes;

        Dispatch window;
        window.kernel = make_kernel_specialization(kSsmConvSnapshotChunkKernel);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.d_conv", 4);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.d_inner", hidden);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.n_t", kPrefill512Tokens);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.n_s", 1);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.state_row_stride", 1);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.state_channel_stride", 3);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.x_row_stride", hidden);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.dst_row_stride", hidden);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.cache_row_stride", 1);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.cache_channel_stride", 3);
        set_parameter(window.kernel, "llm.ssm_conv.snapshot.workgroup_size", 256);
        if (first) {
            window.bindings.push_back({ request.state->id, 0, state_bytes });
        } else {
            window.bindings.push_back({ history[(chunk - 1) % 2], 0, state_bytes });
        }
        window.bindings.push_back({ request.x->id, x_offset, chunk_bytes });
        window.bindings.push_back({ snapshot, 0, window_bytes });
        if (last) {
            window.bindings.push_back({ cache->id, 0, state_bytes });
        } else {
            window.bindings.push_back({ history[chunk % 2], 0, state_bytes });
        }

        Dispatch conv;
        conv.kernel = make_kernel_specialization(kSsmConvPrefill512ChunkKernel);
        set_parameter(conv.kernel, "llm.ssm_conv.prefill.d_conv", 4);
        set_parameter(conv.kernel, "llm.ssm_conv.prefill.d_inner", hidden);
        set_parameter(conv.kernel, "llm.ssm_conv.prefill.n_t", kPrefill512Tokens);
        set_parameter(conv.kernel, "llm.ssm_conv.prefill.n_s", 1);
        set_parameter(conv.kernel, "llm.ssm_conv.prefill.state_row_stride", hidden);
        set_parameter(conv.kernel, "llm.ssm_conv.prefill.x_row_stride", hidden);
        conv.bindings.push_back({ snapshot, 0, window_bytes });
        conv.bindings.push_back({ request.x->id, x_offset, chunk_bytes });
        conv.bindings.push_back({ request.filter->id, 0, request.filter->byte_count });
        conv.bindings.push_back({ request.output->id, x_offset, chunk_bytes });

        dispatch_match.dispatches.push_back(std::move(window));
        dispatch_match.dispatches.push_back(std::move(conv));
    }
    return true;
}

}  // namespace

bool append_ssm_conv_rollback_chunks(const DispatchMatchContext & context,
                                     const SsmConvChunkedRequest & request,
                                     DispatchMatch &               dispatch_match) {
    const int64_t hidden = request.hidden_size;
    const int64_t tokens = request.token_count;
    if (request.state == nullptr || request.x == nullptr || request.filter == nullptr || request.output == nullptr ||
        request.caches.empty() || request.caches.size() > kRollbackCacheSlots || hidden < 32 || hidden > 65536 ||
        hidden % 32 != 0 || tokens <= kSsmConvRollbackMaxTokens || tokens > kUbatchMaxTokens ||
        request.sequence_count < 1) {
        dispatch_match.status.log("chained SSM_CONV: unsupported request (hidden %lld, tokens %lld, sequences %lld, "
                                  "cache slots %zu)",
                                  static_cast<long long>(hidden), static_cast<long long>(tokens),
                                  static_cast<long long>(request.sequence_count), request.caches.size());
        return false;
    }

    if (request.sequence_count == 1 && request.caches.size() == 1 && tokens % kPrefill512Tokens == 0 &&
        hidden >= kPrefill512MinHidden && hidden <= kPrefill512MaxHidden) {
        return append_prefill512_chunks(context, request, dispatch_match);
    }

    // equal chunks (at most two sizes, so at most two kernel specializations per ubatch length);
    // every chunk has more than 256 tokens, so the last one always covers every cache slot
    const int64_t chunk_count = (tokens + kSsmConvRollbackMaxTokens - 1) / kSsmConvRollbackMaxTokens;
    const int64_t base_tokens = tokens / chunk_count;
    const int64_t extra       = tokens % chunk_count;

    const size_t row_bytes   = static_cast<size_t>(hidden) * sizeof(float);
    const size_t state_bytes = 3 * row_bytes;

    // ping-pong conv histories between consecutive chunks
    const ValueId history[2] = {
        ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size())),
        ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) + 1),
    };
    dispatch_match.transients.push_back({ history[0], "llm.ssm_conv.chunk_history_a", state_bytes, 256 });
    dispatch_match.transients.push_back({ history[1], "llm.ssm_conv.chunk_history_b", state_bytes, 256 });

    for (int64_t sequence = 0; sequence < request.sequence_count; ++sequence) {
        const size_t state_offset = static_cast<size_t>(sequence) * state_bytes;
        const size_t x_base       = static_cast<size_t>(sequence * tokens) * row_bytes;
        int64_t      start        = 0;
        for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
            const int64_t chunk_tokens = base_tokens + (chunk < extra ? 1 : 0);
            const bool    first        = chunk == 0;
            const bool    last         = chunk + 1 == chunk_count;

            Dispatch dispatch;
            dispatch.kernel = make_kernel_specialization(kSsmConvRollbackChunkKernel);
            set_parameter(dispatch.kernel, "llm.ssm_conv.rollback.d_inner", hidden);
            set_parameter(dispatch.kernel, "llm.ssm_conv.rollback.n_t", chunk_tokens);
            set_parameter(dispatch.kernel, "llm.ssm_conv.rollback.n_s", 1);
            set_parameter(dispatch.kernel, "llm.ssm_conv.rollback.cache_count",
                          last ? static_cast<int64_t>(request.caches.size()) : 1);
            set_parameter(dispatch.kernel, "llm.ssm_conv.rollback.workgroup_size", 256);

            const size_t x_offset = x_base + static_cast<size_t>(start) * row_bytes;
            const size_t x_bytes  = static_cast<size_t>(chunk_tokens) * row_bytes;
            if (first) {
                dispatch.bindings.push_back({ request.state->id, state_offset, state_bytes });
            } else {
                dispatch.bindings.push_back({ history[(chunk - 1) % 2], 0, state_bytes });
            }
            dispatch.bindings.push_back({ request.x->id, x_offset, x_bytes });
            dispatch.bindings.push_back({ request.filter->id, 0, request.filter->byte_count });
            dispatch.bindings.push_back({ request.output->id, x_offset, x_bytes });
            for (size_t slot = 0; slot < kRollbackCacheSlots; ++slot) {
                if (last) {
                    const Value * cache = request.caches[std::min(slot, request.caches.size() - 1)];
                    dispatch.bindings.push_back({ cache->id, state_offset, state_bytes });
                } else {
                    dispatch.bindings.push_back({ history[chunk % 2], 0, state_bytes });
                }
            }
            dispatch_match.dispatches.push_back(std::move(dispatch));
            start += chunk_tokens;
        }
    }
    return true;
}

}  // namespace ggml::hrx
