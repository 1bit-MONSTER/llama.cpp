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

// llm.ssm_conv.rollback.cache_count range
constexpr size_t kRollbackCacheSlots = 5;

void set_parameter(KernelSpecialization & kernel, const char * name, int64_t value) {
    kernel.compile_parameters.emplace(name, std::to_string(value));
}

}  // namespace

bool append_ssm_conv_rollback_chunks(const DispatchMatchContext & context,
                                     const SsmConvChunkedRequest & request,
                                     DispatchMatch &               dispatch_match) {
    const int64_t hidden = request.hidden_size;
    const int64_t tokens = request.token_count;
    if (request.state == nullptr || request.x == nullptr || request.filter == nullptr || request.output == nullptr ||
        request.caches.empty() || request.caches.size() > kRollbackCacheSlots || hidden < 32 || hidden > 65536 ||
        hidden % 32 != 0 || tokens <= kSsmConvRollbackMaxTokens || tokens > kLinearAttentionMaxTokens ||
        request.sequence_count < 1) {
        dispatch_match.status.log("chained SSM_CONV: unsupported request (hidden %lld, tokens %lld, sequences %lld, "
                                  "cache slots %zu)",
                                  static_cast<long long>(hidden), static_cast<long long>(tokens),
                                  static_cast<long long>(request.sequence_count), request.caches.size());
        return false;
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
