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

#include "../dispatch-registry.h"

#include <cstdint>
#include <vector>

namespace ggml::hrx {

// The pieces of a matched qwen35 conv prefill (CONCAT(state, x^T) -> SSM_CONV -> SILU plus the
// conv-state cache copies) that the chained form needs.
struct SsmConvChunkedRequest {
    const Value *              state  = nullptr;  // [3, hidden, sequences], channel-major
    const Value *              x      = nullptr;  // [hidden, tokens, sequences], token-major
    const Value *              filter = nullptr;  // [4, hidden]
    const Value *              output = nullptr;  // [hidden, tokens, sequences]
    std::vector<const Value *> caches;            // cache slot s = conv history ending at token (tokens - s)
    int64_t                    hidden_size    = 0;
    int64_t                    token_count    = 0;
    int64_t                    sequence_count = 0;
};

// Emits the conv as rollback dispatches of at most kSsmConvRollbackMaxTokens tokens per sequence.
// Each dispatch but the last writes its final 3-row history to a transient that the next one
// reads as its state; the last one writes the real cache slots. Returns false (with a status
// message) when the request cannot be chained.
bool append_ssm_conv_rollback_chunks(const DispatchMatchContext & context,
                                     const SsmConvChunkedRequest & request,
                                     DispatchMatch &               dispatch_match);

}  // namespace ggml::hrx
