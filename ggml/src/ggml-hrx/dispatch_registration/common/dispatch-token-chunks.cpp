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

#include "dispatch-token-chunks.h"

#include "dispatch/dispatch.h"

#include <utility>

namespace ggml::hrx {

bool append_token_chunked_dispatches(const Dispatch &                                 whole,
                                     int64_t                                          token_count,
                                     int64_t                                          max_tokens,
                                     const std::vector<TokenChunkBinding> &           bindings,
                                     const char *                                     count_parameter,
                                     const std::function<void(Dispatch &, int64_t)> & adjust,
                                     DispatchMatch &                                  dispatch_match) {
    if (token_count < 1 || max_tokens < 1) {
        dispatch_match.status.log("token-chunked dispatch: invalid request (tokens %lld, max %lld)",
                                  static_cast<long long>(token_count), static_cast<long long>(max_tokens));
        return false;
    }
    if (token_count <= max_tokens) {
        dispatch_match.dispatches.push_back(whole);
        return true;
    }
    for (const TokenChunkBinding & b : bindings) {
        if (b.binding >= whole.bindings.size() || b.row_bytes == 0 || b.last_row_bytes == 0) {
            dispatch_match.status.log("token-chunked dispatch: invalid binding %zu", b.binding);
            return false;
        }
    }
    const int64_t chunk_count = (token_count + max_tokens - 1) / max_tokens;
    const int64_t base_tokens = token_count / chunk_count;
    const int64_t extra       = token_count % chunk_count;
    int64_t       first       = 0;
    for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int64_t count = base_tokens + (chunk < extra ? 1 : 0);
        Dispatch      part  = whole;
        part.kernel.integer_parameters[count_parameter] = count;
        for (const TokenChunkBinding & b : bindings) {
            const DispatchBinding & rows   = whole.bindings[b.binding];
            const size_t            skip   = static_cast<size_t>(first) * b.row_bytes;
            const size_t            length = static_cast<size_t>(count - 1) * b.row_bytes + b.last_row_bytes;
            if (skip > rows.length || length > rows.length - skip) {
                dispatch_match.status.log("token-chunked dispatch: rows [%lld, %lld) leave binding %zu",
                                          static_cast<long long>(first), static_cast<long long>(first + count),
                                          b.binding);
                return false;
            }
            part.bindings[b.binding].offset = rows.offset + skip;
            part.bindings[b.binding].length = length;
        }
        if (adjust) {
            adjust(part, count);
        }
        dispatch_match.dispatches.push_back(std::move(part));
        first += count;
    }
    return true;
}

}  // namespace ggml::hrx
