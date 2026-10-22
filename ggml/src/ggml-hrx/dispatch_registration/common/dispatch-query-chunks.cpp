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

#include "dispatch-query-chunks.h"

#include "dispatch/dispatch.h"

#include <algorithm>
#include <cstdlib>

namespace ggml::hrx {

int64_t query_chunk_tokens(int64_t kernel_max) {
    static const int64_t limit = [] {
        const char * env = std::getenv("GGML_HRX_QUERY_CHUNK_TOKENS");
        return env == nullptr || *env == '\0' ? int64_t{ 0 } : static_cast<int64_t>(std::atoll(env));
    }();
    return limit > 0 && limit < kernel_max ? limit : kernel_max;
}

bool append_query_chunked_dispatches(const Dispatch &                                whole,
                                     int64_t                                         token_count,
                                     int64_t                                         max_tokens,
                                     const std::vector<std::pair<size_t, size_t>> & row_bindings,
                                     const char *                                    count_parameter,
                                     DispatchMatch &                                 dispatch_match) {
    if (token_count < 1 || max_tokens < 1) {
        dispatch_match.status.log("query-chunked dispatch: invalid request (tokens %lld, max %lld)",
                                  static_cast<long long>(token_count), static_cast<long long>(max_tokens));
        return false;
    }
    if (token_count <= max_tokens) {
        dispatch_match.dispatches.push_back(whole);
        return true;
    }
    for (const auto & [binding, row_bytes] : row_bindings) {
        if (binding >= whole.bindings.size() || row_bytes == 0) {
            dispatch_match.status.log("query-chunked dispatch: invalid row binding %zu", binding);
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
        for (const auto & [binding, row_bytes] : row_bindings) {
            const DispatchBinding & rows = whole.bindings[binding];
            const size_t            skip = static_cast<size_t>(first) * row_bytes;
            if (skip >= rows.length) {
                dispatch_match.status.log("query-chunked dispatch: chunk at row %lld is outside binding %zu",
                                          static_cast<long long>(first), binding);
                return false;
            }
            DispatchBinding & part_rows = part.bindings[binding];
            part_rows.offset            = rows.offset + skip;
            part_rows.length            = std::min(static_cast<size_t>(count) * row_bytes, rows.length - skip);
        }
        dispatch_match.dispatches.push_back(std::move(part));
        first += count;
    }
    return true;
}

}  // namespace ggml::hrx
