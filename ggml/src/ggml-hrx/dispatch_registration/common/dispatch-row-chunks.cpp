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

#include "dispatch-row-chunks.h"

#include "dispatch/dispatch.h"

#include <algorithm>
#include <string>

namespace ggml::hrx {

bool append_row_chunked_dispatches(const Dispatch & whole,
                                   int64_t          token_count,
                                   int64_t          max_tokens,
                                   size_t           row_binding,
                                   size_t           row_bytes,
                                   size_t           index_binding,
                                   size_t           index_bytes,
                                   const char *     count_parameter,
                                   const char *     capacity_parameter,
                                   DispatchMatch &  dispatch_match) {
    if (token_count < 1 || max_tokens < 1 || row_binding >= whole.bindings.size() ||
        index_binding >= whole.bindings.size() || row_binding == index_binding) {
        dispatch_match.status.log("row-chunked dispatch: invalid request (tokens %lld, max %lld, bindings %zu)",
                                  static_cast<long long>(token_count), static_cast<long long>(max_tokens),
                                  whole.bindings.size());
        return false;
    }
    const DispatchBinding & rows    = whole.bindings[row_binding];
    const DispatchBinding & indices = whole.bindings[index_binding];

    const int64_t chunk_count = (token_count + max_tokens - 1) / max_tokens;
    const int64_t base_tokens = token_count / chunk_count;
    const int64_t extra       = token_count % chunk_count;
    int64_t       first       = 0;
    for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
        const int64_t count       = base_tokens + (chunk < extra ? 1 : 0);
        const size_t  row_skip    = static_cast<size_t>(first) * row_bytes;
        const size_t  index_skip  = static_cast<size_t>(first) * index_bytes;
        if (row_skip >= rows.length || index_skip >= indices.length) {
            dispatch_match.status.log("row-chunked dispatch: chunk at row %lld is outside its bindings",
                                      static_cast<long long>(first));
            return false;
        }
        Dispatch part = whole;
        part.kernel.integer_parameters[count_parameter]   = count;
        part.kernel.compile_parameters[capacity_parameter] = std::to_string(count);
        DispatchBinding & part_rows    = part.bindings[row_binding];
        DispatchBinding & part_indices = part.bindings[index_binding];
        part_rows.offset    = rows.offset + row_skip;
        part_rows.length    = std::min(static_cast<size_t>(count) * row_bytes, rows.length - row_skip);
        part_indices.offset = indices.offset + index_skip;
        part_indices.length = std::min(static_cast<size_t>(count) * index_bytes, indices.length - index_skip);
        dispatch_match.dispatches.push_back(std::move(part));
        first += count;
    }
    return true;
}

}  // namespace ggml::hrx
