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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ggml::hrx {

// One binding of a token-chunked dispatch: chunk rows [first, first + count) bind
//   offset = whole.offset + first * row_bytes,  length = (count - 1) * row_bytes + last_row_bytes
// so a strided view gets exactly the extent its rows reach (the last row ends last_row_bytes after its start).
struct TokenChunkBinding {
    size_t binding;
    size_t row_bytes;
    size_t last_row_bytes;
};

// Splits a dispatch whose `token_count` rows are independent into ceil(token_count / max_tokens) chunks of
// equal size (they differ by at most one row, so at most two kernel specializations). Each chunk binds its rows
// of every TokenChunkBinding (other bindings are shared), sets the integer parameter `count_parameter` to its
// row count, and is passed to `adjust` (may be empty) for compile parameters that depend on the row count.
// Fails (with a status line) when a chunk's extent would leave its binding. With token_count <= max_tokens it
// appends `whole` unchanged.
//
// ROPE uses it above the Loom rope kernel's 2048-token range, so a 4096-token ubatch keeps ROPE on HRX.
bool append_token_chunked_dispatches(const Dispatch &                                 whole,
                                     int64_t                                          token_count,
                                     int64_t                                          max_tokens,
                                     const std::vector<TokenChunkBinding> &           bindings,
                                     const char *                                     count_parameter,
                                     const std::function<void(Dispatch &, int64_t)> & adjust,
                                     DispatchMatch &                                  dispatch_match);

}  // namespace ggml::hrx
