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
#include <utility>
#include <vector>

namespace ggml::hrx {

// Splits a dispatch whose `token_count` rows are independent into dispatches of at most `max_tokens` rows
// (equal chunk sizes, so at most two specializations of the kernel). Each (binding, row_bytes) pair in
// `row_bindings` advances by row_bytes per row; every other binding is shared by all chunks. The integer
// parameter `count_parameter` takes each chunk's row count. With token_count <= max_tokens it appends
// `whole` unchanged.
//
// Prompt flash attention uses it above the Loom kernel's 2048 query rows: every query row reads its own
// mask row and all of K and V, so a chunk of rows gives the same result as the whole batch, and -ub 4096
// keeps attention on HRX instead of leaving the node to the CPU (engine #390).
bool append_query_chunked_dispatches(const Dispatch &                                whole,
                                     int64_t                                         token_count,
                                     int64_t                                         max_tokens,
                                     const std::vector<std::pair<size_t, size_t>> & row_bindings,
                                     const char *                                    count_parameter,
                                     DispatchMatch &                                 dispatch_match);

// The chunk size to pass as max_tokens: kernel_max, or GGML_HRX_QUERY_CHUNK_TOKENS when that is set and smaller
// (a test switch: test-backend-ops cases are shorter than the kernels' limits, so a small value runs them in
// chunks).
int64_t query_chunk_tokens(int64_t kernel_max);

}  // namespace ggml::hrx
