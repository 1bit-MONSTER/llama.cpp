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

namespace ggml::hrx {

// Splits a row-wise dispatch, whose `token_count` rows are processed independently, into dispatches of at
// most `max_tokens` rows each (equal chunk sizes, so at most two kernel specializations). The bindings at
// `row_binding` and `index_binding` advance by `row_bytes` / `index_bytes` per row; every other binding is
// shared. The integer parameter `count_parameter` and the compile parameter `capacity_parameter` take each
// chunk's row count. Used where a kernel's token range is shorter than the ubatch (engine #389).
bool append_row_chunked_dispatches(const Dispatch & whole,
                                   int64_t          token_count,
                                   int64_t          max_tokens,
                                   size_t           row_binding,
                                   size_t           row_bytes,
                                   size_t           index_binding,
                                   size_t           index_bytes,
                                   const char *     count_parameter,
                                   const char *     capacity_parameter,
                                   DispatchMatch &  dispatch_match);

}  // namespace ggml::hrx
