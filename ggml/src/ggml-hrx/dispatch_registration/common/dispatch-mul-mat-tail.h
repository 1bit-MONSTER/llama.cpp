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

#include "dispatch-mul-mat-common.h"

namespace ggml::hrx {

// The q8_1 x4 prefill kernel takes token counts that are multiples of 256. For
// a prompt chunk with a remainder, it runs the 256-aligned head and this
// appends one generic F32 WMMA matmul for tokens [head_tokens, token_count):
// input and output are token-major, so the tail is the same buffers bound at
// the head's byte offset. Returns false (appending nothing) for an empty head
// or a tail shorter than 2 tokens.
bool common_append_mul_mat_token_tail(const CommonMulMatMatch & match,
                                      int64_t                   head_tokens,
                                      DispatchMatch &           dispatch_match);

}  // namespace ggml::hrx
