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

// Upload-time repack of exact-ternary Q4_0 weights into the packed ternary layout (dispatch/ternary-q4-0.h).
#pragma once

#include "dispatch/ternary-q4-0.h"
#include "runtime/host-memory.h"
#include "status.h"

#include <cstdint>
#include <vector>

namespace ggml::hrx {

// Verifies that every Q4_0 nibble is 7, 8 or 9 and that the four blocks of each 128-value group share one
// scale, then writes the packed layout. Fails (and writes nothing) on the first block that is not ternary.
Status materialize_ternary_q4_0_k128(const HostWeightSource & source, std::vector<uint8_t> & output);

}  // namespace ggml::hrx
