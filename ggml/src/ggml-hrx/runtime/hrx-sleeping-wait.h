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

#include "hrx_runtime.h"

#include <array>
#include <cstdint>

namespace ggml::hrx {

// Durations of the last few waits at one call site; see hrx-sleeping-wait.cpp.
struct WaitHistory {
    std::array<int64_t, 4> duration_ns{};
    uint32_t               count = 0;
};

// hrx_stream_wait that sleeps through most of a long expected wait instead of spending it in the
// HSA signal wait, which keeps a CPU core busy. See hrx-sleeping-wait.cpp.
hrx_status_t stream_wait_sleeping(hrx_stream_t stream, WaitHistory & history);

// This thread's WaitHistory for the work identified by `key` (a replayed graph), so waits on
// different work do not share one estimate.
WaitHistory & wait_history_for(const void * key);

// hrx_stream_synchronize (flush, then wait) with the sleeping wait.
hrx_status_t stream_synchronize_sleeping(hrx_stream_t stream, WaitHistory & history);

}  // namespace ggml::hrx
