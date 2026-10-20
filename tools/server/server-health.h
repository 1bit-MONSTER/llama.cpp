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

// A stall-aware /health for llama-server. Today /health answers 200 whenever the process is up, so a
// request whose backend call never returns (a wedged GPU queue, a driver that stopped retiring
// work) looks healthy forever and a supervisor never restarts it. The main loop notes its progress;
// /health returns 503 when work is queued and that progress is older than the threshold.
//
// LLAMA_SERVER_HEALTH_STALL_SECS: seconds without main-loop progress before /health says 503
// (default 30; 0 disables the check, the old behaviour).

#include <atomic>
#include <cstdint>

namespace server_health {

int64_t stall_us_from_env();

// Seconds in, microseconds out: 0 disables the check, anything unusable is the 30 s default.
int64_t parse_stall_us(const char * seconds);

// True only when work is queued and the main loop has not noted progress for longer than stall_us.
// A stall_us of 0 disables it, and a server that has not noted any progress yet is never stalled.
bool stalled(int64_t last_progress_us, int64_t now_us, int64_t stall_us, bool work_queued);

// The state itself: one llama-server per process serves one model, and /health must stay answerable
// while the server context is being torn down or rebuilt for sleeping, so it cannot live in the
// context the HTTP handler is told not to touch.
struct State {
    std::atomic<int64_t> last_progress_us{0};
    std::atomic<bool>    work_queued{false};
};

State & state();

}  // namespace server_health
