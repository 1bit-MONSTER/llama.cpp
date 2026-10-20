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

#include "server-health.h"

#include <cstdlib>

namespace server_health {

namespace {
constexpr int64_t kDefaultStallSeconds = 30;
}

int64_t parse_stall_us(const char * seconds) {
    if (seconds == nullptr || *seconds == '\0') {
        return kDefaultStallSeconds * 1000000;
    }
    char *                   end    = nullptr;
    const unsigned long long parsed = std::strtoull(seconds, &end, 10);
    if (end == nullptr || *end != '\0') {
        return kDefaultStallSeconds * 1000000;
    }
    return static_cast<int64_t>(parsed) * 1000000;
}

int64_t stall_us_from_env() {
    static const int64_t stall_us = parse_stall_us(std::getenv("LLAMA_SERVER_HEALTH_STALL_SECS"));
    return stall_us;
}

State & state() {
    static State shared;
    return shared;
}

bool stalled(int64_t last_progress_us, int64_t now_us, int64_t stall_us, bool work_queued) {
    if (stall_us <= 0 || !work_queued || last_progress_us <= 0) {
        return false;
    }
    return now_us - last_progress_us > stall_us;
}

}  // namespace server_health
