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

// Host test for server-health.h: the stall decision behind llama-server's /health. No server, no
// model: the decision and the environment parser are pure.

#include "server-health.h"

#include <cstdio>
#include <cstdlib>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

int main() {
    const int64_t s = 1000000;   // one second in microseconds
    const int64_t stall = 30 * s;

    // no work queued: never stalled, however long since the last progress (idle server)
    REQUIRE(!server_health::stalled(1 * s, 100 * s, stall, false));
    REQUIRE(!server_health::stalled(0, 100 * s, stall, false));

    // work queued and progress recent: healthy
    REQUIRE(!server_health::stalled(100 * s, 110 * s, stall, true));

    // work queued and no progress for longer than the threshold: stalled
    REQUIRE(server_health::stalled(100 * s, 131 * s, stall, true));
    REQUIRE(!server_health::stalled(100 * s, 130 * s, stall, true));  // exactly the threshold is not yet a stall

    // a server that has never noted progress is not judged (it just started)
    REQUIRE(!server_health::stalled(0, 1000 * s, stall, true));

    // 0 disables the check entirely
    REQUIRE(!server_health::stalled(1 * s, 1000 * s, 0, true));

    // the parser: seconds in, microseconds out, 30 s for anything unusable
    REQUIRE(server_health::parse_stall_us(nullptr) == 30 * s);
    REQUIRE(server_health::parse_stall_us("") == 30 * s);
    REQUIRE(server_health::parse_stall_us("0") == 0);
    REQUIRE(server_health::parse_stall_us("5") == 5 * s);
    REQUIRE(server_health::parse_stall_us("120") == 120 * s);
    REQUIRE(server_health::parse_stall_us("soon") == 30 * s);
    REQUIRE(server_health::parse_stall_us("5s") == 30 * s);

    // the environment reader is memoised; whatever the process has, it agrees with the parser
    REQUIRE(server_health::stall_us_from_env() == server_health::stall_us_from_env());

    std::printf("test-server-health: all checks passed\n");
    return 0;
}
