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

// Host test for host-memory-available.h: the MemAvailable the unbounded-residency warning compares against
// (engine #669). Pure: no device, no HRX header, and the /proc body is supplied by the test.

#include "runtime/host-memory-available.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

using ggml::hrx::host_memory_available_bytes;
using ggml::hrx::mem_available_from_meminfo;

static const size_t MiB = size_t{ 1 } << 20;

int main() {
    // A real /proc/meminfo shape: MemAvailable is neither the first line nor the last.
    const char * body =
        "MemTotal:       128140180 kB\n"
        "MemFree:          901888 kB\n"
        "MemAvailable:   55939156 kB\n"
        "Buffers:           12345 kB\n"
        "Cached:          9876543 kB\n";
    REQUIRE(mem_available_from_meminfo(body) == size_t{ 55939156 } * 1024);

    // Whitespace between the colon and the number is a single field width in meminfo; the parse must not depend on it.
    REQUIRE(mem_available_from_meminfo("MemAvailable: 2048 kB\n") == 2048 * 1024);
    REQUIRE(mem_available_from_meminfo("MemAvailable:\t2048 kB\n") == 2048 * 1024);

    // No field, no answer - and a lookalike key must not answer for it.
    REQUIRE(mem_available_from_meminfo("MemTotal: 1024 kB\nMemFree: 512 kB\n") == 0);
    REQUIRE(mem_available_from_meminfo("MemAvailableX: 4096 kB\n") == 0);
    REQUIRE(mem_available_from_meminfo("MemAvailable: 4096 kB\nMemFree: 1 kB\n") == 4096 * 1024);

    // Junk, an empty value, an absent terminator, and an impossible size are "unknown", never a guess.
    REQUIRE(mem_available_from_meminfo("MemAvailable: banana kB\n") == 0);
    REQUIRE(mem_available_from_meminfo("MemAvailable:\n") == 0);
    REQUIRE(mem_available_from_meminfo("MemAvailable: 4096 kB") == 4096 * 1024);
    REQUIRE(mem_available_from_meminfo("MemAvailable: 18446744073709551615 kB\n") == 0);
    REQUIRE(mem_available_from_meminfo("MemAvailable: -2048 kB\n") == 0);
    REQUIRE(mem_available_from_meminfo(nullptr) == 0);
    REQUIRE(mem_available_from_meminfo("") == 0);

    // The live reader on this host: whoever calls it must get either a plausible number or 0 ("unknown", stay quiet).
    const size_t live = host_memory_available_bytes();
    std::printf("host_memory_available_bytes() = %zu bytes (%.1f MiB)\n", live, double(live) / double(MiB));
    FILE * meminfo = std::fopen("/proc/meminfo", "rb");
    if (meminfo != nullptr) {
        std::fclose(meminfo);
        REQUIRE(live > 0);
        REQUIRE(live < size_t{ 1 } << 50);
    }

    std::printf("test-hrx-host-memory-available: all requirements passed\n");
    return 0;
}
