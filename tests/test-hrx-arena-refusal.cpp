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
//
// The host-weight arena refusal has to report the reserve outcome that actually failed (engine #900). Every arm below
// is a number measured in that issue, or a corner the predicate must not over-claim on. No device and no libhrx: the
// message helper is the only thing under test, which is why it lives apart from HostWeightCache.

#include "runtime/host-memory-refusal.h"

#include <cstdio>
#include <cstring>
#include <string>

using ggml::hrx::ArenaMiss;
using ggml::hrx::describe;
using ggml::hrx::fragmented;

static constexpr size_t MiB = 1024 * 1024;

static int failures = 0;

static void expect(bool condition, const char * arm, const char * what) {
    if (!condition) {
        std::printf("FAIL %s: %s\n", arm, what);
        ++failures;
    }
}

static void expect_contains(const std::string & text, const char * needle, const char * arm) {
    if (text.find(needle) == std::string::npos) {
        std::printf("FAIL %s: %s not in \"%s\"\n", arm, needle, text.c_str());
        ++failures;
    }
}

static void expect_absent(const std::string & text, const char * needle, const char * arm) {
    if (text.find(needle) != std::string::npos) {
        std::printf("FAIL %s: %s must not be in \"%s\"\n", arm, needle, text.c_str());
        ++failures;
    }
}

int main() {
    // The issue's own smoking gun (Qwen3-0.6B-Q4_K_M, budget 200 MiB): 193.2 MiB free in 5 blocks, a 121.7 MiB slot.
    {
        ArenaMiss miss;
        miss.need          = (size_t) (121.7 * MiB);
        miss.free_bytes    = (size_t) (193.2 * MiB);
        miss.free_blocks   = 5;
        miss.largest_block = (size_t) (60.4 * MiB);
        miss.capacity      = 200 * MiB;
        miss.limit         = 200 * MiB;
        const std::string text = describe(miss);
        expect(fragmented(miss), "budget-200", "free >= need with no block that large is fragmentation");
        expect_contains(text, "fragmented", "budget-200");
        expect_contains(text, "193.2 MiB free in 5 blocks", "budget-200");
        expect_contains(text, "largest 60.4 MiB", "budget-200");
        expect_absent(text, "Raise it to at least", "budget-200");
        expect(text.back() != '\n', "budget-200", "one sentence, no trailing newline");
    }

    // The issue's contradiction on a real model (76 GiB pack, budget 4096 MiB): "Raise it to at least 4030 MiB" while
    // 4096 MiB was in force. That refusal was a fragmented miss, so it must now carry no raise-to number at all.
    {
        ArenaMiss miss;
        miss.need          = 497 * MiB;
        miss.free_bytes    = (size_t) (564.2 * MiB);
        miss.free_blocks   = 3;
        miss.largest_block = 400 * MiB;
        miss.resident      = 1983 * MiB;
        miss.parked        = 1549 * MiB;
        miss.capacity      = 4096 * MiB;
        miss.limit         = 4096 * MiB;
        const std::string text = describe(miss);
        expect(fragmented(miss), "budget-4096", "a 497 MiB slot against 564 MiB in 3 blocks is fragmentation");
        expect_absent(text, "Raise it to at least", "budget-4096");
        expect_contains(text, "limit 4096.0 MiB", "budget-4096");
    }

    // A genuine capacity miss keeps the engine #788 wording, and its number is still the arithmetic it claims.
    {
        ArenaMiss miss;
        miss.need          = 500 * MiB;
        miss.free_bytes    = 40 * MiB;
        miss.free_blocks   = 2;
        miss.largest_block = 40 * MiB;
        miss.resident      = 1983 * MiB;
        miss.parked        = 1549 * MiB;
        miss.capacity      = 3572 * MiB;
        miss.limit         = 4096 * MiB;
        const std::string text = describe(miss);
        expect(!fragmented(miss), "capacity", "free < need is a capacity miss");
        expect_contains(text, "cannot hold this model's per-execution working set", "capacity");
        expect_contains(text, "Raise it to at least 4032 MiB", "capacity");  // 500 + 1983 + 1549
    }

    // Unbounded (the default) has no limit to raise, and must not divide by it or invent one.
    {
        ArenaMiss miss;
        miss.need          = 64 * MiB;
        miss.free_bytes    = 8 * MiB;
        miss.free_blocks   = 1;
        miss.largest_block = 8 * MiB;
        miss.capacity      = 72 * MiB;
        miss.limit         = 0;
        const std::string text = describe(miss);
        expect(!fragmented(miss), "unbounded", "not fragmented");
        expect_contains(text, "limit 0.0 MiB", "unbounded");
    }

    // The corner the predicate must not over-claim on: a block large enough exists, so the take failed for another
    // reason (a failed growth logs its own line). Calling that fragmentation would be as wrong as the old wording.
    {
        ArenaMiss miss;
        miss.need          = 64 * MiB;
        miss.free_bytes    = 128 * MiB;
        miss.free_blocks   = 1;
        miss.largest_block = 128 * MiB;
        miss.capacity      = 192 * MiB;
        miss.limit         = 192 * MiB;
        const std::string text = describe(miss);
        expect(!fragmented(miss), "corner", "a large enough free block is not fragmentation");
        expect_absent(text, "the free space is fragmented", "corner");
    }

    if (failures != 0) {
        std::printf("test-hrx-arena-refusal: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-hrx-arena-refusal: all cases ok (5 arms)\n");
    return 0;
}
