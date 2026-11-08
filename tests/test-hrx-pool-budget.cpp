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

// Host test for the pool guardrail's budget arithmetic: pure, no device, no HRX link (engine #1502).
//
// The case that motivated the change is the one measured on strixhalo: GTT total 120 GiB with 74.8 GiB in use gave a
// 29.2 GiB budget (ceiling 104 - in use), and `hsa_amd_memory_pool_allocate` then refused a 15.35 GiB buffer with
// OUT_OF_RESOURCES — the host's own memory, not the GTT counter, was the binding limit at that moment. So the budget is
// now the smaller of the GTT term and MemAvailable, which is what the engine's admission check has always used.

#include "hrx-1bit-safety.h"

#include <cstdio>
#include <cstdlib>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

namespace {

constexpr size_t GiB = size_t{ 1 } << 30;
constexpr size_t MiB = size_t{ 1 } << 20;

void clear_env() {
    unsetenv("GGML_HRX_POOL_LIMIT_MB");
    unsetenv("GGML_HRX_GTT_RESERVE_MB");
}

}  // namespace

int main() {
    // --- the #1502 measurement, replayed as arithmetic -----------------------------------------------------------
    {
        clear_env();
        // GTT total 120 GiB, 74.8 GiB in use -> the guardrail's own line said "pool budget 29.2 GiB" with the default
        // 16 GiB reserve. 120 - 16 = 104 ceiling; 104 - 74.8 = 29.2.
        const size_t total = 120 * GiB;
        const size_t free  = total - (size_t) (74.8 * (double) GiB);

        const size_t gtt_only = ggml::hrx::onebit::pool_budget(total, free, 0);
        REQUIRE(gtt_only / MiB == (size_t) (29.2 * 1024.0) || gtt_only / MiB == (size_t) (29.2 * 1024.0) - 1 ||
                gtt_only / MiB == (size_t) (29.2 * 1024.0) + 1);

        // With the host at ~15 GiB available the allocator could not serve 15.35 GiB: the budget must now say so.
        const size_t host_tight = 15 * GiB;
        const size_t budget     = ggml::hrx::onebit::pool_budget(total, free, host_tight);
        REQUIRE(budget == host_tight);
        REQUIRE(budget < (size_t) (15.35 * (double) GiB));  // the refused request no longer looks admissible
        REQUIRE(gtt_only > (size_t) (15.35 * (double) GiB));  // ...while the old arithmetic said it was fine
    }

    // --- no behaviour change when the host has plenty of room ----------------------------------------------------
    {
        clear_env();
        const size_t total = 120 * GiB;
        const size_t free  = total - (size_t) (74.8 * (double) GiB);
        REQUIRE(ggml::hrx::onebit::pool_budget(total, free, 80 * GiB) ==
                ggml::hrx::onebit::pool_budget(total, free, 0));
    }

    // --- a host figure larger than the GTT term never inflates the budget ----------------------------------------
    {
        clear_env();
        const size_t total = 120 * GiB;
        const size_t free  = total - (size_t) (74.8 * (double) GiB);
        REQUIRE(ggml::hrx::onebit::pool_budget(total, free, 900 * GiB) ==
                ggml::hrx::onebit::pool_budget(total, free, 0));
    }

    // --- unknown host memory (0) keeps the old arithmetic, and unknown device (0) stays unfettered ----------------
    {
        clear_env();
        REQUIRE(ggml::hrx::onebit::pool_budget(0, 0, 15 * GiB) == 0);
    }

    // --- the operator override still wins outright over both terms -----------------------------------------------
    {
        clear_env();
        setenv("GGML_HRX_POOL_LIMIT_MB", "2048", 1);
        REQUIRE(ggml::hrx::onebit::pool_budget(120 * GiB, 45 * GiB, 15 * GiB) == 2048 * MiB);
        clear_env();
    }

    // --- the reserve knob still moves the GTT term -----------------------------------------------------------------
    {
        clear_env();
        setenv("GGML_HRX_GTT_RESERVE_MB", "0", 1);
        // ceiling = 120 - 0 = 120; in use 74.8 -> 45.2 GiB of GTT term, and the host term is larger, so the GTT term wins.
        const size_t budget = ggml::hrx::onebit::pool_budget(120 * GiB, 120 * GiB - (size_t) (74.8 * (double) GiB), 80 * GiB);
        REQUIRE(budget == (size_t) (45.2 * (double) GiB) || budget == (size_t) (45.2 * (double) GiB) - 1 ||
                budget == (size_t) (45.2 * (double) GiB) + 1);
        clear_env();
    }

    std::printf("test-hrx-pool-budget: OK\n");
    return 0;
}
