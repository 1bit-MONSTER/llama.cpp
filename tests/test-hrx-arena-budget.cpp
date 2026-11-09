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

// Host test for the transient arena's share of the pool budget (engine#1711): the pure decision, no device, no HRX
// header, no model. The property that matters is that the arena is charged what it HOLDS, not what it asks for each
// launch: a launch that inherits a buffer at least as large as it needs must be charged nothing, or the accounting
// double-counts a buffer that is already paid for and refuses launches that fit.

#include "runtime/transient-arena-budget.h"

#include <cstdio>

static int failures = 0;

#define REQUIRE(condition)                                                       \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

int main() {
    using ggml::hrx::plan_transient_arena_charge;

    {
        // first launch: no arena yet, so the whole request is charged
        const auto p = plan_transient_arena_charge(0, 1024, 4096);
        REQUIRE(p.allocate);
        REQUIRE(!p.release_first);
        REQUIRE(p.release_bytes == 0);
        REQUIRE(p.charge_bytes == 1024);
        REQUIRE(!p.refuse);
    }
    {
        // the no-double-count rule: a launch that fits inside what the arena already holds is charged NOTHING
        const auto p = plan_transient_arena_charge(4096, 1024, 4096);
        REQUIRE(!p.allocate);
        REQUIRE(p.charge_bytes == 0);
        REQUIRE(!p.refuse);
        const auto equal = plan_transient_arena_charge(1024, 1024, 4096);
        REQUIRE(!equal.allocate);
        REQUIRE(equal.charge_bytes == 0);
    }
    {
        // growing: the old capacity goes back to the pool and the new size is charged -- net, the delta
        const auto p = plan_transient_arena_charge(1024, 4096, 8192);
        REQUIRE(p.allocate);
        REQUIRE(p.release_first);
        REQUIRE(p.release_bytes == 1024);
        REQUIRE(p.charge_bytes == 4096);
        REQUIRE(!p.refuse);
    }
    {
        // a request the budget cannot take is a refusal, and it is a refusal BEFORE the allocator is reached
        const auto p = plan_transient_arena_charge(1024, 8192, 4096);
        REQUIRE(p.allocate);
        REQUIRE(p.refuse);
        REQUIRE(p.charge_bytes == 8192);
    }
    {
        // an unset budget (0) is unbounded, as pool_reserve() treats it -- the guardrail is opt-in
        const auto p = plan_transient_arena_charge(0, 1u << 30, 0);
        REQUIRE(p.allocate);
        REQUIRE(!p.refuse);
    }
    {
        // zero-size requests never allocate and never charge, whatever the budget
        const auto p = plan_transient_arena_charge(0, 0, 64);
        REQUIRE(!p.allocate);
        REQUIRE(p.charge_bytes == 0);
        REQUIRE(!p.refuse);
    }
    {
        // exactly at the budget is allowed: the guardrail refuses what does not fit, not what exactly fits
        const auto p = plan_transient_arena_charge(0, 4096, 4096);
        REQUIRE(p.allocate);
        REQUIRE(!p.refuse);
    }

    {
        // THE ORDER (engine#1711 review): reserve before allocate, and the replaced capacity released before either.
        // A first version of this change allocated first and reserved after, so the allocation that can fault ran
        // before admission -- which is the failure the change exists to close, reintroduced by ordering.
        using ggml::hrx::TransientArenaStep;
        using ggml::hrx::transient_arena_steps;
        const auto grow = transient_arena_steps(plan_transient_arena_charge(1024, 4096, 8192));
        REQUIRE(grow.size() == 3);
        REQUIRE(grow[0] == TransientArenaStep::ReleaseOld);
        REQUIRE(grow[1] == TransientArenaStep::Reserve);
        REQUIRE(grow[2] == TransientArenaStep::Allocate);
        const auto first = transient_arena_steps(plan_transient_arena_charge(0, 4096, 8192));
        REQUIRE(first.size() == 2);
        REQUIRE(first[0] == TransientArenaStep::Reserve);
        REQUIRE(first[1] == TransientArenaStep::Allocate);
        const auto none = transient_arena_steps(plan_transient_arena_charge(4096, 1024, 8192));
        REQUIRE(none.empty());
    }

    if (failures != 0) {
        std::printf("test-hrx-arena-budget: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-hrx-arena-budget: all checks passed\n");
    return 0;
}
