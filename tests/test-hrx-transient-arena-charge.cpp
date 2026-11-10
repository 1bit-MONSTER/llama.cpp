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
// What the transient arena owes the pool ledger when it grows (engine #1711). No device and no libhrx: the decision is a
// pure function (runtime/transient-arena-charge.h) so every corner can be pinned here, including the one that makes a
// naive charge under-report - diagnostic mode retires the superseded buffer instead of releasing it, so its charge has
// to stay. The arms are the four states the growth path can be in.

#include "runtime/transient-arena-charge.h"

#include <cstdio>

using ggml::hrx::ArenaChargePlan;
using ggml::hrx::plan_arena_charge;

static int failures = 0;

static void check_sizes(bool ok, const char * what, std::size_t got, std::size_t want) {
    if (!ok) {
        std::printf("FAIL %s (%zu, want %zu)\n", what, got, want);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

static void check(bool ok, const char * what, ArenaChargePlan got, ArenaChargePlan want) {
    if (!ok) {
        std::printf("FAIL %s: reserve %zu want %zu, release %zu want %zu\n", what, got.reserve_bytes, want.reserve_bytes,
                    got.release_bytes, want.release_bytes);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

int main() {
    // 1. First allocation: charge all of it, nothing to give back.
    ArenaChargePlan got = plan_arena_charge(8u << 20, 0, false);
    check(got.reserve_bytes == (8u << 20) && got.release_bytes == 0, "first allocation charges its whole size", got,
          {8u << 20, 0});

    // 2. Growth, buffer freed: charge the delta, give back the superseded allocation's charge. The net charge after this
    //    is the new size - which is what the ledger must show, not the sum of every size the arena ever had.
    got = plan_arena_charge(16u << 20, 8u << 20, false);
    check(got.reserve_bytes == (16u << 20) && got.release_bytes == (8u << 20),
          "growth charges the WHOLE new arena and releases the old", got, {16u << 20, 8u << 20});

    // 3. Growth in diagnostic mode: the old buffer is RETIRED and stays resident, so its charge stays too. This is the arm
    //    a naive "release the old capacity" version fails: it would release memory that is still on the device.
    got = plan_arena_charge(16u << 20, 8u << 20, true);
    check(got.reserve_bytes == (16u << 20) && got.release_bytes == 0,
          "diagnostic growth keeps the retired buffer charged (ledger ends at old + new)", got, {16u << 20, 0});

    // 4. No growth: nothing charged, nothing released - the common case, and the one that must not perturb the ledger.
    got = plan_arena_charge(8u << 20, 8u << 20, false);
    check(got.reserve_bytes == 0 && got.release_bytes == 0, "a request that fits charges nothing", got, {0, 0});

    // 5. A smaller request than the current capacity: same as no growth (the arena never shrinks on its own).
    got = plan_arena_charge(1u << 20, 8u << 20, true);
    check(got.reserve_bytes == 0 && got.release_bytes == 0, "a smaller request charges nothing", got, {0, 0});

    // 6. The host-weight arena reuses this same decision (`retires_old = false`): it grows straight to its budget, so
    //    the first growth charges the whole arena and a later regrow to the same budget charges nothing. Getting this
    //    wrong double-charges a cache that keeps the same arena across many weights, which is the common case.
    got = plan_arena_charge(4096u << 20, 0, false);
    check(got.reserve_bytes == (4096u << 20) && got.release_bytes == 0, "host-weight arena: first growth charges the whole arena", got,
          {4096u << 20, 0});
    got = plan_arena_charge(4096u << 20, 4096u << 20, false);
    check(got.reserve_bytes == 0 && got.release_bytes == 0, "host-weight arena: a regrow to the same budget charges nothing", got,
          {0, 0});
    // and a grow past the budget releases the superseded arena's charge in the same step (the old buffer is freed there)
    got = plan_arena_charge(8192u << 20, 4096u << 20, false);
    check(got.reserve_bytes == (8192u << 20) && got.release_bytes == (4096u << 20),
          "host-weight arena: growing past the budget charges the new arena and releases the old", got,
          {8192u << 20, 4096u << 20});

    // 9. THE INVARIANT the #270 review named, walked over a sequence rather than one step: after every growth the
    //    ledger must equal the arena's live capacity. A delta-shaped plan with a paired release passes a single-step
    //    check and fails this one on the second growth, settling at new - old.
    {
        size_t ledger = 0, capacity = 0;
        bool   ok     = true;
        for (size_t step = 0; step < 5; ++step) {
            const size_t          required = (1u << 20) << step;   // 1, 2, 4, 8, 16 MiB
            const ArenaChargePlan plan     = plan_arena_charge(required, capacity, false);
            ledger += plan.reserve_bytes - plan.release_bytes;
            capacity = required;
            if (ledger != capacity) {
                ok = false;
            }
        }
        check_sizes(ok, "ledger equals the live capacity after every growth (not new - old)", ledger, capacity);
        size_t retired_ledger = 0, retired_held = 0;
        ok = true;
        for (size_t step = 0; step < 4; ++step) {
            const size_t          required = (1u << 20) << step;
            const ArenaChargePlan plan     = plan_arena_charge(required, retired_held, true);
            retired_ledger += plan.reserve_bytes - plan.release_bytes;
            retired_held += required;                              // every superseded arena stays resident
            if (retired_ledger != retired_held) {
                ok = false;
            }
        }
        check_sizes(ok, "retired growth accumulates old + new (they are all still on the device)", retired_ledger, retired_held);
    }

    if (failures != 0) {
        std::printf("%d arm(s) failed\n", failures);
        return 1;
    }
    std::printf("all arms passed\n");
    return 0;
}
