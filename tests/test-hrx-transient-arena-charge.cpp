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
    check(got.reserve_bytes == (8u << 20) && got.release_bytes == (8u << 20), "growth charges the delta and releases the old", got,
          {8u << 20, 8u << 20});

    // 3. Growth in diagnostic mode: the old buffer is RETIRED and stays resident, so its charge stays too. This is the arm
    //    a naive "release the old capacity" version fails: it would release memory that is still on the device.
    got = plan_arena_charge(16u << 20, 8u << 20, true);
    check(got.reserve_bytes == (8u << 20) && got.release_bytes == 0, "diagnostic growth keeps the retired buffer charged", got,
          {8u << 20, 0});

    // 4. No growth: nothing charged, nothing released - the common case, and the one that must not perturb the ledger.
    got = plan_arena_charge(8u << 20, 8u << 20, false);
    check(got.reserve_bytes == 0 && got.release_bytes == 0, "a request that fits charges nothing", got, {0, 0});

    // 5. A smaller request than the current capacity: same as no growth (the arena never shrinks on its own).
    got = plan_arena_charge(1u << 20, 8u << 20, true);
    check(got.reserve_bytes == 0 && got.release_bytes == 0, "a smaller request charges nothing", got, {0, 0});

    if (failures != 0) {
        std::printf("%d arm(s) failed\n", failures);
        return 1;
    }
    std::printf("all arms passed\n");
    return 0;
}
