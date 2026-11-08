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
// The budget after a refusal (engine #1502). The numbers in the first arm are the issue's own: a 29.2 GiB budget admitted a
// 15.35 GiB buffer that the allocator refused, and the guardrail then knew nothing it did not know before. No device, no libhrx:
// the helper is a pure function, which is why it lives beside pool_budget().

#include "hrx-1bit-safety.h"

#include <cstdio>

static constexpr size_t GiB = 1024ull * 1024 * 1024;
static constexpr size_t MiB = 1024 * 1024;

static int failures = 0;

static void expect_true(const char * what, bool cond) {
    if (cond) { printf("  PASS  %s\n", what); return; }
    printf("  FAIL  %s\n", what);
    failures++;
}

static void expect_eq(const char * what, size_t got, size_t want) {
    if (got == want) {
        printf("  PASS  %-58s %zu\n", what, got);
        return;
    }
    printf("  FAIL  %-58s got %zu, want %zu\n", what, got, want);
    failures++;
}

int main() {
    using ggml::hrx::onebit::pool_budget_after_refusal;

    // The issue's own case: the guardrail admitted 15.35 GiB out of 29.2 GiB and the pool refused it.
    const size_t budget   = 29 * GiB + 205 * MiB;   // 29.2 GiB
    const size_t refused  = 15 * GiB + 359 * MiB;   // 15.35 GiB
    const size_t after    = pool_budget_after_refusal(budget, refused);

    expect_eq("29.2 GiB budget after a refused 15.35 GiB is size-1, not budget-size", after, refused - 1);
    expect_true("the refused size no longer fits the new budget", after < refused);
    expect_true("a 4 GiB request still fits the new budget", 4 * GiB <= after);

    // A budget of 0 means "device size unknown, unfettered" and must not become a limit.
    expect_eq("budget 0 stays 0 (unfettered)", pool_budget_after_refusal(0, 15 * GiB), 0);

    // Repeated refusals of the *same* size must not compound: the evidence is one-bound, so the budget settles at size-1.
    size_t b = 20 * GiB;
    for (int i = 0; i < 8; i++) {
        b = pool_budget_after_refusal(b, 5 * GiB);
    }
    expect_eq("eight 5 GiB refusals against a 20 GiB budget settle at size-1, no compounding", b, 5 * GiB - 1);
    expect_true("the repeatedly refused size still does not fit", b < 5 * GiB);

    // A refusal larger than the budget cannot *raise* it, and a budget already below size-1 is left alone.
    expect_eq("refusal of 40 GiB against a 29.2 GiB budget leaves the budget unchanged", pool_budget_after_refusal(budget, 40 * GiB), budget);
    expect_eq("budget already below size-1 is not raised", pool_budget_after_refusal(2 * GiB, 8 * GiB), 2 * GiB);
    expect_eq("refusal of 8 GiB against an 8 GiB budget leaves 8 GiB - 1", pool_budget_after_refusal(8 * GiB, 8 * GiB), 8 * GiB - 1);

    // The arm the review of #251 asked for: a cap is evidence about the pool at one instant, so it must be refutable. These four
    // arms are the reviewer's own numbers - GTT in use swinging between 35.8 GiB of a 68.2 GiB budget and 93.9 GiB of a 0.5 GiB one
    // inside a session, and a 4.7 GiB model refused at 13.6 / 13.5 / 10.8 GiB in three consecutive attempts.
    using ggml::hrx::onebit::pool_budget_refresh;

    const size_t cap      = refused - 1;      // 15.34 GiB, from the refusal above
    const size_t cap_free = 45 * GiB;         // what the pool showed when it refused
    const size_t uncapped = 44 * GiB;         // what pool_budget() would say now (free - 16 GiB reserve)

    expect_eq("no cap in force: refresh returns the uncapped budget", pool_budget_refresh(uncapped, 0, 0, 10 * GiB), uncapped);
    expect_eq("pool unchanged since the refusal: the cap holds", pool_budget_refresh(uncapped, cap, cap_free, cap_free), cap);
    expect_eq("pool SMALLER since the refusal: the cap still holds",
              pool_budget_refresh(uncapped, cap, cap_free, 20 * GiB), cap);
    expect_eq("pool GREW since the refusal: the cap is lifted and the refused size is admissible again",
              pool_budget_refresh(uncapped, cap, cap_free, cap_free + 1), uncapped);
    // The four constants the reviewer cited, as one sequence: refused at 13.6 GiB of budget, then the pool frees.
    const size_t b_small = 136 * MiB;         // 13.6 GiB budget
    const size_t cap4    = pool_budget_after_refusal(b_small, 4 * GiB + 700 * MiB);   // the 4.7 GiB model refused
    expect_true("a 4.7 GiB model refused at a 13.6 GiB budget is capped below its size", cap4 < 4 * GiB + 700 * MiB);
    expect_true("when the pool frees the same model is admissible again",
                pool_budget_refresh(68 * GiB, cap4, 20 * GiB, 60 * GiB) > 4 * GiB + 700 * MiB);

    printf(failures == 0 ? "test-hrx-pool-refusal-budget: all requirements passed\n"
                         : "test-hrx-pool-refusal-budget: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
