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

    printf(failures == 0 ? "test-hrx-pool-refusal-budget: all requirements passed\n"
                         : "test-hrx-pool-refusal-budget: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
