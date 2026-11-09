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

    // The arms the review of #251 asked for: drive the state machine the device actually runs (note_refusal -> observe), not only the
    // pure cap. The numbers are the reviewer's own, and the first one is the doubling they measured.
    using ggml::hrx::onebit::pool_budget_observe;
    using ggml::hrx::onebit::pool_budget_note_refusal;
    using ggml::hrx::onebit::pool_budget_state_t;

    const size_t total   = 120 * GiB;
    const size_t our_mem = 15 * GiB;        // this process's own model: in the counters AND in the ledger

    // Registration: nothing held by us yet, another process holds 20 GiB.
    pool_budget_state_t st;
    st.budget = 84 * GiB;                   // 104 ceiling - 20 held by the other process

    // A refusal while we hold 15 GiB ourselves: cap at size-1, and record the free figure the counters showed.
    const size_t free_at_refusal = total - 20 * GiB - our_mem;      // 85 GiB
    st = pool_budget_note_refusal(st, 80 * GiB, free_at_refusal);
    expect_eq("cap after refusing 80 GiB is size-1", st.cap, 80 * GiB - 1);

    // The other process now holds 1 GiB LESS than at the refusal, so the cap is lifted.
    st = pool_budget_observe(total, free_at_refusal + GiB, our_mem, 0, st);
    expect_eq("lifting the cap restores the registration budget (own memory counted once)", st.budget, 85 * GiB);
    expect_eq("the cap is cleared when it is lifted", st.cap, 0);

    // And the defect the review named: a pool that SHRINKS must not clear the cap, or the low reading freezes forever.
    pool_budget_state_t st2;
    st2.budget = 30 * GiB;
    st2 = pool_budget_note_refusal(st2, 40 * GiB, total - 45 * GiB);          // refusal seen with 75 GiB free
    st2 = pool_budget_observe(total, total - 90 * GiB, our_mem, 0, st2);         // other process grew by 45 GiB
    expect_eq("a shrinking pool keeps the cap (it is not cleared)", st2.cap, 30 * GiB);
    expect_eq("a shrinking pool keeps the cap's evidence", st2.cap_free, total - 45 * GiB);
    expect_true("a shrinking pool never ends up above the cap", st2.budget <= st2.cap);

    // The reviewer's 4.7 GiB sequence, with the typo they caught fixed: 13.6 GiB, not 136 MiB.
    const size_t b_small = 13 * GiB + 614 * MiB;                              // 13.6 GiB
    pool_budget_state_t st3;
    st3.budget = b_small;
    st3 = pool_budget_note_refusal(st3, 4 * GiB + 700 * MiB, 20 * GiB);
    expect_true("a 4.7 GiB model refused at a 13.6 GiB budget is capped below its size", st3.cap < 4 * GiB + 700 * MiB);
    st3 = pool_budget_observe(total, 60 * GiB, 0, 0, st3);                       // the pool frees
    expect_true("when the pool frees the same model is admissible again", st3.budget > 4 * GiB + 700 * MiB);
    expect_eq("no cap in force: an observation changes nothing", pool_budget_observe(total, 10 * GiB, 0, 0, pool_budget_state_t{20 * GiB, 0, 0}).budget, 20 * GiB);

    // ---- the host term (this lane's half of #1502) ------------------------------------------------------------------
    //
    // On an APU the GTT ceiling and DRAM are one pool, so the allocator is bounded by the host as well as by the GTT counter.
    // The issue's measured case: budget 29.2 GiB from the counters, while /proc/meminfo had roughly 15 GiB left - and the pool
    // then refused 15.35 GiB, failing at model load instead of at admission.
    using ggml::hrx::onebit::pool_budget;
    const size_t issue_budget = 29 * GiB + 205 * MiB;                        // 29.2 GiB, as printed by the guardrail
    const size_t issue_free   = total - 74 * GiB - 819 * MiB;                // in use 74.8 GiB
    expect_eq("the GTT arithmetic reproduces the issue's 29.2 GiB budget", pool_budget(total, issue_free, 0), issue_budget);
    const size_t host_tight = 15 * GiB;
    expect_eq("a tight host replaces the GTT term", pool_budget(total, issue_free, host_tight), host_tight);
    expect_true("so the refused 15.35 GiB is no longer admissible", pool_budget(total, issue_free, host_tight) < 15 * GiB + 359 * MiB);
    expect_true("while the GTT-only arithmetic still called it admissible", pool_budget(total, issue_free, 0) > 15 * GiB + 359 * MiB);
    expect_eq("a roomy host leaves the GTT term alone (no behaviour change)", pool_budget(total, issue_free, 80 * GiB), issue_budget);
    expect_eq("a host figure larger than the GTT term never inflates the budget", pool_budget(total, issue_free, 900 * GiB), issue_budget);
    expect_eq("unknown host (0) keeps the old arithmetic", pool_budget(total, issue_free, 0), pool_budget(total, issue_free, 0));
    expect_eq("unknown device stays unfettered", pool_budget(0, 0, 15 * GiB), 0);

    // The state machine must stay honest under a tight host: a cap can only lower the budget, and the host term bounds the
    // recomputation that lifts a cap, so a refuted cap cannot restore an optimistic figure while the host is still full.
    pool_budget_state_t st4;
    st4.budget = issue_budget;
    st4 = pool_budget_note_refusal(st4, 4 * GiB, issue_free);
    st4 = pool_budget_observe(total, issue_free + 10 * GiB, 0, host_tight, st4);
    expect_true("a lifted cap is still bounded by the host term", st4.budget <= host_tight);
    expect_eq("and the cap is cleared, because the GTT pool did grow", st4.cap, 0);

    // engine #1417: the live check is now the DEFAULT, so the decision it makes is worth pinning like the cap above. The
    // numbers are the report's own: a 120 GiB pool, a box that had 74.8 GiB in use at device open and ~90 GiB when the
    // 15.35 GiB allocation was attempted.
    using ggml::hrx::onebit::pool_live_fits;

    const size_t gtt = 120 * GiB;
    const size_t margin_256 = 256 * MiB;

    expect_true("a 15.35 GiB buffer fits 45 GiB free with the default margin",
                pool_live_fits(refused, gtt, gtt - 45 * GiB, margin_256));
    // The case the live check exists for, built from the report's own numbers: the open-time budget was
    // 104 - 74.8 = 29.2 GiB, so a 20 GiB buffer is ADMITTED by the snapshot. If a neighbour then grew the box to
    // 105 GiB in use (15 GiB free), the snapshot still admits it and the failure lands in the allocator; the live
    // check refuses it at admission, which is the whole of engine #1417's ask.
    const size_t snapshot_budget = gtt - 16 * GiB - 748 * GiB / 10;      // 29.2 GiB, printed in the report
    const size_t stale_request   = 20 * GiB;
    expect_true("the open-time snapshot admits a 20 GiB buffer (29.2 GiB budget)",
                stale_request <= snapshot_budget);
    expect_true("a neighbour growing the box to 105 GiB in use makes the live check refuse that buffer",
                !pool_live_fits(stale_request, gtt, 105 * GiB, margin_256));
    expect_true("with the neighbour gone the same buffer passes again",
                pool_live_fits(stale_request, gtt, gtt - 45 * GiB, margin_256));
    expect_true("the margin is required headroom, not decoration: 16 GiB free refuses a 16 GiB request",
                !pool_live_fits(16 * GiB, gtt, gtt - 16 * GiB, margin_256));
    expect_true("the same request passes with the margin set to 0",
                pool_live_fits(16 * GiB, gtt, gtt - 16 * GiB, 0));
    expect_true("used > total (a transient counter artefact) is free == 0, not an underflow",
                !pool_live_fits(1 * MiB, gtt, gtt + 8 * GiB, 0));
    expect_true("used == total refuses anything larger than the margin",
                !pool_live_fits(1 * MiB, gtt, gtt, margin_256));
    expect_true("an absurd margin saturates instead of wrapping",
                !pool_live_fits(1 * MiB, gtt, 0, SIZE_MAX));
    expect_true("a zero-size request with room passes", pool_live_fits(0, gtt, gtt - 1 * GiB, margin_256));

    printf(failures == 0 ? "test-hrx-pool-refusal-budget: all requirements passed\n"
                         : "test-hrx-pool-refusal-budget: %d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
