// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// MtpBudgetTrace's per-sequence state (common/onebit-mtp-budget.h), engine #2433: the MTP drafter calls
// keep()/drafted()/end() for every drafting sequence. A merge dropped its begin() call and every draft
// wrote past an empty vector, so the first MTP request killed llama-server. This checks that state is
// sized for any sequence the drafter reaches, with or without begin(), and that begin() still resets it.

#include "onebit-mtp-budget.h"

#include <cmath>
#include <cstdio>

static int n_fail = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++n_fail;                                                  \
        }                                                              \
    } while (0)

static bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

int main() {
    // no begin(): the shape of the #2433 drafter. Sequence 3 of 4, three draft depths.
    {
        onebit::MtpBudgetTrace b;
        CHECK(b.keep(3, 0.9f, 0.0f));
        b.drafted(3, 1, 0.9f, 11);
        CHECK(b.keep(3, 0.8f, 0.0f));
        b.drafted(3, 2, 0.8f, 12);
        CHECK(near(b.budget_cum(3), 0.72f));
        CHECK(near(b.budget_cum(0), 1.0f));   // sized but untouched
        b.end(4);
        b.accepted(3, 2);
    }

    // begin() sizes and resets: the product starts again at 1 for every sequence.
    {
        onebit::MtpBudgetTrace b;
        b.begin(2);
        CHECK(b.keep(1, 0.5f, 0.0f));
        CHECK(near(b.budget_cum(1), 0.5f));
        b.begin(2);
        CHECK(near(b.budget_cum(1), 1.0f));
        // a sequence past what begin() sized still works (n_seq grew between steps)
        CHECK(b.keep(5, 0.5f, 0.0f));
        CHECK(near(b.budget_cum(5), 0.5f));
        b.end(6);
    }

    // the rule itself: keep while the running product stays above the threshold, off at <= 0.
    {
        onebit::MtpBudgetTrace b;
        b.begin(1);
        CHECK(b.keep(0, 0.9f, 0.6f));    // 0.9
        CHECK(b.keep(0, 0.8f, 0.6f));    // 0.72
        CHECK(!b.keep(0, 0.8f, 0.6f));   // 0.576 <= 0.6: stop
        CHECK(onebit::mtp_keep_drafting(0.0f, 0.0f));
        CHECK(onebit::mtp_keep_drafting(0.0f, -1.0f));
    }

    if (n_fail == 0) {
        std::printf("test-onebit-mtp-budget: ok\n");
    }
    return n_fail == 0 ? 0 : 1;
}
