// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// test-hrx-gtt-reserve -- engine#2986: the GTT reserve is a CAP, not a constant.
//
// A flat 16 GiB reserve left a 16 GB machine (about 8 GiB of GTT by default) and a 32 GB one (about 16 GiB) with a
// ceiling of zero, so ggml-hrx refused the device at registration for every model -- and those are the machines the
// owner's direction is aimed at. The reserve is now min(16 GiB, 12.5 % of the GTT total), capped at the 16 GiB the #680
// measurement established on a 120 GiB GTT.
//
// No device and no HRX runtime: this is the arithmetic of one function, and the arms are the four pools that matter --
// a 16 GB machine, a 32 GB machine, this box, and one large enough to hit the cap -- plus the two escapes (an explicit
// knob, and a caller that cannot measure the total).
#include "ggml-hrx/hrx-1bit-safety.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;
constexpr size_t kGiB = 1ull << 30;

void check(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        failures++;
    }
}

}  // namespace

int main() {
    using ggml::hrx::onebit::pool_reserve_bytes;

    // The pools that exist in the field, and what each must keep free.
    check(pool_reserve_bytes(8 * kGiB) == 1 * kGiB, "an 8 GiB pool (a 16 GB machine) keeps 1 GiB free");
    check(pool_reserve_bytes(16 * kGiB) == 2 * kGiB, "a 16 GiB pool (a 32 GB machine) keeps 2 GiB free");
    check(pool_reserve_bytes(120 * kGiB) == 15 * kGiB, "this box's 120 GiB pool keeps 15 GiB free");
    // The cap: 12.5 % of 200 GiB would be 25 GiB, so the measured value wins instead of scaling past it.
    check(pool_reserve_bytes(200 * kGiB) == 16 * kGiB, "a 200 GiB pool is capped at the measured 16 GiB");
    check(pool_reserve_bytes(1024 * kGiB) == 16 * kGiB, "and stays capped at any larger total");
    // The escapes, in the order they win.
    check(pool_reserve_bytes(0) == 16 * kGiB, "an unknown total keeps the flat 16 GiB (never a smaller reserve)");
    setenv("GGML_HRX_GTT_RESERVE_MB", "512", 1);
    check(pool_reserve_bytes(8 * kGiB) == (512ull << 20), "GGML_HRX_GTT_RESERVE_MB overrides the scaling");
    unsetenv("GGML_HRX_GTT_RESERVE_MB");

    // A pool that cannot hold the scaled reserve outright: the ceiling would be zero, and that is the caller's
    // business (pool_budget_terms reports it) -- what matters here is that the reserve itself never exceeds the pool.
    check(pool_reserve_bytes(1 * kGiB) <= 1 * kGiB, "a 1 GiB pool does not get a reserve larger than itself");

    std::printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
