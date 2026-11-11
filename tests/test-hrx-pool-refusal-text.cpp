// Copyright 2026 bong-water-water-bong. All rights reserved.
// Proprietary and confidential. Distributed only as compiled binaries under the
// 1bit Binary License (LICENSE-BINARY).
//
// A device refusal must name the resource that was actually binding (engine#2069). `budget == 0` can come from the HOST
// term -- pool_budget_terms() takes the smaller of the GTT term and MemAvailable and sets `host_bound` -- but the refusal
// used to word itself from the ceiling alone, so it told the operator to "free GTT" and printed the three GTT figures, none
// of which was binding. Measured: a self-built HRX configuration refused with "GTT total 120.0 GiB with 78.0 GiB free
// leaves nothing under the 104.0 GiB ceiling", which cannot explain anything (78 free against a 104 ceiling leaves ~62 GiB),
// while the real cause was the host figure reading as 0 (the #1942 argument-evaluation-order bug).
//
// Four arms, and the second half of each is what makes them worth having: the host-bound text must NOT advise freeing GTT,
// and the GTT text must NOT mention RAM -- a wording function can otherwise satisfy every arm by mentioning everything.
#include "runtime/failure-status.h"
#include "hrx-1bit-safety.h"

#include <cstdio>
#include <cstdlib>
#include <cstdarg>

// ggml-base is not in this link: the arms exercise the wording, and ggml_abort only appears on paths they never take.
extern "C" void ggml_abort(const char *, int, const char *, ...) { std::abort(); }
extern "C" void ggml_log_internal(enum ggml_log_level, const char *, ...) {}
#include <string>

using ggml::hrx::onebit::pool_budget_terms_t;
using ggml::hrx::pool_refusal_text;

static int fails = 0;
static void check(bool ok, const char * what, const std::string & text = std::string()) {
    if (!ok) { std::printf("FAIL %s\n     text: %s", what, text.c_str()); ++fails; }
    else std::printf("ok   %s\n", what);
}
static bool has(const std::string & h, const char * n) { return h.find(n) != std::string::npos; }

int main() {
    constexpr size_t GiB = 1024ull * 1024ull * 1024ull;
    const size_t total = 120 * GiB, free_gtt = 78 * GiB, ceiling = 104 * GiB;

    // 1. the measured case: the host term was binding (a measured zero from the #1942 argument-order bug)
    pool_budget_terms_t host0 = {};
    host0.host_available = 0; host0.host_bound = true; host0.budget = 0;
    std::string t = pool_refusal_text("HRX0", total, free_gtt, ceiling, host0);
    check(has(t, "host MemAvailable"), "host-bound refusal names the HOST term", t);
    check(has(t, "free RAM"), "and tells the operator to free RAM", t);
    check(!has(t, "free GTT or lower"), "and does NOT tell them to free GTT", t);
    check(has(t, "measured 0.0 GiB"), "a measured zero is called out as a reported zero", t);

    // 2. a real host figure: named, and still no "free GTT" advice
    pool_budget_terms_t host8 = {};
    host8.host_available = 8 * GiB; host8.host_bound = true; host8.budget = 8 * GiB;
    t = pool_refusal_text("HRX0", total, free_gtt, ceiling, host8);
    check(has(t, "8.0 GiB") && has(t, "free RAM") && !has(t, "free GTT or lower"),
          "a non-zero host term is printed with its figure and still no GTT advice", t);
    check(!has(t, "measured 0.0 GiB"), "and no zero warning when nothing was zero", t);

    // 3. GTT binding: the #680 wording survives verbatim
    pool_budget_terms_t gtt = {};
    gtt.host_available = 64 * GiB; gtt.host_bound = false; gtt.budget = 26 * GiB;
    t = pool_refusal_text("HRX0", total, free_gtt, ceiling, gtt);
    check(has(t, "free GTT or lower the reserve") && has(t, "GGML_HRX_GTT_RESERVE_MB") && has(t, "engine#680"),
          "the GTT wording (and #680's rationale) is unchanged when GTT binds", t);
    check(!has(t, "free RAM"), "and it does not mention RAM", t);

    // 4. kHostUnknown: the host term decided nothing, so the GTT wording is the right one
    pool_budget_terms_t unk = {};
    unk.host_available = ggml::hrx::onebit::kHostUnknown; unk.host_bound = false; unk.budget = 0;
    t = pool_refusal_text("HRX0", total, free_gtt, ceiling, unk);
    check(has(t, "free GTT or lower"), "an unreadable host figure leaves the GTT wording", t);

    std::printf(fails ? "%d arm(s) failed\n" : "all arms passed\n", fails);
    return fails ? 1 : 0;
}
