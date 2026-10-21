// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Cumulative-confidence budget for MTP drafting (opt-in, our own file).
//
// The upstream knob (--spec-draft-p-min) is a per-token threshold on the drafter's top-1 probability.
// The measurement behind this file (27B UD-Q4_K_XL + its Q4_0 MTP head, HRX, greedy; 149 verified steps
// at draft length 4, 330 at 3) shows that probability barely decays with depth (0.894 -> 0.846 from
// depth 1 to 4) while the conditional acceptance does (0.90 -> 0.83 -> 0.85 -> 0.83), so a per-token
// threshold either passes everything or nothing. What does discriminate is the running product of those
// probabilities - the quantity DeepInfer's scheduler uses (process_MTPInfo: accept_rates[j] =
// prod_{t<=j} c_t) - so the rule here keeps drafting while that product stays above a threshold and
// stops before the draft that would push it below.
//
// MtpBudgetTrace carries the per-sequence state of that rule, and the optional measurement log
// (ONEBIT_MTP_TRACE=<file>) that priced every drafting policy: per decode step it writes the drafter's
// top-1 probability and the running product at every draft depth, the target's accepted count, and
// timestamps at the start and end of drafting and at the end of verification. It is opened only when the
// environment variable is set, nothing is read back, and it never changes a decision.
//
// The rule is off by default: a threshold <= 0 keeps every draft the caller's own limits allow, which is
// exactly the historical behaviour.

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace onebit {

// Keep drafting while the running product of the drafter's top-1 probabilities exceeds the threshold.
// p_cum: the product including the candidate just sampled. threshold <= 0 disables the rule.
inline bool mtp_keep_drafting(float p_cum, float threshold) {
    if (!(threshold > 0.0f)) {
        return true;
    }
    return p_cum > threshold;
}

// Per-sequence state for the rule above plus the measurement log. Not thread-safe; the draft-mtp
// implementation calls it from one thread, one step at a time.
class MtpBudgetTrace {
  public:
    // Starts a drafting step: clears the per-sequence state and logs "B".
    void begin(uint32_t n_seq) {
        ++step_;
        budget_cum_.assign(n_seq, 1.0f);
        trace_cum_.assign(n_seq, 1.0f);
        drafted_.assign(n_seq, 0);
        if (FILE * f = file()) {
            std::fprintf(f, "B t=%.3f step=%llu\n", now_ms(), (unsigned long long) step_);
            std::fflush(f);
        }
    }

    // The candidate just sampled: folds it into the running product and reports whether to draft it.
    // false means the running product has fallen to or below the threshold, so drafting stops here.
    bool keep(int32_t seq, float p1, float threshold) {
        const size_t i = static_cast<size_t>(seq);
        budget_cum_[i] *= p1;
        return mtp_keep_drafting(budget_cum_[i], threshold);
    }

    // The candidate was added to the draft: remember the depth and log "D" (depth, p1, running product).
    void drafted(int32_t seq, size_t n_drafted, float p1, int32_t token) {
        const size_t i = static_cast<size_t>(seq);
        drafted_[i] = n_drafted;
        trace_cum_[i] *= p1;
        if (FILE * f = file()) {
            std::fprintf(f, "D t=%.3f step=%llu seq=%d depth=%zu p1=%.6f cumprod=%.6f id=%d\n", now_ms(),
                         (unsigned long long) step_, (int) seq, n_drafted, (double) p1, (double) trace_cum_[i],
                         (int) token);
            std::fflush(f);
        }
    }

    // Drafting ended for this step: logs "E" with the drafted count of every sequence.
    void end(uint32_t n_seq) {
        if (FILE * f = file()) {
            for (uint32_t seq = 0; seq < n_seq; ++seq) {
                std::fprintf(f, "E t=%.3f step=%llu seq=%u drafted=%zu\n", now_ms(), (unsigned long long) step_, seq,
                             drafted_[seq]);
            }
            std::fflush(f);
        }
    }

    // The target finished verifying this step: logs "A" with the accepted count.
    void accepted(int32_t seq, uint32_t n_accepted) {
        if (FILE * f = file()) {
            std::fprintf(f, "A t=%.3f step=%llu seq=%d n_accepted=%u\n", now_ms(), (unsigned long long) step_, (int) seq,
                         n_accepted);
            std::fflush(f);
        }
    }

  private:
    static FILE * file() {
        static FILE * f = []() -> FILE * {
            const char * p = std::getenv("ONEBIT_MTP_TRACE");
            return (p != nullptr && *p != '\0') ? std::fopen(p, "a") : nullptr;
        }();
        return f;
    }

    static double now_ms() {
        using clock = std::chrono::steady_clock;
        static const clock::time_point origin = clock::now();
        return std::chrono::duration<double, std::milli>(clock::now() - origin).count();
    }

    uint64_t            step_ = 0;
    std::vector<float>  budget_cum_;
    std::vector<float>  trace_cum_;
    std::vector<size_t> drafted_;
};

}  // namespace onebit
