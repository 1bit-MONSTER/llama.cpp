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

// A blocking hrx_stream_wait goes to ROCr's hsa_signal_wait_scacquire on the stream timeline. The
// timeline advances with every completed command (about 900 per decode token of a 27B model), and
// each advance returns the wait from the KFD event ioctl without sleeping, so the waiting thread
// runs a full CPU core (measured: 35% user + 65% system, 30 context switches in 6 s) for the whole
// token. On a dense 27B that is ~10 W of package power and the difference between a 74 C and a
// 90-93 C APU during decode.
//
// When a wait is expected to be long (the shortest of the last four waits at this call site is over
// 20 ms, so one long prefill wait does not make the next decode token oversleep), sleep through 80%
// of it and leave the rest to the normal wait. Shorter waits are left as they were: sleeping
// through 3-10 ms tokens cost small models 2-5% of decode (the host work between tokens then runs
// on a core that has clocked down), and they were never the thermal problem.
//
// ONEBIT_HRX_BLOCKING_WAIT=1 turns the sleep off.

#include "hrx-sleeping-wait.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sys/prctl.h>

namespace ggml::hrx {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kSleepAboveNs = 20000000;

bool blocking_wait() {
    static const bool blocking = [] {
        const char * value = std::getenv("ONEBIT_HRX_BLOCKING_WAIT");
        return value != nullptr && value[0] == '1';
    }();
    return blocking;
}

void sleep_ns(int64_t ns) {
    // Timer slack defaults to 50 us per thread; keep the wake-up close to the budget.
    thread_local bool slack_set = [] {
        prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);
        return true;
    }();
    (void) slack_set;
    const timespec interval = { static_cast<time_t>(ns / 1000000000), static_cast<long>(ns % 1000000000) };
    nanosleep(&interval, nullptr);
}

// HYPERLOOM_HRX_SLICED_WAIT=1: sleep the same budget, but in short slices with a completion query between
// them. The budget comes from the shortest of the last four waits at the call site, and those can all be
// long prefill waits (llama-bench: pp512 reps, or the 16k-token depth prefill before every tg rep), so the
// first decode tokens after a prefill would otherwise sleep ~80% of a 1-2 s prefill wait each, decaying by
// x0.8 per token. Slicing bounds any oversleep to one slice; in steady-state decode the budget ends before
// the stream completes, so the extra queries change nothing there.
bool sliced_wait() {
    static const bool sliced = [] {
        const char * value = std::getenv("HYPERLOOM_HRX_SLICED_WAIT");
        return value != nullptr && value[0] == '1';
    }();
    return sliced;
}

constexpr int64_t kSleepSliceNs = 1000000;

int64_t expected_ns(const WaitHistory & history) {
    if (history.count == 0) {
        return 0;
    }
    const uint32_t n = std::min<uint32_t>(history.count, history.duration_ns.size());
    return *std::min_element(history.duration_ns.begin(), history.duration_ns.begin() + n);
}

}  // namespace

hrx_status_t stream_wait_sleeping(hrx_stream_t stream, WaitHistory & history) {
    if (blocking_wait() || stream == nullptr) {
        return hrx_stream_wait(stream);
    }
    const auto    start    = Clock::now();
    const int64_t expected = expected_ns(history);
    if (expected > kSleepAboveNs) {
        bool         complete = false;
        hrx_status_t status   = hrx_stream_query(stream, &complete);
        if (!hrx_status_is_ok(status)) {
            return status;
        }
        if (!complete) {
            if (sliced_wait()) {
                const auto deadline = start + std::chrono::nanoseconds(expected * 4 / 5);
                while (!complete) {
                    const int64_t remaining =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - Clock::now()).count();
                    if (remaining <= 0) {
                        break;
                    }
                    sleep_ns(std::min(remaining, kSleepSliceNs));
                    status = hrx_stream_query(stream, &complete);
                    if (!hrx_status_is_ok(status)) {
                        return status;
                    }
                }
            } else {
                sleep_ns(expected * 4 / 5);
            }
        }
    }
    hrx_status_t status = hrx_stream_wait(stream);
    if (hrx_status_is_ok(status)) {
        history.duration_ns[history.count % history.duration_ns.size()] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
        history.count++;
    }
    return status;
}

hrx_status_t stream_synchronize_sleeping(hrx_stream_t stream, WaitHistory & history) {
    hrx_status_t status = hrx_stream_flush(stream);
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    return stream_wait_sleeping(stream, history);
}

}  // namespace ggml::hrx
