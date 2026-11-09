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

#include "hrx-1bit-safety.h"

#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <mutex>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace ggml::hrx::onebit {

namespace {

std::atomic<size_t>   g_pool_allocated{ 0 };
std::atomic<bool>     g_device_poisoned{ false };
std::atomic<uint64_t> g_fault_address{ 0 };
std::atomic<uint32_t> g_fault_kind{ 0 };

constexpr size_t kMiB = 1024 * 1024;

bool read_counter(const char * path, unsigned long long & value) {
    FILE * file = std::fopen(path, "r");
    if (file == nullptr) {
        return false;
    }
    const bool ok = std::fscanf(file, "%llu", &value) == 1;
    std::fclose(file);
    return ok;
}

}  // namespace

size_t pool_reserve_bytes() {
    // The reserve is what the ceiling keeps free for the driver and for kernel/transient spikes. #680 measured a
    // gfxhub page fault and a MODE2 reset at GTT 106.5 GiB of 120, i.e. 13.5 GiB of headroom was not enough once
    // the runtime's own allocations landed; 16 GiB leaves 104 GiB as the ceiling, 2.5 GiB under the fault point.
    constexpr size_t kDefaultReserveMiB = 16384;
    const char *     env                = std::getenv("GGML_HRX_GTT_RESERVE_MB");
    if (env != nullptr) {
        const long long mb = std::atoll(env);
        if (mb >= 0) {
            return static_cast<size_t>(mb) * kMiB;
        }
    }
    return kDefaultReserveMiB * kMiB;
}

bool gtt_counters(size_t & total, size_t & used) {
    // Two reads of the driver's own counters. Cards are probed in order because the render node is not fixed across
    // installs; the first card that answers wins, and a box with no amdgpu simply has no guardrail input.
    for (int card = 0; card < 8; ++card) {
        char total_path[128];
        char used_path[128];
        std::snprintf(total_path, sizeof(total_path), "/sys/class/drm/card%d/device/mem_info_gtt_total", card);
        std::snprintf(used_path, sizeof(used_path), "/sys/class/drm/card%d/device/mem_info_gtt_used", card);
        unsigned long long total_bytes = 0;
        unsigned long long used_bytes  = 0;
        if (read_counter(total_path, total_bytes) && read_counter(used_path, used_bytes)) {
            total = static_cast<size_t>(total_bytes);
            used  = static_cast<size_t>(used_bytes);
            return true;
        }
    }
    return false;
}

size_t pool_budget(size_t memory_total, size_t memory_free, size_t memory_available) {
    const char * env = std::getenv("GGML_HRX_POOL_LIMIT_MB");
    if (env != nullptr) {
        const long long mb = std::atoll(env);
        if (mb > 0) {
            return static_cast<size_t>(mb) * kMiB;
        }
    }
    if (memory_total == 0) {
        return 0;  // unknown device size: keep the unfettered behaviour
    }
    const size_t reserve = pool_reserve_bytes();
    const size_t ceiling = memory_total > reserve ? memory_total - reserve : 0;
    const size_t in_use  = memory_total > memory_free ? memory_total - memory_free : 0;
    const size_t by_gtt  = ceiling > in_use ? ceiling - in_use : 0;
    // engine #1502: the GTT counter is not the allocator's ceiling on an APU. Letting a 15.35 GiB buffer through a 29.2 GiB
    // budget while the host had ~15 GiB left made the failure land inside hsa_amd_memory_pool_allocate and read as
    // "llama_model_load: error loading model" (two Hyperloom measurements scored 0.0 that way). The engine's admission check has
    // always used min(GTT free, MemAvailable) (app/mem_admit.h); this is the same two-term model in the fork's safety net.
    if (memory_available != 0 && memory_available < by_gtt) {
        return memory_available;
    }
    return by_gtt;
}

// engine #1417: the budget above is a snapshot taken at device open, and the ledger is per-process, so another process
// growing its model after we opened is invisible to it - the check passes and the allocation reaches a device that is at its
// ceiling, where it fails inside the allocator instead of at admission. A live consult of the driver's own counters closes
// that window, and it is ON BY DEFAULT (engine #1417's report was a 74.8 GiB reading at device open against ~90 GiB when the
// allocation happened; no fixed threshold on those numbers can separate the two).
//
// The trade the old default made is now the operator's opt-OUT rather than opt-in: admission depends on what other processes
// are doing, so a launch that fits an idle box can be refused on a busy one. GGML_HRX_POOL_LIVE_CHECK=0 restores the
// open-time snapshot exactly.
//
// GGML_HRX_POOL_LIVE_MARGIN_MB leaves headroom for growth that happens between the cached counters and the allocation
// (default 256 MiB - irrelevant on a quiet box's 120 GiB pool, enough to absorb the cache window and the rounding in the
// driver's counters).
static bool pool_live_check_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_POOL_LIVE_CHECK");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        // Fail safe: only an explicit 0/false/no turns the guard off. std::atoi would read "true", "yes" or a typo as 0 and
        // silently remove the check, which is the wrong direction for a guardrail.
        const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(env[0])));
        return !(c == '0' || c == 'f' || c == 'n');
    }();
    return enabled;
}

// Two sysfs reads per allocation is fine for a model load and wasteful in a long run, so the live reading is cached for a
// short window: the guardrail only has to be right about the box, not to the millisecond, and a stale-by-50 ms figure can only
// under-refuse by that window's worth of growth.
static bool pool_live_counters(size_t & total, size_t & used) {
    static std::mutex      mutex;
    static size_t          cached_total = 0;
    static size_t          cached_used  = 0;
    static std::chrono::steady_clock::time_point cached_at{};
    static bool            valid        = false;

    std::lock_guard<std::mutex> lock(mutex);
    const auto                  now = std::chrono::steady_clock::now();
    if (!valid || now - cached_at >= std::chrono::milliseconds(50)) {
        size_t t = 0;
        size_t u = 0;
        if (!gtt_counters(t, u)) {
            valid = false;
            return false;
        }
        cached_total = t;
        cached_used  = u;
        cached_at    = now;
        valid        = true;
    }
    total = cached_total;
    used  = cached_used;
    return true;
}

static size_t pool_live_margin_bytes() {
    static const size_t margin = [] {
        const char * env = std::getenv("GGML_HRX_POOL_LIVE_MARGIN_MB");
        if (env == nullptr || env[0] == '\0') {
            return static_cast<size_t>(256) * kMiB;
        }
        const long long mb = std::atoll(env);
        return mb > 0 ? static_cast<size_t>(mb) * 1024u * 1024u : static_cast<size_t>(0);
    }();
    return margin;
}

size_t pool_budget_after_refusal(size_t budget, size_t size) {
    if (budget == 0) {
        return 0;  // unknown device size: keep the unfettered behaviour
    }
    // Evidence-bounded, not deficit-estimated: the one thing this refusal proves is that *this size* did not fit, so the
    // largest size still admissible is size-1. Subtracting the whole refused size would compound across retries - four
    // failures of 5 GiB against a 20 GiB budget would wedge the process at 0 - while proving no more than size-1 does.
    const size_t next = std::min(budget, size > 0 ? size - 1 : 0);
    GGML_LOG_WARN("%s: the allocator refused %.1f MiB that the guardrail had admitted (budget %.1f MiB); the budget is now "
                  "%.1f MiB, so the next launch of that size is refused at admission instead of failing at model load "
                  "(engine #1502; a later observation of a larger free pool lifts it)\n",
                  __func__, size / (double) kMiB, budget / (double) kMiB, next / (double) kMiB);
    return next;
}

size_t pool_allocated() {
    return g_pool_allocated.load();
}

pool_budget_state_t pool_budget_note_refusal(pool_budget_state_t state, size_t size, size_t free_now) {
    state.budget   = pool_budget_after_refusal(state.budget, size);
    state.cap      = state.budget;
    state.cap_free = free_now;
    return state;
}

pool_budget_state_t pool_budget_observe(size_t memory_total, size_t free_bytes, size_t allocated, size_t memory_available,
                                        pool_budget_state_t state) {
    if (state.cap == 0) {
        return state;  // no cap in force: the registration-time budget stands
    }
    if (free_bytes > state.cap_free) {
        // The pool has grown past the figure the refusal saw, so the cap's evidence no longer holds: a transient refusal must not
        // become a permanent ceiling (engine #1502 review). Recompute from the current figures and drop the cap.
        const size_t uncapped = pool_budget(memory_total, free_bytes + allocated, memory_available);
        GGML_LOG_INFO("%s: the pool grew from %.1f MiB to %.1f MiB free since the last refusal; lifting the %.1f MiB cap to "
                      "%.1f MiB\n",
                      __func__, state.cap_free / (double) kMiB, free_bytes / (double) kMiB, state.cap / (double) kMiB,
                      uncapped / (double) kMiB);
        state.budget   = uncapped;
        state.cap      = 0;
        state.cap_free = 0;
        return state;
    }
    // The pool is no larger than it was at the refusal: keep the cap AND its evidence, and bound only this call's budget by it.
    // Clearing it here would freeze the low reading for the rest of the process (engine #1502 review).
    const size_t capped = pool_budget(memory_total, free_bytes + allocated, memory_available);
    state.budget = state.cap < capped ? state.cap : capped;
    return state;
}

// Does |size| fit the live pool with |margin| of headroom? Pure, so the test drives it: the numbers come from the driver's
// counters, and the arithmetic that matters is the underflow guard (|used| can exceed |total| transiently) and the margin
// saturation (|margin| is an operator input and may be absurd).
bool pool_live_fits(size_t size, size_t total, size_t used, size_t margin) {
    const size_t live_free = used >= total ? 0 : total - used;
    const size_t wanted    = size + (margin > SIZE_MAX - size ? SIZE_MAX - size : margin);
    return wanted <= live_free;
}

bool pool_reserve(size_t budget, size_t size, const char * what) {
    const size_t used = g_pool_allocated.load();
    if (pool_live_check_enabled()) {
        size_t total = 0;
        size_t in_use = 0;
        if (pool_live_counters(total, in_use)) {
            const size_t margin = pool_live_margin_bytes();
            if (!pool_live_fits(size, total, in_use, margin)) {
                const size_t live_free = in_use >= total ? 0 : total - in_use;
                GGML_LOG_ERROR("%s: HRX pool limit: %s of %.1f MiB does not fit the live device budget "
                               "(%.1f MiB free - %.1f MiB margin of %.1f MiB); this is the live check, on by default since "
                               "engine #1417 - set GGML_HRX_POOL_LIVE_CHECK=0 to use the open-time snapshot instead\n",
                               __func__, what, size / (double) kMiB, live_free / (double) kMiB, margin / (double) kMiB,
                               total / (double) kMiB);
                return false;
            }
        }
    }
    if (budget == 0) {
        return true;  // unknown device size: keep the unfettered behaviour
    }
    if (size > budget || used > budget - size) {
        GGML_LOG_ERROR("%s: HRX pool limit: %s of %.1f MiB does not fit (%.1f MiB of %.1f MiB already in use); "
                       "set GGML_HRX_POOL_LIMIT_MB to change the budget\n",
                       __func__, what, size / (double) kMiB, used / (double) kMiB, budget / (double) kMiB);
        return false;
    }
    g_pool_allocated.fetch_add(size);
    return true;
}

void pool_release(size_t size) {
    const size_t used = g_pool_allocated.load();
    g_pool_allocated.store(used > size ? used - size : 0);
}

size_t pool_remaining(size_t budget) {
    const size_t used = g_pool_allocated.load();
    return used >= budget ? 0 : budget - used;
}

void poison_device(uint32_t access_kind, uint64_t address) {
    g_fault_kind.store(access_kind);
    g_fault_address.store(address);
    g_device_poisoned.store(true);
}

bool device_poisoned(const char * what) {
    if (!g_device_poisoned.load(std::memory_order_relaxed)) {
        return false;
    }
    static std::atomic<bool> reported{ false };
    if (!reported.exchange(true)) {
        GGML_LOG_ERROR("%s: HRX device unusable after a fault (access kind %u, address 0x%016" PRIx64
                       "): refusing further work. Restart the process to recover.\n",
                       what, g_fault_kind.load(), g_fault_address.load());
    }
    return true;
}

void poison_selftest() {
    const char * env = std::getenv("GGML_HRX_POISON_SELFTEST");
    if (env != nullptr && std::atoi(env) != 0) {
        poison_device(7, 0xDEADBEE0ull);
    }
}

}  // namespace ggml::hrx::onebit
