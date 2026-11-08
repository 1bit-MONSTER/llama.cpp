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

size_t pool_budget(size_t memory_total, size_t memory_free) {
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
    return ceiling > in_use ? ceiling - in_use : 0;
}

// engine #1417: the budget above is a snapshot taken at device open, and the ledger is per-process, so another process
// growing its model after we opened is invisible to it - the check passes and the allocation reaches a device that is at its
// ceiling. GGML_HRX_POOL_LIVE_CHECK=1 adds a live consult of the driver's own counters to the reservation, so the promise
// "refused here rather than faulting the device" holds under concurrency too.
//
// Off by default, deliberately: enabling it makes admission depend on what other processes are doing, so a launch that works
// on an idle box can refuse on a busy one. That trade is the operator's to take; this code only makes it available.
// GGML_HRX_POOL_LIVE_MARGIN_MB leaves headroom for concurrent growth (default 0 - refuse only when the box truly has no room).
static bool pool_live_check_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_POOL_LIVE_CHECK");
        return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
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
            return static_cast<size_t>(0);
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
                  "(engine #1502; a restart restores the estimate)\n",
                  __func__, size / (double) kMiB, budget / (double) kMiB, next / (double) kMiB);
    return next;
}

size_t pool_budget_refresh(size_t uncapped, size_t cap, size_t cap_free, size_t free_now) {
    if (cap == 0) {
        return uncapped;  // no cap in force
    }
    if (free_now > cap_free) {
        // The pool has grown since the refusal, so the cap's evidence no longer holds: a transient refusal must not become a
        // permanent ceiling (engine #1502 review).
        GGML_LOG_INFO("%s: the pool grew from %.1f MiB to %.1f MiB free since the last refusal; lifting the %.1f MiB cap\n",
                      __func__, cap_free / (double) kMiB, free_now / (double) kMiB, cap / (double) kMiB);
        return uncapped;
    }
    return cap < uncapped ? cap : uncapped;
}

bool pool_reserve(size_t budget, size_t size, const char * what) {
    const size_t used = g_pool_allocated.load();
    if (pool_live_check_enabled()) {
        size_t total = 0;
        size_t in_use = 0;
        if (pool_live_counters(total, in_use)) {
            const size_t live_free = in_use >= total ? 0 : total - in_use;
            const size_t margin    = pool_live_margin_bytes();
            const size_t wanted    = size + (margin > SIZE_MAX - size ? SIZE_MAX - size : margin);
            if (wanted > live_free) {
                GGML_LOG_ERROR("%s: HRX pool limit: %s of %.1f MiB does not fit the live device budget "
                               "(%.1f MiB free - %.1f MiB margin of %.1f MiB); unset GGML_HRX_POOL_LIVE_CHECK to use the "
                               "open-time snapshot instead (engine #1417)\n",
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
