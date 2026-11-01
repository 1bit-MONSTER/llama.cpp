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

#include <atomic>
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

bool pool_reserve(size_t budget, size_t size, const char * what) {
    const size_t used = g_pool_allocated.load();
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
