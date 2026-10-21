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

}  // namespace

size_t pool_budget(size_t memory_total) {
    const char * env = std::getenv("GGML_HRX_POOL_LIMIT_MB");
    if (env != nullptr) {
        const long long mb = std::atoll(env);
        if (mb > 0) {
            return static_cast<size_t>(mb) * kMiB;
        }
    }
    return memory_total;
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
