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

#include "hrx-1bit-staging-pool.h"

#include "ggml-impl.h"
#include "hrx_runtime.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ggml::hrx::onebit {

namespace {

struct Slot {
    size_t       size   = 0;
    hrx_buffer_t buffer = nullptr;
    void *       mapped = nullptr;
};

// One pool per process: HRX has one device per process in practice, and the buffers are only ever bound to
// that device's stream.
constexpr size_t      kPoolCap = 8;  // buffers kept for reuse; more than this is released on return
std::vector<Slot>     g_free;
std::atomic<uint64_t> g_allocations{ 0 };
std::atomic<uint64_t> g_reuses{ 0 };

bool debug_enabled() {
    const char * env = std::getenv("GGML_HRX_STAGING_POOL_DEBUG");
    return env != nullptr && std::atoi(env) != 0;
}

}  // namespace

Status staging_pool_acquire(hrx_device_t device, size_t size, hrx_buffer_t & buffer, void *& mapped) {
    for (size_t i = 0; i < g_free.size(); ++i) {
        if (g_free[i].size >= size) {
            buffer = g_free[i].buffer;
            mapped = g_free[i].mapped;
            g_free.erase(g_free.begin() + static_cast<std::ptrdiff_t>(i));
            const uint64_t n = g_reuses.fetch_add(1) + 1;
            if (debug_enabled()) {
                GGML_LOG_INFO("hrx-staging-pool: reuse %" PRIu64 " (%zu bytes, %zu free)\n", n, size, g_free.size());
            }
            return Status{};
        }
    }
    Status status = allocate_mapped_host_staging_buffer(device, size, buffer, mapped);
    if (status.success()) {
        const uint64_t n = g_allocations.fetch_add(1) + 1;
        if (debug_enabled()) {
            GGML_LOG_INFO("hrx-staging-pool: allocate %" PRIu64 " (%zu bytes, %zu free)\n", n, size, g_free.size());
        }
    }
    return status;
}

void staging_pool_give_back(hrx_buffer_t buffer, void * mapped, size_t size) {
    if (buffer == nullptr) {
        return;
    }
    if (g_free.size() < kPoolCap) {
        g_free.push_back({ size, buffer, mapped });
        return;
    }
    hrx_buffer_release(buffer);
}

void staging_pool_drain() {
    if (!g_free.empty() && debug_enabled()) {
        GGML_LOG_INFO("hrx-staging-pool: drain %zu buffers after %" PRIu64 " allocations and %" PRIu64 " reuses\n",
                      g_free.size(), g_allocations.load(), g_reuses.load());
    }
    for (Slot & slot : g_free) {
        if (slot.buffer != nullptr) {
            hrx_buffer_release(slot.buffer);
            slot.buffer = nullptr;
        }
    }
    g_free.clear();
}

}  // namespace ggml::hrx::onebit
