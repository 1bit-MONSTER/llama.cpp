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

// 1bit's allocation-safety layer for HRX, kept out of AMD's ggml-hrx.cpp: a real pool budget that the
// planner can see, and a poisoned-device state so the first fault stops the process instead of feeding a
// wedged ring. ggml-hrx.cpp only calls in.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml::hrx::onebit {

// The budget every HRX buffer is charged against (engine #680). GGML_HRX_POOL_LIMIT_MB wins when set, otherwise the
// budget is the GTT ceiling (memory_total minus GGML_HRX_GTT_RESERVE_MB, default 16384) less what other processes
// already hold (memory_total - memory_free). A GTT plan past this budget is refused at allocation instead of
// faulting the device: #680 is a gfxhub page fault plus a MODE2 reset at GTT 106.5 of 120 GiB.
size_t pool_budget(size_t memory_total, size_t memory_free);

// The budget after an allocation the allocator refused, given the size it refused (engine #1502).
//
// The guardrail can only estimate what the pool will serve - ROCr exposes no "available" query, so pool_budget() comes from the
// sysfs GTT counter. When the allocator then refuses a size the budget admitted, that refusal is the only information the process
// ever gets about the provider's real ceiling, so spend it - but spend only what it proves: this size did not fit, so cap the
// budget at size-1 (the largest size still admissible) instead of subtracting the whole refused size, which would compound across
// retries and wedge a retrying process at zero. The next launch of that size is then refused at admission with both numbers,
// instead of failing again at model load. A budget of 0 means "device size unknown, unfettered" and stays 0. The cap is not
// permanent: pool_budget_observe() lifts it as soon as the pool is observed to have grown past the refusal's own figure.
size_t pool_budget_after_refusal(size_t budget, size_t size);

// The admission state: the budget admission uses, the cap recorded at the last refusal (0 = none), and the free figure seen when
// that cap was applied. The cap is evidence about the pool at ONE instant, so it is kept with its evidence rather than on its own.
struct pool_budget_state_t {
    size_t budget   = 0;
    size_t cap      = 0;
    size_t cap_free = 0;
};

// Bytes this process currently holds against the pool (the ledger pool_reserve/pool_release maintain).
size_t pool_allocated();

// Records a refusal of |size| bytes, seen with |free_now| free: that size can no longer be admitted, so the budget is capped at
// size-1, and the free figure is kept so the cap can be refuted later.
pool_budget_state_t pool_budget_note_refusal(pool_budget_state_t state, size_t size, size_t free_now);

// Recomputes the budget from a live observation of the pool (|free_bytes| from the driver's counters, which already contain this
// process's own allocations, and |allocated| = pool_allocated()).
//
// Two properties this exists to keep (engine #1502 review):
//  - the process's own memory is counted ONCE: the counters include it, so the basis for pool_budget() adds it back, because
//    pool_remaining() subtracts the ledger again further down the admission path;
//  - a cap is cleared only when it is LIFTED (the pool grew past the figure the refusal saw). When the pool is smaller the cap and
//    its evidence are kept, and only this call's budget is bounded by it - otherwise one low reading becomes a permanent ceiling.
pool_budget_state_t pool_budget_observe(size_t memory_total, size_t free_bytes, size_t allocated, pool_budget_state_t state);

// The reserve the ceiling keeps for the driver and for kernel/transient spikes, in bytes.
size_t pool_reserve_bytes();

// GTT total and currently used bytes, from the driver's own counters (/sys/class/drm/card*/device/mem_info_gtt_*).
// The AMDGPU HAL does not report available memory through hrx_device_memory_info (libhrx returns UNAVAILABLE), and
// these counters are the only cross-process view: the #680 ceiling has to account for what other agents hold too.
bool gtt_counters(size_t & total, size_t & used);

// Charge `size` against `budget`; false (with one log line naming the sizes) when it does not fit.
// True when |size| fits the live pool (|total|, |used| from the driver's counters) with |margin| of headroom. Pure, so the
// test drives it; the underflow and margin-saturation guards live here rather than in the caller.
bool pool_live_fits(size_t size, size_t total, size_t used, size_t margin);

bool pool_reserve(size_t budget, size_t size, const char * what);

// Give bytes back. Every failure path in an allocation must call this.
void pool_release(size_t size);

// What is left of the budget: what the ggml planner is told it may use.
size_t pool_remaining(size_t budget);

// True once a device fault has made the device unusable; logs the reason on the first refusal only.
bool device_poisoned(const char * what);

// Record the first fault (access kind and address, as the runtime reported them).
void poison_device(uint32_t access_kind, uint64_t address);

// GGML_HRX_POISON_SELFTEST=1 poisons at the first allocation, so the refusal path is testable without
// faulting the GPU.
void poison_selftest();

}  // namespace ggml::hrx::onebit
