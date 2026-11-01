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

// The reserve the ceiling keeps for the driver and for kernel/transient spikes, in bytes.
size_t pool_reserve_bytes();

// GTT total and currently used bytes, from the driver's own counters (/sys/class/drm/card*/device/mem_info_gtt_*).
// The AMDGPU HAL does not report available memory through hrx_device_memory_info (libhrx returns UNAVAILABLE), and
// these counters are the only cross-process view: the #680 ceiling has to account for what other agents hold too.
bool gtt_counters(size_t & total, size_t & used);

// Charge `size` against `budget`; false (with one log line naming the sizes) when it does not fit.
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
