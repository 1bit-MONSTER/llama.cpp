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

// The budget every HRX buffer is charged against: GGML_HRX_POOL_LIMIT_MB when set, otherwise the device's
// memory_total.
size_t pool_budget(size_t memory_total);

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
