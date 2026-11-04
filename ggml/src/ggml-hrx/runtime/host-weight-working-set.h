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

#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml::hrx {

// Materialized bytes the model's host weights need resident, learned once from the program bindings before they are
// materialized (engine #669 / #900).
//
// Why a per-model sum and not a per-weight test: the budget is a residency budget, and what has to be resident is the
// weights, not one of them. The per-weight basis is not even knowable at placement time - `HostWeightCache::acquire()`
// charges the *materialized* length, which is the carrier the dispatch builder picks after placement rewrites each
// source row-block (136 B in the file versus 132 B for the shared4 carrier, or 144 B for the row64 ones) - so no
// per-weight comparison can be both tight and safe. And the arena's advice (`need + resident + parked`) is a snapshot
// that moves with the budget: measured, following it cycles 38 <-> 44 MiB forever on Qwen3-4B-IQ4_XS.
//
// The sum is exact for a fixed set of weights and accumulates one entry per distinct weight, keeping the largest
// materialized length seen for each (the same weight can be bound with different layouts by different programs).
// Process-wide, like HostWeightCache::residency_limit().
void host_weight_working_set_observe(uint64_t identity, uint64_t generation, size_t materialized_bytes);

// Sum of the per-weight materialized maxima, or 0 when nothing has been observed yet (unknown, not "zero bytes").
size_t host_weight_working_set_bytes();

// True when the set is known and cannot fit the budget. A zero limit means unbounded, exactly as in the cache.
bool host_weight_working_set_exceeds(size_t limit);

// Tests only: forget what has been observed.
void host_weight_working_set_reset();

}  // namespace ggml::hrx
