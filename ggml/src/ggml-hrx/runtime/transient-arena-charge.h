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
//
// What the transient arena owes the pool ledger when it grows (engine #1711).
//
// The guardrail used to be consulted only by the buffer-type allocator, so the arena - the scratch a launch grows into -
// was allocated uncharged, and "a launch that needs more is refused here" (#680) held for tensors only. Charging it is
// two numbers, and the second one is where a naive version goes wrong: when the superseded allocation is RELEASED its
// charge goes with it, but in diagnostic mode it is RETIRED (diagnostic_retired_buffers_) and stays resident, so its
// charge has to stay too. Releasing it there under-reports by exactly the buffers kept around to diagnose memory
// problems, i.e. the ledger stays comfortable while the device is full.
//
// Pure and header-only so the decision can be tested with no device and no libhrx (the same reason
// host-memory-refusal.h lives apart from HostWeightCache).

#pragma once

#include <cstddef>

namespace ggml::hrx {

struct ArenaChargePlan {
    std::size_t reserve_bytes = 0;  // to take from the pool BEFORE the old allocation is touched
    std::size_t release_bytes = 0;  // to return once the old allocation is gone (0 when it is retired, not freed)
};

// required: the arena size the launch needs; capacity: the arena's current allocation (0 for the first one);
// retires_old: the arena's diagnostic mode, which keeps superseded buffers alive.
inline ArenaChargePlan plan_arena_charge(std::size_t required, std::size_t capacity, bool retires_old) {
    ArenaChargePlan plan;
    if (required <= capacity) {
        return plan;  // no growth, nothing to charge and nothing to give back
    }
    plan.reserve_bytes = required - capacity;
    // A first allocation has nothing to release; a growth releases the superseded buffer's charge only when that
    // buffer is actually freed. Reserved-and-retired bytes stay charged (see this file's header).
    plan.release_bytes = (!retires_old && capacity > 0) ? capacity : 0;
    return plan;
}

}  // namespace ggml::hrx
