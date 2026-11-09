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
// The transient arena's share of the pool budget (engine#1711).
//
// The pool guardrail (#680) charges the ggml tensor buffers and nothing else: the per-launch scratch an op actually grows
// into is allocated straight from the runtime allocator, so "a launch that needs more is refused here rather than faulting
// the device" was only true of one of the two ways a launch can run out of memory. This is the decision that makes it true
// of the scratch too, kept pure and separate from the arena so it can be tested without a device.
//
// Two rules matter and both are here rather than in the caller:
//   - the arena's charge is its CURRENT capacity, released before a new one is taken. A launch that inherits a buffer at
//     least as large as it needs is charged nothing (the arena returns early), which is what keeps the accounting from
//     double-counting a buffer that is already paid for;
//   - a refusal is a decision about the BUDGET, not about the device: `refuse` means the pool cannot take the request,
//     which the caller turns into a Status and the caller of that into a failed build (never a raw allocation that may
//     fault).
#pragma once

#include <cstddef>
#include <vector>

namespace ggml::hrx {

struct TransientArenaCharge {
    bool   allocate      = false;  // the arena has to (re)allocate at required_size
    bool   release_first = false;  // its current capacity must go back to the pool first (it is being replaced)
    size_t release_bytes = 0;
    size_t charge_bytes  = 0;  // what the pool must accept before the allocation may happen
    bool   refuse        = false;  // a request larger than the WHOLE budget: a fast path, not the gate (see below)
};

// The order the arena must perform the steps in, so a test can pin it without an allocator (engine#1711's review: the
// first version reserved AFTER allocating, which left the faulting allocation exactly where it was). Reserve comes
// before allocate, and the replaced capacity is released before either.
enum class TransientArenaStep { ReleaseOld, Reserve, Allocate };

// capacity: the arena's current allocation (0 when it has none). required: the size the caller needs. budget: the pool
// budget the device was admitted with (0 = unset, which the guardrail treats as unbounded).
inline TransientArenaCharge plan_transient_arena_charge(size_t capacity, size_t required, size_t budget) {
    TransientArenaCharge plan;
    if (required == 0 || capacity >= required) {
        return plan;  // nothing to do, and nothing to charge: this is the no-double-count case
    }
    plan.allocate      = true;
    plan.release_first = capacity > 0;
    plan.release_bytes = capacity;
    plan.charge_bytes  = required;
    plan.refuse        = budget != 0 && required > budget;
    return plan;
}

// The steps for a planned grow, in order. `release_old` is dropped when there is nothing to replace, so a test can assert
// the relative order of the two that matter rather than a fixed length.
inline std::vector<TransientArenaStep> transient_arena_steps(const TransientArenaCharge & plan) {
    std::vector<TransientArenaStep> steps;
    if (!plan.allocate) {
        return steps;
    }
    if (plan.release_first) {
        steps.push_back(TransientArenaStep::ReleaseOld);
    }
    steps.push_back(TransientArenaStep::Reserve);
    steps.push_back(TransientArenaStep::Allocate);
    return steps;
}

}  // namespace ggml::hrx
