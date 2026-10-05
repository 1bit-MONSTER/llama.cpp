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

// Transient arena packing that does not add execution barriers.
//
// The stock packer hands a freed arena range to the next value that fits, often the value written by the
// very next command. The graph recorder tracks arena ranges, so that reuse becomes a write-after-read
// dependency on the command that last read the range, and HRX puts an execution barrier before the new
// writer even when the two commands share no data (two projections of one input, for example).
//
// This packer reuses a freed range only for a value whose first command already depends, through the
// program's own data dependencies, on every command that used the range before. Such a reuse adds no
// dependency the recorder would not already have. Otherwise the value gets fresh arena space. A program
// whose unguarded arena exceeds 16 MiB, or whose guarded arena would exceed twice the unguarded one plus
// 8 MiB (512-token prompt programs), keeps the stock packing. GGML_HRX_TRANSIENT_REUSE=legacy restores the stock packer everywhere, and
// GGML_HRX_LOG_TRANSIENT_REUSE=1 logs both arena sizes for every program.

#pragma once

#include "command-program.h"

#include <cstdint>
#include <vector>

namespace ggml::hrx {

struct TransientReuseLifetime {
    TransientAllocation allocation;
    uint32_t            first_use = 0;
    uint32_t            last_use  = 0;
};

// Assigns arena offsets to |lifetimes| (main-program transients) and appends them to |plan|, growing
// plan.arena_size as needed. Returns false, leaving |plan| untouched, when the dependency-aware packer is
// disabled or the program is too large for it; the caller then uses the stock packer.
bool pack_transients_without_false_dependencies(const std::vector<Command> &          commands,
                                                std::vector<TransientReuseLifetime> & lifetimes,
                                                TransientPlan &                       plan);

}  // namespace ggml::hrx
