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
#include <cstdio>
#include <string>

namespace ggml::hrx {

// The arena state at a host-weight refusal (engine #900). "The budget is too small" and "the free space is split"
// are different claims, and the free list tells them apart: when `free_bytes >= need` and no single block is that
// large, the space exists and is fragmented, so the size advice ("raise it to at least N MiB") is both wrong and
// self-contradictory - engine #900 measured "Raise it to at least 4030 MiB" while the budget in force was 4096 MiB.
struct ArenaMiss {
    size_t need          = 0;  // the aligned slot that was wanted
    size_t free_bytes    = 0;  // arena_free_bytes_
    size_t free_blocks   = 0;  // arena_free_.size()
    size_t largest_block = 0;  // the largest of those blocks
    size_t resident      = 0;  // resident_bytes_
    size_t parked        = 0;  // bytes held by the running execution
    size_t capacity      = 0;  // arena_capacity_
    size_t limit         = 0;  // GGML_HRX_HOST_WEIGHT_LIMIT_MB, 0 = unbounded
};

inline constexpr const char * kFragmentedAdvice =
    "the free space is fragmented, not short: no single block can serve the slot, so raising "
    "GGML_HRX_HOST_WEIGHT_LIMIT_MB only helps if it re-creates the arena contiguously (engine #900)";

inline bool fragmented(const ArenaMiss & miss) {
    return miss.free_bytes >= miss.need && miss.largest_block < miss.need;
}

// One sentence for the operator, no trailing newline. Fragmented misses name the split and promise nothing about the
// budget; genuine capacity misses keep the engine #788 wording and its raise-to number, because there it is true.
inline std::string describe(const ArenaMiss & miss) {
    constexpr double kMiB = 1024.0 * 1024.0;
    char             text[640];
    if (fragmented(miss)) {
        std::snprintf(text, sizeof(text),
                      "host weight arena could not place a %.1f MiB slot: %.1f MiB free in %zu blocks, largest %.1f MiB "
                      "(arena %.1f MiB, limit %.1f MiB, resident %.1f MiB, parked %.1f MiB) - %s",
                      miss.need / kMiB, miss.free_bytes / kMiB, miss.free_blocks, miss.largest_block / kMiB,
                      miss.capacity / kMiB, miss.limit / kMiB, miss.resident / kMiB, miss.parked / kMiB,
                      kFragmentedAdvice);
    } else {
        // largest_block >= need here would mean the take failed for another reason (a failed growth logs its own
        // line), so the capacity wording is kept rather than claiming fragmentation that the numbers deny.
        const size_t raise = (miss.need + miss.resident + miss.parked + (1024 * 1024 - 1)) / (1024 * 1024);
        std::snprintf(text, sizeof(text),
                      "host weight arena cannot hold this model's per-execution working set: a %.1f MiB slot does not fit "
                      "with %.1f MiB resident and %.1f MiB parked (arena %.1f MiB, limit %.1f MiB, free %.1f MiB in %zu "
                      "blocks, largest %.1f MiB). Raise it to at least %zu MiB or unset it (engine #788).",
                      miss.need / kMiB, miss.resident / kMiB, miss.parked / kMiB, miss.capacity / kMiB, miss.limit / kMiB,
                      miss.free_bytes / kMiB, miss.free_blocks, miss.largest_block / kMiB, raise);
    }
    return text;
}

}  // namespace ggml::hrx
