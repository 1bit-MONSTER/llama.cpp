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

// Host test for host-weight-working-set.h: the number a bounded run's budget is judged on. No device, no HRX runtime.
// What matters is that a weight is counted once (however many programs bind it), that the largest materialized length
// wins for it, that "unknown" (0) is not "zero bytes", and that a zero budget means unbounded as it does in the cache.

#include "runtime/host-weight-working-set.h"

#include <cstdio>
#include <cstdlib>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

using ggml::hrx::host_weight_working_set_bytes;
using ggml::hrx::host_weight_working_set_exceeds;
using ggml::hrx::host_weight_working_set_observe;
using ggml::hrx::host_weight_working_set_reset;

static const size_t MiB = size_t{ 1 } << 20;

int main() {
    host_weight_working_set_reset();
    // Unknown is not zero: nothing observed yet, so no budget can be judged.
    REQUIRE(host_weight_working_set_bytes() == 0);
    REQUIRE(!host_weight_working_set_exceeds(1));
    REQUIRE(!host_weight_working_set_exceeds(0));

    // Two weights, as two programs bind them: the model's set is their sum.
    host_weight_working_set_observe(/*identity=*/7, /*generation=*/1, 12 * MiB);
    host_weight_working_set_observe(8, 1, 5 * MiB);
    REQUIRE(host_weight_working_set_bytes() == 17 * MiB);

    // The same weight bound again by another program is not a second copy - this is what makes it a model's set
    // rather than a per-program one.
    host_weight_working_set_observe(7, 1, 12 * MiB);
    REQUIRE(host_weight_working_set_bytes() == 17 * MiB);

    // A different layout for the same weight only counts if it is larger (the shared4 carrier is 132 B per row-block
    // against 144 B for the row64 ones), and the increase is the difference, not a replacement.
    host_weight_working_set_observe(7, 1, 13 * MiB);
    REQUIRE(host_weight_working_set_bytes() == 18 * MiB);
    host_weight_working_set_observe(7, 1, 11 * MiB);
    REQUIRE(host_weight_working_set_bytes() == 18 * MiB);

    // A reloaded weight is a new generation, i.e. new storage, so it is counted separately.
    host_weight_working_set_observe(7, 2, 1 * MiB);
    REQUIRE(host_weight_working_set_bytes() == 19 * MiB);

    // Zero bytes is a binding this change does not own (not a residency request): nothing to add.
    host_weight_working_set_observe(9, 1, 0);
    REQUIRE(host_weight_working_set_bytes() == 19 * MiB);

    // The budget comparison, including the boundary and the unbounded case.
    REQUIRE(host_weight_working_set_exceeds(18 * MiB));
    REQUIRE(!host_weight_working_set_exceeds(19 * MiB));
    REQUIRE(!host_weight_working_set_exceeds(19 * MiB + 1));
    REQUIRE(!host_weight_working_set_exceeds(0));

    host_weight_working_set_reset();
    REQUIRE(host_weight_working_set_bytes() == 0);
    REQUIRE(!host_weight_working_set_exceeds(1));

    std::printf("host-weight-working-set: all requirements passed\n");
    return 0;
}
