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

// Host test for failure-status.h: whether a backend failure is a resource failure (engine #1618). Pure - no device, no
// runtime library - because the rule is a function of the messages libhrx renders.
//
// The case this exists for: a Flash-Next decode with flash attention on failed with
// `al...: RESOURCE_EXHAUSTED; [hsa_amd_memory_pool_allocate] HSA_STATUS_ERROR_OUT_OF_RESOURCES` while materializing a host
// weight, and the client was told "Compute error." - a resource failure served as a kernel fault.

#include "runtime/failure-status.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

using ggml::hrx::failure_looks_like_allocation;
using ggml::hrx::failure_status;

int main() {
    // The issue's own line, verbatim from the log that was misread.
    const std::vector<std::string> pool_exhausted = {
        "allocate host memory device buffer: hrx-system/runtime/src/iree/hal/drivers/amdgpu/slab_provider.c:663: "
        "RESOURCE_EXHAUSTED; [hsa_amd_memory_pool_allocate] HSA_STATUS_ERROR_OUT_OF_RESOURCES",
    };
    REQUIRE(failure_looks_like_allocation(pool_exhausted));
    REQUIRE(failure_status(pool_exhausted) == GGML_STATUS_ALLOC_FAILED);

    // HSA's name alone is enough: libhrx renders both, and a caller may see only one of them.
    REQUIRE(failure_looks_like_allocation({ "materialize host weight value 3 failed: HSA_STATUS_ERROR_OUT_OF_RESOURCES" }));
    REQUIRE(failure_status({ "x: RESOURCE_EXHAUSTED" }) == GGML_STATUS_ALLOC_FAILED);

    // A real compute failure must stay a compute failure: this is the direction that would be a regression, because a
    // kernel error reported as "out of memory" sends the reader to free memory that was never the problem.
    const std::vector<std::string> kernel_fault = {
        "execute HRX command program failed: compute status: -1",
        "hip.qsa_attn_select rejected node 20: unsupported layout",
    };
    REQUIRE(!failure_looks_like_allocation(kernel_fault));
    REQUIRE(failure_status(kernel_fault) == GGML_STATUS_FAILED);

    // Empty reasons keep the caller's code rather than inventing one.
    REQUIRE(!failure_looks_like_allocation({}));
    REQUIRE(failure_status({}, GGML_STATUS_FAILED) == GGML_STATUS_FAILED);
    REQUIRE(failure_status({}, GGML_STATUS_ABORTED) == GGML_STATUS_ABORTED);

    // Any message in the list counts, not just the last: graph_compute prints the last reason but classifies the whole set.
    REQUIRE(failure_looks_like_allocation({ "plan rejected: operand layout", "x: RESOURCE_EXHAUSTED" }));

    std::printf("test-hrx-failure-status: all requirements passed\n");
    return 0;
}
