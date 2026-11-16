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

// Host test for host-weight-placement.h: residence is not usage (engine#2494).
//
// The census named two by-design reasons a node sits on the CPU, and both asked host_resident_weight(), which ends
// with `usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. A tensor that is genuinely host-resident but whose buffer is
// USAGE_ANY -- an index, a cache, a bias, an activation kept on the host -- therefore returned false, and the one
// state a GPU cannot dereference without a mapping was the one the census could not print. This pins the split:
// tensor_is_host_resident() answers residence, host_resident_weight() answers residence AND usage.
//
// Uses the CPU backend because it is the only one guaranteed present; no device, no HRX runtime, no model.

#include "host-weight-placement.h"

#include <cstdio>

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml.h"

static int fails = 0;

#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", m); ++fails; } } while (0)

int main() {
    // Null is not resident: both predicates must survive being asked about nothing.
    CHECK(!ggml::hrx::tensor_is_host_resident(nullptr), "null is not host-resident");
    CHECK(!ggml::hrx::host_weight_is_host_resident(nullptr),     "null is not a host weight");

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
        std::printf("test-hrx-host-resident: no CPU backend, nothing to test\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(cpu, nullptr);
    if (backend == nullptr) {
        std::printf("FAIL: could not init the CPU backend\n");
        return 1;
    }

    ggml_init_params params = { /*.mem_size = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
                                /*.mem_buffer = */ nullptr,
                                /*.no_alloc = */ true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 4);
    ggml_tensor * v = ggml_view_2d(ctx, t, 16, 2, t->nb[1], 0);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::printf("FAIL: could not allocate context tensors\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    // 1. A host buffer with the default usage. This is the case that used to be invisible: resident, and NOT a
    //    weight, so host_resident_weight() said false and the census stayed silent about it.
    CHECK(ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_ANY, "a fresh buffer is USAGE_ANY");
    CHECK(ggml::hrx::tensor_is_host_resident(t), "a CPU-backend tensor is host-resident");
    CHECK(!ggml::hrx::host_weight_is_host_resident(t), "USAGE_ANY is not a weight");
    

    // 2. A view answers like its base: residence is a property of the storage, not of the tensor object.
    CHECK(ggml::hrx::tensor_is_host_resident(v), "a view of a host tensor is host-resident");
    CHECK(!ggml::hrx::host_weight_is_host_resident(v), "a view of a USAGE_ANY buffer is not a weight");

    // 3. Mark the same buffer as weights: now BOTH predicates are true, which is the pair the census names.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    CHECK(ggml::hrx::tensor_is_host_resident(t), "still host-resident");
    CHECK(ggml::hrx::host_weight_is_host_resident(t), "a WEIGHTS buffer is a host weight");
    

    // 4. The export only exposes the composition; residence on its own is the new export. Both must be true here,
    //    and the split is what makes the USAGE_ANY case above distinguishable from a non-host tensor.
    CHECK(ggml::hrx::tensor_is_host_resident(t) && ggml::hrx::host_weight_is_host_resident(t),
          "a WEIGHTS buffer is host-resident AND a weight");

    // engine#2494 review: THE CENSUS MUST PARTITION. Pure arithmetic, so it needs no graph and no backend -- the
    // property a reader checks by adding the printed counters up, checked here instead.
    CHECK(ggml::hrx::census_partitions(3, 2, 0, 5), "3 by design + 2 host-non-weight == 5 off-device");
    CHECK(ggml::hrx::census_partitions(0, 0, 4, 4), "all unexplained still partitions");
    CHECK(ggml::hrx::census_partitions(0, 0, 0, 0), "no node on the CPU partitions trivially");
    CHECK(!ggml::hrx::census_partitions(3, 2, 0, 6), "one node counted twice must NOT partition");
    CHECK(!ggml::hrx::census_partitions(3, 2, 0, 4), "one node missed must NOT partition");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (fails == 0) {
        std::printf("test-hrx-host-resident: ok (residence != usage)\n");
    }
    return fails == 0 ? 0 : 1;
}
