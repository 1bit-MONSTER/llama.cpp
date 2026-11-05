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

#include "host-weight-placement.h"

#include "backend-buffer-binding.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "runtime/host-memory.h"
#include "runtime/host-weight-working-set.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

namespace ggml::hrx {

namespace {

// True for a model weight that is NOT in an HRX-allocated buffer, i.e. a tensor the executor would have to copy
// into device memory before it could bind it.
bool host_resident_weight(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr) {
        return false;
    }
    if (buffer->iface.get_base == ggml_backend_hrx_buffer_base) {
        return false;
    }
    if (!ggml_backend_buffer_is_host(buffer)) {
        return false;
    }
    return ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// A decline here moves a node off the device, i.e. it changes the graph's placement, so it has to be greppable. Before
// this line the only witness was the GGML_HRX_REPORT placement census, so a budget that silently reshaped the graph
// looked exactly like one that changed nothing (engine #674, 2026-11-02).
//
// Bounded AND de-duplicated: device_supports_op is queried repeatedly per node (reserve pass, compute pass), so a plain
// cap spends the whole budget on the same two decisions - at a 128 MiB budget that is 8 lines for 2 nodes, and at an
// 8 MiB budget (109 declined tensors) the reader would never see the GEMMs move at all. One line per distinct
// (weight, op) pair, eight pairs, then one line saying the rest are suppressed.
//
// When it can fire, measured (Qwen3-4B-Q4_K_M, -dev HRX0 -fit off -ngl 0, budget 128 MiB): only if the scheduler's
// reserve runs before the host-weight arena's first unservable acquire. At the cli's default verbosity the arena
// refuses the model during load (1.2 s in) and this predicate is never asked, so the log is silent; the same binary
// run with -v or -lv 2 reserves at 0.17 s, logs the two lines, and reaches #173's refusal later at the first
// execution (8.99 s) - 3/3 runs each way, both ending in "Compute error.". So silence here is not evidence that no
// node was declined: it can mean the run never got as far as splitting the graph (engine #674, PR #174 review).
constexpr size_t kDeclineLogLines = 8;

// Static on purpose (one process, one model) - the key is a weight pointer, so a second model or session in the same
// process logs nothing. Fine for a diagnostic; it would need to be per-context if it ever becomes more than that.

void log_declined_placement(const ggml_tensor * op, const ggml_tensor * weight, size_t limit) {
    static std::mutex mutex;
    static std::vector<std::pair<const void *, int>> seen;
    static bool suppressed_reported = false;

    const std::pair<const void *, int> key{weight, static_cast<int>(op->op)};
    std::lock_guard<std::mutex> lock(mutex);
    if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
        return;
    }
    if (seen.size() >= kDeclineLogLines) {
        if (!suppressed_reported) {
            suppressed_reported = true;
            GGML_LOG_WARN("hrx placement: further declines are not logged (%zu distinct shown); the graph is larger "
                          "than this line budget (GGML_HRX_HOST_WEIGHT_LIMIT_MB)\n", seen.size());
        }
        return;
    }
    seen.push_back(key);
    GGML_LOG_WARN("hrx placement: %s is not run on HRX because its weight '%s' (%zu MiB) exceeds the %zu MiB "
                  "residency budget (GGML_HRX_HOST_WEIGHT_LIMIT_MB), so the CPU serves this node\n",
                  ggml_op_name(op->op), weight->name, ggml_nbytes(weight) >> 20, limit >> 20);
}

}  // namespace

bool host_weight_exceeds_residency(const ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }
    const size_t limit = HostWeightCache::residency_limit();
    if (limit == 0) {
        return false;
    }
    // engine #669/#900: once the model's host-weight set is known it is what the budget bounds, so the answer is one
    // number for every host weight instead of a per-weight test against a size that is not the resident size. Before
    // the first execution has been measured there is nothing to compare, so the per-weight file-size test below stays
    // as the first-graph estimate.
    // Only the direction the set can prove: if the model's host weights alone already exceed the budget, no ordering
    // of them fits, so decline every host-weight op and let the CPU serve it. The other direction is NOT implied - the
    // arena also has to hold the slots parked while programs are in flight (measured: 167.3 MiB resident + 343.2 MiB
    // parked for a 304 MiB set at a 512 MiB budget), so a set that fits is not a run that fits.
    if (host_weight_working_set_exceeds(limit)) {
        return true;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * weight = op->src[i];
        if (host_resident_weight(weight) && ggml_nbytes(weight) > limit) {
            log_declined_placement(op, weight, limit);
            return true;
        }
    }
    return false;
}

}  // namespace ggml::hrx
