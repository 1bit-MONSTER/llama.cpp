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
constexpr size_t kDeclineLogLines = 8;

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
