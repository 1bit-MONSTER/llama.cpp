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
#include "ggml.h"
#include "runtime/host-memory.h"
#include "runtime/host-weight-working-set.h"

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
            return true;
        }
    }
    return false;
}

}  // namespace ggml::hrx
