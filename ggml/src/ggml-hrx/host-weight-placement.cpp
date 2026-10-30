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

namespace ggml::hrx {

namespace {

// True for a tensor that is a model weight and is NOT in an HRX-allocated buffer, i.e. a tensor the
// executor would have to copy into device memory before it could bind it.
bool host_resident_weight(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    ggml_backend_buffer_t buffer = tensor->view_src != nullptr ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == nullptr) {
        return false;
    }
    // HRX's own device and host buffers are bound directly, so they are not the case this guards.
    if (buffer->iface.get_base == ggml_backend_hrx_buffer_base) {
        return false;
    }
    if (!ggml_backend_buffer_is_host(buffer)) {
        return false;
    }
    return ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

}  // namespace

bool host_weight_requires_cpu(const ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (host_resident_weight(op->src[i])) {
            return true;
        }
    }
    return false;
}

}  // namespace ggml::hrx
