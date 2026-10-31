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

#include <cstdlib>
#include <cstring>
#include <string>

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
    // TODO(i669): temporary bisect knob, remove before the PR. GGML_HRX_169_DECLINE="MUL_MAT,CONT" restricts the
    // decline to those ops; unset means every op below is declined.
    if (const char * only = std::getenv("GGML_HRX_169_DECLINE")) {
        if (std::strcmp(only, "none") == 0) {
            return false;
        }
        struct Named {
            const char *  name;
            enum ggml_op op;
        };
        static const Named kNamed[] = {
            { "MUL_MAT", GGML_OP_MUL_MAT },     { "MUL_MAT_ID", GGML_OP_MUL_MAT_ID },
            { "GET_ROWS", GGML_OP_GET_ROWS },   { "RMS_NORM", GGML_OP_RMS_NORM },
            { "ROPE", GGML_OP_ROPE },           { "CONT", GGML_OP_CONT },
            { "CPY", GGML_OP_CPY },             { "GLU", GGML_OP_GLU },
            { "MUL", GGML_OP_MUL },             { "ADD", GGML_OP_ADD },
        };
        if (std::strcmp(only, "all") != 0) {
            bool listed = false;
            const std::string list(only);
            size_t            start = 0;
            while (start <= list.size()) {
                const size_t end  = list.find(',', start);
                const std::string name = list.substr(start, end == std::string::npos ? std::string::npos : end - start);
                for (const Named & entry : kNamed) {
                    if (name == entry.name && op->op == entry.op) {
                        listed = true;
                    }
                }
                if (end == std::string::npos) {
                    break;
                }
                start = end + 1;
            }
            if (!listed) {
                return false;
            }
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (host_resident_weight(op->src[i])) {
            return true;
        }
    }
    return false;
}

}  // namespace ggml::hrx
