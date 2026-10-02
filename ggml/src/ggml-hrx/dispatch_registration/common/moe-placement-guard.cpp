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

// Placement guard for gpt-oss's MoE tail ops (ADD_ID, SWIGLU_OAI). When the expert MUL_MAT_IDs they follow run on
// the CPU, running SWIGLU_OAI on HRX between those CPU splits gives wrong results in the full model, although every
// op is right on its own and the same block is right in isolation (root cause open: engine #286). Until that is
// found, these ops are only claimed when their MUL_MAT_ID will run on HRX too.
//
// "Will run on HRX" takes two checks, because the scheduler places a MUL_MAT_ID with its expert weights:
//   - HRX can execute the MUL_MAT_ID (a batch HRX declines, or a disabled dispatch, sends it to the CPU), and
//   - the expert weights sit in a buffer HRX can read. Experts in a CPU-only buffer (CPU_REPACK, where llama.cpp
//     puts them when HRX declined the MUL_MAT_ID at load time, or where an override puts them) keep every
//     MUL_MAT_ID on the CPU, decode included, even when HRX could execute the decode shape.

#include "moe-placement-guard.h"

#include "ggml-backend.h"
#include "ggml.h"

namespace ggml::hrx {

namespace {

// The MUL_MAT_ID an ADD_ID / SWIGLU_OAI input comes from, directly or through ADD_ID, or nullptr.
const ggml_tensor * expert_source(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (tensor->op == GGML_OP_MUL_MAT_ID) {
        return tensor;
    }
    if (tensor->op == GGML_OP_ADD_ID && tensor->src[0] != nullptr && tensor->src[0]->op == GGML_OP_MUL_MAT_ID) {
        return tensor->src[0];
    }
    return nullptr;
}

// False when the expert weights are already in a buffer this device cannot use; true when it can, or before
// allocation (no buffer yet).
bool experts_readable(ggml_backend_dev_t device, const ggml_tensor * experts) {
    const ggml_tensor * weights = experts->src[0];
    if (weights == nullptr) {
        return true;
    }
    const ggml_tensor *   storage = weights->view_src != nullptr ? weights->view_src : weights;
    ggml_backend_buffer_t buffer  = storage->buffer;
    return buffer == nullptr || ggml_backend_dev_supports_buft(device, ggml_backend_buffer_get_type(buffer));
}

}  // namespace

bool moe_tail_claimable(ggml_backend_dev_t device, const ggml_tensor * op) {
    if (op == nullptr) {
        return true;
    }
    const bool add_id     = op->op == GGML_OP_ADD_ID;
    const bool swiglu_oai = op->op == GGML_OP_GLU && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI;
    if (!add_id && !swiglu_oai) {
        return true;
    }
    for (int i = 0; i < 2; ++i) {
        const ggml_tensor * experts = expert_source(op->src[i]);
        if (experts != nullptr &&
            (!experts_readable(device, experts) || !ggml_backend_dev_supports_op(device, experts))) {
            return false;
        }
    }
    return true;
}

}  // namespace ggml::hrx
