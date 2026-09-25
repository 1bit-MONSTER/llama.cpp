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

// Claim for nodes that ggml-hrx executes only inside a fused dispatch.
//
// device_supports_op claims a node when can_execute_standalone_op_as_graph says the
// dispatcher can run it alone, so that everything else falls back to another backend.
// Some nodes of the Qwen3 MoE and Qwen3.5/3.6 graphs have no standalone dispatch and
// run only inside a fused pattern:
//
// - the MoE router chain SOFT_MAX -> ARGSORT -> GET_ROWS -> SUM_ROWS -> CLAMP
//   (llm.moe_router.top8_f32 and the decode projection fusion);
// - the gated-delta-net chain L2_NORM, SOFTPLUS and GATED_DELTA_NET;
// - the per-head RMS_NORM and ROPE of the attention fusions (head_dim 64..256).
//
// With the standalone test alone these nodes went to the CPU, which split every MoE
// and DeltaNet layer: Qwen3.6-35B-A3B prefill fell to 67 tok/s at pp512 and
// Qwen3-30B-A3B prompt decode failed ("value alias target is not transient").
//
// A node keeps the per-op claim when it has the producers of those patterns, which
// only a model graph gives it. The single-op and small multi-op graphs of
// test-backend-ops feed these ops from leaves, from other producers or with shapes the
// fusions do not take (4-D, rows wider than a head), so they still go through the
// standalone test.

#include "ggml.h"

namespace ggml::hrx {

// the op that produced a tensor's storage, looking through views and reshapes; GGML_OP_NONE for a leaf
inline ggml_op fused_context_producer_op(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return GGML_OP_NONE;
    }
    while (tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor->op;
}

// the tensor that owns a tensor's storage, looking through views and reshapes
inline const ggml_tensor * fused_context_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

// true for the router's GET_ROWS: softmax probabilities gathered at the argsort's top-k ids,
// the chain the top-8 router dispatch starts from (a sigmoid router, as in GLM-4.7-Flash, has none)
inline bool fused_context_router_get_rows(const ggml_tensor * op) {
    return op != nullptr && op->op == GGML_OP_GET_ROWS &&
           fused_context_producer_op(op->src[0]) == GGML_OP_SOFT_MAX &&
           fused_context_producer_op(op->src[1]) == GGML_OP_ARGSORT;
}

inline bool fused_context_fed_by_op(const ggml_tensor * op) {
    for (const ggml_tensor * source : op->src) {
        if (source != nullptr && fused_context_producer_op(source) != GGML_OP_NONE) {
            return true;
        }
    }
    return false;
}

inline bool fused_context_is_3d(const ggml_tensor * op) {
    return op->ne[3] == 1;
}

inline bool fused_context_is_head_row(const ggml_tensor * op) {
    return op->ne[0] == 64 || op->ne[0] == 128 || op->ne[0] == 256;
}

// true when op is a node that only a fused dispatch executes, with the producers of that fused pattern
inline bool fused_context_claim(const ggml_tensor * op) {
    if (op == nullptr || !fused_context_is_3d(op)) {
        return false;
    }
    switch (op->op) {
        // MoE router: softmax over [n_expert, n_tokens] router logits, then top-k
        case GGML_OP_SOFT_MAX:
            return op->src[1] == nullptr && fused_context_producer_op(op->src[0]) == GGML_OP_MUL_MAT &&
                   op->ne[2] == 1 && op->ne[0] >= 32 && op->ne[0] <= 512 && op->ne[0] % 32 == 0;
        case GGML_OP_ARGSORT:
            return fused_context_producer_op(op->src[0]) == GGML_OP_SOFT_MAX;
        case GGML_OP_GET_ROWS:
            return fused_context_producer_op(op->src[0]) == GGML_OP_SOFT_MAX &&
                   fused_context_producer_op(op->src[1]) == GGML_OP_ARGSORT;
        case GGML_OP_SUM_ROWS:
            return fused_context_router_get_rows(fused_context_root(op->src[0]));
        case GGML_OP_CLAMP: {
            const ggml_tensor * sum = fused_context_root(op->src[0]);
            return sum != nullptr && sum->op == GGML_OP_SUM_ROWS &&
                   fused_context_router_get_rows(fused_context_root(sum->src[0]));
        }
        // gated delta net
        case GGML_OP_L2_NORM:
        case GGML_OP_GATED_DELTA_NET:
            return fused_context_fed_by_op(op);
        case GGML_OP_UNARY:
            return ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS && fused_context_fed_by_op(op);
        // per-head attention normalisation and rotation
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
            return fused_context_is_head_row(op) && fused_context_fed_by_op(op);
        default:
            return false;
    }
}

}  // namespace ggml::hrx
