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
// - the delta-net conv prefill SSM_CONV fed by CONCAT(state, x^T), above 512 tokens;
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
#include "ubatch-token-limits.h"

#include <cstring>

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

// q or k of a gated delta net: produced by an op, and when that op is the L2_NORM, its input too
inline bool fused_context_gdn_qk_from_op(const ggml_tensor * tensor) {
    const ggml_tensor * root = fused_context_root(tensor);
    if (root == nullptr || root->op == GGML_OP_NONE) {
        return false;
    }
    return root->op != GGML_OP_L2_NORM || fused_context_producer_op(root->src[0]) != GGML_OP_NONE;
}

// true when llama.cpp named the tensor `base` or `base-<layer>`
inline bool fused_context_named(const ggml_tensor * op, const char * base) {
    const size_t length = std::strlen(base);
    return std::strncmp(op->name, base, length) == 0 && (op->name[length] == '\0' || op->name[length] == '-');
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
        // gated delta net, within the token count its matchers take (ubatch-token-limits.h): a claim
        // past it leaves the node with no dispatch and fails the whole graph (engine #389)
        case GGML_OP_L2_NORM:
            return op->ne[2] <= kUbatchMaxTokens && fused_context_fed_by_op(op);
        // The fused GATED_DELTA_NET dispatches read q, k and v from the delta-net conv output: in the
        // qwen35/qwen35moe/qwen3next graphs v is a view of that conv and q, k are L2_NORMs of views of it. The
        // test-backend-ops GATED_DELTA_NET cases feed q, k through L2_NORMs of leaves and v from a leaf; once
        // the HIP add-on ran L2_NORM alone (gpu-kernels #12) "fed by an op" claimed those graphs, which have no
        // dispatch, and 31 cases failed with "unsupported HRX node". So the claim also wants v and the
        // L2_NORM inputs produced by ops.
        case GGML_OP_GATED_DELTA_NET:
            return op->src[0] != nullptr && op->src[1] != nullptr && op->src[2] != nullptr &&
                   op->src[0]->ne[2] <= kUbatchMaxTokens && fused_context_producer_op(op->src[2]) != GGML_OP_NONE &&
                   fused_context_gdn_qk_from_op(op->src[0]) && fused_context_gdn_qk_from_op(op->src[1]);
        case GGML_OP_UNARY:
            return ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS &&
                   op->ne[1] * op->ne[2] <= 3 * kUbatchMaxTokens && fused_context_fed_by_op(op);
        // delta-net conv prefill CONCAT(state, x^T) -> SSM_CONV (d_conv 4) -> SILU: above the standalone conv
        // kernel's 512 tokens only the CONCAT matcher runs it, as chained rollback dispatches
        // (dispatch-ssm-conv-chunked.cpp). A claim cannot see the SILU consumer that matcher needs, so it
        // keys on the names the delta-net conv builder gives these tensors (qwen3next, qwen35, qwen35moe);
        // other conv layouts (Mamba's conv bias before SILU) keep the standalone test.
        case GGML_OP_SSM_CONV:
            return op->src[0] != nullptr && op->src[1] != nullptr && op->src[1]->ne[0] == 4 &&
                   op->src[0]->op == GGML_OP_CONCAT && fused_context_named(op, "conv_output_raw") &&
                   fused_context_named(op->src[0], "conv_input") && op->ne[1] <= kUbatchMaxTokens &&
                   op->ne[2] <= 3 && op->ne[0] >= 32 && op->ne[0] <= 65536 && op->ne[0] % 32 == 0;
        // per-head attention normalisation and rotation
        case GGML_OP_RMS_NORM:
        case GGML_OP_ROPE:
            return fused_context_is_head_row(op) && fused_context_fed_by_op(op);
        default:
            return false;
    }
}

// The conv's x (qkv_mixed): the source CONCAT(state, TRANSPOSE(x)) transposes into the window, through views.
inline const ggml_tensor * fused_context_conv_x(const ggml_tensor * ssm_conv) {
    const ggml_tensor * concat     = ssm_conv->src[0];
    const ggml_tensor * transposed = concat != nullptr ? concat->src[1] : nullptr;
    return transposed != nullptr && transposed->op == GGML_OP_TRANSPOSE ? fused_context_root(transposed->src[0]) :
                                                                          nullptr;
}

// A fused claim also needs the producers its matcher looks at to stay on this backend: when one goes to
// another backend its output becomes a split input, the matcher no longer sees the pattern, and the
// claimed node is left with no dispatch. At -ub 4096 the 27B qkv projection went to the CPU (the prefill
// matmul matchers stop at 2048 tokens) and the claimed SSM_CONV failed the graph. `supported` is the
// backend's own supports_op.
template <typename Supported>
inline bool fused_context_claim(const ggml_tensor * op, Supported && supported) {
    if (!fused_context_claim(op)) {
        return false;
    }
    if (op->op == GGML_OP_SSM_CONV) {
        const ggml_tensor * x = fused_context_conv_x(op);
        return x != nullptr && supported(x);
    }
    return true;
}

}  // namespace ggml::hrx
