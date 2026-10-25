// Copyright 2026 bong-water-water-bong
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
//
// HRX support predicates for the Qwen3.8-Flash-Next (llama.cpp arch qwen4exp)
// decode path:
//   * Interleaved multi-section RoPE (GGML_ROPE_TYPE_IMROPE) collapse to NEOX.
//   * Routed-expert MUL_MAT_ID weight types that have no correct HRX kernel.
// All logic here is ours; AMD corpus/matcher files only call these predicates.

#pragma once

#include "ggml.h"

#include "graph/op-params.h"

#include <cstdint>

namespace ggml::hrx {

// Qwen3.8-Flash-Next drives Q/K RoPE with GGML_ROPE_TYPE_IMROPE and a 4-section
// position layout (p_t, p_h, p_w, p_e). For text decoding llama.cpp sets
// p_t == p_h == p_w == the token position and p_e == 0
// (llm_graph_input_pos::set_input). ggml_mrope_cache_init selects, per rotated
// dimension pair, one of {p_t, p_h, p_w, p_e} from the section layout. When no
// pair ever selects the 4th component (p_e), every selected component carries the
// shared token position, so the interleaved rotation is numerically identical to
// a plain NEOX rotation driven by that position -- which the generic rope_f32
// kernel already computes bit-exactly. This mirrors ggml_mrope_cache_init's
// interleaved selection and reports whether that collapse holds, decided purely
// from the compile-time section configuration.
inline bool imrope_collapses_to_neox(const RopeParams & params, int64_t n_dims) {
    const int sections[4] = { params.sections[0], params.sections[1], params.sections[2], params.sections[3] };
    const int sect_dims   = sections[0] + sections[1] + sections[2] + sections[3];
    if (sect_dims <= 0 || static_cast<int64_t>(sect_dims) > n_dims / 2) {
        return false;
    }
    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
        const int sector = static_cast<int>((i0 / 2) % sect_dims);
        const bool selects_t_h_w = (sector % 3 == 1 && sector < 3 * sections[1]) ||
                                   (sector % 3 == 2 && sector < 3 * sections[2]) ||
                                   (sector % 3 == 0 && sector < 3 * sections[0]);
        if (!selects_t_h_w) {
            return false;  // this pair would read p_e, breaking the NEOX equivalence
        }
    }
    return true;
}

// True when a ROPE node is a Qwen3.8-Flash-Next IMROPE that collapses to NEOX and
// may therefore be executed by the generic NEOX rope_f32 kernel.
inline bool rope_mode_is_collapsible_imrope(const RopeParams & params) {
    return params.mode == GGML_ROPE_TYPE_IMROPE && imrope_collapses_to_neox(params, params.n_dims);
}

// Expected positions ne0 for a supported ROPE node. IMROPE carries four position
// sections of token_count each; the collapse path reads only the first (p_t).
inline int64_t rope_expected_positions_ne0(const RopeParams & params, int64_t token_count) {
    return params.mode == GGML_ROPE_TYPE_IMROPE ? token_count * 4 : token_count;
}

// Mode value handed to the rope_f32 / rope_set_rows_f32 kernels. A collapsible
// IMROPE runs the kernel's NEOX path.
inline int rope_kernel_mode(const RopeParams & params) {
    return params.mode == GGML_ROPE_TYPE_IMROPE ? GGML_ROPE_TYPE_NEOX : params.mode;
}

// Routed-expert weight types that the HRX MUL_MAT_ID kernels cannot execute
// correctly, so the scheduler must keep the expert GEMM on CPU:
//   * IQ4_NL  -- the shared f16 dequantizer mis-applies the 4-bit XOR-12 nibble
//                remap, producing garbage rather than slow-but-correct output.
//   * IQ2_S   -- no HRX MUL_MAT_ID path dequantizes IQ2_S at all.
// Verified by real-model KLD/PPL (test-backend-ops does not exercise MUL_MAT_ID).
inline bool mul_mat_id_weight_type_requires_cpu(ggml_type type) {
    return type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ2_S;
}

// Whole-node decision for whether a MUL_MAT_ID must stay on CPU. In addition to
// the correctness-broken weight types above, the HRX WMMA MUL_MAT_ID kernels
// require the per-expert contraction dimension (src0->ne[0]) to be a multiple of
// 256 (the ggml.mul_mat_id.input_size 'mul' constraint). Qwen3.8-Flash-Next routes
// its Q8_0 expert down-projection with a 640-wide contraction, which no HRX
// MUL_MAT_ID kernel can compile, so it must run on CPU.
inline bool mul_mat_id_requires_cpu(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT_ID || op->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * weight = op->src[0];
    return mul_mat_id_weight_type_requires_cpu(weight->type) || (weight->ne[0] % 256 != 0);
}

}  // namespace ggml::hrx
