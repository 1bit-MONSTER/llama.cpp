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
#include <cstdlib>
#include <cstring>

#ifdef GGML_HRX_HIP_ADDON
#    include "hip/hip-code-objects.h"
#endif

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
#ifdef GGML_HRX_HIP_ADDON
    // With the HIP kernel add-on (gpu-kernels addons/hip-hrx) built in, routed-expert MUL_MAT_ID runs on
    // HRX via the add-on's decode + prefill mmid, which carry the real GGUF code books for IQ4_NL / IQ3_S
    // / Q5_1 / Q8_0 (and Q4_K/Q5_K/Q6_K/MXFP4/NVFP4/Q2_0). IQ2_S runs on the add-on prefill GEMM when the
    // code object carries it (mul_mat_id_iq2s_on_hip below), else on the CPU. (The historical IQ4_NL XOR-12 garbage was the Loom shared-f16-dequant path, which the add-on
    // mmid replaces.)
    return type == GGML_TYPE_IQ2_S;
#else
    // Without the add-on, the only HRX MUL_MAT_ID is the Loom WMMA path: it mis-applies the IQ4_NL XOR-12
    // nibble remap (garbage) and has no IQ2_S path, so both stay on CPU.
    return type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ2_S;
#endif
}

#ifdef GGML_HRX_HIP_ADDON
// True when the add-on prefill GEMM code object carries the IQ2_S routed-expert kernels (plain and gate/up
// SwiGLU). Looked up by export name in the embedded code object (translated through hip_addon_kernel_symbol),
// so the answer does not depend on when the dispatch registry is built.
inline bool hip_addon_has_iq2s_prefill() {
    static const bool present = [] {
        for (const char * target : { "gfx1151", "gfx1100" }) {
            size_t       size = 0;
            const void * data = hip_code_object_data("hrx_prefill_gemm", target, &size);
            if (data == nullptr || size == 0) {
                continue;
            }
            const char * bytes = static_cast<const char *>(data);
            bool         all   = true;
            for (const char * kernel : { "hrx_prefill_mmid_iq2_s_m64_w4", "hrx_prefill_mmid_swiglu_iq2_s_m64_w4" }) {
                // the add-on may compile it under another symbol (hip-code-objects.h)
                const char * name  = hip_addon_kernel_symbol(kernel);
                const size_t n     = std::strlen(name) + 1;  // with the terminator: an exact symbol-table entry
                bool         found = false;
                for (size_t i = 0; !found && i + n <= size; ++i) {
                    found = bytes[i] == name[0] && std::memcmp(bytes + i, name, n) == 0;
                }
                all = all && found;
            }
            if (all) {
                return true;
            }
        }
        return false;
    }();
    return present;
}

// IQ2_S routed experts (Qwen3.8-Flash-Next UD-IQ3_XXS gate / up) run on the add-on IQ2_S prefill GEMM when the
// code object has it, at every batch size: the loader probes support with a 512-token MUL_MAT_ID and then places
// the expert weights in HRX memory, where a CPU decode is far slower (UD-IQ3_XXS tg128 12.82 -> 0.64 tok/s with
// one-token batches left on the CPU). GGML_HRX_HIP_IQ2S_MIN_TOKENS (default 1) raises the floor; the add-on
// matcher reads the same variable.
inline int64_t iq2s_min_tokens() {
    static const int64_t value = [] {
        const char * env = std::getenv("GGML_HRX_HIP_IQ2S_MIN_TOKENS");
        const int64_t v  = env == nullptr || *env == '\0' ? 1 : std::atoll(env);
        return v < 1 ? int64_t{ 1 } : v;
    }();
    return value;
}

inline bool mul_mat_id_iq2s_on_hip(const ggml_tensor * op) {
    const ggml_tensor * ids = op->src[2];
    return op->src[0]->ne[0] % 256 == 0 && ids != nullptr && ids->ne[1] >= iq2s_min_tokens() &&
           hip_addon_has_iq2s_prefill();
}
#endif

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
#ifdef GGML_HRX_HIP_ADDON
    if (weight->type == GGML_TYPE_IQ2_S && mul_mat_id_iq2s_on_hip(op)) {
        return false;
    }
#endif
    if (mul_mat_id_weight_type_requires_cpu(weight->type)) {
        return true;
    }
#ifndef GGML_HRX_HIP_ADDON
    // The Loom WMMA MUL_MAT_ID requires input_size % 256 == 0 ("value 640 violates constraint 'mul'"),
    // so a 640-wide contraction cannot compile there and must stay on CPU. The HIP add-on's mmid handles
    // input_size % 32 (and % 256 for the K-quant / IQ3_S formats), so this restriction lifts when it is built.
    if (weight->ne[0] % 256 != 0) {
        return true;
    }
#endif
    return false;
}

}  // namespace ggml::hrx
