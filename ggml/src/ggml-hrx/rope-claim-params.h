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

// Which ROPE parameter sets can a fused registration actually execute?
//
// The claim in `fused-context-claim.h` says "this node will run inside a fused dispatch", and
// `device_supports_op` uses that to keep the node on this backend. If the claim is broader than the
// matchers, the node is claimed and then refused when the dispatch is built, and the whole graph
// fails instead of the node falling back to the CPU - engine #1467: a vision projector's ROPE
// (`n_dims = 32`, `mode = GGML_ROPE_TYPE_VISION`) was claimed by the head-row rule and then rejected
// by both matchers, so `clip_image_batch_encode` could not compute at all on HRX0.
//
// Two matchers accept rotations and their parameter sets differ:
//
//   * the attention fusion (llm/dispatch-attention-qkv.cpp): n_dims == head_size, mode NORMAL;
//   * the ROPE + SET_ROWS registration (common/dispatch-rope-set-rows.cpp): 4 <= n_dims <= head_size,
//     n_dims a multiple of 4, mode NORMAL / NEOX / a collapsible IMROPE.
//
// Their union is what the claim may assert. Both predicates live here, the two matchers call them and
// the claim calls the union, so a matcher can be narrowed without the claim silently going stale -
// which is how the two earlier attempts at this fix broke text decode: they re-derived the condition
// and used only the SET_ROWS one, which text rotations do not satisfy (they run inside the attention
// fusion).

#include "graph/op-params.h"
#include "qwen4exp-hrx-support.h"  // rope_mode_is_collapsible_imrope

#include <cmath>

namespace ggml::hrx {

// llm/dispatch-attention-qkv.cpp: the per-head rotation consumed by the attention fusions.
inline bool rope_params_supported_by_attention_fusion(const RopeParams & params, int64_t head_size) {
    // AMD's rebase added the n_offs guard to the matcher's inlined copy; it belongs HERE, in the predicate the
    // claim shares with the matcher, so the claim cannot assert a rotation the matcher now declines (engine#1467:
    // a claim broader than its matcher says yes and the node is then refused at dispatch build).
    if (params.n_offs != 0) { return false; }
    return params.n_dims == head_size && params.mode == GGML_ROPE_TYPE_NORMAL && std::isfinite(params.freq_base) &&
           params.freq_base > 0.0f && std::isfinite(params.freq_scale) && params.freq_scale > 0.0f &&
           params.ext_factor == 0.0f && params.attn_factor == 1.0f;
}

// common/dispatch-rope-set-rows.cpp: the rotation that registration builds a theta table for.
inline bool rope_params_supported_by_rope_set_rows(const RopeParams & params, int64_t head_size) {
    // the same n_offs guard as the attention predicate above: neither rope kernel on this line applies an offset
    if (params.n_offs != 0) { return false; }
    const bool supported_mode = params.mode == GGML_ROPE_TYPE_NORMAL || params.mode == GGML_ROPE_TYPE_NEOX ||
                                rope_mode_is_collapsible_imrope(params) || rope_mode_is_sectioned_imrope(params);
    return params.n_dims >= 4 && params.n_dims <= head_size && params.n_dims % 4 == 0 && supported_mode &&
           std::isfinite(params.freq_base) && params.freq_base > 0.0f && std::isfinite(params.freq_scale) &&
           params.freq_scale > 0.0f && std::isfinite(params.ext_factor) && std::isfinite(params.attn_factor);
}

// What the claim may assert: at least one fused registration accepts these parameters. Anything else
// (a vision rotation, a mode nobody implements) must fall back to the CPU rather than be claimed.
inline bool rope_params_supported_by_any_fusion(const RopeParams & params, int64_t head_size) {
    return rope_params_supported_by_attention_fusion(params, head_size) ||
           rope_params_supported_by_rope_set_rows(params, head_size);
}

}  // namespace ggml::hrx
