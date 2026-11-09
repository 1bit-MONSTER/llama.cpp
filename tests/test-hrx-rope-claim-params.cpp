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

// Host test for rope-claim-params.h: which rotations the fused registrations accept, and therefore what
// the claim in fused-context-claim.h may assert. No device and no model - the predicates are pure.
//
// The case this exists for is engine #1467: a vision projector's ROPE carries mode
// GGML_ROPE_TYPE_VISION and n_dims 32 against a 64-wide head, which BOTH matchers reject. The claim used
// to accept it on shape alone, so the node was claimed and then refused when the dispatch was built, and
// the whole CLIP graph failed instead of that node running on the CPU.

#include "fused-context-claim.h"
#include "rope-claim-params.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (0)

namespace {

using ggml::hrx::RopeParams;

RopeParams text_rotation(int n_dims, int head_size) {
    RopeParams params;
    params.n_dims      = n_dims;
    params.mode        = GGML_ROPE_TYPE_NORMAL;
    params.freq_base   = 1000000.0f;
    params.freq_scale  = 1.0f;
    params.ext_factor  = 0.0f;
    params.attn_factor = 1.0f;
    (void) head_size;
    return params;
}

// The projector rotation: two-dimensional vision rotary, per ggml.h "GGML_ROPE_TYPE_VISION n_dims = 4 -->
// [yyyyxxxx]". The node #1467 fails on is n_dims 32 against head_size 64 with this mode.
RopeParams vision_rotation() {
    RopeParams params;
    params.n_dims      = 32;
    params.mode        = GGML_ROPE_TYPE_VISION;
    params.freq_base   = 10000.0f;
    params.freq_scale  = 1.0f;
    params.ext_factor  = 0.0f;
    params.attn_factor = 1.0f;
    return params;
}

// The claim itself, on a graph shaped like the one engine #1467 fails on: a ROPE whose source is a VIEW
// (so the head-row and fed-by-an-op rules both hold) over a 64-wide head. Pre-fix the claim accepted the
// vision rotation on shape alone and the node was then refused at dispatch build, which failed the whole
// CLIP graph instead of running that node on the CPU.
static void check_claim_for(bool vision, bool want_claimed) {
    ggml_init_params params = { /*mem_size*/ 24 * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context *  ctx     = ggml_init(params);
    REQUIRE(ctx != nullptr);

    // 3-D, as the claim's fused_context_is_3d requires - the real node in #1467 is [64, 16, 784, 1], i.e. ne[3] == 1.
    ggml_tensor * base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 16, 784);
    // fed-by-an-op is resolved through the view chain to its ROOT (fused_context_producer_op), so the view has
    // to sit on top of a tensor an op produced - a view of a leaf is GGML_OP_NONE and would make the claim false
    // for the wrong reason, which is what an earlier version of this test did.
    ggml_tensor * ones = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 16, 784);
    ggml_tensor * mul  = ggml_mul(ctx, base, ones);
    ggml_tensor * view = ggml_view_3d(ctx, mul, 64, 16, 784, mul->nb[1], mul->nb[2], 0);
    ggml_tensor * pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, vision ? 3136 : 784);
    ggml_tensor * rope = ggml_rope_ext(ctx, view, pos, nullptr, vision ? 32 : 64,
                                       vision ? GGML_ROPE_TYPE_VISION : GGML_ROPE_TYPE_NORMAL, 0, 1000000.0f, 1.0f,
                                       0.0f, 1.0f, 0.0f, 0.0f);

    // the backend the claim asks about: "is the producer supported" - here, yes for everything, so the
    // only thing under test is the ROPE rule itself.
    const auto supported = [](const ggml_tensor * t) { return t != nullptr; };
    const bool claimed   = ggml::hrx::fused_context_claim(rope, supported);
    if (claimed != want_claimed) {
        std::fprintf(stderr, "claim for %s rotation: got %d, wanted %d (engine #1467)\n",
                     vision ? "vision" : "text", (int) claimed, (int) want_claimed);
        std::abort();
    }
    ggml_free(ctx);
}

}  // namespace

int main() {
    using namespace ggml::hrx;

    {
        // A text attention rotation: n_dims equals the head width, mode NORMAL. The attention fusion takes
        // it, and it is the case that broke twice when the claim was narrowed with the SET_ROWS predicate
        // alone (text rotations do not satisfy that one).
        const RopeParams params = text_rotation(64, 64);
        REQUIRE(rope_params_supported_by_attention_fusion(params, 64));
        REQUIRE(rope_params_supported_by_any_fusion(params, 64));
    }
    {
        // A head-width NEOX rotation with fewer dims than the head: the attention fusion declines (mode and
        // n_dims), the ROPE+SET_ROWS registration takes it, so the claim may still assert it.
        RopeParams params = text_rotation(32, 64);
        params.mode       = GGML_ROPE_TYPE_NEOX;
        REQUIRE(!rope_params_supported_by_attention_fusion(params, 64));
        REQUIRE(rope_params_supported_by_rope_set_rows(params, 64));
        REQUIRE(rope_params_supported_by_any_fusion(params, 64));
    }
    {
        // THE REGRESSION: the vision rotation. Neither matcher accepts it, so the claim must not either -
        // this is the assertion that would have caught engine #1467 before an image request crashed.
        const RopeParams params = vision_rotation();
        REQUIRE(!rope_params_supported_by_attention_fusion(params, 64));
        REQUIRE(!rope_params_supported_by_rope_set_rows(params, 64));
        REQUIRE(!rope_params_supported_by_any_fusion(params, 64));
    }
    {
        // The rejection is on the MODE as well as on n_dims: a vision rotation at head width is still not a
        // rotation any fused registration executes.
        RopeParams params = vision_rotation();
        params.n_dims     = 64;
        REQUIRE(!rope_params_supported_by_attention_fusion(params, 64));
        REQUIRE(!rope_params_supported_by_rope_set_rows(params, 64));
        REQUIRE(!rope_params_supported_by_any_fusion(params, 64));
    }
    {
        // n_dims wider than the head is nobody's case.
        RopeParams params = text_rotation(128, 64);
        REQUIRE(!rope_params_supported_by_any_fusion(params, 64));
    }
    {
        // A degenerate freq_base is declined by both, so it cannot be claimed.
        RopeParams params = text_rotation(64, 64);
        params.freq_base  = 0.0f;
        REQUIRE(!rope_params_supported_by_any_fusion(params, 64));
        params.freq_base = 1000000.0f;
        REQUIRE(rope_params_supported_by_any_fusion(params, 64));
    }
    {
        // ext_factor is only the attention fusion's condition: the SET_ROWS path builds its own theta table
        // and takes it. This is the case where the two matchers genuinely differ, so the union is what the
        // claim has to use.
        RopeParams params = text_rotation(32, 64);
        params.mode       = GGML_ROPE_TYPE_NEOX;
        params.ext_factor = 1.0f;
        REQUIRE(!rope_params_supported_by_attention_fusion(params, 64));
        REQUIRE(rope_params_supported_by_rope_set_rows(params, 64));
        REQUIRE(rope_params_supported_by_any_fusion(params, 64));
    }

    // The regression proper: the claim must refuse the vision rotation and keep the text one.
    check_claim_for(/*vision*/ true, /*want_claimed*/ false);
    check_claim_for(/*vision*/ false, /*want_claimed*/ true);

    std::printf("test-hrx-rope-claim-params: all requirements met\n");
    return 0;
}
