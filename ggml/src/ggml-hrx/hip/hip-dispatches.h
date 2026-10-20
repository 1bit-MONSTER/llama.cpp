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

// The one place HIP matchers are registered (hooked from register_common_dispatches), plus the
// entry points a HIP kernel add-on (GGML_HRX_HIP_ADDON_DIR, see README.md) plugs into.

#pragma once

#include "dispatch_registration/dispatch-registry.h"

#include <cstdint>

namespace ggml::hrx {

void register_hip_dispatches(DispatchRegistryBuilder & registry);

// Matchers built from this directory (one line each in hip-dispatches.cpp).
void register_hip_scale_dispatch(DispatchRegistryBuilder & registry);

// Defined by the add-on when GGML_HRX_HIP_ADDON_DIR is set (the build then defines GGML_HRX_HIP_ADDON);
// registers the add-on's kernels and matchers. Called after the matchers above, once per registry build.
void ggml_hrx_hip_addon_register(DispatchRegistryBuilder & registry);

// Attention-sink rescale hook. dispatch-attention-sink.cpp appends the rescale after FlashAttention; it
// has checked the node (query [tokens][heads][d] f32, key [capacity][kv_heads][d] f16, f16 mask rows
// key_count apart, f32 sinks, output [tokens][heads][dv] f32, no input overlapping the output). A hook
// that returns true has filled dispatch and replaces the Loom kernel; false keeps the Loom kernel.
struct HipAttentionSinkArgs {
    const Value * query;
    const Value * key;
    const Value * mask;
    const Value * sinks;
    const Value * output;
    int64_t       tokens;
    int64_t       key_count;
    int64_t       heads;
    int64_t       kv_heads;
    int64_t       qk_head_size;
    int64_t       value_head_size;
    float         scale;
};
using HipAttentionSinkHook = bool (*)(const HipAttentionSinkArgs & args, Dispatch & dispatch);
void set_hip_attention_sink_hook(HipAttentionSinkHook hook);
bool hip_attention_sink_dispatch(const HipAttentionSinkArgs & args, Dispatch & dispatch);

// SET_ROWS write hook. dispatch-rope-set-rows.cpp matches F16/F32 rows and caches itself; a cache format
// outside that table (q8_0: the quantized KV cache) is offered to this hook, which has already checked the
// node (rows [hidden][tokens] f32 or as the matcher allows, indices contiguous I64, cache contiguous and the
// same storage as the in-place output). A hook that returns true has filled dispatch and replaces the Loom
// kernel; false leaves the node unmatched, as before.
struct HipSetRowsArgs {
    const Value * rows;
    const Value * indices;
    const Value * cache;
    const Value * output;
    size_t        rows_span_bytes;
    int64_t       rows_stride;
    int64_t       hidden_size;
    int64_t       token_count;
    int64_t       cache_row_count;
};
using HipSetRowsHook = bool (*)(const HipSetRowsArgs & args, Dispatch & dispatch);
void set_hip_set_rows_hook(HipSetRowsHook hook);
bool hip_set_rows_dispatch(const HipSetRowsArgs & args, Dispatch & dispatch);
bool hip_set_rows_hook_installed();

}  // namespace ggml::hrx
