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

#include "hip/hip-dispatches.h"

namespace ggml::hrx {
namespace {

HipAttentionSinkHook g_attention_sink_hook = nullptr;

}  // namespace

void register_hip_dispatches(DispatchRegistryBuilder & registry) {
    register_hip_scale_dispatch(registry);
#ifdef GGML_HRX_HIP_ADDON
    ggml_hrx_hip_addon_register(registry);
#endif
}

void set_hip_attention_sink_hook(HipAttentionSinkHook hook) {
    g_attention_sink_hook = hook;
}

bool hip_attention_sink_dispatch(const HipAttentionSinkArgs & args, Dispatch & dispatch) {
    return g_attention_sink_hook != nullptr && g_attention_sink_hook(args, dispatch);
}

}  // namespace ggml::hrx
