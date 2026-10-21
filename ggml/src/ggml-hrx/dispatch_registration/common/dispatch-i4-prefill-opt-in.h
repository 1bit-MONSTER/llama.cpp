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

namespace ggml::hrx {

// The SwiGLU gate/up matmul on symmetric int4 activations (common.mul_mat_swiglu_q5_projection.symmetric_i4_q8_plane,
// dispatch-gated-mul-mat.cpp) quantizes every prompt activation to 4 bits. On Qwen3.8-27B UD-Q4_K_XL it costs a
// mean KLD of 0.0019 (same top 97.9%) against the exact f16 path at the same settings, in every prompt batch of
// up to 2048 tokens. It is off unless GGML_HRX_SYMMETRIC_I4_PREFILL=1.
bool symmetric_i4_prefill_enabled();

}  // namespace ggml::hrx
