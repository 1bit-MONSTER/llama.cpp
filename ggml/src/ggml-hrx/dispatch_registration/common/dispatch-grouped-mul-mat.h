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

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

// MUL_MAT with a batched F16 weight (one matrix per group, no broadcast), such as ZAYA's
// grouped convolution: ggml_grouped_mul_mat_f16_f32.
void register_grouped_mul_mat_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
