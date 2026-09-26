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

// SOFT_MAX (no mask, scale 1), SUM_ROWS, ARGSORT and narrow GET_ROWS on short F32 rows, such
// as a MoE router's, CONT of strided F32 views and broadcast REPEAT: ggml_softmax_rows_f32, ggml_sum_rows_f32,
// ggml_argsort_rows_f32, ggml_get_rows_small_f32 and ggml_copy_strided_f32.
void register_small_rows_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
