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

// CPU backend entry points for PrismML's PQ2_0 and PTQ1_0 (ggml-prism.h).
#pragma once

#include "ggml-prism.h"

#ifdef __cplusplus
extern "C" {
#endif

void quantize_row_pq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_ptq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void ggml_vec_dot_pq2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_ptq1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
