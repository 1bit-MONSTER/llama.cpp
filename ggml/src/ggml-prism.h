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
//
// PrismML's group-128 types PQ2_0 (ggml type 142) and PTQ1_0 (ggml type 143), as their llama.cpp fork
// (https://github.com/PrismML-Eng/llama.cpp, branch prism, MIT License, Copyright (c) 2023-2026 The ggml
// authors) defines them, so that their published GGUFs (Ternary-Bonsai-2-*) load as they are. The type ids
// and block layouts must stay identical to theirs; see ggml-prism-quants.c for the codecs.
//
//   PQ2_0  (34 bytes / 128 values, 2.125 bpw): d (fp16), qs[32]. Value j is code c = (qs[j / 4] >> 2 (j % 4)) & 3,
//          w = (c - 1) * d, so c = 0, 1, 2, 3 mean -1, 0, +1, +2 (the same codec as Q2_0, one scale per 128).
//   PTQ1_0 (28 bytes / 128 values, 1.75 bpw): qs[24], qh[2], d (fp16). Trits t in {-1, 0, +1}, w = t * d,
//          stored base 3 as in TQ1_0: a byte b holds trits n = 0..4 (resp. 0..3 for qh) read as
//          ((uint8_t)(b * 3^n) * 3) >> 8, minus 1. Value order: qs[0..15] give values n * 16 + m (byte m,
//          trit n) for 0..79, qs[16..23] give 80 + n * 8 + m, and qh[0..1] give 120 + n * 2 + h.
#pragma once

#include "ggml-quants.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK_PQ2_0 128
typedef struct {
    ggml_half d;               // scale
    uint8_t   qs[QK_PQ2_0 / 4]; // 2-bit codes, value j at bits 2 (j % 4) of byte j / 4
} block_pq2_0;
static_assert(sizeof(block_pq2_0) == sizeof(ggml_half) + QK_PQ2_0 / 4, "wrong pq2_0 block size/padding");

#define QK_PTQ1_0 128
typedef struct {
    uint8_t   qs[(QK_PTQ1_0 - 4 * QK_PTQ1_0 / 64) / 5]; // 24 bytes, 5 trits each -> 120 values
    uint8_t   qh[QK_PTQ1_0 / 64];                       //  2 bytes, 4 trits each ->   8 values
    ggml_half d;                                        // scale
} block_ptq1_0;
static_assert(sizeof(block_ptq1_0) == sizeof(ggml_half) + QK_PTQ1_0 / 64 + (QK_PTQ1_0 - 4 * QK_PTQ1_0 / 64) / 5,
              "wrong ptq1_0 block size/padding");

GGML_API void quantize_row_pq2_0_ref(const float * GGML_RESTRICT x, block_pq2_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_pq2_0(const block_pq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API size_t quantize_pq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// The 128 trits of one PTQ1_0 block in value order (-1, 0, +1).
GGML_API void ggml_ptq1_0_trits(const block_ptq1_0 * GGML_RESTRICT x, int8_t * GGML_RESTRICT t);

// ggml_validate_row_data for the two types: every scale must be a finite fp16.
GGML_API bool ggml_prism_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);

#ifdef __cplusplus
}
#endif
