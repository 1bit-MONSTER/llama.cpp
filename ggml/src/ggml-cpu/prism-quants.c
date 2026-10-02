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
// CPU dot products for PrismML's PQ2_0 and PTQ1_0 (ggml-prism.h) against Q8_0 activations, as PrismML's fork
// pairs them (four Q8_0 blocks per 128-value block). Plain C: the weights are decoded to int8 per block and
// summed per 32-value Q8_0 block, so the compiler can vectorise the inner loops. This is the reference and
// the fallback for these types; the GPU backends carry the fast paths.

#include "prism-quants.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"

#include <assert.h>

void quantize_row_pq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_pq2_0_ref(x, (block_pq2_0 *) y, k);
}

void quantize_row_ptq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_ptq1_0_ref(x, (block_ptq1_0 *) y, k);
}

static inline float prism_dot_q8_0_x4(const int8_t * GGML_RESTRICT q, const block_q8_0 * GGML_RESTRICT y) {
    float sum = 0.0f;
    for (int b = 0; b < 4; ++b) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; ++j) {
            sumi += (int) q[b * QK8_0 + j] * (int) y[b].qs[j];
        }
        sum += GGML_CPU_FP16_TO_FP32(y[b].d) * (float) sumi;
    }
    return sum;
}

void ggml_vec_dot_pq2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_PQ2_0 == 0);
    assert(nrc == 1);
    (void) bs;
    (void) bx;
    (void) by;
    (void) nrc;

    const block_pq2_0 * GGML_RESTRICT x  = (const block_pq2_0 *) vx;
    const block_q8_0 * GGML_RESTRICT  y  = (const block_q8_0 *) vy;
    const int                         nb = n / QK_PQ2_0;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        int8_t q[QK_PQ2_0];
        for (int j = 0; j < QK_PQ2_0 / 4; ++j) {
            const uint8_t b = x[i].qs[j];
            q[4 * j + 0]    = (int8_t) (((b >> 0) & 3) - 1);
            q[4 * j + 1]    = (int8_t) (((b >> 2) & 3) - 1);
            q[4 * j + 2]    = (int8_t) (((b >> 4) & 3) - 1);
            q[4 * j + 3]    = (int8_t) (((b >> 6) & 3) - 1);
        }
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * prism_dot_q8_0_x4(q, y + 4 * i);
    }
    *s = sumf;
}

void ggml_vec_dot_ptq1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_PTQ1_0 == 0);
    assert(nrc == 1);
    (void) bs;
    (void) bx;
    (void) by;
    (void) nrc;

    const block_ptq1_0 * GGML_RESTRICT x  = (const block_ptq1_0 *) vx;
    const block_q8_0 * GGML_RESTRICT   y  = (const block_q8_0 *) vy;
    const int                          nb = n / QK_PTQ1_0;

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        int8_t q[QK_PTQ1_0];
        ggml_ptq1_0_trits(&x[i], q);
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * prism_dot_q8_0_x4(q, y + 4 * i);
    }
    *s = sumf;
}
