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
// Reference codecs for PrismML's PQ2_0 and PTQ1_0 (ggml-prism.h). The encoders and the PTQ1_0 decoder follow
// quantize_row_pq2_0_ref, quantize_row_ptq1_0_ref and dequantize_row_ptq1_0 in PrismML's llama.cpp fork
// (ggml/src/ggml-quants.c, branch prism, 87268f775), which carries this notice:
//
//   MIT License
//
//   Copyright (c) 2023-2026 The ggml authors
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//   SOFTWARE.

#include "ggml-prism.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// ------------------------------------------------------------------------------------------------ PQ2_0

void quantize_row_pq2_0_ref(const float * GGML_RESTRICT x, block_pq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ2_0 == 0);
    const int64_t nb = k / QK_PQ2_0;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PQ2_0; j++) {
            const float a = fabsf(x[i * QK_PQ2_0 + j]);
            if (a > amax) {
                amax = a;
            }
        }
        const float d  = amax;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);
        memset(y[i].qs, 0, sizeof(y[i].qs));

        // round(w / d) clamped to -1..2, stored + 1
        for (int j = 0; j < QK_PQ2_0; ++j) {
            int q = (int) roundf(x[i * QK_PQ2_0 + j] * id) + 1;
            q     = q < 0 ? 0 : (q > 3 ? 3 : q);
            y[i].qs[j / 4] |= (uint8_t) (q << (2 * (j % 4)));
        }
    }
}

void dequantize_row_pq2_0(const block_pq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PQ2_0 == 0);
    const int64_t nb = k / QK_PQ2_0;

    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < QK_PQ2_0; ++j) {
            const int q = (x[i].qs[j / 4] >> (2 * (j % 4))) & 3;
            y[i * QK_PQ2_0 + j] = (float) (q - 1) * d;
        }
    }
}

size_t quantize_pq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row,
                      const float * imatrix) {
    (void) imatrix;
    const size_t row_size = ggml_row_size(GGML_TYPE_PQ2_0, n_per_row);
    char *       qrow     = (char *) dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_pq2_0_ref(src, (block_pq2_0 *) qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

// ------------------------------------------------------------------------------------------------ PTQ1_0

// The qs bytes are filled in stages of 32, 16 and 8 bytes (TQ1_0's 32-then-16 generalised to 24 bytes):
// a stage of c bytes holds 5c values, value n * c + m in trit n of byte m. With 24 bytes only the 16 and
// 8 stages are used.
static const size_t ptq1_0_stages[3] = { 32, 16, 8 };

void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PTQ1_0; j++) {
            amax = MAX(amax, fabsf(x[j]));
        }
        const float d  = amax;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) {
                        const int xi = lroundf(x[m + n * c] * id) + 1;  // -1, 0, 1 -> 0, 1, 2
                        q *= 3;
                        q += xi;
                    }
                    // ceiling division (243 == 3^5)
                    q              = ((uint16_t) q * 256 + (243 - 1)) / 243;
                    y[i].qs[j + m] = q;
                }
                x += 5 * c;
            }
        }
        for (size_t h = 0; h < sizeof(y->qh); ++h) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                const int xi = lroundf(x[h + m * sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // the first value goes to the most significant trit
            q *= 3;
            q          = ((uint16_t) q * 256 + (243 - 1)) / 243;
            y[i].qh[h] = q;
        }
        x += 4 * sizeof(y->qh);
    }
}

void ggml_ptq1_0_trits(const block_ptq1_0 * GGML_RESTRICT x, int8_t * GGML_RESTRICT t) {
    static const uint8_t pow3[6] = { 1, 3, 9, 27, 81, 243 };

    size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
        const size_t c = ptq1_0_stages[s];
        for (; j + c <= sizeof(x->qs); j += c) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < c; ++m) {
                    const uint8_t q  = x->qs[j + m] * pow3[n];
                    const int16_t xi = ((uint16_t) q * 3) >> 8;
                    *t++             = (int8_t) (xi - 1);
                }
            }
        }
    }
    for (size_t n = 0; n < 4; ++n) {
        for (size_t h = 0; h < sizeof(x->qh); ++h) {
            const uint8_t q  = x->qh[h] * pow3[n];
            const int16_t xi = ((uint16_t) q * 3) >> 8;
            *t++             = (int8_t) (xi - 1);
        }
    }
}

void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    int8_t t[QK_PTQ1_0];
    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        ggml_ptq1_0_trits(&x[i], t);
        for (int j = 0; j < QK_PTQ1_0; ++j) {
            y[i * QK_PTQ1_0 + j] = (float) t[j] * d;
        }
    }
}

size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row,
                       const float * imatrix) {
    (void) imatrix;  // the trits come from the weights themselves
    const size_t row_size = ggml_row_size(GGML_TYPE_PTQ1_0, n_per_row);
    char *       qrow     = (char *) dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_ptq1_0_ref(src, (block_ptq1_0 *) qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

// ------------------------------------------------------------------------------------------------ validation

static bool prism_validate_fp16(ggml_half h, size_t i) {
    const uint16_t f = h;
    if ((f & 0x7c00) == 0x7c00) {
        fprintf(stderr, "ggml_validate_row_data: found %s value at block %zu\n", (f & 0x03ff) ? "nan" : "inf", i);
        return false;
    }
    return true;
}

bool ggml_prism_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
    if (type == GGML_TYPE_PQ2_0) {
        const block_pq2_0 * q = (const block_pq2_0 *) data;
        for (size_t i = 0; i < nbytes / sizeof(block_pq2_0); ++i) {
            if (!prism_validate_fp16(q[i].d, i)) {
                return false;
            }
        }
        return true;
    }
    if (type == GGML_TYPE_PTQ1_0) {
        const block_ptq1_0 * q = (const block_ptq1_0 *) data;
        for (size_t i = 0; i < nbytes / sizeof(block_ptq1_0); ++i) {
            if (!prism_validate_fp16(q[i].d, i)) {
                return false;
            }
        }
        return true;
    }
    return false;
}
