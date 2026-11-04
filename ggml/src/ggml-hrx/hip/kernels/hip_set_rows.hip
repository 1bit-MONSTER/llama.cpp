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

// Standalone GGML_OP_SET_ROWS for HRX (engine #679, P2): cache[indices[i], :] = rows[i, :].
//
// Why this exists: Qwen3.8-Flash-Next's QSA layers rebuild a full-length attention mask once per full-attention
// layer per decode step (src/models/qwen4exp.cpp build_attn_qsa): fill(-inf) over [1, n_kv], then ggml_set_rows
// scatters the 2051 selected cells, then the causal mask is added. HRX had no SET_ROWS kernel for that shape --
// the Loom ones are the f16 element-wise scatter (i64 indices, f16 cache) and the vectorised 2-D writer (i64
// indices, hidden >= 4, <= 2048 tokens), and a quantized cache is the add-on's -- so the node ran on the CPU
// backend and every full-attention layer's graph was split around it. Measured in the model-free QSA probe
// (~/wt/hrx-probe/qsa_layer.cpp): the stage fails with "unsupported HRX node: SET_ROWS output=f32[1,18816]",
// which is the only node of the QSA decode path that cannot run on the device (indexer 0.60 ms + top_k 0.074 ms
// + FA 0.81 ms are all on device, against ~5.9 ms/layer the model's own slope implies at 18.8k).
//
// Shapes (matcher: hip/dispatch-hip-set_rows.cpp): rows f32 or f16 [hidden, tokens] with packed rows, indices
// i32 or i64 [tokens], cache/dst f32 or f16 [hidden, cache_rows] with packed rows, hidden >= 1, tokens >= 1. The
// matcher slices dims 2 and 3 into one dispatch each (batch strides are byte offsets on the bindings), so this
// kernel only ever sees a 2-D slice.
// The QSA mask case is hidden = 1, tokens = 2051, cache_rows = n_kv, i32 indices, f32 in and out; f16 rows are
// widened when the cache is f32, exactly as the CPU reference does (ggml-cpu/ops.cpp set_rows: from_float over a
// widened row).
//
// Geometry: one thread per (token, packet) pair, packet_width elements per thread (4 when both rows are 16-byte
// aligned and hidden is a multiple of 4, else 1), so the element-wise hidden = 1 case costs one thread per
// selected cell instead of a workgroup. Out-of-range indices are skipped rather than written (the CPU reference
// asserts; a device kernel must not fault), and the index arithmetic is done in 64-bit before the range check.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr uint32_t kSetRowsFormatF32 = 0u;
constexpr uint32_t kSetRowsFormatF16 = 1u;
constexpr uint32_t kSetRowsIndexI32  = 0u;
constexpr uint32_t kSetRowsIndexI64  = 1u;

static __device__ __forceinline__ float set_rows_load(const void * __restrict__ rows, const uint32_t row_format,
                                                     const size_t offset) {
    if (row_format == kSetRowsFormatF32) {
        return static_cast<const float *>(rows)[offset];
    }
    return __half2float(static_cast<const __half *>(rows)[offset]);
}

static __device__ __forceinline__ void set_rows_store(void * __restrict__ cache, const uint32_t cache_format,
                                                     const size_t offset, const float value) {
    if (cache_format == kSetRowsFormatF32) {
        static_cast<float *>(cache)[offset] = value;
    } else {
        static_cast<__half *>(cache)[offset] = __float2half(value);
    }
}

extern "C" __global__ void __launch_bounds__(256) hip_set_rows(const void * __restrict__ rows,
                                                             const void * __restrict__ indices,
                                                             void * __restrict__ cache, uint32_t token_count,
                                                             uint32_t packets, uint32_t packet_width,
                                                             uint32_t hidden_size, uint32_t cache_row_count,
                                                             uint32_t row_format, uint32_t index_format,
                                                             uint32_t cache_format) {
    const uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= token_count * packets) {
        return;
    }

    // flat over (token, packet) so the element-wise scatter (packets == 1, hidden_size == 1) and the wide writer
    // (packets == hidden_size / packet_width) share one grid
    const uint32_t token  = work / packets;
    const uint32_t packet = work - token * packets;

    int64_t cache_row = 0;
    if (index_format == kSetRowsIndexI32) {
        cache_row = static_cast<const int32_t *>(indices)[token];
    } else {
        cache_row = static_cast<const int64_t *>(indices)[token];
    }
    if (cache_row < 0 || cache_row >= static_cast<int64_t>(cache_row_count)) {
        return;
    }

    const size_t channel       = static_cast<size_t>(packet) * packet_width;
    const size_t source_offset = static_cast<size_t>(token) * hidden_size + channel;
    const size_t cache_offset  = static_cast<size_t>(cache_row) * hidden_size + channel;

    // packet_width is at most 4 and the loop bound is a launch parameter, so this unrolls into one vector load
    // and one vector store per thread when the matcher picked the aligned path
    for (uint32_t k = 0; k < packet_width; ++k) {
        const float value = set_rows_load(rows, row_format, source_offset + k);
        set_rows_store(cache, cache_format, cache_offset + k, value);
    }
}
