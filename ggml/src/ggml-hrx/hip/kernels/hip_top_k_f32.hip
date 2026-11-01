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

// GGML_OP_TOP_K on HRX (engine #679): the k largest values' indices, one workgroup per row.
//
// Why this exists: Qwen3.8-Flash-Next's QSA layers (src/models/qwen4exp.cpp) call ggml_top_k over the pooled
// indexer scores once per full-attention layer per decode step, with n = n_kv (up to 262144) and
// k = indexer_top_k + r - 1 = 2051. HRX had no TOP_K dispatch, so that node ran on the CPU backend -- a
// std::partial_sort per row with an indirect comparator -- and split every full-attention layer's graph
// around it. Measured on the QSA cliff probe: +9.7 ms per decode step the moment n_kv passed 2051, plus
// ~3.8 us per KV token, i.e. the whole 9.70 vs 31.15 tok/s long-context gap of engine #679.
//
// Algorithm: radix select on a monotone uint32 key map of the f32 bit pattern, so unsigned comparison order is
// float order (-0.0 < +0.0, NaNs above +inf). Three levels of 11/11/10 bits narrow to the exact rank-k key;
// a final emit pass writes every index above that key and then enough indices equal to it to reach k. Three
// passes over the row, no global scratch, no host round trip. Output order is unspecified -- ggml's contract
// for TOP_K is the index *set* (tests/test-backend-ops.cpp test_top_k compares sets and even swaps two entries
// in the CPU reference to say so).
//
// Shapes (matcher: hip/dispatch-hip-top-k.cpp): src0 f32 contiguous [n, rows], dst i32 [k, rows], 1 <= k <= n,
// n <= 262144, rows >= 1. One workgroup of 256 lanes per row and an 8 KiB LDS histogram, so a row of 18,816
// elements costs ~75 KiB x 3 reads instead of a serial sort.

#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr uint32_t kTopKBuckets    = 2048u;
constexpr uint32_t kTopKLanes      = 256u;
constexpr uint32_t kTopKBucketBits = 11u;  // per level; the last level uses 10 and masks with `bits`

__device__ __forceinline__ uint32_t topk_key(float value) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    // sign set -> invert every bit (negatives sort below, more negative lower); sign clear -> set the top bit.
    return bits ^ (0x80000000u | (uint32_t) ((int32_t) bits >> 31));
}

extern "C" __global__ void __launch_bounds__(256) hip_top_k_f32(const float * __restrict__ src,
                                                               int32_t * __restrict__ dst, uint32_t column_count,
                                                               uint32_t top_k) {
    __shared__ uint32_t hist[kTopKBuckets];
    __shared__ uint32_t emitted;
    __shared__ uint32_t threshold_key;
    __shared__ uint32_t next_remaining;

    const uint32_t row  = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const float *  base = src + (size_t) row * column_count;
    int32_t *      out  = dst + (size_t) row * top_k;

    // `prefix` holds the high bits already resolved; `remaining` is how many indices still have to be taken
    // from the elements that match `prefix`.
    uint32_t prefix     = 0;
    uint32_t prefix_len = 0;
    uint32_t remaining  = top_k;

    #pragma unroll
    for (uint32_t level = 0; level < 3u; ++level) {
        // 32 bits in three levels: 11, 11, 10.
        const uint32_t bits  = level < 2u ? kTopKBucketBits : (32u - 2u * kTopKBucketBits);
        const uint32_t shift = 32u - prefix_len - bits;
        const uint32_t mask  = (1u << bits) - 1u;

        for (uint32_t b = lane; b < kTopKBuckets; b += kTopKLanes) {
            hist[b] = 0u;
        }
        __syncthreads();

        for (uint32_t i = lane; i < column_count; i += kTopKLanes) {
            const uint32_t key = topk_key(base[i]);
            if (prefix_len == 0u || (key >> (32u - prefix_len)) == prefix) {
                atomicAdd(&hist[(key >> shift) & mask], 1u);
            }
        }
        __syncthreads();

        // Descending scan: the bucket where the running total first reaches `remaining`.
        if (lane == 0u) {
            uint32_t cumulative = 0u;
            uint32_t bucket     = 0u;
            uint32_t take       = remaining;
            for (uint32_t b = kTopKBuckets; b-- > 0u;) {
                const uint32_t count = hist[b];
                if (cumulative + count >= remaining) {
                    bucket = b;
                    take   = remaining - cumulative;
                    break;
                }
                cumulative += count;
            }
            threshold_key  = (prefix << bits) | bucket;
            next_remaining = take;
        }
        __syncthreads();

        prefix      = threshold_key;          // already includes the bits of this level
        prefix_len += bits;
        remaining   = next_remaining;
    }

    if (lane == 0u) {
        emitted = 0u;
    }
    __syncthreads();

    // Everything strictly above the rank-k key, then enough of the equal group to reach k. Order is not part of
    // the contract, and the second loop stops as soon as k slots are taken.
    for (uint32_t i = lane; i < column_count; i += kTopKLanes) {
        if (topk_key(base[i]) > threshold_key) {
            const uint32_t slot = atomicAdd(&emitted, 1u);
            if (slot < top_k) {
                out[slot] = (int32_t) i;
            }
        }
    }
    __syncthreads();
    for (uint32_t i = lane; i < column_count; i += kTopKLanes) {
        if (topk_key(base[i]) == threshold_key) {
            const uint32_t slot = atomicAdd(&emitted, 1u);
            if (slot < top_k) {
                out[slot] = (int32_t) i;
            }
        }
    }
}
