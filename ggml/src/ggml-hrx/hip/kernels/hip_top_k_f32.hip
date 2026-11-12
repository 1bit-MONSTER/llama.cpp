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

// r3_d0, vectorised visits. The measured limit at rows=1 is memory-level parallelism, not passes: one pass over a
// 200,000-element row is 781 iterations per lane and costs ~143 ns each, i.e. ~300 cycles -- one un-hidden DRAM
// latency per iteration, because a lane has a single outstanding load and the workgroup is only 4 waves. Reading four
// floats per iteration puts 4x the bytes in flight for the same number of latency round trips, and every pass (the
// three histograms and the emit) gets it, since they all walk the row through the same helper.
__device__ __forceinline__ bool topk_vec4_ok(const float * base, uint32_t n) {
    // The row base already includes the pitch (src + row * column_count), so testing this pointer is enough: a pitch
    // that is not a multiple of 4 leaves the following rows unaligned and they simply take the scalar path.
    return n >= 4u && ((reinterpret_cast<uintptr_t>(base) & 15u) == 0u);
}

// Calls f(index, value) for every element of the row, four at a time when the row is 16-byte aligned, with a scalar
// tail. Index order is ascending either way, so slot numbering and the ties rule are unchanged.
template <typename F>
__device__ __forceinline__ void topk_visit(const float * __restrict__ base, uint32_t n, uint32_t lane, F && f) {
    if (topk_vec4_ok(base, n)) {
        const float4 * b4 = reinterpret_cast<const float4 *>(base);
        const uint32_t n4 = n >> 2;
        for (uint32_t i = lane; i < n4; i += kTopKLanes) {
            const float4 v = b4[i];
            const uint32_t e = i << 2;
            f(e + 0u, v.x);
            f(e + 1u, v.y);
            f(e + 2u, v.z);
            f(e + 3u, v.w);
        }
        for (uint32_t i = (n4 << 2) + lane; i < n; i += kTopKLanes) {
            f(i, base[i]);
        }
    } else {
        for (uint32_t i = lane; i < n; i += kTopKLanes) {
            f(i, base[i]);
        }
    }
}

__device__ __forceinline__ uint32_t topk_key(float value) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    // sign set -> invert every bit (negatives sort below, more negative lower); sign clear -> set the top bit.
    return bits ^ (0x80000000u | (uint32_t) ((int32_t) bits >> 31));
}

extern "C" __global__ void __launch_bounds__(256) hip_top_k_f32(const float * __restrict__ src,
                                                               int32_t * __restrict__ dst, uint32_t column_count,
                                                               uint32_t top_k) {
    __shared__ uint32_t hist[kTopKBuckets];
    __shared__ uint32_t lane_sum[kTopKLanes];
    __shared__ uint32_t bucket_found;
    __shared__ uint32_t take_found;
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
        if (lane == 0u) {
            // Exactly one lane satisfies suffix_incl >= remaining > suffix_excl, because the histogram of matching
            // elements sums to >= remaining, so these are always written. Reset anyway: if that invariant ever failed,
            // a stale `take_found` could make `remaining` exceed `top_k`, and `above_slots = top_k - remaining` would
            // wrap the emit's slot guard into an out-of-bounds write. Two stores per level buy that bound.
            bucket_found = 0u;
            take_found   = remaining;   // remaining <= top_k always, so above_slots cannot wrap
        }
        __syncthreads();

        topk_visit(base, column_count, lane, [&](uint32_t i, float value) {
            const uint32_t key = topk_key(value);
            (void) i;
            if (prefix_len == 0u || (key >> (32u - prefix_len)) == prefix) {
                atomicAdd(&hist[(key >> shift) & mask], 1u);
            }
        });
        __syncthreads();

        // Descending scan, cooperatively (r1_d0). The serial form was 2,048 dependent LDS reads in lane 0 per
        // level -- the measured ~100 us fixed cost. Here each lane owns kPerLane = 8 consecutive buckets, a reverse
        // (suffix) inclusive scan over the 256 per-lane chunk sums is done as an LDS Hillis-Steele tree (8 steps,
        // 2 syncs each), and only the crossing lane walks its own 8 buckets. Same bucket and same residue as the
        // serial scan, ties included: the selected bucket is "the highest b whose inclusive suffix count >=
        // remaining", which is exactly what the serial descending loop finds.
        //
        // Deliberately NOT a warp-shuffle scan: the first version used __shfl_down_sync and returned wrong index sets
        // (ERR 4.0, then 0.667 after a range guard) because shfl's out-of-range behaviour and the wave width are not
        // something this kernel should depend on. The LDS tree is wave-agnostic and costs a few hundred nanoseconds.
        constexpr uint32_t kPerLane = kTopKBuckets / kTopKLanes;   // 8
        const uint32_t b0 = lane * kPerLane;

        uint32_t chunk = 0u;
        #pragma unroll
        for (uint32_t i = 0u; i < kPerLane; ++i) {
            chunk += hist[b0 + i];
        }
        // reverse inclusive scan over lane chunk sums: lane_sum[l] = sum of chunks for lanes [l .. kTopKLanes)
        lane_sum[lane] = chunk;
        __syncthreads();
        for (uint32_t off = 1u; off < kTopKLanes; off <<= 1u) {
            const uint32_t v = (lane + off < kTopKLanes) ? lane_sum[lane + off] : 0u;
            __syncthreads();
            lane_sum[lane] += v;
            __syncthreads();
        }
        const uint32_t suffix_incl = lane_sum[lane];
        const uint32_t suffix_excl = suffix_incl - chunk;
        if (suffix_incl >= remaining && suffix_excl < remaining) {
            uint32_t cumulative = suffix_excl;
            uint32_t bucket     = b0;
            uint32_t take       = remaining - cumulative;
            #pragma unroll
            for (uint32_t i = kPerLane; i-- > 0u;) {
                const uint32_t count = hist[b0 + i];
                if (cumulative + count >= remaining) {
                    bucket = b0 + i;
                    take   = remaining - cumulative;
                    break;
                }
                cumulative += count;
            }
            bucket_found = bucket;
            take_found   = take;
        }
        __syncthreads();
        threshold_key  = (prefix << bits) | bucket_found;
        next_remaining = take_found;
        // engine #2146: every lane must have read bucket_found / take_found (and the shared threshold_key must be
        // settled) before lane 0 resets them at the start of the next level. Without this barrier a fast lane 0 zeroes
        // them under a slow lane, which then publishes a wrong threshold_key: invalid and run-to-run different rows.
        __syncthreads();

        prefix      = threshold_key;          // already includes the bits of this level
        prefix_len += bits;
        remaining   = next_remaining;
    }

    // above_slots is `top_k - remaining` with remaining <= top_k by construction, so this cannot wrap; the guards below
    // are kept as a cheap second line of defence rather than a load-bearing condition.
    const uint32_t above_slots = top_k - remaining;

    // Emit, deterministically (engine #2146). The ">" group goes into [0, above_slots) and the group equal to the
    // rank-k key into [above_slots, top_k), as before. What changed is how a lane finds its slots: an atomicAdd per
    // element made the order differ from run to run, and in the tie group it made the *set* differ too (whichever lanes
    // got there first took the `remaining` slots). On Flash-Next the sparse-attention indexer picks 2051 of 2560 KV
    // rows with this kernel, and identical requests gave different replies. Now each lane counts its two groups, an
    // exclusive scan over lanes gives it fixed offsets, and a second pass writes in visit order. The result is a pure
    // function of the row: same order every run, and ties go to the lowest (lane, visit) positions.
    uint32_t n_gt = 0u, n_eq = 0u;
    topk_visit(base, column_count, lane, [&](uint32_t i, float value) {
        const uint32_t key = topk_key(value);
        (void) i;
        n_gt += key > threshold_key;
        n_eq += key == threshold_key;
    });
    // forward inclusive scan of both counts over lanes, in LDS (hist is free after the levels): [0, 256) and [256, 512)
    __syncthreads();
    hist[lane]              = n_gt;
    hist[kTopKLanes + lane] = n_eq;
    __syncthreads();
    for (uint32_t off = 1u; off < kTopKLanes; off <<= 1u) {
        const uint32_t g = lane >= off ? hist[lane - off] : 0u;
        const uint32_t e = lane >= off ? hist[kTopKLanes + lane - off] : 0u;
        __syncthreads();
        hist[lane] += g;
        hist[kTopKLanes + lane] += e;
        __syncthreads();
    }
    uint32_t slot_gt = hist[lane] - n_gt;
    uint32_t slot_eq = hist[kTopKLanes + lane] - n_eq;
    topk_visit(base, column_count, lane, [&](uint32_t i, float value) {
        const uint32_t key = topk_key(value);
        if (key > threshold_key) {
            if (slot_gt < above_slots) {
                out[slot_gt] = (int32_t) i;
            }
            ++slot_gt;
        } else if (key == threshold_key) {
            if (slot_eq < remaining) {
                out[above_slots + slot_eq] = (int32_t) i;
            }
            ++slot_eq;
        }
    });
}
