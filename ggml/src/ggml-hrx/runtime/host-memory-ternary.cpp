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

#include "runtime/host-memory-ternary.h"

#include "dispatch/ternary-q4-0.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <cmath>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

namespace ggml::hrx {

namespace {

// One 256-value block: four Q4_0 blocks per 128-value group, two groups. Returns false if a nibble is not
// 7, 8 or 9 or the blocks of a group that hold a nonzero value disagree on the scale. An all-zero block
// (every nibble 8) decodes to zeros whatever its scale, so it does not take part in that check; a group
// with no nonzero value gets scale 0.
bool pack_block(const block_q4_0 * in, uint8_t * out) {
    uint16_t scales[2] = { 0, 0 };
    std::memset(out + 4, 0, 64);
    for (int group = 0; group < 2; ++group) {
        const block_q4_0 * g          = in + 4 * group;
        bool               have_scale = false;
        for (int b = 0; b < 4; ++b) {
            bool nonzero = false;
            for (int j = 0; j < QK4_0 / 2; ++j) {
                const int lo = g[b].qs[j] & 0x0F;
                const int hi = g[b].qs[j] >> 4;
                if (lo < 7 || lo > 9 || hi < 7 || hi > 9) {
                    return false;
                }
                nonzero        = nonzero || lo != 8 || hi != 8;
                // values j and j + 16 of Q4_0 block b: index within the 256-value block
                const int i_lo = 128 * group + 32 * b + j;
                const int i_hi = i_lo + 16;
                out[4 + i_lo / 4] |= static_cast<uint8_t>((lo - 7) << (2 * (i_lo % 4)));
                out[4 + i_hi / 4] |= static_cast<uint8_t>((hi - 7) << (2 * (i_hi % 4)));
            }
            if (!nonzero) {
                continue;
            }
            uint16_t d;
            std::memcpy(&d, &g[b].d, sizeof(d));
            if (have_scale && d != scales[group]) {
                return false;
            }
            scales[group] = d;
            have_scale    = true;
        }
    }
    std::memcpy(out, scales, sizeof(scales));
    return true;
}

// One 256-value TQ2_0 block (ternary {-d, 0, +d}, one fp16 scale d for the whole block) -> the same 68-byte
// k128 layout as pack_block: d0, d1 (both the block's d), then 64 bytes of 2-bit codes t + 1. The values are
// decoded with ggml's canonical TQ2_0 dequantizer (type_traits to_float), so the k128 codes reproduce the
// GGUF weights bit for bit (same trit, same d). Returns false if a value is not a trit of this d (a TQ2_0
// code outside {-1, 0, +1} after scaling) -- a genuine BitNet/ternary TQ2_0 never trips this. A 128-value
// group with no nonzero value gets scale 0 (its codes are all 1 and decode to 0 whatever the scale), matching
// pack_block.
bool pack_block_tq2_0(const ggml_type_traits * traits, const block_tq2_0 * in, uint8_t * out) {
    float values[256];
    traits->to_float(in, values, 256);
    const float d = ggml_fp16_to_fp32(in->d);
    uint16_t dh;
    std::memcpy(&dh, &in->d, sizeof(dh));
    std::memset(out + 4, 0, 64);
    bool nonzero[2] = { false, false };
    for (int i = 0; i < 256; ++i) {
        int trit = 0;
        if (d != 0.0f) {
            const float scaled = values[i] / d;
            trit = static_cast<int>(std::lround(scaled));
            if (trit < -1 || trit > 1 || std::fabs(scaled - static_cast<float>(trit)) > 1e-3f) {
                return false;
            }
        }
        out[4 + i / 4] |= static_cast<uint8_t>((trit + 1) << (2 * (i % 4)));
        if (trit != 0) {
            nonzero[i / 128] = true;
        }
    }
    const uint16_t scales[2] = { static_cast<uint16_t>(nonzero[0] ? dh : 0),
                                 static_cast<uint16_t>(nonzero[1] ? dh : 0) };
    std::memcpy(out, scales, sizeof(scales));
    return true;
}

}  // namespace

Status materialize_ternary_q4_0_k128(const HostWeightSource & source, std::vector<uint8_t> & output) {
    Status status;
    const bool is_q4_0  = source.source_type == GGML_TYPE_Q4_0;
    const bool is_tq2_0 = source.source_type == GGML_TYPE_TQ2_0;
    if ((!is_q4_0 && !is_tq2_0) || source.input_size <= 0 || source.input_size % 256 != 0 ||
        source.output_size <= 0) {
        status.log("layout %s requires Q4_0 or TQ2_0 with K divisible by 256", source.layout.c_str());
        return status;
    }
    static_assert(sizeof(block_q4_0) == 18);
    static_assert(sizeof(block_tq2_0) == 66);
    const size_t blocks    = static_cast<size_t>(source.input_size / 256);
    const size_t rows      = static_cast<size_t>(source.output_size);
    const size_t in_bytes  = rows * blocks * (is_q4_0 ? 8 * sizeof(block_q4_0) : sizeof(block_tq2_0));
    const size_t out_bytes = ternary_q4_0_k128_bytes(source.input_size, source.output_size);
    if (source.length != in_bytes || source.materialized_length != out_bytes) {
        status.log("layout %s has inconsistent source/materialized lengths", source.layout.c_str());
        return status;
    }
    output.resize(out_bytes);
    const uint8_t * input_base = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const ggml_type_traits * tq2_0_traits = is_tq2_0 ? ggml_get_type_traits(GGML_TYPE_TQ2_0) : nullptr;
    const size_t units = rows * blocks;
    const size_t threads =
        std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), (units + 4095) / 4096));
    std::atomic<bool>        ok{ true };
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (size_t u = units * t / threads; u < units * (t + 1) / threads && ok.load(std::memory_order_relaxed);
                 ++u) {
                const bool unit_ok = is_q4_0
                    ? pack_block(reinterpret_cast<const block_q4_0 *>(input_base) + 8 * u, output.data() + 68 * u)
                    : pack_block_tq2_0(tq2_0_traits, reinterpret_cast<const block_tq2_0 *>(input_base) + u,
                                       output.data() + 68 * u);
                if (!unit_ok) {
                    ok.store(false, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread & thread : pool) {
        thread.join();
    }
    if (!ok.load()) {
        output.clear();
        status.log(
            "layout %s: weight is not exact ternary (a Q4_0 nibble outside 7..9 / differing scales in a "
            "128-value group, or a TQ2_0 code outside {-1,0,+1}); for Q4_0 unset GGML_HRX_TERNARY_Q4_0",
            source.layout.c_str());
    }
    return status;
}

}  // namespace ggml::hrx
