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
#include "ggml-quants.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

namespace ggml::hrx {

namespace {

// One 256-value block: four Q4_0 blocks per 128-value group, two groups. Returns false if a nibble is not
// 7, 8 or 9 or a group's four scales differ.
bool pack_block(const block_q4_0 * in, uint8_t * out) {
    uint16_t scales[2];
    std::memset(out + 4, 0, 64);
    for (int group = 0; group < 2; ++group) {
        const block_q4_0 * g = in + 4 * group;
        uint16_t           d;
        std::memcpy(&d, &g[0].d, sizeof(d));
        for (int b = 0; b < 4; ++b) {
            uint16_t db;
            std::memcpy(&db, &g[b].d, sizeof(db));
            if (db != d) {
                return false;
            }
            for (int j = 0; j < QK4_0 / 2; ++j) {
                const int lo = g[b].qs[j] & 0x0F;
                const int hi = g[b].qs[j] >> 4;
                if (lo < 7 || lo > 9 || hi < 7 || hi > 9) {
                    return false;
                }
                // values j and j + 16 of Q4_0 block b: index within the 256-value block
                const int i_lo = 128 * group + 32 * b + j;
                const int i_hi = i_lo + 16;
                out[4 + i_lo / 4] |= static_cast<uint8_t>((lo - 7) << (2 * (i_lo % 4)));
                out[4 + i_hi / 4] |= static_cast<uint8_t>((hi - 7) << (2 * (i_hi % 4)));
            }
        }
        scales[group] = d;
    }
    std::memcpy(out, scales, sizeof(scales));
    return true;
}

}  // namespace

Status materialize_ternary_q4_0_k128(const HostWeightSource & source, std::vector<uint8_t> & output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q4_0 || source.input_size <= 0 || source.input_size % 256 != 0 ||
        source.output_size <= 0) {
        status.log("layout %s requires Q4_0 with K divisible by 256", source.layout.c_str());
        return status;
    }
    static_assert(sizeof(block_q4_0) == 18);
    const size_t blocks    = static_cast<size_t>(source.input_size / 256);
    const size_t rows      = static_cast<size_t>(source.output_size);
    const size_t in_bytes  = rows * blocks * 8 * sizeof(block_q4_0);
    const size_t out_bytes = ternary_q4_0_k128_bytes(source.input_size, source.output_size);
    if (source.length != in_bytes || source.materialized_length != out_bytes) {
        status.log("layout %s has inconsistent source/materialized lengths", source.layout.c_str());
        return status;
    }
    output.resize(out_bytes);
    const auto * input =
        reinterpret_cast<const block_q4_0 *>(static_cast<const uint8_t *>(source.host_data) + source.offset);
    const size_t units = rows * blocks;
    const size_t threads =
        std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), (units + 4095) / 4096));
    std::atomic<bool>        ok{ true };
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (size_t u = units * t / threads; u < units * (t + 1) / threads && ok.load(std::memory_order_relaxed);
                 ++u) {
                if (!pack_block(input + 8 * u, output.data() + 68 * u)) {
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
            "layout %s: weight is not exact ternary Q4_0 (a nibble outside 7..9 or differing scales in a "
            "128-value group); unset GGML_HRX_TERNARY_Q4_0 for this model",
            source.layout.c_str());
    }
    return status;
}

}  // namespace ggml::hrx
