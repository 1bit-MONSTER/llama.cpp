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

// Packed ternary weights for the K-quant decode kernels. A ternary model written as exact Q4_0 (each
// 128-value group of trits t in {-1, 0, 1} with one fp16 scale d becomes four Q4_0 blocks with nibbles
// t + 8 and the same d; engine tools/ternary_to_q4_0.py) is repacked at upload into 68 bytes per 256
// values: d0, d1 (the two group scales), then 64 bytes of 2-bit codes t + 1. Decode then reads 2.125
// instead of 4.5 bits per weight. Opt-in with GGML_HRX_TERNARY_Q4_0=1 (1bit serve sets it for files
// stamped onebit.ternary_q4_0); the upload checks every block and fails rather than change a weight.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ggml::hrx {

inline constexpr const char kTernaryQ40K128Layout[] = "ternary-q4_0-k128-t2";

// The K-quant decode kernels' weight format value for this layout (ops/kquant_decode_f32.loom).
inline constexpr int64_t kTernaryQ40K128FormatValue = 90;

inline bool ternary_q4_0_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_TERNARY_Q4_0");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

// Bytes of the packed layout for a K x rows weight (K a multiple of 256).
inline size_t ternary_q4_0_k128_bytes(int64_t input_size, int64_t rows) {
    return static_cast<size_t>(input_size / 256) * 68u * static_cast<size_t>(rows);
}

}  // namespace ggml::hrx
