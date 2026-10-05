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

// Example HIP kernel for the HRX HIP recipe (hip/README.md): GGML_OP_SCALE on packed f32,
// y = x * scale + bias. Floats travel as uint32 bit patterns because HRX launch parameters are
// uint32. Matcher: hip/dispatch-hip-scale.cpp (opt-in, GGML_HRX_HIP_EXAMPLE_SCALE=1).

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256)
hip_scale_f32(const float * __restrict__ x, float * __restrict__ y, uint32_t n, uint32_t scale_bits,
              uint32_t bias_bits) {
    const float    scale  = __builtin_bit_cast(float, scale_bits);
    const float    bias   = __builtin_bit_cast(float, bias_bits);
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = x[i] * scale + bias;
    }
}
