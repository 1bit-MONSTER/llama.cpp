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

// Smoke kernels for the HRX HIP path (ggml-hrx-hip-smoke): pointer + by-value arguments,
// blockDim/gridDim (hidden kernargs), LDS and a wave32 shuffle reduction.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256)
hip_smoke_axpy(const float * __restrict__ x, float * __restrict__ y, uint32_t n, float a) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// out[b] = sum of x over block b (256 threads per block, grid = gridDim.x blocks).
extern "C" __global__ void __launch_bounds__(256)
hip_smoke_block_sum(const float * __restrict__ x, float * __restrict__ out, uint32_t n) {
    __shared__ float partial[256 / 32];
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float          v = i < n ? x[i] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, 32);
    }
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x / 32] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (uint32_t w = 0; w < blockDim.x / 32; ++w) {
            s += partial[w];
        }
        out[blockIdx.x] = s + (gridDim.x == 0 ? 1.0f : 0.0f);
    }
}
