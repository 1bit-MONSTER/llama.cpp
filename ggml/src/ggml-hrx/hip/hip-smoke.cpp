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

// ggml-hrx-hip-smoke: loads the embedded hip_smoke code object through HRX (no HIP runtime),
// dispatches both kernels and checks the results on the host. Exit 0 = pass.

#include "hip/hip-code-objects.h"
#include "hrx_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(expr)                                                                   \
    do {                                                                              \
        hrx_status_t _s = (expr);                                                     \
        if (!hrx_status_is_ok(_s)) {                                                  \
            char * _m = nullptr;                                                      \
            size_t _n = 0;                                                            \
            hrx_status_to_string(_s, &_m, &_n);                                       \
            std::fprintf(stderr, "FAIL %s: %.*s\n", #expr, (int) _n, _m ? _m : "");   \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

int main() {
    CHECK(hrx_gpu_initialize(0));
    hrx_device_t device = nullptr;
    CHECK(hrx_gpu_device_get(0, &device));
    char arch[64] = {};
    CHECK(hrx_device_get_property(device, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof(arch)));
    size_t       size = 0;
    const void * data = ggml::hrx::hip_code_object_data("hip_smoke", arch, &size);
    if (data == nullptr) {
        std::fprintf(stderr, "FAIL no hip_smoke code object for %s\n", arch);
        return 1;
    }
    hrx_executable_t executable = nullptr;
    CHECK(hrx_executable_load_data(device, data, size, "amdgpu", arch, &executable));

    uint32_t axpy = 0, block_sum = 0;
    CHECK(hrx_executable_lookup_export_by_name(executable, "hip_smoke_axpy", &axpy));
    CHECK(hrx_executable_lookup_export_by_name(executable, "hip_smoke_block_sum", &block_sum));
    for (uint32_t ordinal : { axpy, block_sum }) {
        hrx_executable_export_info_t info = {};
        CHECK(hrx_executable_export_info(executable, ordinal, &info));
        std::printf("export %s: bindings=%u constants=%uB parameters=%u\n", info.name, info.binding_count,
                    info.constant_byte_length, info.parameter_count);
    }

    hrx_stream_t stream = nullptr;
    CHECK(hrx_stream_create(device, 0, &stream));
    const uint32_t n = 1000003, blocks = (n + 255) / 256;
    hrx_buffer_t   x = nullptr, y = nullptr, out = nullptr;
    const hrx_memory_type_t mem = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
    CHECK(hrx_buffer_allocate(stream, n * sizeof(float), mem, HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &x));
    CHECK(hrx_buffer_allocate(stream, n * sizeof(float), mem, HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &y));
    CHECK(hrx_buffer_allocate(stream, blocks * sizeof(float), mem, HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED, &out));
    CHECK(hrx_stream_synchronize(stream));
    float * px = nullptr;
    float * py = nullptr;
    CHECK(hrx_buffer_map(x, HRX_MAP_WRITE, 0, n * sizeof(float), (void **) &px));
    CHECK(hrx_buffer_map(y, HRX_MAP_WRITE, 0, n * sizeof(float), (void **) &py));
    for (uint32_t i = 0; i < n; ++i) {
        px[i] = (float) (i % 97) * 0.25f;
        py[i] = (float) (i % 13);
    }
    CHECK(hrx_buffer_unmap(x));
    CHECK(hrx_buffer_unmap(y));

    struct {
        uint32_t n;
        float    a;
    } axpy_constants = { n, 3.0f };
    hrx_buffer_ref_t      axpy_refs[2] = { { x, 0, n * sizeof(float) }, { y, 0, n * sizeof(float) } };
    hrx_dispatch_config_t config       = { { blocks, 1, 1 }, { 256, 1, 1 }, 32 };
    CHECK(hrx_stream_dispatch(stream, executable, axpy, &config, &axpy_constants, sizeof(axpy_constants), axpy_refs, 2, 0));
    uint32_t         sum_constants = n;
    hrx_buffer_ref_t sum_refs[2]   = { { y, 0, n * sizeof(float) }, { out, 0, blocks * sizeof(float) } };
    CHECK(hrx_stream_dispatch(stream, executable, block_sum, &config, &sum_constants, sizeof(sum_constants), sum_refs, 2, 0));
    CHECK(hrx_stream_synchronize(stream));

    CHECK(hrx_buffer_map(y, HRX_MAP_READ, 0, n * sizeof(float), (void **) &py));
    float * pout = nullptr;
    CHECK(hrx_buffer_map(out, HRX_MAP_READ, 0, blocks * sizeof(float), (void **) &pout));
    size_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const float want = 3.0f * ((float) (i % 97) * 0.25f) + (float) (i % 13);
        bad += py[i] != want;
    }
    double max_err = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
        double want = 0;
        for (uint32_t i = b * 256; i < std::min(n, (b + 1) * 256); ++i) {
            want += 3.0 * ((i % 97) * 0.25) + (i % 13);
        }
        max_err = std::fmax(max_err, std::fabs(want - pout[b]) / std::fmax(1.0, std::fabs(want)));
    }
    CHECK(hrx_buffer_unmap(y));
    CHECK(hrx_buffer_unmap(out));
    std::printf("arch %s: axpy mismatches %zu/%u, block_sum max rel err %.3g over %u blocks\n", arch, bad, n, max_err,
                blocks);
    hrx_buffer_release(x);
    hrx_buffer_release(y);
    hrx_buffer_release(out);
    hrx_stream_release(stream);
    hrx_executable_release(executable);
    const bool pass = bad == 0 && max_err < 1e-5;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
