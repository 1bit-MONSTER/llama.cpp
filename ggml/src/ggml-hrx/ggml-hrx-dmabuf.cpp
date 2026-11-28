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

#include "ggml-hrx-dmabuf.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>

namespace ggml::hrx {

bool export_dmabuf(void * device_ptr, size_t size, int * fd, size_t * offset) {
    using export_fn = int (*)(const void *, size_t, int *, uint64_t *);
    // libhsa is already loaded by the HRX runtime; look the symbol up in that copy
    static export_fn hsa_export = []() -> export_fn {
        const char * path = getenv("IREE_HAL_AMDGPU_LIBHSA_PATH");
        void * lib = dlopen(path != nullptr ? path : "libhsa-runtime64.so.1", RTLD_NOW | RTLD_NOLOAD);
        return lib != nullptr ? (export_fn) dlsym(lib, "hsa_amd_portable_export_dmabuf") : nullptr;
    }();
    // [1bit] hrx-system v0.3.0 runs on libamdf, not ROCr: no libhsa is in the process, so there is no export to look up.
    // Say so once instead of failing silently (cl-hrxsys STATE: libhrx has no public buffer->dma-buf export yet).
    if (hsa_export == nullptr) {
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "ggml_hrx: dma-buf export unavailable: no ROCr libhsa in this process (hrx-system on "
                            "libamdf has no public buffer->dma-buf export); zero-copy GPU->NPU sharing is off\n");
        }
        return false;
    }
    uint64_t off = 0;
    if (device_ptr == nullptr || hsa_export(device_ptr, size, fd, &off) != 0) {
        return false;
    }
    *offset = (size_t) off;
    return true;
}

}  // namespace ggml::hrx
