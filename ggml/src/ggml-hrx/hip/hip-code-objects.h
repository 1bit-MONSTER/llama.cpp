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

// HIP code objects embedded at build time (ggml-hrx-hip.cmake): one raw AMDGPU ELF per
// hip/kernels/<stem>.hip and target. No HRX or Loom dependencies, so tools can link it alone.

#pragma once

#include <cstddef>

// A kernel add-on may compile its kernels under symbols other than their readable names (the private gpu-kernels
// add-on uses opaque ids). Fork code that looks an add-on kernel up by name inside a code object translates the
// name through this hook, which such an add-on defines; without one (no add-on, or an add-on that keeps the
// readable names) the name is the symbol. GGML_HRX_HIP_ADDON_KERNEL_SYMBOL_HOOK tells an add-on the hook exists.
#define GGML_HRX_HIP_ADDON_KERNEL_SYMBOL_HOOK 1
extern "C" const char * ggml_hrx_hip_addon_kernel_symbol(const char * name) __attribute__((weak));

namespace ggml::hrx {

// The code-object symbol of the add-on kernel |name| (see ggml_hrx_hip_addon_kernel_symbol above).
inline const char * hip_addon_kernel_symbol(const char * name) {
    return ggml_hrx_hip_addon_kernel_symbol != nullptr ? ggml_hrx_hip_addon_kernel_symbol(name) : name;
}

// Code object bytes for (stem, target), or nullptr; target nullptr = any target.
const void * hip_code_object_data(const char * stem, const char * target, size_t * size);

// True when the build embedded this stem for at least one target.
bool hip_code_object_embedded(const char * stem);

// Short content digest of the stem's first embedded code object, or nullptr.
const char * hip_code_object_digest(const char * stem);

}  // namespace ggml::hrx
