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

// HIP kernels dispatched through HRX. A .hip file under hip/kernels is compiled by amdclang++ at
// build time (one raw code object per GPU target, see ggml-hrx-hip.cmake), embedded in
// libggml-hrx, and loaded with hrx_executable_load_data like a Loom JIT result. A matcher refers
// to a HIP kernel with hip_kernel_ref("name") and fills Dispatch exactly as for a Loom kernel:
//
//   - bindings      -> the kernel's pointer arguments, in order;
//   - launch_parameters -> uint32_t by-value arguments after the pointers (packed as constants);
//   - workload_parameters -> integer parameters the launch function reads to size the grid.
//     They are part of the executable cache key, so keep them to what changes the geometry.
//
// See hip/README.md for the full recipe.

#pragma once

#include "hip/hip-code-objects.h"
#include "kernel-corpus/kernel-corpus.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ggml::hrx {

inline constexpr const char kHipKernelFamily[] = "hip";

constexpr KernelCatalogRef hip_kernel_ref(const char * name) {
    return kernel_catalog_ref(kHipKernelFamily, name);
}

struct HipLaunchGeometry {
    std::array<uint32_t, 3> workgroup_count = { 1, 1, 1 };
    std::array<uint32_t, 3> workgroup_size  = { 1, 1, 1 };
};

// Computes the grid from the dispatch's integer parameters; false rejects the dispatch.
using HipLaunchFunction = bool (*)(const std::map<std::string, int64_t> & parameters, HipLaunchGeometry & geometry);

struct HipKernel {
    const char *                         name        = "";  // extern "C" __global__ symbol
    const char *                         code_object = "";  // stem of the .hip file that defines it
    std::vector<KernelBindingDefinition> bindings;
    std::vector<const char *>            launch_parameters;    // uint32_t arguments after the pointers
    std::vector<const char *>            workload_parameters;  // read by launch
    HipLaunchFunction                    launch = nullptr;
};

// Registers a kernel (idempotent per name). Call it from the matcher's register_* function.
void register_hip_kernel(const HipKernel & kernel);

// The registered kernel for a catalog id, or nullptr (hook in resolve_kernel_definition).
const KernelDefinition * find_hip_kernel_definition(uint64_t kernel_id);

bool is_hip_kernel_definition(const KernelDefinition & definition);

// The launch function of a registered kernel, or nullptr.
HipLaunchFunction find_hip_kernel_launch(uint64_t kernel_id);

}  // namespace ggml::hrx
