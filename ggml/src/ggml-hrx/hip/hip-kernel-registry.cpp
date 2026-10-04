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

#include "hip/hip-kernel-registry.h"

#include "hip/hip-code-objects.h"

#include <cstring>
#include <deque>
#include <mutex>

namespace ggml::hrx {
namespace {

struct RegisteredHipKernel {
    std::string                          name;
    std::string                          code_object;
    std::string                          digest;
    std::vector<KernelBindingDefinition> bindings;
    std::vector<KernelScalarDefinition>  launch_parameters;
    std::vector<KernelScalarDefinition>  workload_parameters;
    HipLaunchFunction                    launch = nullptr;
    KernelDefinition                     definition;
};

std::mutex                      g_mutex;
std::deque<RegisteredHipKernel> g_kernels;  // deque: definitions keep their addresses

const RegisteredHipKernel * find_registered(uint64_t kernel_id) {
    for (const RegisteredHipKernel & kernel : g_kernels) {
        if (kernel.definition.id == kernel_id) {
            return &kernel;
        }
    }
    return nullptr;
}

}  // namespace

void register_hip_kernel(const HipKernel & kernel) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const uint64_t              id = hip_kernel_ref(kernel.name).id;
    if (find_registered(id) != nullptr) {
        return;
    }
    RegisteredHipKernel & entry = g_kernels.emplace_back();
    entry.name                  = kernel.name;
    entry.code_object           = kernel.code_object;
    // The digest of any target's code object for this stem identifies the build in cache keys.
    const char * digest = hip_code_object_digest(kernel.code_object);
    entry.digest        = digest != nullptr ? digest : "missing";
    entry.bindings                       = kernel.bindings;
    for (const char * name : kernel.launch_parameters) {
        entry.launch_parameters.push_back({ name, "index" });
    }
    for (const char * name : kernel.workload_parameters) {
        entry.workload_parameters.push_back({ name, "index" });
    }
    entry.launch = kernel.launch;

    KernelDefinition & definition   = entry.definition;
    definition.family               = kHipKernelFamily;
    definition.name                 = entry.name.c_str();
    definition.id                   = id;
    definition.source               = entry.code_object.c_str();
    definition.symbol               = entry.name.c_str();
    definition.backend              = "amdgpu";
    definition.source_digest        = entry.digest.c_str();
    definition.bindings             = { entry.bindings.data(), entry.bindings.size() };
    definition.launch_parameters    = { entry.launch_parameters.data(), entry.launch_parameters.size() };
    definition.workload_parameters  = { entry.workload_parameters.data(), entry.workload_parameters.size() };
    definition.compile_recipe.mode  = "hip";
}

const KernelDefinition * find_hip_kernel_definition(uint64_t kernel_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const RegisteredHipKernel * kernel = find_registered(kernel_id);
    return kernel != nullptr ? &kernel->definition : nullptr;
}

bool is_hip_kernel_definition(const KernelDefinition & definition) {
    return definition.compile_recipe.mode != nullptr && std::strcmp(definition.compile_recipe.mode, "hip") == 0;
}

HipLaunchFunction find_hip_kernel_launch(uint64_t kernel_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const RegisteredHipKernel * kernel = find_registered(kernel_id);
    return kernel != nullptr ? kernel->launch : nullptr;
}

}  // namespace ggml::hrx
