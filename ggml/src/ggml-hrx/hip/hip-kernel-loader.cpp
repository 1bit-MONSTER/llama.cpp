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

#include "hip/hip-kernel-loader.h"

#include "ggml-impl.h"
#include "hip/hip-code-objects.h"
#include "hip/hip-kernel-registry.h"
#include "hrx_runtime.h"

#include <cstring>

namespace ggml::hrx {

LoomCompiledKernelRef make_hip_compiled_kernel(const std::string &      key,
                                               const KernelDefinition & definition,
                                               const Dispatch &         dispatch,
                                               const char *             target) {
    auto                             compiled_ref = std::make_shared<LoomCompiledKernel>(key, LoomKernelCompileRequest{});
    ggml_hrx_loom_jit_compile_result result;
    std::string                      error;

    const HipLaunchFunction launch = find_hip_kernel_launch(definition.id);
    size_t       size = 0;
    const void * data = hip_code_object_data(definition.source, target, &size);
    HipLaunchGeometry geometry;
    if (launch == nullptr) {
        error = "HIP kernel " + kernel_definition_name(definition) + " is not registered";
    } else if (data == nullptr) {
        error = std::string("no ") + (target != nullptr ? target : "?") + " code object '" + definition.source +
                "' embedded for HIP kernel " + kernel_definition_name(definition);
    } else if (!launch(dispatch.kernel.integer_parameters, geometry)) {
        error = "HIP kernel " + kernel_definition_name(definition) + " rejected its launch parameters";
    } else {
        void *       copy   = nullptr;
        hrx_status_t status = hrx_host_allocator_malloc_uninitialized(hrx_host_allocator_system(), size, &copy);
        if (!hrx_status_is_ok(status)) {
            hrx_status_ignore(status);
            error = "HIP code object allocation failed";
        } else {
            std::memcpy(copy, data, size);
            result.hsaco_data                   = copy;
            result.hsaco_size                   = size;
            result.launch_config.workgroup_count = geometry.workgroup_count;
            result.launch_config.workgroup_size  = geometry.workgroup_size;
            result.launch_config.subgroup_size   = (target != nullptr && std::strncmp(target, "gfx9", 4) == 0) ? 64 : 32;
        }
    }
    if (!error.empty()) {
        GGML_LOG_ERROR("%s: %s\n", __func__, error.c_str());
    }
    const bool ok = error.empty();
    HipCodeObjectLoader::complete(*compiled_ref, std::move(result), ok, std::move(error));
    return compiled_ref;
}

}  // namespace ggml::hrx
