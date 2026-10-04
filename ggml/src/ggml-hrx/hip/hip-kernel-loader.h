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

// Turns a registered HIP kernel (hip-kernel-registry.h) into the compile result the HRX kernel
// executable cache loads: the embedded code object for the device target plus the launch
// geometry from the kernel's launch function. Hooked into KernelExecutableCache::get_or_compile.

#pragma once

#include "dispatch/dispatch.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/loom-kernel-jit.h"

#include <string>

namespace ggml::hrx {

LoomCompiledKernelRef make_hip_compiled_kernel(const std::string &      key,
                                               const KernelDefinition & definition,
                                               const Dispatch &         dispatch,
                                               const char *             target);

// Lets make_hip_compiled_kernel complete a LoomCompiledKernel without a Loom compile.
struct HipCodeObjectLoader {
    static void complete(LoomCompiledKernel & kernel, ggml_hrx_loom_jit_compile_result compiled, bool success,
                         std::string error) {
        kernel.complete(std::move(compiled), success, std::move(error));
    }
};

}  // namespace ggml::hrx
