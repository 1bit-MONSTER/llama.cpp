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

namespace ggml::hrx {

// Code object bytes for (stem, target), or nullptr; target nullptr = any target.
const void * hip_code_object_data(const char * stem, const char * target, size_t * size);

// True when the build embedded this stem for at least one target.
bool hip_code_object_embedded(const char * stem);

// Short content digest of the stem's first embedded code object, or nullptr.
const char * hip_code_object_digest(const char * stem);

}  // namespace ggml::hrx
