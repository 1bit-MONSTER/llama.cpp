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

// Disk cache for Loom JIT results. HRX compiles a kernel the first time each specialization is
// dispatched (every new compile parameter or workload value), and the compiled code only lives in
// the process, so every server start pays the compiles again (~1 s for each new prompt-length
// remainder on ZAYA1-8B). This keeps each result under $GGML_HRX_JIT_CACHE_DIR (default
// ~/.cache/1bit/hrx-jit), keyed on every compile input plus the identity of the library that
// holds the compiler, so a rebuilt or updated HRX never reuses old code.
//
// GGML_HRX_JIT_CACHE=0 turns it off. It is also off while a Loom sanitizer is enabled.

#pragma once

#include "loom-jit.h"
#include "loom-kernel-jit.h"

namespace ggml::hrx {

// Records the JIT target (e.g. gfx1151); part of every key.
void loom_jit_disk_cache_set_target(const char * target);

// Fills `compiled` from the cache; false when there is no usable entry.
bool loom_jit_disk_cache_load(const LoomKernelCompileRequest & request, ggml_hrx_loom_jit_compile_result & compiled);

// Stores a successful compile; failures are ignored (the cache is only an accelerator).
void loom_jit_disk_cache_store(const LoomKernelCompileRequest &         request,
                               const ggml_hrx_loom_jit_compile_result & compiled);

}  // namespace ggml::hrx
