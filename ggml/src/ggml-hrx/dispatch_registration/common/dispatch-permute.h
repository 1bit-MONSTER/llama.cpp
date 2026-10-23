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

#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

// Standalone GGML_OP_PERMUTE for f32 values whose permuted result already owns its storage.
//
// The view form (a permute whose output shares storage with its input, which is what ggml produces by
// default) is deliberately NOT claimed here: the value has no destination buffer, so the materialisation
// happens at the following CONT, which register_copy_dispatch() already lowers through the strided-source
// copy kernel. Claiming the view form would need a layout-aware copy that writes through a permuted
// destination layout; that is a kernel-side variant (see ~/briefs/TSPECTREE-STATE.md, "kernel asks").
void register_permute_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
