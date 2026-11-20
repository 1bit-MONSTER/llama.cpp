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

// FLASH_ATTN_EXT over several KV streams (ne[3] > 1: n_seq_max > 1 without a unified KV cache), one
// ggml_flash_attention_f32_f16_wmma launch per stream (engine#2918).
void register_flash_attention_streams_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
