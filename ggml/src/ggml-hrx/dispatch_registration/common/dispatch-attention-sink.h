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
#include "graph/graph.h"

namespace ggml::hrx {

// A 5-input FLASH_ATTN_EXT whose fifth input is F32 sinks, one per query head.
bool attention_sinks_supported(const Graph & graph, const GraphNode & node);

// Appends the in-place sink rescale of the node's output (run after the FlashAttention dispatch).
bool append_attention_sink_dispatch(const Graph & graph, const GraphNode & node, DispatchMatch & dispatch_match);

}  // namespace ggml::hrx
