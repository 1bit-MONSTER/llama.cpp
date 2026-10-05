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

// Order in which a command program's kernels are recorded into an HRX graph.
//
// HRX puts an execution barrier before a recorded kernel when one of its dependencies was recorded after
// the last barrier. Recording in program order therefore costs a barrier at almost every kernel of a
// dependent chain, even where the program has independent kernels that could share one barrier window
// (the q and kv projections of one input, the router next to a shared expert). Here a kernel that would
// need a barrier first lets a ready, barrier-free kernel from the next eight in program order go ahead.
// The dependencies are the same memory-range hazards the graph recorder uses, and a kernel is only moved
// once all of its dependencies are recorded, so every hazard pair keeps its program order.
//
// Measured on decode programs this reaches the dependency-level minimum (GLM-4.7-Flash 1351 -> 1211
// barriers, Qwen3.6-35B-A3B 1003 -> 973, gpt-oss-20b 628 -> 556 on its multi-token programs) while
// moving kernels only a few places; full dependency-level order gave the same counts but Qwen3.6-35B
// decode read 0.8% lower. Programs where reordering saves no barrier, and prompt programs (transient
// arena extent over 16 MiB), keep program order.
// GGML_HRX_GRAPH_ORDER=program records in program order. GGML_HRX_LOG_GRAPH_ORDER=1 logs the barrier
// count of both orders for every recorded graph.

#pragma once

#include "runtime/command-program-executor.h"

#include <vector>

namespace ggml::hrx {

// Returns |commands| in recording order, or an empty vector when program order should be used.
std::vector<PreparedCommand> order_prepared_commands_for_graph(const std::vector<PreparedCommand> & commands);

}  // namespace ggml::hrx
