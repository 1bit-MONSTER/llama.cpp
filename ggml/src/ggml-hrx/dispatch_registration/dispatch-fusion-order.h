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

#include "dispatch-registry.h"

namespace ggml::hrx {

// Read-before-write guard for every registry match (engine #533 / #570).
//
// The scheduler appends a match's dispatches at its root node's position. A matcher that covers nodes beyond
// its root can therefore read a value whose producer has not been scheduled yet, read a value its own programme
// does not write, or read a value an earlier programme fused away. Each of those is a "reads transient value N
// before write" in the command-program verifier and a read of stale memory on the device.
//
// fusion_order_accepts() refuses such a candidate, so the registry falls through to the next registration
// (ultimately the root's single-op matcher, whose inputs the traversal has already produced). For every binding
// the kernel does not declare write-only, following views back to the node that produced the data:
//   1. the producer must already be covered by an earlier match, or be covered by this candidate;
//   2. a producer covered by this candidate must have its storage written by an earlier dispatch of the
//      candidate;
//   3. a producer covered by an earlier match must have its storage written by a dispatch already in the plan.
// Checks 2 and 3 apply, as in verify_command_program, to read-only bindings of transient storage.
// Leaves, zero-byte values, plan transients and storages rooted at a graph input are always available.
//
// GGML_HRX_FUSION_ORDER_GUARD=0 disables the guard (A/B and bisection); GGML_HRX_LOG_FUSION_ORDER=1 prints every
// refusal to stderr. The refusal reason is also appended to match.status.
bool fusion_order_accepts(const DispatchMatchContext & context, DispatchMatch & match, const char * registration_name);

// True when every consumer of `value` is a MUL_MAT_ID whose weight (input 0) `weight_qualifies` accepts. A
// programme that writes only an alternate of `value` (for example an f16 copy) uses this to require that every
// reader is a projection that takes the alternate; any other reader would read a value nothing has written.
bool every_consumer_is_mul_mat_id_with_weight(const Graph & graph, ValueId value,
                                              bool (*weight_qualifies)(const Value & weight));

}  // namespace ggml::hrx
