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

struct ggml_tensor;

namespace ggml::hrx {

// Placement guard for host-resident model weights (engine #669).
//
// A weight that lives in a host buffer the HRX backend did not allocate (llama.cpp's CPU_Mapped model buffers, and
// any other host buffer) is not device addressable, so serving an op that reads it makes the executor materialize
// the tensor into a device buffer (command-program-executor.cpp, materialize_host_bindings -> HostWeightCache).
// That residency is bounded (HostWeightCache::limit), but a weight larger than the whole budget can never be
// resident, and the staging fallback would copy it per execution - for a 103 GiB embedding table that is the same
// wall in a different shape. So an op whose weight input cannot fit the residency budget is declined here and stays
// on the CPU. Weights the scheduler placed on HRX live in HRX's own buffers and are unaffected.
//
// The invariant this file keeps testable (engine #674, 2026-11-02): a placement change must not reintroduce the
// weightless-op / weight-op split - a weight-consuming node declined to the CPU while the weightless node that
// consumes its output stays on HRX. Measured at pin d8bd0ff6052: the bisected corrupting pair {MUL_MAT, MUL} needs the
// norm MUL declined, and this predicate is the only weight-dependent decline in device_supports_op, so a budget would
// have to fall below a norm scale (512 B - 10 KiB) while the knob's floor is 1 MiB - the pair is not expressible.
// Empirically, every budget that declines a node (32/128/256/300/304 MiB -> token_embd's MUL_MAT and GET_ROWS) also
// refuses to run (#173), and 305 MiB declines nothing and still refuses, so no declined placement ever executes. The
// decline is logged in the .cpp so that if either half changes, the graph change is visible in the log and not only in
// the GGML_HRX_REPORT census.
bool host_weight_exceeds_residency(const ggml_tensor * op);

}  // namespace ggml::hrx
