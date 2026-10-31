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
bool host_weight_exceeds_residency(const ggml_tensor * op);

}  // namespace ggml::hrx
