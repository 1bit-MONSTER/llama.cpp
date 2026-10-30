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
// A weight that lives in a host buffer the HRX backend did not allocate (llama.cpp's CPU_Mapped model
// buffers, and any other host buffer) is not device addressable. Running an op that reads it therefore
// makes the executor materialize the tensor into a fresh DEVICE_LOCAL buffer on first use
// (command-program-executor.cpp, materialize_host_bindings -> HostWeightCache). That residency is
// unbounded and is never returned to the allocator, so with the device present but nothing offloaded
// (-ngl 0) a whole model's weights end up claimed as device memory -- on this APU that is GTT, i.e.
// system memory outside the process RSS. That is the ~107 GiB that engine #669 measured.
//
// The backend cannot make that memory safe: the bytes have to exist somewhere for the dispatch. So an op
// whose weight input is host resident is declined here and stays on the CPU. Weights the scheduler
// actually placed on HRX live in HRX's own buffers, are directly bindable, and are unaffected -- a fully
// offloaded model keeps every op it has today.
bool host_weight_requires_cpu(const ggml_tensor * op);

}  // namespace ggml::hrx
