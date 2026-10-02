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

#include "ggml-backend.h"

struct ggml_tensor;

namespace ggml::hrx {

// False for an ADD_ID / SWIGLU_OAI whose expert MUL_MAT_ID (directly, or through ADD_ID) will not run on this
// device: the device declines the op, or the expert weights sit in a buffer it cannot read (see
// moe-placement-guard.cpp). True for everything else.
bool moe_tail_claimable(ggml_backend_dev_t device, const ggml_tensor * op);

}  // namespace ggml::hrx
