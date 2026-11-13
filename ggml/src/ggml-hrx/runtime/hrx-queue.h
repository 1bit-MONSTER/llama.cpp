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

// The queue ggml-hrx submits to. libhrx streams, synchronous copies and graph instantiation take an exact
// hrx_queue_t; this backend uses one per device: the first provisioned queue of the first family that can both
// dispatch and transfer. The queue is borrowed from the device and lives as long as it does.

#pragma once

#include "hrx_runtime.h"

namespace ggml::hrx {

// nullptr when the device has no dispatch+transfer family.
hrx_queue_t device_queue(hrx_device_t device);

// Device address of the first byte of |buffer| as kernels on |device|'s queue see it; false when it has none.
bool buffer_device_address(hrx_device_t device, hrx_buffer_t buffer, void ** address);

}  // namespace ggml::hrx
