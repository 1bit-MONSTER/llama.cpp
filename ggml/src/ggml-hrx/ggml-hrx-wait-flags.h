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
//
// [1bit] hrx-system v0.3.0 helpers: the wait-flag knob and the default queue.
// Every host wait (hrx_stream_wait / hrx_stream_synchronize) and every device's completion
// observation take an explicit hrx_wait_flags_t. One switch sets both, so the spin lever (completion_spin_ns: how long a
// wait polls before it sleeps - heat on this box, memory hrx-wait-spin-thermal) is one knob:
//     GGML_HRX_COMPLETION_WAIT = none | yield (default, as libhrx's own HIP binding) | active
#pragma once

#include "hrx_runtime.h"   // hrx_queue_t, hrx_device_get_queue (and, through it, the wait flags)

#include <cstdlib>
#include <cstring>

namespace ggml::hrx {

inline hrx_wait_flags_t completion_wait_flags() {
    static const hrx_wait_flags_t flags = [] {
        const char * value = std::getenv("GGML_HRX_COMPLETION_WAIT");
        if (value != nullptr && std::strcmp(value, "none") == 0) {
            return (hrx_wait_flags_t) HRX_WAIT_FLAG_NONE;
        }
        if (value != nullptr && std::strcmp(value, "active") == 0) {
            return (hrx_wait_flags_t) HRX_WAIT_FLAG_ACTIVE;
        }
        return (hrx_wait_flags_t) HRX_WAIT_FLAG_YIELD;
    }();
    return flags;
}

// [1bit] hrx-system v0.3.0: streams, graph instances and synchronous copies are created on a queue, not a device. The
// device's first provisioned queue (family 0, queue 0) is what the old device-level calls used implicitly; it is borrowed
// (the device owns it) and lives as long as the device.
inline hrx_queue_t device_queue(hrx_device_t device) {
    return device != nullptr ? hrx_device_get_queue(device, 0, 0) : nullptr;
}

}  // namespace ggml::hrx
