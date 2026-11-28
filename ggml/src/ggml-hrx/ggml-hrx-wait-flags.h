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
#include <vector>

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

// Opens every "amdgpu" device as one sealed group (the v0.3.0 replacement for hrx_gpu_initialize + device_get), with
// the completion_wait_flags() policy. On success *runtime and *group own references the caller releases (group first);
// zero devices leaves *group NULL and still returns ok. The backend's registry uses its own copy (it also sets the
// event sink); this one is for tools and tests.
inline hrx_status_t open_gpu_group(hrx_runtime_t * runtime, hrx_device_group_t * group) {
    *runtime = nullptr;
    *group   = nullptr;
    hrx_runtime_options_t options = {};
    hrx_status_t status = hrx_runtime_create(&options, hrx_host_allocator_system(), runtime);
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    hrx_device_info_t * infos = nullptr;
    size_t count = 0;
    status = hrx_runtime_query_available_devices(*runtime, "amdgpu", hrx_host_allocator_system(), &count, &infos);
    if (hrx_status_is_ok(status) && count > 0) {
        std::vector<hrx_device_create_params_t> params(count);
        for (size_t i = 0; i < count; ++i) {
            params[i].uri                   = infos[i].uri;
            params[i].completion_wait_flags = completion_wait_flags();
        }
        status = hrx_device_group_create(*runtime, count, params.data(), group);
    }
    if (infos != nullptr) {
        hrx_host_allocator_free(hrx_host_allocator_system(), infos);
    }
    return status;
}

}  // namespace ggml::hrx
