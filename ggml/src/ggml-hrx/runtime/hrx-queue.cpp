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

#include "hrx-queue.h"

#include "hrx_memory.h"

#include <cstdint>

namespace ggml::hrx {

static bool find_queue_family(hrx_device_t device, uint32_t * family) {
    const size_t count = hrx_device_queue_family_count(device);
    for (uint32_t i = 0; i < count; ++i) {
        hrx_queue_family_info_t info = {};
        hrx_status_t            status = hrx_device_query_queue_family(device, i, &info);
        if (!hrx_status_is_ok(status)) {
            hrx_status_ignore(status);
            continue;
        }
        const hrx_queue_role_flags_t needed = HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER;
        if ((info.roles & needed) == needed && info.provisioned_queue_count > 0) {
            *family = i;
            return true;
        }
    }
    return false;
}

hrx_queue_t device_queue(hrx_device_t device) {
    uint32_t family = 0;
    return device != nullptr && find_queue_family(device, &family) ? hrx_device_get_queue(device, family, 0) : nullptr;
}

bool buffer_device_address(hrx_device_t device, hrx_buffer_t buffer, void ** address) {
    *address        = nullptr;
    uint32_t family = 0;
    if (device == nullptr || buffer == nullptr || !find_queue_family(device, &family)) {
        return false;
    }
    hrx_memory_scope_t scope  = {};
    hrx_status_t       status = hrx_device_query_queue_family_memory_scope(device, family, HRX_MEMORY_SITE_PROGRAM, &scope);
    uint64_t           value  = 0;
    if (hrx_status_is_ok(status)) {
        status = hrx_buffer_query_address(buffer, scope, HRX_BUFFER_INTERFACE_DEVICE_ADDRESS, &value);
    }
    if (!hrx_status_is_ok(status)) {
        hrx_status_ignore(status);
        return false;
    }
    *address = reinterpret_cast<void *>(static_cast<uintptr_t>(value));
    return value != 0;
}

}  // namespace ggml::hrx
