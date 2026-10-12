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

#include <cstddef>

namespace ggml::hrx {

// Free bytes for an HRX device whose reported total is device_total.
// free = GTT total - GTT used of the matching amdgpu DRM card (sysfs mem_info_gtt_*), clamped by /proc/meminfo
// MemAvailable and by device_total. Falls back to MemAvailable alone when no card matches, and to device_total
// when neither source is readable. GGML_HRX_FREE_MEMORY=total restores the old free == total report.
size_t device_free_memory(size_t device_total);

}  // namespace ggml::hrx
