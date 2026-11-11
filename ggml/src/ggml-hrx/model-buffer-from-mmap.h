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

#include <cstddef>

struct ggml_backend_hrx_device_context;

namespace ggml::hrx {

// engine #856: the model weights stay in the loader's mmap and are not uploaded to a device buffer. The host-weight cache
// packs each weight from these pages, so GTT holds one copy of the model instead of two. GGML_HRX_MODEL_BUFFER_COPY=1
// restores the old upload, and so does a residency limit (GGML_HRX_HOST_WEIGHT_LIMIT_MB).
bool model_buffer_from_mmap_enabled();

// True for a buffer of the mapped type, i.e. weights that are host pages of the model file and not a device upload. A
// weight in such a buffer costs a device copy for every layout the kernels read it in (engine #2137).
bool buffer_is_file_mapped(ggml_backend_buffer_t buffer);

// The buffer type of these buffers ("<device>_Mapped"). It is host memory; only this device claims it.
void model_buffer_from_mmap_init_type(ggml_backend_hrx_device_context * device);

// ggml_backend_dev_buffer_from_host_ptr for HRX: wrap [ptr, ptr + size) without a copy or a device allocation.
ggml_backend_buffer_t model_buffer_from_mmap(ggml_backend_hrx_device_context * device, void * ptr, size_t size);

}  // namespace ggml::hrx
