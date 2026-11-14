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

// engine #2393: build an HRX buffer whose storage is the caller's pointer (no allocation, no pool charge).
// Used by model_buffer_from_mmap instead of retagging a CPU buffer, which made the HRX buffer-context accessor
// read a CPU context. Global scope on purpose: the definition sits outside the anonymous ggml::hrx namespace.
ggml_backend_buffer_t hrx_imported_buffer_from_ptr(ggml_backend_hrx_device_context * device, void * ptr, size_t size);
// engine #2393: the same window shape for a mapped weight -- mapped_buft, no parent allocation to inherit.
ggml_backend_buffer_t hrx_mapped_buffer_from_ptr(ggml_backend_hrx_device_context * device, void * ptr, size_t size);

namespace ggml::hrx {

// engine #856: the model weights stay in the loader's mmap and are not uploaded to a device buffer. The host-weight cache
// packs each weight from these pages, so GTT holds one copy of the model instead of two. GGML_HRX_MODEL_BUFFER_COPY=1
// restores the old upload, and so does a residency limit (GGML_HRX_HOST_WEIGHT_LIMIT_MB).
bool model_buffer_from_mmap_enabled();

// engine #2393: build an HRX buffer whose storage is the caller's pointer (no allocation, no pool charge).
// Used by model_buffer_from_mmap instead of retagging a CPU buffer, which made the HRX buffer-context accessor
// read a CPU context.

// True for a buffer of the mapped type, i.e. weights that are host pages of the model file and not a device upload. A
// weight in such a buffer costs a device copy for every layout the kernels read it in (engine #2137).
bool buffer_is_file_mapped(ggml_backend_buffer_t buffer);

// engine #2137: weights in the file mapping that have a device copy in the stored (GGUF) layout, recorded when the copy
// is made. A matcher that can read a weight either stored or packed asks this before choosing, so it does not add a
// packed copy next to a stored one that a decode kernel already made.
//
// The key is the host address of the bound tensor's bytes: tensor->data on the matcher side, and the binding's buffer
// base plus its offset on the copy side, which is the same address (backend-buffer-binding.cpp computes the offset as
// tensor->data - base). stored_copy_key() is that expression for a tensor; both sides use it.
//
// Lifetime: an entry describes bytes of one mapped model buffer. model_buffer_from_mmap() forgets every entry inside
// the range it wraps, so a model mapped where an earlier one was starts with none. Within one mapping the entries stay
// true: a file-mapped weight is never evicted (the mapping is only used when there is no host-weight limit, see
// model_buffer_from_mmap_enabled()). A stale entry would only make a matcher choose the stored layout, whose copy is
// then made on first use; it cannot make a kernel read bytes that were not written.
const void * stored_copy_key(const ggml_tensor * tensor);
void         note_stored_device_copy(const void * host_address);
bool         has_stored_device_copy(const void * host_address);
void         forget_stored_device_copies(const void * begin, size_t size);

// The buffer type of these buffers ("<device>_Mapped"). It is host memory; only this device claims it.
void model_buffer_from_mmap_init_type(ggml_backend_hrx_device_context * device);

// ggml_backend_dev_buffer_from_host_ptr for HRX: wrap [ptr, ptr + size) without a copy or a device allocation.
ggml_backend_buffer_t model_buffer_from_mmap(ggml_backend_hrx_device_context * device, void * ptr, size_t size);

}  // namespace ggml::hrx
