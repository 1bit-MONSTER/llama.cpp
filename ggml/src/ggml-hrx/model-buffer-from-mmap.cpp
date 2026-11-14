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

#include "model-buffer-from-mmap.h"

#include "backend-context.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "runtime/host-memory.h"

#include <iterator>
#include <mutex>
#include <unordered_set>
#include <cstdlib>
#include <cstring>

namespace ggml::hrx {

namespace {

const char * mapped_type_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context)->name.c_str();
}

// Only the loader's mmap path makes these buffers. Anything else that asks this type for memory gets plain host memory
// with the same type, so it is still host data to the executor.
ggml_backend_buffer_t mapped_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (buffer != nullptr) {
        buffer->buft = buft;
    }
    return buffer;
}

size_t mapped_type_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    GGML_UNUSED(buft);
}

bool mapped_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;
    GGML_UNUSED(buft);
}

}  // namespace

bool model_buffer_from_mmap_enabled() {
    const char * copy = std::getenv("GGML_HRX_MODEL_BUFFER_COPY");
    if (copy != nullptr && copy[0] != '\0' && std::strcmp(copy, "0") != 0) {
        return false;
    }
    // Under GGML_HRX_HOST_WEIGHT_LIMIT_MB a host weight over the budget runs on the CPU. Keep the upload there, so a
    // limit does not move device weights off the device.
    return HostWeightCache::residency_limit() == 0;
}

namespace {
std::mutex & stored_copy_mutex() {
    static std::mutex mutex;
    return mutex;
}
std::unordered_set<const void *> & stored_copy_addresses() {
    static std::unordered_set<const void *> addresses;
    return addresses;
}
}  // namespace

void note_stored_device_copy(const void * host_address) {
    if (host_address == nullptr) {
        return;
    }
    const std::lock_guard<std::mutex> lock(stored_copy_mutex());
    stored_copy_addresses().insert(host_address);
}

const void * stored_copy_key(const ggml_tensor * tensor) {
    return tensor == nullptr ? nullptr : tensor->data;
}

void forget_stored_device_copies(const void * begin, size_t size) {
    if (begin == nullptr || size == 0) {
        return;
    }
    const uint8_t *             first = static_cast<const uint8_t *>(begin);
    const uint8_t *             last  = first + size;
    const std::lock_guard<std::mutex> lock(stored_copy_mutex());
    auto &                      addresses = stored_copy_addresses();
    for (auto it = addresses.begin(); it != addresses.end();) {
        const uint8_t * address = static_cast<const uint8_t *>(*it);
        it = address >= first && address < last ? addresses.erase(it) : std::next(it);
    }
}

bool has_stored_device_copy(const void * host_address) {
    if (host_address == nullptr) {
        return false;
    }
    const std::lock_guard<std::mutex> lock(stored_copy_mutex());
    return stored_copy_addresses().count(host_address) != 0;
}

bool buffer_is_file_mapped(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft != nullptr && buffer->buft->iface.get_name == mapped_type_name;
}

void model_buffer_from_mmap_init_type(ggml_backend_hrx_device_context * device) {
    device->mapped_buft_context = { device, device->name + "_Mapped", true };
    device->mapped_buft         = {
        // designated: the merged ggml interface also has alloc_buffer_n and get_alloc_size_n (left null)
        { .get_name = mapped_type_name, .alloc_buffer = mapped_type_alloc, .get_alignment = mapped_type_alignment,
          .get_max_size = nullptr, .get_alloc_size = nullptr, .is_host = mapped_type_is_host },
        nullptr,
        &device->mapped_buft_context,
    };
}

ggml_backend_buffer_t model_buffer_from_mmap(ggml_backend_hrx_device_context * device, void * ptr, size_t size) {
    // The CPU from-pointer buffer neither copies nor frees: the loader owns the mapping. Retag it so the scheduler places
    // these weights on this device; the binding resolver then reads them as host data and packs them once.
    if (ptr == nullptr || size == 0 || reinterpret_cast<uintptr_t>(ptr) % TENSOR_ALIGNMENT != 0) {
        GGML_LOG_ERROR("%s: refusing host pointer %p (size %zu): it must be non-null and %d-byte aligned\n", __func__,
                       ptr, size, TENSOR_ALIGNMENT);
        return nullptr;
    }
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    if (buffer != nullptr) {
        buffer->buft = &device->mapped_buft;
        // engine #2137: entries from a mapping that was here before describe other bytes
        forget_stored_device_copies(ptr, size);
    }
    return buffer;
}

}  // namespace ggml::hrx
