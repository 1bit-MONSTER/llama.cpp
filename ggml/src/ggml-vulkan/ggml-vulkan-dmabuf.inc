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

// Included by ggml-vulkan.cpp. Imports a dma-buf (device memory another API exported, for example HRX) as a
// Vulkan device buffer with no copy. Reached through the proc "ggml_backend_dev_buffer_from_dmabuf".

#include <unistd.h>

static ggml_backend_buffer_t ggml_backend_vk_dev_buffer_from_dmabuf(ggml_backend_dev_t dev, int fd, size_t offset, size_t size) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    vk_device device = ggml_vk_get_device(ctx->device);
    const off_t fd_size = lseek(fd, 0, SEEK_END);
    if (!device->external_memory_dma_buf || fd_size <= 0 || offset + size > (size_t) fd_size) {
        return nullptr;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }
    vk::ExternalMemoryBufferCreateInfo external_bci { vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT };
    vk::BufferCreateInfo bci { vk::BufferCreateFlags(), size, usage_flags, vk::SharingMode::eExclusive, 0, nullptr };
    bci.setPNext(&external_bci);

    vk_buffer buf = std::make_shared<vk_buffer_struct>();
    buf->buffer = device->device.createBuffer(bci);
    const vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);
    const vk::MemoryFdPropertiesKHR fd_props = device->device.getMemoryFdPropertiesKHR(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, fd);
    const vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();
    const uint32_t bits = fd_props.memoryTypeBits & mem_req.memoryTypeBits;
    // the memory type only describes CPU mappings here: GPU access to exported device memory runs at full speed
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount && type == UINT32_MAX; i++) {
        if ((bits & (1u << i)) && !(mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostCached)) {
            type = i;
        }
    }
    for (uint32_t i = 0; i < mem_props.memoryTypeCount && type == UINT32_MAX; i++) {
        if (bits & (1u << i)) {
            type = i;
        }
    }
    if (type == UINT32_MAX || offset % mem_req.alignment != 0) {
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }

    const int import_fd = dup(fd); // on success Vulkan owns it
    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };
    vk::ImportMemoryFdInfoKHR import_info { vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, import_fd };
    import_info.setPNext(&mem_flags_info);
    try {
        buf->device_memory = device->device.allocateMemory({ (vk::DeviceSize) fd_size, type, &import_info });
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: dma-buf import failed (%s)\n", e.what());
        close(import_fd);
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }
    device->device.bindBufferMemory(buf->buffer, buf->device_memory, offset);
    // keep ggml on GPU-side paths: never map the imported memory on the host
    buf->memory_property_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    buf->ptr = nullptr;
    buf->device = device;
    buf->size = size;
    if (device->buffer_device_address) {
        buf->bda_addr = device->device.getBufferAddress(vk::BufferDeviceAddressInfo(buf->buffer));
    }

    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(device, std::move(buf), device->name);
    return ggml_backend_buffer_init(ggml_backend_vk_device_get_buffer_type(dev), ggml_backend_vk_buffer_interface, bufctx, size);
}
