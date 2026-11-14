#pragma once

#include "ggml-backend-impl.h"
#include "graph/value-map.h"
#include "runtime/graph-program-cache.h"
#include "runtime/host-buffer-registry.h"
#include "runtime/host-memory.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;

struct ggml_backend_hrx_device_context;

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device;
    std::string                       name;
    bool                              host_visible = false;
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device;
    hrx_buffer_t                      buffer;
    uint8_t *                         base;
    uint64_t                          identity;
    uint64_t                          generation;
    bool                              direct_host_binding;
    bool                              imported;
    size_t                            physical_offset = 0;
    size_t                            physical_capacity = 0;
    uint8_t *                         retained_base = nullptr;
};

struct ggml_backend_hrx_device_context {
    hrx_device_t                                   device = nullptr;
    std::string                                    name;
    std::string                                    description;
    std::string                                    architecture;
    size_t                                         memory_total = 0;
    // What this process may plan to put on the device: the GTT ceiling less what other processes already hold at
    // registration (engine #680). Every HRX buffer is charged against it.
    size_t                                         memory_budget = 0;
    // Cap applied after an allocation the allocator refused, and the free figure seen at that moment (engine #1502). The cap is
    // lifted as soon as a later observation shows a larger free pool.
    size_t                                         memory_budget_cap      = 0;
    size_t                                         memory_budget_cap_free = 0;
    // Guards the three budget fields above: buffer_alloc writes them on a refusal while device_memory rewrites them from an
    // observation, and both can run concurrently during a load (engine #1502 review).
    std::mutex                                     memory_budget_mutex;
    bool                                           use_direct_host_bindings = false;
    ggml_backend_buffer_type                       buft               = {};
    ggml_backend_hrx_buffer_type_context           buft_context       = {};
    ggml_backend_buffer_type                       host_buft          = {};
    ggml_backend_hrx_buffer_type_context           host_buft_context  = {};
    ggml_backend_buffer_type                       mapped_buft        = {};
    ggml_backend_hrx_buffer_type_context           mapped_buft_context = {};
    // engine #2393: imported host memory the device addresses directly. NOT mapped_buft: that one is host-typed on
    // purpose (mapped weights ARE host memory, and the scheduler's placement depends on it), while an import is a
    // direct binding whose buffer must not claim to be a host allocation.
    ggml_backend_buffer_type                       imported_buft        = {};
    ggml_backend_hrx_buffer_type_context           imported_buft_context = {};
    ggml::hrx::HostBufferRegistry                  host_buffers;
    std::atomic<uint64_t>                          synchronous_upload_fallbacks{ 0 };
    std::atomic<uint64_t>                          synchronous_download_fallbacks{ 0 };
    std::mutex                                     buffer_stream_mutex;
    // engine #696: one reusable host-mapped staging buffer per device for the ordered host copies (buffer_set /
    // buffer_get). Guarded by buffer_stream_mutex.
    hrx_buffer_t                                   copy_staging        = nullptr;
    size_t                                         copy_staging_size   = 0;
    void *                                         copy_staging_mapped = nullptr;
    hrx_stream_t                                   buffer_stream = nullptr;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context *      device;
    hrx_stream_t                           stream;
    ggml::hrx::KernelExecutableCache       kernel_executables;
    ggml::hrx::GraphProgramCache           graph_programs;
    ggml::hrx::PreparedCommandProgramCache prepared_programs;
    ggml::hrx::TransientArena              transient_arena;
    ggml::hrx::HostTransferManager         host_transfers;
    ggml::hrx::HostWeightCache             host_weights;
    ggml::hrx::GraphReplayStreamState      graph_replay_state;
    std::string                            name;
};

struct ggml_backend_hrx_reg_context {
    bool                                                          initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device>                              devices;

    ~ggml_backend_hrx_reg_context();
};
