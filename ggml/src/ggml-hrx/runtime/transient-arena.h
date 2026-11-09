#pragma once

#include "dispatch/command-program-resolver.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

typedef struct hrx_device_s * hrx_device_t;
typedef struct hrx_stream_s * hrx_stream_t;

namespace ggml::hrx {

class TransientArena {
  public:
    class AllocationLease {
      public:
        AllocationLease()                                        = default;
        AllocationLease(AllocationLease &&) noexcept             = default;
        AllocationLease & operator=(AllocationLease &&) noexcept = default;

        AllocationLease(const AllocationLease &)             = delete;
        AllocationLease & operator=(const AllocationLease &) = delete;

        Status                      ensure_capacity(hrx_device_t device, hrx_stream_t stream, size_t required_size,
                                                      size_t pool_budget = 0);
        TransientArenaAllocationRef current_allocation() const;

      private:
        friend class TransientArena;

        AllocationLease(TransientArena & arena, std::unique_lock<std::mutex> lock);

        TransientArena *             arena_ = nullptr;
        std::unique_lock<std::mutex> lock_;
    };

    TransientArena() = default;
    ~TransientArena();

    TransientArena(const TransientArena &)             = delete;
    TransientArena & operator=(const TransientArena &) = delete;

    Status          ensure_capacity(hrx_device_t device, hrx_stream_t stream, size_t required_size,
                                    size_t pool_budget = 0);
    AllocationLease acquire_allocation_lease();
    void            clear();

    TransientArenaAllocationRef current_allocation() const;

    size_t capacity() const;

    uint64_t allocation_id() const;

  private:
    uint64_t                    next_allocation_id();
    Status                      ensure_capacity_locked(hrx_device_t device, hrx_stream_t stream, size_t required_size,
                                                           size_t pool_budget);
    TransientArenaAllocationRef current_allocation_locked() const;

    mutable std::mutex mutex_;
    hrx_buffer_t       buffer_              = nullptr;
    std::vector<hrx_buffer_t> diagnostic_retired_buffers_;
    // Bytes this arena currently holds against the pool ledger (engine#1711). It is NOT allocation_capacity_ in
    // diagnostic mode, where superseded buffers are retired and stay resident: the tally covers retired + current, so
    // releasing allocation_capacity_ there would under-report exactly the mode used to diagnose out-of-memory.
    size_t                     charged_bytes_ = 0;
    size_t             allocation_capacity_ = 0;
    uint64_t           allocation_id_       = kInvalidTransientArenaAllocationId;
    uint64_t           next_allocation_id_  = 1;
};

}  // namespace ggml::hrx
