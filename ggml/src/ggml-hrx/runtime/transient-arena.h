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

        Status                      ensure_capacity(hrx_device_t device, hrx_stream_t stream, size_t required_size);
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

    Status          ensure_capacity(hrx_device_t device, hrx_stream_t stream, size_t required_size);
    AllocationLease acquire_allocation_lease();
    void            clear();

    // engine #1711: the arena's buffer is pool memory like any other, so it is charged against the same ledger the
    // backend's buffer type allocator uses (`pool_reserve`/`pool_release`, hrx-1bit-safety.h). Before this, only ggml
    // tensor buffers were charged: a launch whose scratch outgrew the pool failed inside libhrx AFTER admission said it
    // fit, which is engine #1417's shape -- the guardrail promised "a launch that needs more is refused here" and the
    // one allocation a large case actually grows into was invisible to it.
    // 0 means "the device size is unknown", the same convention `pool_reserve` uses: charge nothing, keep the old
    // behaviour, and let the allocator decide.
    void set_pool_budget(size_t budget);

    TransientArenaAllocationRef current_allocation() const;

    size_t capacity() const;

    uint64_t allocation_id() const;

  private:
    uint64_t                    next_allocation_id();
    Status                      ensure_capacity_locked(hrx_device_t device, hrx_stream_t stream, size_t required_size);
    TransientArenaAllocationRef current_allocation_locked() const;

    mutable std::mutex mutex_;
    size_t             pool_budget_         = 0;   // 0 = unknown device size: uncharged, as pool_reserve does
    hrx_buffer_t       buffer_              = nullptr;
    std::vector<hrx_buffer_t> diagnostic_retired_buffers_;
    size_t             allocation_capacity_ = 0;
    uint64_t           allocation_id_       = kInvalidTransientArenaAllocationId;
    uint64_t           next_allocation_id_  = 1;
};

}  // namespace ggml::hrx
