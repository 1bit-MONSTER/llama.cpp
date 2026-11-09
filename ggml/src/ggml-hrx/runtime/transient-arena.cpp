#include "transient-arena.h"

#include "hrx-1bit-safety.h"
#include "hrx-interop-utils.h"
#include "hrx_runtime.h"

#include <cstdlib>
#include <utility>

namespace ggml::hrx {

TransientArena::AllocationLease::AllocationLease(TransientArena & arena, std::unique_lock<std::mutex> lock) :
    arena_(&arena),
    lock_(std::move(lock)) {}

Status TransientArena::AllocationLease::ensure_capacity(hrx_device_t device,
                                                        hrx_stream_t stream,
                                                        size_t       required_size) {
    Status status;
    if (arena_ == nullptr || !lock_.owns_lock()) {
        status.log("missing transient arena allocation lease");
        return status;
    }
    return arena_->ensure_capacity_locked(device, stream, required_size);
}

TransientArenaAllocationRef TransientArena::AllocationLease::current_allocation() const {
    if (arena_ == nullptr || !lock_.owns_lock()) {
        return {};
    }
    return arena_->current_allocation_locked();
}

TransientArena::~TransientArena() {
    clear();
}

void TransientArena::set_pool_budget(size_t budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_budget_ = budget;
}

void TransientArena::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_ != nullptr) {
        hrx_buffer_release(buffer_);
        buffer_ = nullptr;
        // engine #1711: the arena held this much of the pool, so clearing it gives the pool those bytes back. A
        // diagnostic run keeps retired buffers alive in `diagnostic_retired_buffers_` and those stay charged, which is
        // exactly the pressure that mode exists to reproduce.
        if (pool_budget_ != 0 && allocation_capacity_ != 0) {
            ggml::hrx::onebit::pool_release(allocation_capacity_);
        }
    }
    for (hrx_buffer_t buffer : diagnostic_retired_buffers_) {
        hrx_buffer_release(buffer);
    }
    diagnostic_retired_buffers_.clear();
    allocation_capacity_ = 0;
    allocation_id_       = kInvalidTransientArenaAllocationId;
}

uint64_t TransientArena::next_allocation_id() {
    const uint64_t id = next_allocation_id_++;
    if (next_allocation_id_ == kInvalidTransientArenaAllocationId) {
        ++next_allocation_id_;
    }
    return id;
}

Status TransientArena::ensure_capacity(hrx_device_t device, hrx_stream_t stream, size_t required_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ensure_capacity_locked(device, stream, required_size);
}

TransientArena::AllocationLease TransientArena::acquire_allocation_lease() {
    return AllocationLease(*this, std::unique_lock<std::mutex>(mutex_));
}

Status TransientArena::ensure_capacity_locked(hrx_device_t device, hrx_stream_t stream, size_t required_size) {
    Status status;
    const bool diagnostic_fresh = std::getenv("GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA") != nullptr;
    if (required_size == 0 || (!diagnostic_fresh && allocation_capacity_ >= required_size)) {
        return status;
    }
    if (device == nullptr) {
        status.log("missing HRX device for transient arena allocation");
        return status;
    }
    if (stream == nullptr) {
        status.log("missing HRX stream for transient arena allocation");
        return status;
    }
    if (buffer_ != nullptr) {
        if (diagnostic_fresh) {
            diagnostic_retired_buffers_.push_back(buffer_);
        } else {
        if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
            status.log("synchronize before growing transient arena: %s", error->c_str());
            return status;
        }
        hrx_buffer_release(buffer_);
        // engine #1711: release the old charge BEFORE reserving the new one, so the ledger holds at most one arena per
        // process and a grow from A to B is charged as B, not A+B. This is why the fix is not "charge the delta": the
        // old buffer is released first (its bytes are gone either way), so the honest sequence is release-then-reserve.
        if (pool_budget_ != 0 && allocation_capacity_ != 0) {
            ggml::hrx::onebit::pool_release(allocation_capacity_);
        }
        }
        buffer_              = nullptr;
        allocation_capacity_ = 0;
        allocation_id_       = kInvalidTransientArenaAllocationId;
    }

    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    // engine #1711: admission. A launch whose scratch does not fit the pool is refused HERE, with both numbers, instead of
    // reaching the runtime allocator and failing after the ceiling said it fit (engine #1417's shape).
    if (pool_budget_ != 0 && !ggml::hrx::onebit::pool_reserve(pool_budget_, required_size, "the transient arena")) {
        status.log("the transient arena needs %.1f MiB and the pool budget is %.1f MiB with %.1f MiB already charged; "
                   "refusing this launch at admission rather than letting the runtime allocator fail (engine #1711)",
                   required_size / (double) (1u << 20), pool_budget_ / (double) (1u << 20),
                   ggml::hrx::onebit::pool_allocated() / (double) (1u << 20));
        return status;
    }
    hrx_buffer_t allocation = nullptr;
    if (ErrorResult error = take_status(
            hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, required_size, &allocation))) {
        status.log("allocate transient arena: %s", error->c_str());
        if (pool_budget_ != 0) {
            ggml::hrx::onebit::pool_release(required_size);   // nothing was allocated: give the charge back
        }
        return status;
    }

    buffer_              = allocation;
    allocation_capacity_ = required_size;
    allocation_id_       = next_allocation_id();
    return status;
}

TransientArenaAllocationRef TransientArena::current_allocation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_allocation_locked();
}

size_t TransientArena::capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_capacity_;
}

uint64_t TransientArena::allocation_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_id_;
}

TransientArenaAllocationRef TransientArena::current_allocation_locked() const {
    return { buffer_, allocation_capacity_, allocation_id_ };
}

}  // namespace ggml::hrx
