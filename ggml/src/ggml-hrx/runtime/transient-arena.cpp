#include "transient-arena.h"

#include "hrx-1bit-safety.h"   // engine#1711: the pool ledger (pool_reserve/pool_release)
#include "transient-arena-charge.h"

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
                                                        size_t       required_size,
                                                        size_t       pool_budget) {
    Status status;
    if (arena_ == nullptr || !lock_.owns_lock()) {
        status.log("missing transient arena allocation lease");
        return status;
    }
    return arena_->ensure_capacity_locked(device, stream, required_size, pool_budget);
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

void TransientArena::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_ != nullptr) {
        hrx_buffer_release(buffer_);
        buffer_ = nullptr;
    }
    for (hrx_buffer_t buffer : diagnostic_retired_buffers_) {
        hrx_buffer_release(buffer);
    }
    diagnostic_retired_buffers_.clear();
    if (charged_bytes_ > 0) {
        ggml::hrx::onebit::pool_release(charged_bytes_);   // engine#1711: retired buffers were charged too
        charged_bytes_ = 0;
    }
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

Status TransientArena::ensure_capacity(hrx_device_t device, hrx_stream_t stream, size_t required_size,
                                      size_t pool_budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ensure_capacity_locked(device, stream, required_size, pool_budget);
}

TransientArena::AllocationLease TransientArena::acquire_allocation_lease() {
    return AllocationLease(*this, std::unique_lock<std::mutex>(mutex_));
}

Status TransientArena::ensure_capacity_locked(hrx_device_t device, hrx_stream_t stream, size_t required_size,
                                               size_t pool_budget) {
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
    // engine#1711: the guardrail used to be consulted only by the buffer-type allocator, so the arena grew uncharged and a
    // launch that ran out of pool memory faulted instead of being refused (#680's promise covered tensors only). The charge
    // is taken BEFORE the old buffer is released: a refusal must leave the arena with the capacity it already had.
    const ArenaChargePlan charge = plan_arena_charge(required_size, allocation_capacity_, diagnostic_fresh);
    bool                  charge_taken = false;
    if (charge.reserve_bytes > 0 && pool_budget > 0) {
        if (!ggml::hrx::onebit::pool_reserve(pool_budget, charge.reserve_bytes, "the transient arena")) {
            status.log("refused: the transient arena needs %zu more bytes and the pool has no room", charge.reserve_bytes);
            return status;
        }
        charge_taken   = true;
        charged_bytes_ += charge.reserve_bytes;
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
            // the superseded allocation is gone from the device, so its charge goes with it; in diagnostic mode the
            // buffer was retired above and stays resident, so its charge stays too.
            if (charge.release_bytes > 0) {
                ggml::hrx::onebit::pool_release(charge.release_bytes);
                charged_bytes_ -= charged_bytes_ >= charge.release_bytes ? charge.release_bytes : charged_bytes_;
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
    hrx_buffer_t allocation = nullptr;
    if (ErrorResult error = take_status(
            hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, required_size, &allocation))) {
        status.log("allocate transient arena: %s", error->c_str());
        if (charge_taken) {
            ggml::hrx::onebit::pool_release(charge.reserve_bytes);
            charged_bytes_ -= charged_bytes_ >= charge.reserve_bytes ? charge.reserve_bytes : charged_bytes_;
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
