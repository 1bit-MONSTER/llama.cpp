#include "transient-arena.h"

#include "transient-arena-budget.h"

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

void TransientArena::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_budget_locked();
    if (buffer_ != nullptr) {
        hrx_buffer_release(buffer_);
        buffer_ = nullptr;
    }
    for (hrx_buffer_t buffer : diagnostic_retired_buffers_) {
        hrx_buffer_release(buffer);
    }
    diagnostic_retired_buffers_.clear();
    allocation_capacity_ = 0;
    allocation_id_       = kInvalidTransientArenaAllocationId;
}

void TransientArena::release_budget_locked() {
    if (budget_charged_ != 0 && budget_ != nullptr && *budget_ != 0) {
        onebit::pool_release(budget_charged_);
    }
    budget_charged_ = 0;
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
    // engine#1711: the scratch is charged against the pool budget like a tensor buffer. The COARSE check here is a fast
    // path for the obvious case only -- a request larger than the whole budget cannot fit however empty the pool is. It
    // is not the gate: the gate is `pool_reserve` below, which is the only thing that can see what the tensor buffers
    // have already taken, and it runs BEFORE the allocator (review of this PR: reserving after the allocation lets the
    // allocation that can fault happen first, which is the whole failure this change exists to close).
    if (budget_ != nullptr && *budget_ != 0) {
        const TransientArenaCharge plan = plan_transient_arena_charge(allocation_capacity_, required_size, *budget_);
        if (plan.refuse) {
            status.log("transient arena: refusing a %zu byte scratch: the whole pool budget is %zu bytes (GGML_HRX_POOL_LIMIT_MB "
                       "or the GTT ceiling, engine #680/#1711)",
                       plan.charge_bytes, *budget_);
            return status;
        }
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
        }
        buffer_              = nullptr;
        allocation_capacity_ = 0;
        allocation_id_       = kInvalidTransientArenaAllocationId;
    }

    // The replaced capacity goes back to the pool first, then the request is admitted, and only an admitted request
    // reaches the allocator: reserve-then-allocate is the order that makes "refused here rather than faulting the
    // device" true. The reverse order was in the first version of this change and it left the faulting allocation
    // exactly where it was.
    release_budget_locked();
    const bool charged = budget_ != nullptr && *budget_ != 0;
    if (charged && !onebit::pool_reserve(*budget_, required_size, "transient arena scratch")) {
        status.log("transient arena: refusing a %zu byte scratch: the pool budget is %zu bytes and the tensor buffers "
                   "are already charged against it (GGML_HRX_POOL_LIMIT_MB or the GTT ceiling, engine #680/#1711)",
                   required_size, *budget_);
        return status;
    }
    budget_charged_ = charged ? required_size : 0;

    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    hrx_buffer_t allocation = nullptr;
    if (ErrorResult error = take_status(
            hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, required_size, &allocation))) {
        release_budget_locked();  // the reservation is not kept for an allocation that did not happen
        status.log("allocate transient arena: %s", error->c_str());
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
