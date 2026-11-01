#pragma once

#include "dispatch/dispatch.h"
#include "ggml.h"
#include "runtime/host-buffer-registry.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

typedef struct hrx_buffer_s * hrx_buffer_t;
typedef struct hrx_device_s * hrx_device_t;
typedef struct hrx_stream_s * hrx_stream_t;

namespace ggml::hrx {

struct HostTransferStats {
    uint64_t uploads        = 0;
    uint64_t downloads      = 0;
    size_t   upload_bytes   = 0;
    size_t   download_bytes = 0;
};

class HostTransferManager {
  public:
    Status upload_synchronous(hrx_stream_t stream,
                              const void * host_source,
                              hrx_buffer_t destination,
                              size_t       offset,
                              size_t       size);
    Status upload_async(hrx_stream_t stream,
                        const void * host_source,
                        hrx_buffer_t destination,
                        size_t       offset,
                        size_t       size);
    Status download_synchronous(hrx_stream_t stream,
                                hrx_buffer_t source,
                                size_t       offset,
                                void *       host_destination,
                                size_t       size);

    HostTransferStats stats() const;
    void              clear();

  private:
    mutable std::mutex mutex_;
    HostTransferStats  stats_;
};

struct HostWeightSource {
    const void * host_data  = nullptr;
    hrx_buffer_t device_buffer       = nullptr;
    uint64_t     identity   = 0;
    uint64_t     generation = 0;
    size_t       capacity   = 0;
    size_t       offset     = 0;
    size_t       length     = 0;
    size_t       materialized_length = 0;
    std::string  layout              = kNativeWeightLayout;
    ggml_type    source_type         = GGML_TYPE_COUNT;
    int64_t      input_size          = 0;
    int64_t      output_size         = 0;
};

struct HostWeightCacheStats {
    uint64_t hits             = 0;
    uint64_t misses           = 0;
    uint64_t evictions        = 0;
    size_t   allocation_count = 0;
    size_t   resident_bytes   = 0;
};

class HostWeightLease {
  public:
    HostWeightLease() = default;

    bool                valid() const;
    hrx_buffer_t        buffer() const;
    // Every resident weight lives in one arena buffer (engine #669), so the binding is (buffer, offset, length).
    size_t              offset() const;
    size_t              length() const;
    const std::string & layout() const;

  private:
    struct Entry;
    // The cache owns the entry, so a lease is a *borrowed* snapshot: `buffer_`/`length_`/`layout_` are what the
    // caller binds, and `entry_` only says whether it is still there. On eviction (which synchronizes the stream
    // first) the entry goes away, valid() turns false, and the next execution re-acquires and rebinds. This is what
    // makes the residency bounded: a prepared program no longer keeps a materialized weight alive by holding it.
    std::weak_ptr<Entry> entry_;
    hrx_buffer_t         buffer_ = nullptr;
    size_t               offset_ = 0;
    size_t               length_ = 0;
    std::string          layout_;

    explicit HostWeightLease(const std::shared_ptr<Entry> & entry);
    friend class HostWeightCache;
};

struct HostWeightAcquireResult {
    HostWeightLease lease;
    Status          status;

    bool valid() const { return lease.valid() && status.success(); }
};

class HostWeightCache {
  public:
    HostWeightCache() = default;
    ~HostWeightCache();

    HostWeightCache(const HostWeightCache &)             = delete;
    HostWeightCache & operator=(const HostWeightCache &) = delete;

    HostWeightAcquireResult acquire(hrx_device_t             device,
                                    hrx_stream_t             stream,
                                    HostTransferManager &    transfers,
                                    const HostWeightSource & source);
    HostWeightCacheStats    stats() const;
    void                    clear();

    // Bytes of materialized host weights the cache may keep resident (GGML_HRX_HOST_WEIGHT_LIMIT_MB, default
    // 2048). A single weight larger than the whole budget is refused rather than made resident.
    // engine #669: one command-program execution at a time. While one is in flight the slots the cache frees are
    // parked and only returned to the free list when it ends: a lease is a weak snapshot, so "evicted" is not
    // "unbound", and handing a slot to another weight while the running program still binds its old offset is
    // what produced NaN logits.
    void        begin_execution();
    void        end_execution();

    size_t        limit() const;
    // The same budget without a cache instance, for the placement guard (host-weight-placement.cpp).
    static size_t residency_limit();

  private:
    struct SourceKey {
        uint64_t identity   = 0;
        uint64_t generation = 0;
        size_t   capacity   = 0;
        size_t   offset     = 0;
        size_t   length     = 0;
        size_t      materialized_length = 0;
        std::string layout;
        ggml_type   source_type = GGML_TYPE_COUNT;
        int64_t     input_size  = 0;
        int64_t     output_size = 0;

        bool operator==(const SourceKey & other) const {
            return identity == other.identity && generation == other.generation && capacity == other.capacity &&
                   offset == other.offset && length == other.length &&
                   materialized_length == other.materialized_length && layout == other.layout &&
                   source_type == other.source_type && input_size == other.input_size &&
                   output_size == other.output_size;
        }
    };

    struct SourceKeyHash {
        size_t operator()(const SourceKey & key) const;
    };

    mutable std::mutex                                                                    mutex_;
    std::unordered_map<SourceKey, std::shared_ptr<HostWeightLease::Entry>, SourceKeyHash> entries_;
    std::unordered_map<SourceKey, uint64_t, SourceKeyHash>                                last_use_;
    HostWeightCacheStats                                                                  stats_;
    size_t                                                                                limit_          = 0;
    size_t                                                                                resident_bytes_ = 0;
    uint64_t                                                                              use_clock_      = 0;
    bool                                                                                  limit_resolved_ = false;
    std::mutex                                                                            evict_mutex_;

    // One device buffer holds every resident weight (engine #669). Materializing each weight with its own
    // allocate/release made libhrx's per-size allocator pools retain the churn (~6700 releases returned nothing:
    // GTT sat at 7393 MiB with 61 MiB resident). One arena is one size class, and eviction frees a slot, not a
    // buffer. The arena grows by doubling up to the residency limit, and a growth drops every entry (the stream
    // is synchronized first), so the peak stays at the working set rather than at the knob.
    hrx_buffer_t                           arena_          = nullptr;
    size_t                                 arena_capacity_ = 0;
    size_t                                 arena_free_bytes_ = 0;
    std::vector<std::pair<size_t, size_t>> arena_free_;

    bool        arena_reserve(hrx_device_t device,
                              hrx_stream_t stream,
                              size_t       size,
                              hrx_buffer_t & buffer,
                              size_t &     offset,
                              size_t &     slot);
    void        arena_reclaim_locked(size_t offset, size_t slot);
    bool        arena_take_locked(size_t aligned, hrx_buffer_t & buffer, size_t & offset, size_t & slot);
    // Free the oldest residents until their slots add up to `aligned`; the caller then retries the take.
    bool        arena_make_room(hrx_stream_t stream, size_t aligned);
    size_t                                    executions_    = 0;
    std::vector<std::pair<size_t, size_t>>    arena_deferred_;
    // The bytes the arena may hand out: its own capacity, so eviction reacts to the buffer that exists rather
    // than to the knob. Called with `mutex_` held.
    size_t      arena_budget_locked() const;

    void        resolve_limit();
    // Drop least recently used entries until `needed` more bytes fit. Called with `mutex_` NOT held; it
    // synchronizes the stream before freeing anything, because a dispatch from an earlier command program may
    // still be reading the buffer.
    void        evict_for(hrx_stream_t stream, size_t needed);
};

// RAII over HostWeightCache::begin_execution/end_execution for one command-program execution.
class HostWeightExecutionScope {
  public:
    explicit HostWeightExecutionScope(HostWeightCache * cache) : cache_(cache) {
        if (cache_ != nullptr) {
            cache_->begin_execution();
        }
    }
    ~HostWeightExecutionScope() {
        if (cache_ != nullptr) {
            cache_->end_execution();
        }
    }

    HostWeightExecutionScope(const HostWeightExecutionScope &)             = delete;
    HostWeightExecutionScope & operator=(const HostWeightExecutionScope &) = delete;

  private:
    HostWeightCache * cache_ = nullptr;
};

struct HostStagingBuffer {
    HostStagingBuffer() = default;
    ~HostStagingBuffer();

    HostStagingBuffer(HostStagingBuffer && other) noexcept;
    HostStagingBuffer & operator=(HostStagingBuffer && other) noexcept;

    HostStagingBuffer(const HostStagingBuffer &)             = delete;
    HostStagingBuffer & operator=(const HostStagingBuffer &) = delete;

    hrx_buffer_t buffer    = nullptr;
    void *       host_data = nullptr;
    HostBufferRef source_host_buffer;
    int32_t      value     = -1;
    size_t       length    = 0;
    bool         upload    = false;
    bool         download  = false;

    void clear();
};

Status allocate_host_staging_buffer(hrx_device_t device, size_t size, HostStagingBuffer & staging);
Status allocate_mapped_host_staging_buffer(hrx_device_t device,
                                           size_t       size,
                                           hrx_buffer_t & buffer,
                                           void *&      host_data);

}  // namespace ggml::hrx
