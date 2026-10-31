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

    void        resolve_limit();
    // Drop least recently used entries until `needed` more bytes fit. Called with `mutex_` NOT held; it
    // synchronizes the stream before freeing anything, because a dispatch from an earlier command program may
    // still be reading the buffer.
    void        evict_for(hrx_stream_t stream, size_t needed);
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
