#pragma once

#include "ggml.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
typedef struct hrx_buffer_s * hrx_buffer_t;

namespace ggml::hrx {

struct ValueId {
    ValueId() : value(-1) {}

    explicit ValueId(int32_t value) : value(value) {}

    int32_t value;
};

struct ValueStorageId {
    ValueStorageId() : value(-1) {}

    explicit ValueStorageId(int32_t value) : value(value) {}

    int32_t value;
};

inline bool operator==(ValueId lhs, ValueId rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(ValueId lhs, ValueId rhs) {
    return !(lhs == rhs);
}

inline bool operator==(ValueStorageId lhs, ValueStorageId rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(ValueStorageId lhs, ValueStorageId rhs) {
    return !(lhs == rhs);
}

// Recorded when get_or_add_tensor_value() creates a storage rather than aliasing (review of #262).
// How the alias source was (or was not) found, recorded when the storage is chosen. The two "found" cases are the
// interesting ones: with an alias source the view joins *that* value's storage, which need not be the storage of the
// node's own input - and that is what makes is_layout_alias_node() false while an alias source exists at all
// (review of #262: the earlier ViewSrcRegisteredStorageDiffers label was unreachable, because a registered view_src is
// an early return in find_alias_source()).
enum class AliasMiss : uint8_t {
    None,            // not a view, so there is nothing to alias
    NoAliasSource,   // view_src unregistered and no value shares its storage root: a storage of its own
    ExactViewSrc,    // the alias source was the view_src value itself
    RangeFallback,   // the alias source was found by the storage-root fallback, not by identity
};

enum class ValueKind : uint8_t {
    External,
    Transient,
};

struct ValueBufferBinding {
    // A buffer is directly bindable by an HRX command program. Host data requires residency or staging before
    // execution. These are alternate storage forms and should not both be populated.
    hrx_buffer_t buffer     = nullptr;
    size_t       offset     = 0;
    size_t       length     = 0;
    uint64_t     identity   = 0;
    uint64_t     generation = 0;
    size_t       capacity   = 0;
    void *       host_data  = nullptr;
    bool         weight     = false;

    bool requires_materialization() const { return host_data != nullptr; }
};

struct Value {
    ValueId                            id;
    ValueKind                          kind;
    ValueStorageId                     storage;
    ValueId                            storage_root;
    ValueId                            alias_source;
    size_t                             storage_offset     = 0;
    size_t                             storage_byte_count = 0;
    ggml_type                          type;
    std::array<int64_t, GGML_MAX_DIMS> ne;
    std::array<size_t, GGML_MAX_DIMS>  nb;
    int64_t                            element_count = 0;
    size_t                             byte_count    = 0;
    bool                               contiguous    = false;
    const ggml_tensor *                tensor        = nullptr;
    // Why this value was given a storage of its own instead of aliasing its input, recorded AT ASSIGNMENT TIME.
    // A diagnostic that asks the map later gets a different answer: values added between the assignment and the
    // message make `find_tensor(view_src)` succeed where it failed then, and the message would then name the wrong
    // repair (review of #262).
    AliasMiss                          alias_miss    = AliasMiss::None;
    std::optional<ValueBufferBinding>  buffer;
};

struct ValueStorage {
    ValueStorageId id;
    ValueId        root;
    size_t         byte_count = 0;
};

class ValueMap {
  public:
    ValueMap() = default;

    ValueId get_or_add_tensor_value(const ggml_tensor * tensor, ValueKind kind);

    const Value *                     find(ValueId id) const;
    const Value *                     find_tensor(const ggml_tensor * tensor) const;
    const ValueStorage *              find_storage(ValueStorageId id) const;
    bool                              bind_buffer(ValueId id, ValueBufferBinding binding);
    std::optional<ValueBufferBinding> resolve_buffer_binding(ValueId id) const;
    std::vector<ValueId>              external_value_ids() const;
    Status                            alias_storage(ValueId target, ValueId source);
    ValueId                           storage_root(ValueId id) const;
    bool                              same_storage(ValueId lhs, ValueId rhs) const;

    const std::vector<Value> & values() const { return values_; }

    const std::vector<ValueStorage> & storages() const { return storages_; }

    size_t size() const { return values_.size(); }

    Status add_snapshot_storage(ValueStorage storage);
    Status add_snapshot_value(Value value);

  private:
    const Value * find_alias_source(const ggml_tensor * tensor) const;
    void          promote_storage_root_external(ValueId storage_root);

    std::vector<Value>                              values_;
    std::vector<ValueStorage>                       storages_;
    std::unordered_map<const ggml_tensor *, size_t> tensor_values_;
};

}  // namespace ggml::hrx
