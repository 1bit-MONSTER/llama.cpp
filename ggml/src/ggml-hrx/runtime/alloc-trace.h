// DIAGNOSTIC — not part of any PR, not in the merged base. Built only in this lane's tree to attribute a device Buffer
// Object seen in the `amdgpu_bo_create` tracepoint to the fork allocation site that asked for it: the tracepoint carries
// a size, a thread name and a pid, but no origin, and the sizes below 4 MiB are suballocated inside slabs so a histogram
// cannot separate them (FACTS-memory §7.10, §8.1). With GGML_HRX_ALLOC_TRACE set, every device-side allocation the
// ggml-hrx runtime makes prints one line with its request size and its call site. Off by default; no behavioural change
// (the only effect is one fprintf per allocation, and only when the variable is set).
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace ggml::hrx {

inline void alloc_trace(const char * tag, size_t size) {
    static const bool enabled = std::getenv("GGML_HRX_ALLOC_TRACE") != nullptr;
    if (enabled) {
        std::fprintf(stderr, "hrx alloc trace: %-22s size=%zu (%.3f MiB)\n", tag, size,
                     static_cast<double>(size) / (1024.0 * 1024.0));
    }
}

// Detail variant: the same line plus whatever an allocation site knows about *what* it is allocating. Used for the
// host-weight materialisation path, where the request length is a packed/transformed length and cannot be matched to a
// GGUF tensor by arithmetic alone (FACTS-memory §7.20(c)).
inline void alloc_trace_detail(const char * tag, size_t size, const char * layout, int source_type, long long in_size,
                               long long out_size, size_t source_length) {
    static const bool enabled = std::getenv("GGML_HRX_ALLOC_TRACE") != nullptr;
    if (enabled) {
        std::fprintf(stderr,
                     "hrx alloc trace: %-22s size=%zu (%.3f MiB) layout=%s type=%d in=%lld out=%lld src=%zu\n", tag, size,
                     static_cast<double>(size) / (1024.0 * 1024.0), layout != nullptr ? layout : "?", source_type,
                     in_size, out_size, source_length);
    }
}

}  // namespace ggml::hrx
