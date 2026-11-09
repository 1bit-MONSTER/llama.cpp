#include "runtime/failure-status.h"

#include <cctype>
#include <cstring>

namespace ggml::hrx {

namespace {

// iree's status name for a pool that cannot serve the request, and HSA's for the same condition one layer down. Both appear
// verbatim in the failures this exists for:
//   al...: RESOURCE_EXHAUSTED; [hsa_amd_memory_pool_allocate] HSA_STATUS_ERROR_OUT_OF_RESOURCES
constexpr const char * kResourceNeedles[] = { "RESOURCE_EXHAUSTED", "OUT_OF_RESOURCES" };

// Case-insensitive on purpose: the *classification* needles are exact because they are identifiers as iree and HSA render
// them, but the *hint* exists to catch spellings we have not seen -- `hipErrorOutOfMemory` is real hip and mixed case, and a
// case-sensitive match would have missed exactly the case this probe was added for (the test caught that).
constexpr const char * kAllocationHints[] = { "memory", "alloc" };

bool contains_any_of_ci(const std::string & message, const char * const * needles, size_t count) {
    std::string lowered;
    lowered.reserve(message.size());
    for (char c : message) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    for (size_t i = 0; i < count; ++i) {
        if (lowered.find(needles[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

const std::string * failure_unmatched_allocation(const std::vector<std::string> & messages) {
    if (failure_looks_like_allocation(messages)) {
        return nullptr;  // it matched: nothing to warn about
    }
    for (const std::string & message : messages) {
        if (contains_any_of_ci(message, kAllocationHints, sizeof(kAllocationHints) / sizeof(kAllocationHints[0]))) {
            return &message;
        }
    }
    return nullptr;
}

bool failure_looks_like_allocation(const std::vector<std::string> & messages) {
    for (const std::string & message : messages) {
        for (const char * needle : kResourceNeedles) {
            if (std::strstr(message.c_str(), needle) != nullptr) {
                return true;
            }
        }
    }
    return false;
}

enum ggml_status failure_status(const std::vector<std::string> & messages, enum ggml_status fallback) {
    return failure_looks_like_allocation(messages) ? GGML_STATUS_ALLOC_FAILED : fallback;
}

}  // namespace ggml::hrx
