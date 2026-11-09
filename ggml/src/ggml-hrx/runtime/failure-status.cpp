#include "runtime/failure-status.h"

#include <cstring>

namespace ggml::hrx {

namespace {

// iree's status name for a pool that cannot serve the request, and HSA's for the same condition one layer down. Both appear
// verbatim in the failures this exists for:
//   al...: RESOURCE_EXHAUSTED; [hsa_amd_memory_pool_allocate] HSA_STATUS_ERROR_OUT_OF_RESOURCES
constexpr const char * kResourceNeedles[] = { "RESOURCE_EXHAUSTED", "OUT_OF_RESOURCES" };

}  // namespace

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
