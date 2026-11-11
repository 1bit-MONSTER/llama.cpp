#include "runtime/failure-status.h"
#include "hrx-1bit-safety.h"   // engine#2069: the terms the refusal is worded from

#include <cstdio>

#include <cctype>
#include <cstring>

namespace ggml::hrx {

namespace {

// iree's status name for a pool that cannot serve the request, and HSA's for the same condition one layer down. Both appear
// verbatim in the failures this exists for:
//   al...: RESOURCE_EXHAUSTED; [hsa_amd_memory_pool_allocate] HSA_STATUS_ERROR_OUT_OF_RESOURCES
// kPoolRefusalToken is the same condition one layer UP: the pool ledger refused before the allocator was asked.
constexpr const char * kResourceNeedles[] = { "RESOURCE_EXHAUSTED", "OUT_OF_RESOURCES", kPoolRefusalToken };

// Case-insensitive on purpose: the *classification* needles are exact because they are identifiers as iree and HSA render
// them, but the *hint* exists to catch spellings we have not seen -- `hipErrorOutOfMemory` is real hip and mixed case, and a
// case-sensitive match would have missed exactly the case this probe was added for (the test caught that).
// "pool" is here for our own refusals: one written without kPoolRefusalToken is then at least named in the log.
constexpr const char * kAllocationHints[] = { "memory", "alloc", "pool" };

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

std::string pool_refusal_text(const char * device_name, size_t device_total, size_t device_free, size_t ceiling_bytes,
                              const onebit::pool_budget_terms_t & terms) {
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    const char * name = device_name ? device_name : "(unnamed)";
    char buf[1024];
    // The host term wins when it is smaller than the GTT term, and then GTT is not what to free. The GTT figures are still
    // printed, because they are part of why the budget is what it is, but they are labelled as the non-binding side.
    if (terms.host_bound) {
        // A measured zero is a symptom of its own (engine#1942: an argument-evaluation-order bug made a readable figure read as
        // zero), so it says "measured 0.0 GiB" rather than printing a number that looks like a reading.
        snprintf(buf, sizeof buf,
                 "ggml_hrx: %s: refusing to use the device: host MemAvailable %.1f GiB is below the GTT term "
                 "(GTT total %.1f GiB with %.1f GiB free, ceiling %.1f GiB), so HOST memory is what is binding here; "
                 "free RAM%s (engine#680, engine#2069)\n",
                 name, terms.host_available / GiB, device_total / GiB, device_free / GiB, ceiling_bytes / GiB,
                 terms.host_available == 0 ? " - and note the host figure measured 0.0 GiB, which is a reported zero rather than a reading"
                                           : "");
        return std::string(buf);
    }
    snprintf(buf, sizeof buf,
             "ggml_hrx: %s: refusing to use the device: GTT total %.1f GiB with %.1f GiB free leaves nothing under the "
             "%.1f GiB ceiling (reserve GGML_HRX_GTT_RESERVE_MB, override GGML_HRX_POOL_LIMIT_MB); free GTT or lower the "
             "reserve (engine#680)\n",
             name, device_total / GiB, device_free / GiB, ceiling_bytes / GiB);
    return std::string(buf);
}

}  // namespace ggml::hrx
