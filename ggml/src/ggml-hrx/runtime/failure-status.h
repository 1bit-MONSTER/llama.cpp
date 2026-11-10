#pragma once

#include "ggml.h"

#include <string>
#include <vector>

namespace ggml::hrx {

// Whether a backend failure is a *resource* failure rather than a compute one (engine #1618).
//
// The distinction matters outside this backend: `src/llama-context.cpp` maps GGML_STATUS_ALLOC_FAILED to llama_decode's -2
// and GGML_STATUS_FAILED to -3, and the server prints a different sentence for each. Before this, every HRX failure left
// `GraphExecutionResult`'s default GGML_STATUS_FAILED, so a host-memory allocation the pool could not serve reached the client
// as "Compute error." - i.e. a resource failure read as a kernel fault, which is how #1618 was filed and how it stayed
// misdiagnosed for a day (the log said HSA_STATUS_ERROR_OUT_OF_RESOURCES while the client said "Compute error").
//
// Text-based on purpose: the code lives in libhrx's `iree_status_t`, and `status.h` keeps only the rendered messages by the
// time a failure reaches this backend. The two strings matched below are iree's own status name and HSA's error name, so they
// survive rewording of the prose around them.
bool failure_looks_like_allocation(const std::vector<std::string> & messages);

// The third identifier, and the only one this backend writes itself: the token a compute-time refusal by the pool LEDGER
// carries in its message. The two names above come from the allocator, i.e. from a launch the guardrail admitted and the
// device then could not serve. Since engine #1711/#1823 the ledger refuses first (the transient arena, a host staging buffer,
// a materialized host weight), and those refusals are this backend's own prose: "refused: a 121.7 MiB host weight does not
// fit the pool" names neither RESOURCE_EXHAUSTED nor OUT_OF_RESOURCES, so the better the accounting got, the more often an
// out-of-memory condition went back to being "Compute error." for the client. A refusal site appends this token instead of
// relying on its wording, so rewording a sentence cannot change the status.
inline constexpr char kPoolRefusalToken[] = "POOL_EXHAUSTED";

// A message that *looks* like an allocation failure (it mentions memory, allocation or the pool) but matched no needle, so it is
// about to be classified as a compute failure (engine #1618 review). Returns nullptr when nothing looks suspicious. This exists
// so the next spelling of an out-of-memory status -- `hipErrorOutOfMemory`, `HSA_STATUS_ERROR_INVALID_ALLOCATION`, whatever the
// runtime renders tomorrow -- costs a log line rather than another day of misdiagnosis: the fallback stays the safe direction
// (a false OOM is worse), but it stops being silent.
const std::string * failure_unmatched_allocation(const std::vector<std::string> & messages);

// The ggml status to return for these failure messages: GGML_STATUS_ALLOC_FAILED when they describe a resource failure,
// |fallback| otherwise. graph_compute() passes its own code (so a caller that already has one keeps it); the default is what a
// caller with nothing better to say gets, and it is what the test exercises for the preserved-code case.
enum ggml_status failure_status(const std::vector<std::string> & messages, enum ggml_status fallback = GGML_STATUS_FAILED);

}  // namespace ggml::hrx
