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

// The ggml status to return for these failure messages: GGML_STATUS_ALLOC_FAILED when they describe a resource failure,
// |fallback| otherwise (so a caller that already has a code keeps it).
enum ggml_status failure_status(const std::vector<std::string> & messages, enum ggml_status fallback = GGML_STATUS_FAILED);

}  // namespace ggml::hrx
