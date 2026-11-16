#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_get_rows_dispatches(DispatchRegistryBuilder & registry);

// Bounded per-node claim for GET_ROWS (engine#2704): the shapes THIS file's matchers execute -- the rows predicate and
// the narrow scalar_3d one -- and nothing else for a source they gate. Consulted from device_supports_op so a shape a
// matcher declines keeps its CPU fallback instead of becoming an unsupported HRX node at compute time.
bool get_rows_tensor_supported(const struct ggml_tensor * op);

}  // namespace ggml::hrx
