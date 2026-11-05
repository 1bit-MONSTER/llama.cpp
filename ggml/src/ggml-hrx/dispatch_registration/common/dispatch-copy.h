#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_copy_dispatch(DispatchRegistryBuilder & registry);

// Fill `dispatch` with a copy of `source` into a packed F32 transient of `destination_bytes` at `destination`, going through
// the strided-source kernel when the rows are not packed. Used by matchers that need a packed activation but must not
// reject a graph just because the producer handed them a row-padded view. False when the source cannot be expressed.
bool append_f32_densify_dispatch(const Value & source, ValueId destination, size_t destination_bytes, Dispatch & dispatch);

}  // namespace ggml::hrx
