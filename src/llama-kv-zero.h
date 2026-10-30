// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// The padding of a KV cache must be finite (engine #528).
//
// Every attention kernel masks the columns past the sequence end with -INFINITY and relies on those columns
// holding zeros: `-infinity + NaN` is NaN and `0 * NaN` is NaN, so one padded column with garbage in it makes
// a whole output row NaN. llama.cpp clears the KV buffer when it allocates it
// ("allocate tensors and initialize the buffers to avoid NaNs in the padding"), but on some backends that
// clear is best-effort - ggml-hrx logs a failed fill and carries on - and a clear that did not take leaves
// whatever the device memory held before.
//
// This verifies the clear instead of trusting it, and fills the buffer through the backend's own tensor writes
// when it did not take: at allocation, and on every reset - which is once per request in a server, because
// llama-server clears the cache for each one. That is why the fast path samples a few tensors (a handful of
// readbacks) instead of scanning the buffer, and why a repair, which is rare, verifies all of it.

#include "ggml-backend.h"

#include <cstddef>

struct ggml_context;

// Returns true when the buffer reads zero afterwards. `what` names the caller for the log line ("KV cache",
// "KV cache reset", ...). A false return means the buffer could not be made zero: the caller must not serve
// from it.
//
// The fast path samples three windows in each of a few tensors spread over the buffer, because the clear is
// one fill submit per buffer; a repair (the clear did not take) fills everything and verifies everything.
bool llama_kv_zero_fill(ggml_backend_buffer_t buf, ggml_context * ctx, const char * what);
