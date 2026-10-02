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

// A bound on the graph programs one HRX backend keeps. Every new graph shape (each new prompt or
// ubatch length) builds a GraphProgram that retains its HRX buffers, recorded graph and transient
// arena binding, and GraphProgramCache never evicted: a llama-server answering prompts of varying
// length grew about 1 GiB of GTT per request (ZAYA1-8B, 2026-10-01) until the box ran out of memory.

#include <cstddef>
#include <cstdint>

struct ggml_backend_hrx_context;

namespace ggml::hrx {

class GraphProgram;

// GGML_HRX_GRAPH_PROGRAM_CACHE: programs kept per backend (default 64; 0 keeps them all, the old
// behaviour).
size_t graph_program_cache_limit();

// After a lookup: if it built a new program (the cache's build count moved past builds_before) and
// the cache now holds more than the limit, wait for the stream (no command of a dropped program may
// still be running), then drop every cached program except `current`, and the prepared programs.
// A lookup that reused a program, by uid or by structure, never drops anything, so steady decode
// does not churn. Returns false only if the stream wait failed; the caches are then kept.
bool trim_graph_programs(ggml_backend_hrx_context & context, GraphProgram * current, uint64_t builds_before);

}  // namespace ggml::hrx
