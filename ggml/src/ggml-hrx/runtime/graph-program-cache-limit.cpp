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

#include "graph-program-cache-limit.h"

#include "backend-context.h"
#include "ggml-impl.h"
#include "graph-program-budget.h"
#include "runtime/hrx-sleeping-wait.h"

#include <atomic>
#include <cstdlib>

namespace ggml::hrx {

size_t graph_program_cache_limit() {
    static const size_t limit = [] {
        const char * value = std::getenv("GGML_HRX_GRAPH_PROGRAM_CACHE");
        if (value == nullptr || *value == '\0') {
            return size_t{ 64 };
        }
        char *                   end    = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end != nullptr && *end == '\0' ? static_cast<size_t>(parsed) : size_t{ 64 };
    }();
    return limit;
}

bool trim_graph_programs(ggml_backend_hrx_context & context, GraphProgram * current, uint64_t builds_before) {
    const size_t max_programs = graph_program_cache_limit();
    const size_t budget       = graph_program_cache_budget_bytes();
    if (current == nullptr || context.graph_programs.stats().builds == builds_before) {
        return true;
    }
    // A program's transient arena stays alive while it is cached, about 1 GiB for a large prefill
    // shape, so the count alone does not bound the GTT: evict least recently used programs until both
    // the count and the byte budget hold, never the program being executed.
    const std::vector<uint64_t> victims = context.graph_programs.plan_evictions(budget, max_programs, current);
    if (victims.empty()) {
        return true;
    }
    static thread_local WaitHistory history;
    hrx_status_t status = stream_synchronize_sleeping(context.stream, history);
    if (!hrx_status_is_ok(status)) {
        GGML_LOG_ERROR("%s: stream wait failed; graph programs kept\n", __func__);
        hrx_status_ignore(status);
        return false;
    }
    context.graph_replay_state.mark_stream_synchronized();
    const size_t held_before   = context.graph_programs.size();
    const size_t bytes_before  = context.graph_programs.bytes();
    context.graph_programs.evict(victims);
    context.prepared_programs.clear();
    // one line per trim: it should come when a new shape pushes the cache over, not during steady
    // decode that reuses the programs already there
    static std::atomic<uint64_t> trims{ 0 };
    GGML_LOG_WARN("%s: HRX graph program cache trimmed (%zu of %zu programs, %zu -> %zu MiB of arenas, budget "
                  "%zu MiB, %zu max): kept the current one, trim %llu\n",
                  __func__, victims.size(), held_before, bytes_before >> 20, context.graph_programs.bytes() >> 20,
                  budget >> 20, max_programs, static_cast<unsigned long long>(++trims));
    return true;
}

}  // namespace ggml::hrx
