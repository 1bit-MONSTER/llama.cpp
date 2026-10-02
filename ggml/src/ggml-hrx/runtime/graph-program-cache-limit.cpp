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
    const size_t limit = graph_program_cache_limit();
    if (limit == 0 || current == nullptr || context.graph_programs.stats().builds == builds_before) {
        return true;
    }
    const size_t held = context.graph_programs.size();
    if (held <= limit) {
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
    context.graph_programs.retain_only(current);
    context.prepared_programs.clear();
    // one line per flush: it should come every `limit` new shapes, never during steady decode
    static std::atomic<uint64_t> flushes{ 0 };
    GGML_LOG_WARN("%s: HRX graph program cache over its limit (%zu programs, limit %zu): kept the current one, "
                  "flush %llu\n",
                  __func__, held, limit, static_cast<unsigned long long>(++flushes));
    return true;
}

}  // namespace ggml::hrx
