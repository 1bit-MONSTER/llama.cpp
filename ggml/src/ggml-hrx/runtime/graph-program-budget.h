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

// A byte budget for the HRX graph program cache (graph-program-cache.h). A cached program keeps its
// transient arena alive, about 1 GiB for a large prefill shape, and the cache used to hold every
// shape it ever built: a long-lived server answering prompts of varying length grew its GTT without
// a bound (112 GiB was seen on 2026-10-01). The count limit in graph-program-cache-limit.h bounds the
// number of programs but not their size, and drops everything but the current program when it trips.
// This bounds the bytes and drops least-recently-used programs only.
//
// GGML_HRX_GRAPH_PROGRAM_CACHE_MB: budget in MiB (default 8192: a 27B model's hot shapes plus another
// model's fit; 0 keeps every program, the old behaviour).

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

size_t graph_program_cache_budget_bytes();

// The parser behind it, so a host test can check the values without touching the environment:
// MiB in, bytes out; 0 means unbounded; anything unparsable is the 8192 MiB default.
size_t parse_graph_program_cache_budget_bytes(const char * mib);

class GraphProgramBudget {
  public:
    // A program entered the cache (or its size changed): it is now the most recently used.
    void note(uint64_t uid, size_t bytes);
    // A lookup reused it: now the most recently used.
    void touch(uint64_t uid);
    // It left the cache by another path.
    void forget(uint64_t uid);

    size_t total_bytes() const;
    size_t size() const;

    // The programs to drop so that at most `budget` bytes and `max_programs` entries remain, least
    // recently used first. `keep` (when `has_keep`) is never returned and is not counted against the
    // budget, so the program being executed cannot be swapped out from under the caller. Empty when
    // nothing has to go; `budget == 0` and `max_programs == 0` mean no limit.
    std::vector<uint64_t> plan_evictions(size_t budget, size_t max_programs, uint64_t keep, bool has_keep) const;

  private:
    struct Entry {
        size_t   bytes = 0;
        uint64_t used  = 0;
    };

    std::unordered_map<uint64_t, Entry> entries_;
    uint64_t                            tick_ = 0;
};

}  // namespace ggml::hrx
