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

// Host test for graph-program-budget.h: the byte budget and LRU order of the HRX graph program
// cache. No device, no HRX header, no model: the policy is pure.

#include "runtime/graph-program-budget.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

using ggml::hrx::GraphProgramBudget;

static const size_t MiB = size_t{ 1 } << 20;

static std::vector<uint64_t> plan(const GraphProgramBudget & budget,
                                  size_t                     bytes,
                                  size_t                     max_programs,
                                  uint64_t                   keep,
                                  bool                       has_keep) {
    return budget.plan_evictions(bytes, max_programs, keep, has_keep);
}

int main() {
    {
        // nothing cached, nothing to evict
        GraphProgramBudget budget;
        REQUIRE(plan(budget, 8 * MiB, 64, 0, false).empty());
        REQUIRE(budget.size() == 0);
        REQUIRE(budget.total_bytes() == 0);
    }
    {
        // under the byte budget and the count: no eviction, and steady reuse touches nothing away
        GraphProgramBudget budget;
        budget.note(1, 1 * MiB);
        budget.note(2, 1 * MiB);
        budget.touch(1);
        REQUIRE(plan(budget, 8 * MiB, 64, 0, false).empty());
        REQUIRE(budget.total_bytes() == 2 * MiB);
    }
    {
        // over the byte budget: least recently used first, until the total fits
        GraphProgramBudget budget;
        budget.note(1, 1 * MiB);  // oldest
        budget.note(2, 1 * MiB);
        budget.note(3, 1 * MiB);
        budget.touch(1);  // 1 is now the newest: the order is 2, 3, 1
        const auto victims = plan(budget, 2 * MiB, 0, 0, false);
        REQUIRE(victims.size() == 1);
        REQUIRE(victims[0] == 2);
        const auto both = plan(budget, 1 * MiB, 0, 0, false);
        REQUIRE(both.size() == 2);
        REQUIRE(both[0] == 2);
        REQUIRE(both[1] == 3);
    }
    {
        // the program being executed is never a victim, even when it is the oldest
        GraphProgramBudget budget;
        budget.note(7, 4 * MiB);  // the current program, and the oldest
        budget.note(8, 1 * MiB);
        budget.note(9, 4 * MiB);
        const auto victims = plan(budget, 1 * MiB, 0, 7, true);
        REQUIRE(victims.size() == 2);
        for (uint64_t uid : victims) {
            REQUIRE(uid != 7);
        }
        // and with only the current program cached, nothing is left to evict
        GraphProgramBudget only_current;
        only_current.note(7, 64 * MiB);
        REQUIRE(plan(only_current, 1 * MiB, 0, 7, true).empty());
    }
    {
        // a single program larger than the whole budget is evicted when it is not the current one
        GraphProgramBudget budget;
        budget.note(4, 32 * MiB);
        const auto victims = plan(budget, 8 * MiB, 0, 0, false);
        REQUIRE(victims.size() == 1);
        REQUIRE(victims[0] == 4);
    }
    {
        // the count cap still holds on its own (the old GGML_HRX_GRAPH_PROGRAM_CACHE behaviour)
        GraphProgramBudget budget;
        for (uint64_t uid = 1; uid <= 5; ++uid) {
            budget.note(uid, 1 * MiB);
        }
        const auto victims = plan(budget, 0, 2, 0, false);
        REQUIRE(victims.size() == 3);  // 5 -> 2 entries
        REQUIRE(victims[0] == 1);      // oldest first
        REQUIRE(victims[2] == 3);
    }
    {
        // 0 means no limit on either axis, the old unbounded behaviour
        GraphProgramBudget budget;
        for (uint64_t uid = 1; uid <= 200; ++uid) {
            budget.note(uid, 1 * MiB);
        }
        REQUIRE(plan(budget, 0, 0, 0, false).empty());
    }
    {
        // note() on a known program updates its size and its place in the order; forget() drops it
        GraphProgramBudget budget;
        budget.note(1, 1 * MiB);
        budget.note(2, 1 * MiB);
        budget.note(1, 2 * MiB);
        REQUIRE(budget.size() == 2);
        REQUIRE(budget.total_bytes() == 3 * MiB);
        budget.forget(2);
        REQUIRE(budget.size() == 1);
        REQUIRE(budget.total_bytes() == 2 * MiB);
        REQUIRE(plan(budget, 2 * MiB, 0, 0, false).empty());
    }
    {
        // resize() changes the byte count without changing the LRU (engine #841: what a cached program retains is
        // re-derived before planning evictions, and doing that must not make every program look freshly used)
        GraphProgramBudget budget;
        budget.note(1, 1 * MiB);  // oldest
        budget.note(2, 1 * MiB);
        budget.note(3, 1 * MiB);
        // re-derived in reverse, so a resize() that reordered the LRU would leave 3 as the oldest and be caught here
        for (uint64_t uid = 3; uid >= 1; --uid) {
            budget.resize(uid, 8 * MiB);
        }
        REQUIRE(budget.size() == 3);
        REQUIRE(budget.total_bytes() == 24 * MiB);
        const auto victims = plan(budget, 16 * MiB, 0, 0, false);
        REQUIRE(victims.size() == 1);
        REQUIRE(victims[0] == 1);  // still the oldest, so resize() left the order alone
        // and it is a no-op on a uid that is not cached, not a new entry
        budget.resize(42, 64 * MiB);
        REQUIRE(budget.size() == 3);
        REQUIRE(budget.total_bytes() == 24 * MiB);
    }
    {
        // the byte-bound order is selectable (engine #841): "size" drops the largest first so each trim returns
        // real bytes, age keeps the shipped order, and the count bound stays oldest-first either way
        GraphProgramBudget budget;
        budget.note(1, 1 * MiB);   // oldest, small
        budget.note(2, 2 * MiB);
        budget.note(3, 32 * MiB);  // newest, and the only one that matters to a byte budget
        ::setenv("GGML_HRX_GRAPH_PROGRAM_EVICT", "size", 1);
        const auto by_size = plan(budget, 4 * MiB, 0, 0, false);
        REQUIRE(by_size.size() == 1);
        REQUIRE(by_size[0] == 3);  // largest first, not oldest first
        const auto by_size_both = plan(budget, 2 * MiB, 0, 0, false);
        REQUIRE(by_size_both.size() == 2);  // 32 then 2 leaves 1 MiB, which fits, so it stops there
        REQUIRE(by_size_both[0] == 3);
        REQUIRE(by_size_both[1] == 2);
        // equal sizes fall back to age, so the order stays deterministic
        GraphProgramBudget ties;
        ties.note(11, 8 * MiB);
        ties.note(12, 8 * MiB);
        ties.note(13, 8 * MiB);
        REQUIRE(plan(ties, 8 * MiB, 0, 0, false)[0] == 11);
        // the count bound is still answered by age under the same env
        REQUIRE(plan(ties, 0, 1, 0, false)[0] == 11);
        // and when ONE trim has to satisfy both bounds, this order drives the count bound's victims too: there is one
        // victim list, and the byte constraint is the one that needs the most drops. Pinned rather than left implicit,
        // because "the count bound stays oldest-first" is only true while the byte bound is not also tripped.
        REQUIRE(plan(budget, 4 * MiB, 1, 0, false).size() == 2);   // size: [3, 2] (a byte-bound trip, count also over)
        REQUIRE(plan(budget, 4 * MiB, 1, 0, false)[0] == 3);
        ::unsetenv("GGML_HRX_GRAPH_PROGRAM_EVICT");
        const auto by_age = plan(budget, 4 * MiB, 0, 0, false);
        REQUIRE(by_age.size() == 3);    // 35 MiB against 4 MiB needs all three out of the cache either way
        REQUIRE(by_age[0] == 1);        // but the shipped order still takes the oldest first
        // and an unparsable value is the age order, not a third policy
        ::setenv("GGML_HRX_GRAPH_PROGRAM_EVICT", "largest", 1);
        REQUIRE(plan(budget, 4 * MiB, 0, 0, false)[0] == 1);  // unparsable -> age order
        ::unsetenv("GGML_HRX_GRAPH_PROGRAM_EVICT");
    }
    {
        // the policy the trim log line prints has to be the policy that actually ran, or the log lies about the arm
        ::unsetenv("GGML_HRX_GRAPH_PROGRAM_EVICT");
        REQUIRE(std::strcmp(ggml::hrx::graph_program_evict_policy_name(), "age") == 0);
        ::setenv("GGML_HRX_GRAPH_PROGRAM_EVICT", "size", 1);
        REQUIRE(std::strcmp(ggml::hrx::graph_program_evict_policy_name(), "size") == 0);
        ::setenv("GGML_HRX_GRAPH_PROGRAM_EVICT", "largest", 1);
        REQUIRE(std::strcmp(ggml::hrx::graph_program_evict_policy_name(), "age") == 0);
        ::unsetenv("GGML_HRX_GRAPH_PROGRAM_EVICT");
    }
    {
        // the env parser: MiB in, bytes out, and the documented default for anything unusable
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes(nullptr) == 8192 * MiB);
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes("") == 8192 * MiB);
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes("0") == 0);          // unbounded
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes("256") == 256 * MiB);
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes("4096") == 4096 * MiB);
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes("lots") == 8192 * MiB);
        REQUIRE(ggml::hrx::parse_graph_program_cache_budget_bytes("12x") == 8192 * MiB);
    }
    std::printf("test-hrx-graph-program-budget: all checks passed\n");
    return 0;
}
