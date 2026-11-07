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

#include "graph-program-budget.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ggml::hrx {

namespace {

// Which program the byte bound drops first (engine #841). Age is the shipped order; "size" drops the largest first
// so that each trim actually moves the byte budget, because retention is skewed: the measured trims freed 0-1 MiB
// per program under the age order while the largest programs hold tens of MiB each. Read per call rather than
// cached, so a host test can flip it between cases; a trim happens when a new shape arrives, not per token.
bool evict_largest_first() {
    const char * policy = std::getenv("GGML_HRX_GRAPH_PROGRAM_EVICT");
    return policy != nullptr && std::strcmp(policy, "size") == 0;
}

}  // namespace

size_t parse_graph_program_cache_budget_bytes(const char * mib) {
    if (mib == nullptr || *mib == '\0') {
        return size_t{ 8192 } << 20;
    }
    char *                   end    = nullptr;
    const unsigned long long parsed = std::strtoull(mib, &end, 10);
    if (end == nullptr || *end != '\0') {
        return size_t{ 8192 } << 20;
    }
    return static_cast<size_t>(parsed) << 20;
}

size_t graph_program_cache_budget_bytes() {
    static const size_t bytes = parse_graph_program_cache_budget_bytes(std::getenv("GGML_HRX_GRAPH_PROGRAM_CACHE_MB"));
    return bytes;
}

const char * graph_program_evict_policy_name() {
    return evict_largest_first() ? "size" : "age";
}

void GraphProgramBudget::note(uint64_t uid, size_t bytes) {
    entries_[uid] = Entry{ bytes, ++tick_ };
}

void GraphProgramBudget::touch(uint64_t uid) {
    const auto found = entries_.find(uid);
    if (found != entries_.end()) {
        found->second.used = ++tick_;
    }
}

void GraphProgramBudget::resize(uint64_t uid, size_t bytes) {
    const auto found = entries_.find(uid);
    if (found != entries_.end()) {
        found->second.bytes = bytes;
    }
}

void GraphProgramBudget::forget(uint64_t uid) {
    entries_.erase(uid);
}

size_t GraphProgramBudget::total_bytes() const {
    size_t total = 0;
    for (const auto & entry : entries_) {
        total += entry.second.bytes;
    }
    return total;
}

size_t GraphProgramBudget::size() const {
    return entries_.size();
}

std::vector<uint64_t> GraphProgramBudget::plan_evictions(size_t budget,
                                                         size_t max_programs,
                                                         uint64_t keep,
                                                         bool     has_keep) const {
    std::vector<uint64_t> victims;
    size_t                total  = 0;
    size_t                count  = 0;
    std::vector<std::pair<uint64_t, uint64_t>> by_age;  // (used, uid), oldest first
    by_age.reserve(entries_.size());
    for (const auto & entry : entries_) {
        if (has_keep && entry.first == keep) {
            continue;
        }
        total += entry.second.bytes;
        ++count;
        by_age.emplace_back(entry.second.used, entry.first);
    }
    const bool over_bytes  = budget != 0 && total > budget;
    const bool over_count  = max_programs != 0 && count > max_programs;
    if (!over_bytes && !over_count) {
        return victims;
    }
    std::sort(by_age.begin(), by_age.end());
    if (over_bytes && evict_largest_first()) {
        // The byte bound is about bytes, so drop the programs that give the most back. There is one victim list, so
        // when a single trim has to satisfy both bounds this order drives the count bound's victims too: the byte
        // constraint is the one that needs the most drops to satisfy, and a largest-first pass satisfies both sooner
        // than an oldest-first pass satisfies the bytes (engine #841). With the byte bound not tripped the count bound
        // is untouched and stays oldest-first, which is the question age answers.
        std::stable_sort(by_age.begin(), by_age.end(), [this](const auto & a, const auto & b) {
            return entries_.at(a.second).bytes > entries_.at(b.second).bytes;
        });
    }
    for (const auto & candidate : by_age) {
        if ((!over_bytes || total <= budget) && (!over_count || count <= max_programs)) {
            break;
        }
        victims.push_back(candidate.second);
        const auto found = entries_.find(candidate.second);
        if (found != entries_.end()) {
            total -= found->second.bytes;
        }
        if (count > 0) {
            --count;
        }
    }
    return victims;
}

}  // namespace ggml::hrx
