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

namespace ggml::hrx {

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

void GraphProgramBudget::note(uint64_t uid, size_t bytes) {
    entries_[uid] = Entry{ bytes, ++tick_ };
}

void GraphProgramBudget::touch(uint64_t uid) {
    const auto found = entries_.find(uid);
    if (found != entries_.end()) {
        found->second.used = ++tick_;
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
