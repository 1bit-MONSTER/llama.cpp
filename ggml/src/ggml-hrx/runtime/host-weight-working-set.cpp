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

#include "host-weight-working-set.h"

#include <map>
#include <mutex>
#include <utility>

namespace ggml::hrx {
namespace {

using WeightKey = std::pair<uint64_t, uint64_t>;

std::mutex                       working_set_mutex;
std::map<WeightKey, size_t>     working_set_bytes;
size_t                           working_set_total = 0;

}  // namespace

void host_weight_working_set_observe(uint64_t identity, uint64_t generation, size_t materialized_bytes) {
    if (materialized_bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(working_set_mutex);
    const WeightKey             key{ identity, generation };
    const auto                  found = working_set_bytes.find(key);
    if (found == working_set_bytes.end()) {
        working_set_bytes.emplace(key, materialized_bytes);
        working_set_total += materialized_bytes;
        return;
    }
    if (materialized_bytes > found->second) {
        working_set_total += materialized_bytes - found->second;
        found->second = materialized_bytes;
    }
}

size_t host_weight_working_set_bytes() {
    std::lock_guard<std::mutex> lock(working_set_mutex);
    return working_set_total;
}

bool host_weight_working_set_exceeds(size_t limit) {
    if (limit == 0) {
        return false;
    }
    return host_weight_working_set_bytes() > limit;
}

void host_weight_working_set_reset() {
    std::lock_guard<std::mutex> lock(working_set_mutex);
    working_set_bytes.clear();
    working_set_total = 0;
}

}  // namespace ggml::hrx
