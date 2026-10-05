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

#include "runtime/graph-record-order.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <unordered_map>

namespace ggml::hrx {
namespace {

constexpr size_t kMaxTransientExtent = size_t{ 16 } << 20;

bool level_order_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_GRAPH_ORDER");
        return value == nullptr || std::strcmp(value, "program") != 0;
    }();
    return enabled;
}

bool log_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_LOG_GRAPH_ORDER");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool access_writes(ResourceAccess access) {
    return access == ResourceAccess::Write || access == ResourceAccess::ReadWrite;
}

// Memory-range hazards between commands, in program order: a command depends on the last writer of
// every byte it touches and, when it writes, on every reader since that writer.
class RangeHazards {
  public:
    void collect(hrx_buffer_t buffer, size_t offset, size_t length, bool writes, std::vector<uint32_t> & deps) const {
        if (buffer == nullptr || length == 0) {
            return;
        }
        const auto found = buffers_.find(buffer);
        if (found == buffers_.end()) {
            return;
        }
        const size_t end   = range_end(offset, length);
        auto         range = found->second.upper_bound(offset);
        if (range != found->second.begin()) {
            --range;
            if (range->second.end <= offset) {
                ++range;
            }
        }
        for (; range != found->second.end() && range->first < end; ++range) {
            if (range->second.last_writer >= 0) {
                deps.push_back(static_cast<uint32_t>(range->second.last_writer));
            }
            if (writes) {
                deps.insert(deps.end(), range->second.readers.begin(), range->second.readers.end());
            }
        }
    }

    void update(uint32_t command, hrx_buffer_t buffer, size_t offset, size_t length, bool writes) {
        if (buffer == nullptr || length == 0) {
            return;
        }
        const size_t end    = range_end(offset, length);
        Ranges &     ranges = buffers_[buffer];
        split(ranges, offset);
        split(ranges, end);
        size_t cursor = offset;
        auto   range  = ranges.lower_bound(offset);
        while (cursor < end) {
            if (range == ranges.end() || range->first > cursor) {
                Segment segment;
                segment.end = range == ranges.end() ? end : std::min(end, range->first);
                if (writes) {
                    segment.last_writer = static_cast<int32_t>(command);
                } else {
                    segment.readers.push_back(command);
                }
                const size_t next = segment.end;
                range             = std::next(ranges.emplace(cursor, std::move(segment)).first);
                cursor            = next;
                continue;
            }
            if (writes) {
                range->second.last_writer = static_cast<int32_t>(command);
                range->second.readers.clear();
            } else if (range->second.readers.empty() || range->second.readers.back() != command) {
                range->second.readers.push_back(command);
            }
            cursor = range->second.end;
            ++range;
        }
    }

  private:
    struct Segment {
        size_t                end         = 0;
        int32_t               last_writer = -1;
        std::vector<uint32_t> readers;
    };
    using Ranges = std::map<size_t, Segment>;

    static size_t range_end(size_t offset, size_t length) {
        return length > std::numeric_limits<size_t>::max() - offset ? std::numeric_limits<size_t>::max() :
                                                                      offset + length;
    }

    static void split(Ranges & ranges, size_t offset) {
        auto upper = ranges.upper_bound(offset);
        if (upper == ranges.begin()) {
            return;
        }
        auto current = std::prev(upper);
        if (offset <= current->first || offset >= current->second.end) {
            return;
        }
        Segment right       = current->second;
        current->second.end = offset;
        ranges.emplace(offset, std::move(right));
    }

    std::unordered_map<hrx_buffer_t, Ranges> buffers_;
};

// Barriers HRX records for |order|: one before a command with a dependency recorded since the last barrier.
size_t count_barriers(const std::vector<std::vector<uint32_t>> & deps, const std::vector<uint32_t> & order) {
    std::vector<uint32_t> window_epoch(deps.size(), 0);
    uint32_t              epoch    = 1;
    size_t                barriers = 0;
    for (uint32_t command : order) {
        for (uint32_t dep : deps[command]) {
            if (window_epoch[dep] == epoch) {
                ++barriers;
                ++epoch;
                break;
            }
        }
        window_epoch[command] = epoch;
    }
    return barriers;
}

// Dependency-level (as-soon-as-possible) order: barriers = longest dependency chain. Logged for reference.
std::vector<uint32_t> asap_order(const std::vector<std::vector<uint32_t>> & deps) {
    std::vector<uint32_t> level(deps.size(), 0), order(deps.size());
    for (uint32_t c = 0; c < deps.size(); ++c) {
        order[c] = c;
        for (uint32_t dep : deps[c]) {
            level[c] = std::max(level[c], level[dep] + 1);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) { return level[lhs] < level[rhs]; });
    return order;
}

// Program order, except that when the next command needs a barrier, a command from the next
// kLookahead unscheduled ones that is ready (all its dependencies recorded) and needs no barrier
// (none recorded since the last barrier) is recorded first. Every hazard pair keeps its program order
// because a command is only taken once all of its dependencies are recorded.
std::vector<uint32_t> lookahead_order(const std::vector<std::vector<uint32_t>> & deps) {
    constexpr size_t      kLookahead = 8;
    const size_t          n          = deps.size();
    std::vector<uint8_t>  scheduled(n, 0);
    std::vector<uint32_t> window_epoch(n, 0);
    std::vector<uint32_t> order;
    order.reserve(n);
    uint32_t epoch = 1;
    size_t   first = 0;
    auto ready = [&](uint32_t c) {
        for (uint32_t dep : deps[c]) {
            if (!scheduled[dep]) {
                return false;
            }
        }
        return true;
    };
    auto in_window = [&](uint32_t c) {
        for (uint32_t dep : deps[c]) {
            if (window_epoch[dep] == epoch) {
                return true;
            }
        }
        return false;
    };
    while (order.size() < n) {
        while (scheduled[first]) {
            ++first;
        }
        uint32_t pick    = static_cast<uint32_t>(first);
        bool     found   = false;
        size_t   examined = 0;
        for (size_t c = first; c < n && examined < kLookahead; ++c) {
            if (scheduled[c]) {
                continue;
            }
            ++examined;
            if (ready(static_cast<uint32_t>(c)) && !in_window(static_cast<uint32_t>(c))) {
                pick  = static_cast<uint32_t>(c);
                found = true;
                break;
            }
        }
        if (!found && in_window(pick)) {
            ++epoch;  // barrier
        }
        scheduled[pick]    = 1;
        window_epoch[pick] = epoch;
        order.push_back(pick);
    }
    return order;
}

}  // namespace

std::vector<PreparedCommand> order_prepared_commands_for_graph(const std::vector<PreparedCommand> & commands) {
    const bool enabled = level_order_enabled();
    if ((!enabled && !log_enabled()) || commands.size() < 2) {
        return {};
    }

    std::vector<std::vector<uint32_t>> deps(commands.size());
    RangeHazards                       hazards;
    for (uint32_t c = 0; c < commands.size(); ++c) {
        for (const PreparedCommandBinding & binding : commands[c].kernel.bindings) {
            hazards.collect(binding.ref.buffer, binding.ref.offset, binding.ref.length,
                            access_writes(binding.binding.access), deps[c]);
        }
        std::sort(deps[c].begin(), deps[c].end());
        deps[c].erase(std::unique(deps[c].begin(), deps[c].end()), deps[c].end());
        deps[c].erase(std::remove(deps[c].begin(), deps[c].end(), c), deps[c].end());
        for (const PreparedCommandBinding & binding : commands[c].kernel.bindings) {
            hazards.update(c, binding.ref.buffer, binding.ref.offset, binding.ref.length,
                           access_writes(binding.binding.access));
        }
    }

    std::vector<uint32_t> program_order(commands.size());
    for (uint32_t c = 0; c < commands.size(); ++c) {
        program_order[c] = c;
    }
    const std::vector<uint32_t> level_order = lookahead_order(deps);
    // Reordering changes which kernels overlap. Where it saves no barrier it is not used: Qwen3-0.6B's
    // 512-token prompt program has 254 barriers in either order and its pp512 read 1.1% lower reordered.
    const size_t program_barriers = count_barriers(deps, program_order);
    const size_t level_barriers   = count_barriers(deps, level_order);
    const uint32_t levels = commands.empty() ? 0u : static_cast<uint32_t>(count_barriers(deps, asap_order(deps)) + 1);
    // Prompt programs keep program order too: their barriers are a small share of the long kernels.
    // They are told apart by their transient arena extent: decode programs measured 0.2-6.5 MB, 512-token
    // prompt programs 12-335 MB.
    size_t transient_extent = 0;
    for (const PreparedCommand & command : commands) {
        for (const PreparedCommandBinding & binding : command.kernel.bindings) {
            if (binding.binding.origin == CommandBindingOrigin::Transient) {
                transient_extent = std::max(transient_extent, binding.ref.offset + binding.ref.length);
            }
        }
    }
    const bool use_level = enabled && level_barriers < program_barriers && transient_extent <= kMaxTransientExtent;
    if (log_enabled()) {
        GGML_LOG_WARN("ggml-hrx graph order: %zu commands, %u levels, barriers %zu in program order, %zu in lookahead order, transient extent %zu (%s used)\n",
                      commands.size(), levels,
                      program_barriers, level_barriers, transient_extent, use_level ? "lookahead order" : "program order");
    }
    if (!use_level) {
        return {};
    }
    std::vector<PreparedCommand> ordered;
    ordered.reserve(commands.size());
    for (uint32_t c : level_order) {
        ordered.push_back(commands[c]);
    }
    return ordered;
}

}  // namespace ggml::hrx
