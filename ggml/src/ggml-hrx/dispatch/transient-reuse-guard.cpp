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

#include "transient-reuse-guard.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

namespace ggml::hrx {
namespace {

// Ancestor bitsets cost commands^2 / 8 bytes; 8192 commands = 8 MiB, built once per program.
constexpr size_t kMaxCommands = 8192;

// The guarded packing is used while its arena stays within 2x the unguarded one plus this much.
constexpr size_t kMaxGuardGrowth = size_t{ 8 } << 20;

// Programs whose unguarded arena exceeds this are prompt programs (decode measured 0.1-2.4 MB unguarded).
constexpr size_t kMaxUnguardedArena = size_t{ 16 } << 20;

bool reuse_guard_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_TRANSIENT_REUSE");
        return value == nullptr || (std::strcmp(value, "legacy") != 0 && std::strcmp(value, "0") != 0);
    }();
    return enabled;
}

size_t align_up(size_t value, size_t alignment) {
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

bool writes(ResourceAccess access) {
    return access == ResourceAccess::Write || access == ResourceAccess::ReadWrite;
}

bool reads(ResourceAccess access) {
    return access == ResourceAccess::Read || access == ResourceAccess::ReadWrite;
}

// ancestors[c] holds bit d when command d must finish before command c, following read/write hazards on
// the same value (transients by value id, before arena placement; graph values and program constants by
// value id). Different graph values that alias one buffer are not linked here; a missing link only makes
// the packer reuse less, never more.
class CommandAncestors {
  public:
    explicit CommandAncestors(const std::vector<Command> & commands) :
        words_((commands.size() + 63) / 64),
        bits_(commands.size() * words_, 0) {
        struct Access {
            int32_t               last_writer = -1;
            std::vector<uint32_t> readers;
        };
        std::map<std::pair<int, int32_t>, Access> accesses;
        std::vector<uint32_t>                     deps;
        for (size_t c = 0; c < commands.size(); ++c) {
            deps.clear();
            for (const CommandBinding & binding : commands[c].bindings) {
                const auto key  = std::make_pair(static_cast<int>(binding.origin), binding.value.value);
                auto       found = accesses.find(key);
                if (found == accesses.end()) {
                    continue;
                }
                if (found->second.last_writer >= 0) {
                    deps.push_back(static_cast<uint32_t>(found->second.last_writer));
                }
                if (writes(binding.access)) {
                    deps.insert(deps.end(), found->second.readers.begin(), found->second.readers.end());
                }
            }
            uint64_t * row = &bits_[c * words_];
            for (uint32_t d : deps) {
                if (d == c) {
                    continue;
                }
                const uint64_t * dep_row = &bits_[static_cast<size_t>(d) * words_];
                for (size_t w = 0; w < words_; ++w) {
                    row[w] |= dep_row[w];
                }
                row[d / 64] |= uint64_t{ 1 } << (d % 64);
            }
            for (const CommandBinding & binding : commands[c].bindings) {
                Access & access = accesses[std::make_pair(static_cast<int>(binding.origin), binding.value.value)];
                if (writes(binding.access)) {
                    access.last_writer = static_cast<int32_t>(c);
                    access.readers.clear();
                }
                if (reads(binding.access)) {
                    access.readers.push_back(static_cast<uint32_t>(c));
                }
            }
        }
    }

    bool precedes(uint32_t before, uint32_t command) const {
        return (bits_[static_cast<size_t>(command) * words_ + before / 64] >> (before % 64)) & 1;
    }

  private:
    size_t                words_;
    std::vector<uint64_t> bits_;
};

struct FreeRange {
    size_t                offset = 0;
    size_t                size   = 0;
    std::vector<uint32_t> users;  // every command that touched a value previously placed here
};

struct LiveRange {
    size_t                offset   = 0;
    size_t                size     = 0;
    uint32_t              last_use = 0;
    std::vector<uint32_t> users;
};

bool log_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_LOG_TRANSIENT_REUSE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool pack(const std::vector<Command> &          commands,
          const CommandAncestors *              ancestors,
          std::vector<TransientReuseLifetime> & lifetimes,
          TransientPlan &                       plan);

}  // namespace

bool pack_transients_without_false_dependencies(const std::vector<Command> &          commands,
                                                std::vector<TransientReuseLifetime> & lifetimes,
                                                TransientPlan &                       plan) {
    if (!reuse_guard_enabled() || commands.empty() || commands.size() > kMaxCommands) {
        return false;
    }
    for (size_t i = 0; i < commands.size(); ++i) {
        if (commands[i].ordinal != i) {
            return false;
        }
    }
    // The guard costs arena space: every value whose range cannot be reused gets fresh space. Decode
    // programs grow by a few MiB at most (measured ZAYA1-8B 2.2 -> 2.9 MB, GLM-4.7-Flash 0.2 -> 0.9 MB);
    // 512-token prompt programs grew 3-15x (ZAYA1-8B 19 -> 299 MB), and there the barriers are a small
    // share of the long kernels, so such programs keep the stock packing.
    const CommandAncestors              ancestors(commands);
    std::vector<TransientReuseLifetime> guarded_lifetimes = lifetimes;
    TransientPlan                       guarded           = plan;
    std::vector<TransientReuseLifetime> unguarded_lifetimes = lifetimes;
    TransientPlan                       unguarded           = plan;
    if (!pack(commands, &ancestors, guarded_lifetimes, guarded) ||
        !pack(commands, nullptr, unguarded_lifetimes, unguarded)) {
        return false;
    }
    // Prompt programs (unguarded arena over 16 MiB) keep the stock packing as well: with the guard, pp512
    // read 2.2% lower on Qwen3-8B (larger arena, more activation traffic).
    const size_t limit = 2 * unguarded.arena_size + kMaxGuardGrowth;
    const bool   use   = guarded.arena_size <= limit && unguarded.arena_size <= kMaxUnguardedArena;
    if (log_enabled()) {
        GGML_LOG_WARN("ggml-hrx transient reuse: %zu commands, %zu values, arena %zu bytes guarded, %zu unguarded%s\n",
                      commands.size(), lifetimes.size(), guarded.arena_size, unguarded.arena_size,
                      use ? "" : " (prompt-sized or over the growth limit: stock packing)");
    }
    if (!use) {
        return false;
    }
    lifetimes = std::move(guarded_lifetimes);
    plan      = std::move(guarded);
    return true;
}

namespace {

// |ancestors| == nullptr packs without the guard (any freed range is reusable); used to bound the
// guarded arena's growth.
bool pack(const std::vector<Command> &          commands,
          const CommandAncestors *              ancestors,
          std::vector<TransientReuseLifetime> & lifetimes,
          TransientPlan &                       plan) {

    std::map<int32_t, std::vector<uint32_t>> users_by_value;
    for (const Command & command : commands) {
        for (const CommandBinding & binding : command.bindings) {
            if (binding.origin == CommandBindingOrigin::Transient) {
                std::vector<uint32_t> & users = users_by_value[binding.value.value];
                if (users.empty() || users.back() != command.ordinal) {
                    users.push_back(command.ordinal);
                }
            }
        }
    }

    std::sort(lifetimes.begin(), lifetimes.end(), [](const TransientReuseLifetime & lhs, const TransientReuseLifetime & rhs) {
        if (lhs.first_use != rhs.first_use) {
            return lhs.first_use < rhs.first_use;
        }
        if (lhs.last_use != rhs.last_use) {
            return lhs.last_use < rhs.last_use;
        }
        return lhs.allocation.value.value < rhs.allocation.value.value;
    });

    TransientPlan             packed = plan;
    std::vector<LiveRange>    live;
    std::vector<FreeRange>    free_ranges;
    for (TransientReuseLifetime & lifetime : lifetimes) {
        const uint32_t writer = lifetime.first_use;
        if (writer >= commands.size()) {
            return false;
        }
        // Release values whose last use is before this value's first command.
        for (size_t i = 0; i < live.size();) {
            if (live[i].last_use < writer) {
                free_ranges.push_back({ live[i].offset, live[i].size, std::move(live[i].users) });
                live[i] = std::move(live.back());
                live.pop_back();
            } else {
                ++i;
            }
        }
        std::sort(free_ranges.begin(), free_ranges.end(),
                  [](const FreeRange & lhs, const FreeRange & rhs) { return lhs.offset < rhs.offset; });

        auto reusable = [&](const FreeRange & range) {
            if (ancestors == nullptr) {
                return true;
            }
            for (uint32_t user : range.users) {
                if (user >= writer || !ancestors->precedes(user, writer)) {
                    return false;
                }
            }
            return true;
        };

        // Best fit over runs of adjacent reusable ranges.
        const size_t size      = lifetime.allocation.size;
        const size_t alignment = std::max<size_t>(lifetime.allocation.alignment, 1);
        size_t       best_offset = 0, best_waste = SIZE_MAX, best_first = 0, best_last = 0;
        bool         found = false;
        for (size_t first = 0; first < free_ranges.size(); ++first) {
            if (!reusable(free_ranges[first]) ||
                align_up(free_ranges[first].offset, alignment) >= free_ranges[first].offset + free_ranges[first].size) {
                continue;
            }
            size_t end = free_ranges[first].offset + free_ranges[first].size;
            size_t last = first;
            while (true) {
                const size_t offset = align_up(free_ranges[first].offset, alignment);
                if (offset + size <= end) {
                    const size_t waste = end - free_ranges[first].offset - size;
                    if (waste < best_waste) {
                        best_waste = waste, best_offset = offset, best_first = first, best_last = last, found = true;
                    }
                    break;
                }
                if (last + 1 >= free_ranges.size() || free_ranges[last + 1].offset != end ||
                    !reusable(free_ranges[last + 1])) {
                    break;
                }
                ++last;
                end = free_ranges[last].offset + free_ranges[last].size;
            }
        }

        size_t offset = 0;
        if (found) {
            offset = best_offset;
            // Keep the parts of the run outside [offset, offset + size) free, with their own users.
            std::vector<FreeRange> kept;
            const FreeRange & head = free_ranges[best_first];
            if (offset > head.offset) {
                kept.push_back({ head.offset, offset - head.offset, head.users });
            }
            const FreeRange & tail     = free_ranges[best_last];
            const size_t      tail_end = tail.offset + tail.size;
            if (offset + size < tail_end) {
                const size_t begin = std::max(offset + size, tail.offset);
                kept.push_back({ begin, tail_end - begin, tail.users });
            }
            free_ranges.erase(free_ranges.begin() + static_cast<std::ptrdiff_t>(best_first),
                              free_ranges.begin() + static_cast<std::ptrdiff_t>(best_last) + 1);
            free_ranges.insert(free_ranges.end(), kept.begin(), kept.end());
        } else {
            offset             = align_up(packed.arena_size, alignment);
            packed.arena_size  = offset + size;
        }

        lifetime.allocation.arena_offset = offset;
        LiveRange range;
        range.offset   = offset;
        range.size     = size;
        range.last_use = lifetime.last_use;
        const auto users = users_by_value.find(lifetime.allocation.value.value);
        if (users != users_by_value.end()) {
            range.users = users->second;
        } else {
            range.users = { lifetime.first_use, lifetime.last_use };
        }
        live.push_back(std::move(range));
        packed.allocations.push_back(lifetime.allocation);
    }

    plan = std::move(packed);
    return true;
}

}  // namespace
}  // namespace ggml::hrx
