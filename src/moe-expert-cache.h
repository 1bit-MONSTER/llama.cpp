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

// The routed-expert cache for streaming MoE models from NVMe (docs/moe-streaming.md):
// experts live in pinned RAM slots, one slot per expert holding all of its parts (gate, up,
// down). Misses are read with O_DIRECT by a pool of reader threads, several reads in flight;
// demand reads go ahead of prefetches. acquire() blocks until an expert is resident;
// prefetch() only queues reads, so the next layer's experts can load while this layer
// computes.

#include "moe-gguf-index.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace onebit::moe {

struct CacheOptions {
    int slots = 1024;         // experts held in RAM, all layers together (each layer keeps the same share)
    bool per_layer = false;   // false: one LRU over all layers (it hit more in replays); true: slots / layers each
    int io_threads = 8;       // reads in flight
    bool pin = true;          // mlock the slots
};

struct CacheStats {
    uint64_t hits = 0;              // resident when acquired, and not by a prefetch still unused
    uint64_t prefetch_hits = 0;     // resident (or loading) because of a prefetch
    uint64_t misses = 0;            // read on demand
    uint64_t prefetches = 0;        // experts a prefetch started reading
    uint64_t prefetch_wasted = 0;   // prefetched, then evicted before any use
    uint64_t bytes_read = 0;
    double stall_ms = 0;            // time acquire() waited for reads
};

class ExpertCache {
public:
    // One expert's parts in its slot, in GgufIndex::experts order.
    struct Resident { std::vector<const uint8_t*> parts; };

    ExpertCache(const GgufIndex& index, const CacheOptions& opt);
    ~ExpertCache();
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    // Makes the experts resident (reading misses on demand) and pins them until release().
    std::vector<Resident> acquire(int layer, const std::vector<int>& experts);
    void release(int layer, const std::vector<int>& experts);
    // Queues reads for the experts not resident yet; returns at once.
    void prefetch(int layer, const std::vector<int>& experts);

    CacheStats stats() const;
    void reset_stats();
    size_t slot_bytes() const { return slot_bytes_; }  // the largest slot
    size_t pinned_bytes() const { return mem_bytes_; }
    int slots() const { return (int) slots_.size(); }

private:
    enum class State { Empty, Loading, Ready };
    struct Slot {
        uint64_t key = ~0ull;
        State state = State::Empty;
        int refs = 0;
        int pending = 0;          // reads not finished
        bool prefetched = false;  // loaded by prefetch() and not acquired since
        std::vector<size_t> part_off;  // byte offset of each part's data inside the slot
        size_t off = 0;           // byte offset of the slot in the pinned region
        std::list<int>::iterator lru_it;
        int lru_list = 0;
    };
    struct Read { int slot; int fd; uint64_t off, len; uint8_t* dst; };

    uint64_t key(int layer, int e) const { return (uint64_t(layer) << 32) | uint32_t(e); }
    int lru_of(int layer) const;
    int take_slot(int layer);   // evicts; lock held
    int start_load(int layer, int e, bool demand);  // lock held
    void reader();

    const GgufIndex& index_;
    CacheOptions opt_;
    std::vector<int> fds_;
    uint8_t* mem_ = nullptr;
    size_t mem_bytes_ = 0, slot_bytes_ = 0;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, int> where_;
    std::vector<std::list<int>> lru_;         // front = most recent; per layer or one list
    std::unordered_map<int, int> layer_lru_;  // MoE layer -> lru_ index
    mutable std::mutex mu_;
    std::condition_variable ready_cv_, work_cv_;
    std::deque<Read> demand_q_, prefetch_q_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
    CacheStats st_;
};

}  // namespace onebit::moe
