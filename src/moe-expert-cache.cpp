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
#include "moe-expert-cache.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace onebit::moe {

namespace {
constexpr uint64_t kAlign = 4096;  // O_DIRECT offset, length and buffer alignment
uint64_t down(uint64_t v) { return v / kAlign * kAlign; }
uint64_t up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
}  // namespace

ExpertCache::ExpertCache(const GgufIndex& index, const CacheOptions& opt) : index_(index), opt_(opt) {
    if (index.experts.empty()) throw std::runtime_error("the model has no routed expert tensors");
    // A slot holds every part's aligned read window, [down(off), up(off + bytes)). Mixed-quant
    // models (Unsloth's UD) use bigger types in some layers, so layers are grouped by that size:
    // each size class has its own slots, and LRU (shared mode) or one per layer.
    // Each part has a fixed window, up(bytes) + one alignment unit; after a read the part's data
    // is moved to the start of its window, so a class's slots share one layout (a device can
    // view them as a strided tensor per part).
    std::map<std::vector<size_t>, std::vector<int>> classes;  // part windows -> layers
    for (const auto& [l, parts] : index.experts) {
        std::vector<size_t> w;
        size_t n = 0;
        for (const auto& p : parts) { w.push_back(up(p.bytes) + kAlign); n += w.back(); }
        classes[w].push_back(l);
        slot_bytes_ = std::max(slot_bytes_, n);
    }
    // the budget is in experts; every layer keeps the same share of its experts
    const double share = std::min(1.0, double(opt_.slots) / (double(index.experts.size()) * index.n_expert));
    struct List { size_t bytes; int slots; std::vector<size_t> windows; };
    std::vector<List> lists;
    for (const auto& [windows, layers] : classes) {
        size_t bytes = 0;
        for (size_t w : windows) bytes += w;
        if (opt_.per_layer) {
            for (int l : layers) {
                layer_lru_[l] = (int) lists.size();
                lists.push_back({bytes, std::max(1, (int) (share * index.n_expert)), windows});
            }
        } else {
            for (int l : layers) layer_lru_[l] = (int) lists.size();
            lists.push_back({bytes, std::max(1, (int) (share * index.n_expert * layers.size())), windows});
        }
    }
    int n_slots = 0;
    for (const auto& li : lists) { n_slots += li.slots; mem_bytes_ += size_t(li.slots) * li.bytes; }
    void* p = ::mmap(nullptr, mem_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot map " + std::to_string(mem_bytes_ >> 20) + " MiB of expert slots");
    mem_ = static_cast<uint8_t*>(p);
    ::madvise(mem_, mem_bytes_, MADV_HUGEPAGE);
    if (opt_.pin && ::mlock(mem_, mem_bytes_) != 0)
        throw std::runtime_error(std::string("mlock of the expert slots failed: ") + std::strerror(errno));
    for (const auto& f : index.files) {
        const int fd = ::open(f.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("cannot open " + f + " with O_DIRECT: " + std::strerror(errno));
        fds_.push_back(fd);
    }
    slots_.resize(n_slots);
    lru_.resize(lists.size());
    size_t off = 0;
    int si = 0;
    for (int li = 0; li < (int) lists.size(); ++li) {
        Layout lay;
        lay.base = mem_ + off;
        lay.slot_bytes = lists[li].bytes;
        lay.n_slots = lists[li].slots;
        size_t at = 0;
        for (size_t w : lists[li].windows) { lay.part_off.push_back(at); at += w; }
        layouts_.push_back(lay);
        for (int k = 0; k < lists[li].slots; ++k, ++si) {
            slots_[si].lru_list = li;
            slots_[si].index = k;
            slots_[si].off = off;
            off += lists[li].bytes;
            lru_[li].push_back(si);  // empty slots at the back: taken first
            slots_[si].lru_it = std::prev(lru_[li].end());
        }
    }
    for (int i = 0; i < std::max(1, opt_.io_threads); ++i) threads_.emplace_back([this] { reader(); });
}

ExpertCache::~ExpertCache() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (auto& t : threads_) t.join();
    for (int fd : fds_) ::close(fd);
    if (mem_) {
        if (opt_.pin) ::munlock(mem_, mem_bytes_);
        ::munmap(mem_, mem_bytes_);
    }
}

int ExpertCache::lru_of(int layer) const {
    auto it = layer_lru_.find(layer);
    if (it == layer_lru_.end()) throw std::runtime_error("layer " + std::to_string(layer) + " has no routed experts");
    return it->second;
}

int ExpertCache::take_slot(int layer) {
    auto& lru = lru_[lru_of(layer)];
    for (auto it = lru.rbegin(); it != lru.rend(); ++it) {  // least recent first
        Slot& s = slots_[*it];
        if (s.refs > 0 || s.state == State::Loading) continue;
        if (s.state == State::Ready) {
            where_.erase(s.key);
            if (s.prefetched) st_.prefetch_wasted++;
        }
        s.state = State::Empty;
        s.prefetched = false;
        return *it;
    }
    throw std::runtime_error("no free expert slot: every slot of this layer is in use or loading");
}

int ExpertCache::start_load(int layer, int e, bool demand) {
    const int si = take_slot(layer);
    Slot& s = slots_[si];
    s.key = key(layer, e);
    s.state = State::Loading;
    s.prefetched = !demand;
    s.part_off = layouts_[s.lru_list].part_off;
    where_[s.key] = si;
    auto& lru = lru_[s.lru_list];
    lru.splice(lru.begin(), lru, s.lru_it);
    uint8_t* base = mem_ + s.off;
    size_t at = 0;
    const auto& parts = index_.experts.at(layer);
    s.pending = (int) parts.size();
    for (const auto& p : parts) {
        const uint64_t off = p.base + uint64_t(e) * p.stride;
        const uint64_t a = down(off), b = up(off + p.bytes);
        Read r{si, fds_[p.file], a, b - a, base + at, off - a, p.bytes};
        (demand ? demand_q_ : prefetch_q_).push_back(r);
        at += up(p.bytes) + kAlign;  // the part's fixed window
        st_.bytes_read += b - a;
    }
    work_cv_.notify_all();
    return si;
}

void ExpertCache::reader() {
    for (;;) {
        Read r;
        {
            std::unique_lock<std::mutex> lk(mu_);
            work_cv_.wait(lk, [&] { return stop_ || !demand_q_.empty() || !prefetch_q_.empty(); });
            if (stop_) return;
            auto& q = demand_q_.empty() ? prefetch_q_ : demand_q_;
            r = q.front();
            q.pop_front();
        }
        uint64_t done = 0;
        while (done < r.len) {
            const ssize_t n = ::pread(r.fd, r.dst + done, r.len - done, (off_t) (r.off + done));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                // short read at the end of the file: the window's tail past EOF is padding
                std::memset(r.dst + done, 0, r.len - done);
                break;
            }
            done += (uint64_t) n;
        }
        if (r.lead) std::memmove(r.dst, r.dst + r.lead, r.bytes);  // the part at its window's start
        {
            std::lock_guard<std::mutex> lk(mu_);
            Slot& s = slots_[r.slot];
            if (--s.pending == 0) s.state = State::Ready;
        }
        ready_cv_.notify_all();
    }
}

std::vector<ExpertCache::Resident> ExpertCache::acquire(int layer, const std::vector<int>& experts) {
    std::unique_lock<std::mutex> lk(mu_);
    std::vector<int> si(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) {
        auto it = where_.find(key(layer, experts[i]));
        if (it == where_.end()) {
            st_.misses++;
            si[i] = start_load(layer, experts[i], true);
        } else {
            Slot& s = slots_[it->second];
            if (s.prefetched) st_.prefetch_hits++;
            else st_.hits++;
            si[i] = it->second;
            auto& lru = lru_[s.lru_list];
            lru.splice(lru.begin(), lru, s.lru_it);
        }
        slots_[si[i]].refs++;
        slots_[si[i]].prefetched = false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ready_cv_.wait(lk, [&] {
        for (int s : si) if (slots_[s].state != State::Ready) return false;
        return true;
    });
    st_.stall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::vector<Resident> out(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) {
        const Slot& s = slots_[si[i]];
        const uint8_t* base = mem_ + s.off;
        for (size_t off : s.part_off) out[i].parts.push_back(base + off);
        out[i].slot = s.index;
    }
    return out;
}

void ExpertCache::release(int layer, const std::vector<int>& experts) {
    std::lock_guard<std::mutex> lk(mu_);
    for (int e : experts) {
        auto it = where_.find(key(layer, e));
        if (it != where_.end() && slots_[it->second].refs > 0) slots_[it->second].refs--;
    }
}

void ExpertCache::prefetch(int layer, const std::vector<int>& experts) {
    std::lock_guard<std::mutex> lk(mu_);
    for (int e : experts) {
        if (where_.count(key(layer, e))) continue;
        try {
            start_load(layer, e, false);
        } catch (const std::runtime_error&) {
            return;  // every slot busy: a prefetch is optional
        }
        st_.prefetches++;
    }
}

CacheStats ExpertCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return st_;
}

void ExpertCache::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    st_ = CacheStats{};
}

}  // namespace onebit::moe
