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
//
// The host-weight arena's pool charge (engine#1856, the arm llama.cpp #275's review asked for). #269 charged the arena
// and #275 charged the unbounded per-weight path, but nothing asserted either against the LEDGER: the staging buffer has
// had that arm since #270, and the arena never did, so both charges could have been wrong in the one direction that does
// not fail loudly. An under-charge leaves the ledger saying less memory is held than is, the guardrail admits a launch it
// should refuse, and the result lands in the allocator as "llama_model_load: error loading model" -- the shape of
// engine #1502 and #1417.
//
// No device: this simulates the charge the way test-hrx-staging-ledger.cpp does -- reserve against the pool, record the
// tally on the object, then exercise the paths that must return it. The size of a growth is NOT re-derived here: that is
// plan_arena_charge()'s arithmetic and test-hrx-transient-arena-charge.cpp already pins it. What is under test here is
// conservation: what goes in comes back out, exactly once.

#include "ggml-hrx/hrx-1bit-safety.h"
#include "runtime/host-memory.h"

#include <cstdio>

using ggml::hrx::HostWeightCache;

static int failures = 0;

static void check(bool ok, const char * what, size_t got, size_t want) {
    if (!ok) {
        std::printf("FAIL %s (ledger %zu, want %zu)\n", what, got, want);
        ++failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

int main() {
    using ggml::hrx::onebit::pool_allocated;   // the LEDGER (not pool_reserve_bytes, which is the 16 GiB headroom reserve)
    using ggml::hrx::onebit::pool_reserve;

    const size_t base = pool_allocated();

    // 1. clear() on a cache that holds nothing must not touch the ledger. This is the state of every freshly constructed
    //    cache and of every one already cleared, so an unconditional release here would underflow every later figure.
    {
        HostWeightCache empty;
        empty.clear();
        check(pool_allocated() == base, "clear() on an arena holding nothing leaves the ledger alone", pool_allocated(), base);
        empty.clear();
        check(pool_allocated() == base, "and calling it twice does not either", pool_allocated(), base);
        check(empty.held_bytes_for_test() == 0, "and the cache's own tally stays zero", empty.held_bytes_for_test(), 0);
    }

    // 2. A held charge is returned exactly once. Budget 1 TiB because pool_reserve() returns true WITHOUT touching the
    //    ledger when budget == 0, which would make every assertion below vacuous.
    {
        HostWeightCache cache;
        const size_t    bytes = 8u << 20;
        const bool      reserved = pool_reserve(1ull << 40, bytes, "the host-weight arena");
        check(reserved, "the pool accepts an 8 MiB arena growth", 0, 0);
        cache.set_held_bytes_for_test(bytes);
        check(pool_allocated() == base + bytes, "the ledger carries the arena's bytes", pool_allocated(), base + bytes);
        cache.clear();
        check(pool_allocated() == base, "clear() returns exactly what was charged", pool_allocated(), base);
        check(cache.held_bytes_for_test() == 0, "and zeroes its own tally, so a second clear() is a no-op",
              cache.held_bytes_for_test(), 0);
        cache.clear();
        check(pool_allocated() == base, "a second clear() does not release it again", pool_allocated(), base);
    }

    // 3. The refusal path must take nothing. A growth the pool cannot hold returns before the tally is incremented -- the
    //    direction that matters, because a charge taken for an allocation that never happened would make the guardrail
    //    stricter than the memory and refuse launches that fit. Simulated with a budget that this growth cannot fit.
    {
        HostWeightCache cache;
        const size_t    bytes = 64u << 20;
        const size_t    before = pool_allocated();
        const bool      refused = pool_reserve(/*budget*/ 1u << 10, bytes, "the host-weight arena");
        check(!refused, "a growth larger than the budget is refused", 0, 0);
        check(pool_allocated() == before, "... and the refusal leaves the ledger untouched", pool_allocated(), before);
        cache.clear();
        check(pool_allocated() == before, "... and a clear() after it returns nothing it did not take", pool_allocated(), before);
    }

    // 4. Two caches are independent: clearing one must not release the other's charge. The arena is per-cache, and a
    //    shared tally would make the ledger right in total and wrong per model -- invisible until two models load.
    {
        HostWeightCache a, b;
        const size_t    ab = 4u << 20;   // identical sizes on purpose: a shared tally bugs out on the second release
        check(pool_reserve(1ull << 40, ab, "the host-weight arena (a)"), "the pool accepts a's growth", 0, 0);
        a.set_held_bytes_for_test(ab);
        check(pool_reserve(1ull << 40, ab, "the host-weight arena (b)"), "the pool accepts b's growth", 0, 0);
        b.set_held_bytes_for_test(ab);
        check(pool_allocated() == base + 2 * ab, "both charges are in the ledger", pool_allocated(), base + 2 * ab);
        a.clear();
        check(pool_allocated() == base + ab, "clearing a returns a's bytes and leaves b's", pool_allocated(), base + ab);
        b.clear();
        check(pool_allocated() == base, "and clearing b returns the rest", pool_allocated(), base);
    }

    if (failures == 0) std::printf("all host-arena ledger checks passed\n");
    return failures == 0 ? 0 : 1;
}
