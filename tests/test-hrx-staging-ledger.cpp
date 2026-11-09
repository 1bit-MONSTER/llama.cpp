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
// The host staging buffer's pool charge (engine#1711). Unlike the two arenas this one carries no delta -- the object
// owns exactly one buffer -- so the whole contract is that clear() gives the bytes back EXACTLY once. clear() is reached
// by the destructor, by move-assignment and by the next allocate, so a tally that survived a release would hand the same
// bytes back two or three times and inflate the budget for everything afterwards.
//
// No device: the ledger is the only thing under test, which is why this does not need libhrx.

#include "hrx-1bit-safety.h"
#include "runtime/host-memory.h"

#include <cstdio>

using ggml::hrx::HostStagingBuffer;

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
    using ggml::hrx::onebit::pool_reserve;
    using ggml::hrx::onebit::pool_allocated;   // the ledger (NOT pool_reserve_bytes, which is the 16 GiB headroom reserve)
    using ggml::hrx::onebit::pool_release;

    const size_t base = pool_allocated();

    // 1. clear() on a buffer that holds nothing must not touch the ledger: this is the state of every default-constructed
    //    staging buffer and of every one already cleared, so an unconditional release here would underflow everything.
    {
        HostStagingBuffer empty;
        empty.clear();
        check(pool_allocated() == base, "clear() on an empty staging buffer leaves the ledger alone", pool_allocated(), base);
        empty.clear();
        check(pool_allocated() == base, "and calling it twice does not either", pool_allocated(), base);
    }

    // 2. A held charge is returned exactly once. The buffer is simulated: reserve the bytes, record them on the object,
    //    clear it, and the ledger must be back where it started.
    {
        HostStagingBuffer held;
        const size_t     bytes = 8u << 20;
        // budget 1 TiB: pool_reserve() returns true WITHOUT touching the ledger when budget == 0, so a real budget is
        // needed for the bytes to actually be charged
        const bool reserved = pool_reserve(1ull << 40, bytes, "a host staging buffer");
        check(reserved, "the pool accepts an 8 MiB staging request", 0, 0);
        held.charged_bytes_ = bytes;
        check(pool_allocated() == base + bytes, "the ledger carries the staging bytes", pool_allocated(), base + bytes);
        held.clear();
        check(pool_allocated() == base, "clear() returns exactly what was charged", pool_allocated(), base);
        held.clear();
        check(pool_allocated() == base, "a second clear() does not release it again", pool_allocated(), base);
        (void) pool_release;  // the release under test happens inside clear()
    }

    if (failures != 0) {
        std::printf("%d arm(s) failed\n", failures);
        return 1;
    }
    std::printf("all arms passed\n");
    return 0;
}
