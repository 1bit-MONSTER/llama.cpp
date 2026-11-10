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
#include <utility>

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

    // 3. The MOVE path carries the tally (the defect review of #270 found): the destination ends up holding the
    //    buffer, so it must hold the charge; the source ends up owning nothing, so it must hold no tally. Dropping
    //    either half leaves a ledger that disagrees with the memory in both objects at once.
    {
        HostStagingBuffer from, to;
        const size_t     bytes = 4u << 20;
        const bool       reserved = pool_reserve(1ull << 40, bytes, "a host staging buffer");
        check(reserved, "the pool accepts a 4 MiB staging request for the move arm", 0, 0);
        from.charged_bytes_ = bytes;
        const size_t before = pool_allocated();
        to = std::move(from);
        check(pool_allocated() == before, "a move does not touch the ledger", pool_allocated(), before);
        check(to.charged_bytes_ == bytes, "the destination carries the charge", to.charged_bytes_, bytes);
        check(from.charged_bytes_ == 0, "the source keeps no tally for a buffer it no longer owns", from.charged_bytes_, 0);
        to.clear();
        check(pool_allocated() == base, "clearing the destination returns the bytes exactly once", pool_allocated(), base);
        from.clear();                                     // must be a no-op: it holds nothing and is charged nothing
        check(pool_allocated() == base, "clearing the moved-from source does not release again", pool_allocated(), base);
    }

    // 4. The pairing helper the host-weight entry calls (engine #1823): charge refuses without touching anything, release
    //    returns exactly once, and the uncharged mode (budget 0) reserves nothing. Same contract as the staging object
    //    above, now on the shared function, which is what makes it testable without a device.
    {
        using ggml::hrx::pool_tally::charge;
        using ggml::hrx::pool_tally::release;
        // A bystander holds 1 MiB for the whole block. Without it the ledger is at zero when the second release runs,
        // pool_release clamps there, and "a second release is a no-op" passes for a release() that never zeroes its tally.
        size_t bystander = 0;
        check(charge(1ull << 40, 1u << 20, bystander, "a bystander"), "a bystander holds 1 MiB, so an over-release has bytes to take", 0, 0);
        size_t        tally = 0;
        const size_t  before = pool_allocated();
        const size_t  bytes  = 6u << 20;

        check(charge(1ull << 40, bytes, tally, "test weight"), "charge takes the bytes from the pool", 0, 0);
        check(tally == bytes && pool_allocated() == before + bytes, "and records the tally and the ledger", pool_allocated(), before + bytes);
        release(tally);
        check(tally == 0 && pool_allocated() == before, "release returns them and zeroes the tally", pool_allocated(), before);
        release(tally);
        check(pool_allocated() == before, "a second release is a no-op, not an underflow of somebody else's bytes", pool_allocated(), before);

        // a refusal leaves both untouched: this is the arm that matters on the default path, where the weight used to be
        // materialised uncharged. Note `budget == size` with nothing held EXACTLY FITS and is accepted -- a first version
        // of this arm asserted otherwise and was wrong -- so the pressure has to come from bytes already held.
        size_t       holder = 0;
        const size_t held   = 4u << 20;
        check(charge(64u << 20, held, holder, "a holder"), "a holder takes 4 MiB", 0, 0);
        tally = 0;
        const bool refused = charge(5u << 20, bytes, tally, "test weight");   // 4 held + 6 wanted > 5 budget
        check(!refused && tally == 0, "a refused charge leaves the tally alone", tally, 0);
        check(pool_allocated() == before + held, "and does not touch the ledger either", pool_allocated(), before + held);

        // the uncharged mode: budget 0 means "device size unknown", so nothing is reserved and nothing must be returned
        check(charge(0, bytes, tally, "test weight") && tally == 0, "budget 0 reserves nothing and reports success", tally, 0);
        release(holder);
        check(pool_allocated() == before, "and the holder's release returns everything", pool_allocated(), before);
        release(bystander);
        check(pool_allocated() == base, "and the bystander's release leaves the ledger where the test found it", pool_allocated(), base);
    }

    if (failures != 0) {
        std::printf("%d arm(s) failed\n", failures);
        return 1;
    }
    std::printf("all arms passed\n");
    return 0;
}
