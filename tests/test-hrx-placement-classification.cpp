// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// engine#2450: the two sides of GGML_HRX_STRICT, without a device and without a ggml link.
//
// The patch that followed #2450 made strict abort only on an off-HRX node that is NOT explainable. A reviewer's
// objection was the right one: nothing then exercised either branch, so strict could have become a mode that
// never fires. These arms pin both -- the named classes must be classified by design, and a node with no named
// reason must NOT be, because that is the case strict still aborts on.

#include "placement-classification.h"

#include <cstdio>

using ggml::hrx::placement_node_facts;
using ggml::hrx::placement_node_is_by_design;

static int failures = 0;

static void expect(bool got, bool want, const char * what) {
    if (got != want) {
        std::printf("FAIL %s: got %s, want %s\n", what, got ? "by-design" : "unexplained", want ? "by-design" : "unexplained");
        ++failures;
    }
}

int main() {
    // 1. An unaligned contraction: no HRX dense schedule exists (engine#991), so this is by design.
    {
        placement_node_facts f;
        f.is_matmul = true, f.src0_present = true, f.src0_ne0 = 1000;
        expect(placement_node_is_by_design(f), true, "unaligned contraction (ne0 = 1000) is by design");
    }
    // 2. A host-resident weight: a file-mapped model's weights are in ggml core's CPU from-ptr buffer type, so the
    //    CPU owns the consumer (engine#2450). This is the case the patch exists for.
    {
        placement_node_facts f;
        f.is_matmul = true, f.src0_present = true, f.src0_ne0 = 1024, f.src_host_weight = true;
        expect(placement_node_is_by_design(f), true, "a host-resident weight is by design");
    }
    // 3. THE OTHER SIDE: a 256-aligned contraction with no host weight and no other reason. Strat mode must still
    //    abort here, so this must be classified as unexplained. Without this arm the mode could never fire.
    {
        placement_node_facts f;
        f.is_matmul = true, f.src0_present = true, f.src0_ne0 = 1024;
        expect(placement_node_is_by_design(f), false, "an aligned contraction with no host weight stays unexplained");
    }
    // 4. Alignment only excuses a contraction; another op with a misaligned width is not the #991 class.
    {
        placement_node_facts f;
        f.is_matmul = false, f.src0_present = true, f.src0_ne0 = 1000;
        expect(placement_node_is_by_design(f), false, "a non-contraction is not excused by its src0 width");
    }
    // 5. No src0 at all: never crash, never classify.
    {
        placement_node_facts f;
        f.is_matmul = true;
        expect(placement_node_is_by_design(f), false, "a contraction with no src0 is unexplained, not a crash");
    }
    // 6. Default-constructed facts: nothing named, nothing excused.
    expect(placement_node_is_by_design(placement_node_facts{}), false, "empty facts are unexplained");

    if (failures != 0) {
        std::printf("test-hrx-placement-classification: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-hrx-placement-classification: all cases ok (6 arms, both sides of STRICT)\n");
    return 0;
}
