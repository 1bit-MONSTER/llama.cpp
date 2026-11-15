// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// engine#2450: whether an off-HRX node is off for a reason the code calls BY DESIGN. Kept in its own
// translation unit with no ggml dependency, so the two sides of GGML_HRX_STRICT -- the named classes and the
// unexplained case that must still abort -- are testable without a device (tests/test-hrx-placement-classification.cpp).

#pragma once

#include <cstdint>

namespace ggml::hrx {

// What the placement observer can see about one graph node, reduced to the questions the policy asks.
struct placement_node_facts {
    bool    is_matmul       = false;  // MUL_MAT or MUL_MAT_ID
    bool    src0_present    = false;
    int64_t src0_ne0        = 0;
    bool    src_host_weight = false;  // some source is a host-resident weight (host_weight_is_host_resident)
};

// true  -> by design: an unaligned contraction (no HRX dense schedule exists, engine#991) or a host-resident
//          weight (a file-mapped model's weights live in ggml core's CPU from-ptr buffer type, so the CPU owns
//          the consumer, engine#2450).
// false -> unexplained: strict mode aborts on exactly this.
bool placement_node_is_by_design(const placement_node_facts & facts);

}  // namespace ggml::hrx
