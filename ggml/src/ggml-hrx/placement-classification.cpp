// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0

#include "placement-classification.h"

namespace ggml::hrx {

bool placement_node_is_by_design(const placement_node_facts & facts) {
    // #991: an unaligned contraction has no HRX dense schedule at all (input_size % 256 != 0), so the CPU is
    // the only owner it can have.
    if (facts.is_matmul && facts.src0_present && facts.src0_ne0 % 256 != 0) {
        return true;
    }
    // #2450: a host-resident weight is served by the CPU by design, so the node that consumes it belongs there.
    return facts.src_host_weight;
}

}  // namespace ggml::hrx
