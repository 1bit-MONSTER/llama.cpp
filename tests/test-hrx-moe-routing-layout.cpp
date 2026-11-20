// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// The MoE partition table is written with one descriptor layout and read with another (engine #2861).
//
// The router chooses the layout from the expert count; the common mul_mat_id decodes with fixed constants. Nothing
// compared the two, and the byte sizes agree, so a 128-expert routing bundle was handed to a common-layout reader and
// the partition bits landed inside its expert field: the reader gathered past the expert weight tensor and the device
// faulted. These arms pin the boundary and the arithmetic that makes it dangerous, on the host, without a device.

#include "dispatch/command-plan-metadata.h"
#include "dispatch_registration/common/dispatch-moe-routing-layout.h"

#include <cstdio>
#include <cstdlib>

using namespace ggml::hrx;

static int failures = 0;

static void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// The encoder the partition builder runs (dispatch-moe-router.cpp) and the decoder the common consumer runs
// (command-plan-metadata.h constants). Both are reproduced from the source constants so that drift in either, or in
// moe_router_descriptor_layout(), fails an arm here instead of on the device.
static int32_t encode_descriptor(int32_t expert, int32_t partition, int32_t row_count, const MoeRoutingDescriptorLayout & layout) {
    return expert | (partition << layout.partition_shift_value) | ((row_count - 1) << layout.row_count_shift_value);
}

static int32_t decoded_expert(int32_t descriptor) {
    return descriptor & kMoePartitionExpertMask;
}

static int32_t decoded_partition(int32_t descriptor) {
    return (descriptor >> kMoePartitionShift) & 63;
}

static int32_t decoded_row_count(int32_t descriptor) {
    return ((descriptor >> kMoePartitionRowCountShift) & 31) + 1;
}

int main() {
    // 1. The layout boundary is the expert count, and it is 128.
    const MoeRoutingDescriptorLayout qwen   = moe_router_descriptor_layout(128);
    const MoeRoutingDescriptorLayout common = moe_router_descriptor_layout(129);
    check(qwen.expert_mask_value == 127 && qwen.partition_shift_value == 7 && qwen.row_count_shift_value == 13,
          "128 experts selects the Qwen layout (127/7/13)");
    check(common.expert_mask_value == 511 && common.partition_shift_value == 9 && common.row_count_shift_value == 15,
          "129 experts selects the common layout (511/9/15)");

    // 2. The reuse guard: a table written with one layout must not be read with the other.
    check(!moe_routing_descriptor_is_common_layout(qwen.expert_mask_value, qwen.partition_shift_value,
                                                   qwen.row_count_shift_value),
          "a Qwen-laid bundle is refused by the common reader");
    {
        // The same question asked of a whole bundle, which is what common_mul_mat_id_bundle_matches() asks before it
        // reuses the router's table. Sizes are identical between the two layouts, so only the layout field can tell
        // them apart (engine #2861).
        CommandPlanMoeRoutingBundle router_bundle{};
        router_bundle.descriptor_expert_mask     = qwen.expert_mask_value;
        router_bundle.descriptor_partition_shift = qwen.partition_shift_value;
        router_bundle.descriptor_row_count_shift = qwen.row_count_shift_value;
        check(!moe_routing_bundle_readable_as_common(router_bundle),
              "a routing bundle written with the Qwen layout is not reusable by the common mul_mat_id");

        router_bundle.descriptor_expert_mask     = kMoePartitionExpertMask;
        router_bundle.descriptor_partition_shift = kMoePartitionShift;
        router_bundle.descriptor_row_count_shift = kMoePartitionRowCountShift;
        check(moe_routing_bundle_readable_as_common(router_bundle),
              "and a bundle written with the common layout still is, so the guard costs no table rebuild");
    }
    check(moe_routing_descriptor_is_common_layout(common.expert_mask_value, common.partition_shift_value,
                                                  common.row_count_shift_value),
          "a common-laid bundle is still reused, so the guard does not cost the fast path");
    check(!moe_routing_descriptor_is_common_layout(0, 0, 0),
          "an unlabelled bundle is refused rather than assumed readable");

    // 3. Why the guard is needed, at the boundary that produced the fault. The Qwen partition field sits at bit 7, the
    // common reader's expert field is 9 bits wide, so the low TWO bits of a Qwen partition are read as expert bits:
    // partitions congruent to 1, 2 or 3 (mod 4) push the expert index past the table, and the sizes still agree.
    int32_t escaped = 0;
    for (int32_t partition = 0; partition < 16; ++partition) {
        const int32_t descriptor = encode_descriptor(127, partition, 32, qwen);
        const int32_t expert     = decoded_expert(descriptor);
        const int32_t bleed      = partition & 3;
        if (bleed == 0) {
            check(expert == 127, "partitions congruent to 0 mod 4 keep the expert index in range");
        } else {
            ++escaped;
            check(expert == 127 + bleed * 128, "partitions 1/2/3 mod 4 decode to an expert 128/256/384 beyond the table");
            check(expert > 127, "and that expert does not exist, so the gather leaves the weight tensor");
            check(decoded_partition(descriptor) != partition, "the partition index it decodes is not the one written");
        }
    }
    check(escaped == 12, "12 of the 16 partitions of a 512-row expert escape");

    // 4. Two reasons a smaller batch is unaffected, and one reason sizes cannot prove safety. Up to 32 rows an expert
    // stays in partition 0, so nothing bleeds. The row count is WRONG even then (the Qwen row-count field starts at
    // bit 13, the common reader looks at bit 15), which is why "the tables are the same size" proves nothing.
    for (int32_t row_count = 1; row_count <= 32; ++row_count) {
        const int32_t descriptor = encode_descriptor(120, 0, row_count, qwen);
        check(decoded_expert(descriptor) == 120, "up to 32 rows in one partition keeps the expert index in range");
        check(decoded_row_count(descriptor) == (((row_count - 1) >> 2) & 31) + 1,
              "the common reader decodes a row count that is not the one written, in range or not");
    }
    {
        const int32_t descriptor = encode_descriptor(120, 1, 1, qwen);  // the 33rd row of that expert
        check(decoded_expert(descriptor) > 127,
              "the 33rd row of an expert is the first descriptor that escapes (tokens=33 is the minimal repro)");
    }

    if (failures != 0) {
        std::fprintf(stderr, "test-hrx-moe-routing-layout: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-hrx-moe-routing-layout: ok\n");
    return 0;
}
