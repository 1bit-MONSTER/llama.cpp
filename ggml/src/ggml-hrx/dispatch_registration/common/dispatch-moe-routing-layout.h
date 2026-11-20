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

// Descriptor layout of the MoE partition table that the MoE router dispatch builds.
//
// ggml_moe_build_expert_partition_table packs each 32-row partition as
// expert | partition << partition_shift | (row_count - 1) << row_count_shift, and it
// runs one workgroup with one lane per expert. Two layouts are in use:
//
// - Qwen layout (7-bit expert: mask 127, shifts 7 and 13, 128 lanes). The qwen3_moe
//   routed kernels decode this layout with hard-coded constants, and they only run
//   for 128 experts.
// - Common layout (9-bit expert: mask 511, shifts 9 and 15, 512 lanes). The common
//   mul_mat_id kernels (ggml_moe_unpack_expert_partition_descriptor) decode this one
//   with hard-coded constants, for up to 512 experts.
//
// The router used the Qwen layout for every expert count. With more than 128 experts
// (Qwen3.6-35B-A3B has 256) the table builder masked expert ids to 7 bits, gave no
// partition to experts 128 and up (128 lanes), and the common mul_mat_id kernels that
// consume the table decoded row counts as partition ordinals. They then read assignment
// ordinals past the expert table and faulted (HSA_STATUS_ERROR_MEMORY_FAULT) on any
// prompt batch in which one expert received two or more tokens. No Qwen-layout
// consumer matches above 128 experts, so the router emits the common layout there.

#include "dispatch/command-plan-metadata.h"
#include <cstdint>

namespace ggml::hrx {

struct MoeRoutingDescriptorLayout {
    const char * expert_mask;
    const char * partition_shift;
    const char * row_count_shift;
    const char * partition_workgroup_size;
    // the same three numbers as values, for comparing a written bundle against the layout a reader decodes
    int32_t expert_mask_value;
    int32_t partition_shift_value;
    int32_t row_count_shift_value;
};

inline constexpr int64_t kMoeRoutingQwenLayoutMaxExpertCount = 128;

// The layout has to be the one the CONSUMER decodes, not the one that follows from the expert count. The qwen fused
// routed kernels decode the Qwen layout and only run at 128 experts; the common mul_mat_id decodes the common layout
// and runs for 1..512 experts. A plan carries one table per route-ids value, so the writer cannot satisfy both and must
// ask who is reading (engine#2861).
inline MoeRoutingDescriptorLayout moe_router_descriptor_layout(int64_t expert_count, bool consumer_reads_common_layout) {
    if (!consumer_reads_common_layout && expert_count <= kMoeRoutingQwenLayoutMaxExpertCount) {
        return { "127", "7", "13", "128", 127, 7, 13 };
    }
    return { "511", "9", "15", "512", 511, 9, 15 };
}

// True when a partition table written with (mask, partition_shift, row_count_shift) can be decoded by a reader that
// uses the common layout. The sizes matching is not enough: the two layouts place the partition and row-count fields
// at different bit offsets, so a table written with one and read with the other yields a wrong expert index.
inline bool moe_routing_descriptor_is_common_layout(int32_t expert_mask,
                                                   int32_t partition_shift,
                                                   int32_t row_count_shift) {
    return expert_mask == kMoePartitionExpertMask && partition_shift == kMoePartitionShift &&
           row_count_shift == kMoePartitionRowCountShift;
}

// True when a routing bundle may be reused by a reader that decodes the common layout. A bundle carries the layout it
// was written with precisely so that this question has an answer; before engine #2861 the only fields compared were the
// sizes, which agree between the two layouts.
inline bool moe_routing_bundle_readable_as_common(const CommandPlanMoeRoutingBundle & bundle) {
    return moe_routing_descriptor_is_common_layout(bundle.descriptor_expert_mask, bundle.descriptor_partition_shift,
                                                   bundle.descriptor_row_count_shift);
}

}  // namespace ggml::hrx
