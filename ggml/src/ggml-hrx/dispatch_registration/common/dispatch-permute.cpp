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

#include "dispatch-permute.h"

#include "dispatch-layout-utils.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

// Why this exists (engine #444's sibling work, TSPECTREE phase (a)): the DFlash2 drafter's encoder and
// decoder permute activations between the head-major and token-major layouts, and llama.cpp's graph leaves
// those as GGML_OP_PERMUTE nodes. Our dispatcher had no registration for PERMUTE at all, so a permuted
// value that is not followed by a CONT was placed on the CPU, which splits the drafter graph once per draft.
//
// A permute does not move data (the result is a view with the strides travelling with the dims), so the
// materialised case is a strided read into a contiguous destination - the same lowering register_copy_dispatch
// already performs for CONT. The view case has no destination buffer and stays where it was (see the header).

namespace ggml::hrx {
namespace {

static constexpr uint64_t         kMaximumPermuteElements = uint64_t{ 1 } << 30;
static constexpr KernelCatalogRef kCopyStridedSourceF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_source_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool packed_f32_layout(const Value & value) {
    size_t expected_stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static bool element_strides(const Value & value) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] <= 0 || value.nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    return value.byte_count % sizeof(float) == 0;
}

static bool match_permute_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_PERMUTE || node->inputs.size() != 1) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || output == nullptr || source->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return false;
    }

    // Only the materialised form: a permuted value that owns a contiguous destination buffer. The view form
    // (source and output share storage) has no destination and is left to the following CONT or to the CPU.
    if (source->storage == output->storage || !output->contiguous || !packed_f32_layout(*output) ||
        output->element_count <= 0 || output->element_count != source->element_count ||
        output->byte_count != source->byte_count || !element_strides(*source) ||
        static_cast<uint64_t>(output->element_count) > kMaximumPermuteElements) {
        return false;
    }

    // A permute keeps each dimension's stride with the dimension, so the destination's nb[i] is exactly the
    // source stride to read for destination dim i - no axes have to be recovered from op_params.
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kCopyStridedSourceF32Kernel);
    dispatch.kernel.integer_parameters.emplace("source_span", source->byte_count / sizeof(float));
    dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne0", std::to_string(output->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne1", std::to_string(output->ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne2", std::to_string(output->ne[2]));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.stride" + std::to_string(i),
                                                   std::to_string(output->nb[i] / sizeof(float)));
    }
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ source->storage_root, source->storage_offset, source->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_permute_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.permute_f32",
        GGML_OP_PERMUTE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_permute_f32_dispatch,
    });
}

}  // namespace ggml::hrx
