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
#include "dispatch-small-rows.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kSoftmaxRowsKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_softmax_rows_f32");
static constexpr KernelCatalogRef kSumRowsKernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_sum_rows_f32");
static constexpr KernelCatalogRef kArgsortRowsKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_argsort_rows_f32");
static constexpr KernelCatalogRef kGetRowsSmallKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_small_f32");
static constexpr KernelCatalogRef kCopyStridedKernel  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_f32");

static bool packed(const Value & value, size_t element_size) {
    size_t stride = element_size;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] != stride) {
            return false;
        }
        stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static int64_t rows_of(const Value & value) {
    return value.ne[1] * value.ne[2] * value.ne[3];
}

static bool same_shape(const Value & a, const Value & b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a.ne[i] != b.ne[i]) {
            return false;
        }
    }
    return true;
}

// the single input and the output of a row op, both packed F32 (the output may be I32)
static bool row_op_values(const DispatchMatchContext & context, ggml_op op, ggml_type output_type,
                          const Value *& input, const Value *& output) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != op || node->inputs.size() != 1) {
        return false;
    }
    input  = context.graph.values().find(node->inputs[0]);
    output = context.graph.values().find(node->output);
    return input != nullptr && output != nullptr && input->type == GGML_TYPE_F32 && output->type == output_type &&
           packed(*input, sizeof(float)) && packed(*output, ggml_type_size(output_type)) &&
           output->alias_source.value < 0 && input->storage != output->storage && rows_of(*input) <= 16777216;
}

static void finish(const DispatchMatchContext & context, DispatchMatch & match, Dispatch && dispatch) {
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
}

static bool match_softmax_rows(const DispatchMatchContext & context, DispatchMatch & match) {
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!row_op_values(context, GGML_OP_SOFT_MAX, GGML_TYPE_F32, input, output) || !same_shape(*input, *output) ||
        input->ne[0] > 4096) {
        return false;
    }
    const SoftMaxParams * params = op_params_as<SoftMaxParams>(context.root_node->params);
    if (params == nullptr || params->scale != 1.0f || params->max_bias != 0.0f) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSoftmaxRowsKernel);
    dispatch.kernel.integer_parameters.emplace("column_count", input->ne[0]);
    dispatch.kernel.integer_parameters.emplace("row_count", rows_of(*input));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}

static bool match_sum_rows(const DispatchMatchContext & context, DispatchMatch & match) {
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!row_op_values(context, GGML_OP_SUM_ROWS, GGML_TYPE_F32, input, output) || input->ne[0] > 4096 ||
        output->ne[0] != 1 || output->ne[1] != input->ne[1] || output->ne[2] != input->ne[2] ||
        output->ne[3] != input->ne[3]) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSumRowsKernel);
    dispatch.kernel.integer_parameters.emplace("column_count", input->ne[0]);
    dispatch.kernel.integer_parameters.emplace("row_count", rows_of(*input));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}

static bool match_argsort_rows(const DispatchMatchContext & context, DispatchMatch & match) {
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!row_op_values(context, GGML_OP_ARGSORT, GGML_TYPE_I32, input, output) || !same_shape(*input, *output) ||
        input->ne[0] > 1024) {
        return false;
    }
    const ArgsortParams * params = op_params_as<ArgsortParams>(context.root_node->params);
    if (params == nullptr) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kArgsortRowsKernel);
    dispatch.kernel.integer_parameters.emplace("column_count", input->ne[0]);
    dispatch.kernel.integer_parameters.emplace("row_count", rows_of(*input));
    dispatch.kernel.integer_parameters.emplace("descending", params->order == GGML_SORT_ORDER_DESC ? 1 : 0);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}

// GET_ROWS within batches: source [W, S, B], ids [R, B] (I32, rows may be strided, as the first k
// columns of an ARGSORT are), output [W, R, B]. Only rows the
// regular get_rows kernel does not take (narrower than 4 floats, or not a multiple of 4).
static bool match_get_rows_small(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return false;
    }
    const Value * source = context.graph.values().find(node->inputs[0]);
    const Value * ids    = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    if (source == nullptr || ids == nullptr || output == nullptr || source->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || output->type != GGML_TYPE_F32 || !packed(*source, sizeof(float)) ||
        !packed(*output, sizeof(float)) || output->alias_source.value >= 0 || ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] % sizeof(int32_t) != 0 || static_cast<int64_t>(ids->nb[1] / sizeof(int32_t)) < ids->ne[0]) {
        return false;
    }
    const int64_t w = source->ne[0];
    const int64_t s = source->ne[1];
    const int64_t b = source->ne[2];
    const int64_t r = ids->ne[0];
    if ((w >= 4 && w % 4 == 0) || w > 65536 || s > 65536 || b > 65536 || r > 65536 || source->ne[3] != 1 ||
        ids->ne[1] != b || ids->ne[2] != 1 || ids->ne[3] != 1 || output->ne[0] != w || output->ne[1] != r ||
        output->ne[2] != b || output->ne[3] != 1) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsSmallKernel);
    dispatch.kernel.integer_parameters.emplace("width", w);
    dispatch.kernel.integer_parameters.emplace("id_count", r);
    dispatch.kernel.integer_parameters.emplace("batch_count", b);
    dispatch.kernel.integer_parameters.emplace("source_rows", s);
    dispatch.kernel.integer_parameters.emplace("id_stride", static_cast<int64_t>(ids->nb[1] / sizeof(int32_t)));
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}


// CONT of a strided F32 view (permuted, transposed or sliced) into a packed output. Only sources
// that are not packed: packed copies belong to the regular copy kernel.
static bool match_copy_strided(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONT || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        packed(*input, sizeof(float)) || !packed(*output, sizeof(float)) || output->alias_source.value >= 0 ||
        input->storage == output->storage || input->element_count != output->element_count) {
        return false;
    }
    int64_t strides[GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (input->nb[i] % sizeof(float) != 0 || input->ne[i] != output->ne[i]) {
            return false;
        }
        strides[i] = static_cast<int64_t>(input->nb[i] / sizeof(float));
    }
    const int64_t extent = static_cast<int64_t>(input->byte_count / sizeof(float));
    if (extent < 1 || extent > 268435456 || output->element_count > 268435456) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kCopyStridedKernel);
    static const char * const ne_names[] = { "ne0", "ne1", "ne2", "ne3" };
    static const char * const s_names[]  = { "s0", "s1", "s2", "s3" };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.integer_parameters.emplace(ne_names[i], output->ne[i]);
        dispatch.kernel.integer_parameters.emplace(s_names[i], strides[i]);
    }
    dispatch.kernel.integer_parameters.emplace("source_extent", extent);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}


// REPEAT that only broadcasts (every input dim is 1 or the output's): the strided copy with a zero
// stride on the broadcast dims. Tiling repeats (output a multiple of a larger input) are not claimed.
static bool match_repeat_broadcast(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_REPEAT || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !packed(*output, sizeof(float)) || output->alias_source.value >= 0 || input->storage == output->storage ||
        output->element_count > 268435456) {
        return false;
    }
    int64_t strides[GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (input->nb[i] % sizeof(float) != 0 || (input->ne[i] != 1 && input->ne[i] != output->ne[i])) {
            return false;
        }
        strides[i] = input->ne[i] == 1 ? 0 : static_cast<int64_t>(input->nb[i] / sizeof(float));
    }
    const int64_t extent = static_cast<int64_t>(input->byte_count / sizeof(float));
    if (extent < 1 || extent > 268435456) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kCopyStridedKernel);
    static const char * const ne_names[] = { "ne0", "ne1", "ne2", "ne3" };
    static const char * const s_names[]  = { "s0", "s1", "s2", "s3" };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.integer_parameters.emplace(ne_names[i], output->ne[i]);
        dispatch.kernel.integer_parameters.emplace(s_names[i], strides[i]);
    }
    dispatch.kernel.integer_parameters.emplace("source_extent", extent);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}

}  // namespace

void register_small_rows_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({ "common.softmax_rows_f32", GGML_OP_SOFT_MAX, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_softmax_rows });
    registry.add({ "common.sum_rows_f32", GGML_OP_SUM_ROWS, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_sum_rows });
    registry.add({ "common.argsort_rows_f32", GGML_OP_ARGSORT, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_argsort_rows });
    registry.add({ "common.get_rows_small_f32", GGML_OP_GET_ROWS, DispatchMatchKind::SingleOp, 0,
                   DispatchSource::Common, match_get_rows_small });
    registry.add({ "common.copy_strided_f32", GGML_OP_CONT, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_copy_strided });
    registry.add({ "common.repeat_broadcast_f32", GGML_OP_REPEAT, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_repeat_broadcast });
}

}  // namespace ggml::hrx
