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
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kSoftmaxRowsKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_softmax_rows_f32");
static constexpr KernelCatalogRef kSumRowsKernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_sum_rows_f32");
static constexpr KernelCatalogRef kArgsortRowsKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_argsort_rows_f32");
static constexpr KernelCatalogRef kGetRowsSmallKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_small_f32");
static constexpr KernelCatalogRef kCopyStridedKernel  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_f32");
static constexpr KernelCatalogRef kNormRowsKernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_norm_rows_f32");
static constexpr KernelCatalogRef kBinaryStridedKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_binary_strided_f32");
static constexpr KernelCatalogRef kClampKernel         = GGML_HRX_KERNEL_REF("loom_libs", "ggml_clamp_f32");
static constexpr KernelCatalogRef kClampInplaceKernel  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_clamp_inplace_f32");
static constexpr KernelCatalogRef kCopyF32F16Kernel    = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_f32_f16");
static constexpr KernelCatalogRef kAttentionStridedKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_attention_strided_f32_f16");
static constexpr KernelCatalogRef kAttentionRowsKernel    = GGML_HRX_KERNEL_REF("loom_libs", "ggml_attention_rows_f32_f16");
static constexpr KernelCatalogRef kMulMatSmallF16Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_small_f16_f32");
static constexpr KernelCatalogRef kMulMatSmallF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_small_f32_f32");
static constexpr KernelCatalogRef kMulMatSmallQ8Kernel  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_small_q8_0_f32");
static constexpr KernelCatalogRef kMulMatRowsQ8Kernel   = GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_rows_q8_0_f32");

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

// NORM (LayerNorm without affine; its weight and bias are separate MUL/ADD nodes) on packed F32 rows
static bool match_norm_rows(const DispatchMatchContext & context, DispatchMatch & match) {
    const Value * input  = nullptr;
    const Value * output = nullptr;
    if (!row_op_values(context, GGML_OP_NORM, GGML_TYPE_F32, input, output) || !same_shape(*input, *output) ||
        input->ne[0] > 65536) {
        return false;
    }
    const RmsNormParams * params = op_params_as<RmsNormParams>(context.root_node->params);
    if (params == nullptr) {
        return false;
    }
    std::ostringstream eps;
    eps.precision(9);
    eps << params->eps;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kNormRowsKernel);
    dispatch.kernel.integer_parameters.emplace("column_count", input->ne[0]);
    dispatch.kernel.integer_parameters.emplace("row_count", rows_of(*input));
    dispatch.kernel.compile_parameters.emplace("ggml.norm_rows_f32.epsilon", eps.str());
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

// ADD / SUB / MUL / DIV of F32 values with any element strides, broadcasting either input (ggml's
// rule: an input dim of 1 against a larger output dim), into a packed output. Registered below the
// packed binary kernels (priority -10), so it only takes what they refuse: strided views mostly.
static bool match_binary_strided(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 2) {
        return false;
    }
    int64_t op = -1;
    switch (node->op) {
        case GGML_OP_ADD: op = 0; break;
        case GGML_OP_SUB: op = 1; break;
        case GGML_OP_MUL: op = 2; break;
        case GGML_OP_DIV: op = 3; break;
        default: return false;
    }
    const Value * lhs    = context.graph.values().find(node->inputs[0]);
    const Value * rhs    = context.graph.values().find(node->inputs[1]);
    const Value * output = context.graph.values().find(node->output);
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->type != GGML_TYPE_F32 || rhs->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !packed(*output, sizeof(float)) || output->alias_source.value >= 0 ||
        lhs->storage == output->storage || rhs->storage == output->storage || output->element_count > 268435456) {
        return false;
    }
    int64_t a[GGML_MAX_DIMS], b[GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        for (const Value * v : { lhs, rhs }) {
            if (v->nb[i] % sizeof(float) != 0 || (v->ne[i] != 1 && v->ne[i] != output->ne[i])) {
                return false;
            }
        }
        a[i] = lhs->ne[i] == 1 ? 0 : static_cast<int64_t>(lhs->nb[i] / sizeof(float));
        b[i] = rhs->ne[i] == 1 ? 0 : static_cast<int64_t>(rhs->nb[i] / sizeof(float));
    }
    const int64_t a_extent = static_cast<int64_t>(lhs->byte_count / sizeof(float));
    const int64_t b_extent = static_cast<int64_t>(rhs->byte_count / sizeof(float));
    if (a_extent < 1 || b_extent < 1 || a_extent > 268435456 || b_extent > 268435456) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kBinaryStridedKernel);
    static const char * const ne_names[] = { "ne0", "ne1", "ne2", "ne3" };
    static const char * const a_names[]  = { "a0", "a1", "a2", "a3" };
    static const char * const b_names[]  = { "b0", "b1", "b2", "b3" };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        dispatch.kernel.integer_parameters.emplace(ne_names[i], output->ne[i]);
        dispatch.kernel.integer_parameters.emplace(a_names[i], a[i]);
        dispatch.kernel.integer_parameters.emplace(b_names[i], b[i]);
    }
    dispatch.kernel.integer_parameters.emplace("a_extent", a_extent);
    dispatch.kernel.integer_parameters.emplace("b_extent", b_extent);
    dispatch.kernel.integer_parameters.emplace("op", op);
    dispatch.bindings.push_back({ lhs->id, 0, lhs->byte_count });
    dispatch.bindings.push_back({ rhs->id, 0, rhs->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}

static std::string f32_config(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

// CLAMP of a packed F32 tensor on its own (fused router clamps match first, at their own priority)
static bool match_clamp(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CLAMP || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    const ClampParams * params = op_params_as<ClampParams>(node->params);
    if (input == nullptr || output == nullptr || params == nullptr || input->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !packed(*input, sizeof(float)) || !packed(*output, sizeof(float)) ||
        !same_shape(*input, *output) || output->element_count > 268435456) {
        return false;
    }
    // ggml_clamp is in place: the output is a view of the input, same layout
    const bool in_place = output->alias_source.value >= 0;
    if (in_place ? (output->storage != input->storage || output->storage_offset != input->storage_offset)
                 : input->storage == output->storage) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(in_place ? kClampInplaceKernel : kClampKernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.clamp_f32.min", f32_config(params->min));
    dispatch.kernel.compile_parameters.emplace("ggml.clamp_f32.max", f32_config(params->max));
    if (!in_place) {
        dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    }
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    finish(context, match, std::move(dispatch));
    return true;
}

// CPY F32 (any element strides) -> packed F16: ggml_cpy(src, dst) returns a view of dst, so the
// output aliases the second input, which only gives the destination's layout
static bool match_copy_f32_f16(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CPY || node->inputs.empty() || node->inputs.size() > 2) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F16 ||
        !packed(*output, ggml_type_size(GGML_TYPE_F16)) || input->storage == output->storage ||
        input->element_count != output->element_count || output->element_count > 268435456) {
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
    if (extent < 1 || extent > 268435456) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kCopyF32F16Kernel);
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

// FLASH_ATTN_EXT with the layouts the flash-attention kernels refuse (one contiguous block per head,
// as encoders lay them out): F32 query [d, n_q, h], F16 key/value [d, n_kv, h_kv], F16 mask
// [n_kv, >= n_q], output [dv, h, n_q] packed; no ALiBi, no softcap. Registered below them.
static bool match_attention_strided(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4) {
        return false;
    }
    const Value * q    = context.graph.values().find(node->inputs[0]);
    const Value * k    = context.graph.values().find(node->inputs[1]);
    const Value * v    = context.graph.values().find(node->inputs[2]);
    const Value * mask = context.graph.values().find(node->inputs[3]);
    const Value * out  = context.graph.values().find(node->output);
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node->params);
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr || out == nullptr || params == nullptr ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
        out->type != GGML_TYPE_F32 || params->max_bias != 0.0f || params->logit_softcap != 0.0f) {
        return false;
    }
    const int64_t d = q->ne[0], dv = v->ne[0], nq = q->ne[1], nkv = k->ne[1], nh = q->ne[2], nhkv = k->ne[2];
    if (k->ne[0] != d || v->ne[1] != nkv || v->ne[2] != nhkv || nhkv < 1 || nh % nhkv != 0 || d > 1024 || dv > 1024 ||
        dv > 1024 || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || mask->ne[0] < nkv || mask->ne[1] < nq ||
        mask->ne[2] != 1 || mask->ne[3] != 1 || out->ne[0] != dv || out->ne[1] != nh || out->ne[2] != nq ||
        out->ne[3] != 1 || !packed(*out, sizeof(float)) || q->nb[0] != sizeof(float) || k->nb[0] != 2 || v->nb[0] != 2 ||
        mask->nb[0] != 2 || q->nb[1] % 4 || q->nb[2] % 4 || k->nb[1] % 2 || k->nb[2] % 2 || v->nb[1] % 2 || v->nb[2] % 2 ||
        mask->nb[1] % 2 || nq > 65536 || nkv > 65536 || nh > 1024 || dv < 1 || dv > 1024 || (dv % 32) != 0) {
        return false;
    }
    Dispatch dispatch;
    // scores once per (query, head) in workgroup memory when they fit (ONEBIT_HRX_ATTN_PER_LANE=1: the old way)
    const bool rows = nkv <= 2048 && d <= 256 && dv <= 256 && std::getenv("ONEBIT_HRX_ATTN_PER_LANE") == nullptr;
    dispatch.kernel = make_kernel_specialization(rows ? kAttentionRowsKernel : kAttentionStridedKernel);
    auto & ip = dispatch.kernel.integer_parameters;
    ip.emplace("qk_size", d);
    ip.emplace("v_size", dv);
    ip.emplace("q_count", nq);
    ip.emplace("kv_count", nkv);
    ip.emplace("head_count", nh);
    ip.emplace("kv_head_count", nhkv);
    ip.emplace("q_s1", static_cast<int64_t>(q->nb[1] / 4));
    ip.emplace("q_s2", static_cast<int64_t>(q->nb[2] / 4));
    ip.emplace("k_s1", static_cast<int64_t>(k->nb[1] / 2));
    ip.emplace("k_s2", static_cast<int64_t>(k->nb[2] / 2));
    ip.emplace("v_s1", static_cast<int64_t>(v->nb[1] / 2));
    ip.emplace("v_s2", static_cast<int64_t>(v->nb[2] / 2));
    ip.emplace("m_s1", static_cast<int64_t>(mask->nb[1] / 2));
    ip.emplace("q_extent", static_cast<int64_t>(q->byte_count / 4));
    ip.emplace("k_extent", static_cast<int64_t>(k->byte_count / 2));
    ip.emplace("v_extent", static_cast<int64_t>(v->byte_count / 2));
    ip.emplace("m_extent", static_cast<int64_t>(mask->byte_count / 2));
    dispatch.kernel.compile_parameters.emplace("ggml.attention_strided.scale", f32_config(params->scale));
    for (const Value * b : { q, k, v, mask, out }) {
        dispatch.bindings.push_back({ b->id, 0, b->byte_count });
    }
    finish(context, match, std::move(dispatch));
    return true;
}

// MUL_MAT of a small F16/F32 weight [K, N] (rows packed, any K) or Q8_0 weight (K a multiple of 32) with F32 columns [K, T] into a
// packed [N, T]: heads and projections the tiled matmul kernels refuse (K not a multiple of 256).
// Registered below them; one workitem per output, so it is for small N x T only.
static bool match_mul_mat_small(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return false;
    }
    const Value * w   = context.graph.values().find(node->inputs[0]);
    const Value * x   = context.graph.values().find(node->inputs[1]);
    const Value * out = context.graph.values().find(node->output);
    const bool q8 = w != nullptr && w->type == GGML_TYPE_Q8_0;
    if (w == nullptr || x == nullptr || out == nullptr ||
        (w->type != GGML_TYPE_F16 && w->type != GGML_TYPE_F32 && !q8) || x->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || !packed(*out, sizeof(float))) {
        return false;
    }
    // element size, or for Q8_0 the block size (w_s1 then counts blocks)
    const size_t  wsz = q8 ? ggml_type_size(GGML_TYPE_Q8_0) : ggml_type_size(w->type);
    const int64_t k = w->ne[0], n = w->ne[1], t = x->ne[1];
    if (x->ne[0] != k || w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || out->ne[0] != n ||
        out->ne[1] != t || out->ne[2] != 1 || out->ne[3] != 1 || (!q8 && w->nb[0] != wsz) || w->nb[1] % wsz != 0 ||
        (q8 && k % 32 != 0) || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || n * t > (1 << 22) ||
        k > 1048576) {
        return false;
    }
    Dispatch dispatch;
    // Q8_0 with enough K: a workgroup per output, lanes over the blocks (ONEBIT_HRX_Q8_PER_OUTPUT=1: the old way)
    const bool rows = q8 && k >= 32 * 64 && std::getenv("ONEBIT_HRX_Q8_PER_OUTPUT") == nullptr;
    dispatch.kernel = make_kernel_specialization(rows ? kMulMatRowsQ8Kernel
                                                 : q8 ? kMulMatSmallQ8Kernel
                                                 : w->type == GGML_TYPE_F16 ? kMulMatSmallF16Kernel : kMulMatSmallF32Kernel);
    auto & ip = dispatch.kernel.integer_parameters;
    ip.emplace("k_size", k);
    ip.emplace("n_size", n);
    ip.emplace("t_count", t);
    ip.emplace("w_s1", static_cast<int64_t>(w->nb[1] / wsz));
    ip.emplace("x_s1", static_cast<int64_t>(x->nb[1] / sizeof(float)));
    ip.emplace("w_extent", static_cast<int64_t>(w->byte_count / wsz));
    ip.emplace("x_extent", static_cast<int64_t>(x->byte_count / sizeof(float)));
    for (const Value * b : { w, x, out }) {
        dispatch.bindings.push_back({ b->id, 0, b->byte_count });
    }
    finish(context, match, std::move(dispatch));
    return true;
}

}  // namespace

void register_small_rows_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({ "common.softmax_rows_f32", GGML_OP_SOFT_MAX, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_softmax_rows });
    registry.add({ "common.binary_strided_f32.add", GGML_OP_ADD, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_binary_strided });
    registry.add({ "common.binary_strided_f32.sub", GGML_OP_SUB, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_binary_strided });
    registry.add({ "common.binary_strided_f32.mul", GGML_OP_MUL, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_binary_strided });
    registry.add({ "common.binary_strided_f32.div", GGML_OP_DIV, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_binary_strided });
    registry.add({ "common.mul_mat_small_f32", GGML_OP_MUL_MAT, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_mul_mat_small });
    registry.add({ "common.attention_strided_f32_f16", GGML_OP_FLASH_ATTN_EXT, DispatchMatchKind::SingleOp, -10,
                   DispatchSource::Common, match_attention_strided });
    registry.add({ "common.copy_strided_f32_f16", GGML_OP_CPY, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_copy_f32_f16 });
    registry.add({ "common.clamp_f32", GGML_OP_CLAMP, DispatchMatchKind::SingleOp, -10, DispatchSource::Common,
                   match_clamp });
    registry.add({ "common.norm_rows_f32", GGML_OP_NORM, DispatchMatchKind::SingleOp, 0, DispatchSource::Common,
                   match_norm_rows });
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
