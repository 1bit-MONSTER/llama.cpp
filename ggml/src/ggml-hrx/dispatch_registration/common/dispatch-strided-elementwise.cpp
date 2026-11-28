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
// engine#3491: a SCALE or a UNARY whose input is a STRIDED f32 view - Inkling's MoE SITU activation scales and sigmoids
// `gate` and `up`, the two halves of the fused gate_up expert output (1152 of every 2304 columns) - was off HRX: the
// scale_bias / unary kernels read a packed input, and their matchers decline anything else. These matchers copy the
// view into a packed transient (append_f32_densify_dispatch, the strided-source copy) and then run the same kernel on
// it. A packed input keeps the existing matchers (common.scale_bias_f32, common.unary_f32), which run first.

#include "dispatch-strided-elementwise.h"
#include "dispatch-copy.h"

#include "ggml.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kUnaryF32Kernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_unary_f32");
static constexpr KernelCatalogRef kScaleBiasF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_scale_bias_f32");

static std::string float_config_value(float value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

// the strided case only: an f32 input that is NOT contiguous (the packed one has its own matcher), a packed output
static bool strided_elementwise_values(const DispatchMatchContext & context, const Value *& input,
                                       const Value *& output) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) {
        return false;
    }
    input  = context.graph.values().find(node->inputs[0]);
    output = context.graph.values().find(node->output);
    return input != nullptr && output != nullptr && input->type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32 &&
           !input->contiguous && output->contiguous && same_shape(*input, *output) && output->element_count > 0 &&
           static_cast<uint64_t>(output->element_count) <= std::numeric_limits<uint32_t>::max() &&
           output->alias_source.value < 0 && input->storage != output->storage;
}

// densify the input into a fresh transient and return the binding the elementwise kernel reads
static bool densify_input(const DispatchMatchContext & context, const Value & input, DispatchMatch & match,
                          DispatchBinding & packed) {
    const size_t  bytes = static_cast<size_t>(input.element_count) * sizeof(float);
    const ValueId id    = context.next_plan_value;
    Dispatch      densify;
    if (!append_f32_densify_dispatch(input, id, bytes, densify)) {
        return false;
    }
    match.transients.push_back({ id, "common.strided_elementwise.input_packed", bytes, 256 });
    match.dispatches.push_back(std::move(densify));
    packed = { id, 0, bytes };
    return true;
}

static bool match_scale_strided_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    if (context.root_node == nullptr || context.root_node->op != GGML_OP_SCALE) {
        return false;
    }
    const ScaleParams * params = op_params_as<ScaleParams>(context.root_node->params);
    const Value *       input  = nullptr;
    const Value *       output = nullptr;
    if (params == nullptr || !std::isfinite(params->scale) || !std::isfinite(params->bias) ||
        !strided_elementwise_values(context, input, output)) {
        return false;
    }
    DispatchBinding packed;
    if (!densify_input(context, *input, match, packed)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleBiasF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale.scale", float_config_value(params->scale));
    dispatch.kernel.compile_parameters.emplace("ggml.scale.bias", float_config_value(params->bias));
    dispatch.bindings.push_back(packed);
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_unary_strided_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    if (context.root_node == nullptr) {
        return false;
    }
    const UnaryParams * params = op_params_as<UnaryParams>(context.root_node->params);
    const Value *       input  = nullptr;
    const Value *       output = nullptr;
    if (params == nullptr || !unary_kind_supported(params->op) || !strided_elementwise_values(context, input, output)) {
        return false;
    }
    DispatchBinding packed;
    if (!densify_input(context, *input, match, packed)) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kUnaryF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.unary_f32.op",
                                               std::to_string(unary_kind_config_value(params->op)));
    dispatch.bindings.push_back(packed);
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_strided_elementwise_dispatches(DispatchRegistryBuilder & registry) {
    // priority -5: below the packed matchers (0), which decline a strided input anyway
    registry.add({ "common.scale_strided_f32", GGML_OP_SCALE, DispatchMatchKind::SingleOp, -5, DispatchSource::Common,
                   match_scale_strided_f32 });
    for (ggml_op op : { GGML_OP_UNARY, GGML_OP_SQR, GGML_OP_SQRT, GGML_OP_LOG, GGML_OP_SIN, GGML_OP_COS }) {
        registry.add({ "common.unary_strided_f32", op, DispatchMatchKind::SingleOp, -5, DispatchSource::Common,
                       match_unary_strided_f32 });
    }
}

}  // namespace ggml::hrx
