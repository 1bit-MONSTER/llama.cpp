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
// engine#3554: OpenELM's per-head q/k RMS_NORM reads a VIEW of the fused QKV output, f32[head_dim, n_head, tokens]
// whose head stride is head_dim but whose token stride is the whole fused row (q + k + v heads). The rmsnorm kernel
// reads row r at r * input_stride, one stride, so common.rmsnorm_f32 (and every fused RMS_NORM matcher, which share its
// layout guard) declines such a view, and the node HRX had claimed was left with no matcher: graph_compute failed
// ("unsupported HRX node ... RMS_NORM ... <-VIEW"). This matcher copies the view into a packed transient
// (append_f32_densify_dispatch, the strided-source copy) and runs the same ggml_rmsnorm_f32 kernel on it. It is
// registered below common.rmsnorm_f32, so every layout that matcher reads directly still goes to it.

#include "dispatch-rmsnorm-strided.h"
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
#include <type_traits>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kRmsNormF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rmsnorm_f32");

template <typename T> static std::string config_value(T value) {
    std::ostringstream stream;
    if constexpr (std::is_floating_point_v<T>) {
        stream << std::setprecision(std::numeric_limits<T>::max_digits10);
    }
    stream << value;
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

static bool match_rmsnorm_strided_f32(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_RMS_NORM || node->inputs.size() != 1 ||
        context.root_index >= context.covered_nodes.size() || context.covered_nodes[context.root_index]) {
        return false;
    }
    const RmsNormParams * params = op_params_as<RmsNormParams>(node->params);
    const Value *         input  = context.graph.values().find(node->inputs[0]);
    const Value *         output = context.graph.values().find(node->output);
    if (params == nullptr || !std::isfinite(params->eps) || params->eps <= 0.0f || input == nullptr ||
        output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || input->contiguous ||
        input->nb[0] != sizeof(float) || !output->contiguous || output->alias_source.value >= 0 ||
        !same_shape(*input, *output) || input->storage_root == output->storage_root) {
        return false;
    }
    // the same row sizes the packed kernel takes (common.rmsnorm_f32's is_supported_f32_hidden_size)
    const int64_t hidden_size = output->ne[0];
    if (hidden_size < 64 || hidden_size > 32768 || hidden_size % 64 != 0 || output->element_count <= 0 ||
        output->element_count % hidden_size != 0) {
        return false;
    }
    const int64_t token_count = output->element_count / hidden_size;
    if (token_count < 1 || token_count > (int64_t(1) << 20)) {
        return false;
    }

    const size_t  packed_bytes = static_cast<size_t>(input->element_count) * sizeof(float);
    const ValueId packed       = context.next_plan_value;
    Dispatch      densify;
    if (!append_f32_densify_dispatch(*input, packed, packed_bytes, densify)) {
        return false;
    }
    match.transients.push_back({ packed, "common.rmsnorm_strided.input_packed", packed_bytes, 256 });
    match.dispatches.push_back(std::move(densify));

    Dispatch dispatch;
    dispatch.kernel                         = make_kernel_specialization(kRmsNormF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("token_count", token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.hidden_size", config_value(hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.rms_epsilon", config_value(params->eps));
    dispatch.kernel.compile_parameters.emplace("ggml.rmsnorm_f32.input_stride", config_value(hidden_size));
    dispatch.bindings.push_back({ packed, 0, packed_bytes });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_rmsnorm_strided_dispatches(DispatchRegistryBuilder & registry) {
    // priority -5: below common.rmsnorm_f32 (0), which takes every layout its kernel reads directly
    registry.add({ "common.rmsnorm_strided_f32", GGML_OP_RMS_NORM, DispatchMatchKind::SingleOp, -5,
                   DispatchSource::Common, match_rmsnorm_strided_f32 });
}

}  // namespace ggml::hrx
