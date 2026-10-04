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

// Example matcher for the HRX HIP recipe: GGML_OP_SCALE (packed f32) -> hip_scale_f32
// (hip/kernels/hip_scale_f32.hip). Opt-in with GGML_HRX_HIP_EXAMPLE_SCALE=1 so it never replaces
// the Loom scale kernel in normal runs; it exists to prove and document the matcher path.

#include "hip/hip-dispatches.h"
#include "hip/hip-kernel-registry.h"

#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace ggml::hrx {
namespace {

constexpr KernelCatalogRef kHipScaleKernel = hip_kernel_ref("hip_scale_f32");

bool packed_f32(const Value & value) {
    size_t stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] != stride) {
            return false;
        }
        stride *= static_cast<size_t>(value.ne[i]);
    }
    return value.type == GGML_TYPE_F32 && value.contiguous;
}

bool launch_hip_scale(const std::map<std::string, int64_t> & parameters, HipLaunchGeometry & geometry) {
    const auto n = parameters.find("element_count");
    if (n == parameters.end() || n->second <= 0) {
        return false;
    }
    const int64_t groups       = (n->second + 255) / 256;
    geometry.workgroup_count   = { static_cast<uint32_t>(groups < 4096 ? groups : 4096), 1, 1 };
    geometry.workgroup_size    = { 256, 1, 1 };
    return true;
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool match_hip_scale(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SCALE || node->inputs.size() != 1) {
        return false;
    }
    const ScaleParams * params = op_params_as<ScaleParams>(node->params);
    const Value *       output = context.graph.values().find(node->output);
    const Value *       input  = context.graph.values().find(node->inputs[0]);
    if (params == nullptr || output == nullptr || input == nullptr || !packed_f32(*output) || !packed_f32(*input) ||
        input->element_count != output->element_count || output->alias_source.value >= 0 ||
        input->alias_source.value >= 0 || input->storage == output->storage ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kHipScaleKernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("scale_bits", float_bits(params->scale));
    dispatch.kernel.integer_parameters.emplace("bias_bits", float_bits(params->bias));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_hip_scale_dispatch(DispatchRegistryBuilder & registry) {
    const char * enabled = std::getenv("GGML_HRX_HIP_EXAMPLE_SCALE");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0 || !hip_code_object_embedded("hip_scale_f32")) {
        return;
    }
    register_hip_kernel({
        "hip_scale_f32",
        "hip_scale_f32",
        { { "input", ResourceAccess::Read }, { "output", ResourceAccess::Write } },
        { "element_count", "scale_bits", "bias_bits" },
        { "element_count" },
        launch_hip_scale,
    });
    registry.add({ "hip.scale_f32", GGML_OP_SCALE, DispatchMatchKind::SingleOp, 10, DispatchSource::Common,
                   match_hip_scale });
}

}  // namespace ggml::hrx
