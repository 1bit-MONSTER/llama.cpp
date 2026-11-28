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

// Standalone ROPE, mode GGML_ROPE_TYPE_VISION, f32 (ops/rope_vision_f32.loom). engine#3549: a vision projector rotates
// strided Q/K views with ggml_rope_multi(..., VISION); every other HRX rotation is fused (attention, ROPE+SET_ROWS),
// so these nodes ran on the CPU and strict placement aborted the image encode (rope-claim-params.h, engine#1467).

#include "dispatch-rope-vision.h"

#include "dispatch-mul-mat-common.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kRopeVisionF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_vision_f32");
static constexpr uint64_t         kMaximumElements     = uint64_t{ 1 } << 30;

bool match_rope_vision_f32(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 2) {
        return false;  // a third input (freq factors) is not implemented
    }
    const RopeParams * params = op_params_as<RopeParams>(node->params);
    if (params == nullptr || params->mode != GGML_ROPE_TYPE_VISION || params->n_offs != 0 ||
        params->ext_factor != 0.0f || !std::isfinite(params->freq_base) || params->freq_base <= 0.0f ||
        !std::isfinite(params->freq_scale) || !std::isfinite(params->attn_factor) || params->n_dims <= 0) {
        return false;
    }
    const Value * source    = context.graph.values().find(node->inputs[0]);
    const Value * positions = context.graph.values().find(node->inputs[1]);
    const Value * output    = context.graph.values().find(node->output);
    if (source == nullptr || positions == nullptr || output == nullptr || source->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || positions->type != GGML_TYPE_I32 || !output->contiguous ||
        !positions->contiguous || source->nb[0] != sizeof(float) || source->ne != output->ne ||
        source->ne[0] != 2 * static_cast<int64_t>(params->n_dims) ||
        positions->element_count < 4 * source->ne[2] || output->element_count <= 0 ||
        static_cast<uint64_t>(output->element_count) > kMaximumElements) {
        return false;
    }
    const int s0 = params->sections[0], s1 = params->sections[1], s2 = params->sections[2], s3 = params->sections[3];
    if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0 || s0 + s1 + s2 + s3 <= 0) {
        return false;
    }
    uint64_t span = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source->ne[i] <= 0 || source->nb[i] == 0 || source->nb[i] % sizeof(float) != 0) {
            return false;
        }
        span += static_cast<uint64_t>(source->ne[i] - 1) * (source->nb[i] / sizeof(float));
    }
    if (span > kMaximumElements) {
        return false;
    }
    // ggml_compute_forward_rope_flt's theta_scale, in the same float arithmetic
    const float theta_scale = powf(params->freq_base, -2.0f / static_cast<float>(params->n_dims));
    Dispatch          dispatch;
    const std::string k = "ggml.rope_vision_f32.";
    dispatch.kernel     = make_kernel_specialization(kRopeVisionF32Kernel);
    const auto set_i    = [&](const char * name, int64_t value) {
        dispatch.kernel.compile_parameters.emplace(k + name, std::to_string(value));
    };
    set_i("n_dims", params->n_dims);
    set_i("ne1", source->ne[1]);
    set_i("ne2", source->ne[2]);
    set_i("stride1", static_cast<int64_t>(source->nb[1] / sizeof(float)));
    set_i("stride2", static_cast<int64_t>(source->nb[2] / sizeof(float)));
    set_i("stride3", static_cast<int64_t>(source->nb[3] / sizeof(float)));
    set_i("sect_dims", s0 + s1 + s2 + s3);
    set_i("sec_h", s0);
    set_i("sec_w", s0 + s1);
    set_i("sec_e", s0 + s1 + s2);
    dispatch.kernel.compile_parameters.emplace(k + "theta_scale", common_to_config_value(theta_scale));
    dispatch.kernel.compile_parameters.emplace(k + "freq_scale", common_to_config_value(params->freq_scale));
    dispatch.kernel.compile_parameters.emplace(k + "attn_factor", common_to_config_value(params->attn_factor));
    dispatch.kernel.integer_parameters.emplace("pair_count", output->element_count / 2);
    dispatch.kernel.integer_parameters.emplace("source_span", static_cast<int64_t>(span));
    dispatch.kernel.integer_parameters.emplace("position_count", positions->element_count);
    dispatch.bindings.push_back({ source->storage_root, source->storage_offset, static_cast<size_t>(span) * sizeof(float) });
    dispatch.bindings.push_back({ positions->id, 0, positions->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_rope_vision_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.rope_vision_f32",
        GGML_OP_ROPE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_rope_vision_f32,
    });
}

}  // namespace ggml::hrx
