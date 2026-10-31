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

// GGML_OP_TOP_K for HRX (engine #679) -> hip_top_k_f32 (hip/kernels/hip_top_k_f32.hip).
//
// Qwen3.8-Flash-Next's QSA selection is the reason this matcher exists: without it the node runs on the CPU
// backend and every full-attention layer's graph is split around it (measured: +9.7 ms per decode step at
// n_kv > 2051 and ~3.8 us per KV token after that). It is a SingleOp matcher at priority 10; there is no
// competing TOP_K matcher, and the claim is bounded in ggml-hrx.cpp (device_supports_op) so a shape this
// matcher cannot take still falls back to the CPU backend instead of failing the graph.

#include "hip/hip-capabilities.h"
#include "hip/hip-dispatches.h"
#include "hip/hip-kernel-registry.h"

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {

// Bounds the kernel's 32-bit key map and loop counters allow. ggml-hrx.cpp's device_supports_op asks
// hip_top_k_tensor_supported, which checks the same limit, so the claim and the matcher cannot drift apart.
constexpr int64_t kTopKMaxColumnCount = 262144;

// Shared with ggml-hrx.cpp's per-node support check so the claim never promises a node that no matcher takes.
bool top_k_shape_supported(const Value & input, const Value & output) {
    if (input.type != GGML_TYPE_F32 || output.type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t column_count = input.ne[0];
    const int64_t top_k        = output.ne[0];
    const int64_t row_count    = output.ne[1] * output.ne[2] * output.ne[3];
    if (column_count <= 0 || column_count > kTopKMaxColumnCount || top_k <= 0 || top_k > column_count) {
        return false;
    }
    if (row_count <= 0 || row_count > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    // ggml_top_k keeps the leading dimensions of its source and replaces ne[0] with k.
    return input.ne[1] == output.ne[1] && input.ne[2] == output.ne[2] && input.ne[3] == output.ne[3] &&
           output.nb[0] == sizeof(int32_t);
}

namespace {

constexpr KernelCatalogRef kHipTopKKernel = hip_kernel_ref("hip_top_k_f32");

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

bool launch_hip_top_k(const std::map<std::string, int64_t> & parameters, HipLaunchGeometry & geometry) {
    const auto rows = parameters.find("row_count");
    const auto k    = parameters.find("top_k");
    if (rows == parameters.end() || k == parameters.end() || rows->second <= 0 || k->second <= 0) {
        return false;
    }
    geometry.workgroup_count = { static_cast<uint32_t>(rows->second), 1, 1 };
    geometry.workgroup_size  = { 256, 1, 1 };
    return true;
}

bool match_hip_top_k(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_TOP_K || node->inputs.size() != 1) {
        return false;
    }
    const Value * input  = context.graph.values().find(node->inputs[0]);
    const Value * output = context.graph.values().find(node->output);
    if (input == nullptr || output == nullptr || !packed_f32(*input)) {
        return false;
    }
    if (output->type != GGML_TYPE_I32 || !output->contiguous || !top_k_shape_supported(*input, *output)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kHipTopKKernel);
    dispatch.kernel.integer_parameters.emplace("column_count", input->ne[0]);
    dispatch.kernel.integer_parameters.emplace("top_k", output->ne[0]);
    dispatch.kernel.integer_parameters.emplace("row_count", output->ne[1] * output->ne[2] * output->ne[3]);
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

bool hip_top_k_tensor_supported(const struct ggml_tensor * op) {
    // The claim must follow the matcher, including the debug switch: with GGML_HRX_DISABLE_DISPATCH naming this
    // registration the node has to fall back to the CPU backend, not become an unsupported HRX node.
    const char * disabled = std::getenv("GGML_HRX_DISABLE_DISPATCH");
    if (disabled != nullptr && std::string(disabled).find("hip.top_k_f32") != std::string::npos) {
        return false;
    }
    if (op == nullptr || op->src[0] == nullptr || !ggml_is_contiguous(op->src[0]) ||
        !ggml_is_contiguous(op)) {
        return false;
    }
    const int64_t column_count = op->src[0]->ne[0];
    const int64_t top_k        = op->ne[0];
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32 || column_count <= 0 ||
        column_count > kTopKMaxColumnCount || top_k <= 0 || top_k > column_count) {
        return false;
    }
    return op->src[0]->ne[1] == op->ne[1] && op->src[0]->ne[2] == op->ne[2] && op->src[0]->ne[3] == op->ne[3];
}

void register_hip_top_k_dispatch(DispatchRegistryBuilder & registry) {
    if (!hip_code_object_embedded("hip_top_k_f32")) {
        return;
    }
    ggml::hrx::hip_declare_eager_op(GGML_OP_TOP_K);
    ggml::hrx::register_hip_kernel({
        "hip_top_k_f32",
        "hip_top_k_f32",
        { { "source", ResourceAccess::Read }, { "output", ResourceAccess::Write } },
        { "column_count", "top_k" },   // by-value kernel arguments, in order
        { "row_count" },               // read by launch_hip_top_k to size the grid
        launch_hip_top_k,
    });
    registry.add({ "hip.top_k_f32", GGML_OP_TOP_K, DispatchMatchKind::SingleOp, 10, DispatchSource::Common,
                   match_hip_top_k });
}

}  // namespace ggml::hrx
