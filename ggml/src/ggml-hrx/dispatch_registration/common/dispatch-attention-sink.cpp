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

// Attention sinks for FLASH_ATTN_EXT (gpt-oss): the FlashAttention dispatch runs without the sink and
// ops/attention_sink_f32.loom then rescales its output in place, row by row, by S / (S + exp(sink - M)), which is
// exact (see the kernel). dispatch-flash-attention.cpp accepts the 5-input node when attention_sinks_supported
// holds and appends this dispatch after its own.

#include "dispatch-attention-sink.h"

#include "dispatch-mul-mat-common.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kAttentionSinkF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_attention_sink_f32");

std::string index_config(int64_t value) {
    return std::to_string(value);
}

// True when the two values share storage and their byte ranges intersect.
bool overlaps(const Graph & graph, const Value & lhs, const Value & rhs) {
    if (lhs.id == rhs.id) {
        return true;
    }
    if (!graph.values().same_storage(lhs.id, rhs.id)) {
        return false;
    }
    const size_t lhs_end = lhs.storage_offset + lhs.byte_count;
    const size_t rhs_end = rhs.storage_offset + rhs.byte_count;
    return lhs.storage_offset < rhs_end && rhs.storage_offset < lhs_end;
}

}  // namespace

bool attention_sinks_supported(const Graph & graph, const GraphNode & node) {
    if (node.op != GGML_OP_FLASH_ATTN_EXT || node.inputs.size() != 5) {
        return false;
    }
    const Value * query = graph.values().find(node.inputs[0]);
    const Value * sinks = graph.values().find(node.inputs[4]);
    if (query == nullptr || sinks == nullptr || sinks->type != GGML_TYPE_F32 || !sinks->contiguous) {
        return false;
    }
    const int64_t query_head_count = query->ne[2];
    return sinks->ne[0] == query_head_count && sinks->ne[1] == 1 && sinks->ne[2] == 1 && sinks->ne[3] == 1;
}

bool append_attention_sink_dispatch(const Graph & graph, const GraphNode & node, DispatchMatch & dispatch_match) {
    if (!attention_sinks_supported(graph, node)) {
        return false;
    }
    const Value *              query  = graph.values().find(node.inputs[0]);
    const Value *              key    = graph.values().find(node.inputs[1]);
    const Value *              mask   = graph.values().find(node.inputs[3]);
    const Value *              sinks  = graph.values().find(node.inputs[4]);
    const Value *              output = graph.values().find(node.output);
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node.params);
    if (key == nullptr || mask == nullptr || output == nullptr || params == nullptr) {
        return false;
    }
    // The FlashAttention matchers have checked these layouts (query [tokens][heads][d] f32, key
    // [capacity][kv_heads][d] f16, output [tokens][heads][dv] f32); the mask rows must be key_count apart.
    const int64_t tokens               = query->ne[1];
    const int64_t query_head_count     = query->ne[2];
    const int64_t qk_head_size         = query->ne[0];
    const int64_t key_capacity         = key->ne[1];
    const int64_t key_value_head_count = key->ne[2];
    const int64_t value_head_size      = output->ne[0];
    const int64_t key_count            = mask->ne[0];
    if (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(ggml_fp16_t) ||
        mask->nb[1] != static_cast<size_t>(key_count) * sizeof(ggml_fp16_t) || key_count < 1 ||
        key_count > key_capacity || query_head_count > 256 || key_value_head_count > 256 || qk_head_size % 16 != 0 ||
        value_head_size % 16 != 0 || qk_head_size > 576 || value_head_size > 576) {
        return false;
    }
    // The output is rewritten in place after FlashAttention wrote it; no input may overlap its bytes (views of one
    // allocation share a storage root but are distinct values, so compare storage, not value ids).
    for (const Value * input : { query, key, mask, sinks }) {
        if (overlaps(graph, *input, *output)) {
            return false;
        }
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAttentionSinkF32Kernel);
    dispatch.kernel.integer_parameters.emplace("tokens", tokens);
    dispatch.kernel.integer_parameters.emplace("key_count", key_count);
    dispatch.kernel.integer_parameters.emplace("key_capacity", key_capacity);
    dispatch.kernel.compile_parameters.emplace("ggml.attention_sink.query_head_count", index_config(query_head_count));
    dispatch.kernel.compile_parameters.emplace("ggml.attention_sink.key_value_head_count",
                                               index_config(key_value_head_count));
    dispatch.kernel.compile_parameters.emplace("ggml.attention_sink.qk_head_size", index_config(qk_head_size));
    dispatch.kernel.compile_parameters.emplace("ggml.attention_sink.value_head_size", index_config(value_head_size));
    dispatch.kernel.compile_parameters.emplace("ggml.attention_sink.scale", common_to_config_value(params->scale));
    dispatch.bindings.push_back({ query->id, 0, query->byte_count });
    dispatch.bindings.push_back({ key->id, 0, key->byte_count });
    dispatch.bindings.push_back({ mask->id, 0, mask->byte_count });
    dispatch.bindings.push_back({ sinks->id, 0, sinks->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace ggml::hrx
