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

// Flash attention over several KV streams (engine#2918).
//
// A context with n_seq_max > 1 and no unified KV cache (llama-perplexity's 4 parallel sequences, llama-server
// --parallel N without -kvu) gives every sequence its own KV stream, and FLASH_ATTN_EXT then carries the stream
// in dim 3: q [D, T, H, S], k/v [D, n_kv, Hkv, S], mask [n_kv, T, 1, S], out [D, H, T, S]. Every other HRX
// flash-attention matcher requires ne[3] == 1, so the node was unsupported, llama_context::resolve_fused_ops saw
// it on the CPU in its 1-token probe and turned flash attention off for the whole context (LFM3-1.7B under
// llama-perplexity, "Flash Attention (2 occurrences)").
//
// The streams are independent: stream s only reads its own q rows, its own K/V cache stream and its own mask
// plane. So this matcher runs the existing ggml_flash_attention_f32_f16_wmma kernel once per stream, with each
// binding offset by that stream's nb[3]; within a stream the layout is exactly the single-stream one the
// common.flash_attention_f32_f16_wmma matcher accepts. V stays row-major (no transposed copy) and no sinks are
// taken: the smallest claim that covers the multi-stream node.

#include "dispatch-flash-attention-streams.h"

#include "dispatch-query-chunks.h"

#include "ggml.h"
#include "graph/graph-matcher.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kFlashAttentionStreamsKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_flash_attention_f32_f16_wmma");

// The same bounds the single-stream prompt matcher applies to one stream (dispatch-flash-attention.cpp):
// Q/K head size a multiple of 16 in [16, 576], V head size a multiple of 64 in [64, 576], 1..64 heads,
// at most 2048 query rows per launch (longer streams run as query chunks).
static constexpr int64_t kQkHeadSizeBlock     = 16;
static constexpr int64_t kValueHeadSizeBlock  = 64;
static constexpr int64_t kMaxHeadSize         = 576;
static constexpr int64_t kMaxHeads            = 64;
static constexpr int64_t kMaxQueryTokens      = 2048;
static constexpr int64_t kMaxKeyValueTokens   = 262144;
// llama.cpp's LLAMA_MAX_SEQ is 256; a stream count past that is not a graph llama builds.
static constexpr int64_t kMaxStreams          = 256;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static std::string config_value(int64_t value) {
    return std::to_string(value);
}

static std::string config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

struct StreamsMatch {
    const Value * query  = nullptr;
    const Value * key    = nullptr;
    const Value * value  = nullptr;
    const Value * mask   = nullptr;
    const Value * output = nullptr;
    int64_t       stream_count          = 0;
    int64_t       query_token_count     = 0;  // per stream
    int64_t       key_value_token_count = 0;  // per stream (the cache view's capacity)
    int64_t       query_head_count      = 0;
    int64_t       key_value_head_count  = 0;
    int64_t       qk_head_size          = 0;
    int64_t       value_head_size       = 0;
    float         attention_scale       = 0.0f;
};

static bool match_streams(const Graph & graph, const GraphNode * node, StreamsMatch & match) {
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4) {
        return false;
    }
    const Value * q = graph_value(graph, node->inputs[0]);
    const Value * k = graph_value(graph, node->inputs[1]);
    const Value * v = graph_value(graph, node->inputs[2]);
    const Value * m = graph_value(graph, node->inputs[3]);
    const Value * o = graph_value(graph, node->output);
    if (q == nullptr || k == nullptr || v == nullptr || m == nullptr || o == nullptr) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        m->type != GGML_TYPE_F16 || o->type != GGML_TYPE_F32) {
        return false;
    }
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node->params);
    if (params == nullptr || std::fabs(params->max_bias) > 1.0e-6f || std::fabs(params->logit_softcap) > 1.0e-6f ||
        (params->prec != GGML_PREC_DEFAULT && params->prec != GGML_PREC_F32)) {
        return false;
    }

    // Only the multi-stream shape: ne[3] == 1 is the single-stream matchers' node.
    const int64_t S = q->ne[3];
    if (S < 2 || S > kMaxStreams || k->ne[3] != S || v->ne[3] != S || m->ne[3] != S || o->ne[3] != S ||
        m->ne[2] != 1) {
        return false;
    }

    const int64_t D   = q->ne[0];
    const int64_t Dv  = v->ne[0];
    const int64_t T   = q->ne[1];
    const int64_t H   = q->ne[2];
    const int64_t N   = k->ne[1];
    const int64_t Hkv = k->ne[2];
    if (D < kQkHeadSizeBlock || D > kMaxHeadSize || D % kQkHeadSizeBlock != 0 || Dv < kValueHeadSizeBlock ||
        Dv > kMaxHeadSize || Dv % kValueHeadSizeBlock != 0 || k->ne[0] != D || o->ne[0] != Dv) {
        return false;
    }
    if (T < 1 || N < T || N > kMaxKeyValueTokens || H < 1 || H > kMaxHeads || Hkv < 1 || Hkv > kMaxHeads ||
        H % Hkv != 0) {
        return false;
    }
    if (v->ne[1] != N || v->ne[2] != Hkv || m->ne[0] != N || m->ne[1] != T || o->ne[1] != H || o->ne[2] != T) {
        return false;
    }

    // Per stream, the single-stream layout: q rows of H*D floats with heads D apart, K/V rows of Hkv*D halves
    // with heads D apart (V with its own head size), mask rows of N halves, output rows of Dv floats with
    // tokens H*Dv apart. Each stream then starts nb[3] after the previous one; the planes must not overlap.
    const size_t f32 = sizeof(float);
    const size_t f16 = sizeof(ggml_fp16_t);
    if (q->nb[0] != f32 || q->nb[1] != static_cast<size_t>(H * D) * f32 || q->nb[2] != static_cast<size_t>(D) * f32 ||
        q->nb[3] < static_cast<size_t>(T) * q->nb[1]) {
        return false;
    }
    if (k->nb[0] != f16 || k->nb[1] != static_cast<size_t>(Hkv * D) * f16 || k->nb[2] != static_cast<size_t>(D) * f16 ||
        k->nb[3] < static_cast<size_t>(N) * k->nb[1]) {
        return false;
    }
    if (v->nb[0] != f16 || v->nb[1] != static_cast<size_t>(Hkv * Dv) * f16 ||
        v->nb[2] != static_cast<size_t>(Dv) * f16 || v->nb[3] < static_cast<size_t>(N) * v->nb[1]) {
        return false;
    }
    if (m->nb[0] != f16 || m->nb[1] != static_cast<size_t>(N) * f16 || m->nb[3] < static_cast<size_t>(T) * m->nb[1]) {
        return false;
    }
    if (o->nb[0] != f32 || o->nb[1] != static_cast<size_t>(Dv) * f32 ||
        o->nb[2] != static_cast<size_t>(H * Dv) * f32 || o->nb[3] < static_cast<size_t>(T) * o->nb[2]) {
        return false;
    }
    // Every per-stream slice has to lie inside the value's own bytes (the bindings below are offset + length).
    const size_t last = static_cast<size_t>(S - 1);
    if (last * q->nb[3] + static_cast<size_t>(T) * q->nb[1] > q->byte_count ||
        last * k->nb[3] + static_cast<size_t>(N) * k->nb[1] > k->byte_count ||
        last * v->nb[3] + static_cast<size_t>(N) * v->nb[1] > v->byte_count ||
        last * m->nb[3] + static_cast<size_t>(T) * m->nb[1] > m->byte_count ||
        last * o->nb[3] + static_cast<size_t>(T) * o->nb[2] > o->byte_count) {
        return false;
    }

    match.query                 = q;
    match.key                   = k;
    match.value                 = v;
    match.mask                  = m;
    match.output                = o;
    match.stream_count          = S;
    match.query_token_count     = T;
    match.key_value_token_count = N;
    match.query_head_count      = H;
    match.key_value_head_count  = Hkv;
    match.qk_head_size          = D;
    match.value_head_size       = Dv;
    match.attention_scale       = params->scale;
    return true;
}

static bool match_flash_attention_streams_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match) {
    StreamsMatch match;
    if (!match_streams(context.graph, context.root_node, match)) {
        return false;
    }

    const size_t q_bytes = static_cast<size_t>(match.query_token_count) * match.query->nb[1];
    const size_t k_bytes = static_cast<size_t>(match.key_value_token_count) * match.key->nb[1];
    const size_t v_bytes = static_cast<size_t>(match.key_value_token_count) * match.value->nb[1];
    const size_t m_bytes = static_cast<size_t>(match.query_token_count) * match.mask->nb[1];
    const size_t o_bytes = static_cast<size_t>(match.query_token_count) * match.output->nb[2];

    for (int64_t s = 0; s < match.stream_count; ++s) {
        const size_t stream = static_cast<size_t>(s);
        Dispatch     dispatch;
        dispatch.kernel = make_kernel_specialization(kFlashAttentionStreamsKernel);
        dispatch.kernel.integer_parameters.emplace("query_token_count", match.query_token_count);
        dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
        auto & cp = dispatch.kernel.compile_parameters;
        cp.emplace("ggml.flash_attention.query_head_count", config_value(match.query_head_count));
        cp.emplace("ggml.flash_attention.key_value_head_count", config_value(match.key_value_head_count));
        cp.emplace("ggml.flash_attention.qk_head_size", config_value(match.qk_head_size));
        cp.emplace("ggml.flash_attention.value_head_size", config_value(match.value_head_size));
        cp.emplace("ggml.flash_attention.value_stride",
                   config_value(static_cast<int64_t>(match.value->nb[1] / sizeof(ggml_fp16_t))));
        cp.emplace("ggml.flash_attention.attention_scale", config_value(match.attention_scale));
        cp.emplace("ggml.flash_attention.apply_gate", "0");
        cp.emplace("ggml.flash_attention.gate_stride_head", "1");
        cp.emplace("ggml.flash_attention.gate_stride_token", "1");
        // binding order of the kernel: query, key, value, mask, gate (unused: the query again), output
        dispatch.bindings.push_back({ match.query->id, stream * match.query->nb[3], q_bytes });
        dispatch.bindings.push_back({ match.key->id, stream * match.key->nb[3], k_bytes });
        dispatch.bindings.push_back({ match.value->id, stream * match.value->nb[3], v_bytes });
        dispatch.bindings.push_back({ match.mask->id, stream * match.mask->nb[3], m_bytes });
        dispatch.bindings.push_back({ match.query->id, stream * match.query->nb[3], q_bytes });
        dispatch.bindings.push_back({ match.output->id, stream * match.output->nb[3], o_bytes });
        if (!append_query_chunked_dispatches(dispatch, match.query_token_count, query_chunk_tokens(kMaxQueryTokens),
                                             { { 0, match.query->nb[1] },
                                               { 3, match.mask->nb[1] },
                                               { 4, match.query->nb[1] },
                                               { 5, match.output->nb[2] } },
                                             "query_token_count", dispatch_match)) {
            return false;
        }
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    const GraphNode * layout = find_single_layout_alias_consumer(context.graph, match.output->id);
    if (layout != nullptr &&
        !append_covered_node_index_once(context.graph, context.covered_nodes, layout, dispatch_match.covered_nodes)) {
        return false;
    }
    return true;
}

}  // namespace

void register_flash_attention_streams_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "onebit.flash_attention_streams_f32_f16_wmma",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        45,
        DispatchSource::Common,
        match_flash_attention_streams_dispatch,
    });
}

}  // namespace ggml::hrx
