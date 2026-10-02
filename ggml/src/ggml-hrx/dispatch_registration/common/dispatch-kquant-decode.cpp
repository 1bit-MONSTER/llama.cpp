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

// Decode projections (1 token, or 2-8 for MTP / speculative verify batches) on Q2_K, Q3_K, Q4_K, Q5_K,
// Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and Q8_0 weights, read in their GGUF block layout
// (and exact-ternary Q4_0 repacked to 2 bits, see kquant_ternary)
// (ops/kquant_decode_f32.loom): FFN gate/up pairs fused with SwiGLU, and plain projections with an
// optional following residual ADD. Mixed-quant models (Unsloth UD-Q4_K_XL and similar) pair these
// types freely per layer; without this they take the generic dequantize-4-values-at-a-time
// kernels.

#include "dispatch-kquant-decode.h"

#include "dispatch-mul-mat-common.h"
#include "dispatch/ternary-q4-0.h"
#include "graph/graph-matcher.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

namespace {

static constexpr KernelCatalogRef kKQuantSwiGLUDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_swiglu_decode_f32");
static constexpr KernelCatalogRef kKQuantMulMatDecodeKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_mul_mat_decode_f32");
// 2..8 tokens (MTP / speculative verify batches): weights dequantized once per lane for all tokens.
static constexpr KernelCatalogRef kKQuantSwiGLUDecodeTokensKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_swiglu_decode_tokens_f32");
static constexpr KernelCatalogRef kKQuantMulMatDecodeTokensKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_mul_mat_decode_tokens_f32");
// The same 2..8 token kernels reading a Q8_1 x4 copy of the input for Q4_K, Q5_K, Q6_K, IQ4_NL,
// IQ4_XS and Q8_0 weights (dot4i per word and token instead of an f32 FMA per weight and token).
static constexpr KernelCatalogRef kKQuantSwiGLUDecodeTokensQ8Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_swiglu_decode_tokens_q8_1");
static constexpr KernelCatalogRef kKQuantMulMatDecodeTokensQ8Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_kquant_mul_mat_decode_tokens_q8_1");

bool kquant_format(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::Q2K:
        case CommonMulMatWeightFormat::IQ1_S:
        case CommonMulMatWeightFormat::IQ1_M:
        case CommonMulMatWeightFormat::IQ2_XXS:
        case CommonMulMatWeightFormat::IQ2_XS:
        case CommonMulMatWeightFormat::IQ3_XXS:
        case CommonMulMatWeightFormat::IQ2_S:
        case CommonMulMatWeightFormat::Q3K:
        case CommonMulMatWeightFormat::Q4K:
        case CommonMulMatWeightFormat::Q5K:
        case CommonMulMatWeightFormat::Q6K:
        case CommonMulMatWeightFormat::IQ3_S:
        case CommonMulMatWeightFormat::IQ4_NL:
        case CommonMulMatWeightFormat::IQ4_XS:
        case CommonMulMatWeightFormat::Q8_0:
            return true;
        default:
            return false;
    }
}

// Q4_0 joins as packed ternary (format 90, dispatch/ternary-q4-0.h) when GGML_HRX_TERNARY_Q4_0 is set: the
// weight binding asks for the repacked layout, which the upload verifies value by value.
bool kquant_ternary(const CommonMulMatMatch & match) {
    return match.weight_format == CommonMulMatWeightFormat::Q4_0 && ternary_q4_0_enabled() &&
           match.input_size % 256 == 0;
}

bool kquant_supported(const CommonMulMatMatch & match) {
    return kquant_format(match.weight_format) || kquant_ternary(match);
}

// Formats the Q8_1 tokens kernels read (ggml_kquant_lane_q8w in the .loom).
bool kquant_q8_format(const CommonMulMatMatch & match) {
    if (kquant_ternary(match)) {
        return false;
    }
    switch (match.weight_format) {
        case CommonMulMatWeightFormat::Q4K:
        case CommonMulMatWeightFormat::Q5K:
        case CommonMulMatWeightFormat::Q6K:
        case CommonMulMatWeightFormat::IQ4_NL:
        case CommonMulMatWeightFormat::IQ4_XS:
        case CommonMulMatWeightFormat::Q8_0:
            return true;
        default:
            return false;
    }
}

// GGML_HRX_KQUANT_TOKENS_Q8=0 keeps 2..8 token projections on the F32 tokens kernels.
bool kquant_tokens_q8_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_KQUANT_TOKENS_Q8");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// ggml_quantize_q8_1_x4_f32 takes input_size 128..32768.
constexpr int64_t kMaxQ8InputSize = 32768;

bool use_tokens_q8(int64_t token_count, int64_t input_size, bool q8_formats) {
    return kquant_tokens_q8_enabled() && token_count > 1 && q8_formats && input_size <= kMaxQ8InputSize;
}

// The Q8_1 x4 copy of a 2..8 token input, quantized once (ggml_quantize_q8_1_x4_f32) and shared by
// every kquant projection of that input. It is published under its own name and not as a
// GGML_TYPE_Q8_1 alternate, so the other matchers (which pick q8 kernels whenever a Q8_1 copy of
// their input exists) route exactly as before.
constexpr const char * kTokensQ8AlternateName = "kquant.tokens.q8_1_x4";

bool prepare_tokens_q8_input(const DispatchMatchContext & context, const Value & input, int64_t input_size,
                             int64_t token_count, DispatchMatch & match, DispatchBinding & binding) {
    const size_t bytes = static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(context.graph, context.plan, input.id, GGML_TYPE_COUNT, bytes);
    if (alternate != nullptr) {
        if (alternate->name != kTokensQ8AlternateName) {
            return false;
        }
        binding = { alternate->alternate_value, 0, bytes };
        return true;
    }

    constexpr KernelCatalogRef kernel     = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_quantize_q8_1_x4_f32");
    const ValueId              activation = context.next_plan_value;
    match.transients.push_back({ activation, kTokensQ8AlternateName, bytes, 256 });
    Dispatch quantize;
    quantize.kernel = make_kernel_specialization(kernel);
    quantize.kernel.integer_parameters.emplace("token_count", token_count);
    quantize.kernel.integer_parameters.emplace("input_size", input_size);
    quantize.kernel.compile_parameters.emplace("ggml.quantize_q8_1_x4.group_capacity",
                                                std::to_string(token_count * input_size / 128));
    quantize.bindings.push_back({ input.id, 0, input.byte_count });
    quantize.bindings.push_back({ activation, 0, bytes });
    match.dispatches.push_back(std::move(quantize));

    Status status;
    if (!match.metadata.append_alternate_value({ input.id, activation, GGML_TYPE_COUNT, bytes, kTokensQ8AlternateName },
                                               status)) {
        match.status.append(status);
        return false;
    }
    binding = { activation, 0, bytes };
    return true;
}

int64_t kquant_format_value(const CommonMulMatMatch & match) {
    return kquant_ternary(match) ? kTernaryQ40K128FormatValue : common_mul_mat_format_config_value(match.weight_format);
}

DispatchBinding kquant_weight_binding(const CommonMulMatMatch & match) {
    if (kquant_ternary(match)) {
        return { match.weight->id,
                 0,
                 ternary_q4_0_k128_bytes(match.input_size, match.output_size),
                 kTernaryQ40K128Layout,
                 match.weight->type,
                 match.input_size,
                 match.output_size,
                 match.weight->byte_count };
    }
    return { match.weight->id, 0, match.weight->byte_count };
}

// The codebook a format's lane functions read from workgroup memory (IQ1_S and IQ1_M share one),
// or 0. A gate/up pair shares that one buffer, so its two formats may not need different codebooks.
int kquant_grid(CommonMulMatWeightFormat format) {
    switch (format) {
        case CommonMulMatWeightFormat::IQ3_S:
            return 21;
        case CommonMulMatWeightFormat::IQ2_XXS:
            return 24;
        case CommonMulMatWeightFormat::IQ2_XS:
            return 25;
        case CommonMulMatWeightFormat::IQ2_S:
            return 22;
        case CommonMulMatWeightFormat::IQ3_XXS:
            return 28;
        case CommonMulMatWeightFormat::IQ1_S:
        case CommonMulMatWeightFormat::IQ1_M:
            return 26;
        default:
            return 0;
    }
}

// The kernels' config ranges (ops/kquant_decode_f32.loom).
constexpr int64_t kMaxInputSize  = 65536;
constexpr int64_t kMaxOutputSize = 1048576;
constexpr int64_t kMaxTokens     = 8;

// A MUL_MAT with 1..kMaxTokens tokens: the common matcher's decode form admits exactly one token and
// its prefill form two or more.
CommonMulMatMatch match_few_token_mul_mat(const Graph & graph, const GraphNode * node, KernelCatalogRef kernel) {
    CommonMulMatMatch match = common_match_mul_mat_any_format(graph, node, kernel, true);
    if (!match.matched()) {
        match = common_match_mul_mat_any_format(graph, node, kernel, false);
    }
    if (!match.matched() || match.token_count > kMaxTokens) {
        return {};
    }
    return match;
}

// A value is ready at this root when its producer (looking through layout aliases) is a graph input
// or an already-claimed node. A dispatch is emitted at its root's position, so reading a value that
// a later-rooted fusion will produce fails with "reads transient value before write".
bool value_ready(const DispatchMatchContext & context, ValueId id) {
    const Graph & graph = context.graph;
    for (int depth = 0; depth < 16; ++depth) {
        const GraphNode * producer = graph.index().producer(id);
        if (producer == nullptr) {
            return true;
        }
        size_t index = 0;
        if (!graph.index().node_index(producer, index) || index >= context.covered_nodes.size()) {
            return false;
        }
        if (context.covered_nodes[index]) {
            return true;
        }
        if (!is_layout_alias_node(graph, *producer) || producer->inputs.empty()) {
            return false;
        }
        id = producer->inputs[0];
    }
    return false;
}

// Some fused producers (e.g. qwen3_moe's routed down + next-layer norm) publish the next input only
// as a Q8_1 alternate and never write the F32 value when every consumer reads the alternate.
// These kernels (F32 and the Q8_1 tokens kernels, which quantize their own copy) leave such inputs
// to the q8 consumers.
bool has_q8_alternate(const DispatchMatchContext & context, const Value & input, int64_t input_size,
                      int64_t token_count) {
    const size_t bytes = static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
    return find_alternate_value(context.graph, context.plan, input.id, GGML_TYPE_Q8_1, bytes) != nullptr;
}

bool uncovered(const DispatchMatchContext & context, const GraphNode * node) {
    size_t index = 0;
    return context.graph.index().node_index(node, index) && index < context.covered_nodes.size() &&
           !context.covered_nodes[index];
}

bool match_kquant_swiglu_decode(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const Graph & graph = context.graph;
    if (!graph.has_index()) {
        return false;
    }
    const CommonMulMatMatch root =
        match_few_token_mul_mat(graph, context.root_node, kKQuantSwiGLUDecodeKernel);
    if (!root.matched() || root.token_count < 1 || root.token_count > kMaxTokens || !kquant_supported(root) ||
        root.input_size % 256 != 0 || root.input_size > kMaxInputSize || root.output_size > kMaxOutputSize ||
        !value_ready(context, root.input->id)) {
        return false;
    }

    // The root's only consumer is a SwiGLU whose other operand is a matching MUL_MAT of the same input.
    const std::vector<const GraphNode *> & root_consumers = graph.index().consumers(context.root_node->output);
    const GraphNode * glu = root_consumers.size() == 1 ? root_consumers.front() : nullptr;
    BinaryKind        op;
    if (glu == nullptr || glu->inputs.size() != 2 || !common_fused_binary_kind_from_params(glu->params, op) ||
        op != BinaryKind::SwiGLU || !uncovered(context, glu)) {
        return false;
    }
    const bool root_is_gate = glu->inputs[0] == context.root_node->output;
    if (!root_is_gate && glu->inputs[1] != context.root_node->output) {
        return false;
    }
    const ValueId     peer_id = root_is_gate ? glu->inputs[1] : glu->inputs[0];
    const GraphNode * peer    = graph.index().producer(peer_id);
    if (peer == nullptr || peer == context.root_node || !uncovered(context, peer) ||
        graph.index().consumers(peer_id).size() != 1) {
        return false;
    }
    const CommonMulMatMatch other = match_few_token_mul_mat(graph, peer, kKQuantSwiGLUDecodeKernel);
    if (other.matched() && kquant_grid(other.weight_format) != 0 && kquant_grid(root.weight_format) != 0 &&
        kquant_grid(other.weight_format) != kquant_grid(root.weight_format)) {
        return false;
    }
    if (!other.matched() || !kquant_supported(other) || other.input->id != root.input->id ||
        other.input_size != root.input_size || other.output_size != root.output_size ||
        other.token_count != root.token_count) {
        return false;
    }
    const Value * output = common_graph_value(graph, glu->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !common_same_shape(*output, *root.output)) {
        return false;
    }

    const CommonMulMatMatch & gate = root_is_gate ? root : other;
    const CommonMulMatMatch & up   = root_is_gate ? other : root;
    const bool q8_path =
        use_tokens_q8(root.token_count, root.input_size, kquant_q8_format(gate) && kquant_q8_format(up));
    if (has_q8_alternate(context, *root.input, root.input_size, root.token_count)) {
        return false;
    }
    for (const GraphNode * node : { context.root_node, peer, glu }) {
        if (!append_covered_node_index_once(graph, context.covered_nodes, node, dispatch_match.covered_nodes)) {
            dispatch_match.covered_nodes.clear();
            return false;
        }
    }

    // 2..8 tokens with Q8_1 formats on both sides: quantize the input once (or reuse the copy an
    // earlier kquant projection of the same input made) for the Q8_1 tokens kernel.
    DispatchBinding q8_binding{};
    const bool      q8 = q8_path && prepare_tokens_q8_input(context, *root.input, root.input_size,
                                                            root.token_count, dispatch_match, q8_binding);
    if (!dispatch_match.status.success()) {
        dispatch_match.covered_nodes.clear();
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(root.token_count == 1 ? kKQuantSwiGLUDecodeKernel :
                                                 q8                    ? kKQuantSwiGLUDecodeTokensQ8Kernel :
                                                                         kKQuantSwiGLUDecodeTokensKernel);
    if (root.token_count > 1) {
        dispatch.kernel.compile_parameters.emplace("ggml.kquant_decode.token_count",
                                                   std::to_string(root.token_count));
    }
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_swiglu_decode.input_size",
                                               std::to_string(root.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_swiglu_decode.output_size",
                                               std::to_string(root.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_swiglu_decode.gate_weight_format",
                                               std::to_string(kquant_format_value(gate)));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_swiglu_decode.up_weight_format",
                                               std::to_string(kquant_format_value(up)));
    if (q8) {
        dispatch.bindings.push_back(q8_binding);
    } else {
        dispatch.bindings.push_back({ root.input->id, 0, root.input->byte_count });
    }
    dispatch.bindings.push_back(kquant_weight_binding(gate));
    dispatch.bindings.push_back(kquant_weight_binding(up));
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}


bool match_kquant_mul_mat_decode_impl(const DispatchMatchContext & context, DispatchMatch & dispatch_match,
                                      bool require_add) {
    const Graph & graph = context.graph;
    if (!graph.has_index()) {
        return false;
    }
    const CommonMulMatMatch root =
        match_few_token_mul_mat(graph, context.root_node, kKQuantMulMatDecodeKernel);
    if (!root.matched() || root.token_count < 1 || root.token_count > kMaxTokens || !kquant_supported(root) ||
        root.input_size % 256 != 0 || root.input_size > kMaxInputSize || root.output_size > kMaxOutputSize ||
        !value_ready(context, root.input->id)) {
        return false;
    }
    const bool q8_path = use_tokens_q8(root.token_count, root.input_size, kquant_q8_format(root));
    if (has_q8_alternate(context, *root.input, root.input_size, root.token_count)) {
        return false;
    }

    // Fold a following residual ADD (the projection's only consumer) into the store.
    const Value *     addend = nullptr;
    const Value *     output = root.output;
    const GraphNode * add    = common_find_only_consumer_with_op(graph, root.output->id, GGML_OP_ADD);
    if (add != nullptr && common_binary_node_is_add(*add) && uncovered(context, add)) {
        const bool    root_is_lhs = add->inputs[0] == root.output->id;
        const Value * other       = common_graph_value(graph, root_is_lhs ? add->inputs[1] : add->inputs[0]);
        const Value * sum         = common_graph_value(graph, add->output);
        if ((root_is_lhs || add->inputs[1] == root.output->id) && other != nullptr && sum != nullptr &&
            other->type == GGML_TYPE_F32 && sum->type == GGML_TYPE_F32 && other->contiguous && sum->contiguous &&
            common_same_shape(*root.output, *other) && common_same_shape(*root.output, *sum) &&
            value_ready(context, other->id)) {
            addend = other;
            output = sum;
        } else {
            add = nullptr;
        }
    } else {
        add = nullptr;
    }
    if (require_add && add == nullptr) {
        return false;
    }

    if (!append_covered_node_index_once(graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        (add != nullptr &&
         !append_covered_node_index_once(graph, context.covered_nodes, add, dispatch_match.covered_nodes))) {
        dispatch_match.covered_nodes.clear();
        return false;
    }

    DispatchBinding q8_binding{};
    const bool      q8 = q8_path && prepare_tokens_q8_input(context, *root.input, root.input_size,
                                                            root.token_count, dispatch_match, q8_binding);
    if (!dispatch_match.status.success()) {
        dispatch_match.covered_nodes.clear();
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(root.token_count == 1 ? kKQuantMulMatDecodeKernel :
                                                 q8                    ? kKQuantMulMatDecodeTokensQ8Kernel :
                                                                         kKQuantMulMatDecodeTokensKernel);
    if (root.token_count > 1) {
        dispatch.kernel.compile_parameters.emplace("ggml.kquant_decode.token_count",
                                                   std::to_string(root.token_count));
    }
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.input_size",
                                               std::to_string(root.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.output_size",
                                               std::to_string(root.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.weight_format",
                                               std::to_string(kquant_format_value(root)));
    dispatch.kernel.compile_parameters.emplace("ggml.kquant_mul_mat_decode.add", addend != nullptr ? "1" : "0");
    if (q8) {
        dispatch.bindings.push_back(q8_binding);
    } else {
        dispatch.bindings.push_back({ root.input->id, 0, root.input->byte_count });
    }
    dispatch.bindings.push_back(kquant_weight_binding(root));
    // Without an ADD the addend is never read; bind the (Q8_1 or F32) input in its place.
    if (addend != nullptr) {
        dispatch.bindings.push_back({ addend->id, 0, addend->byte_count });
    } else if (q8) {
        dispatch.bindings.push_back(q8_binding);
    } else {
        dispatch.bindings.push_back({ root.input->id, 0, root.input->byte_count });
    }
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

bool match_kquant_mul_mat_decode(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    return match_kquant_mul_mat_decode_impl(context, dispatch_match, false);
}

bool match_kquant_mul_mat_add_decode(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    return match_kquant_mul_mat_decode_impl(context, dispatch_match, true);
}

}  // namespace

void register_kquant_decode_dispatches(DispatchRegistryBuilder & registry) {
    // Above common.mul_mat_swiglu.symmetric_i4_lowrow_adjacent_dual (310): that path repacks the
    // weights to int4 at first use and quantizes activations to int4, and under llama-server it
    // made 27B decode 10.7 tok/s against 11.9 with this kernel.
    registry.add({
        "kquant.swiglu.decode_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        315,
        DispatchSource::Common,
        match_kquant_swiglu_decode,
    });
    // Projection + residual ADD above common.mul_mat_postops.f32_f32_wmma (180), which otherwise takes
    // the 2-8 token (MTP verify) projections through the prefill WMMA kernel.
    registry.add({
        "kquant.mul_mat_add.decode_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        185,
        DispatchSource::Common,
        match_kquant_mul_mat_add_decode,
    });
    // Plain projections above the generic common.mul_mat f32 matchers (80/70/60), below every
    // specialized one.
    registry.add({
        "kquant.mul_mat.decode_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        85,
        DispatchSource::Common,
        match_kquant_mul_mat_decode,
    });
}

}  // namespace ggml::hrx
