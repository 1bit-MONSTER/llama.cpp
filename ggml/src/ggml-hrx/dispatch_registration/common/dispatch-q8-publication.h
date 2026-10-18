// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatch/command-plan.h"
#include "graph/graph.h"

#include <cstdint>

namespace ggml::hrx {

// Consumer qualification for consumer-driven Q8_1 x4 publication (engine #449).
//
// A producer that can publish its normalized activation as Q8_1 x4 may only do so when the value is
// actually consumed by a projection kernel that reads Q8_1 x4. These helpers answer that question on
// the graph alone, so the publication decision is a property of the consumer and not of the producer.
// They are deliberately conservative: an unqualified consumer means "publish nothing", which costs one
// separate quantize dispatch at worst, while a wrong "yes" costs a value nobody reads.

// Decode-shaped dense projection, i.e. the shapes the kquant decode kernels serve for 1-5 tokens.
inline bool common_q8_consumer_is_qualified_projection(const Graph &     graph,
                                                       const GraphNode & consumer,
                                                       const Value &     activation) {
    if (consumer.op != GGML_OP_MUL_MAT || consumer.inputs.size() != 2 || consumer.inputs[1] != activation.id) {
        return false;
    }
    const Value * weight = graph.values().find(consumer.inputs[0]);
    const Value * output = graph.values().find(consumer.output);
    if (weight == nullptr || output == nullptr || !weight->contiguous || !output->contiguous ||
        weight->alias_source.value >= 0 || output->type != GGML_TYPE_F32) {
        return false;
    }
    if (activation.ne[0] < 256 || activation.ne[0] > 32768 || activation.ne[0] % 256 != 0 ||
        activation.ne[2] != 1 || activation.ne[3] != 1) {
        return false;
    }
    if (weight->ne[0] != activation.ne[0] || weight->ne[1] < 64 || weight->ne[1] > 262144 ||
        weight->ne[1] % 64 != 0 || weight->ne[2] != 1 || weight->ne[3] != 1) {
        return false;
    }
    if (output->ne[0] != weight->ne[1] || output->ne[1] != activation.ne[1] || output->ne[2] != 1 ||
        output->ne[3] != 1) {
        return false;
    }
    const bool decode = activation.ne[1] >= 1 && activation.ne[1] <= 5 &&
                        (weight->type == GGML_TYPE_Q4_K ||
                         (weight->type == GGML_TYPE_Q6_K && !graph.index().consumers(output->id).empty()));
    const bool prefill = (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_IQ4_XS) &&
                         activation.ne[1] >= 256 && activation.ne[1] <= 2048 && activation.ne[1] % 256 == 0;
    return decode || prefill;
}

// True when `value` feeds a qualified q8 projection, either directly or through ordered layout
// aliases that cover the whole value (a reshape of the normalized row does not disqualify it).
inline bool common_has_qualified_q8_projection_consumer(const Graph & graph, const Value & value) {
    if (!graph.has_index() || value.type != GGML_TYPE_F32 || !value.contiguous) {
        return false;
    }
    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        const Value * activation = &value;
        while (consumer != nullptr && is_layout_alias_node(graph, *consumer) && consumer->inputs.size() == 1 &&
               consumer->inputs[0] == activation->id) {
            const Value * aliased = graph.values().find(consumer->output);
            if (aliased == nullptr || aliased->type != GGML_TYPE_F32 || !aliased->contiguous ||
                !same_full_value_range(*aliased, *activation)) {
                activation = nullptr;
                break;
            }
            const auto & alias_consumers = graph.index().consumers(aliased->id);
            consumer = alias_consumers.size() == 1 ? alias_consumers.front() : nullptr;
            activation = aliased;
        }
        if (consumer != nullptr && activation != nullptr &&
            common_q8_consumer_is_qualified_projection(graph, *consumer, *activation)) {
            return true;
        }
    }
    return false;
}

// Routed-MoE qualification. A routed gate/up projection can consume the q8 hidden only when the whole
// routed FFN around it is the fused shape: the same route ids feed the down projection through the
// GLU, with the profile's expert geometry. A different route tensor (or a different route count) means
// the chain will not fuse, so publishing q8 there would quantize a value nobody reads.
struct CommonQ8RoutedProfile {
    int64_t expert_hidden_size = 0;
    int64_t expert_count       = 0;
    int64_t route_count        = 0;
};

inline bool common_q8_consumer_is_qualified_routed_projection(const Graph &                  graph,
                                                              const GraphNode &              consumer,
                                                              const Value &                  activation,
                                                              const CommonQ8RoutedProfile &  profile) {
    if (consumer.op != GGML_OP_MUL_MAT_ID || consumer.inputs.size() != 3 || consumer.inputs[1] != activation.id ||
        profile.expert_hidden_size <= 0 || profile.expert_count <= 0 || profile.route_count <= 0) {
        return false;
    }
    const Value * gate_weight = graph.values().find(consumer.inputs[0]);
    const Value * route_ids   = graph.values().find(consumer.inputs[2]);
    const Value * gate_output = graph.values().find(consumer.output);
    if (gate_weight == nullptr || route_ids == nullptr || gate_output == nullptr || !graph.has_index() ||
        route_ids->type != GGML_TYPE_I32 || route_ids->ne[0] != profile.route_count ||
        route_ids->ne[1] != activation.ne[1] || route_ids->ne[2] != 1 || route_ids->ne[3] != 1) {
        return false;
    }
    if ((gate_weight->type != GGML_TYPE_Q4_K && gate_weight->type != GGML_TYPE_Q6_K) ||
        gate_weight->ne[0] != activation.ne[0] || gate_weight->ne[1] != profile.expert_hidden_size ||
        gate_weight->ne[2] != profile.expert_count || gate_weight->ne[3] != 1) {
        return false;
    }
    for (const GraphNode * glu : graph.index().consumers(gate_output->id)) {
        if (glu == nullptr || glu->op != GGML_OP_GLU || glu->inputs.size() != 2 ||
            (glu->inputs[0] != gate_output->id && glu->inputs[1] != gate_output->id)) {
            continue;
        }
        const Value * glu_output = graph.values().find(glu->output);
        if (glu_output == nullptr || glu_output->type != GGML_TYPE_F32) {
            continue;
        }
        for (const GraphNode * down : graph.index().consumers(glu_output->id)) {
            if (down == nullptr || down->op != GGML_OP_MUL_MAT_ID || down->inputs.size() != 3 ||
                down->inputs[1] != glu_output->id || down->inputs[2] != route_ids->id) {
                continue;
            }
            const Value * down_weight = graph.values().find(down->inputs[0]);
            if (down_weight != nullptr && down_weight->ne[0] == profile.expert_hidden_size &&
                down_weight->ne[2] == profile.expert_count) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace ggml::hrx
