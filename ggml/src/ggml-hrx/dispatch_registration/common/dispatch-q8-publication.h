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

}  // namespace ggml::hrx
