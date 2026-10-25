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

#include "dispatch-fusion-order.h"

#include "kernel-corpus/kernel-corpus.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace ggml::hrx {
namespace {

static bool env_flag(const char * name, bool fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

static bool guard_enabled() {
    static const bool enabled = env_flag("GGML_HRX_FUSION_ORDER_GUARD", true);
    return enabled;
}

static bool log_enabled() {
    static const bool enabled = env_flag("GGML_HRX_LOG_FUSION_ORDER", false);
    return enabled;
}

// The binding ABI (names and access) of a kernel. Access does not depend on the target, so the first catalogued
// variant is enough; HIP kernels resolve through the same call.
static const KernelDefinition * kernel_definition(uint64_t kernel_id) {
    thread_local std::unordered_map<uint64_t, const KernelDefinition *> cache;
    const auto found = cache.find(kernel_id);
    if (found != cache.end()) {
        return found->second;
    }
    const KernelResolveResult resolved = resolve_kernel_definition(get_qwen_kernel_corpus(), "", kernel_id);
    if (resolved.definition != nullptr) {
        cache.emplace(kernel_id, resolved.definition);
    }
    return resolved.definition;
}

enum class BindingUse {
    Unknown,
    Read,
    Write,
    ReadWrite,
};

static BindingUse binding_use(const KernelDefinition * definition, size_t index) {
    if (definition == nullptr || index >= definition->bindings.size()) {
        return BindingUse::Unknown;
    }
    switch (definition->bindings[index].access) {
        case ResourceAccess::Read:
            return BindingUse::Read;
        case ResourceAccess::Write:
            return BindingUse::Write;
        case ResourceAccess::ReadWrite:
            return BindingUse::ReadWrite;
    }
    return BindingUse::Unknown;
}

static bool binding_may_write(BindingUse use) {
    return use != BindingUse::Read;
}

// The node whose dispatch produces the bytes of `value`: views, reshapes, permutes and transposes are followed back
// to their source. nullptr for leaves (graph inputs, weights, caches).
static const GraphNode * data_producer(const Graph & graph, ValueId value) {
    for (size_t step = 0; step < 256; ++step) {
        const GraphNode * producer = graph.index().producer(value);
        if (producer == nullptr || producer->op == GGML_OP_NONE) {
            return nullptr;
        }
        if (!is_layout_alias_op(producer->op) || producer->inputs.empty()) {
            return producer;
        }
        value = producer->inputs[0];
    }
    return nullptr;
}

// Storage rooted at a graph input (its root value has no producing node) is bound from outside and therefore
// defined before the first command.
static bool storage_is_graph_input(const Graph & graph, const Value & value) {
    const Value * root = graph.values().find(value.storage_root);
    return root == nullptr || data_producer(graph, root->id) == nullptr;
}

// The command program binds a value as a transient exactly when the value and its storage root are transient
// (command_binding_origin). Only those reads are checked for a prior write, as in verify_command_program: a
// graph-visible buffer can legitimately be bound as a placeholder the kernel never reads (hip.pad_f32 binds its
// output as the source of an empty pad).
static bool storage_is_transient(const Graph & graph, const Value & value) {
    if (value.kind != ValueKind::Transient) {
        return false;
    }
    const Value * root = graph.values().find(value.storage_root);
    return root == nullptr || root->kind == ValueKind::Transient;
}

static void add_written_storages(const Graph & graph, const Dispatch & dispatch, std::unordered_set<int32_t> & storages) {
    const KernelDefinition * definition = kernel_definition(dispatch.kernel.kernel_id);
    for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
        if (!binding_may_write(binding_use(definition, i))) {
            continue;
        }
        const Value * value = graph.values().find(dispatch.bindings[i].value);
        if (value != nullptr) {
            storages.insert(value->storage.value);
        }
    }
}

// Storages written by the dispatches already in the plan. The scheduler only appends to its plan and asks the
// registry at every node, so the set is extended on every guard call and rebuilt whenever the plan, its storage
// or the graph changes.
class PlanWrites {
  public:
    const std::unordered_set<int32_t> & refresh(const Graph & graph, const CommandPlan & plan) {
        if (graph_ != &graph || plan_ != &plan || dispatch_data_ != plan.dispatches.data() ||
            initialization_data_ != plan.initialization_dispatches.data() || dispatches_ > plan.dispatches.size() ||
            initializations_ > plan.initialization_dispatches.size() ||
            constants_ > plan.constant_initializations.size() ||
            (dispatches_ > 0 && !same_fingerprint(plan.dispatches[dispatches_ - 1]))) {
            graph_               = &graph;
            plan_                = &plan;
            dispatch_data_       = plan.dispatches.data();
            initialization_data_ = plan.initialization_dispatches.data();
            dispatches_          = 0;
            initializations_     = 0;
            constants_           = 0;
            storages_.clear();
        }
        for (; initializations_ < plan.initialization_dispatches.size(); ++initializations_) {
            add_written_storages(graph, plan.initialization_dispatches[initializations_], storages_);
        }
        for (; dispatches_ < plan.dispatches.size(); ++dispatches_) {
            add_written_storages(graph, plan.dispatches[dispatches_], storages_);
        }
        if (dispatches_ > 0) {
            const Dispatch & last = plan.dispatches[dispatches_ - 1];
            last_kernel_id_       = last.kernel.kernel_id;
            last_binding_count_   = last.bindings.size();
            last_first_value_     = last.bindings.empty() ? -1 : last.bindings.front().value.value;
        }
        for (; constants_ < plan.constant_initializations.size(); ++constants_) {
            const Value * value = graph.values().find(plan.constant_initializations[constants_].value);
            if (value != nullptr) {
                storages_.insert(value->storage.value);
            }
        }
        return storages_;
    }

  private:
    // A plan object at a reused address could otherwise be mistaken for the one already scanned.
    bool same_fingerprint(const Dispatch & last) const {
        return last.kernel.kernel_id == last_kernel_id_ && last.bindings.size() == last_binding_count_ &&
               (last.bindings.empty() ? -1 : last.bindings.front().value.value) == last_first_value_;
    }

    uint64_t                    last_kernel_id_      = 0;
    size_t                      last_binding_count_  = 0;
    int32_t                     last_first_value_    = -1;
    const Graph *               graph_               = nullptr;
    const CommandPlan *         plan_                = nullptr;
    const Dispatch *            dispatch_data_       = nullptr;
    const Dispatch *            initialization_data_ = nullptr;
    size_t                      dispatches_          = 0;
    size_t                      initializations_     = 0;
    size_t                      constants_           = 0;
    std::unordered_set<int32_t> storages_;
};

static bool refuse(const DispatchMatchContext & context,
                   DispatchMatch &              match,
                   const char *                 registration_name,
                   const Dispatch &             dispatch,
                   size_t                       dispatch_index,
                   size_t                       binding_index,
                   const Value &                value,
                   const GraphNode &            producer,
                   const char *                 reason) {
    size_t producer_index = 0;
    context.graph.index().node_index(&producer, producer_index);
    const KernelDefinition * definition = kernel_definition(dispatch.kernel.kernel_id);
    const std::string        kernel     = definition != nullptr ? kernel_definition_name(*definition) : std::string("?");
    const char *             binding    = definition != nullptr && binding_index < definition->bindings.size() &&
                                       definition->bindings[binding_index].name != nullptr ?
                                              definition->bindings[binding_index].name :
                                              "?";
    match.status.log(
        "fusion order: %s at node %zu (%s) refused: dispatch %zu %s binding %zu %s reads value %d produced by node %zu "
        "(%s) %s",
        registration_name != nullptr ? registration_name : "", context.root_index, ggml_op_name(context.root_node->op),
        dispatch_index, kernel.c_str(), binding_index, binding, value.id.value, producer_index,
        ggml_op_name(producer.op), reason);
    if (log_enabled()) {
        std::fprintf(stderr, "ggml_hrx %s\n", match.status.errors().back().c_str());
    }
    return false;
}

}  // namespace

bool fusion_order_accepts(const DispatchMatchContext & context, DispatchMatch & match, const char * registration_name) {
    if (!guard_enabled() || context.root_node == nullptr || !context.graph.has_index()) {
        return true;
    }
    const Graph & graph = context.graph;

    const std::unordered_set<size_t> internal(match.covered_nodes.begin(), match.covered_nodes.end());
    std::unordered_set<int32_t>      written_by_match;
    for (const Dispatch & dispatch : match.initialization_dispatches) {
        add_written_storages(graph, dispatch, written_by_match);
    }
    for (const CommandPlanConstantInitialization & initialization : match.constant_initializations) {
        const Value * value = graph.values().find(initialization.value);
        if (value != nullptr) {
            written_by_match.insert(value->storage.value);
        }
    }

    thread_local PlanWrites             plan_writes;
    const std::unordered_set<int32_t> & written_by_plan = plan_writes.refresh(graph, context.plan);

    for (size_t d = 0; d < match.dispatches.size(); ++d) {
        const Dispatch &         dispatch   = match.dispatches[d];
        const KernelDefinition * definition = kernel_definition(dispatch.kernel.kernel_id);
        for (size_t b = 0; b < dispatch.bindings.size(); ++b) {
            const BindingUse use = binding_use(definition, b);
            if (use == BindingUse::Write) {
                continue;
            }
            const Value * value = graph.values().find(dispatch.bindings[b].value);
            if (value == nullptr || value->byte_count == 0) {
                continue;  // plan transient, completion counter or empty value
            }
            const GraphNode * producer       = data_producer(graph, value->id);
            size_t            producer_index = 0;
            if (producer == nullptr || !graph.index().node_index(producer, producer_index)) {
                continue;
            }
            if (internal.count(producer_index) != 0) {
                if (use == BindingUse::Read && storage_is_transient(graph, *value) &&
                    written_by_match.count(value->storage.value) == 0 && !storage_is_graph_input(graph, *value)) {
                    return refuse(context, match, registration_name, dispatch, d, b, *value, *producer,
                                  "inside the programme before any of its dispatches writes it");
                }
                continue;
            }
            if (producer_index >= context.covered_nodes.size() || !context.covered_nodes[producer_index]) {
                return refuse(context, match, registration_name, dispatch, d, b, *value, *producer,
                              "that is not scheduled yet");
            }
            if (use == BindingUse::Read && storage_is_transient(graph, *value) && !storage_is_graph_input(graph, *value)) {
                if (written_by_plan.count(value->storage.value) == 0 &&
                    written_by_match.count(value->storage.value) == 0) {
                    return refuse(context, match, registration_name, dispatch, d, b, *value, *producer,
                                  "whose programme did not write it");
                }
            }
        }
        add_written_storages(graph, dispatch, written_by_match);
    }
    return true;
}

bool every_consumer_is_mul_mat_id_with_weight(const Graph & graph, ValueId value,
                                              bool (*weight_qualifies)(const Value & weight)) {
    if (!graph.has_index() || weight_qualifies == nullptr) {
        return false;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    if (consumers.empty()) {
        return false;
    }
    for (const GraphNode * consumer : consumers) {
        if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT_ID || consumer->inputs.size() != 3 ||
            consumer->inputs[1] != value) {
            return false;
        }
        const Value * weight = graph.values().find(consumer->inputs[0]);
        if (weight == nullptr || !weight_qualifies(*weight)) {
            return false;
        }
    }
    return true;
}

}  // namespace ggml::hrx
