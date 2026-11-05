#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include "dispatch-registry.h"
#include "dispatch-fusion-order.h"

#include "common/dispatch-common.h"
#include "llm/dispatch-attention-qkv.h"
#include "llm/dispatch-gated-delta-net.h"
#include "llm/dispatch-ssm-conv.h"
#include "qwen/dispatch-qwen.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr size_t kNoNode = static_cast<size_t>(-1);

static constexpr size_t kOpCount = static_cast<size_t>(GGML_OP_COUNT);

static const std::vector<DispatchRegistration> kEmptyRegistrations;

static bool valid_root_op(ggml_op op) {
    return op >= 0 && static_cast<size_t>(op) < kOpCount;
}

static void sort_registrations(std::vector<DispatchRegistration> & registrations) {
    std::stable_sort(
        registrations.begin(), registrations.end(),
        [](const DispatchRegistration & lhs, const DispatchRegistration & rhs) { return lhs.priority > rhs.priority; });
}

static bool qwen_dispatch_disabled_from_environment() {
    const char * value = std::getenv("GGML_HRX_DISABLE_QWEN_DISPATCH");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

static DispatchRegistry build_registry(bool include_qwen) {
    DispatchRegistryBuilder builder;
    register_common_dispatches(builder);
    register_llm_attention_qkv_dispatches(builder);
    register_llm_gated_delta_net_dispatch(builder);
    register_llm_ssm_conv_dispatch(builder);
    if (include_qwen) {
        register_qwen_dispatches(builder);
    }
    return builder.build();
}

}  // namespace

// Debug switches. GGML_HRX_DISABLE_DISPATCH: comma-separated substrings of registration names to
// skip ("fused" skips every fused registration). GGML_HRX_LOG_DISPATCH=1: print each
// registration name the first time it matches.
static bool dispatch_disabled(const DispatchRegistration & registration) {
    static const std::vector<std::string> patterns = [] {
        std::vector<std::string> out;
        const char * env = std::getenv("GGML_HRX_DISABLE_DISPATCH");
        std::string  list = env != nullptr ? env : "";
        size_t       start = 0;
        while (start <= list.size()) {
            size_t end = list.find(',', start);
            if (end == std::string::npos) {
                end = list.size();
            }
            if (end > start) {
                out.push_back(list.substr(start, end - start));
            }
            start = end + 1;
        }
        return out;
    }();
    const std::string name = registration.name != nullptr ? registration.name : "";
    for (const std::string & p : patterns) {
        if ((p == "fused" && registration.kind == DispatchMatchKind::Fused) || name.find(p) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// A tensor that something outside the graph will read back after compute must still exist then. A fused match
// that *covers* such a node (it is not the match's root) keeps the only copy of that value inside the fused
// program, so the read-back sees the untouched buffer - measured on the HRX route as all-zero pre-RoPE q/k
// tensors from the zha dump (engine #1078). Declining the match falls through to a narrower registration that
// writes the node itself.
//
// The marker is the explicit name prefix llama_zha_dump applies (src/llama-zha-dump.cpp, "zha-readback."), not
// GGML_TENSOR_FLAG_OUTPUT on its own: llama.cpp marks several tensors as outputs for API reasons (result_norm is
// t_embd), and honouring the bare flag declined the final norm fusion in every ordinary run. Nothing else sets
// this prefix, so a run without the dump is planned exactly as before. The root is exempt as well - a match
// always writes its root's output.
static bool tensor_is_readback_request(const ggml_tensor * tensor) {
    constexpr const char * k_readback_prefix = "zha-readback.";
    constexpr size_t       k_prefix_len      = sizeof("zha-readback.") - 1;
    return tensor != nullptr && (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
           std::strncmp(tensor->name, k_readback_prefix, k_prefix_len) == 0;
}

static size_t first_covered_requested_output(const DispatchMatchContext & context, const DispatchMatch & match) {
    const std::vector<GraphNode> & nodes = context.graph.nodes();
    for (const size_t node_index : match.covered_nodes) {
        if (node_index == context.root_index || node_index >= nodes.size()) {
            continue;
        }
        const Value * value = context.graph.values().find(nodes[node_index].output);
        if (value != nullptr && tensor_is_readback_request(value->tensor)) {
            return node_index;
        }
    }
    return kNoNode;
}

static void dispatch_log_requested_output(const DispatchRegistration & registration, const Graph & graph,
                                          size_t node_index) {
    static const bool enabled = std::getenv("GGML_HRX_LOG_DISPATCH") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex  mutex;
    static long long   logged = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (logged >= 8) {
        if (logged++ == 8) {
            std::fprintf(stderr, "ggml_hrx dispatch: further declined registrations suppressed\n");
        }
        return;
    }
    logged++;
    const std::vector<GraphNode> & nodes = graph.nodes();
    const char * node_op   = node_index < nodes.size() ? ggml_op_name(nodes[node_index].op) : "?";
    const Value * value    = node_index < nodes.size() ? graph.values().find(nodes[node_index].output) : nullptr;
    const char * tensor_nm = (value != nullptr && value->tensor != nullptr) ? value->tensor->name : "?";
    std::fprintf(stderr, "ggml_hrx dispatch: %s declined at node %zu (%s) - it covers tensor %s, marked as a graph output\n",
                 registration.name != nullptr ? registration.name : "", node_index, node_op, tensor_nm);
}

static void dispatch_log_match(const DispatchRegistration & registration) {
    static const bool enabled = std::getenv("GGML_HRX_LOG_DISPATCH") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex            mutex;
    static std::set<std::string> seen;
    const std::string            name = registration.name != nullptr ? registration.name : "";
    std::lock_guard<std::mutex>  lock(mutex);
    if (seen.insert(name).second) {
        std::fprintf(stderr, "ggml_hrx dispatch: %s (%s)\n", name.c_str(),
                     registration.kind == DispatchMatchKind::Fused ? "fused" : "single");
    }
}

bool DispatchRegistry::match(const DispatchMatchContext & context, DispatchMatch & match) const {
    return this->match(context, match, nullptr);
}

bool DispatchRegistry::match(const DispatchMatchContext & context,
                             DispatchMatch &              match,
                             DispatchMatchDiagnostics *   diagnostics) const {
    if (context.root_node == nullptr || !valid_root_op(context.root_node->op) || registrations_by_root_.empty()) {
        return false;
    }
    if (diagnostics != nullptr) {
        diagnostics->root_op = context.root_node->op;
        diagnostics->attempts.clear();
    }
    const std::vector<DispatchRegistration> & registrations =
        registrations_by_root_[static_cast<size_t>(context.root_node->op)].ordered;
    for (const DispatchRegistration & registration : registrations) {
        if (dispatch_disabled(registration)) {
            continue;
        }
        DispatchMatch candidate;
        const bool    matched = registration.matcher != nullptr && registration.matcher(context, candidate);
        const size_t  covered_output = matched ? first_covered_requested_output(context, candidate) : kNoNode;
        if (matched && covered_output != kNoNode) {
            dispatch_log_requested_output(registration, context.graph, covered_output);
            if (diagnostics != nullptr) {
                diagnostics->attempts.push_back({
                    registration.name != nullptr ? registration.name : "",
                    registration.root_op,
                    registration.kind,
                    registration.priority,
                    registration.source,
                    false,
                    candidate.covered_nodes,
                    candidate.status.errors(),
                });
            }
            match.status.append(candidate.status);
            continue;
        }
        if (matched && fusion_order_accepts(context, candidate, registration.name)) {
            dispatch_log_match(registration);
            if (diagnostics != nullptr) {
                diagnostics->attempts.push_back({
                    registration.name != nullptr ? registration.name : "",
                    registration.root_op,
                    registration.kind,
                    registration.priority,
                    registration.source,
                    true,
                    candidate.covered_nodes,
                    candidate.status.errors(),
                });
            }
            match = std::move(candidate);
            return true;
        }
        if (diagnostics != nullptr) {
            diagnostics->attempts.push_back({
                registration.name != nullptr ? registration.name : "",
                registration.root_op,
                registration.kind,
                registration.priority,
                registration.source,
                false,
                candidate.covered_nodes,
                candidate.status.errors(),
            });
        }
        match.status.append(candidate.status);
    }
    return false;
}

const std::vector<DispatchRegistration> & DispatchRegistry::registrations_for_root(ggml_op root_op) const {
    if (!valid_root_op(root_op) || registrations_by_root_.empty()) {
        return kEmptyRegistrations;
    }
    return registrations_by_root_[static_cast<size_t>(root_op)].ordered;
}

void DispatchRegistryBuilder::add(DispatchRegistration registration) {
    if (!valid_root_op(registration.root_op) || registration.matcher == nullptr) {
        return;
    }
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    DispatchRegistry::RegistrationGroup & group =
        registry_.registrations_by_root_[static_cast<size_t>(registration.root_op)];
    if (registration.kind == DispatchMatchKind::Fused) {
        group.fused.push_back(std::move(registration));
    } else {
        group.single_op.push_back(registration);
        registry_.single_op_registrations_.push_back(std::move(registration));
    }
}

DispatchRegistry DispatchRegistryBuilder::build() {
    if (registry_.registrations_by_root_.empty()) {
        registry_.registrations_by_root_.resize(kOpCount);
    }
    for (DispatchRegistry::RegistrationGroup & group : registry_.registrations_by_root_) {
        sort_registrations(group.fused);
        sort_registrations(group.single_op);
        group.ordered = group.fused;
        group.ordered.insert(group.ordered.end(), group.single_op.begin(), group.single_op.end());
    }
    sort_registrations(registry_.single_op_registrations_);
    return std::move(registry_);
}

const DispatchRegistry * find_dispatch_registry(const DispatchTarget & target) {
    static const DispatchRegistry gfx1100_registry              = build_registry(true);
    static const DispatchRegistry gfx1100_generic_only_registry = build_registry(false);
    static const DispatchRegistry gfx1151_registry              = build_registry(true);
    static const DispatchRegistry gfx1151_generic_only_registry = build_registry(false);

    const bool qwen_disabled = qwen_dispatch_disabled_from_environment();

    if (target.architecture == "gfx1100") {
        return qwen_disabled ? &gfx1100_generic_only_registry : &gfx1100_registry;
    }
    if (target.architecture == "gfx1151") {
        return qwen_disabled ? &gfx1151_generic_only_registry : &gfx1151_registry;
    }
    return nullptr;
}

}  // namespace ggml::hrx
