#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include "dispatch-registry.h"
#include "dispatch-fusion-order.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include "backend-buffer-binding.h"

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

static bool dispatch_has_eligible_weight_layouts(const DispatchMatchContext & context, const DispatchMatch & match) {
    auto eligible = [&](const std::vector<Dispatch> & dispatches) {
        for (const Dispatch & dispatch : dispatches) {
            for (const DispatchBinding & binding : dispatch.bindings) {
                if (binding.layout == kNativeWeightLayout) {
                    continue;
                }
                const Value * value = context.graph.values().find(binding.value);
                const bool has_graph_value = value != nullptr;
                const bool layout_is_eligible = !has_graph_value || dispatch_weight_layout_eligible(*value);
                const bool has_ineligible_weight = has_graph_value && !layout_is_eligible;
                if (has_ineligible_weight) {
                    return false;
                }
            }
        }
        return true;
    };
    return eligible(match.initialization_dispatches) && eligible(match.dispatches);
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
// skip. The bare word "fused" is not a substring here: it is special-cased below to skip EVERY
// registration whose kind is DispatchMatchKind::Fused, which is a large fraction of this backend
// (engine#2146 counted the fallout). That difference is invisible from the command line -- the
// switch looks like the per-name form the README documents -- so it is announced once, and the
// one-registration form is named. A category-wide disable is a state no test covers, and it can
// change the answer without failing a node: measured on LFM2.5-1.2B, `=fused` turned a coherent
// completion into fragments with no error, which is how it was read as a kernel defect twice.
// GGML_HRX_LOG_DISPATCH=1: print each registration name the first time it matches.
bool dispatch_name_disabled(const char * registration_name, DispatchMatchKind kind) {
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
        for (const std::string & pattern : out) {
            if (pattern == "fused") {
                std::fprintf(stderr,
                             "hrx: GGML_HRX_DISABLE_DISPATCH=fused disables EVERY registration whose kind is "
                             "DispatchMatchKind::Fused, not the names containing \"fused\" (engine#2146). That is a "
                             "configuration no test covers and it can change the answer without failing a node; to "
                             "A/B one kernel, name that registration instead, as hip/README.md documents.\n");
            }
        }
        return out;
    }();
    const std::string name = registration_name != nullptr ? registration_name : "";
    for (const std::string & p : patterns) {
        if ((p == "fused" && kind == DispatchMatchKind::Fused) || name.find(p) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool dispatch_disabled(const DispatchRegistration & registration) {
    return dispatch_name_disabled(registration.name, registration.kind);
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
    // Two callers need a covered value to stay host-readable: the ZHA dump ("zha-readback.", engine #1078) and the
    // backend-ops suite, whose `fusion_test_nodes()` are the values it compares ("hrx-readback.", engine #1259). A fused
    // match must not swallow either: the value has to exist outside the fused program, or the read-back sees whatever the
    // buffer held before - measured as `selected_experts` returning the ARGSORT input's float bytes.
    if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
        return false;
    }
    static constexpr const char * k_prefixes[] = { "zha-readback.", "hrx-readback." };
    for (const char * prefix : k_prefixes) {
        if (std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
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

// engine #1316: a match may not ask for a packed layout of a weight that another node reads in the native layout - a
// tied token embedding, gathered by GET_ROWS and contracted by the output MUL_MAT. A command program holds one binding per
// value, so the packing request and the native one collide and the executor refuses the whole program
// ("weight value 0 has conflicting resident layout requests", llama_decode ret = -3). Declining the match here lets the
// next registration for the same root op - the native-format one - handle the node, so both consumers share a single
// resident copy. Before this, moving the input layer onto the device was a hard failure rather than an option.
static bool match_requests_packed_layout_of_natively_read_weight(const DispatchMatchContext & context,
                                                                const DispatchMatch &        candidate,
                                                                const char *                 registration_name) {
    // A/B control, default on: GGML_HRX_LAYOUT_CONFLICT_NO_PACK=0 restores the previous behaviour (the packed match is
    // taken and the program is refused), which is what makes the cost of this gate measurable in one box state.
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_LAYOUT_CONFLICT_NO_PACK");
        return env == nullptr || env[0] == '\0' || env[0] != '0';
    }();
    if (!enabled || !context.graph.has_index()) {
        return false;
    }
    const ValueMap & values = context.graph.values();
    for (const Dispatch & dispatch : candidate.dispatches) {
        for (const DispatchBinding & binding : dispatch.bindings) {
            if (binding.layout == kNativeWeightLayout) {
                continue;
            }
            // Only a weight that is already device-resident can hit the conflict: the packing request and the native gather
            // then both need a resident binding of one value. When the weight is in a host buffer instead - the default,
            // where the input layer stays on the CPU - the gather is not on this backend at all and this gate must stay out
            // of the way, which is the difference between 0 nodes off with LLAMA_INPUT_LAYER_ON_DEV=1 and a needless
            // de-optimisation of the shipped path.
            const Value * value = values.find(binding.value);
            if (value == nullptr || value->tensor == nullptr || value->tensor->buffer == nullptr ||
                ggml_backend_buffer_is_host(value->tensor->buffer)) {
                continue;
            }
            for (const GraphNode * consumer : context.graph.index().consumers(binding.value)) {
                if (consumer != nullptr && consumer->op == GGML_OP_GET_ROWS) {
                    // greppable, because this changes a graph's placement: same rule as the host-weight decline in
                    // host-weight-placement.cpp. Once per (registration, weight), and GGML_LOG_ERROR because the CLI's log
                    // sink drops WARN.
                    static std::mutex            log_mutex;
                    static std::set<std::string> logged;
                    const std::string            key = std::string(registration_name != nullptr ? registration_name : "?") +
                                            "|" + (value->tensor->name != nullptr ? value->tensor->name : "?");
                    std::lock_guard<std::mutex> lock(log_mutex);
                    if (logged.insert(key).second) {
                        GGML_LOG_ERROR("hrx dispatch: %s declined - it asks for layout '%s' of weight '%s', which GET_ROWS "
                                       "also reads natively, so one program would need two resident layouts of it "
                                       "(engine #1316)\n",
                                       key.c_str(), binding.layout.c_str(), value->tensor->name);
                    }
                    return true;
                }
            }
        }
    }
    return false;
}

bool dispatch_weight_layout_eligible(const Value & value) {
    return ggml_backend_hrx_weight_layout_eligible(value.tensor);
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
        bool          matched = registration.matcher != nullptr && registration.matcher(context, candidate);
        if (matched && match_requests_packed_layout_of_natively_read_weight(context, candidate, registration.name)) {
            // engine #1316: see the helper - the next registration for this root op gets the node instead.
            matched = false;
        }
        // AMD: a match whose weights cannot take the layout it asks for declines, so a narrower registration gets the node
        if (matched && !dispatch_has_eligible_weight_layouts(context, candidate)) {
            matched = false;
        }
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
    if (registry_.activation_consumers_by_root_.empty()) {
        registry_.activation_consumers_by_root_.resize(kOpCount);
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


static const std::vector<DispatchActivationConsumerRegistration> kEmptyActivationConsumers;

const std::vector<DispatchActivationConsumerRegistration> &
DispatchRegistry::activation_consumers_for_root(ggml_op root_op) const {
    if (!valid_root_op(root_op) || activation_consumers_by_root_.empty()) {
        return kEmptyActivationConsumers;
    }
    return activation_consumers_by_root_[static_cast<size_t>(root_op)];
}

void DispatchRegistryBuilder::add_activation_consumer(DispatchActivationConsumerRegistration registration) {
    if (!valid_root_op(registration.root_op) || registration.accepts == nullptr ||
        registration.accepted_formats == DispatchActivationInputNone) {
        return;
    }
    if (registry_.activation_consumers_by_root_.empty()) {
        registry_.activation_consumers_by_root_.resize(kOpCount);
    }
    registry_.activation_consumers_by_root_[static_cast<size_t>(registration.root_op)].push_back(
        std::move(registration));
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
