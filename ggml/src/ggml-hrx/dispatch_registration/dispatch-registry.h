#pragma once

#include "dispatch/command-plan.h"
#include "ggml.h"
#include "graph/graph.h"
#include "status.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ggml::hrx {

struct DispatchTarget {
    std::string architecture;
};

enum class DispatchMatchKind {
    Fused,
    SingleOp,
};

enum class DispatchSource {
    Common,
    Llm,
    Qwen,
};

struct DispatchMatchContext {
    const Graph &             graph;
    const GraphNode *         root_node  = nullptr;
    size_t                    root_index = 0;
    const std::vector<bool> & covered_nodes;
    const CommandPlan &       plan;
    ValueId                   next_plan_value;
};

struct DispatchValueAliasRequest {
    ValueId source_value;
    ValueId target_value;
};

struct DispatchMatch {
    std::vector<Dispatch>                            initialization_dispatches;
    std::vector<size_t>                              covered_nodes;
    std::vector<Dispatch>                            dispatches;
    std::vector<CommandPlanTransient>                transients;
    std::vector<CommandPlanConstantInitialization>   constant_initializations;
    std::vector<CommandPlanCompletionCounterRequest> completion_counter_requests;
    std::vector<DispatchValueAliasRequest>           value_aliases;
    CommandPlanMetadata                              metadata;
    Status                                           status;
};

using DispatchMatcher = bool (*)(const DispatchMatchContext & context, DispatchMatch & match);

struct DispatchRegistration {
    const char *      name     = "";
    ggml_op           root_op  = GGML_OP_NONE;
    DispatchMatchKind kind     = DispatchMatchKind::SingleOp;
    int               priority = 0;
    DispatchSource    source   = DispatchSource::Common;
    DispatchMatcher   matcher  = nullptr;
};

struct DispatchRegistrationAttempt {
    std::string              name;
    ggml_op                  root_op  = GGML_OP_NONE;
    DispatchMatchKind        kind     = DispatchMatchKind::SingleOp;
    int                      priority = 0;
    DispatchSource           source   = DispatchSource::Common;
    bool                     matched  = false;
    std::vector<size_t>      covered_nodes;
    std::vector<std::string> errors;
};

struct DispatchMatchDiagnostics {
    ggml_op                                  root_op = GGML_OP_NONE;
    std::vector<DispatchRegistrationAttempt> attempts;
};

class DispatchRegistry {
  public:
    bool match(const DispatchMatchContext & context, DispatchMatch & match) const;
    bool match(const DispatchMatchContext & context,
               DispatchMatch &              match,
               DispatchMatchDiagnostics *   diagnostics) const;

    const std::vector<DispatchRegistration> & registrations_for_root(ggml_op root_op) const;

    const std::vector<DispatchRegistration> & single_op_registrations() const { return single_op_registrations_; }

  private:
    friend class DispatchRegistryBuilder;

    struct RegistrationGroup {
        std::vector<DispatchRegistration> fused;
        std::vector<DispatchRegistration> single_op;
        std::vector<DispatchRegistration> ordered;
    };

    std::vector<RegistrationGroup>    registrations_by_root_;
    std::vector<DispatchRegistration> single_op_registrations_;
};

class DispatchRegistryBuilder {
  public:
    void             add(DispatchRegistration registration);
    DispatchRegistry build();

  private:
    DispatchRegistry registry_;
};

const DispatchRegistry * find_dispatch_registry(const DispatchTarget & target);

// GGML_HRX_DISABLE_DISPATCH, the one rule for it: true when a comma-separated pattern is a substring of `name`, or is
// "fused" and kind is Fused. The registry skips such a registration, and a supports_op claim for a node that only that
// registration executes must decline with it, or the scheduler keeps the node on HRX with no dispatch and the graph
// fails ("unsupported HRX node", engine #2146: =top_k left TOP_K claimed because the claim looked for the full name
// "hip.top_k_f32" inside the variable instead of the pattern inside the name).
bool dispatch_name_disabled(const char * name, DispatchMatchKind kind = DispatchMatchKind::SingleOp);

}  // namespace ggml::hrx
