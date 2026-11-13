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

// AMD's activation-publication consumer classes: dispatch-activation-publication.h (restored from the pin) reports which
// consumer classes a publication serves, and the merge kept the reporting without this enum.
enum DispatchActivationConsumerSource : uint32_t {
    DispatchActivationConsumerSourceCommon = 1u << 0,
    DispatchActivationConsumerSourceLlm    = 1u << 1,
    DispatchActivationConsumerSourceQwen   = 1u << 2,
    DispatchActivationConsumerSourceAll    = DispatchActivationConsumerSourceCommon |
                                             DispatchActivationConsumerSourceLlm |
                                             DispatchActivationConsumerSourceQwen,
};

class DispatchRegistry;

struct DispatchMatchContext {
    const Graph &             graph;
    const GraphNode *         root_node  = nullptr;
    size_t                    root_index = 0;
    const std::vector<bool> & covered_nodes;
    const CommandPlan &       plan;
    ValueId                   next_plan_value;
    // AMD's dispatch-activation-publication.h and the ops test both read the registry back out of the context. Our line had
    // removed this field; the merge kept the readers, so it comes back -- and now at the position AMD's own initializers expect.
    const DispatchRegistry *  registry = nullptr;
};

// Non-native weight bindings require bytes that remain available after graph import.
bool dispatch_weight_layout_eligible(const Value & value);

enum DispatchActivationInputFormat : uint32_t {
    DispatchActivationInputNone           = 0,
    DispatchActivationInputF16Row         = 1u << 0,
    DispatchActivationInputF16K16Major    = 1u << 1,
    DispatchActivationInputQ8_1X4         = 1u << 2,
    DispatchActivationInputSymmetricI4K32 = 1u << 3,
};

enum DispatchActivationProducerRequirement : uint32_t {
    DispatchActivationProducerRequirementNone             = 0,
    DispatchActivationProducerRequirementPackedK16Economy = 1u << 0,
    DispatchActivationProducerRequirementFactoredTiledF16 = 1u << 1,
};

enum DispatchActivationConsumerTrait : uint32_t {
    DispatchActivationConsumerTraitNone                   = 0,
    DispatchActivationConsumerTraitQ8PublicationPreferred = 1u << 0,
};

enum class DispatchActivationConsumerUse {
    BandwidthLimited,
    Tiled,
    Row,
    LowRow,
};

using DispatchActivationConsumerPredicate = bool (*)(const DispatchMatchContext & context,
                                                      const GraphNode &            consumer,
                                                      const Value &                input);

struct DispatchActivationConsumerRegistration {
    const char *                        name = "";
    ggml_op                             root_op = GGML_OP_NONE;
    uint32_t                            accepted_formats = DispatchActivationInputNone;
    DispatchActivationConsumerUse       use = DispatchActivationConsumerUse::Row;
    uint32_t                            producer_requirements = DispatchActivationProducerRequirementNone;
    DispatchSource                      source = DispatchSource::Common;
    DispatchActivationConsumerPredicate accepts = nullptr;
    uint32_t                            traits = DispatchActivationConsumerTraitNone;
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

    // AMD's publication bookkeeping: which consumer classes a publication for this root op serves.
    const std::vector<DispatchActivationConsumerRegistration> &
    activation_consumers_for_root(ggml_op root_op) const;

    const std::vector<DispatchRegistration> & single_op_registrations() const { return single_op_registrations_; }

  private:
    friend class DispatchRegistryBuilder;

    struct RegistrationGroup {
        std::vector<DispatchRegistration> fused;
        std::vector<DispatchRegistration> single_op;
        std::vector<DispatchRegistration> ordered;
    };

    std::vector<RegistrationGroup>    registrations_by_root_;
    std::vector<std::vector<DispatchActivationConsumerRegistration>> activation_consumers_by_root_;
    std::vector<DispatchRegistration> single_op_registrations_;
};

class DispatchRegistryBuilder {
  public:
    void             add(DispatchRegistration registration);
    void             add_activation_consumer(DispatchActivationConsumerRegistration registration);
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
