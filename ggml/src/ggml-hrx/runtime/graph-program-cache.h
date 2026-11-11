#pragma once

#include "dispatch/command-program.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/graph-program-budget.h"
#include "runtime/command-program-executor.h"
#include "runtime/prepared-command-program-cache.h"
#include "status.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;

namespace ggml::hrx {

struct GraphProgramExternalBinding {
    ValueId             value;
    const ggml_tensor * tensor = nullptr;
};

enum class GraphProgramExternalSlotKind {
    Node,
    Source,
};

struct GraphProgramExternalSlot {
    ValueId                      value;
    GraphProgramExternalSlotKind kind         = GraphProgramExternalSlotKind::Node;
    size_t                       node_index   = 0;
    int32_t                      source_index = -1;
    bool                         weight_layout_eligible = true;
};

struct GraphProgramMatch {
    std::vector<GraphProgramExternalBinding> external_bindings;
    Status                                   status;

    bool valid() const { return status.success(); }
};

class GraphProgram {
  public:
    GraphProgram(uint64_t                        uid,
                 std::string                     target,
                 std::unique_ptr<Graph>          graph,
                 std::unique_ptr<CommandProgram> commands,
                 std::string                     command_shape);

    uint64_t uid() const { return uid_; }

    const std::string & target() const { return target_; }

    const std::string & command_shape() const { return command_shape_; }

    uint64_t command_shape_hash() const { return command_shape_hash_; }

    // What this program keeps alive while it stays cached: its transient arena plus its own prepared command
    // program's staging set, about 1 GiB for a large prefill shape together (graph-program-budget.h).
    size_t retained_bytes() const;

    const Graph & graph() const { return *graph_; }

    Graph & graph() { return *graph_; }

    const CommandProgram & commands() const { return *commands_; }

    GraphProgramMatch match_current_graph(const ggml_cgraph & graph) const;
    GraphProgramMatch match_trusted_graph(const ggml_cgraph & graph, bool bind_external = true) const;
    GraphProgramMatch match_host_staging_graph(const ggml_cgraph & graph) const;
    Status capture_external_slots(const ggml_cgraph & graph, const GraphProgramMatch & match);

    bool can_use_prepared_fast_path(const ggml_cgraph & graph) const;
    bool mutable_sources_current() const;
    std::shared_ptr<PreparedCommandProgramCacheEntry> prepared_entry() const;
    void bind_prepared_graph(const ggml_cgraph & graph, const GraphProgramMatch & match,
                             const CommandProgramBindings & bindings,
                             std::shared_ptr<PreparedCommandProgramCacheEntry> entry);

  private:
    const GraphProgramExternalSlot * find_external_slot(ValueId value) const;
    const ggml_tensor *              resolve_external_slot(const ggml_cgraph &              graph,
                                                           const GraphProgramExternalSlot & slot,
                                                           Status &                         status) const;

    uint64_t                        uid_ = 0;
    std::string                     target_;
    std::unique_ptr<Graph>          graph_;
    std::unique_ptr<const CommandProgram> commands_;
    std::string                     command_shape_;
    uint64_t                        command_shape_hash_ = 0;

    std::vector<GraphProgramExternalSlot> external_slots_;
    struct MutableSourceWatch {
        const ggml_tensor * tensor;
        uint64_t            identity;
        size_t              offset;
        size_t              length;
    };
    std::vector<MutableSourceWatch> mutable_source_watches_;
    std::unordered_map<int32_t, size_t>   external_slot_by_value_;
    uint64_t                              fast_path_uid_ = 0;
    const ggml_tensor * const *           fast_path_nodes_ = nullptr;
    mutable std::mutex                    prepared_mutex_;
    std::shared_ptr<PreparedCommandProgramCacheEntry> prepared_entry_;
};

struct GraphProgramCacheStats {
    uint64_t builds                  = 0;
    uint64_t hits                    = 0;
    // lifetime totals, not levels: the counts of programs this cache has already dropped are folded into the cache
    // as they are erased, so these only ever grow and a trim cannot make them go backwards (engine #841).
    uint64_t prepared_program_builds = 0;
    uint64_t prepared_program_hits   = 0;
};

struct GraphProgramLookup {
    GraphProgram *                program = nullptr;
    std::unique_ptr<GraphProgram> uncached_program;
    GraphProgramMatch             match;
    Status                        status;

    bool valid() const { return program != nullptr && status.success() && match.valid(); }
};

struct GraphProgramSupportResult {
    bool   supported = false;
    Status status;

    bool valid() const { return supported && status.success(); }
};

class GraphProgramCache {
  public:
    bool can_execute(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target) const;

    GraphProgramSupportResult check_support(const ggml_cgraph &  graph,
                                            const KernelCorpus & corpus,
                                            const std::string &  target) const;

    GraphProgramLookup get_or_build(const ggml_cgraph & graph, const KernelCorpus & corpus, const std::string & target);
    GraphProgramCacheStats stats() const;

    size_t size() const;
    // bytes held by the cached programs (arena + prepared command programs; graph-program-budget.h)
    size_t bytes() const;
    // the programs to drop so that at most budget_bytes and max_programs remain, least recently used
    // first, never `keep`; empty when nothing has to go
    std::vector<uint64_t> plan_evictions(size_t budget_bytes, size_t max_programs, const GraphProgram * keep) const;
    // Re-derive what each cached program retains, before the byte budget is consulted (engine #841): the transient
    // arena is not the whole story, and the rest (~1 GiB per prefill shape in the prepared command programs) is
    // exactly what the knob was there to bound. `extra_bytes` supplies the bytes attributed to a uid outside the
    // program itself; the LRU order is left alone.
    void retally_retained_bytes(const std::function<size_t(uint64_t)> & extra_bytes);
    // drop these programs, freeing their arenas; the caller has already waited for the stream
    void evict(const std::vector<uint64_t> & uids);
    // drop every cached program except `keep` (graph-program-cache-limit.h)
    void   retain_only(const GraphProgram * keep);

    void clear();

  private:
    // carry a program's own prepared counters into the cache before it is destroyed, so stats() stays a lifetime total
    void fold_prepared_stats(const GraphProgram & program);
    // destroy one cached program, folding its counters in first; `evict` and `retain_only` go through it, `clear`
    // folds in bulk because it is dropping everything anyway
    void drop_program(uint64_t uid);

    GraphProgramLookup get_or_build_impl(const ggml_cgraph &  graph,
                                         const KernelCorpus & corpus,
                                         const std::string &  target);

    struct ValidatedGraphMatch {
        GraphProgram *              program = nullptr;
        const ggml_tensor * const * nodes   = nullptr;
    };

    GraphProgramLookup build_from_imported(const ggml_cgraph &  graph,
                                           Graph &&             imported_graph,
                                           const KernelCorpus & corpus,
                                           const std::string &  target);

    std::unique_ptr<GraphProgram> build_program_from_imported(uint64_t             uid,
                                                              Graph &&             imported_graph,
                                                              const KernelCorpus & corpus,
                                                              const std::string &  target,
                                                              Status &             errors) const;

    mutable std::mutex                                          mutex_;
    std::unordered_map<uint64_t, std::unique_ptr<GraphProgram>> programs_;
    std::unordered_map<uint64_t, ValidatedGraphMatch>           validated_matches_;
    GraphProgram *                                              last_program_ = nullptr;
    GraphProgramCacheStats                                      stats_;
    // last-use order and byte accounting of the cached programs (graph-program-budget.h)
    GraphProgramBudget                                          budget_;
};

bool can_execute_standalone_op_as_graph(const ggml_tensor * op, const std::string & target);

}  // namespace ggml::hrx
