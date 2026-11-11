#include "graph-executor.h"

#include "backend-buffer-binding.h"
#include "hrx-interop-utils.h"
#include "ggml-impl.h"
<<<<<<< 7a1f2b81ce17aaa84a40b766d1b06e5fd4601196
#include "runtime/graph-program-cache-limit.h"
#include "runtime/graph-replay.h"
#include "runtime/host-gap-profile.h"
=======
#include "hrx-interop-utils.h"
#include "hrx_runtime.h"
>>>>>>> 9bf5871310a8d7de74f4cb0c57009835fbf91c27
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <utility>
#include <vector>

namespace ggml::hrx {

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

Status GraphExecutor::context_valid_for_graph_programs() const {
    Status status;
    if (context_.device == nullptr) {
        status.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        status.log("missing HRX target");
    }
    return status;
}

Status GraphExecutor::context_valid_for_execution() const {
    return context_valid_for_graph_programs();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    result.status = context_valid_for_graph_programs();
    if (!result.status.success()) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.status.append(support.status);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    Status                             status;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        if (ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding)) {
<<<<<<< 7a1f2b81ce17aaa84a40b766d1b06e5fd4601196
            binding.buffer     = value_binding.buffer;
            binding.host_data  = value_binding.host_data;
            binding.offset     = value_binding.offset;
            binding.length     = value_binding.length;
            binding.identity   = value_binding.identity;
            binding.generation = value_binding.generation;
            binding.capacity   = value_binding.capacity;
            binding.weight     = value_binding.weight;
            binding.empty_value = ggml_nbytes(external.tensor) == 0;
            binding.graph_input = (external.tensor->flags & GGML_TENSOR_FLAG_INPUT) != 0 ||
                                  (external.tensor->view_src != nullptr &&
                                   (external.tensor->view_src->flags & GGML_TENSOR_FLAG_INPUT) != 0);
=======
            bindings.push_back(make_command_program_binding(external.value, value_binding,
                                                            ggml_nbytes(external.tensor) == 0));
>>>>>>> 9bf5871310a8d7de74f4cb0c57009835fbf91c27
        } else {
            status.log("external value %d is not bound", external.value.value);
        }
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), status);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    CompileTraceExecution trace(context_.kernel_executables.trace_id(), graph.uid);
    if (graph.n_nodes == 0) {
        result.code = GGML_STATUS_SUCCESS;
        trace.success = true;
        return result;
    }
    result.status = context_valid_for_execution();
    if (!result.status.success()) {
        return result;
    }

    const uint64_t     graph_start_ns = host_gap_enabled() ? hrx_graph_replay_now_ns() : 0;
    host_gap_begin_graph();
    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    const uint64_t       builds_before = context_.graph_programs.stats().builds;
    GraphProgramLookup   lookup;
    {
        // a graph-program cache hit, or a match of every node against the dispatch registry
        const HostGapScope scope("graph.lookup");
        lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    }
    if (lookup.valid()) {
        trim_graph_programs(context_, lookup.program, builds_before);  // graph-program-cache-limit.h
    }
    if (!lookup.valid()) {
        result.status.append(lookup.status);
        result.status.append(lookup.match.status);
        if (result.status.success()) {
            result.status.log("build HRX graph program failed");
        }
        return result;
    }

    const bool reusable_owner = lookup.program->can_use_prepared_fast_path(graph);
    const bool mutable_sources_current = reusable_owner && lookup.program->mutable_sources_current();
    const bool trusted_weights = reusable_owner && mutable_sources_current;
    GraphProgramMatch binding_match = std::move(lookup.match);
<<<<<<< 7a1f2b81ce17aaa84a40b766d1b06e5fd4601196
    if (use_graph_prepared && lookup.program->has_prepared_program()) {
        binding_match = lookup.program->match_host_staging_graph(graph);
        if (!binding_match.valid()) {
            result.status.append(binding_match.status);
            if (result.status.success()) {
                result.status.log("match HRX host-staging graph failed");
            }
            return result;
        }
=======
    if (reusable_owner) {
        binding_match = trusted_weights ? lookup.program->match_host_staging_graph(graph) :
                                          lookup.program->match_trusted_graph(graph, true);
    }
    if (!binding_match.valid()) {
        result.status.append(binding_match.status);
        return result;
>>>>>>> 9bf5871310a8d7de74f4cb0c57009835fbf91c27
    }

    CommandProgramBindings bindings;
    {
        const HostGapScope scope("graph.bind");
        bindings = bind_external_value_buffers(binding_match);
    }
    if (!bindings.valid()) {
        result.status.append(bindings.status);
        if (result.status.success()) {
            result.status.log("bind HRX external value buffers failed");
        }
        return result;
    }
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.kernel_executables,
        &context_.transient_arena,
        &context_.host_transfers,
        &context_.host_weights,
        &context_.device->host_buffers,
        &context_.graph_replay_state,
<<<<<<< 7a1f2b81ce17aaa84a40b766d1b06e5fd4601196
        context_.device->memory_budget,
    };
    PreparedCommandProgramCacheExecutionResult execution;
    {
        const HostGapScope scope("graph.execute");
        execution = use_graph_prepared ? lookup.program->execute_with_result(execution_context, bindings) :
                                        context_.prepared_programs.execute_with_result(execution_context, lookup.program->uid(),
                                                                                       lookup.program->command_shape_hash(),
                                                                                       lookup.program->commands(), bindings);
    }
    host_gap_end_graph(hrx_graph_replay_now_ns() - graph_start_ns, execution.graph_replay_build_ns,
                       execution.graph_replay_launch_ns, execution.graph_replay_dispatches);
=======
        trace.ids.cache_id != 0 ? &trace : nullptr,
    };
    std::shared_ptr<PreparedCommandProgramCacheEntry> entry;
    if (trusted_weights) {
        entry = lookup.program->prepared_entry();
    } else if (graph.uid != 0) {
        entry = context_.prepared_programs.get_or_create(execution_context, lookup.program->uid(),
                                                        lookup.program->command_shape_hash(), bindings);
    }
    const PreparedCommandProgramCacheExecutionResult execution = entry != nullptr ?
        context_.prepared_programs.execute_with_result(execution_context, lookup.program->commands(), bindings, entry) :
        context_.prepared_programs.execute_with_result(execution_context, 0, lookup.program->command_shape_hash(),
                                                       lookup.program->commands(), bindings);
    result.graph_replay_event = execution.graph_replay_event;
>>>>>>> 9bf5871310a8d7de74f4cb0c57009835fbf91c27
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }

    const bool validated_entry = entry != nullptr && !trusted_weights;
    if (validated_entry) {
        lookup.program->bind_prepared_graph(graph, binding_match, bindings, std::move(entry));
    }

    result.code = GGML_STATUS_SUCCESS;
<<<<<<< 7a1f2b81ce17aaa84a40b766d1b06e5fd4601196

    // engine #1251: a download this program staged is only a *queued* memcpy until mark_stream_synchronized() runs
    // (GraphReplayStreamState::add_host_writeback defers the publish to the next backend synchronize). ggml's
    // scheduler shares a host tensor between the HRX split and a CPU split without inserting a copy — it is the same
    // memory, so there is nothing to copy — which means a CPU node in the next split can read that tensor before
    // anything synchronizes HRX. The Qwen4-exp QSA mask is exactly that shape: a CPU fill of -inf, a device SET_ROWS
    // that unmasks the top-k cells, and a CPU add that consumes the result; without this publish the add reads the
    // fill's -inf for every cell and every logit is NaN. Publish here, i.e. copy the materialised host output back
    // before the graph compute returns.
    if (context_.graph_replay_state.has_pending_host_writebacks()) {
        if (ErrorResult error = take_status(hrx_stream_synchronize(context_.stream))) {
            result.status.log("synchronize HRX stream to publish host writebacks: %s", error->c_str());
            result.code = GGML_STATUS_FAILED;
        }
        context_.graph_replay_state.mark_stream_synchronized();
    }
=======
    trace.success = true;
>>>>>>> 9bf5871310a8d7de74f4cb0c57009835fbf91c27
    return result;
}

}  // namespace ggml::hrx
