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

#include "dispatch-shape-log.h"

#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <vector>

namespace ggml::hrx {
namespace {

constexpr const char * kShapeLogEnvironment = "GGML_HRX_DISPATCH_SHAPE_LOG";

std::string json_escape(const char * text) {
    std::string out;
    for (const char * c = text != nullptr ? text : ""; *c != '\0'; ++c) {
        switch (*c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                if (static_cast<unsigned char>(*c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(*c));
                    out += buf;
                } else {
                    out += *c;
                }
        }
    }
    return out;
}

void append_tensor(std::ostringstream & out, const Graph & graph, ValueId id) {
    const Value * value = graph.values().find(id);
    if (value == nullptr) {
        out << "null";
        return;
    }
    const char * name = value->tensor != nullptr ? value->tensor->name : "";
    out << "{\"id\":" << value->id.value << ",\"name\":\"" << json_escape(name) << "\",\"type\":\""
        << json_escape(ggml_type_name(value->type)) << "\",\"ne\":[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out << (i ? "," : "") << value->ne[i];
    }
    out << "],\"nb\":[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out << (i ? "," : "") << value->nb[i];
    }
    out << "]}";
}

// The match that emitted plan dispatch |index| of one phase (-1 when none recorded it).
std::vector<int> command_matches(const CommandPlan & plan, size_t command_count, bool initialization) {
    std::vector<int> matches(command_count, -1);
    for (size_t m = 0; m < plan.match_records.size(); ++m) {
        const CommandPlanMatchRecord & record = plan.match_records[m];
        const size_t first = initialization ? record.first_initialization_dispatch : record.first_dispatch;
        const size_t count = initialization ? record.initialization_dispatch_count : record.dispatch_count;
        for (size_t i = first; i < first + count && i < command_count; ++i) {
            matches[i] = static_cast<int>(m);
        }
    }
    return matches;
}

void append_commands(std::ostringstream &          out,
                     const std::vector<Command> &  commands,
                     const std::vector<int> &      matches,
                     const char *                  phase,
                     const KernelCorpus &          corpus,
                     const std::string &           target,
                     bool &                        first) {
    for (size_t i = 0; i < commands.size(); ++i) {
        const Command &           command  = commands[i];
        const KernelResolveResult resolved = resolve_kernel_definition(corpus, target, command.kernel.kernel_id);
        const std::string         kernel   = resolved.found() ? kernel_definition_name(*resolved.definition)
                                                              : "kernel_id=" + std::to_string(command.kernel.kernel_id);
        const char * symbol = resolved.found() && resolved.definition->symbol != nullptr ? resolved.definition->symbol : "";
        out << (first ? "" : ",") << "{\"phase\":\"" << phase << "\",\"ordinal\":" << command.ordinal
            << ",\"kernel\":\"" << json_escape(kernel.c_str()) << "\",\"symbol\":\"" << json_escape(symbol)
            << "\",\"match\":" << (i < matches.size() ? matches[i] : -1) << "}";
        first = false;
    }
}

}  // namespace

bool dispatch_shape_log_enabled() {
    static const bool enabled = [] {
        const char * path = std::getenv(kShapeLogEnvironment);
        return path != nullptr && path[0] != '\0';
    }();
    return enabled;
}

std::string format_command_program_shape_line(const Graph &          graph,
                                              const CommandPlan &    plan,
                                              const CommandProgram & program,
                                              const KernelCorpus &   corpus,
                                              const std::string &    target,
                                              uint64_t               graph_uid,
                                              uint64_t               program_id) {
    std::ostringstream out;
    out << "{\"type\":\"hrx_command_program\",\"program\":" << program_id << ",\"graph_uid\":" << graph_uid
        << ",\"commands\":[";
    bool first = true;
    append_commands(out, program.initialization_commands,
                    command_matches(plan, program.initialization_commands.size(), true), "init", corpus, target, first);
    append_commands(out, program.commands, command_matches(plan, program.commands.size(), false), "main", corpus,
                    target, first);
    out << "],\"matches\":[";
    const std::vector<GraphNode> & nodes = graph.nodes();
    for (size_t m = 0; m < plan.match_records.size(); ++m) {
        out << (m ? "," : "") << "{\"nodes\":[";
        const std::vector<size_t> & covered = plan.match_records[m].covered_nodes;
        for (size_t c = 0; c < covered.size(); ++c) {
            out << (c ? "," : "");
            if (covered[c] >= nodes.size()) {
                out << "null";
                continue;
            }
            const GraphNode &   node   = nodes[covered[c]];
            const Value *       output = graph.values().find(node.output);
            const ggml_tensor * tensor = output != nullptr ? output->tensor : nullptr;
            out << "{\"op\":\"" << json_escape(ggml_op_name(node.op)) << "\",\"index\":" << covered[c]
                << ",\"desc\":\"" << json_escape(tensor != nullptr ? ggml_op_desc(tensor) : ggml_op_name(node.op))
                << "\",\"out\":";
            append_tensor(out, graph, node.output);
            out << ",\"src\":[";
            for (size_t s = 0; s < node.inputs.size(); ++s) {
                out << (s ? "," : "");
                append_tensor(out, graph, node.inputs[s]);
            }
            out << "]}";
        }
        out << "]}";
    }
    out << "]}";
    return out.str();
}

void write_command_program_shape_log_if_requested(const Graph &          graph,
                                                  const CommandPlan &    plan,
                                                  const CommandProgram & program,
                                                  const KernelCorpus &   corpus,
                                                  const std::string &    target,
                                                  uint64_t               graph_uid) {
    if (!dispatch_shape_log_enabled()) {
        return;
    }
    static std::atomic<uint64_t> next_program{ 0 };
    static std::mutex            mutex;
    const std::string            line =
        format_command_program_shape_line(graph, plan, program, corpus, target, graph_uid, next_program++);
    std::lock_guard<std::mutex> lock(mutex);
    if (FILE * file = std::fopen(std::getenv(kShapeLogEnvironment), "a")) {
        std::fputs(line.c_str(), file);
        std::fputc('\n', file);
        std::fclose(file);
    }
}

}  // namespace ggml::hrx
