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

// GGML_HRX_DISPATCH_SHAPE_LOG=<file>: one JSON line per built HRX command program, mapping every command to the ggml
// dispatch match that emitted it and every match to its ggml nodes and tensors. The engine's
// tools/hyperloom/hrx2kineto.py --shapes reads it next to an HRX dispatch profile (HRX_PROFILE_MODE=dispatch) and turns
// kernel names into hrx:: ops with shapes, FLOPs and bytes for TraceLens (engine #679). Off, the scheduler records
// nothing and no line is written. On, kernels are recorded in program order (graph-record-order.h), so the profile
// keys line up with the commands listed here; compare a profiled arm only against another shape-log or
// GGML_HRX_GRAPH_ORDER=program arm.
//
// Line (type "hrx_command_program"): program (a process-unique, increasing id), graph_uid, commands[] of
// {phase: "init"|"main", ordinal, kernel (definition name), symbol (the code-object export, the profile's event key),
// match (index into matches, -1 when no match emitted it)}, matches[] of {nodes: [{op, index, desc, out, src[]}]};
// tensors are {id, name, type, ne[4], nb[4]}, src in HRX's own input order (SET_ROWS: rows, ids, destination).

#pragma once

#include "command-plan.h"
#include "command-program.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

#include <cstdint>
#include <string>

namespace ggml::hrx {

bool dispatch_shape_log_enabled();

std::string format_command_program_shape_line(const Graph &          graph,
                                              const CommandPlan &    plan,
                                              const CommandProgram & program,
                                              const KernelCorpus &   corpus,
                                              const std::string &    target,
                                              uint64_t               graph_uid,
                                              uint64_t               program_id);

// Appends the line for this program to the shape log when it is enabled.
void write_command_program_shape_log_if_requested(const Graph &          graph,
                                                  const CommandPlan &    plan,
                                                  const CommandProgram & program,
                                                  const KernelCorpus &   corpus,
                                                  const std::string &    target,
                                                  uint64_t               graph_uid);

}  // namespace ggml::hrx
