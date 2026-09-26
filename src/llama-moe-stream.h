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
#pragma once

// Routed experts streamed from the model file (1bit engine, docs/moe-streaming.md): a MoE
// layer's expert FFN runs out of pinned RAM slots that an O_DIRECT reader fills on demand, so a
// model's experts need not fit in RAM. Enabled by environment:
//   ONEBIT_MOE_FILE      the GGUF (first shard of a split file) the model was loaded from
//   ONEBIT_MOE_SLOTS     experts held in RAM, all layers together
//   ONEBIT_MOE_IO        reads in flight (default 8)
//   ONEBIT_MOE_MAX_BATCH largest batch that streams (default 8); bigger batches (prompts) use
//                        the regular path over the mmap-ed expert tensors
//   ONEBIT_MOE_PREFETCH  MoE layers ahead whose experts the gate-ahead prefetch queues
//                        (default 1; 0 turns it off)
//   ONEBIT_MOE_STATS     1 prints where decode time went, at exit
//   ONEBIT_MOE_SUBST     r > 0: a routed expert that is not resident is replaced by a resident one
//                        among the router's next top-k choices, if that one scores at least r
//                        times the missing one (0, the default: exact routing)
//   ONEBIT_MOE_DEVICE    where streamed experts compute: a GPU device name (default: the first
//                        GPU that imports host memory) or "cpu"
// On a GPU (Strix Halo's is unified memory), each layer's pinned slots are imported as a device
// buffer and viewed as expert tensors; a CPU op maps the routed expert ids to slot indices and
// the regular MUL_MAT_ID path runs on the GPU over the slots.
// Load the model with the expert tensors on the CPU and mmap-ed (-ot exps=CPU), so they are
// never read into memory except by a prompt batch.

#include "ggml.h"

struct llama_moe_stream;

// The process-wide streamer, or nullptr when ONEBIT_MOE_SLOTS is unset (or setup failed).
llama_moe_stream * llama_moe_stream_get();

// Whether a batch of n_tokens streams, and whether these tensors can (separate gate and up,
// no biases or per-expert scales).
// A batch streams only when all of its routed experts fit in one layer's slots at once.
bool llama_moe_stream_applies(const llama_moe_stream * s, int64_t n_tokens, int64_t n_expert_used,
                              const ggml_tensor * gate_exps, const ggml_tensor * up_exps, const ggml_tensor * down_exps);

// GPU mode: pins the batch's experts (ids -> slot indices) and returns the slot tensors to use
// in place of the layer's expert tensors. False when the layer cannot stream on the GPU.
// cur: the layer's FFN input [n_embd, n_tokens], for the next layer's gate-ahead prefetch.
bool llama_moe_stream_gpu(ggml_context * ctx, llama_moe_stream * s, int il, ggml_tensor * cur, ggml_tensor * ids,
                          ggml_tensor * gate_exps, ggml_tensor * up_exps, ggml_tensor * down_exps,
                          ggml_tensor ** slot_ids, ggml_tensor ** slot_gate, ggml_tensor ** slot_up,
                          ggml_tensor ** slot_down);

// Resident-aware top-k (ONEBIT_MOE_SUBST): the experts of layer il chosen from selection_probs
// [n_expert, n_tokens] F32, preferring resident ones (see ONEBIT_MOE_SUBST); nullptr when off.
ggml_tensor * llama_moe_stream_select(ggml_context * ctx, llama_moe_stream * s, int il, ggml_tensor * selection_probs,
                                      int64_t n_expert_used);

// CPU mode: the expert FFN of layer il with SwiGLU, from the streamed slots: returns the per-expert down
// projections [n_embd, n_expert_used, n_tokens], like the down MUL_MAT_ID it replaces.
// cur: [n_embd, n_tokens] F32; ids: [n_expert_used, n_tokens] I32 (may be a view).
ggml_tensor * llama_moe_stream_build(ggml_context * ctx, llama_moe_stream * s, int il, ggml_tensor * cur,
                                     ggml_tensor * ids, ggml_tensor * gate_exps, ggml_tensor * up_exps,
                                     ggml_tensor * down_exps);
