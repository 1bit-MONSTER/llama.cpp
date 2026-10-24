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

// Next-layer-block routing predictor for a streamed MoE (docs/moe-streaming.md, "Next").
//
// A model whose routed experts come from the drive pays one full layer of reads per layer. The
// prefetch that hides them needs layer L+1's experts while layer L computes, i.e. before L's own
// MoE output exists. The cheap predictor -- L+1's router applied to L's FFN input -- is wrong on
// this family: with hyper-connections L+1's router input is L+1's own mixing step of the residual
// *after* L's MoE, so the mixing step is missing and two errors stack.
//
// So the graph predicts through L+1's own token-mixer block: its PLE injection, its attention or
// DeltaNet block, its combine, its FFN mixer, its router. State writes are suppressed where a
// later exact read would see them (the recurrent state and the conv histories); an attention
// successor needs none, because the exact block rewrites the same cache cell before anything
// reads it. Only the ranking of the router logits is used, and more candidates than the layer
// routes to are kept, so the near misses are read too.
//
// Environment (read once):
//   ONEBIT_MOE_PREDICT=off|router|block   off (default), router only (no block, cheap), block
//   ONEBIT_MOE_PREDICT_DEPTH=N            candidates kept per layer (default: the layer's top-k)
//   ONEBIT_MOE_PREDICT_MARGIN=F           read a tail candidate only while its logit is within F
//                                         of the best one (default 0: read every kept candidate)
//   ONEBIT_MOE_PREDICT_MIN=N              never read fewer than N candidates per layer
//   ONEBIT_MOE_PREDICT_LAYERS=1,2,5-9,47  predict only these successor layers (default: all)
//   ONEBIT_MOE_PREDICT_NO_ATTN=1          skip the block for sparse-attention successors
//   ONEBIT_MOE_PREDICT_OUT=FILE           append one P/A line per predicted layer per step
//   ONEBIT_MOE_PREDICT_STATS=1            one hit-rate line per layer when the context closes
//
// The prediction is emitted as a graph output tensor, read back on the host after the step, rated
// against the routing the step actually used, and handed to whatever prefetch sink is installed.
// A streamer's in-graph op registers itself with set_prefetch(); with nothing installed the sink
// is the trace, so the hit rate is measurable before the streamer exists.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_backend_sched;
struct ggml_tensor;

namespace onebit::moe {

enum class predict_mode {
    off,
    router,
    block,
};

struct predict_config {
    predict_mode mode    = predict_mode::off;
    int          depth   = 0;      // 0: the layer's own top-k
    int          min     = 0;      // 0: depth
    float        margin  = 0.0f;   // 0: no gate
    bool         attn    = true;   // block for sparse-attention successors too
    std::vector<uint8_t> layers;   // empty: every layer
    std::string  trace_path;
    bool         stats   = false;
};

// Parsed from the environment on first use; immutable afterwards.
const predict_config & config();

bool enabled();                       // mode != off
bool wanted(int il_successor);        // predict this successor layer
int  depth(int n_expert_used);        // candidates kept for a layer that routes to n_expert_used

// The candidate policy, pure host code (tests/test-onebit-moe-predict.cpp).
// `scores` are the candidate logits, best first; the ids run in the same order, so the answer is
// how many of the leading candidates to read: min(depth, n), minus a tail whose margin to the best
// exceeds the configured one, and never fewer than `keep_min`.
int keep_count(const predict_config & cfg, const float * scores, int n, int keep_min);

struct layer_stat {
    uint64_t asked  = 0;   // steps this layer was predicted
    uint64_t hits   = 0;   // predicted candidates the layer really used
    uint64_t pred   = 0;   // predicted candidates
    uint64_t actual = 0;   // experts the layer really used
    uint64_t read   = 0;   // candidates the gate would read
    uint64_t gated  = 0;   // candidates the gate dropped
};

struct totals_t {
    uint64_t steps = 0, asked = 0, hits = 0, pred = 0, actual = 0, read = 0, gated = 0;
};

const std::vector<layer_stat> & layer_stats();
const totals_t & totals();
void reset_stats();
void dump_stats();   // per layer + totals, to stderr

// --- graph side -------------------------------------------------------------------------------
// Called once when the model graph is built with the predictor on: allocates this graph's
// prediction/actual readback tensors. False when the predictor is off.
bool graph_begin(ggml_context * ctx0, ggml_cgraph * gf, int n_layer, int depth, int n_used);
// Per layer: copy the prediction (ids best first, and their logits) into column `il`.
void graph_prediction(ggml_context * ctx0, ggml_cgraph * gf, int il, ggml_tensor * ids, ggml_tensor * scores);
// Per layer: copy the routing the layer actually used into column `il`.
void graph_actual(ggml_context * ctx0, ggml_cgraph * gf, int il, ggml_tensor * ids);
// Called after the step's compute (llama_context::graph_compute): read the tensors back, rate the
// step, write the trace, and hand each layer's gated candidate set to the prefetch sink.
void step_end(ggml_backend_sched * sched);

// The prefetch consumer (the streamer's in-graph op). `ids` are valid only during the call.
using prefetch_fn = void (*)(void * ud, int il, const int32_t * ids, int n);
void set_prefetch(prefetch_fn fn, void * ud);

}  // namespace onebit::moe
