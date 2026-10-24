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

#include "onebit-moe-predict.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace onebit::moe {

void dump_stats();

namespace {

// "1,2,5-9,47" -> a per-layer flag vector; anything unparsable is ignored, which leaves the
// layer unpredicted rather than aborting a run
std::vector<uint8_t> parse_layers(const char * s) {
    std::vector<uint8_t> out;
    if (s == nullptr || *s == '\0') {
        return out;
    }
    const char * p = s;
    while (*p != '\0') {
        while (*p == ',' || *p == ' ') {
            ++p;
        }
        if (*p < '0' || *p > '9') {
            break;
        }
        char * end = nullptr;
        const long a = strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        p = end;
        long b = a;
        if (*p == '-') {
            ++p;
            b = strtol(p, &end, 10);
            if (end == p) {
                b = a;
            } else {
                p = end;
            }
        }
        if (a < 0 || b < a || b > 4095) {
            break;
        }
        if ((long) out.size() <= b) {
            out.resize((size_t) b + 1, 0);
        }
        for (long l = a; l <= b; ++l) {
            out[(size_t) l] = 1;
        }
        while (*p == ' ') {
            ++p;
        }
    }
    return out;
}

predict_config parse() {
    predict_config cfg;

    const char * m = getenv("ONEBIT_MOE_PREDICT");
    if (m != nullptr && strcmp(m, "block") == 0) {
        cfg.mode = predict_mode::block;
    } else if (m != nullptr && strcmp(m, "router") == 0) {
        cfg.mode = predict_mode::router;
    } else if (m != nullptr && strcmp(m, "off") != 0 && *m != '\0') {
        fprintf(stderr, "[moe-predict] unknown ONEBIT_MOE_PREDICT=%s, using off (block|router|off)\n", m);
    }

    if (const char * v = getenv("ONEBIT_MOE_PREDICT_DEPTH")) cfg.depth  = atoi(v);
    if (const char * v = getenv("ONEBIT_MOE_PREDICT_MIN"))   cfg.min    = atoi(v);
    if (const char * v = getenv("ONEBIT_MOE_PREDICT_MARGIN")) cfg.margin = (float) atof(v);
    if (const char * v = getenv("ONEBIT_MOE_PREDICT_NO_ATTN")) cfg.attn = atoi(v) == 0;
    if (const char * v = getenv("ONEBIT_MOE_PREDICT_LAYERS")) cfg.layers = parse_layers(v);
    if (const char * v = getenv("ONEBIT_MOE_PREDICT_OUT"))    cfg.trace_path = v;
    if (const char * v = getenv("ONEBIT_MOE_PREDICT_STATS"))  cfg.stats = atoi(v) != 0;

    if (cfg.depth  < 0) cfg.depth  = 0;
    if (cfg.min    < 0) cfg.min    = 0;
    if (cfg.margin < 0.0f) cfg.margin = 0.0f;

    if (cfg.mode != predict_mode::off) {
        size_t n_sel = 0;
        for (uint8_t v : cfg.layers) n_sel += v != 0;
        fprintf(stderr, "[moe-predict] %s, depth %d, margin %.2f, min %d, %zu layer(s)%s%s\n",
                cfg.mode == predict_mode::block ? "block" : "router (no block)",
                cfg.depth, cfg.margin, cfg.min, cfg.layers.empty() ? (size_t) 0 : n_sel,
                cfg.attn ? "" : ", attention successors skipped",
                cfg.trace_path.empty() ? "" : (", trace " + cfg.trace_path).c_str());
    }
    return cfg;
}

const predict_config & cfg() {
    static const predict_config c = parse();
    static const bool at_exit = [] {
        if (c.stats) {
            std::atexit([] { dump_stats(); });
        }
        return true;
    }();
    (void) at_exit;
    return c;
}

// --- this graph's readback tensors ------------------------------------------------------------
ggml_tensor * t_pred_ids   = nullptr;   // I32 [depth, n_layer]
ggml_tensor * t_pred_score = nullptr;   // F32 [depth, n_layer]
ggml_tensor * t_act_ids    = nullptr;   // I32 [n_used, n_layer]
std::vector<uint8_t> has_pred;          // per layer: a prediction was copied this build
int g_depth = 0;
int g_n_used = 0;

std::vector<layer_stat> g_layer;
totals_t  g_totals;
std::vector<int32_t> g_ids;
std::vector<float>   g_scores;
std::vector<int32_t> g_act;

prefetch_fn g_sink    = nullptr;
void *      g_sink_ud = nullptr;

FILE * g_trace = nullptr;
bool   g_trace_open_tried = false;

void trace_open() {
    if (g_trace_open_tried) {
        return;
    }
    g_trace_open_tried = true;
    if (cfg().trace_path.empty()) {
        return;
    }
    g_trace = fopen(cfg().trace_path.c_str(), "a");
    if (g_trace == nullptr) {
        fprintf(stderr, "[moe-predict] cannot open %s for the trace\n", cfg().trace_path.c_str());
    }
}

void ensure_layer_stats(int n_layer) {
    if ((int) g_layer.size() < n_layer) {
        g_layer.resize((size_t) n_layer);
    }
}

}  // namespace

const predict_config & config() { return cfg(); }

bool enabled() { return cfg().mode != predict_mode::off; }

bool wanted(int il) {
    const auto & c = cfg();
    if (c.mode == predict_mode::off || il < 0) {
        return false;
    }
    return c.layers.empty() || ((size_t) il < c.layers.size() && c.layers[(size_t) il] != 0);
}

int depth(int n_expert_used) {
    const int d = cfg().depth > 0 ? cfg().depth : n_expert_used;
    return std::max(1, d);
}

int keep_count(const predict_config & c, const float * scores, int n, int keep_min) {
    if (n <= 0 || scores == nullptr) {
        return 0;
    }
    const int want = c.depth > 0 ? std::min(c.depth, n) : n;
    // a candidate beyond the depth is read only while it sits close enough to the best one to
    // plausibly be the expert the true routing swapped in; the margin gate is what keeps a deeper
    // prediction from costing proportionally more bytes. A wrong prefetch still only costs
    // bandwidth, so the gate only trims a tail it is sure about.
    int keep = want;
    if (c.margin > 0.0f) {
        const float best = scores[0];
        for (int i = 1; i < want; ++i) {
            if (best - scores[i] > c.margin) {
                keep = i;
                break;
            }
        }
    }
    keep = std::max(keep, std::min(keep_min, want));
    return keep;
}

const std::vector<layer_stat> & layer_stats() { return g_layer; }
const totals_t & totals() { return g_totals; }

void reset_stats() {
    std::fill(g_layer.begin(), g_layer.end(), layer_stat{});
    g_totals = totals_t{};
}

void dump_stats() {
    if (g_totals.asked == 0) {
        return;
    }
    fprintf(stderr, "[moe-predict] %llu step(s), %llu prediction(s): %llu/%llu experts hit (%.1f%%), "
                    "%llu/%llu candidates read (%.1f%% gated)\n",
            (unsigned long long) g_totals.steps, (unsigned long long) g_totals.asked,
            (unsigned long long) g_totals.hits, (unsigned long long) g_totals.actual,
            g_totals.actual ? 100.0 * (double) g_totals.hits / (double) g_totals.actual : 0.0,
            (unsigned long long) g_totals.read, (unsigned long long) g_totals.pred,
            g_totals.pred ? 100.0 * (double) g_totals.gated / (double) g_totals.pred : 0.0);
    for (size_t il = 0; il < g_layer.size(); ++il) {
        const auto & s = g_layer[il];
        if (s.asked == 0) {
            continue;
        }
        fprintf(stderr, "[moe-predict]   layer %3zu: hit %5.1f%% (%llu/%llu), read %.1f/layer, gated %llu\n",
                il, s.actual ? 100.0 * (double) s.hits / (double) s.actual : 0.0,
                (unsigned long long) s.hits, (unsigned long long) s.actual,
                (double) s.read / (double) s.asked, (unsigned long long) s.gated);
    }
}

bool graph_begin(ggml_context * ctx0, ggml_cgraph * gf, int n_layer, int n_depth, int n_used) {
    if (!enabled() || ctx0 == nullptr || gf == nullptr || n_layer <= 0) {
        t_pred_ids = t_pred_score = t_act_ids = nullptr;
        return false;
    }
    g_depth  = std::max(1, n_depth);
    g_n_used = std::max(1, n_used);
    has_pred.assign((size_t) n_layer, 0);
    ensure_layer_stats(n_layer);

    t_pred_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, g_depth, n_layer);
    t_pred_score = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, g_depth, n_layer);
    t_act_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, g_n_used, n_layer);
    ggml_set_output(t_pred_ids);
    ggml_set_output(t_pred_score);
    ggml_set_output(t_act_ids);
    // keep them allocated even when no layer writes them (a graph with the predictor off-handed)
    ggml_build_forward_expand(gf, t_pred_ids);
    ggml_build_forward_expand(gf, t_pred_score);
    ggml_build_forward_expand(gf, t_act_ids);
    return true;
}

void graph_prediction(ggml_context * ctx0, ggml_cgraph * gf, int il, ggml_tensor * ids, ggml_tensor * scores) {
    if (t_pred_ids == nullptr || ids == nullptr || scores == nullptr) {
        return;
    }
    const int64_t depth_ = ids->ne[0];
    if (depth_ <= 0 || ids->ne[1] != 1 || il < 0 || (size_t) il >= has_pred.size() || depth_ > g_depth) {
        return;
    }
    ggml_tensor * dst_ids = ggml_view_1d(ctx0, t_pred_ids, depth_, (size_t) il * t_pred_ids->nb[1]);
    ggml_tensor * dst_sco = ggml_view_1d(ctx0, t_pred_score, depth_, (size_t) il * t_pred_score->nb[1]);
    ggml_tensor * ids2 = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx0, ids);
    ggml_tensor * sco2 = ggml_is_contiguous(scores) ? scores : ggml_cont(ctx0, scores);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ids2, dst_ids));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, sco2, dst_sco));
    has_pred[(size_t) il] = 1;
}

void graph_actual(ggml_context * ctx0, ggml_cgraph * gf, int il, ggml_tensor * ids) {
    if (t_act_ids == nullptr || ids == nullptr) {
        return;
    }
    // the routing is per token and the predictor only runs for a single-token step
    if (ids->ne[0] != g_n_used || ids->ne[1] != 1 || il < 0 || il >= t_act_ids->ne[1]) {
        return;
    }
    ggml_tensor * dst = ggml_view_1d(ctx0, t_act_ids, g_n_used, (size_t) il * t_act_ids->nb[1]);
    ggml_tensor * src = ggml_is_contiguous(ids) ? ids : ggml_cont(ctx0, ids);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));
}

void set_prefetch(prefetch_fn fn, void * ud) {
    g_sink    = fn;
    g_sink_ud = ud;
}

void step_end(ggml_backend_sched * sched) {
    if (t_pred_ids == nullptr || sched == nullptr || has_pred.empty()) {
        return;
    }
    ggml_backend_t b_ids = ggml_backend_sched_get_tensor_backend(sched, t_pred_ids);
    ggml_backend_t b_sco = ggml_backend_sched_get_tensor_backend(sched, t_pred_score);
    ggml_backend_t b_act = ggml_backend_sched_get_tensor_backend(sched, t_act_ids);
    if (b_ids == nullptr || b_sco == nullptr || b_act == nullptr) {
        // not scheduled in the graph that just ran (a graph built without the predictor)
        return;
    }

    const int64_t n_layer = t_pred_ids->ne[1];
    g_ids.assign((size_t) g_depth, 0);
    g_scores.assign((size_t) g_depth, 0.0f);
    g_act.assign((size_t) g_n_used, 0);

    const auto & c = cfg();
    trace_open();

    g_totals.steps++;
    const uint64_t step = g_totals.steps - 1;

    for (int64_t il = 0; il < n_layer; ++il) {
        if ((size_t) il >= has_pred.size() || has_pred[(size_t) il] == 0) {
            continue;
        }
        ggml_backend_tensor_get(t_pred_ids, g_ids.data(), (size_t) il * t_pred_ids->nb[1], (size_t) g_depth * sizeof(int32_t));
        ggml_backend_tensor_get(t_pred_score, g_scores.data(), (size_t) il * t_pred_score->nb[1], (size_t) g_depth * sizeof(float));
        ggml_backend_tensor_get(t_act_ids, g_act.data(), (size_t) il * t_act_ids->nb[1], (size_t) g_n_used * sizeof(int32_t));

        uint64_t hits = 0;
        for (int i = 0; i < g_depth; ++i) {
            for (int k = 0; k < g_n_used; ++k) {
                if (g_ids[(size_t) i] == g_act[(size_t) k]) {
                    hits++;
                    break;
                }
            }
        }
        const int keep = keep_count(c, g_scores.data(), g_depth, c.min > 0 ? c.min : g_n_used);

        auto & st = g_layer[(size_t) il];
        st.asked++;  st.hits += hits;  st.pred += (uint64_t) g_depth;
        st.actual += (uint64_t) g_n_used;  st.read += (uint64_t) keep;
        st.gated += (uint64_t) (g_depth - keep);

        g_totals.asked++;  g_totals.hits += hits;  g_totals.pred += (uint64_t) g_depth;
        g_totals.actual += (uint64_t) g_n_used;  g_totals.read += (uint64_t) keep;
        g_totals.gated += (uint64_t) (g_depth - keep);

        if (g_trace != nullptr) {
            fprintf(g_trace, "P %llu %lld %d", (unsigned long long) step, (long long) il, keep);
            for (int i = 0; i < keep; ++i) {
                fprintf(g_trace, " %d:%.6g", g_ids[(size_t) i], (double) g_scores[(size_t) i]);
            }
            fprintf(g_trace, "\n");
            fprintf(g_trace, "A %llu %lld %d", (unsigned long long) step, (long long) il, g_n_used);
            for (int k = 0; k < g_n_used; ++k) {
                fprintf(g_trace, " %d", g_act[(size_t) k]);
            }
            fprintf(g_trace, "\n");
        }

        if (g_sink != nullptr) {
            g_sink(g_sink_ud, (int) il, g_ids.data(), keep);
        }
    }
    if (g_trace != nullptr) {
        fflush(g_trace);
    }
}

}  // namespace onebit::moe
