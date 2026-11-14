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

// The routed-down hook (hip/hip-dispatches.h) on llm.routed_ffn.decode_down_next_q8: a HIP add-on may replace that
// registration's kernel with one of the same binding contract. This test builds the graph the registration fuses, at
// the active MoE profile's shapes (hidden 2048, expert hidden 768, 128 experts, 8 routes), as llama.cpp lays out a
// Qwen3-MoE layer:
//
//   res_in = x + attn                     n = rms_norm(res_in) * ffn_norm
//   gate = mul_mat_id(W_gate, n, ids)     up = mul_mat_id(W_up, n, ids)      h = swiglu(gate, up)
//   w = normalized get_rows(probs, ids)   e = mul_mat_id(W_down, h, ids) * w
//   res = res_in + sum_r view_r(e)        n2 = rms_norm(res) * next_norm     proj = W_next n2      out = res + z
//
// The registration writes the residual in place and publishes n2 only as its q8 alternate, so the test reads the
// residual through `out` and n2 through `proj`, a Q4_K projection that reads that alternate.
//
// Arms (argv): --arm loom sets GGML_HRX_HIP_MOE_DOWN=0, --arm hip sets it to 1, both before the registry is built
// (the add-on reads it once). Cases:
//   t1_q6k_oracle  1 token, Q6_K down, h also a graph output: the residual is checked per element against an fp64
//                  host product of HRX's own h, and proj against an fp64 host projection of HRX's own residual;
//   t1_q6k_model   1 token, Q6_K down, the model's topology;
//   t1_q4k         1 token, Q4_K down: the registration still matches (its Loom Q4_K kernel
//                  qwen3_moe_routed_down_q4k_q8_1_x4_next_q8 must be in the plan), so the hook is asked and must decline;
//   t2_q6k         2 tokens: the one-token registration does not apply.
// Every case is also checked against the same graph on the CPU backend.
//
// Path taken, two ways:
//   1. host, no device: the graph is imported and scheduled for gfx1151 and the plan's kernel names are checked (hip
//      arm: the add-on's moe_down_q6k_f32_next_q8 exactly once in the Q6_K one-token cases and the Loom
//      qwen3_moe_routed_down_q6k_f32_wave64_next_q8 never; loom arm: the reverse; the add-on kernel never in t1_q4k
//      or t2_q6k);
//   2. device: GGML_HRX_DUMP_COMMAND_PROGRAM_DIR is set per case and the kernel lists of the command programs HRX
//      built and ran are checked the same way.
// Each device case runs three times in one scheduler and the three results must be bit-identical.
//
// --plan-only runs part 1 and exits without initialising any backend; --cpu-only runs part 1 and then the device part's
// graphs on the CPU backend alone. With no HRX0 device the device part returns 77. TEST_HRX_ROUTED_DOWN_HIP_VERBOSE=1
// prints every planned kernel.

#include "dispatch/dispatch-scheduler.h"
#include "dispatch_registration/dispatch-registry.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "graph/graph.h"
#include "hip/hip-code-objects.h"
#include "hip/hip-kernel-registry.h"
#include "kernel-corpus/kernel-corpus.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

namespace {

constexpr int64_t kHidden    = 2048;  // kActiveLlmMoeDispatchProfile.hidden_size
constexpr int64_t kExpertFfn = 768;   // .expert_hidden_size
constexpr int64_t kExperts   = 128;   // .expert_count
constexpr int64_t kRoutes    = 8;     // .route_count
constexpr int64_t kNextOut   = 4096;  // the next layer's q projection in Qwen3-Coder-30B-A3B
constexpr float   kEps       = 1e-6f;

constexpr const char * kHipKernel  = "moe_down_q6k_f32_next_q8";
constexpr const char * kLoomQ6K    = "qwen3_moe_routed_down_q6k_f32_wave64_next_q8";
constexpr const char * kLoomQ4K    = "qwen3_moe_routed_down_q4k_q8_1_x4_next_q8";

struct Case {
    const char * name;
    int64_t      tokens;
    ggml_type    down_type;
    bool         expose_h;   // h is also a graph output (the fp64 oracle needs HRX's own h)
    bool         hook_case;  // the hook must take this case when the arm is on
};

const Case kCases[] = {
    { "t1_q6k_oracle", 1, GGML_TYPE_Q6_K, true, true },
    { "t1_q6k_model", 1, GGML_TYPE_Q6_K, false, true },
    { "t1_q4k", 1, GGML_TYPE_Q4_K, false, false },
    { "t2_q6k", 2, GGML_TYPE_Q6_K, false, false },
};

// ---- deterministic data -------------------------------------------------------------------------------------------

struct Data {
    std::vector<uint8_t> w_gate, w_up, w_down, w_next;  // quantized
    std::vector<float>   ffn_norm, next_norm, x, attn, z, probs;
    std::vector<int32_t> ids;  // [kExperts, tokens], the first kRoutes of each column are the routes
};

// Quantizes a [rows, cols] f32 matrix (row-major, cols contiguous) of N(0,1)/sqrt(cols) values through ggml.
std::vector<uint8_t> quantize_random(std::mt19937 & rng, ggml_type type, int64_t cols, int64_t rows) {
    std::normal_distribution<float> n(0.0f, 1.0f / std::sqrt((float) cols));
    std::vector<float>              f((size_t) (cols * rows));
    for (float & v : f) {
        v = n(rng);
    }
    std::vector<uint8_t> q(ggml_row_size(type, cols) * rows);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, rows, cols, nullptr);
    return q;
}

// [cols, rows, kExperts] experts: the routed experts get their own random matrices, every other expert a copy of
// one decoy matrix, so a wrong expert index changes the answer without quantizing all 128 experts.
std::vector<uint8_t> make_experts(std::mt19937 & rng, ggml_type type, int64_t cols, int64_t rows,
                                  const std::vector<bool> & routed) {
    const size_t         expert_bytes = ggml_row_size(type, cols) * rows;
    std::vector<uint8_t> out(expert_bytes * kExperts);
    const auto           decoy = quantize_random(rng, type, cols, rows);
    for (int64_t e = 0; e < kExperts; ++e) {
        const auto src = routed[e] ? quantize_random(rng, type, cols, rows) : decoy;
        std::memcpy(out.data() + e * expert_bytes, src.data(), expert_bytes);
    }
    return out;
}

Data make_data(const Case & c) {
    std::mt19937                          rng(20261113u + (uint32_t) c.tokens * 7u + (uint32_t) c.down_type);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float>       n(0.0f, 1.0f);
    Data                                  d;
    auto fill = [&](std::vector<float> & v, size_t count, float offset, float scale) {
        v.resize(count);
        for (float & f : v) {
            f = offset + scale * u(rng);
        }
    };
    fill(d.ffn_norm, kHidden, 1.0f, 0.25f);
    fill(d.next_norm, kHidden, 1.0f, 0.25f);
    fill(d.x, kHidden * c.tokens, 0.0f, 1.0f);
    fill(d.attn, kHidden * c.tokens, 0.0f, 0.5f);
    fill(d.z, kHidden * c.tokens, 0.0f, 1.0f);

    d.ids.resize(kExperts * c.tokens);
    d.probs.resize(kExperts * c.tokens);
    std::vector<bool> routed(kExperts, false);
    for (int64_t t = 0; t < c.tokens; ++t) {
        std::vector<int32_t> order(kExperts);
        for (int64_t e = 0; e < kExperts; ++e) {
            order[e] = (int32_t) e;
        }
        std::shuffle(order.begin(), order.end(), rng);
        std::copy(order.begin(), order.end(), d.ids.begin() + t * kExperts);
        for (int64_t r = 0; r < kRoutes; ++r) {
            routed[order[r]] = true;
        }
        double sum = 0.0;
        std::vector<double> logits(kExperts);
        for (int64_t e = 0; e < kExperts; ++e) {
            logits[e] = 2.0 * n(rng);
            sum += std::exp(logits[e]);
        }
        for (int64_t e = 0; e < kExperts; ++e) {
            d.probs[t * kExperts + e] = (float) (std::exp(logits[e]) / sum);
        }
    }
    d.w_gate = make_experts(rng, GGML_TYPE_Q4_K, kHidden, kExpertFfn, routed);
    d.w_up   = make_experts(rng, GGML_TYPE_Q4_K, kHidden, kExpertFfn, routed);
    d.w_down = make_experts(rng, c.down_type, kExpertFfn, kHidden, routed);
    d.w_next = quantize_random(rng, GGML_TYPE_Q4_K, kHidden, kNextOut);
    return d;
}

// ---- the graph ----------------------------------------------------------------------------------------------------

struct Tensors {
    ggml_tensor *w_gate, *w_up, *w_down, *w_next, *ffn_norm, *next_norm;  // weights
    ggml_tensor *x, *attn, *z, *probs, *ids_all;                           // inputs
    ggml_tensor *h, *out, *proj;                                           // read back
};

void make_leaves(ggml_context * wctx, ggml_context * ictx, const Case & c, Tensors & t) {
    t.w_gate    = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, kHidden, kExpertFfn, kExperts);
    t.w_up      = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, kHidden, kExpertFfn, kExperts);
    t.w_down    = ggml_new_tensor_3d(wctx, c.down_type, kExpertFfn, kHidden, kExperts);
    t.w_next    = ggml_new_tensor_2d(wctx, GGML_TYPE_Q4_K, kHidden, kNextOut);
    t.ffn_norm  = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    t.next_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, kHidden);
    t.x         = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kHidden, c.tokens);
    t.attn      = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kHidden, c.tokens);
    t.z         = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kHidden, c.tokens);
    t.probs     = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, kExperts, c.tokens);
    t.ids_all   = ggml_new_tensor_2d(ictx, GGML_TYPE_I32, kExperts, c.tokens);
    for (ggml_tensor * in : { t.x, t.attn, t.z, t.probs, t.ids_all }) {
        ggml_set_input(in);
    }
}

ggml_cgraph * build_graph(ggml_context * gctx, const Case & c, Tensors & t) {
    const int64_t T      = c.tokens;
    ggml_tensor * res_in = ggml_add(gctx, t.x, t.attn);
    ggml_tensor * n      = ggml_mul(gctx, ggml_rms_norm(gctx, res_in, kEps), t.ffn_norm);
    // llama.cpp passes [n_embd, 1, tokens]; for one token that is n itself
    ggml_tensor * cur    = T == 1 ? n : ggml_reshape_3d(gctx, n, kHidden, 1, T);
    // the routes are the first kRoutes rows of a [kExperts, tokens] tensor, as llama.cpp passes its argsort
    ggml_tensor * ids    = ggml_view_2d(gctx, t.ids_all, kRoutes, T, t.ids_all->nb[1], 0);
    // the route weights come before the experts in the graph's order, as llama.cpp builds them (a fused match may
    // only read values whose producers are already scheduled)
    ggml_tensor * w = ggml_get_rows(gctx, ggml_reshape_3d(gctx, t.probs, 1, kExperts, T), ids);
    w               = ggml_reshape_2d(gctx, w, kRoutes, T);
    ggml_tensor * s = ggml_clamp(gctx, ggml_sum_rows(gctx, w), 6.103515625e-5f, INFINITY);
    w               = ggml_reshape_3d(gctx, ggml_div(gctx, w, s), 1, kRoutes, T);
    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, w);

    ggml_tensor * gate   = ggml_mul_mat_id(gctx, t.w_gate, cur, ids);
    ggml_tensor * up     = ggml_mul_mat_id(gctx, t.w_up, cur, ids);
    t.h                  = ggml_swiglu_split(gctx, gate, up);
    ggml_tensor * down   = ggml_mul_mat_id(gctx, t.w_down, t.h, ids);

    ggml_tensor * e   = ggml_mul(gctx, down, w);
    ggml_tensor * moe = nullptr;
    for (int64_t r = 0; r < kRoutes; ++r) {
        ggml_tensor * v = ggml_view_2d(gctx, e, kHidden, T, e->nb[2], r * e->nb[1]);
        moe             = moe == nullptr ? v : ggml_add(gctx, moe, v);
    }
    ggml_tensor * res = ggml_add(gctx, moe, res_in);
    ggml_tensor * n2  = ggml_mul(gctx, ggml_rms_norm(gctx, res, kEps), t.next_norm);
    t.proj            = ggml_mul_mat(gctx, t.w_next, n2);
    t.out             = ggml_add(gctx, res, t.z);
    ggml_set_output(t.proj);
    ggml_set_output(t.out);
    if (c.expose_h) {
        ggml_set_output(t.h);
    }
    ggml_build_forward_expand(graph, t.proj);
    ggml_build_forward_expand(graph, t.out);
    if (c.expose_h) {
        ggml_build_forward_expand(graph, t.h);
    }
    return graph;
}

// ---- path taken -----------------------------------------------------------------------------------------------------

struct KernelCount {
    int hip = 0, loom_q6k = 0, loom_q4k = 0;
};

// The add-on's kernel as kernel_definition_name() spells it: the readable name goes through the add-on's symbol hook
// (hip/hip-code-objects.h) to the name it registered; empty when this build has no such HIP kernel.
std::string g_hip_name;

void count_name(const std::string & name, KernelCount & k) {
    // names are family:name (kernel_definition_name)
    if (!g_hip_name.empty() && name == g_hip_name) {
        ++k.hip;
    } else if (name == std::string("qwen3_moe:") + kLoomQ6K) {
        ++k.loom_q6k;
    } else if (name == std::string("qwen3_moe:") + kLoomQ4K) {
        ++k.loom_q4k;
    }
}

// The expected counts for one case and arm; prints and returns false on a mismatch.
bool check_counts(const char * where, const Case & c, bool hip_arm, const KernelCount & k) {
    bool ok = true;
    if (c.hook_case) {
        ok = hip_arm ? (k.hip == 1 && k.loom_q6k == 0) : (k.hip == 0 && k.loom_q6k == 1);
    } else {
        ok = k.hip == 0 && k.loom_q6k == 0;
        if (c.down_type == GGML_TYPE_Q4_K) {
            ok = ok && k.loom_q4k == 1;  // the registration matched, so the hook was asked and declined
        }
    }
    std::printf("  %s %-14s hip=%d loom_q6k=%d loom_q4k=%d %s\n", where, c.name, k.hip, k.loom_q6k, k.loom_q4k,
                ok ? "OK" : "FAIL");
    return ok;
}

bool plan_case(const Case & c, bool hip_arm) {
    ggml_init_params p    = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx  = ggml_init(p);
    Tensors          t{};
    make_leaves(ctx, ctx, c, t);
    ggml_cgraph *                graph    = build_graph(ctx, c, t);
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    const bool                   scheduled = scheduler.schedule_graph(imported.graph, { "gfx1151" });
    if (!scheduled || !scheduler.plan().valid()) {
        std::printf("  plan   %-14s not scheduled: %s FAIL\n", c.name, scheduler.error().c_str());
        ggml_free(ctx);
        return false;
    }
    KernelCount k;
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        const ggml::hrx::KernelResolveResult resolved = ggml::hrx::resolve_kernel_definition(
            ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", dispatch.kernel.kernel_id);
        const std::string name = resolved.found() ? ggml::hrx::kernel_definition_name(*resolved.definition) : "";
        if (std::getenv("TEST_HRX_ROUTED_DOWN_HIP_VERBOSE") != nullptr) {
            const auto format = dispatch.kernel.compile_parameters.find("ggml.mul_mat.activation_format");
            std::printf("    %s%s%s\n", name.empty() ? "(unresolved)" : name.c_str(),
                        format != dispatch.kernel.compile_parameters.end() ? " activation_format=" : "",
                        format != dispatch.kernel.compile_parameters.end() ? format->second.c_str() : "");
        }
        count_name(name, k);
    }
    ggml_free(ctx);
    return check_counts("plan  ", c, hip_arm, k);
}

KernelCount count_dumped_kernels(const std::filesystem::path & dir) {
    KernelCount k;
    if (!std::filesystem::is_directory(dir)) {
        return k;
    }
    for (const auto & entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.path().filename() != "kernels.txt") {
            continue;
        }
        std::ifstream in(entry.path());
        std::string   phase, ordinal, name;
        std::string   line;
        while (std::getline(in, line)) {
            // "<phase> <ordinal> <family:name>[ deps=...]"
            const size_t a = line.find(' ');
            const size_t b = a == std::string::npos ? a : line.find(' ', a + 1);
            if (b == std::string::npos) {
                continue;
            }
            const size_t e = line.find(' ', b + 1);
            count_name(line.substr(b + 1, e == std::string::npos ? std::string::npos : e - b - 1), k);
        }
    }
    return k;
}

// ---- device run -----------------------------------------------------------------------------------------------------

struct Result {
    std::vector<float> h, out, proj;
};

bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Runs the case on `backend` (HRX first with the CPU as fallback, or the CPU alone) `repeats` times; returns false in
// *stable when the repeats differ.
Result run(ggml_backend_t hrx, ggml_backend_t cpu, const Case & c, const Data & d, int repeats, bool * stable) {
    ggml_init_params p    = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   wctx = ggml_init(p);
    ggml_context *   ictx = ggml_init(p);
    Tensors          t{};
    make_leaves(wctx, ictx, c, t);
    ggml_backend_t        home = hrx != nullptr ? hrx : cpu;
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, home);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors(ictx, home);
    REQUIRE(wbuf != nullptr && ibuf != nullptr);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(t.w_gate, d.w_gate.data(), 0, ggml_nbytes(t.w_gate));
    ggml_backend_tensor_set(t.w_up, d.w_up.data(), 0, ggml_nbytes(t.w_up));
    ggml_backend_tensor_set(t.w_down, d.w_down.data(), 0, ggml_nbytes(t.w_down));
    ggml_backend_tensor_set(t.w_next, d.w_next.data(), 0, ggml_nbytes(t.w_next));
    ggml_backend_tensor_set(t.ffn_norm, d.ffn_norm.data(), 0, ggml_nbytes(t.ffn_norm));
    ggml_backend_tensor_set(t.next_norm, d.next_norm.data(), 0, ggml_nbytes(t.next_norm));
    ggml_backend_tensor_set(t.x, d.x.data(), 0, ggml_nbytes(t.x));
    ggml_backend_tensor_set(t.attn, d.attn.data(), 0, ggml_nbytes(t.attn));
    ggml_backend_tensor_set(t.z, d.z.data(), 0, ggml_nbytes(t.z));
    ggml_backend_tensor_set(t.probs, d.probs.data(), 0, ggml_nbytes(t.probs));
    ggml_backend_tensor_set(t.ids_all, d.ids.data(), 0, ggml_nbytes(t.ids_all));

    ggml_init_params gp   = { 128 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   gctx = ggml_init(gp);
    ggml_cgraph *    graph = build_graph(gctx, c, t);

    std::vector<ggml_backend_t> backends;
    if (hrx != nullptr) {
        backends.push_back(hrx);
    }
    backends.push_back(cpu);
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 2048, false, true);
    REQUIRE(sched != nullptr);
    Result first;
    *stable = true;
    for (int i = 0; i < repeats; ++i) {
        REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        Result r;
        r.out.resize(ggml_nelements(t.out));
        r.proj.resize(ggml_nelements(t.proj));
        ggml_backend_tensor_get(t.out, r.out.data(), 0, ggml_nbytes(t.out));
        ggml_backend_tensor_get(t.proj, r.proj.data(), 0, ggml_nbytes(t.proj));
        if (c.expose_h) {
            r.h.resize(ggml_nelements(t.h));
            ggml_backend_tensor_get(t.h, r.h.data(), 0, ggml_nbytes(t.h));
        }
        if (i == 0) {
            first = std::move(r);
        } else if (!same_bits(first.out, r.out) || !same_bits(first.proj, r.proj) || !same_bits(first.h, r.h)) {
            *stable = false;
        }
    }
    if (hrx != nullptr) {
        int on_cpu = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (ggml_backend_sched_get_tensor_backend(sched, node) != hrx && node->op != GGML_OP_VIEW &&
                node->op != GGML_OP_RESHAPE) {
                std::printf("    on the CPU: %s (%s)\n", ggml_op_desc(node), node->name);
                ++on_cpu;
            }
        }
        std::printf("    %d compute node(s) off HRX\n", on_cpu);
    }
    ggml_backend_sched_free(sched);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_backend_buffer_free(ibuf);
    ggml_free(wctx);
    ggml_free(ictx);
    return first;
}

double nmse(const std::vector<double> & got, const std::vector<double> & ref) {
    double err = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        err += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0.0 ? err / den : err;
}

// out minus (x + attn + z): the routed sum alone, so the residual's own magnitude does not hide an error in it.
std::vector<double> routed_part(const std::vector<float> & out, const Data & d) {
    std::vector<double> r(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        r[i] = (double) out[i] - ((double) d.x[i] + d.attn[i] + d.z[i]);
    }
    return r;
}

std::vector<float> dequantize(ggml_type type, const uint8_t * data, int64_t count) {
    std::vector<float> f((size_t) count);
    ggml_get_type_traits(type)->to_float(data, f.data(), count);
    return f;
}

// The fp64 oracle for t1_q6k_oracle (one token), built from HRX's own h and residual.
//
// Residual: res_i = res_in_i + sum_r w_r sum_k W_r[i,k] h_r[k]. Any fp32 evaluation order of these K*R+R+1 terms
// (K = 768, R = 8) has error <= (K*R + R + 1) * u * S_i with u = 2^-24 and S_i = |res_in_i| + sum_r |w_r| sum_k
// |W_r[i,k] h_r[k]| (the standard recursive-summation bound, which also covers a tree or FMA order); the dequantized
// weight is d * scale * q, one more rounding per term. The bound used is 2 * (K*R + R + 8) * u * S_i, so 2.9e-4 * S_i;
// a dropped route, a wrong expert or a wrong row moves the value by a large fraction of S_i.
//
// Projection: n2 reaches proj as its q8 alternate (32-value blocks, step amax/127). Rounding to the nearest step
// leaves an error of variance step^2 / 12 per element, so proj_j's error has sigma_j^2 = sum_k Wn[j,k]^2 step(k)^2 / 12;
// the bound is 8 sigma_j (covers truncation instead of rounding, sigma x 2, and the maximum over 4096 rows, about
// 3.7 sigma) plus 1e-5 * sum_k |Wn[j,k] n2_k| for the projection's own f32 arithmetic.
bool check_oracle(const Data & d, const Result & got) {
    const double u = std::ldexp(1.0, -24);
    // route weights, as the graph computes them
    std::vector<double> w(kRoutes);
    double              wsum = 0.0;
    for (int64_t r = 0; r < kRoutes; ++r) {
        w[r] = d.probs[d.ids[r]];
        wsum += w[r];
    }
    for (double & v : w) {
        v /= std::max(wsum, 6.103515625e-5);
    }
    const size_t       expert_bytes = ggml_row_size(GGML_TYPE_Q6_K, kExpertFfn) * kHidden;
    std::vector<double> ref(kHidden), mag(kHidden);
    for (int64_t i = 0; i < kHidden; ++i) {
        const double res_in = (double) (d.x[i] + d.attn[i]);
        ref[i]              = res_in;
        mag[i]              = std::fabs(res_in);
    }
    for (int64_t r = 0; r < kRoutes; ++r) {
        const std::vector<float> W =
            dequantize(GGML_TYPE_Q6_K, d.w_down.data() + d.ids[r] * expert_bytes, kExpertFfn * kHidden);
        const float * h = got.h.data() + r * kExpertFfn;
        for (int64_t i = 0; i < kHidden; ++i) {
            double acc = 0.0, abs_acc = 0.0;
            for (int64_t k = 0; k < kExpertFfn; ++k) {
                const double t = (double) W[i * kExpertFfn + k] * h[k];
                acc += t;
                abs_acc += std::fabs(t);
            }
            ref[i] += w[r] * acc;
            mag[i] += std::fabs(w[r]) * abs_acc;
        }
    }
    const double factor   = 2.0 * (double) (kExpertFfn * kRoutes + kRoutes + 8) * u;
    double       worst    = 0.0;  // max |err| / bound
    int          failures = 0;
    for (int64_t i = 0; i < kHidden; ++i) {
        const double res = (double) got.out[i] - d.z[i];  // out = res + z, one more f32 rounding
        const double err = std::fabs(res - ref[i]);
        const double bound = factor * mag[i] + 2.0 * u * (std::fabs(res) + std::fabs((double) d.z[i]));
        worst = std::max(worst, err / bound);
        failures += err <= bound ? 0 : 1;
    }
    std::printf("    residual vs fp64 oracle: worst |err|/bound = %.3g over %lld values, %d over\n", worst,
                (long long) kHidden, failures);

    // projection, from HRX's own residual
    std::vector<double> res(kHidden), n2(kHidden);
    double              ss = 0.0;
    for (int64_t i = 0; i < kHidden; ++i) {
        res[i] = (double) got.out[i] - d.z[i];
        ss += res[i] * res[i];
    }
    const double scale = 1.0 / std::sqrt(ss / kHidden + kEps);
    for (int64_t i = 0; i < kHidden; ++i) {
        n2[i] = res[i] * scale * d.next_norm[i];
    }
    std::vector<double> step(kHidden);
    for (int64_t b = 0; b < kHidden / 32; ++b) {
        double amax = 0.0;
        for (int64_t k = 0; k < 32; ++k) {
            amax = std::max(amax, std::fabs(n2[b * 32 + k]));
        }
        for (int64_t k = 0; k < 32; ++k) {
            step[b * 32 + k] = amax / 127.0;
        }
    }
    const std::vector<float> Wn = dequantize(GGML_TYPE_Q4_K, d.w_next.data(), kHidden * kNextOut);
    double                   pworst    = 0.0;
    int                      pfailures = 0;
    for (int64_t j = 0; j < kNextOut; ++j) {
        double acc = 0.0, abs_acc = 0.0, var = 0.0;
        for (int64_t k = 0; k < kHidden; ++k) {
            const double wk = Wn[j * kHidden + k];
            acc += wk * n2[k];
            abs_acc += std::fabs(wk * n2[k]);
            var += wk * wk * step[k] * step[k] / 12.0;
        }
        const double bound = 8.0 * std::sqrt(var) + 1e-5 * abs_acc;
        const double err   = std::fabs((double) got.proj[j] - acc);
        pworst             = std::max(pworst, err / bound);
        pfailures += err <= bound ? 0 : 1;
    }
    std::printf("    proj vs fp64 oracle:     worst |err|/bound = %.3g over %lld values, %d over\n", pworst,
                (long long) kNextOut, pfailures);
    return failures == 0 && pfailures == 0;
}

// HRX against the CPU backend. Both sides quantize activations to 8 bits before the quantized products (the CPU to
// its Q8_K, HRX to q8 for the Q4_K products), which alone gives each product a relative nmse of about 1e-4 at these
// value ranges; three products in series, so the bound is 2e-3. The routed part is compared, not the residual, so
// the residual's magnitude does not hide an error.
bool check_against_cpu(const Data & d, const Result & got, const Result & cpu) {
    const double out_nmse  = nmse(routed_part(got.out, d), routed_part(cpu.out, d));
    const double proj_nmse = nmse(std::vector<double>(got.proj.begin(), got.proj.end()),
                                  std::vector<double>(cpu.proj.begin(), cpu.proj.end()));
    const bool   ok        = out_nmse <= 2e-3 && proj_nmse <= 2e-3;
    std::printf("    vs CPU backend: routed-part nmse %.3g, proj nmse %.3g (bound 2e-3)\n", out_nmse, proj_nmse);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep the lines printed before a failed REQUIRE
    const char *          arm       = nullptr;
    bool                  plan_only = false;
    bool                  cpu_only  = false;
    std::filesystem::path dump_root = "test-hrx-routed-down-hip-dump";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--arm" && i + 1 < argc) {
            arm = argv[++i];
        } else if (a == "--plan-only") {
            plan_only = true;
        } else if (a == "--cpu-only") {
            cpu_only = true;
        } else if (a == "--dump-dir" && i + 1 < argc) {
            dump_root = argv[++i];
        } else {
            arm = nullptr;
            break;
        }
    }
    if (arm == nullptr || (std::strcmp(arm, "hip") != 0 && std::strcmp(arm, "loom") != 0)) {
        std::fprintf(stderr, "usage: %s --arm hip|loom [--plan-only | --cpu-only] [--dump-dir DIR]\n", argv[0]);
        return 2;
    }
    const bool hip_arm = std::strcmp(arm, "hip") == 0;
    // before anything builds the dispatch registry: the add-on reads it once
    setenv("GGML_HRX_HIP_MOE_DOWN", hip_arm ? "1" : "0", 1);
    std::printf("test-hrx-routed-down-hip: arm %s (GGML_HRX_HIP_MOE_DOWN=%s)\n", arm, hip_arm ? "1" : "0");

    REQUIRE(ggml::hrx::find_dispatch_registry({ "gfx1151" }) != nullptr);  // registers the HIP kernels
    if (const ggml::hrx::KernelDefinition * hip =
            ggml::hrx::find_hip_kernel_definition(
                ggml::hrx::hip_kernel_ref(ggml::hrx::hip_addon_kernel_symbol(kHipKernel)).id)) {
        g_hip_name = ggml::hrx::kernel_definition_name(*hip);
    }
    std::printf("  HIP kernel %s: %s\n", kHipKernel, g_hip_name.empty() ? "not in this build" : g_hip_name.c_str());
    if (hip_arm && g_hip_name.empty()) {
        std::printf("test-hrx-routed-down-hip: --arm hip needs a build with the HIP add-on's %s, FAIL\n", kHipKernel);
        return 1;
    }

    int failures = 0;
    for (const Case & c : kCases) {
        failures += plan_case(c, hip_arm) ? 0 : 1;
    }
    if (plan_only) {
        REQUIRE(failures == 0);
        std::printf("test-hrx-routed-down-hip: plan cases OK (--plan-only, no backend initialised)\n");
        return 0;
    }

    if (cpu_only) {
        // the device part's data, graphs and checks on the CPU backend alone, no HRX backend: the oracle's bounds do
        // not hold for the CPU (it quantizes h before the down product), so its numbers are printed, not asserted
        ggml_backend_t cpu = ggml_backend_cpu_init();
        REQUIRE(cpu != nullptr);
        for (const Case & c : kCases) {
            std::printf("  cpu %s\n", c.name);
            const Data   d      = make_data(c);
            bool         stable = true;
            const Result a      = run(nullptr, cpu, c, d, 2, &stable);
            std::printf("    two runs bit-identical: %s\n", stable ? "yes" : "NO");
            if (c.expose_h) {
                check_oracle(d, a);
            }
            check_against_cpu(d, a, a);
            failures += stable ? 0 : 1;
        }
        ggml_backend_free(cpu);
        REQUIRE(failures == 0);
        std::printf("test-hrx-routed-down-hip: plan cases OK, CPU-only pass ran\n");
        return 0;
    }

    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        REQUIRE(failures == 0);
        std::printf("test-hrx-routed-down-hip: plan cases OK; no HRX0 device, the device cases are skipped\n");
        return 77;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);
    for (const Case & c : kCases) {
        std::printf("  device %s\n", c.name);
        const Data                  d   = make_data(c);
        const std::filesystem::path dir = dump_root / (std::string(arm) + "-" + c.name);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        setenv("GGML_HRX_DUMP_COMMAND_PROGRAM_DIR", dir.c_str(), 1);
        bool         stable = true;
        const Result got    = run(hrx, cpu, c, d, 3, &stable);
        unsetenv("GGML_HRX_DUMP_COMMAND_PROGRAM_DIR");
        bool ok = check_counts("device", c, hip_arm, count_dumped_kernels(dir));
        std::printf("    three runs bit-identical: %s\n", stable ? "yes" : "NO");
        ok = ok && stable;
        if (c.expose_h) {
            ok = check_oracle(d, got) && ok;
        }
        bool         cpu_stable = true;
        const Result ref        = run(nullptr, cpu, c, d, 1, &cpu_stable);
        ok                      = check_against_cpu(d, got, ref) && ok;
        std::printf("  device %-14s %s\n", c.name, ok ? "OK" : "FAIL");
        failures += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    REQUIRE(failures == 0);
    std::printf("test-hrx-routed-down-hip: arm %s, all cases OK\n", arm);
    return 0;
}
