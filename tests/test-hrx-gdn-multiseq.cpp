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

// Multi-sequence gated delta net on HRX: one Qwen3.5 / Qwen3.8 linear-attention layer (hrx-gdn-layer.h) with
// 1..4 sequences per ubatch, in both GDN modes (the n_seqs-wide scan, and GGML_HRX_GDN_PER_SEQUENCE=1).
//
// Part 1 (any host, no GPU): the gfx1151 dispatch plan must claim the q/k L2_NORM and the GATED_DELTA_NET
// per op, take every node, keep every dispatch binding inside its value, and give a command program that
// verifies.
// Part 2 (HRX0 on gfx1151 / gfx1100): the layer output and both caches must match the CPU backend (normalized
// MSE, as test-backend-ops), with s_copy reversed against the rows written (sequence s reads row
// kv_head + n_seqs - 1 - s). Each case runs `repeats` times on HRX and every repeat must give identical bytes.
//
// Usage: test-hrx-gdn-multiseq [--plan-only] [--verbose] [--repeats N] [--filter SUBSTRING]
//                              [--mode native|per-seq|both]   (default both; one mode per process isolates a GPU fault)

#include "dispatch/command-program-resolver.h"
#include "dispatch/command-program.h"
#include "dispatch/dispatch-scheduler.h"
#include "fused-context-claim.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "graph/graph.h"
#include "hrx-gdn-layer.h"
#include "kernel-corpus/kernel-corpus.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

using hrx_gdn_layer::Case;
using hrx_gdn_layer::Layer;

static constexpr double kMaxNmse = 5e-4;

struct Built {
    ggml_context * ctx   = nullptr;
    ggml_cgraph *  graph = nullptr;
    Layer          layer;
};

static Built build(const Case & c) {
    const int64_t    nodes  = hrx_gdn_layer::graph_size();
    ggml_init_params params = { static_cast<size_t>(nodes) * ggml_tensor_overhead() +
                                    ggml_graph_overhead_custom(nodes, false),
                                nullptr, true };
    Built b;
    b.ctx = ggml_init(params);
    REQUIRE(b.ctx != nullptr);
    b.graph = ggml_new_graph_custom(b.ctx, nodes, false);
    b.layer = hrx_gdn_layer::build_layer(b.ctx, b.graph, c);
    return b;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    return resolved.found() ? ggml::hrx::kernel_definition_name(*resolved.definition) : "?";
}

static std::string case_name(const Case & c) {
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), "%s T=%lld S=%lld K=%lld rows=%lld+%lld", c.shape.name,
                  (long long) c.tokens, (long long) c.seqs, (long long) c.snapshots, (long long) c.kv_head,
                  (long long) c.mem_size);
    return buffer;
}

static void set_mode(bool per_sequence) {
    if (per_sequence) {
        setenv("GGML_HRX_GDN_PER_SEQUENCE", "1", 1);
    } else {
        unsetenv("GGML_HRX_GDN_PER_SEQUENCE");
    }
}

// Part 1: plan for gfx1151. Returns the number of problems.
static int check_plan(const Case & c, bool per_sequence, bool verbose) {
    set_mode(per_sequence);
    Built b        = build(c);
    int   problems = 0;
    // the per-op claim the scheduler asks before it places a node on HRX
    for (int i = 0; i < ggml_graph_n_nodes(b.graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(b.graph, i);
        if ((node->op == GGML_OP_L2_NORM || node->op == GGML_OP_GATED_DELTA_NET) &&
            !ggml::hrx::fused_context_claim(node)) {
            std::printf("  %s: %s [%lld,%lld,%lld,%lld] not claimed for HRX\n", case_name(c).c_str(),
                        ggml_op_desc(node), (long long) node->ne[0], (long long) node->ne[1],
                        (long long) node->ne[2], (long long) node->ne[3]);
            ++problems;
        }
    }

    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*b.graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics) || !scheduler.plan().valid()) {
        std::printf("  %s: no HRX plan: %s\n", case_name(c).c_str(), diagnostics.unsupported_message.c_str());
        ggml_free(b.ctx);
        return problems + 1;
    }
    const ggml::hrx::CommandPlan & plan = scheduler.plan();
    std::string                    gdn_kernels;
    int                            gdn_dispatches = 0;
    for (const ggml::hrx::Dispatch & dispatch : plan.dispatches) {
        const std::string name = kernel_name_for_id(dispatch.kernel.kernel_id);
        if (name.find("gated_delta_net") != std::string::npos) {
            ++gdn_dispatches;
            if (gdn_kernels.find(name) == std::string::npos) {
                gdn_kernels += (gdn_kernels.empty() ? "" : ",") + name.substr(name.find(':') + 1);
            }
        }
        for (const ggml::hrx::DispatchBinding & binding : dispatch.bindings) {
            const ggml::hrx::Value * value = imported.graph.values().find(binding.value);
            if (value == nullptr || binding.layout != ggml::hrx::kNativeWeightLayout) {
                continue;  // plan transients and materialized weight layouts
            }
            if (binding.offset > value->byte_count || binding.length > value->byte_count - binding.offset) {
                std::printf("  %s: %s binds [%zu, +%zu) of a %zu-byte value\n", case_name(c).c_str(), name.c_str(),
                            binding.offset, binding.length, value->byte_count);
                ++problems;
            }
        }
        if (verbose) {
            std::string params;
            for (const auto & [key, val] : dispatch.kernel.compile_parameters) {
                if (key.find("sequence_count") != std::string::npos || key.find(".n_s") != std::string::npos ||
                    key.find("token_count") != std::string::npos) {
                    params += " " + key + "=" + val;
                }
            }
            std::printf("    %s%s\n", name.c_str(), params.c_str());
        }
    }
    const ggml::hrx::CommandProgram commands =
        ggml::hrx::build_command_program(imported.graph, plan, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151");
    if (!commands.valid() ||
        !ggml::hrx::verify_command_program(commands, ggml::hrx::get_qwen_kernel_corpus(), "gfx1151").valid()) {
        std::printf("  %s: command program does not verify\n", case_name(c).c_str());
        ++problems;
    }
    if (per_sequence && c.seqs > 1 && c.snapshots == 1 && c.tokens <= 16 &&
        gdn_kernels.find("_inplace") == std::string::npos) {
        std::printf("  %s: per-sequence mode did not take the in-place scan\n", case_name(c).c_str());
        ++problems;
    }
    std::printf("  plan %-9s %-44s %3zu dispatches, %d gdn (%s) %s\n", per_sequence ? "per-seq" : "native",
                case_name(c).c_str(), plan.dispatches.size(), gdn_dispatches, gdn_kernels.c_str(),
                problems == 0 ? "OK" : "FAIL");
    ggml_free(b.ctx);
    return problems;
}

// Part 2: HRX vs CPU on the same random data.
struct Outputs {
    std::vector<float> out;
    std::vector<float> conv_cache;
    std::vector<float> ssm_cache;
};

static void fill(ggml_tensor * tensor, std::mt19937 & rng, float scale) {
    std::uniform_real_distribution<float> uniform(-scale, scale);
    std::vector<float>                    values(static_cast<size_t>(ggml_nelements(tensor)));
    for (float & v : values) {
        v = uniform(rng);
    }
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
        return;
    }
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    ggml_quantize_chunk(tensor->type, values.data(), quantized.data(), 0, tensor->ne[1], tensor->ne[0], nullptr);
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
}

static bool run(ggml_backend_t backend, const Case & c, uint32_t seed, Outputs & result) {
    Built                 b      = build(c);
    Layer &               l      = b.layer;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    REQUIRE(buffer != nullptr);
    std::mt19937 rng(seed);
    fill(l.x, rng, 1.0f);
    for (ggml_tensor * w : { l.w_qkv, l.w_z, l.w_alpha, l.w_beta, l.w_out }) {
        fill(w, rng, 0.05f);
    }
    fill(l.dt, rng, 1.0f);
    fill(l.conv_w, rng, 0.5f);
    fill(l.norm_w, rng, 1.0f);
    fill(l.conv_cache, rng, 1.0f);
    fill(l.ssm_cache, rng, 0.05f);
    std::vector<float> a(static_cast<size_t>(c.shape.v_heads));
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = hrx_gdn_layer::a_value(static_cast<int64_t>(i));
    }
    ggml_backend_tensor_set(l.a, a.data(), 0, a.size() * sizeof(float));
    std::vector<int32_t> s_copy(static_cast<size_t>(c.seqs));
    for (int64_t s = 0; s < c.seqs; ++s) {
        s_copy[static_cast<size_t>(s)] = hrx_gdn_layer::s_copy_value(c, s);
    }
    ggml_backend_tensor_set(l.s_copy, s_copy.data(), 0, s_copy.size() * sizeof(int32_t));

    const bool ok = ggml_backend_graph_compute(backend, b.graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        auto get = [](ggml_tensor * t, std::vector<float> & v) {
            v.resize(static_cast<size_t>(ggml_nelements(t)));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        };
        get(l.out, result.out);
        get(l.conv_cache, result.conv_cache);
        get(l.ssm_cache, result.ssm_cache);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(b.ctx);
    return ok;
}

static double nmse(const std::vector<float> & got, const std::vector<float> & expected) {
    if (got.size() != expected.size()) {
        return 1e30;
    }
    double err = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = static_cast<double>(got[i]) - expected[i];
        err += std::isfinite(diff) ? diff * diff : 1e30;
        ref += static_cast<double>(expected[i]) * expected[i];
    }
    return ref > 0.0 ? err / ref : err;
}

static bool same_bytes(const Outputs & lhs, const Outputs & rhs) {
    auto eq = [](const std::vector<float> & a, const std::vector<float> & b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    };
    return eq(lhs.out, rhs.out) && eq(lhs.conv_cache, rhs.conv_cache) && eq(lhs.ssm_cache, rhs.ssm_cache);
}

int main(int argc, char ** argv) {
    bool        plan_only = false;
    bool        verbose   = false;
    int         repeats   = 3;
    std::string filter;
    std::string mode = "both";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--plan-only") == 0) {
            plan_only = true;
        } else if (std::strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        }
    }

    std::vector<Case> cases;
    for (const hrx_gdn_layer::Shape & shape :
         { hrx_gdn_layer::kQwen38_27B, hrx_gdn_layer::kQwen35_A3B, hrx_gdn_layer::kTiny }) {
        for (int64_t seqs : { 1, 2, 3, 4 }) {
            for (int64_t tokens : { 1, 2, 8, 32 }) {
                cases.push_back({ shape, tokens, seqs, 4, 0, 1 });
            }
        }
        cases.push_back({ shape, 1, 2, 8, 5, 1 });   // cache rows [5, 7) of 8, s_copy {6, 5}
        cases.push_back({ shape, 4, 4, 8, 3, 1 });   // rows [3, 7)
        cases.push_back({ shape, 3, 2, 4, 0, 4 });   // MTP-style rollback snapshots, two sequences
    }
    if (!filter.empty()) {
        std::vector<Case> kept;
        for (const Case & c : cases) {
            if (case_name(c).find(filter) != std::string::npos) {
                kept.push_back(c);
            }
        }
        cases.swap(kept);
    }

    std::vector<bool> modes;
    if (mode == "native" || mode == "both") {
        modes.push_back(false);
    }
    if (mode == "per-seq" || mode == "both") {
        modes.push_back(true);
    }
    REQUIRE(!modes.empty());

    int problems = 0;
    std::printf("test-hrx-gdn-multiseq: plans for gfx1151\n");
    for (bool per_sequence : modes) {
        for (const Case & c : cases) {
            problems += check_plan(c, per_sequence, verbose);
        }
    }

    if (!plan_only) {
        ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
        if (device == nullptr) {
            ggml_backend_load_all();
            device = ggml_backend_dev_by_name("HRX0");
        }
        const char * description = device != nullptr ? ggml_backend_dev_description(device) : "";
        if (device == nullptr ||
            (std::strstr(description, "gfx1151") == nullptr && std::strstr(description, "gfx1100") == nullptr)) {
            std::printf("test-hrx-gdn-multiseq: no gfx1151/gfx1100 HRX0 device (%s), runtime part skipped\n",
                        description);
        } else {
            ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
            ggml_backend_t cpu = ggml_backend_cpu_init();
            REQUIRE(hrx != nullptr && cpu != nullptr);
            uint32_t seed = 1;
            for (const Case & c : cases) {
                Outputs expected;
                REQUIRE(run(cpu, c, seed, expected));
                for (bool per_sequence : modes) {
                    set_mode(per_sequence);
                    Outputs first;
                    bool    ok        = true;
                    bool    identical = true;
                    for (int r = 0; r < repeats && ok; ++r) {
                        Outputs got;
                        ok = run(hrx, c, seed, got);
                        if (ok && r == 0) {
                            first = std::move(got);
                        } else if (ok) {
                            identical = identical && same_bytes(first, got);
                        }
                    }
                    if (!ok) {
                        std::printf("  run  %-9s %-44s HRX compute failed\n", per_sequence ? "per-seq" : "native",
                                    case_name(c).c_str());
                        ++problems;
                        continue;
                    }
                    const double e_out  = nmse(first.out, expected.out);
                    const double e_conv = nmse(first.conv_cache, expected.conv_cache);
                    const double e_ssm  = nmse(first.ssm_cache, expected.ssm_cache);
                    const bool   pass   = e_out <= kMaxNmse && e_conv <= kMaxNmse && e_ssm <= kMaxNmse && identical;
                    std::printf("  run  %-9s %-44s out=%.3g conv=%.3g ssm=%.3g repeats=%d %s %s\n",
                                per_sequence ? "per-seq" : "native", case_name(c).c_str(), e_out, e_conv, e_ssm,
                                repeats, identical ? "identical" : "DIFFER", pass ? "OK" : "FAIL");
                    problems += pass ? 0 : 1;
                }
                ++seed;
            }
            set_mode(false);
            ggml_backend_free(cpu);
            ggml_backend_free(hrx);
        }
    }
    std::printf("test-hrx-gdn-multiseq: %zu cases x %zu modes, %d problems\n", cases.size(), modes.size(), problems);
    return problems == 0 ? 0 : 1;
}
