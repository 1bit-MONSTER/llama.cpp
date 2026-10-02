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

// gpt-oss's MoE block with its MXFP4 experts in CPU memory and its biases and routing in HRX memory, as llama.cpp
// lays out gpt-oss when HRX declines the expert MUL_MAT_ID at load time:
//   gate = add_id(mul_mat_id(W_gate, x, ids), b_gate, ids)     up = add_id(mul_mat_id(W_up, x, ids), b_up, ids)
//   h = swiglu_oai(gate, up, 1.702, 7)                          out = add_id(mul_mat_id(W_down, h, ids), b_down, ids)
// ids is a [n_used of n_expert] view of a [n_expert, tokens] tensor in HRX memory, as llama.cpp passes the
// argsort and gpt-oss computes it on HRX. Two passes: experts in a CPU-only buffer (CPU_REPACK, the layout above,
// where every MUL_MAT_ID stays on the CPU), and experts in a plain host buffer (HRX may read them). Checks the
// result against the same graph on the CPU alone, and the placement guard (moe-placement-guard.cpp): an ADD_ID or
// SWIGLU_OAI never runs on HRX while the MUL_MAT_ID it follows runs on the CPU (engine #286).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static constexpr int64_t kHidden  = 256;
static constexpr int64_t kFfn     = 256;
static constexpr int64_t kExperts = 8;
static constexpr int64_t kUsed    = 4;

struct Model {
    std::vector<uint8_t> w_gate, w_up, w_down;  // MXFP4
    std::vector<float>   b_gate, b_up, b_down, x;
    std::vector<int32_t> ids;
};

static Model make_model(int64_t tokens) {
    std::mt19937                          rng(1234);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    auto fill = [&](std::vector<float> & v, size_t n) {
        v.resize(n);
        for (float & f : v) {
            f = u(rng);
        }
    };
    auto experts = [&](std::vector<uint8_t> & out, int64_t cols, int64_t rows) {
        std::vector<float> f;
        fill(f, cols * rows * kExperts);
        out.resize(ggml_row_size(GGML_TYPE_MXFP4, cols) * rows * kExperts);
        ggml_quantize_chunk(GGML_TYPE_MXFP4, f.data(), out.data(), 0, rows * kExperts, cols, nullptr);
    };
    Model m;
    experts(m.w_gate, kHidden, kFfn);
    experts(m.w_up, kHidden, kFfn);
    experts(m.w_down, kFfn, kHidden);
    fill(m.b_gate, kFfn * kExperts);
    fill(m.b_up, kFfn * kExperts);
    fill(m.b_down, kHidden * kExperts);
    fill(m.x, kHidden * tokens);
    m.ids.resize(kExperts * tokens);
    for (int64_t t = 0; t < tokens; ++t) {
        std::vector<int32_t> order(kExperts);
        for (int64_t e = 0; e < kExperts; ++e) {
            order[e] = (int32_t) e;
        }
        std::shuffle(order.begin(), order.end(), rng);
        std::copy(order.begin(), order.end(), m.ids.begin() + t * kExperts);
    }
    return m;
}

// The CPU's CPU_REPACK buffer type, or nullptr when this build has none.
static ggml_backend_buffer_type_t cpu_repack_buft() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        return nullptr;
    }
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (get_extra == nullptr) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * b = get_extra(dev); b != nullptr && *b != nullptr; ++b) {
        if (std::strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) {
            return *b;
        }
    }
    return nullptr;
}

// Runs the block; with hrx != nullptr the experts go to expert_buft and the biases and routing to HRX memory.
// Returns false in *guard_ok when an ADD_ID / SWIGLU_OAI ran on HRX while its MUL_MAT_ID ran elsewhere.
static std::vector<float> run(ggml_backend_t hrx, ggml_backend_t cpu, ggml_backend_buffer_type_t expert_buft,
                              const Model & m, int64_t tokens, bool * guard_ok) {
    ggml_init_params wp   = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context *   ectx = ggml_init(wp);  // experts
    ggml_context *   xctx = ggml_init(wp);  // the activations, in CPU memory
    ggml_context *   bctx = ggml_init(wp);  // biases and routing (HRX-resident when hrx != nullptr)
    ggml_tensor *    w_gate = ggml_new_tensor_3d(ectx, GGML_TYPE_MXFP4, kHidden, kFfn, kExperts);
    ggml_tensor *    w_up   = ggml_new_tensor_3d(ectx, GGML_TYPE_MXFP4, kHidden, kFfn, kExperts);
    ggml_tensor *    w_down = ggml_new_tensor_3d(ectx, GGML_TYPE_MXFP4, kFfn, kHidden, kExperts);
    ggml_tensor *    x      = ggml_new_tensor_3d(xctx, GGML_TYPE_F32, kHidden, 1, tokens);
    ggml_tensor *    b_gate = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, kFfn, kExperts);
    ggml_tensor *    b_up   = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, kFfn, kExperts);
    ggml_tensor *    b_down = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, kHidden, kExperts);
    ggml_tensor *    ids_all = ggml_new_tensor_2d(bctx, GGML_TYPE_I32, kExperts, tokens);
    ggml_set_input(x);
    ggml_set_input(ids_all);
    ggml_backend_buffer_t ebuf = ggml_backend_alloc_ctx_tensors_from_buft(
        ectx, hrx != nullptr && expert_buft != nullptr ? expert_buft : ggml_backend_get_default_buffer_type(cpu));
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors(xctx, cpu);
    ggml_backend_buffer_t bbuf = ggml_backend_alloc_ctx_tensors(bctx, hrx != nullptr ? hrx : cpu);
    REQUIRE(ebuf != nullptr && xbuf != nullptr && bbuf != nullptr);
    ggml_backend_buffer_set_usage(ebuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(bbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w_gate, m.w_gate.data(), 0, ggml_nbytes(w_gate));
    ggml_backend_tensor_set(w_up, m.w_up.data(), 0, ggml_nbytes(w_up));
    ggml_backend_tensor_set(w_down, m.w_down.data(), 0, ggml_nbytes(w_down));
    ggml_backend_tensor_set(b_gate, m.b_gate.data(), 0, ggml_nbytes(b_gate));
    ggml_backend_tensor_set(b_up, m.b_up.data(), 0, ggml_nbytes(b_up));
    ggml_backend_tensor_set(b_down, m.b_down.data(), 0, ggml_nbytes(b_down));
    ggml_backend_tensor_set(x, m.x.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids_all, m.ids.data(), 0, ggml_nbytes(ids_all));

    ggml_init_params gp   = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   gctx = ggml_init(gp);
    ggml_tensor *    ids  = ggml_view_2d(gctx, ids_all, kUsed, tokens, ids_all->nb[1], 0);
    ggml_tensor *    mm_gate = ggml_mul_mat_id(gctx, w_gate, x, ids);
    ggml_tensor *    mm_up   = ggml_mul_mat_id(gctx, w_up, x, ids);
    ggml_tensor *    gate    = ggml_add_id(gctx, mm_gate, b_gate, ids);
    ggml_tensor *    up      = ggml_add_id(gctx, mm_up, b_up, ids);
    ggml_tensor *    h       = ggml_swiglu_oai(gctx, gate, up, 1.702f, 7.0f);
    ggml_tensor *    mm_down = ggml_mul_mat_id(gctx, w_down, h, ids);
    ggml_tensor *    out     = ggml_add_id(gctx, mm_down, b_down, ids);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, out);

    std::vector<ggml_backend_t> backends;
    if (hrx != nullptr) {
        backends.push_back(hrx);
    }
    backends.push_back(cpu);
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 1024, false, true);
    REQUIRE(sched != nullptr);
    REQUIRE(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    *guard_ok = true;
    if (hrx != nullptr) {
        auto on_hrx = [&](ggml_tensor * t) { return ggml_backend_sched_get_tensor_backend(sched, t) == hrx; };
        std::printf("  placement:");
        for (ggml_tensor * node : { mm_gate, gate, mm_up, up, h, mm_down, out }) {
            std::printf(" %s=%s", ggml_op_desc(node), on_hrx(node) ? "HRX" : "CPU");
        }
        std::printf("\n");
        const bool tail_ok = (!on_hrx(gate) || on_hrx(mm_gate)) && (!on_hrx(up) || on_hrx(mm_up)) &&
                             (!on_hrx(h) || (on_hrx(mm_gate) && on_hrx(mm_up))) && (!on_hrx(out) || on_hrx(mm_down));
        *guard_ok = tail_ok;
    }
    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
    ggml_backend_sched_free(sched);
    ggml_free(gctx);
    ggml_backend_buffer_free(ebuf);
    ggml_backend_buffer_free(xbuf);
    ggml_backend_buffer_free(bbuf);
    ggml_free(ectx);
    ggml_free(xctx);
    ggml_free(bctx);
    return result;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep the lines printed before a failed REQUIRE
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-moe-split: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(hrx != nullptr && cpu != nullptr);
    ggml_backend_buffer_type_t repack = cpu_repack_buft();
    if (repack == nullptr) {
        std::printf("test-hrx-moe-split: no CPU_REPACK buffer type, the CPU-only expert pass is skipped\n");
    }
    int failures = 0;
    for (ggml_backend_buffer_type_t buft : { repack, ggml_backend_get_default_buffer_type(cpu) }) {
        if (buft == nullptr) {
            continue;
        }
        for (int64_t tokens : { 1, 3, 12, 40 }) {
            const Model        m = make_model(tokens);
            bool               ref_ok = true, guard_ok = true;
            std::vector<float> expected = run(nullptr, cpu, nullptr, m, tokens, &ref_ok);
            std::vector<float> got      = run(hrx, cpu, buft, m, tokens, &guard_ok);
            double err = 0.0, ref = 0.0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double d = (double) got[i] - expected[i];
                err += d * d;
                ref += (double) expected[i] * expected[i];
            }
            const double nmse = ref > 0.0 ? err / ref : err;
            const bool   ok   = nmse <= 5e-4 && guard_ok;
            std::printf("experts=%-10s tokens=%2lld nmse=%.3g guard=%s %s\n", ggml_backend_buft_name(buft),
                        (long long) tokens, nmse, guard_ok ? "ok" : "VIOLATED", ok ? "OK" : "FAIL");
            failures += ok ? 0 : 1;
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    REQUIRE(failures == 0);
    std::printf("test-hrx-moe-split: all cases OK\n");
    return 0;
}
