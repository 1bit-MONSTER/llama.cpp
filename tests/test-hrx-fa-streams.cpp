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

// engine#2918: FLASH_ATTN_EXT over several KV streams, built the way llama.cpp builds it for a context with
// n_seq_max > 1 and no unified KV cache (llm_graph_context::build_attn_mha + llama_kv_cache::get_k/get_v), run on
// HRX0 and on the CPU, compared element by element. test-backend-ops' multi-stream cases use a different K/V view
// (a doubled head dim), so they never reach the layout the model graph has.
//
// Exit 0 = every case agrees, 1 = a case disagrees or HRX0 could not compute it, 77 = no HRX0 device. Without the
// onebit.flash_attention_streams matcher every case fails (measured: GGML_HRX_DISABLE_DISPATCH and the pre-change libs).

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// D/Dv: Q-K and V head sizes. q_gap: extra query rows between streams (a q view whose stream planes are not packed).
// broadcast_mask: one mask plane for every stream (ne[3] == 1). sinks: per-head attention sinks.
// required: the case has the layout llama builds and the matcher admits, so HRX0 must claim it; an optional case may
// be declined (CPU fallback), but if HRX0 claims it, it must compute it and agree with the CPU.
struct Case {
    int64_t D, Dv, H, Hkv, S, T, n_kv, kv_size;
    int64_t q_gap          = 0;
    bool    broadcast_mask = false;
    bool    sinks          = false;
    bool    required       = true;
};

struct Built {
    ggml_context * ctx = nullptr;
    ggml_tensor *  q = nullptr, *k_cache = nullptr, *v_cache = nullptr, *mask = nullptr, *sinks = nullptr, *out = nullptr;
    ggml_cgraph *  graph = nullptr;
};

Built build(const Case & c) {
    Built b;
    ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    b.ctx = ggml_init(params);
    // q as the model has it before build_attn_mha: [D, H, T*S], viewed per stream and permuted to [D, T, H, S]
    b.q = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, c.D, c.H, (c.T + c.q_gap) * c.S);
    ggml_tensor * q = ggml_view_4d(b.ctx, b.q, c.D, c.H, c.T, c.S, b.q->nb[1], b.q->nb[2], b.q->nb[2] * (c.T + c.q_gap), 0);
    q = ggml_permute(b.ctx, q, 0, 2, 1, 3);
    // the K/V cache: one [Hkv*D, kv_size] (V: [Hkv*Dv, kv_size]) plane per stream; the graph views n_kv cells of
    // every stream
    b.k_cache = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F16, c.Hkv * c.D, c.kv_size, c.S);
    b.v_cache = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F16, c.Hkv * c.Dv, c.kv_size, c.S);
    const size_t f16 = ggml_type_size(GGML_TYPE_F16);
    ggml_tensor * k  = ggml_view_4d(b.ctx, b.k_cache, c.D, c.Hkv, c.n_kv, c.S, c.D * f16, c.Hkv * c.D * f16,
                                    c.Hkv * c.D * c.kv_size * f16, 0);
    ggml_tensor * v  = ggml_view_4d(b.ctx, b.v_cache, c.Dv, c.Hkv, c.n_kv, c.S, c.Dv * f16, c.Hkv * c.Dv * f16,
                                    c.Hkv * c.Dv * c.kv_size * f16, 0);
    k = ggml_permute(b.ctx, k, 0, 2, 1, 3);
    v = ggml_permute(b.ctx, v, 0, 2, 1, 3);
    b.mask = ggml_new_tensor_4d(b.ctx, GGML_TYPE_F16, c.n_kv, c.T, 1, c.broadcast_mask ? 1 : c.S);
    b.out  = ggml_flash_attn_ext(b.ctx, q, k, v, b.mask, 1.0f / std::sqrt(static_cast<float>(c.D)), 0.0f, 0.0f);
    if (c.sinks) {
        b.sinks = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, c.H);
        ggml_flash_attn_ext_add_sinks(b.out, b.sinks);
    }
    ggml_prec_set_acc(b.out, GGML_PREC_F32);
    b.graph = ggml_new_graph(b.ctx);
    ggml_build_forward_expand(b.graph, b.out);
    return b;
}

struct Inputs {
    std::vector<float>       q, sinks;
    std::vector<ggml_fp16_t> k, v, mask;
};

Inputs make_inputs(const Case & c, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Inputs                                in;
    in.q.resize(static_cast<size_t>(c.D * c.H * (c.T + c.q_gap) * c.S));
    for (float & x : in.q) {
        x = dist(rng);
    }
    const size_t k_count = static_cast<size_t>(c.Hkv * c.D * c.kv_size * c.S);
    const size_t v_count = static_cast<size_t>(c.Hkv * c.Dv * c.kv_size * c.S);
    in.k.resize(k_count);
    in.v.resize(v_count);
    for (size_t i = 0; i < k_count; ++i) {
        in.k[i] = ggml_fp32_to_fp16(dist(rng));
    }
    for (size_t i = 0; i < v_count; ++i) {
        in.v[i] = ggml_fp32_to_fp16(dist(rng));
    }
    if (c.sinks) {
        in.sinks.resize(static_cast<size_t>(c.H));
        for (float & x : in.sinks) {
            x = 4.0f * dist(rng);
        }
    }
    // a causal mask whose length differs per stream, so a stream that read another stream's plane would show
    const int64_t planes = c.broadcast_mask ? 1 : c.S;
    in.mask.resize(static_cast<size_t>(c.n_kv * c.T * planes));
    for (int64_t s = 0; s < planes; ++s) {
        const int64_t filled = c.n_kv - (s * 37) % (c.n_kv / 2);  // cells in use in this stream
        for (int64_t t = 0; t < c.T; ++t) {
            const int64_t last = filled - c.T + t;  // the query's own position
            for (int64_t j = 0; j < c.n_kv; ++j) {
                const bool  visible = j <= last && j < filled;
                in.mask[(s * c.T + t) * c.n_kv + j] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
        }
    }
    return in;
}

bool run(ggml_backend_t backend, const Case & c, const Inputs & in, std::vector<float> & out, bool & claimed) {
    Built b = build(c);
    // Ask before allocating, as the scheduler does: HRX0 answers yes for any node whose output already sits in its
    // memory (ggml-hrx.cpp, "a node already placed in HRX device memory"), which says nothing about the matchers.
    claimed = ggml_backend_supports_op(backend, b.out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (buffer == nullptr) {
        ggml_free(b.ctx);
        return false;
    }
    ggml_backend_tensor_set(b.q, in.q.data(), 0, in.q.size() * sizeof(float));
    ggml_backend_tensor_set(b.k_cache, in.k.data(), 0, in.k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b.v_cache, in.v.data(), 0, in.v.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b.mask, in.mask.data(), 0, in.mask.size() * sizeof(ggml_fp16_t));
    if (b.sinks != nullptr) {
        ggml_backend_tensor_set(b.sinks, in.sinks.data(), 0, in.sinks.size() * sizeof(float));
    }
    const bool ok = claimed && ggml_backend_graph_compute(backend, b.graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize(static_cast<size_t>(ggml_nelements(b.out)));
        ggml_backend_tensor_get(b.out, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(b.ctx);
    return ok;
}

}  // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HRX0");
    if (dev == nullptr) {
        std::printf("skip: no HRX0 device\n");
        return 77;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (hrx == nullptr || cpu == nullptr) {
        std::printf("FAIL: backend init\n");
        return 1;
    }

    // { D, Dv, H, Hkv, S, T, n_kv, kv_size, q_gap, broadcast_mask, sinks, required }
    const Case cases[] = {
        // LFM3-1.7B's attention (head 128, 16 query heads over 4 KV heads), plus a 64-head-size GQA shape
        { 128, 128, 16, 4, 4, 1, 512, 512 },
        { 128, 128, 16, 4, 4, 1, 1024, 4096 },
        { 128, 128, 16, 4, 2, 7, 256, 768 },
        { 128, 128, 16, 4, 4, 64, 512, 512 },
        { 128, 128, 16, 4, 4, 300, 1024, 2048 },
        { 128, 128, 16, 4, 2, 512, 8192, 8192 },
        { 64, 64, 8, 2, 3, 33, 512, 1024 },
        { 128, 128, 16, 4, 8, 1, 256, 256 },
        // review of fork #450: shapes the matcher admits that the first eight did not reach
        { 192, 128, 16, 4, 2, 64, 512, 1024 },        // Q-K head size != V head size, several KV heads
        { 96, 64, 8, 2, 3, 17, 256, 512 },            // MiniCPM3's 96/64
        { 256, 256, 8, 1, 3, 17, 768, 1024 },         // one KV head, head 256
        { 64, 64, 8, 8, 2, 40, 256, 512 },            // no grouping (H == Hkv)
        { 128, 128, 32, 4, 2, 3, 1024, 1024 },        // 8 query heads per KV head
        { 64, 64, 4, 1, 2, 2100, 2304, 2304 },        // more than one query chunk inside a stream
        { 128, 128, 16, 4, 3, 9, 512, 512, 5 },       // q stream planes 5 rows apart (not packed)
        { 576, 512, 16, 1, 2, 20, 512, 1024 },        // the largest head sizes, separate V cache
        // optional: either declined or correct
        { 64, 64, 8, 2, 3, 33, 113, 128, 0, false, false, false },   // odd key count: mask planes not 16-byte aligned
        { 576, 576, 4, 1, 2, 4, 256, 256, 0, false, false, false },  // V head size past the kernel's 512
        { 128, 128, 16, 4, 3, 9, 512, 512, 0, true, false, false },  // one mask plane for every stream
        { 64, 64, 8, 2, 2, 9, 256, 256, 0, false, true, false },     // attention sinks
    };
    int failures = 0;
    uint32_t seed = 2918;
    for (const Case & c : cases) {
        const Inputs       in = make_inputs(c, seed++);
        std::vector<float> ref, got;
        bool               cpu_claimed = false, hrx_claimed = false;
        if (!run(cpu, c, in, ref, cpu_claimed)) {
            std::printf("FAIL: CPU reference did not run\n");
            return 1;
        }
        const bool ran = run(hrx, c, in, got, hrx_claimed);
        double     err = 0.0, norm = 0.0, max_abs = 0.0;
        if (ran) {
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = static_cast<double>(got[i]) - ref[i];
                err += d * d;
                norm += static_cast<double>(ref[i]) * ref[i];
                max_abs = std::fmax(max_abs, std::fabs(d));
                if (!std::isfinite(got[i])) {
                    max_abs = INFINITY;
                }
            }
        }
        const double nmse     = norm > 0.0 ? err / norm : err;
        // test-backend-ops' bound for FLASH_ATTN_EXT
        const bool   agrees   = ran && std::isfinite(max_abs) && nmse < 5e-4;
        // a required case must be claimed and agree; an optional one may be declined, but a claim must run and agree
        const bool   declined = !c.required && !hrx_claimed;
        const bool   pass     = (hrx_claimed && agrees) || declined;
        std::printf("%s D=%lld Dv=%lld H=%lld Hkv=%lld streams=%lld tokens=%lld n_kv=%lld kv_size=%lld q_gap=%lld "
                    "broadcast_mask=%d sinks=%d: supports_op=%d nmse=%.3g max_abs=%.3g%s\n",
                    pass ? "ok  " : "FAIL", (long long) c.D, (long long) c.Dv, (long long) c.H, (long long) c.Hkv,
                    (long long) c.S, (long long) c.T, (long long) c.n_kv, (long long) c.kv_size, (long long) c.q_gap,
                    c.broadcast_mask ? 1 : 0, c.sinks ? 1 : 0, hrx_claimed ? 1 : 0, nmse, max_abs,
                    declined ? " (declined)" : "");
        failures += pass ? 0 : 1;
    }
    ggml_backend_free(hrx);
    ggml_backend_free(cpu);
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
