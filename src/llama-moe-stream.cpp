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
#include "llama-moe-stream.h"

#ifdef LLAMA_MOE_STREAM

#include "llama-impl.h"
#include "moe-expert-cache.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// What one layer's streamed FFN keeps between its three ops of one graph run.
struct layer_state {
    llama_moe_stream * s = nullptr;
    int il = 0;
    std::vector<int> held;                    // experts acquired for the current batch
    std::unordered_map<int, int> slot_of;     // expert -> index into res
    std::vector<onebit::moe::ExpertCache::Resident> res;
    int part_gate = -1, part_up = -1, part_down = -1;
    ggml_type t_gate = GGML_TYPE_F32, t_up = GGML_TYPE_F32, t_down = GGML_TYPE_F32;
    int64_t n_embd = 0, n_ff = 0;
    std::vector<uint8_t> qx_gate, qx_up;      // the batch's inputs in gate's / up's vec_dot type
};

}  // namespace

struct gpu_layer {
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    bool failed = false;
};

struct llama_moe_stream {
    onebit::moe::GgufIndex index;
    std::unique_ptr<onebit::moe::ExpertCache> cache;
    ggml_backend_dev_t gpu = nullptr;          // null: experts compute on the CPU
    ggml_context * tctx = nullptr;             // the slot tensors
    std::map<int, gpu_layer> gpu_layers;
    std::map<uint8_t *, ggml_backend_buffer_t> region_bufs;  // slot regions the device allocated
    int max_batch = 8;
    int layer_slots = 0;  // the fewest slots any one layer can use
    // Gate-ahead prefetch: the routers of the next `prefetch` MoE layers (F32, from the file)
    // applied to layer l's FFN input; their top-k experts are queued for reading. A worker thread
    // scores them, so decode never waits for it.
    int prefetch = 1;
    std::map<int, std::vector<float>> router;  // MoE layer -> [n_expert][n_embd]
    std::map<int, int> next_layer;             // MoE layer -> the next MoE layer
    std::thread pf_thread;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    bool pf_stop = false, pf_has = false;
    int pf_from = 0, pf_used = 0;              // the job: layer whose FFN input pf_x is, experts per token
    uint64_t pf_seq = 0;                       // the remap count when it was queued
    std::vector<float> pf_x;
    std::atomic<uint64_t> remap_seq{0};        // remaps started (GPU and CPU mode)
    std::map<int, layer_state> layers;
    layer_state * last = nullptr;  // the layer whose experts are pinned (layers run in order)
    // ONEBIT_MOE_STATS=1: where decode time goes, printed at exit
    bool report = false;
    float subst = 0;           // ONEBIT_MOE_SUBST
    std::atomic<uint64_t> substituted{0}, selected{0};
    uint64_t remaps = 0;
    double remap_ms = 0, gap_ms = 0;  // inside the remap op; between one remap's end and the next's start
    std::chrono::steady_clock::time_point last_remap_end{};
    ~llama_moe_stream() {
        if (pf_thread.joinable()) {
            {
                std::lock_guard<std::mutex> lk(pf_mu);
                pf_stop = true;
            }
            pf_cv.notify_all();
            pf_thread.join();
        }
        if (!report || !cache) return;
        const auto st = cache->stats();
        const uint64_t used = st.hits + st.prefetch_hits + st.misses;
        fprintf(stderr, "moe-stream: %llu remaps, %.3f ms in remap (%.3f waiting for reads), %.3f ms between remaps (per remap)\n",
                (unsigned long long) remaps, remap_ms / std::max<uint64_t>(remaps, 1), st.stall_ms / std::max<uint64_t>(remaps, 1),
                gap_ms / std::max<uint64_t>(remaps, 1));
        if (subst > 0)
            fprintf(stderr, "moe-stream: %llu of %llu routed experts replaced by resident ones (ONEBIT_MOE_SUBST %.2f)\n",
                    (unsigned long long) substituted.load(), (unsigned long long) selected.load(), subst);
        fprintf(stderr, "moe-stream: %llu experts used: %.1f%% hits, %.1f%% prefetch hits, %.1f%% misses; %llu prefetched, %llu wasted; %.2f GiB read\n",
                (unsigned long long) used, 100.0 * st.hits / std::max<uint64_t>(used, 1),
                100.0 * st.prefetch_hits / std::max<uint64_t>(used, 1), 100.0 * st.misses / std::max<uint64_t>(used, 1),
                (unsigned long long) st.prefetches, (unsigned long long) st.prefetch_wasted, st.bytes_read / double(1u << 30));
    }
};

namespace {

float dot_f32(const float * a, const float * b, int64_t n) {
    float acc[8] = {};  // eight lanes, so the compiler vectorizes without reassociating one sum
    int64_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
    float sum = 0;
    for (float v : acc) sum += v;
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// Scores the next layers' routers on a queued FFN input and prefetches their top-k experts,
// skipping a layer whose remap has already started.
void prefetch_worker(llama_moe_stream * s) {
    std::vector<float> x;
    std::vector<std::pair<float, int>> sc;
    std::vector<int> want;
    for (;;) {
        int from, used;
        uint64_t seq;
        {
            std::unique_lock<std::mutex> lk(s->pf_mu);
            s->pf_cv.wait(lk, [&] { return s->pf_stop || s->pf_has; });
            if (s->pf_stop) return;
            x.swap(s->pf_x);
            from = s->pf_from;
            used = s->pf_used;
            seq = s->pf_seq;
            s->pf_has = false;
        }
        const int64_t n_embd = (int64_t) x.size();
        int l = from;
        for (int d = 1; d <= s->prefetch; ++d) {
            auto nl = s->next_layer.find(l);
            if (nl == s->next_layer.end()) break;
            l = nl->second;
            if (s->remap_seq.load() >= seq + d) continue;  // that layer is already running
            auto rw = s->router.find(l);
            if (rw == s->router.end() || rw->second.size() % n_embd) break;
            const int64_t n_exp = rw->second.size() / n_embd;
            sc.resize(n_exp);
            for (int64_t e = 0; e < n_exp; ++e) sc[e] = { -dot_f32(rw->second.data() + e * n_embd, x.data(), n_embd), (int) e };
            const int64_t k = std::min<int64_t>(used, n_exp);
            std::partial_sort(sc.begin(), sc.begin() + k, sc.end());
            want.clear();
            for (int64_t i = 0; i < k; ++i) want.push_back(sc[i].second);
            s->cache->prefetch(l, want);
        }
    }
}

}  // namespace

llama_moe_stream * llama_moe_stream_get() {
    static std::once_flag once;
    static std::unique_ptr<llama_moe_stream> s;
    std::call_once(once, [] {
        const char * slots = getenv("ONEBIT_MOE_SLOTS");
        const char * file = getenv("ONEBIT_MOE_FILE");
        if (!slots) return;
        if (!file) {
            LLAMA_LOG_ERROR("%s: ONEBIT_MOE_SLOTS needs ONEBIT_MOE_FILE (the model's GGUF)\n", __func__);
            return;
        }
        try {
            auto st = std::make_unique<llama_moe_stream>();
            st->index = onebit::moe::GgufIndex::open(file);
            onebit::moe::CacheOptions opt;
            opt.slots = atoi(slots);
            const char * devname = getenv("ONEBIT_MOE_DEVICE");
            if (!devname || strcmp(devname, "cpu") != 0) {
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    const auto t = ggml_backend_dev_type(d);
                    if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
                    if (devname && strcmp(devname, ggml_backend_dev_name(d)) != 0) continue;
                    st->gpu = d;
                    break;
                }
                if (devname && !st->gpu) LLAMA_LOG_WARN("%s: no device %s; experts compute on the CPU\n", __func__, devname);
            }
            opt.per_layer = st->gpu != nullptr;  // one region per layer: one device buffer each
            if (st->gpu) {
                // The slots live in device buffers the host can map (UMA): the GPU reads them at
                // full speed, where imported host memory runs about 100x slower.
                using get_host_ptr_t = void * (*)(ggml_backend_buffer_t);
                auto get_ptr = (get_host_ptr_t) ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(st->gpu), "ggml_backend_vk_buffer_get_host_ptr");
                if (get_ptr) {
                    llama_moe_stream * raw = st.get();
                    opt.region = [raw, get_ptr](int, size_t bytes) -> uint8_t * {
                        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(raw->gpu), bytes);
                        if (!b) return nullptr;
                        uint8_t * p = (uint8_t *) get_ptr(b);
                        if (!p) { ggml_backend_buffer_free(b); return nullptr; }
                        raw->region_bufs[p] = b;
                        return p;
                    };
                }
            }
            if (const char * io = getenv("ONEBIT_MOE_IO")) opt.io_threads = atoi(io);
            if (const char * mb = getenv("ONEBIT_MOE_MAX_BATCH")) st->max_batch = atoi(mb);
            st->cache = std::make_unique<onebit::moe::ExpertCache>(st->index, opt);
            st->layer_slots = int(double(opt.slots) / st->index.experts.size());
            if (const char * pf = getenv("ONEBIT_MOE_PREFETCH")) st->prefetch = std::max(0, atoi(pf));
            if (const char * r = getenv("ONEBIT_MOE_STATS")) st->report = atoi(r) != 0;
            if (const char * r = getenv("ONEBIT_MOE_SUBST")) st->subst = std::max(0.0f, (float) atof(r));
            if (st->prefetch) {  // the routers, read once (F32 only; others skip prefetch)
                int prev = -1;
                for (const auto & [l, parts] : st->index.experts) {
                    if (prev >= 0) st->next_layer[prev] = l;
                    prev = l;
                    const std::string name = "blk." + std::to_string(l) + ".ffn_gate_inp.weight";
                    for (const auto & t : st->index.tensors) {
                        if (t.name != name || t.type != 0 /* F32 */) continue;
                        std::vector<float> w(t.bytes / 4);
                        FILE * f = fopen(st->index.files[t.file].c_str(), "rb");
                        if (f && fseeko(f, (off_t) t.offset, SEEK_SET) == 0 && fread(w.data(), 4, w.size(), f) == w.size())
                            st->router[l] = std::move(w);
                        if (f) fclose(f);
                    }
                }
            }
            if (st->prefetch && !st->router.empty()) st->pf_thread = std::thread(prefetch_worker, st.get());
            if (st->gpu) {
                ggml_init_params ip = { 3 * 1024 * ggml_tensor_overhead(), nullptr, true };
                st->tctx = ggml_init(ip);
            }
            LLAMA_LOG_INFO("%s: streaming routed experts from %s: %d slots, %.1f GiB pinned, batches up to %d, computed on %s\n",
                           __func__, file, st->cache->slots(), st->cache->pinned_bytes() / double(1u << 30), st->max_batch,
                           st->gpu ? ggml_backend_dev_name(st->gpu) : "CPU");
            s = std::move(st);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: expert streaming disabled: %s\n", __func__, e.what());
        }
    });
    return s.get();
}

bool llama_moe_stream_applies(const llama_moe_stream * s, int64_t n_tokens, int64_t n_expert_used,
                              const ggml_tensor * gate_exps, const ggml_tensor * up_exps, const ggml_tensor * down_exps) {
    return s && n_tokens <= s->max_batch && n_tokens * n_expert_used <= s->layer_slots && gate_exps && up_exps &&
           down_exps;
}

namespace {

int part_named(const onebit::moe::GgufIndex & idx, int il, const char * suffix) {
    auto it = idx.experts.find(il);
    if (it == idx.experts.end()) return -1;
    for (size_t i = 0; i < it->second.size(); ++i)
        if (it->second[i].tensor.find(suffix) != std::string::npos) return (int) i;
    return -1;
}

int32_t id_at(const ggml_tensor * ids, int64_t k, int64_t t) {
    return *(const int32_t *) ((const char *) ids->data + t * ids->nb[1] + k * ids->nb[0]);
}

// Gate-ahead: hand this layer's FFN input to the prefetch worker (the latest job replaces an
// unstarted one)
void prefetch_next(layer_state & L, const ggml_tensor * cur, int64_t n_used) {
    llama_moe_stream & s = *L.s;
    if (!s.pf_thread.joinable() || !cur || cur->ne[1] != 1) return;  // single-token steps
    {
        std::lock_guard<std::mutex> lk(s.pf_mu);
        s.pf_x.assign((const float *) cur->data, (const float *) cur->data + cur->ne[0]);
        s.pf_from = L.il;
        s.pf_used = (int) n_used;
        s.pf_seq = s.remap_seq.load();
        s.pf_has = true;
    }
    s.pf_cv.notify_one();
}

// op 1, one thread: pin the batch's experts (reading misses) and convert its inputs for vec_dot
void op_acquire(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    layer_state & L = *(layer_state *) ud;
    L.s->remap_seq++;
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * cur = dst->src[1];
    auto & cache = *L.s->cache;
    for (layer_state * prev : { L.s->last, &L })  // the previous layer has finished with its experts
        if (prev && !prev->held.empty()) {
            cache.release(prev->il, prev->held);
            prev->held.clear();
        }
    L.s->last = &L;
    L.held.clear();
    L.slot_of.clear();
    for (int64_t t = 0; t < ids->ne[1]; ++t)
        for (int64_t k = 0; k < ids->ne[0]; ++k) {
            const int e = id_at(ids, k, t);
            if (L.slot_of.emplace(e, (int) L.held.size()).second) L.held.push_back(e);
        }
    L.res = cache.acquire(L.il, L.held);
    auto convert = [&](ggml_type t, std::vector<uint8_t> & buf) {
        const auto * tr = ggml_get_type_traits_cpu(t);
        const ggml_type vt = tr->vec_dot_type;
        const size_t row = ggml_row_size(vt, L.n_embd);
        buf.resize(row * cur->ne[1]);
        const auto * vtr = ggml_get_type_traits_cpu(vt);
        for (int64_t i = 0; i < cur->ne[1]; ++i) {
            const float * x = (const float *) ((const char *) cur->data + i * cur->nb[1]);
            if (vt == GGML_TYPE_F32) memcpy(buf.data() + i * row, x, row);
            else vtr->from_float(x, buf.data() + i * row, L.n_embd);
        }
    };
    prefetch_next(L, cur, ids->ne[0]);
    convert(L.t_gate, L.qx_gate);
    if (L.t_up == L.t_gate) L.qx_up = L.qx_gate;
    else convert(L.t_up, L.qx_up);
    *(int32_t *) dst->data = (int32_t) L.held.size();
}

// op 2, all threads: act[r, k, t] = silu(gate_e[r] . x_t) * (up_e[r] . x_t), rows split across threads
void op_gate_up(ggml_tensor * dst, int ith, int nth, void * ud) {
    layer_state & L = *(layer_state *) ud;
    const ggml_tensor * ids = dst->src[0];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1], n_ff = L.n_ff;
    const auto * tg = ggml_get_type_traits_cpu(L.t_gate);
    const auto * tu = ggml_get_type_traits_cpu(L.t_up);
    const size_t row_g = ggml_row_size(L.t_gate, L.n_embd), row_u = ggml_row_size(L.t_up, L.n_embd);
    const size_t qrow_g = ggml_row_size(tg->vec_dot_type, L.n_embd), qrow_u = ggml_row_size(tu->vec_dot_type, L.n_embd);
    const int64_t total = n_tok * n_used * n_ff;
    const int64_t per = (total + nth - 1) / nth, r0 = ith * per, r1 = std::min(total, r0 + per);
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t row = r % n_ff, k = (r / n_ff) % n_used, t = r / (n_ff * n_used);
        const auto & res = L.res[L.slot_of.at(id_at(ids, k, t))];
        float g = 0, u = 0;
        tg->vec_dot((int) L.n_embd, &g, 0, res.parts[L.part_gate] + row * row_g, 0, L.qx_gate.data() + t * qrow_g, 0, 1);
        tu->vec_dot((int) L.n_embd, &u, 0, res.parts[L.part_up] + row * row_u, 0, L.qx_up.data() + t * qrow_u, 0, 1);
        const float silu = g / (1.0f + expf(-g));
        *(float *) ((char *) dst->data + t * dst->nb[2] + k * dst->nb[1] + row * dst->nb[0]) = silu * u;
    }
}

// op 3, all threads: out[c, k, t] = down_e[c] . act[:, k, t], rows split across threads
void op_down(ggml_tensor * dst, int ith, int nth, void * ud) {
    layer_state & L = *(layer_state *) ud;
    const ggml_tensor * act = dst->src[0];
    const ggml_tensor * ids = dst->src[1];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1], n_embd = L.n_embd, n_ff = L.n_ff;
    const auto * td = ggml_get_type_traits_cpu(L.t_down);
    const ggml_type vt = td->vec_dot_type;
    const auto * vtr = ggml_get_type_traits_cpu(vt);
    const size_t row_d = ggml_row_size(L.t_down, n_ff), qrow = ggml_row_size(vt, n_ff);
    std::vector<uint8_t> qa(qrow);
    int64_t have = -1;  // the (t, k) whose act is in qa
    const int64_t total = n_tok * n_used * n_embd;
    const int64_t per = (total + nth - 1) / nth, r0 = ith * per, r1 = std::min(total, r0 + per);
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t c = r % n_embd, tk = r / n_embd, k = tk % n_used, t = tk / n_used;
        if (tk != have) {
            const float * a = (const float *) ((const char *) act->data + t * act->nb[2] + k * act->nb[1]);
            if (vt == GGML_TYPE_F32) memcpy(qa.data(), a, qrow);
            else vtr->from_float(a, qa.data(), n_ff);
            have = tk;
        }
        const auto & res = L.res[L.slot_of.at(id_at(ids, k, t))];
        float o = 0;
        td->vec_dot((int) n_ff, &o, 0, res.parts[L.part_down] + c * row_d, 0, qa.data(), 0, 1);
        *(float *) ((char *) dst->data + t * dst->nb[2] + k * dst->nb[1] + c * dst->nb[0]) = o;
    }
}

// ONEBIT_MOE_SUBST, one thread: top-k of each token's selection scores, where a chosen expert
// that is not resident gives way to the best resident one among the next k candidates if that
// one scores at least subst times as much. Scores may be logits or probabilities, so the ratio
// is taken on softmax-free values only when both are positive; otherwise on exp(difference).
struct select_args { llama_moe_stream * s; int il; };

void op_select(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    const auto & a = *(const select_args *) ud;
    llama_moe_stream & s = *a.s;
    const ggml_tensor * sp = dst->src[0];
    const int64_t n_exp = sp->ne[0], n_tok = sp->ne[1], k = dst->ne[0];
    const int64_t n_cand = std::min<int64_t>(2 * k, n_exp);
    std::vector<std::pair<float, int>> sc(n_exp);
    std::vector<char> taken(n_exp);
    uint64_t n_sub = 0;
    for (int64_t t = 0; t < n_tok; ++t) {
        const float * row = (const float *) ((const char *) sp->data + t * sp->nb[1]);
        for (int64_t e = 0; e < n_exp; ++e) sc[e] = { -row[e], (int) e };
        std::partial_sort(sc.begin(), sc.begin() + n_cand, sc.end());
        std::fill(taken.begin(), taken.end(), 0);
        std::vector<char> res(n_cand);
        for (int64_t i = 0; i < n_cand; ++i) res[i] = s.cache->resident(a.il, sc[i].second);
        for (int64_t i = 0; i < k; ++i) taken[sc[i].second] = 1;
        int32_t * out = (int32_t *) ((char *) dst->data + t * dst->nb[1]);
        for (int64_t i = 0; i < k; ++i) {
            int e = sc[i].second;
            if (!res[i]) {
                const float pe = -sc[i].first;
                for (int64_t j = k; j < n_cand; ++j) {
                    const int c = sc[j].second;
                    if (!res[j] || taken[c]) continue;
                    const float pc = -sc[j].first;
                    const float ratio = pe > 0 && pc >= 0 ? pc / pe : std::exp(pc - pe);
                    if (ratio < s.subst) break;  // candidates only get worse
                    taken[c] = 1;
                    e = c;
                    ++n_sub;
                    break;
                }
            }
            out[i] = e;
        }
    }
    s.substituted += n_sub;
    s.selected += (uint64_t) (n_tok * k);
}

// GPU mode, one thread: pin the batch's experts and write their slot indices
void op_remap(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    layer_state & L = *(layer_state *) ud;
    const auto t_in = std::chrono::steady_clock::now();
    L.s->remap_seq++;
    if (L.s->remaps) L.s->gap_ms += std::chrono::duration<double, std::milli>(t_in - L.s->last_remap_end).count();
    const ggml_tensor * ids = dst->src[0];
    auto & cache = *L.s->cache;
    for (layer_state * prev : { L.s->last, &L })  // the GPU has finished the previous layer
        if (prev && !prev->held.empty()) {
            cache.release(prev->il, prev->held);
            prev->held.clear();
        }
    L.s->last = &L;
    L.slot_of.clear();
    for (int64_t t = 0; t < ids->ne[1]; ++t)
        for (int64_t k = 0; k < ids->ne[0]; ++k) {
            const int e = id_at(ids, k, t);
            if (L.slot_of.emplace(e, (int) L.held.size()).second) L.held.push_back(e);
        }
    L.res = cache.acquire(L.il, L.held);
    for (int64_t t = 0; t < ids->ne[1]; ++t)
        for (int64_t k = 0; k < ids->ne[0]; ++k)
            ((int32_t *) dst->data)[t * ids->ne[0] + k] = L.res[L.slot_of.at(id_at(ids, k, t))].slot;
    prefetch_next(L, dst->src[1], ids->ne[0]);
    L.s->last_remap_end = std::chrono::steady_clock::now();
    L.s->remap_ms += std::chrono::duration<double, std::milli>(L.s->last_remap_end - t_in).count();
    L.s->remaps++;
}

}  // namespace

bool llama_moe_stream_gpu(ggml_context * ctx, llama_moe_stream * s, int il, ggml_tensor * cur, ggml_tensor * ids,
                          ggml_tensor * gate_exps, ggml_tensor * up_exps, ggml_tensor * down_exps,
                          ggml_tensor ** slot_ids, ggml_tensor ** slot_gate, ggml_tensor ** slot_up,
                          ggml_tensor ** slot_down) {
    if (!s->gpu) return false;
    gpu_layer & G = s->gpu_layers[il];
    if (G.failed) return false;
    layer_state & L = s->layers[il];
    if (!G.buf) {
        L.s = s;
        L.il = il;
        L.part_gate = part_named(s->index, il, "ffn_gate_exps");
        L.part_up = part_named(s->index, il, "ffn_up_exps");
        L.part_down = part_named(s->index, il, "ffn_down_exps");
        const auto & lay = s->cache->layout(il);
        if (L.part_gate < 0 || L.part_up < 0 || L.part_down < 0) { G.failed = true; return false; }
        auto own = s->region_bufs.find(lay.base);
        G.buf = own != s->region_bufs.end() ? own->second
                                            : ggml_backend_dev_buffer_from_host_ptr(s->gpu, lay.base, lay.region_bytes, lay.region_bytes);
        if (!G.buf) {
            LLAMA_LOG_WARN("%s: %s cannot import layer %d's slots; it streams on the CPU\n", __func__, ggml_backend_dev_name(s->gpu), il);
            G.failed = true;
            return false;
        }
        auto view = [&](const ggml_tensor * like, int part) {
            ggml_tensor * t = ggml_new_tensor_3d(s->tctx, like->type, like->ne[0], like->ne[1], lay.n_slots);
            GGML_ASSERT(ggml_nbytes(t) == lay.part_bytes[part] * lay.n_slots);  // the cache packs parts like ggml
            ggml_format_name(t, "blk.%d.%s_slots", il, part == L.part_gate ? "ffn_gate_exps" : part == L.part_up ? "ffn_up_exps" : "ffn_down_exps");
            // a device buffer's addresses are its own (Vulkan's base is not the host pointer)
            char * at = (char *) ggml_backend_buffer_get_base(G.buf) + lay.part_off[part];
            if (ggml_backend_tensor_alloc(G.buf, t, at) != GGML_STATUS_SUCCESS) return (ggml_tensor *) nullptr;
            return t;
        };
        G.gate = view(gate_exps, L.part_gate);
        G.up = view(up_exps, L.part_up);
        G.down = view(down_exps, L.part_down);
        if (!G.gate || !G.up || !G.down) { G.failed = true; return false; }
    }
    ggml_tensor * args[] = { ids, cur };
    *slot_ids = ggml_custom_4d(ctx, GGML_TYPE_I32, ids->ne[0], ids->ne[1], 1, 1, args, 2, op_remap, 1, &L);
    *slot_gate = G.gate;
    *slot_up = G.up;
    *slot_down = G.down;
    return true;
}

ggml_tensor * llama_moe_stream_build(ggml_context * ctx, llama_moe_stream * s, int il, ggml_tensor * cur,
                                     ggml_tensor * ids, ggml_tensor * gate_exps, ggml_tensor * up_exps,
                                     ggml_tensor * down_exps) {
    layer_state & L = s->layers[il];
    if (!L.s) {
        L.s = s;
        L.il = il;
        L.part_gate = part_named(s->index, il, "ffn_gate_exps");
        L.part_up = part_named(s->index, il, "ffn_up_exps");
        L.part_down = part_named(s->index, il, "ffn_down_exps");
        GGML_ASSERT(L.part_gate >= 0 && L.part_up >= 0 && L.part_down >= 0 && "the streamed file lacks this layer's expert tensors");
    }
    L.t_gate = gate_exps->type;
    L.t_up = up_exps->type;
    L.t_down = down_exps->type;
    L.n_embd = gate_exps->ne[0];
    L.n_ff = gate_exps->ne[1];
    const int64_t n_used = ids->ne[0], n_tok = ids->ne[1];

    ggml_tensor * args1[] = { ids, cur };
    ggml_tensor * pinned = ggml_custom_4d(ctx, GGML_TYPE_I32, 1, 1, 1, 1, args1, 2, op_acquire, 1, &L);
    ggml_tensor * args2[] = { ids, pinned };
    ggml_tensor * act = ggml_custom_4d(ctx, GGML_TYPE_F32, L.n_ff, n_used, n_tok, 1, args2, 2, op_gate_up, GGML_N_TASKS_MAX, &L);
    ggml_tensor * args3[] = { act, ids };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, L.n_embd, n_used, n_tok, 1, args3, 2, op_down, GGML_N_TASKS_MAX, &L);
}

ggml_tensor * llama_moe_stream_select(ggml_context * ctx, llama_moe_stream * s, int il, ggml_tensor * selection_probs,
                                      int64_t n_expert_used) {
    if (!s || s->subst <= 0 || selection_probs->type != GGML_TYPE_F32) return nullptr;
    static std::map<int, select_args> args;  // one per layer, alive as long as the graphs
    args[il] = { s, il };
    ggml_tensor * src[] = { selection_probs };
    return ggml_custom_4d(ctx, GGML_TYPE_I32, n_expert_used, selection_probs->ne[1], 1, 1, src, 1, op_select, 1, &args[il]);
}

#else  // not built with the expert cache (non-Linux)

llama_moe_stream * llama_moe_stream_get() { return nullptr; }

bool llama_moe_stream_gpu(ggml_context *, llama_moe_stream *, int, ggml_tensor *, ggml_tensor *, ggml_tensor *,
                          ggml_tensor *, ggml_tensor *, ggml_tensor **, ggml_tensor **, ggml_tensor **, ggml_tensor **) {
    return false;
}

bool llama_moe_stream_applies(const llama_moe_stream *, int64_t, int64_t, const ggml_tensor *, const ggml_tensor *,
                              const ggml_tensor *) {
    return false;
}

ggml_tensor * llama_moe_stream_build(ggml_context *, llama_moe_stream *, int, ggml_tensor *, ggml_tensor *,
                                     ggml_tensor *, ggml_tensor *, ggml_tensor *) {
    return nullptr;
}

ggml_tensor * llama_moe_stream_select(ggml_context *, llama_moe_stream *, int, ggml_tensor *, int64_t) {
    return nullptr;
}

#endif
