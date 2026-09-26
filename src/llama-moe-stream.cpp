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

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
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

struct llama_moe_stream {
    onebit::moe::GgufIndex index;
    std::unique_ptr<onebit::moe::ExpertCache> cache;
    int max_batch = 8;
    std::map<int, layer_state> layers;
    layer_state * last = nullptr;  // the layer whose experts are pinned (layers run in order)
};

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
            if (const char * io = getenv("ONEBIT_MOE_IO")) opt.io_threads = atoi(io);
            if (const char * mb = getenv("ONEBIT_MOE_MAX_BATCH")) st->max_batch = atoi(mb);
            st->cache = std::make_unique<onebit::moe::ExpertCache>(st->index, opt);
            LLAMA_LOG_INFO("%s: streaming routed experts from %s: %d slots, %.1f GiB pinned, batches up to %d\n",
                           __func__, file, st->cache->slots(), st->cache->pinned_bytes() / double(1u << 30), st->max_batch);
            s = std::move(st);
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: expert streaming disabled: %s\n", __func__, e.what());
        }
    });
    return s.get();
}

bool llama_moe_stream_applies(const llama_moe_stream * s, int64_t n_tokens, const ggml_tensor * gate_exps,
                              const ggml_tensor * up_exps, const ggml_tensor * down_exps) {
    return s && n_tokens <= s->max_batch && gate_exps && up_exps && down_exps;
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

// op 1, one thread: pin the batch's experts (reading misses) and convert its inputs for vec_dot
void op_acquire(ggml_tensor * dst, int ith, int nth, void * ud) {
    (void) nth;
    if (ith != 0) return;
    layer_state & L = *(layer_state *) ud;
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

}  // namespace

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

#else  // not built with the expert cache (non-Linux)

llama_moe_stream * llama_moe_stream_get() { return nullptr; }

bool llama_moe_stream_applies(const llama_moe_stream *, int64_t, const ggml_tensor *, const ggml_tensor *,
                              const ggml_tensor *) {
    return false;
}

ggml_tensor * llama_moe_stream_build(ggml_context *, llama_moe_stream *, int, ggml_tensor *, ggml_tensor *,
                                     ggml_tensor *, ggml_tensor *, ggml_tensor *) {
    return nullptr;
}

#endif
