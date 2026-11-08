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

#include "llama-kv-share.h"

#include "llama-context.h"
#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <vector>

// exported by ggml-base (ggml-backend-impl.h): a buffer made of several, freed together
extern "C" { GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers); }

namespace {

thread_local bool g_share_next = false;
thread_local bool g_share      = false;

// every non-view tensor at a 4 KiB boundary, in chunks of at most 1 GiB: both devices compute the same layout
// whatever their own alignment, and each chunk stays below Vulkan's per-buffer limit (reads past 4 GiB go wrong)
constexpr size_t SHARE_ALIGN = 4096;
constexpr size_t SHARE_CHUNK = size_t(1) << 30;

struct share_chunk {
    size_t              size   = 0;
    uint64_t            layout = 1469598103934665603ull; // FNV-1a over type and shape of every tensor, in order
    std::vector<size_t> offs;                             // offset of each of its tensors
};

std::vector<share_chunk> share_layout(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    std::vector<share_chunk> chunks(1);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != nullptr) {
            continue;
        }
        const size_t n = GGML_PAD(std::max(ggml_nbytes(t), ggml_backend_buft_get_alloc_size(buft, t)), SHARE_ALIGN);
        if (!chunks.back().offs.empty() && chunks.back().size + n > SHARE_CHUNK) {
            chunks.emplace_back();
        }
        share_chunk & c = chunks.back();
        auto mix = [&](uint64_t v) { c.layout = (c.layout ^ v) * 1099511628211ull; };
        c.offs.push_back(c.size);
        c.size += n;
        mix(t->type);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            mix((uint64_t) t->ne[d]);
        }
    }
    for (auto & c : chunks) {
        c.size = std::max(c.size, SHARE_ALIGN);
    }
    return chunks;
}

// place the tensors of ctx into the chunk buffers, in the order share_layout used
void share_place(ggml_context * ctx, const std::vector<share_chunk> & chunks, const std::vector<ggml_backend_buffer_t> & bufs) {
    size_t c = 0, i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != nullptr) {
            continue;
        }
        if (i == chunks[c].offs.size()) {
            c++;
            i = 0;
        }
        t->buffer = nullptr;
        t->data   = nullptr;
        ggml_backend_tensor_alloc(bufs[c], t, (char *) ggml_backend_buffer_get_base(bufs[c]) + chunks[c].offs[i++]);
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != nullptr) {
            t->buffer = nullptr;
            t->data   = nullptr;
            ggml_backend_view_init(t);
        }
    }
}

// one buffer for the KV cache's bookkeeping, like ggml_backend_alloc_ctx_tensors_from_buft returns
ggml_backend_buffer_t share_wrap(std::vector<ggml_backend_buffer_t> & bufs) {
    return bufs.size() == 1 ? bufs[0] : ggml_backend_multi_buffer_alloc_buffer(bufs.data(), bufs.size());
}

void * share_proc(ggml_backend_buffer_type_t buft, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg != nullptr) {
        return ggml_backend_reg_get_proc_address(reg, name);
    }
    // [1bit] a buffer type with no device: ggml-cpu's static CPU buft is one (ggml-backend.cpp marks it NULL), so
    // without this the CPU could never be the importing side of a shared KV region. Ask every registered backend
    // and take the first that provides the proc -- but only for the IMPORT proc, whose caller passes the buft's own
    // device (null for the CPU, which ggml-cpu's importer accepts). The export proc must never be resolved this way:
    // no CPU backend exports, so the lookup would hand this backend's buffer whatever other backend happens to
    // implement it (ggml-hrx) and call it on a foreign buffer. Measured 2026-11-05, a CPU decoder with
    // ONEBIT_KV_SHARE=1: "shared KV: CPU allocation or dma-buf export failed" at load, which killed the backend.
    if (std::strcmp(name, "ggml_backend_dev_buffer_from_dmabuf") != 0) {
        return nullptr;
    }
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        if (void * proc = ggml_backend_reg_get_proc_address(ggml_backend_reg_get(i), name)) {
            return proc;
        }
    }
    return nullptr;
}

template <typename F>
bool share_rec(llama_memory_i & dst, const llama_memory_i & src, F && f) {
    // [1bit] target 1: every memory class this tree can hand over, including the hybrid wrappers, whose shared
    // memory is the attention cache plus the recurrent state (a model like qwen35 keeps most of its state there)
    if (auto * d = dynamic_cast<llama_kv_cache *>(&dst)) {
        auto * s = dynamic_cast<const llama_kv_cache *>(&src);
        return s != nullptr && f(*d, *s);
    }
    if (auto * d = dynamic_cast<llama_kv_cache_iswa *>(&dst)) {
        auto * s = dynamic_cast<const llama_kv_cache_iswa *>(&src);
        return s != nullptr && share_rec(*d->get_base(), *s->get_base(), f) && share_rec(*d->get_swa(), *s->get_swa(), f);
    }
    if (auto * d = dynamic_cast<llama_memory_hybrid *>(&dst)) { // covers llama_memory_hybrid_idx
        auto * s = dynamic_cast<const llama_memory_hybrid *>(&src);
        return s != nullptr && share_rec(*d->get_mem_attn(), *s->get_mem_attn(), f) &&
                                share_rec(*d->get_mem_recr(), *s->get_mem_recr(), f);
    }
    if (auto * d = dynamic_cast<llama_memory_hybrid_iswa *>(&dst)) {
        auto * s = dynamic_cast<const llama_memory_hybrid_iswa *>(&src);
        return s != nullptr && share_rec(*d->get_mem_attn(), *s->get_mem_attn(), f) &&
                                share_rec(*d->get_mem_recr(), *s->get_mem_recr(), f);
    }
    if (auto * d = dynamic_cast<llama_memory_recurrent *>(&dst)) {
        auto * s = dynamic_cast<const llama_memory_recurrent *>(&src);
        return s != nullptr && f(*d, *s);
    }
    return false;
}

template <typename F>
bool share_each(llama_context * dst, const llama_context * src, F && f) {
    llama_memory_i * d = llama_get_memory(dst);
    const llama_memory_i * s = llama_get_memory(src);
    return d != nullptr && s != nullptr && share_rec(*d, *s, f);
}

} // namespace

// A backend that cannot export is not a misconfiguration, it is a backend that cannot: ggml-cpu is the one that
// matters here, and a hybrid route can legitimately decode on the CPU. Failing the context would make
// ONEBIT_KV_SHARE=1 turn a working server into one that does not start (measured 2026-11-05); instead the KV gets an
// ordinary buffer, llama_kv_share_regions reports no region, `?action=share` refuses that context per request, and the
// engine's route falls back to its state file for that request. The backend's own name is in the warning so the
// operator can tell "this decoder cannot" from "this decoder was not asked to".
static ggml_backend_buffer_t share_degrade(ggml_context * ctx, ggml_backend_buffer_type_t buft,
        std::vector<llama_kv_shared_region> & out, const char * why) {
    LLAMA_LOG_WARN("%s: %s %s; its KV cache is allocated normally and a shared handoff will be refused for this "
                   "context (the state-file handoff still works)\n", __func__, ggml_backend_buft_name(buft), why);
    out.clear();
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
}

ggml_backend_buffer_t llama_share_alloc(ggml_context * ctx, ggml_backend_buffer_type_t buft,
        std::vector<llama_kv_shared_region> & out) {
    using export_fn = bool (*)(ggml_backend_buffer_t, int *, size_t *);
    auto export_dmabuf = (export_fn) share_proc(buft, "ggml_backend_buffer_export_dmabuf");
    if (export_dmabuf == nullptr) {
        return share_degrade(ctx, buft, out, "cannot export its memory");
    }
    const std::vector<share_chunk> chunks = share_layout(ctx, buft);
    std::vector<ggml_backend_buffer_t> bufs;
    out.clear();
    out.resize(chunks.size());
    for (size_t c = 0; c < chunks.size(); ++c) {
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, chunks[c].size);
        if (buf == nullptr || !export_dmabuf(buf, &out[c].fd, &out[c].offset)) {
            if (buf != nullptr) {
                ggml_backend_buffer_free(buf);
            }
            for (auto * b : bufs) {
                ggml_backend_buffer_free(b);
            }
            for (const auto & r : out) {
#ifndef _WIN32
                if (r.fd >= 0) { ::close(r.fd); }
#endif
            }
            return share_degrade(ctx, buft, out, "allocation or dma-buf export failed");
        }
        out[c].size   = chunks[c].size;
        out[c].layout = chunks[c].layout;
        bufs.push_back(buf);
    }
    share_place(ctx, chunks, bufs);
    return share_wrap(bufs);
}

ggml_backend_buffer_t llama_share_import(ggml_context * ctx, ggml_backend_buffer_type_t buft,
        const std::vector<llama_kv_shared_region_pod> & regions, size_t & next) {
    const std::vector<share_chunk> chunks = share_layout(ctx, buft);
    if (next + chunks.size() > regions.size()) {
        LLAMA_LOG_ERROR("%s: %zu region(s) left, this buffer needs %zu\n", __func__, regions.size() - next, chunks.size());
        return nullptr;
    }
    for (size_t c = 0; c < chunks.size(); ++c) {
        if (chunks[c].size != regions[next + c].size || chunks[c].layout != regions[next + c].layout) {
            LLAMA_LOG_ERROR("%s: KV layout differs from the source (K/V types, shapes or flash attention)\n", __func__);
            return nullptr;
        }
    }
    using import_fn = ggml_backend_buffer_t (*)(ggml_backend_dev_t, int, size_t, size_t);
    auto import_dmabuf = (import_fn) share_proc(buft, "ggml_backend_dev_buffer_from_dmabuf");
    std::vector<ggml_backend_buffer_t> bufs;
    size_t total = 0;
    for (size_t c = 0; c < chunks.size(); ++c) {
        const auto & region = regions[next + c];
        ggml_backend_buffer_t buf = import_dmabuf ? import_dmabuf(ggml_backend_buft_get_device(buft), region.fd, region.offset, region.size) : nullptr;
        if (buf == nullptr) {
            for (auto * b : bufs) {
                ggml_backend_buffer_free(b);
            }
            LLAMA_LOG_ERROR("%s: %s cannot import the shared region\n", __func__, ggml_backend_buft_name(buft));
            return nullptr;
        }
        bufs.push_back(buf);
        total += region.size;
    }
    next += chunks.size();
    share_place(ctx, chunks, bufs);
    LLAMA_LOG_INFO("%s: %10s buffer now maps the shared region (%.2f MiB in %zu chunks, zero copy)\n", __func__,
                   ggml_backend_buft_name(buft), total / 1024.0 / 1024.0, chunks.size());
    return share_wrap(bufs);
}

llama_kv_shared_region::~llama_kv_shared_region() {
#ifndef _WIN32
    if (fd >= 0) {
        close(fd);   // a dma-buf, Linux only
    }
#endif
}

bool llama_kv_share_take_next(bool no_alloc) {
    const bool take = !no_alloc && g_share_next;
    if (!no_alloc) {
        g_share_next = false;
    }
    return take;
}

void llama_kv_share_begin(bool enabled) { g_share = enabled; }
void llama_kv_share_end() { g_share = false; }
bool llama_kv_share_enabled() { return g_share; }

ggml_backend_buffer_t llama_kv_cache::share_alloc(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    std::vector<llama_kv_shared_region> regions;
    ggml_backend_buffer_t buf = llama_share_alloc(ctx, buft, regions); // [1bit] exportable region
    shared_.push_back(std::move(regions));
    return buf;
}

bool llama_kv_cache::share_from(const llama_kv_cache & src) {
    if (other || src.other || src.shared_.empty()) {
        LLAMA_LOG_ERROR("%s: the source KV cache has %zu shared buffers\n", __func__, src.shared_.size());
        return false;
    }
    std::vector<llama_kv_shared_region_pod> regions;
    for (const auto & chunks : src.shared_) {
        for (const auto & r : chunks) {
            regions.push_back({ r.fd, r.offset, r.size, r.layout });
        }
    }
    size_t next = 0;
    return share_import(regions, next) && next == regions.size();
}

// [1bit] target 1: the same mapping as share_from(), driven by descriptors instead of by the owner context, so a
// provider in another process (which only ever sees fds that came over a unix socket) can map the decoder's KV.
// The chunk boundaries of this cache are recomputed, not transmitted: share_layout() is a pure function of the
// cache's tensors, which is why the layout hash can be checked against the owner's.
bool llama_kv_cache::share_import(const std::vector<llama_kv_shared_region_pod> & regions, size_t & next) {
    if (other || regions.empty()) {
        LLAMA_LOG_ERROR("%s: no shared regions to import\n", __func__);
        return false;
    }
    for (size_t i = 0; i < ctxs_bufs.size(); ++i) {
        ggml_context * ctx = ctxs_bufs[i].first.get();
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(ctxs_bufs[i].second.get());
        ggml_backend_buffer_t buf = llama_share_import(ctx, buft, regions, next);
        if (buf == nullptr) {
            return false;
        }
        ctxs_bufs[i].second.reset(buf); // frees the buffer this cache had
    }
    return true;
}

bool llama_kv_cache::cells_copy_from(const llama_kv_cache & src) {
    if (other || src.other || v_cells.size() != src.v_cells.size()) {
        return false;
    }
    for (size_t s = 0; s < v_cells.size(); ++s) {
        if (v_cells[s].size() != src.v_cells[s].size()) {
            return false;
        }
    }
    v_cells = src.v_cells;
    v_heads = src.v_heads;
    return true;
}

void llama_context::kv_memory_rebound() {
    gf_res_prev->reset();
    ggml_backend_sched_reset(sched.get());
    sched_need_reserve = true;
}

void llama_kv_share_next(bool enabled) {
    g_share_next = enabled;
}

bool llama_kv_share_from(llama_context * dst, llama_context * src) {
    if (!share_each(dst, src, [](auto & d, const auto & s) { return d.share_from(s); })) {
        return false;
    }
    dst->kv_memory_rebound();
    return true;
}

// [1bit] target 1: the tensors a prefill provider writes, and where they are. A tensor's offset is taken from its own
// buffer's base, which is the region it was mapped into (ggml_backend_dev_buffer_from_dmabuf returns a buffer over the
// region window), and `region` is the ordinal of the first buffer the walk meets -- the same order share_place fills,
// so it is the index into the region list the decoder sent.
bool llama_kv_cache::share_tensors(std::vector<llama_kv_shared_tensor> & out) const {
    std::vector<const void *> bases;
    auto region_of = [&](const void * base) {
        for (size_t i = 0; i < bases.size(); ++i) {
            if (bases[i] == base) return uint32_t(i);
        }
        bases.push_back(base);
        return uint32_t(bases.size() - 1);
    };
    for (const auto & L : layers) {
        for (uint32_t is_v = 0; is_v < 2; ++is_v) {
            ggml_tensor * t = is_v ? L.v : L.k;
            if (t == nullptr || t->buffer == nullptr || t->data == nullptr) {
                continue;
            }
            const char * base = (const char *) ggml_backend_buffer_get_base(t->buffer);
            out.push_back({ L.il, is_v, region_of(base), uint64_t((const char *) t->data - base),
                            uint64_t(ggml_nbytes(t)), uint64_t(t->ne[0]) });
        }
    }
    return true;
}

bool llama_memory_recurrent::share_tensors(std::vector<llama_kv_shared_tensor> &) const {
    return true;   // the recurrent state is not K/V rows; a prefill provider does not write it
}

size_t llama_kv_share_tensor_map(llama_context * ctx, llama_kv_share_tensor * out, size_t n_out) {
    std::vector<llama_kv_shared_tensor> tensors;
    if (!share_each(ctx, ctx, [&](auto & d, const auto &) { return d.share_tensors(tensors); })) {
        return 0;
    }
    const size_t n = std::min(n_out, tensors.size());
    for (size_t i = 0; i < n; ++i) {
        out[i].layer     = tensors[i].layer;
        out[i].is_v      = tensors[i].is_v;
        out[i].region    = tensors[i].region;
        out[i].offset    = tensors[i].offset;
        out[i].bytes     = tensors[i].bytes;
        out[i].row_elems = tensors[i].row_elems;
    }
    return tensors.size();
}

bool llama_kv_cells_copy(llama_context * dst, const llama_context * src) {
    return share_each(dst, src, [](auto & d, const auto & s) { return d.cells_copy_from(s); });
}

// [1bit] target 1, both ends of a handoff whose provider lives in another process.
size_t llama_kv_share_regions(llama_context * ctx, llama_kv_share_region * out, size_t n_out) {
    std::vector<llama_kv_shared_region_pod> regions;
    if (!share_each(ctx, ctx, [&](auto & d, const auto &) {
            return d.share_regions(regions);
        })) {
        return 0;
    }
    const size_t n = std::min(n_out, regions.size());
    for (size_t i = 0; i < n; ++i) {
        out[i].fd     = regions[i].fd;
        out[i].offset = regions[i].offset;
        out[i].size   = regions[i].size;
        out[i].layout = regions[i].layout;
    }
    return regions.size();
}

bool llama_kv_share_import(llama_context * ctx, const llama_kv_share_region * regions, size_t n_regions) {
    std::vector<llama_kv_shared_region_pod> in;
    in.reserve(n_regions);
    for (size_t i = 0; i < n_regions; ++i) {
        in.push_back({ regions[i].fd, regions[i].offset, regions[i].size, regions[i].layout });
    }
    size_t next = 0;
    if (!share_each(ctx, ctx, [&](auto & d, const auto &) { return d.share_import(in, next); }) || next != in.size()) {
        return false;
    }
    ctx->kv_memory_rebound();
    return true;
}

bool llama_kv_cells_adopt(llama_context * ctx, uint32_t n_tokens, llama_seq_id seq_id) {
    return share_each(ctx, ctx, [&](auto & d, const auto &) {
        return d.cells_adopt(n_tokens, seq_id);
    });
}

// [1bit] what a provider in another process has to reproduce for its own cache to lay out like this one's.
bool llama_kv_share_params(const llama_context * ctx, struct llama_kv_share_cparams * out) {
    if (ctx == nullptr || out == nullptr) {
        return false;
    }
    const llama_cparams & cp = ctx->get_cparams();
    out->n_ctx        = cp.n_ctx;
    out->n_batch      = cp.n_batch;
    out->n_ubatch     = cp.n_ubatch;
    out->n_seq_max    = cp.n_seq_max;
    out->flash_attn   = cp.flash_attn ? 1 : 0;
    out->type_k       = (int32_t) cp.type_k;
    out->type_v       = (int32_t) cp.type_v;
    out->n_rs_seq     = cp.n_rs_seq; // [1bit] widens the recurrent state's rows: a provider must build the same
    return true;
}
