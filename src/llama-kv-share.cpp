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

#include "ggml-backend.h"

#include <algorithm>
#include <stdexcept>
#include <unistd.h>
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
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

template <typename F>
bool share_each(llama_context * dst, const llama_context * src, F && f) {
    if (auto * d = dynamic_cast<llama_kv_cache *>(llama_get_memory(dst))) {
        auto * s = dynamic_cast<const llama_kv_cache *>(llama_get_memory(src));
        return s != nullptr && f(*d, *s);
    }
    if (auto * d = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(dst))) {
        auto * s = dynamic_cast<const llama_kv_cache_iswa *>(llama_get_memory(src));
        return s != nullptr && f(*d->get_base(), *s->get_base()) && f(*d->get_swa(), *s->get_swa());
    }
    return false;
}

} // namespace

llama_kv_shared_region::~llama_kv_shared_region() {
    if (fd >= 0) {
        close(fd);
    }
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
    using export_fn = bool (*)(ggml_backend_buffer_t, int *, size_t *);
    auto export_dmabuf = (export_fn) share_proc(buft, "ggml_backend_buffer_export_dmabuf");
    if (export_dmabuf == nullptr) {
        throw std::runtime_error(format("shared KV: %s cannot export its memory", ggml_backend_buft_name(buft)));
    }
    const std::vector<share_chunk> chunks = share_layout(ctx, buft);
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<llama_kv_shared_region> regions(chunks.size());
    for (size_t c = 0; c < chunks.size(); ++c) {
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, chunks[c].size);
        if (buf == nullptr || !export_dmabuf(buf, &regions[c].fd, &regions[c].offset)) {
            if (buf != nullptr) {
                ggml_backend_buffer_free(buf);
            }
            for (auto * b : bufs) {
                ggml_backend_buffer_free(b);
            }
            throw std::runtime_error(format("shared KV: %s allocation or dma-buf export failed", ggml_backend_buft_name(buft)));
        }
        regions[c].size   = chunks[c].size;
        regions[c].layout = chunks[c].layout;
        bufs.push_back(buf);
    }
    share_place(ctx, chunks, bufs);
    shared_.push_back(std::move(regions));
    return share_wrap(bufs);
}

bool llama_kv_cache::share_from(const llama_kv_cache & src) {
    if (other || src.other || src.shared_.size() != ctxs_bufs.size()) {
        LLAMA_LOG_ERROR("%s: the source KV cache has %zu shared buffers, this one has %zu\n", __func__, src.shared_.size(), ctxs_bufs.size());
        return false;
    }
    using import_fn = ggml_backend_buffer_t (*)(ggml_backend_dev_t, int, size_t, size_t);
    for (size_t i = 0; i < ctxs_bufs.size(); ++i) {
        ggml_context * ctx = ctxs_bufs[i].first.get();
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(ctxs_bufs[i].second.get());
        const auto & regions = src.shared_[i];
        const std::vector<share_chunk> chunks = share_layout(ctx, buft);
        bool same = chunks.size() == regions.size();
        for (size_t c = 0; same && c < chunks.size(); ++c) {
            same = chunks[c].size == regions[c].size && chunks[c].layout == regions[c].layout;
        }
        if (!same) {
            LLAMA_LOG_ERROR("%s: KV layout differs from the source (K/V types, shapes or flash attention)\n", __func__);
            return false;
        }
        auto import_dmabuf = (import_fn) share_proc(buft, "ggml_backend_dev_buffer_from_dmabuf");
        std::vector<ggml_backend_buffer_t> bufs;
        size_t total = 0;
        for (const auto & region : regions) {
            ggml_backend_buffer_t buf = import_dmabuf ? import_dmabuf(ggml_backend_buft_get_device(buft), region.fd, region.offset, region.size) : nullptr;
            if (buf == nullptr) {
                for (auto * b : bufs) {
                    ggml_backend_buffer_free(b);
                }
                LLAMA_LOG_ERROR("%s: %s cannot import the shared KV region\n", __func__, ggml_backend_buft_name(buft));
                return false;
            }
            bufs.push_back(buf);
            total += region.size;
        }
        share_place(ctx, chunks, bufs);
        ctxs_bufs[i].second.reset(share_wrap(bufs)); // frees the buffer this cache had
        LLAMA_LOG_INFO("%s: %10s KV buffer now maps the shared region (%.2f MiB in %zu chunks, zero copy)\n", __func__,
                       ggml_backend_buft_name(buft), total / 1024.0 / 1024.0, regions.size());
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
    if (!share_each(dst, src, [](llama_kv_cache & d, const llama_kv_cache & s) { return d.share_from(s); })) {
        return false;
    }
    dst->kv_memory_rebound();
    return true;
}

bool llama_kv_cells_copy(llama_context * dst, const llama_context * src) {
    return share_each(dst, src, [](llama_kv_cache & d, const llama_kv_cache & s) { return d.cells_copy_from(s); });
}
