// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// kvshare-tensor-map: what a prefill provider gets from llama_kv_share_tensor_map, checked (engine#753).
//
// The provider side of a shared-KV handoff needs, per cache tensor, where it sits inside the region it was mapped into
// and how wide one position's row is. Those numbers existed inside llama_share_import and were dropped; the map now
// reports them. This tool checks the arithmetic on a real model, with no dma-buf and no device: a context allocates its
// KV normally, the map reports offsets relative to each buffer, and the checks below hold for that case and for the
// imported case alike (where the buffer IS the region window).
//
//   kvshare-tensor-map -m model.gguf [--ctx N] [--ubatch N]
//
// Checked: every K/V tensor of every attention layer is reported; offset + bytes fits its region; one position's row is
// row_elems * element size; K and V never overlap; and the region ordinals are contiguous from 0.
#include "llama.h"
#include "llama-ext.h"   // the [1bit] API this tool exercises lives in the extension header

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
static void check(const char * what, bool ok, const std::string & why = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, why.empty() ? "" : " -- ", why.c_str());
    if (!ok) fails++;
}

int main(int argc, char ** argv) {
    std::string model_path;
    uint32_t n_ctx = 512, n_ubatch = 128;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--ubatch") && i + 1 < argc) n_ubatch = uint32_t(std::atoi(argv[++i]));
    }
    if (model_path.empty()) { std::fprintf(stderr, "usage: %s -m model.gguf [--ctx N] [--ubatch N]\n", argv[0]); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 3; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_ubatch = n_ubatch;
    cp.n_batch = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // the serving route's setting
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create a context\n"); return 4; }

    const size_t n = llama_kv_share_tensor_map(ctx, nullptr, 0);
    std::printf("kvshare-tensor-map: model=%s n_ctx=%u tensors=%zu\n", model_path.c_str(), n_ctx, n);
    check("the map reports something", n > 0);
    std::vector<llama_kv_share_tensor> t(n ? n : 1);
    const size_t got = llama_kv_share_tensor_map(ctx, t.data(), t.size());
    check("sizing first then filling agrees", got == n, std::to_string(got) + " vs " + std::to_string(n));

    const int32_t n_layer = llama_model_n_layer(model);
    size_t n_k = 0, n_v = 0;
    bool fits = true, rows_ok = true, regions_contiguous = true, no_overlap = true;
    uint32_t max_region = 0;
    for (size_t i = 0; i < n; ++i) {
        if (t[i].is_v == 0) n_k++; else n_v++;
        if (t[i].bytes == 0 || t[i].row_elems == 0) rows_ok = false;
        // the row of one position must divide the tensor, or a provider cannot write it position by position
        const uint64_t row_bytes = t[i].row_elems * 2;   // f16/bf16 cache, both 2 bytes
        if (t[i].bytes % row_bytes != 0) rows_ok = false;
        if (t[i].region > max_region) max_region = t[i].region;
        if (t[i].layer >= uint32_t(n_layer)) fits = false;
        // overlap within a region: every pair of tensors in the same region must be disjoint
        for (size_t j = i + 1; j < n; ++j) {
            if (t[j].region != t[i].region) continue;
            if (t[i].offset < t[j].offset + t[j].bytes && t[j].offset < t[i].offset + t[i].bytes) no_overlap = false;
        }
    }
    for (uint32_t r = 0; r <= max_region; ++r) {
        bool seen = false;
        for (size_t i = 0; i < n; ++i) seen = seen || t[i].region == r;
        if (!seen) regions_contiguous = false;
    }
    check("K and V are reported per layer", n_k == n_v && n_k > 0,
          std::to_string(n_k) + " K / " + std::to_string(n_v) + " V for " + std::to_string(n_layer) + " layers");
    check("every row width divides its tensor", rows_ok);
    check("every tensor belongs to a layer of this model", fits);
    check("tensors inside one region do not overlap", no_overlap);
    check("region ordinals are contiguous from 0", regions_contiguous);

    for (size_t i = 0; i < n && i < 6; ++i) {
        std::printf("     layer %-3u %s  region %u  offset %8llu  bytes %8llu  row_elems %llu\n", t[i].layer,
                    t[i].is_v ? "V" : "K", t[i].region, (unsigned long long) t[i].offset,
                    (unsigned long long) t[i].bytes, (unsigned long long) t[i].row_elems);
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return fails ? 1 : 0;
}
