// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// kvshare-layout-roundtrip: the provider's validation path, with no GPU (engine#753).
//
// A provider's first real step is `llama_kv_share_import`, which checks each region's `layout` hash against what the
// context would have allocated and then maps the dma-bufs over its KV tensors. Gating that needs a region with the right
// hash, and until now the only way to get one was to *export* from a real decoder -- ggml-hrx, i.e. a GPU.
//
// `llama_kv_share_layout` reports the chunks a context EXPECTS (size + hash) with no exporter in the loop, so this tool can
// fabricate them: a udmabuf dma-buf per chunk (a sealed memfd, no GPU in the path), carrying the hash the context just
// reported. Then the import must succeed -- and must fail when one hash is corrupted, which is the half that shows the
// check is live rather than vacuous.
//
//   kvshare-layout-roundtrip -m model.gguf [--ctx N]
//
// Needs `/dev/udmabuf` (root:kvm on this box).
#include "llama.h"
#include "llama-ext.h"

#include <linux/memfd.h>
#include <linux/udmabuf.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int fails = 0;
void check(const char * what, bool ok, const std::string & why = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, why.empty() ? "" : " -- ", why.c_str());
    if (!ok) fails++;
}

// one dma-buf per chunk: a sealed memfd handed to /dev/udmabuf, which is a dma-buf an importer can pin like any other
struct Udmabuf {
    int fd = -1;
    void * map = nullptr;
    size_t size = 0;
    bool create(size_t n) {
        size = n;
        const int mem = int(syscall(SYS_memfd_create, "kvshare-layout", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        if (mem < 0 || ftruncate(mem, off_t(n)) != 0 ||
            fcntl(mem, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL) != 0) {
            return false;
        }
        const int udma = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        if (udma < 0) { close(mem); return false; }
        struct udmabuf_create c = {};
        c.memfd = uint32_t(mem);
        c.flags = UDMABUF_FLAGS_CLOEXEC;
        c.size = uint64_t(n);
        fd = ioctl(udma, UDMABUF_CREATE, &c);
        close(udma);
        close(mem);
        if (fd < 0) return false;
        map = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        return map != MAP_FAILED;
    }
    ~Udmabuf() {
        if (map && map != MAP_FAILED) munmap(map, size);
        if (fd >= 0) close(fd);
    }
};

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    uint32_t n_ctx = 512;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = uint32_t(std::atoi(argv[++i]));
    }
    if (model_path.empty()) { std::fprintf(stderr, "usage: %s -m model.gguf [--ctx N]\n", argv[0]); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 3; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // the serving route's setting
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create a context\n"); return 4; }

    // 1. what does this context expect?
    const size_t n = llama_kv_share_layout(ctx, nullptr, 0);
    std::printf("kvshare-layout-roundtrip: model=%s n_ctx=%u chunks=%zu\n", model_path.c_str(), n_ctx, n);
    check("the context reports a layout before exporting anything", n > 0);
    std::vector<llama_kv_share_region> regions(n ? n : 1);
    const size_t got = llama_kv_share_layout(ctx, regions.data(), regions.size());
    check("sizing first then filling agrees", got == n, std::to_string(got) + " vs " + std::to_string(n));

    // 2. fabricate regions of exactly those sizes carrying exactly those hashes
    std::vector<Udmabuf> bufs(n);
    bool made = true;
    for (size_t i = 0; i < n; i++) {
        if (!bufs[i].create(regions[i].size)) { std::perror("udmabuf"); made = false; break; }
        regions[i].fd = bufs[i].fd;
        regions[i].offset = 0;
        std::memset(bufs[i].map, 0, bufs[i].size);
    }
    check("a udmabuf was created for every chunk (no GPU in the path)", made);
    if (!made) return 5;

    // 3. the provider's first real step: the import must accept them
    check("llama_kv_share_import accepts the fabricated regions",
          llama_kv_share_import(ctx, regions.data(), regions.size()));

    // 4. and the table the writer takes must now describe that imported memory
    const size_t tensors = llama_kv_share_tensor_map(ctx, nullptr, 0);
    std::vector<llama_kv_share_tensor> table(tensors ? tensors : 1);
    const size_t filled = llama_kv_share_tensor_map(ctx, table.data(), table.size());
    check("the tensor table is reported for the imported region", filled > 0 && filled == tensors,
          std::to_string(filled) + " tensors");

    // 5. the control: one corrupted hash must be refused, or step 3 proved nothing about the check
    {
        llama_context * c2 = llama_init_from_model(model, cp);
        std::vector<llama_kv_share_region> bad(n);
        llama_kv_share_layout(c2, bad.data(), bad.size());
        for (size_t i = 0; i < n; i++) { bad[i].fd = bufs[i].fd; bad[i].offset = 0; }
        bad[0].layout ^= 1;
        check("a corrupted layout hash is refused", !llama_kv_share_import(c2, bad.data(), bad.size()));
        llama_free(c2);
    }

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return fails ? 1 : 0;
}
