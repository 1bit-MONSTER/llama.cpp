// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// kvshare-adopt-check: does a decoder actually work from K/V an NPU prefill wrote? (engine#753)
//
// This is the semantic half of the handoff. Byte equality of the rows is one claim (#39/#40 prove it); this is the other:
// a context that IMPORTS the fabricated region, ADOPTS the cells and then decodes must reproduce the prediction of the
// route that computed the rows -- not the prediction of some other implementation of the same model, which is a different
// quantisation and would disagree by design (measured: 11853 against 861, with nothing wrong).
//
//   kvshare-adopt-check -m model.gguf --rows DIR --writer PATH --elf PATH
//
// DIR is a dump written by the NPU prefill: `tokens.i32` (the prefix it prefilled), `kv-layerNN.bin` (one file per layer),
// and the two files that make this comparison possible -- `probe.i32`, the token that route fed at position N, and
// `probe_pred.i32`, what it predicted from there. The rows are written into the region by `--writer` (a separate command, so
// the NPU work stays under its own hold), then the decoder is asked the same question.
//
// bf16 on both sides: the writer moves 16-bit pieces with no conversion, so the cache must BE bf16 or this measures a
// reinterpretation rather than the array's arithmetic (docs/npu-dx-kvwrite.md). Needs `/dev/udmabuf`.
#include "llama.h"
#include "llama-ext.h"

#include <linux/memfd.h>
#include <linux/udmabuf.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int fails = 0;
void check(const char * what, bool ok, const std::string & why = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, why.empty() ? "" : " -- ", why.c_str());
    if (!ok) fails++;
}

struct Udmabuf {
    int fd = -1;
    void * map = nullptr;
    size_t size = 0;
    bool create(size_t n) {
        size = n;
        const int mem = int(syscall(SYS_memfd_create, "kvshare-adopt", MFD_ALLOW_SEALING | MFD_CLOEXEC));
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

std::vector<char> slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>{std::istreambuf_iterator<char>(f), {}};
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path, rows_dir, writer, elf;
    uint32_t n_ctx = 512;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) rows_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--writer") && i + 1 < argc) writer = argv[++i];
        else if (!std::strcmp(argv[i], "--elf") && i + 1 < argc) elf = argv[++i];
    }
    if (model_path.empty() || rows_dir.empty() || writer.empty() || elf.empty()) {
        std::fprintf(stderr, "usage: %s -m model.gguf --rows DIR --writer PATH --elf PATH [--ctx N]\n", argv[0]);
        return 2;
    }

    // the dump's own description: the prefix it prefilled and the prediction to reproduce
    const std::vector<char> tok_raw = slurp(rows_dir + "/tokens.i32");
    const std::vector<char> probe_raw = slurp(rows_dir + "/probe.i32");
    const std::vector<char> pred_raw = slurp(rows_dir + "/probe_pred.i32");
    const size_t P = tok_raw.size() / 4;
    check("the dump names its prefix (tokens.i32)", P >= 2, std::to_string(P) + " tokens");
    check("the dump names its probe and prediction", probe_raw.size() == 4 && pred_raw.size() == 4);
    if (P < 2 || probe_raw.size() != 4 || pred_raw.size() != 4) return 3;
    std::vector<int32_t> tok(P);
    std::memcpy(tok.data(), tok_raw.data(), P * 4);
    int32_t probe = 0, want = 0;
    std::memcpy(&probe, probe_raw.data(), 4);
    std::memcpy(&want, pred_raw.data(), 4);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 4; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_BF16;   // the writer emits bf16 and converts nothing
    cp.type_v = GGML_TYPE_BF16;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create a context\n"); return 5; }

    // the region the decoder expects, and the table the writer needs -- both without an exporter
    const size_t n = llama_kv_share_layout(ctx, nullptr, 0);
    std::vector<llama_kv_share_region> regions(n ? n : 1);
    check("the context reports its layout", n > 0 && llama_kv_share_layout(ctx, regions.data(), regions.size()) == n);
    std::vector<Udmabuf> bufs(n);
    bool made = n > 0;
    for (size_t i = 0; made && i < n; i++) {
        made = bufs[i].create(regions[i].size);
        regions[i].fd = bufs[i].fd;
        regions[i].offset = 0;
        if (made) std::memset(bufs[i].map, 0, bufs[i].size);
    }
    check("a udmabuf exists for every chunk (no GPU in the path)", made);
    if (!made) return 6;
    check("the import accepts them", llama_kv_share_import(ctx, regions.data(), regions.size()));

    const size_t nt = llama_kv_share_tensor_map(ctx, nullptr, 0);
    std::vector<llama_kv_share_tensor> table(nt ? nt : 1);
    check("the tensor table describes the imported region",
          llama_kv_share_tensor_map(ctx, table.data(), table.size()) == nt && nt > 0,
          std::to_string(nt) + " tensors");

    // write the NPU prefill's rows into those tensors: one call to the writer per layer, with the fds inherited
    int wrote = 0, failed = 0;
    for (size_t t = 0; t + 1 < table.size(); t += 2) {
        // the dump names its files zero-padded (kv-layer00.bin); an unpadded name silently skipped layers 0-9 the first
        // time this ran, and "18 written, 0 failed" is exactly how that looked
        char rows_name[64];
        std::snprintf(rows_name, sizeof rows_name, "%s/kv-layer%02u.bin", rows_dir.c_str(), table[t].layer);
        const std::string rows_path = rows_name;
        if (slurp(rows_path).empty()) continue;
        for (size_t i = 0; i < n; i++) fcntl(bufs[i].fd, F_SETFD, 0);
        char cmd[8192];
        std::snprintf(cmd, sizeof cmd,
                      "%s --fd %d --bytes %zu --elf %s "
                      "--tensor %u:K:%llu:%llu:%llu --tensor %u:V:%llu:%llu:%llu "
                      "--write %u:0:%zu:%llu:%llu:%s",
                      writer.c_str(), bufs[table[t].region].fd, regions[table[t].region].size, elf.c_str(),
                      table[t].layer, (unsigned long long) table[t].offset, (unsigned long long) table[t].bytes,
                      (unsigned long long) table[t].row_elems, table[t + 1].layer,
                      (unsigned long long) table[t + 1].offset, (unsigned long long) table[t + 1].bytes,
                      (unsigned long long) table[t + 1].row_elems, table[t].layer, P,
                      (unsigned long long) table[t].row_elems, (unsigned long long) table[t + 1].row_elems,
                      rows_path.c_str());
        const int rc = std::system(cmd);
        if (rc == 0) wrote++; else { if (failed++ == 0) std::printf("     first failure: %s\n", cmd); }
    }
    check("every layer's rows were written into the imported region", failed == 0 && wrote > 0,
          std::to_string(wrote) + " written, " + std::to_string(failed) + " failed");
    check("the decoder adopts the cells the NPU filled", llama_kv_cells_adopt(ctx, uint32_t(P), 0));

    // the same question the NPU route answered: given `probe` at position P, which token comes next?
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1;
    b.token[0] = probe;
    b.pos[0] = llama_pos(P);
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    int32_t got = -1;
    if (llama_decode(ctx, b) == 0) {
        const float * lg = llama_get_logits_ith(ctx, 0);
        got = int32_t(std::max_element(lg, lg + llama_vocab_n_tokens(llama_model_get_vocab(model))) - lg);
    }
    llama_batch_free(b);
    check("a decoder on the NPU's K/V reproduces the NPU's own next token", got >= 0 && got == want,
          "decoder=" + std::to_string(got) + " npu=" + std::to_string(want));

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return fails ? 1 : 0;
}
