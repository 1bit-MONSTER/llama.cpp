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
//
// [1bit] TZEROCOPY target 1: the reference prefill provider for a shared KV region.
//
// The decoder (llama-server with ONEBIT_KV_SHARE=1) exports its KV cache as dma-buf regions and hands them to
// this process as inherited fds + a JSON header (`POST /slots/<id>?action=share {"socket": ...}`). This program
// maps those regions into its own context's KV tensors -- no copy, no state file -- prefills the prompt into
// them, and exits. The decoder then takes the cells over with `POST /slots/<id>?action=adopt`.
//
//   kvshare-prefill --model <gguf> --tokens <ids file> --layout <header file> --fds 3,4 [--gpu-layers N]
//   kvshare-prefill --model <gguf> --tokens <ids file> --out <state file>   (the file route)
//
// The two modes exist so one provider can serve both transports: `--fds` prefills into the decoder's own
// KV memory, `--out` prefills into its own and writes the sequence state the decoder restores. Same math,
// same model, same engine -- which is what makes fd-vs-file a controlled comparison.
//
// stdout: one JSON line {"n_tokens": N, "prefill_ms": x, "argmax": id}. exit 0 on success.
//
// The real provider on this box is the NPU runtime (npu/forward), which imports the same descriptors as XDNA BOs
// instead of mmap'ing them. This one exists so the fd plumbing can be gated end to end with the CPU, and so the
// contract between the engine and any provider has a written-down implementation.

#include "llama.h"
#include "../../src/llama-ext.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s --model <gguf> --tokens <ids file> --layout <header file> --fds <n[,n...]>\n"
            "          [--gpu-layers N] [--device NAME]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string tokens_path;
    std::string layout_path;
    std::string fds_arg;
    std::string out_path;
    std::string device;
    int32_t n_gpu_layers = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s: %s needs a value\n", argv[0], what); std::exit(2); }
            return argv[++i];
        };
        if (arg == "--model")       { model_path   = next("--model"); }
        else if (arg == "--tokens") { tokens_path  = next("--tokens"); }
        else if (arg == "--layout") { layout_path  = next("--layout"); }
        else if (arg == "--fds")    { fds_arg      = next("--fds"); }
        else if (arg == "--out")    { out_path      = next("--out"); }
        else if (arg == "--device") { device       = next("--device"); }
        else if (arg == "--gpu-layers") { n_gpu_layers = std::atoi(next("--gpu-layers").c_str()); }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "%s: unknown argument %s\n", argv[0], arg.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || tokens_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    const bool shared = !fds_arg.empty() && !layout_path.empty();
    if (!shared && out_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    // the prompt the decoder tokenized, so both sides prefill exactly the same ids
    std::vector<llama_token> tokens;
    {
        std::ifstream f(tokens_path);
        if (!f) { std::fprintf(stderr, "kvshare-prefill: cannot read %s\n", tokens_path.c_str()); return 1; }
        long v;
        while (f >> v) tokens.push_back((llama_token) v);
    }
    if (tokens.empty()) { std::fprintf(stderr, "kvshare-prefill: %s has no tokens\n", tokens_path.c_str()); return 1; }

    // the header the decoder sent: region descriptors plus the context params its cache was built with
    json  header = json{{"params", json::object()}, {"regions", json::array()}};
    if (shared) try {
        std::ifstream  f(layout_path);
        if (!f) throw std::runtime_error("cannot read " + layout_path);
        f >> header;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "kvshare-prefill: %s\n", e.what());
        return 1;
    }
    const json & params = header.at("params");
    const json & jregions = header.at("regions");

    // the fds: inherited from the engine, which received them over the unix socket from the decoder
    std::vector<int> fds;
    for (size_t pos = 0; pos < fds_arg.size();) {
        const size_t comma = fds_arg.find(',', pos);
        const std::string one = fds_arg.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        fds.push_back(std::atoi(one.c_str()));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (shared && fds.size() != jregions.size()) {
        std::fprintf(stderr, "kvshare-prefill: %zu fd(s) for %zu region(s)\n", fds.size(), jregions.size());
        return 1;
    }

    std::vector<llama_kv_share_region> regions(jregions.size());
    for (size_t i = 0; i < regions.size(); ++i) {
        regions[i].fd     = fds[i];
        regions[i].offset = jregions[i].at("offset").get<size_t>();
        regions[i].size   = jregions[i].at("size").get<size_t>();
        regions[i].layout = jregions[i].at("layout").get<uint64_t>();
    }

    // the model, and a context built the way the decoder's was: the region's layout hash is checked against this
    const auto t_load0 = std::chrono::steady_clock::now();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) { std::fprintf(stderr, "kvshare-prefill: cannot load %s\n", model_path.c_str()); return 1; }

    llama_context_params cparams = llama_context_default_params();
    if (shared) {
        cparams.n_ctx         = params.at("n_ctx").get<uint32_t>();
        cparams.n_batch       = params.at("n_batch").get<uint32_t>();
        cparams.n_ubatch      = params.at("n_ubatch").get<uint32_t>();
        cparams.n_seq_max     = params.at("n_seq_max").get<uint32_t>();
        cparams.flash_attn_type = params.at("flash_attn").get<int32_t>() != 0
                ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.type_k        = (ggml_type) params.at("type_k").get<int32_t>();
        cparams.type_v        = (ggml_type) params.at("type_v").get<int32_t>();
    } else {
        // file route: the state the decoder restores must match its cache, and the decoder runs -fa on
        cparams.n_ctx           = (uint32_t) std::max<size_t>(tokens.size() + 8, 256);
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    cparams.n_threads     = 8;
    cparams.n_threads_batch = 8;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) { std::fprintf(stderr, "kvshare-prefill: cannot create a context\n"); return 1; }

    // shared route: map the decoder's KV memory as this context's KV tensors (zero copy: the tensors ARE it)
    const auto t_import0 = std::chrono::steady_clock::now();
    if (shared && !llama_kv_share_import(ctx, regions.data(), regions.size())) {
        std::fprintf(stderr, "kvshare-prefill: cannot map the shared KV region (layout differs: same model, -fa, "
                             "-c, -b/-ub and KV types as the decoder are required)\n");
        return 1;
    }
    const double import_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_import0).count();
    const double load_ms   = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

    // prefill into it, in chunks of n_ubatch, exactly as a normal prompt prefill would
    const int32_t n_batch_max = (int32_t) std::min<size_t>(cparams.n_batch, tokens.size());
    llama_batch batch = llama_batch_init(n_batch_max, 0, 1);
    const auto t_pre0 = std::chrono::steady_clock::now();
    for (size_t done = 0; done < tokens.size();) {
        const size_t k = std::min<size_t>(cparams.n_ubatch, tokens.size() - done);
        batch.n_tokens = (int32_t) k;
        for (size_t i = 0; i < k; ++i) {
            batch.token[i]     = tokens[done + i];
            batch.pos[i]       = (llama_pos) (done + i);
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (done + i == tokens.size() - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "kvshare-prefill: decode failed at token %zu\n", done);
            return 1;
        }
        done += k;
    }
    llama_synchronize(ctx);
    const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pre0).count();

    // the last token's argmax, so the engine can see that the provider and the decoder agree on the prefix
    int32_t argmax = -1;
    if (const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1)) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        float best = logits[0];
        argmax = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > best) { best = logits[i]; argmax = i; }
        }
    }

    // file route: the same cells, written where the decoder's ?action=restore will read them
    if (!shared) {
        const size_t written = llama_state_seq_save_file(ctx, out_path.c_str(), 0, tokens.data(), tokens.size());
        if (written == 0) {
            std::fprintf(stderr, "kvshare-prefill: cannot write %s\n", out_path.c_str());
            return 1;
        }
    }

    std::printf("{\"n_tokens\": %zu, \"prefill_ms\": %.1f, \"import_ms\": %.3f, \"load_ms\": %.1f, \"argmax\": %d}\n",
            tokens.size(), prefill_ms, import_ms, load_ms, argmax);
    std::fflush(stdout);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
