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
// Tests the in-place LoRA factor write (llama_adapter_lora_set_factors).
//
// The test builds a tiny llama model in memory and two rank-2 adapter GGUFs on disk,
// so no model files are needed. Both adapter files are removed before the in-place
// update: a re-parse would fail, so a successful update proves that only the resident
// tensors were written. The test also checks that a refused update leaves the factors
// and the logits untouched.

#include "llama.h"

#include "llama-adapter.h"
#include "llama-arch.h"
#include "llama-model-saver.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

static constexpr uint32_t n_vocab = 64;
static constexpr uint32_t n_embd  = 32;
static constexpr uint32_t n_head  = 2;
static constexpr uint32_t n_ff    = 64;
static constexpr uint32_t n_layer = 1;
static constexpr uint32_t n_ctx   = 16;
static constexpr uint32_t n_rank  = 2;

static constexpr float adapter_alpha = 2.0f; // rank 2 -> scale 1.0

static const char * const tensor_attn = "blk.0.attn_q.weight";
static const char * const tensor_ffn  = "blk.0.ffn_down.weight";

static const llama_token tokens[] = { 1, 2, 3, 4 };
static constexpr int n_tokens = sizeof(tokens)/sizeof(tokens[0]);

#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return 1; \
        } \
    } while (0)

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double mse_ab = 0.0;
    double mse_a0 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        mse_ab += (double) (a[i] - b[i])*(a[i] - b[i]);
        mse_a0 += (double) a[i]*a[i];
    }
    return mse_a0 > 0.0 ? mse_ab/mse_a0 : mse_ab;
}

static bool equal_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static struct gguf_context * make_model_gguf() {
    struct gguf_context * ctx = gguf_init_empty();

    llama_model_saver ms(LLM_ARCH_LLAMA, ctx);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, "llama");
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd/n_head);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,           "no_vocab");

    return ctx;
}

// deterministic small weights per tensor name
static void set_model_tensor_data(struct ggml_tensor * t, void * userdata) {
    const size_t seed = *(const size_t *) userdata;

    std::hash<std::string> hasher;
    std::mt19937 gen((uint32_t) (seed ^ hasher(std::string(t->name))));
    std::normal_distribution<float> dis(0.0f, 0.1f);

    // norm weights stay near 1, like a trained model
    const bool is_norm = strstr(t->name, "norm") != nullptr;

    if (t->type == GGML_TYPE_F32) {
        std::vector<float> data(ggml_nelements(t));
        for (auto & v : data) {
            v = is_norm ? 1.0f + 0.01f*dis(gen) : dis(gen);
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data(ggml_nelements(t));
        for (auto & v : data) {
            v = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> data(ggml_nbytes(t), 0);
        ggml_backend_tensor_set(t, data.data(), 0, data.size());
    }
}

static bool silent_progress(float, void *) {
    return true;
}

static std::vector<float> read_f32_tensor(struct ggml_tensor * t) {
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    return data;
}

static struct llama_model * make_model() {
    gguf_context_ptr ctx(make_model_gguf());

    llama_model_params params = llama_model_default_params();
    params.progress_callback = silent_progress;
    params.n_gpu_layers = 0;

    size_t seed = 1;
    return llama_model_init_from_user(ctx.get(), set_model_tensor_data, &seed, params);
}

struct adapter_pair {
    std::string name;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    std::vector<float> a;
    std::vector<float> b;
};

static void add_f32_tensor(struct gguf_context * ctx, const std::string & name, int64_t ne0, int64_t ne1, const std::vector<float> & data) {
    ggml_tensor t;
    memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = t.nb[0]*ne0;
    t.nb[2] = t.nb[1]*ne1;
    t.nb[3] = t.nb[2];
    ggml_set_name(&t, name.c_str());

    gguf_add_tensor(ctx, &t);
    gguf_set_tensor_data(ctx, name.c_str(), data.data());
}

static std::vector<float> make_factors(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 0.3f);
    std::vector<float> data(n);
    for (auto & v : data) {
        v = dis(gen);
    }
    return data;
}

// rank-2 factors for the two weights the test updates
static adapter_pair make_pair(const std::string & name, int64_t ne0, int64_t ne1, uint32_t seed) {
    adapter_pair p;
    p.name = name;
    p.ne0 = ne0;
    p.ne1 = ne1;
    p.a = make_factors((size_t) ne0*n_rank, seed + 1);
    p.b = make_factors((size_t) ne1*n_rank, seed + 2);
    return p;
}

static void write_adapter_gguf(const std::string & path, const std::vector<adapter_pair> & pairs) {
    gguf_context_ptr ctx(gguf_init_empty());

    gguf_set_val_str(ctx.get(), "general.type", "adapter");
    gguf_set_val_str(ctx.get(), "general.architecture", "llama");
    gguf_set_val_str(ctx.get(), "adapter.type", "lora");
    gguf_set_val_f32(ctx.get(), "adapter.lora.alpha", adapter_alpha);

    for (const auto & p : pairs) {
        add_f32_tensor(ctx.get(), p.name + ".lora_a", p.ne0, n_rank, p.a);
        add_f32_tensor(ctx.get(), p.name + ".lora_b", n_rank, p.ne1, p.b);
    }

    if (!gguf_write_to_file(ctx.get(), path.c_str(), false)) {
        fprintf(stderr, "FAIL: could not write %s\n", path.c_str());
        GGML_ABORT("gguf write failed");
    }
}

static struct llama_context * make_ctx(struct llama_model * model) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_tokens;
    params.n_ubatch = n_tokens;
    params.n_seq_max = 1;
    params.n_threads = 2;
    params.n_threads_batch = 2;

    return llama_init_from_model(model, params);
}

static std::vector<float> eval(struct llama_context * ctx) {
    // the probe re-runs the same positions, so the KV of the previous probe must go;
    // the graph and the adapter set stay as they are
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        GGML_ABORT("llama_decode failed");
    }

    std::vector<float> logits(n_vocab);
    memcpy(logits.data(), llama_get_logits_ith(ctx, n_tokens - 1), n_vocab*sizeof(float));

    llama_batch_free(batch);
    return logits;
}

int main() {
    llama_backend_init();

    // both adapter files are removed after loading and before any in-place update:
    // a re-parse would fail, so a successful update can only have written the tensors
    const std::string path1 = "test-lora-inplace-v1.gguf";
    const std::string path2 = "test-lora-inplace-v2.gguf";

    // generation 1: both weights; generation 2: attn_q only, ffn_down stays at 1
    const adapter_pair a1_attn = make_pair(tensor_attn, n_embd, n_embd, 100);
    const adapter_pair a1_ffn  = make_pair(tensor_ffn,  n_ff,   n_embd, 200);
    const adapter_pair a2_attn = make_pair(tensor_attn, n_embd, n_embd, 300);

    write_adapter_gguf(path1, { a1_attn, a1_ffn });
    {
        adapter_pair a2_ffn = a1_ffn;
        write_adapter_gguf(path2, { a2_attn, a2_ffn });
    }

    struct llama_model * model = make_model();
    CHECK(model != nullptr);

    struct llama_adapter_lora * adapter1 = llama_adapter_lora_init(model, path1.c_str());
    CHECK(adapter1 != nullptr);
    struct llama_adapter_lora * adapter2 = llama_adapter_lora_init(model, path2.c_str());
    CHECK(adapter2 != nullptr);

    // the model keeps its own reference to the loaded adapters; free them at the end
    CHECK(llama_adapter_lora_n_factors(adapter1) == 2);

    {
        // the loaded factors must hold the values written to the GGUF
        const std::vector<float> a = read_f32_tensor(adapter1->ab_map.at(tensor_attn).a);
        const std::vector<float> b = read_f32_tensor(adapter1->ab_map.at(tensor_attn).b);
        CHECK(a.size() == a1_attn.a.size());
        CHECK(b.size() == a1_attn.b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i] == a1_attn.a[i]);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            CHECK(b[i] == a1_attn.b[i]);
        }
    }

    // (c) structural evidence for "in place": the tensors keep their addresses
    ggml_tensor * attn_a_before = adapter1->ab_map.at(tensor_attn).a;
    ggml_tensor * attn_b_before = adapter1->ab_map.at(tensor_attn).b;

    // structural evidence for "no GGUF parse": the files are gone before the update
    std::filesystem::remove(path1);
    std::filesystem::remove(path2);
    CHECK(!std::filesystem::exists(path1));
    CHECK(!std::filesystem::exists(path2));

    // without adapter
    struct llama_context * ctx = make_ctx(model);
    CHECK(ctx != nullptr);
    const std::vector<float> logits_base = eval(ctx);

    // (a) apply the adapter: the logits move
    float scale = 1.0f;
    CHECK(llama_set_adapters_lora(ctx, &adapter1, 1, &scale) == 0);
    const std::vector<float> logits1 = eval(ctx);
    {
        double dmax = 0.0; double lmax = 0.0;
        for (size_t i = 0; i < logits_base.size(); ++i) {
            dmax = std::max(dmax, (double) std::fabs(logits1[i] - logits_base[i]));
            lmax = std::max(lmax, (double) std::fabs(logits_base[i]));
        }
        printf("applied: nmse = %.3e dmax = %.3e logits_absmax = %.3e\n", nmse(logits1, logits_base), dmax, lmax);
    }
    CHECK(nmse(logits1, logits_base) > 1e-6);

    // (b) rewrite one weight in place; the same context picks the new values up
    CHECK(llama_adapter_lora_set_factors(adapter1, tensor_attn,
                a2_attn.a.data(), a2_attn.a.size()*sizeof(float),
                a2_attn.b.data(), a2_attn.b.size()*sizeof(float)) == 0);

    CHECK(adapter1->ab_map.at(tensor_attn).a == attn_a_before);
    CHECK(adapter1->ab_map.at(tensor_attn).b == attn_b_before);

    const std::vector<float> logits2 = eval(ctx);
    CHECK(nmse(logits2, logits1) > 1e-6);
    printf("updated: nmse(logits2, logits1) = %.3e\n", nmse(logits2, logits1));

    // the in-place write must give the same result as a fresh load of the same factors
    struct llama_context * ctx2 = make_ctx(model);
    CHECK(ctx2 != nullptr);
    CHECK(llama_set_adapters_lora(ctx2, &adapter2, 1, &scale) == 0);
    const std::vector<float> logits_expected = eval(ctx2);
    printf("updated vs freshly loaded: nmse = %.3e\n", nmse(logits2, logits_expected));
    CHECK(nmse(logits2, logits_expected) < 1e-6);
    CHECK(!equal_bits(logits2, logits_base));

    // (d) refused updates keep the old values
    const std::vector<float> logits_before_refusal = logits2;

    CHECK(llama_adapter_lora_set_factors(adapter1, "blk.0.attn_k.weight",
                a2_attn.a.data(), a2_attn.a.size()*sizeof(float),
                a2_attn.b.data(), a2_attn.b.size()*sizeof(float)) == -2);
    CHECK(llama_adapter_lora_set_factors(adapter1, tensor_attn,
                a2_attn.a.data(), a2_attn.a.size()*sizeof(float) - sizeof(float),
                a2_attn.b.data(), a2_attn.b.size()*sizeof(float)) == -3);
    CHECK(llama_adapter_lora_set_factors(adapter1, tensor_attn, nullptr, 0, nullptr, 0) == -1);
    CHECK(llama_adapter_lora_set_factors(adapter1, nullptr,
                a2_attn.a.data(), a2_attn.a.size()*sizeof(float),
                a2_attn.b.data(), a2_attn.b.size()*sizeof(float)) == -1);

    // a refused batch writes nothing, not even its valid entry
    const llama_adapter_lora_factor batch_refused[] = {
        { tensor_attn, a1_attn.a.data(), a1_attn.a.size()*sizeof(float), a1_attn.b.data(), a1_attn.b.size()*sizeof(float) },
        { "blk.0.attn_v.weight", a1_attn.a.data(), a1_attn.a.size()*sizeof(float), a1_attn.b.data(), a1_attn.b.size()*sizeof(float) },
    };
    CHECK(llama_adapter_lora_set_factors_multi(adapter1, batch_refused, 2) == -2);

    CHECK(adapter1->ab_map.at(tensor_attn).a == attn_a_before);
    CHECK(adapter1->ab_map.at(tensor_attn).b == attn_b_before);

    const std::vector<float> logits_after_refusal = eval(ctx);
    CHECK(equal_bits(logits_after_refusal, logits_before_refusal));
    printf("refused update: logits unchanged\n");

    // rewrite every factor of the adapter in one call: enumerate the names first
    std::vector<std::string> names;
    names.reserve(llama_adapter_lora_n_factors(adapter1));
    for (size_t i = 0; i < llama_adapter_lora_n_factors(adapter1); ++i) {
        char name[128] = {0};
        CHECK(llama_adapter_lora_factor_name_by_index(adapter1, i, name, sizeof(name)) > 0);
        names.emplace_back(name);
    }
    CHECK(names.size() == 2);
    CHECK(llama_adapter_lora_factor_name_by_index(adapter1, 2, nullptr, 0) == -1);

    std::vector<llama_adapter_lora_factor> all;
    all.reserve(names.size());
    for (const auto & name : names) {
        if (name == tensor_attn) {
            all.push_back({ name.c_str(), a1_attn.a.data(), a1_attn.a.size()*sizeof(float), a1_attn.b.data(), a1_attn.b.size()*sizeof(float) });
        } else if (name == tensor_ffn) {
            all.push_back({ name.c_str(), a1_ffn.a.data(),  a1_ffn.a.size()*sizeof(float),  a1_ffn.b.data(),  a1_ffn.b.size()*sizeof(float)  });
        } else {
            fprintf(stderr, "FAIL: unexpected adapter tensor '%s'\n", name.c_str());
            return 1;
        }
    }
    CHECK(llama_adapter_lora_set_factors_multi(adapter1, all.data(), all.size()) == 0);

    // the factors are back to the values the adapter was loaded with, so the result
    // must be identical to the first application
    const std::vector<float> logits3 = eval(ctx);
    CHECK(equal_bits(logits3, logits1));
    CHECK(nmse(logits3, logits2) > 1e-6);

    llama_free(ctx2);
    llama_free(ctx);

    llama_adapter_lora_free(adapter2);
    llama_adapter_lora_free(adapter1);
    llama_model_free(model);
    llama_backend_free();

    printf("OK: in-place adapter factor update\n");
    return 0;
}
