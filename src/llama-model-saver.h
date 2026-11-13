#pragma once

#include "gguf.h"
#include "llama.h"
#include "llama-arch.h"

#include <string>
#include <unordered_set>
#include <vector>

struct llama_model_saver {
    struct gguf_context * gguf_ctx = nullptr;
    const bool gguf_ctx_owned = true;   // ours: only free a context this saver created
    const struct llama_model * model;
    const struct LLM_KV llm_kv;

    llama_model_saver(const struct llama_model * model);
    // ours: build a saver over an empty gguf context (test-lora-inplace uses it)
    llama_model_saver(enum llm_arch arch, struct gguf_context * gguf_ctx);
    ~llama_model_saver();

    void add_kv(enum llm_kv key, uint32_t     value);
    void add_kv(enum llm_kv key, int32_t      value);
    void add_kv(enum llm_kv key, uint64_t     value);
    void add_kv(enum llm_kv key, float        value);
    void add_kv(enum llm_kv key, bool         value);
    void add_kv(enum llm_kv key, const char * value);

    [[noreturn]]
    void add_kv(enum llm_kv key, char value); // needed to make the template below compile

    template <typename Container>
    void add_kv(enum llm_kv key, const Container & value, bool per_layer = false);

    void add_kv(enum llm_kv key, const std::vector<std::string> & value);

    void add_tensor(const struct ggml_tensor * tensor);

    // tensors already written, so a shared submodule reached from several layers (Zamba v1's shared
    // transformer block) is written once instead of tripping the duplicate-name check
    std::unordered_set<const struct ggml_tensor *> added_tensors;

    void add_kv_from_model();

    void add_tensors_from_model();

    void save(const std::string & path_model);
};
