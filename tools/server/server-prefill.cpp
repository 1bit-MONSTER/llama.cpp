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

#include "server-prefill.h"

#include "../../src/llama-ext.h"
#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <stdexcept>

bool server_prefill_device::load(const char * device_name, common_params & params, llama_context * ctx_tgt) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_name);
    if (dev == nullptr) {
        LOG_WRN("prefill device '%s' not found, prefill stays on the main device\n", device_name);
        return false;
    }
    if (!params.lora_adapters.empty()) {
        LOG_WRN("%s", "prefill device is not used with LoRA adapters\n");
        return false;
    }
    if (const char * v = getenv("ONEBIT_PREFILL_MIN_TOKENS")) {
        min_tokens = std::max(1, atoi(v));
    }
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    llama_model_params mparams = common_model_params_to_llama(params);
    mparams.devices           = devs;
    mparams.progress_callback = nullptr;
    model.reset(llama_model_load_from_file(params.model.path.c_str(), mparams));
    if (!model) {
        LOG_WRN("failed to load the model on prefill device '%s'\n", device_name);
        return false;
    }
    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_ctx     = llama_n_ctx(ctx_tgt);
    cparams.n_batch   = llama_n_batch(ctx_tgt);
    cparams.n_ubatch  = llama_n_ubatch(ctx_tgt);
    cparams.n_seq_max = llama_n_seq_max(ctx_tgt);
    llama_kv_share_next(true); // this context owns the KV region; ctx_tgt maps it
    try {
        ctx.reset(llama_init_from_model(model.get(), cparams));
    } catch (const std::exception & e) {
        LOG_WRN("prefill context: %s\n", e.what());
    }
    if (!ctx || !llama_kv_share_from(ctx_tgt, ctx.get()) || !llama_kv_cells_copy(ctx.get(), ctx_tgt)) {
        LOG_WRN("prefill device '%s' cannot share the KV cache (set -fa on explicitly so both devices lay it out the same way)\n", device_name);
        reset();
        return false;
    }
    LOG_INF("prompt prefill on %s (from %d tokens), decode on the main device; KV cache shared, zero copy\n", device_name, min_tokens);
    return true;
}

int32_t server_prefill_device::prefill(llama_context * ctx_tgt, const llama_batch & view) {
    if (!ctx || view.embd != nullptr || view.logits == nullptr) {
        return 0;
    }
    int32_t k = 0;
    while (k < view.n_tokens && !view.logits[k]) {
        k++;
    }
    // HRX prefill halves on a partial last ubatch (Qwen2.5-7B: 1085 tok/s at 2047 tokens, 2246 at 2048): whole
    // ubatches only, the remainder runs on the main device with the token that needs logits
    k -= k % (int32_t) llama_n_ubatch(ctx.get());
    if (k < min_tokens || !llama_kv_cells_copy(ctx.get(), ctx_tgt)) {
        return 0;
    }
    llama_batch prefix = view;
    prefix.n_tokens    = k;
    if (llama_decode(ctx.get(), prefix) != 0) {
        LOG_WRN("prefill device decode failed (n = %d), using the main device\n", k);
        return 0;
    }
    llama_synchronize(ctx.get());
    if (!llama_kv_cells_copy(ctx_tgt, ctx.get())) {
        throw std::runtime_error("prefill device: KV cells handoff failed");
    }
    return k;
}
