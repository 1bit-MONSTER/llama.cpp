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

#pragma once

#include "common.h"
#include "llama-cpp.h"
#include "llama.h"

#include <cstdint>

// ONEBIT_PREFILL_DEVICE=<device>: prompt prefill runs on a second device of the same GPU (for example HRX0 while the
// server decodes on Vulkan0). Both contexts use one KV cache (dma-buf, zero copy); only the cell metadata moves.
// ONEBIT_PREFILL_MIN_TOKENS sets the shortest prompt prefix sent there (default 1024: below it the split does not pay).
struct server_prefill_device {
    llama_model_ptr   model;
    llama_context_ptr ctx;
    int32_t           min_tokens = 1024;

    // loads a second copy of the model on device_name; its KV region becomes ctx_tgt's KV cache
    bool load(const char * device_name, common_params & params, llama_context * ctx_tgt);

    void reset() {
        ctx.reset();
        model.reset();
    }

    // runs the prompt prefix of view (the tokens before the first one that needs logits) on this device;
    // returns how many tokens it ran, 0 when view stays on ctx_tgt
    int32_t prefill(llama_context * ctx_tgt, const llama_batch & view);
};
