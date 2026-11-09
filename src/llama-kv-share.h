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

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Zero-copy KV sharing between two contexts of one model on two devices of the same GPU (API in llama-ext.h).
// The owner context allocates each KV buffer as one exportable region with a fixed layout; the other maps it.

struct llama_kv_shared_region {
    int      fd     = -1;
    size_t   offset = 0;
    size_t   size   = 0;
    uint64_t layout = 0; // hash of tensor types and shapes: both sides must lay the cache out the same way

    llama_kv_shared_region() = default;
    llama_kv_shared_region(llama_kv_shared_region && o) noexcept : fd(o.fd), offset(o.offset), size(o.size), layout(o.layout) { o.fd = -1; }
    llama_kv_shared_region(const llama_kv_shared_region &) = delete;
    ~llama_kv_shared_region();
};

// [1bit] target 1, the table a prefill provider needs: which cache tensor sits where inside its region, and how wide
// one position's row is. `share_layout()` computes exactly this and `llama_share_import` maps the regions over the
// tensors; until now the offsets were dropped on the way (engine#753).
struct llama_kv_shared_tensor {
    uint32_t layer     = 0;   // the model's layer index
    uint32_t is_v      = 0;   // 0 = K, 1 = V
    uint32_t region    = 0;   // which entry of the region list this tensor lands in
    uint64_t offset    = 0;   // bytes from the start of that region
    uint64_t bytes     = 0;   // the whole tensor
    uint64_t row_elems = 0;   // elements per position (ne[0]): the row a provider writes
};

// the same region, handed out: a POD, so whoever receives the list does not close the owner's fd
struct llama_kv_shared_region_pod {
    int      fd     = -1;
    size_t   offset = 0;
    size_t   size   = 0;
    uint64_t layout = 0;
};

// the pending llama_kv_share_next request; a real (not no_alloc) context takes and clears it
bool llama_kv_share_take_next(bool no_alloc);

// [1bit] target 1, the two halves of a shared region, generic over the memory class: llama_kv_cache and
// llama_memory_recurrent both allocate their tensors through these, so a model whose memory is hybrid (qwen35 and
// the like) can hand off its whole memory and not only the attention half. The recurrent state of the 27B is
// 151 MB per request that the file route writes and reads on every handoff, and it never has to leave the box.
//   llama_share_alloc: allocate ctx's tensors as exportable chunks (throws, like share_alloc did)
ggml_backend_buffer_t llama_share_alloc(ggml_context * ctx, ggml_backend_buffer_type_t buft,
        std::vector<llama_kv_shared_region> & out);
//   llama_share_import: map the owner's chunks over ctx's tensors, consuming regions[next...]; nullptr unless the
//   layout matches this context's (types, shapes, flash attention)
ggml_backend_buffer_t llama_share_import(ggml_context * ctx, ggml_backend_buffer_type_t buft,
        const std::vector<llama_kv_shared_region_pod> & regions, size_t & next);

// set by llama_context around memory creation, read by llama_kv_cache while it allocates
void llama_kv_share_begin(bool enabled);
void llama_kv_share_end();
bool llama_kv_share_enabled();
