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

#include <cstddef>
#include <cstdint>

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

// the pending llama_kv_share_next request; a real (not no_alloc) context takes and clears it
bool llama_kv_share_take_next(bool no_alloc);

// set by llama_context around memory creation, read by llama_kv_cache while it allocates
void llama_kv_share_begin(bool enabled);
void llama_kv_share_end();
bool llama_kv_share_enabled();
