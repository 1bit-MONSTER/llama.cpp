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

// Size-keyed pool of driver-owned host staging buffers, for the graph replay download path. Every host
// readback used to allocate a fresh driver buffer and release it, which is one allocation, one map and one
// release per downloaded value on every decode step. The pool keeps a few buffers and hands one back by size.
// ggml-hrx only calls in.

#pragma once

#include "runtime/host-memory.h"

#include <cstddef>

namespace ggml::hrx::onebit {

// A staging buffer of at least `size`, owned by the caller (one reference): a pooled one when the pool has a
// fitting slot, otherwise a fresh driver allocation.
Status staging_pool_acquire(hrx_device_t device, size_t size, hrx_buffer_t & buffer, void *& mapped);

// Hand the buffer back for reuse. The pool takes over the reference it is given; the caller must not use or
// release the buffer afterwards.
void staging_pool_give_back(hrx_buffer_t buffer, void * mapped, size_t size);

// Release every pooled buffer (end of a graph replay state, or shutdown). Also logs the acquire/reuse counts
// under GGML_HRX_STAGING_POOL_DEBUG=1.
void staging_pool_drain();

}  // namespace ggml::hrx::onebit
