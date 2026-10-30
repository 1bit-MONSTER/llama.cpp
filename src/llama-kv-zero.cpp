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

#include "llama-kv-zero.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-impl.h"

#include <cstdlib>
#include <cstring>
#include <vector>

// Windows read back per tensor: start, middle, end. Three 4 KiB windows catch a fill that did not happen.
static const size_t kSampleBytes = 4096;
static const size_t kSampleCount = 3;

// Tensors sampled by the fast path, spread over the buffer (first, middle, last). The clear is one fill submit
// per buffer, so sampling a few tensors out of it says as much as reading all of them would, and costs a
// handful of readbacks instead of one per layer - this runs on every KV reset, i.e. once per request in a
// server.
static const size_t kSampledTensors = 3;

// One repair chunk. Large enough that the per-call overhead disappears, small enough that the staging buffer
// stays in cache while it is written many times.
static const size_t kChunkBytes = 1024 * 1024;

// ONEBIT_KV_ZERO=0 skips the verification, for measuring what it costs; the padding is then only as
// trustworthy as the backend's own clear, which is the state engine #528 came from. Default: on.
static bool verify_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("ONEBIT_KV_ZERO");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

static bool tensor_is_zero(ggml_tensor * t, size_t & nonzero_bytes) {
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        return true;
    }

    std::vector<uint8_t> sample(kSampleBytes < nbytes ? kSampleBytes : nbytes);
    for (size_t i = 0; i < kSampleCount; ++i) {
        const size_t offset = i == 0 ? 0
                             : i == 1 ? (nbytes / 2) & ~size_t(7)
                                      : (nbytes - sample.size()) & ~size_t(7);
        ggml_backend_tensor_get(t, sample.data(), offset, sample.size());
        for (uint8_t byte : sample) {
            if (byte != 0) {
                nonzero_bytes += sample.size();
                break;
            }
        }
    }
    return nonzero_bytes == 0;
}

// The tensors of a context, in order (the KV tensors of one buffer type are packed into one buffer in this
// order, so a spread of indices is a spread over the buffer).
static std::vector<ggml_tensor *> context_tensors(ggml_context * ctx) {
    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (ggml_nbytes(t) > 0) {
            tensors.push_back(t);
        }
    }
    return tensors;
}

// Sample kSampledTensors tensors out of the context, first/middle/last.
static bool sampled_tensors_are_zero(const std::vector<ggml_tensor *> & tensors, size_t & nonzero_bytes) {
    if (tensors.empty()) {
        return true;
    }
    const size_t sampled = tensors.size() < kSampledTensors ? tensors.size() : kSampledTensors;
    for (size_t i = 0; i < sampled; ++i) {
        const size_t index = sampled == 1 ? 0 : i * (tensors.size() - 1) / (sampled - 1);
        if (!tensor_is_zero(tensors[index], nonzero_bytes)) {
            return false;
        }
    }
    return true;
}

static bool all_tensors_are_zero(const std::vector<ggml_tensor *> & tensors, size_t & nonzero_bytes) {
    for (ggml_tensor * t : tensors) {
        if (!tensor_is_zero(t, nonzero_bytes)) {
            return false;
        }
    }
    return true;
}

static void tensor_zero(ggml_tensor * t, std::vector<uint8_t> & chunk) {
    const size_t nbytes = ggml_nbytes(t);
    for (size_t offset = 0; offset < nbytes; offset += chunk.size()) {
        const size_t size = nbytes - offset < chunk.size() ? nbytes - offset : chunk.size();
        ggml_backend_tensor_set(t, chunk.data(), offset, size);
    }
}

bool llama_kv_zero_fill(ggml_backend_buffer_t buf, ggml_context * ctx, const char * what) {
    if (buf == nullptr || ctx == nullptr || ggml_backend_buffer_get_size(buf) == 0 || !verify_enabled()) {
        return true;
    }

    const std::vector<ggml_tensor *> tensors = context_tensors(ctx);
    const double mib = ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0;

    size_t nonzero_bytes = 0;
    if (sampled_tensors_are_zero(tensors, nonzero_bytes)) {
        LLAMA_LOG_INFO("%s: %s verified zero (%s, %.2f MiB, %zu of %zu tensors sampled) (engine #528)\n", __func__,
                       what, ggml_backend_buffer_name(buf), mib,
                       tensors.empty() ? 0 : std::min(kSampledTensors, tensors.size()), tensors.size());
        return true;
    }

    // The clear did not take: this is the state that turns one padded column into a NaN output row, so say so
    // and repair it through the backend's own writes, which fail loudly instead of silently.
    LLAMA_LOG_WARN("%s: %s was not zero after the clear (%s, %zu bytes non-zero in %zu samples): zero-filling "
                   "%.2f MiB and verifying again (engine #528)\n",
                   __func__, what, ggml_backend_buffer_name(buf), nonzero_bytes, kSampleCount, mib);

    std::vector<uint8_t> chunk(kChunkBytes, 0);
    for (ggml_tensor * t : tensors) {
        tensor_zero(t, chunk);
    }

    // A repair is rare, so verify all of it rather than the sample.
    nonzero_bytes = 0;
    if (!all_tensors_are_zero(tensors, nonzero_bytes)) {
        LLAMA_LOG_ERROR("%s: %s is still not zero after the fill (%zu bytes non-zero), the KV cache cannot be "
                        "used (engine #528)\n",
                        __func__, what, nonzero_bytes);
        return false;
    }

    LLAMA_LOG_INFO("%s: %s zero-filled and verified (%s, %.2f MiB) (engine #528)\n", __func__, what,
                   ggml_backend_buffer_name(buf), mib);
    return true;
}
