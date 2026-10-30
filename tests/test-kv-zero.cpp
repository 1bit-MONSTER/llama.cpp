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

// Tests for llama-kv-zero.cpp (engine #528): the KV padding must read zero, and a buffer whose clear did not
// take must be repaired. CPU buffers only, so this runs anywhere.

#include "llama-kv-zero.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define REQUIRE(condition)                                                                        \
    do {                                                                                          \
        if (!(condition)) {                                                                       \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                         \
        }                                                                                         \
    } while (false)

struct kv_zero_test_context {
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    std::vector<ggml_tensor *> tensors;

    // one context per (tensor count, tensor bytes), all F16 like a KV cache
    kv_zero_test_context(size_t n_tensors, size_t bytes) {
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        const int64_t n = (int64_t) (bytes / sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n_tensors; ++i) {
            ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n);
            ggml_format_name(t, "kv_%zu", i);
            tensors.push_back(t);
        }
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        REQUIRE(backend != nullptr);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_free(backend);
        REQUIRE(buf != nullptr);
    }

    ~kv_zero_test_context() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    void poison(ggml_tensor * t, uint8_t byte, size_t offset, size_t size) {
        std::vector<uint8_t> data(size, byte);
        ggml_backend_tensor_set(t, data.data(), offset, size);
    }

    // true when every byte of the buffer's tensors reads zero
    bool all_zero() const {
        for (ggml_tensor * t : tensors) {
            std::vector<uint8_t> data(ggml_nbytes(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size());
            for (uint8_t byte : data) {
                if (byte != 0) {
                    return false;
                }
            }
        }
        return true;
    }
};

static void test_clean_buffer_is_verified() {
    kv_zero_test_context c(4, 64 * 1024);
    REQUIRE(c.all_zero());
    REQUIRE(llama_kv_zero_fill(c.buf, c.ctx, "test"));
    REQUIRE(c.all_zero());
    std::printf("clean buffer verified\n");
}

static void test_poisoned_buffer_is_repaired() {
    kv_zero_test_context c(4, 64 * 1024);
    // f16 0x7E00 (NaN) over one whole tensor
    c.poison(c.tensors[3], 0x7E, 0, ggml_nbytes(c.tensors[3]));
    REQUIRE(!c.all_zero());
    REQUIRE(llama_kv_zero_fill(c.buf, c.ctx, "test"));
    REQUIRE(c.all_zero());
    std::printf("poisoned tensor repaired\n");
}

// The fast path samples three windows per tensor, so a poison that only touches one window must still be found
// and repaired: last window of the first tensor, middle window of the last one.
static void test_sampled_windows_are_checked() {
    kv_zero_test_context c(6, 128 * 1024);
    for (ggml_tensor * t : { c.tensors.front(), c.tensors.back() }) {
        const size_t nbytes = ggml_nbytes(t);
        for (size_t offset : { size_t(0), nbytes / 2, nbytes - 4096 }) {
            c.poison(t, 0x7E, offset, 4096);
            REQUIRE(!c.all_zero());
            REQUIRE(llama_kv_zero_fill(c.buf, c.ctx, "test"));
            REQUIRE(c.all_zero());
        }
    }
    std::printf("all sampled windows checked\n");
}

// A buffer larger than the 1 MiB repair chunk exercises the chunked fill and the second verification.
static void test_chunked_repair() {
    kv_zero_test_context c(2, 3 * 1024 * 1024);
    for (ggml_tensor * t : c.tensors) {
        c.poison(t, 0x7E, 0, ggml_nbytes(t));
    }
    REQUIRE(!c.all_zero());
    REQUIRE(llama_kv_zero_fill(c.buf, c.ctx, "test"));
    REQUIRE(c.all_zero());
    std::printf("chunked repair over %zu MiB\n", (2 * 3 * 1024 * 1024) / 1024 / 1024);
}

int main() {
    test_clean_buffer_is_verified();
    test_poisoned_buffer_is_repaired();
    test_sampled_windows_are_checked();
    test_chunked_repair();
    std::printf("test-kv-zero: all tests passed\n");
    return 0;
}
