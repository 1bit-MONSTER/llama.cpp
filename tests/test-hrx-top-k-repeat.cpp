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

// TOP_K on HRX must be a pure function of its input (engine #2146). The sparse-attention indexer on Qwen3.8-Flash-Next
// picks 2051 of 2560 KV rows with it, and when the indices came back in a different order - or, with tied scores, as a
// different set - identical greedy requests gave different replies. Each case runs the same TOP_K several times on the
// HRX device and requires: every row a valid top-k (indices in range, no duplicate, nothing unchosen above anything
// chosen), bitwise-identical output across runs, and with distinct inputs the same index set as the CPU backend.
// Cases cover the indexer shape, a router-like 128 -> 8, an odd row pitch (the scalar path), and heavy ties.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    int cols, k, rows, levels;  // levels > 0: inputs take that many distinct values (ties); 0: continuous
};

std::vector<int32_t> run_top_k(ggml_backend_t backend, const Case & c, const std::vector<float> & input) {
    ggml_init_params params = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    ggml_tensor *    x      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.cols, c.rows);
    ggml_tensor *    y      = ggml_top_k(ctx, x, c.k);
    ggml_cgraph *    graph  = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    std::vector<int32_t>  out;
    if (buffer != nullptr && ggml_backend_supports_op(backend, y)) {
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) {
            out.resize(static_cast<size_t>(c.k) * c.rows);
            ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(int32_t));
        }
    }
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return out;
}

int invalid_rows(const Case & c, const std::vector<float> & input, const std::vector<int32_t> & out) {
    int bad = 0;
    for (int r = 0; r < c.rows; ++r) {
        std::vector<char> picked(c.cols, 0);
        float             lowest_picked = 1e30f, highest_unpicked = -1e30f;
        bool              broken        = false;
        for (int j = 0; j < c.k && !broken; ++j) {
            const int32_t i = out[static_cast<size_t>(r) * c.k + j];
            broken          = i < 0 || i >= c.cols || picked[i];
            if (!broken) {
                picked[i]     = 1;
                lowest_picked = std::min(lowest_picked, input[static_cast<size_t>(r) * c.cols + i]);
            }
        }
        for (int i = 0; i < c.cols && !broken; ++i) {
            if (!picked[i]) {
                highest_unpicked = std::max(highest_unpicked, input[static_cast<size_t>(r) * c.cols + i]);
            }
        }
        bad += broken || highest_unpicked > lowest_picked;
    }
    return bad;
}

std::vector<int32_t> sorted_rows(const Case & c, std::vector<int32_t> out) {
    for (int r = 0; r < c.rows; ++r) {
        std::sort(out.begin() + static_cast<size_t>(r) * c.k, out.begin() + static_cast<size_t>(r + 1) * c.k);
    }
    return out;
}

}  // namespace

int main() {
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-top-k-repeat: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (hrx == nullptr || cpu == nullptr) {
        std::printf("test-hrx-top-k-repeat: FAIL: backend init\n");
        return 1;
    }
    const Case cases[] = {
        { 2560, 2051, 512, 0 },   // the Flash-Next indexer shape, distinct scores
        { 2560, 2051, 512, 64 },  // the same with heavy ties
        { 4096, 2048, 256, 64 },
        { 128, 8, 512, 0 },       // router-like
        { 128, 8, 512, 4 },
        { 2561, 2051, 64, 64 },   // odd pitch: rows after the first are unaligned, the scalar visit path
        { 7, 7, 3, 0 },           // k == cols
    };
    constexpr int kRuns    = 12;
    int           failures = 0;
    for (const Case & c : cases) {
        std::vector<float> input(static_cast<size_t>(c.cols) * c.rows);
        std::mt19937       rng(2146u + static_cast<unsigned>(c.cols * 31 + c.k));
        for (float & v : input) {
            v = c.levels > 0 ? static_cast<float>(rng() % c.levels) / c.levels - 0.5f
                             : std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
        }
        const std::vector<int32_t> first = run_top_k(hrx, c, input);
        if (first.empty()) {
            std::printf("  cols=%d k=%d rows=%d: HRX does not run this TOP_K, skipped\n", c.cols, c.k, c.rows);
            continue;
        }
        int bad = invalid_rows(c, input, first), differ = 0;
        for (int run = 1; run < kRuns; ++run) {
            const std::vector<int32_t> again = run_top_k(hrx, c, input);
            bad += invalid_rows(c, input, again);
            differ += again.size() != first.size() || std::memcmp(again.data(), first.data(), first.size() * 4) != 0;
        }
        bool cpu_set_ok = true;
        if (c.levels == 0) {
            const std::vector<int32_t> ref = run_top_k(cpu, c, input);
            cpu_set_ok = !ref.empty() && sorted_rows(c, ref) == sorted_rows(c, first);
        }
        const bool ok = bad == 0 && differ == 0 && cpu_set_ok;
        failures += !ok;
        std::printf("  %s cols=%d k=%d rows=%d levels=%d: invalid rows %d, runs differing from run 1: %d of %d%s\n",
                    ok ? "ok  " : "FAIL", c.cols, c.k, c.rows, c.levels, bad, differ, kRuns - 1,
                    c.levels == 0 ? (cpu_set_ok ? ", set == CPU" : ", set != CPU") : "");
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hrx);
    std::printf("test-hrx-top-k-repeat: %s\n", failures ? "FAIL" : "ok");
    return failures ? 1 : 0;
}
