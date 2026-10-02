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

// Known-answer check for NVFP4 weights on HRX: GET_ROWS of NVFP4 rows must equal ggml's dequantize_row_nvfp4
// bit for bit. NVFP4 is 64-value blocks of four UE4M3 scales (one per 16 values) and 32 bytes of E2M1 codes;
// every halved UE4M3 scale is m * 2^-10 or (8 + m) * 2^(e - 11) and every code an integer up to 12, so each value
// is exact in f32 (no subnormals: the smallest nonzero is 2^-10). The 16 rows of 256 values hold 16 scales each,
// together all 256 scale bytes (0x7F reads as 0, bit 7 is ignored), and every 16-value group holds all 16 codes
// split across both nibbles. The graph runs on the HRX device itself (no scheduler, so no CPU fallback), and the
// HRX dispatch plan for it must contain the get_rows kernel.

#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "graph/graph.h"
#include "kernel-corpus/kernel-corpus.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static constexpr int kRows       = 16;
static constexpr int kValues     = 256;  // four 64-value blocks per row
static constexpr int kBlocks     = kValues / 64;
static constexpr int kBlockBytes = 36;   // d[4] (UE4M3), qs[32]
static constexpr int kRowBytes   = kBlocks * kBlockBytes;

static std::vector<uint8_t> make_weights() {
    std::vector<uint8_t> weights(static_cast<size_t>(kRowBytes) * kRows);
    for (int r = 0; r < kRows; ++r) {
        for (int b = 0; b < kBlocks; ++b) {
            uint8_t * block = weights.data() + static_cast<size_t>(r) * kRowBytes + b * kBlockBytes;
            for (int s = 0; s < 4; ++s) {
                block[s] = static_cast<uint8_t>(r * 16 + b * 4 + s);
                for (int j = 0; j < 8; ++j) {
                    const int low        = (j + 2 * s + b + r) & 15;
                    const int high       = (low + 8) & 15;
                    block[4 + s * 8 + j] = static_cast<uint8_t>(low | (high << 4));
                }
            }
        }
    }
    return weights;
}

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), "gfx1151", kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

// The HRX dispatch plan for the graph: every node must be covered, and a get_rows kernel must run.
static void require_hrx_get_rows_plan(ggml_cgraph * graph) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    if (!scheduler.schedule_graph(imported.graph, { "gfx1151" }, &diagnostics)) {
        std::fprintf(stderr, "unsupported: %s\n", diagnostics.unsupported_message.c_str());
        std::abort();
    }
    REQUIRE(scheduler.plan().valid());
    bool found = false;
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        const std::string name = kernel_name_for_id(dispatch.kernel.kernel_id);
        std::printf("dispatch: %s\n", name.c_str());
        found = found || name.find("get_rows") != std::string::npos;
    }
    REQUIRE(found);
}

int main() {
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-nvfp4: no HRX0 device, skipped\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    REQUIRE(backend != nullptr);

    ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx    = ggml_init(params);
    REQUIRE(ctx != nullptr);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, kValues, kRows);
    ggml_tensor * ids     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kRows);
    ggml_tensor * rows    = ggml_get_rows(ctx, weights, ids);
    ggml_cgraph * graph   = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rows);
    REQUIRE(ggml_backend_supports_op(backend, rows));
    require_hrx_get_rows_plan(graph);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    const std::vector<uint8_t> host_weights = make_weights();
    std::vector<int32_t>       host_ids(kRows);
    for (int r = 0; r < kRows; ++r) {
        host_ids[r] = kRows - 1 - r;
    }
    ggml_backend_tensor_set(weights, host_weights.data(), 0, host_weights.size());
    ggml_backend_tensor_set(ids, host_ids.data(), 0, host_ids.size() * sizeof(int32_t));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> got(static_cast<size_t>(kValues) * kRows);
    std::vector<float> expected(kValues);
    ggml_backend_tensor_get(rows, got.data(), 0, got.size() * sizeof(float));
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_NVFP4);
    REQUIRE(traits->to_float != nullptr);
    int mismatches = 0;
    for (int r = 0; r < kRows; ++r) {
        const int source = host_ids[r];
        traits->to_float(host_weights.data() + static_cast<size_t>(source) * kRowBytes, expected.data(), kValues);
        for (int k = 0; k < kValues; ++k) {
            const float value = got[static_cast<size_t>(r) * kValues + k];
            if (std::memcmp(&value, &expected[k], sizeof(float)) == 0) {
                continue;
            }
            if (mismatches < 8) {
                std::fprintf(stderr, "row %d value %d: got %.9g, expected %.9g\n", source, k, value, expected[k]);
            }
            ++mismatches;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    REQUIRE(mismatches == 0);
    std::printf("test-hrx-nvfp4: %d rows x %d values bit-exact, all 256 scale bytes\n", kRows, kValues);
    return 0;
}
