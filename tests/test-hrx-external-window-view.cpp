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

// A view into storage the graph does not hold (engine #1467). llama.cpp gives a deepstack model (Qwen3-VL) one wide
// input embedding, n_embd x (1 + deepstack layers) values per token, and takes two kinds of view of it: the first
// n_embd values of every token (t_inp_embd), and for each deepstack layer a view OF THAT VIEW one slice further in,
// which is added to the layer's output. When the node of the first view is in another backend's split, the wide tensor
// is not a value of the HRX graph at all, so the deepstack view cannot alias anything the graph knows: it got a storage
// of its own, was not an alias of its input, no registration takes a VIEW, and the whole graph was refused - on the
// text model's own warmup, before any image.
//
// Two checks. The plan: the same sub-graph the scheduler hands over (ggml_graph_view without the first view's node),
// through the real importer, must plan, and a window that this graph is expected to FILL (a transient) must stay
// refused. The values: on HRX0 the window must read the slice it names, compared with the CPU backend; skipped
// without a device.

#include "dispatch/dispatch-scheduler.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"  // ggml_cgraph and ggml_graph_view: the sub-graph the scheduler builds
#include "ggml.h"
#include "graph/graph.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static constexpr int64_t kWidth  = 2048;  // n_embd
static constexpr int64_t kSlices = 4;     // 1 + deepstack layers
static constexpr int64_t kTokens = 2;

struct WindowGraph {
    ggml_context * ctx   = nullptr;
    ggml_tensor *  wide  = nullptr;
    ggml_tensor *  a     = nullptr;
    ggml_tensor *  out   = nullptr;
    ggml_cgraph *  graph = nullptr;  // every node
    ggml_cgraph    split = {};       // what the scheduler hands over: without the node of the first view
};

// out = a + view(view(wide, first slice), slice `slice`), as llm_graph_context::build_inp_embd and the deepstack loop
// build it.
static WindowGraph make_window_graph(int slice) {
    WindowGraph      g;
    ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx                   = ggml_init(params);
    REQUIRE(g.ctx != nullptr);
    g.wide               = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kWidth * kSlices, kTokens);
    g.a                  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kWidth, kTokens);
    ggml_tensor * first  = ggml_view_2d(g.ctx, g.wide, kWidth, kTokens, g.wide->nb[1], 0);
    ggml_tensor * window = ggml_view_2d(g.ctx, first, kWidth, kTokens, first->nb[1], slice * kWidth * sizeof(float));
    g.out                = ggml_add(g.ctx, g.a, window);
    g.graph              = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.out);
    // nodes: the first view, the window, the add
    REQUIRE(ggml_graph_n_nodes(g.graph) == 3 && ggml_graph_node(g.graph, 0) == first &&
            ggml_graph_node(g.graph, 1) == window && ggml_graph_node(g.graph, 2) == g.out);
    g.split = ggml_graph_view(g.graph, 1, 3);
    return g;
}

static bool plans(ggml_cgraph & graph, bool window_is_transient, std::string & error) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(graph);
    REQUIRE(imported.valid());
    if (window_is_transient) {
        // What the importer never produces for such a window, built by hand: the same nodes with the window registered
        // as a value this graph must fill. Nothing would fill it, so it must be refused.
        ggml::hrx::Graph by_hand;
        std::vector<ggml::hrx::ValueId> ids;
        for (const ggml::hrx::GraphNode & node : imported.graph.nodes()) {
            std::vector<ggml::hrx::ValueId> inputs;
            for (const ggml::hrx::ValueId input : node.inputs) {
                const ggml::hrx::Value * value = imported.graph.values().find(input);
                REQUIRE(value != nullptr && value->tensor != nullptr);
                const bool is_window = value->tensor == ggml_graph_node(&graph, 0);
                inputs.push_back(by_hand.values().get_or_add_tensor_value(
                    value->tensor, is_window ? ggml::hrx::ValueKind::Transient : value->kind));
            }
            const ggml::hrx::Value * output = imported.graph.values().find(node.output);
            REQUIRE(output != nullptr && output->tensor != nullptr);
            const ggml::hrx::ValueId id = by_hand.values().get_or_add_tensor_value(
                output->tensor, ggml::hrx::ValueKind::Transient);
            ggml::hrx::GraphNode & added = by_hand.add_node(node.op, id, std::move(inputs));
            added.params                 = node.params;
        }
        REQUIRE(by_hand.build_index().success());
        ggml::hrx::DispatchScheduler scheduler;
        const bool                   ok = scheduler.schedule_graph(by_hand, { "gfx1151" });
        error = ok || scheduler.plan().status.errors().empty() ? "" : scheduler.plan().status.errors().back();
        return ok;
    }
    ggml::hrx::DispatchScheduler scheduler;
    const bool                   ok = scheduler.schedule_graph(imported.graph, { "gfx1151" });
    error = ok || scheduler.plan().status.errors().empty() ? "" : scheduler.plan().status.errors().back();
    return ok;
}

// wide[slice s, token t, i] = a value that names all three, so a window at the wrong slice or stride cannot agree
static std::vector<float> wide_data() {
    std::vector<float> data(static_cast<size_t>(kWidth * kSlices * kTokens));
    for (int64_t t = 0; t < kTokens; ++t) {
        for (int64_t s = 0; s < kSlices; ++s) {
            for (int64_t i = 0; i < kWidth; ++i) {
                data[static_cast<size_t>((t * kSlices + s) * kWidth + i)] =
                    static_cast<float>(1000 * (s + 1) + 100 * t) + static_cast<float>(i) / 4096.0f;
            }
        }
    }
    return data;
}

static std::vector<float> compute(ggml_backend_t backend, int slice) {
    WindowGraph           g      = make_window_graph(slice);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    REQUIRE(buffer != nullptr);
    const std::vector<float> wide = wide_data();
    const std::vector<float> a(static_cast<size_t>(kWidth * kTokens), 0.25f);
    ggml_backend_tensor_set(g.wide, wide.data(), 0, wide.size() * sizeof(float));
    ggml_backend_tensor_set(g.a, a.data(), 0, a.size() * sizeof(float));
    REQUIRE(ggml_backend_graph_compute(backend, &g.split) == GGML_STATUS_SUCCESS);
    std::vector<float> result(static_cast<size_t>(ggml_nelements(g.out)));
    ggml_backend_tensor_get(g.out, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(g.ctx);
    return result;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    int failures = 0;

    // the plan
    for (int slice = 1; slice < kSlices; ++slice) {
        WindowGraph g = make_window_graph(slice);
        std::string error;
        const bool  external  = plans(g.split, false, error);
        std::printf("window at slice %d, as imported          : %s%s%s\n", slice, external ? "plans" : "REFUSED",
                    external ? "" : " - ", error.c_str());
        failures += external ? 0 : 1;
        const bool transient = plans(g.split, true, error);
        std::printf("window at slice %d, registered transient : %s\n", slice, transient ? "plans" : "REFUSED");
        failures += transient ? 1 : 0;
        ggml_free(g.ctx);
    }

    // the values
    ggml_backend_dev_t device = ggml_backend_dev_by_name("HRX0");
    if (device == nullptr) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_name("HRX0");
    }
    if (device == nullptr) {
        std::printf("test-hrx-external-window-view: no HRX0 device, the value check is skipped\n");
    } else {
        ggml_backend_t hrx = ggml_backend_dev_init(device, nullptr);
        ggml_backend_t cpu = ggml_backend_cpu_init();
        REQUIRE(hrx != nullptr && cpu != nullptr);
        for (int slice = 1; slice < kSlices; ++slice) {
            const std::vector<float> expected = compute(cpu, slice);
            const std::vector<float> got      = compute(hrx, slice);
            REQUIRE(got.size() == expected.size());
            size_t wrong = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                wrong += got[i] == expected[i] ? 0 : 1;
            }
            // the CPU result itself must be the slice it names: 0.25 + wide[slice, token 1, 0]
            const float first_of_token1 = expected[static_cast<size_t>(kWidth)];
            const bool  names_slice     = std::fabs(first_of_token1 - (0.25f + 1000.0f * (slice + 1) + 100.0f)) < 1e-3f;
            std::printf("window at slice %d on HRX0 vs CPU         : %zu of %zu values differ%s\n", slice, wrong,
                        got.size(), names_slice ? "" : " (the reference does not read the slice)");
            failures += wrong == 0 && names_slice ? 0 : 1;
        }
        ggml_backend_free(cpu);
        ggml_backend_free(hrx);
    }

    std::printf("test-hrx-external-window-view: %d failures\n", failures);
    REQUIRE(failures == 0);
    return 0;
}
