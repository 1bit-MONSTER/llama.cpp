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

// Host test for how a layout-alias node is treated when its input is PRODUCED vs EXTERNAL, which is the
// divergence engine #1467's projector hits: with the projector on HRX the graph is refused at
//   unsupported HRX node 20: VIEW output=f32[2048,2,1,1] <- VIEW inputs=[f32[2048,2,1,1]] consumers=[21:ADD]
// while `can_execute_standalone_op_as_graph` reports the same node supported, because the single-node graph it
// builds feeds the view from an External leaf and `value_is_available()` returns true when a value has no
// producer (dispatch-scheduler.cpp:102-106).
//
// No device: the scheduler, the registry and the graph are all host-side; only the tensors are real ggml
// objects, and no backend is initialised.

#include "dispatch/dispatch-scheduler.h"
#include "graph/graph.h"
#include "graph/op-params.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (0)

namespace {

using namespace ggml::hrx;

const DispatchTarget kTarget{ "gfx1151" };

// A graph shaped like the failing one: an external leaf, one op that produces a value on this backend, and a
// no-op layout alias on top of it. `produced` selects whether the alias's input is that produced value (the
// real graph) or a second external leaf (what can_execute_standalone_op_as_graph builds).
enum class ViewKind { NoOpAlias, OffsetAlias, StridedView };

bool plans_with_view(ViewKind kind, bool produced) {
    ggml_init_params params = { /*mem_size*/ 32 * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context *  ctx     = ggml_init(params);
    REQUIRE(ctx != nullptr);

    // 3 rows so an offset view stays in bounds
    ggml_tensor * a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 3);
    ggml_tensor * b    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 3);
    ggml_tensor * add  = ggml_add(ctx, a, b);
    ggml_tensor * base = produced ? add : a;
    // The three relations that matter: a true alias (same storage), an alias at an offset (same buffers, different
    // start - the question is whether ValueMap gives it the same storage), and a genuinely strided view (same shape,
    // different nb[1]) which is NOT an alias and therefore needs a real dispatch that no registration provides.
    ggml_tensor * view = nullptr;
    switch (kind) {
        case ViewKind::NoOpAlias:    view = ggml_view_2d(ctx, base, 2048, 2, base->nb[1], 0); break;
        case ViewKind::OffsetAlias:  view = ggml_view_2d(ctx, base, 2048, 2, base->nb[1], base->nb[1]); break;
        case ViewKind::StridedView:  view = ggml_view_2d(ctx, base, 2048, 2, 2 * base->nb[1], 0); break;
    }
    REQUIRE(view != nullptr);
    ggml_tensor * out  = ggml_scale(ctx, view, 0.5f);

    Graph                graph;
    std::vector<ValueId> no_inputs;
    // the leaves
    const ValueId va = graph.values().get_or_add_tensor_value(a, ValueKind::External);
    const ValueId vb = graph.values().get_or_add_tensor_value(b, ValueKind::External);
    // the producer of the alias's input
    ValueId vadd = va;
    if (produced) {
        const ValueId vadd_out = graph.values().get_or_add_tensor_value(add, ValueKind::Transient);
        GraphNode &   n_add    = graph.add_node(GGML_OP_ADD, vadd_out, { va, vb });
        n_add.params           = import_op_params(*add);
        vadd                   = vadd_out;
    }
    // the alias itself
    const ValueId vview    = graph.values().get_or_add_tensor_value(view, ValueKind::Transient);
    GraphNode &   n_view   = graph.add_node(GGML_OP_VIEW, vview, { vadd });
    n_view.params          = import_op_params(*view);
    // something that consumes the alias, so the alias is not the graph's output
    const ValueId vout     = graph.values().get_or_add_tensor_value(out, ValueKind::Transient);
    GraphNode &   n_scale  = graph.add_node(GGML_OP_SCALE, vout, { vview });
    n_scale.params         = import_op_params(*out);
    (void) no_inputs;
    REQUIRE(graph.build_index().success());

    DispatchScheduler sched;
    const bool        planned = sched.schedule_graph(graph, kTarget);
    if (!planned) {
        // print whatever the scheduler recorded, so the diagnostic added for engine #1467 is visible here
        for (const std::string & e : sched.plan().status.errors()) {
            std::printf("  status: %s" "\n", e.c_str());
        }
    }
    ggml_free(ctx);
    return planned;
}

}  // namespace

int main() {
    // Measured behaviour of the four relations, pinned so a change in the alias rule shows up here and not in an
    // image request two days later.
    const bool noop_external = plans_with_view(ViewKind::NoOpAlias, false);
    const bool noop_produced = plans_with_view(ViewKind::NoOpAlias, true);
    const bool offset        = plans_with_view(ViewKind::OffsetAlias, true);
    const bool strided       = plans_with_view(ViewKind::StridedView, true);

    std::printf("no-op alias, external input : %s\n", noop_external ? "plans" : "REFUSED");
    std::printf("no-op alias, produced input : %s\n", noop_produced ? "plans" : "REFUSED");
    std::printf("offset alias, produced input: %s\n", offset ? "plans" : "REFUSED");
    std::printf("strided view, produced input: %s\n", strided ? "plans" : "REFUSED");

    // A no-op layout alias plans whether its input is an External leaf (the single-node graph that
    // can_execute_standalone_op_as_graph builds) or a value this backend produces (the real graph). That REFUTES
    // the reading that engine #1467's VIEW refusal comes from value_is_available() returning true for a
    // producer-less value: if that shortcut decided it, the produced case would be refused.
    REQUIRE(noop_external);
    REQUIRE(noop_produced);
    // An offset alias is still one storage root, so it is elided too.
    REQUIRE(offset);
    // A STRIDED view of the same shape is elided as well - same_storage() compares the storage root, not the layout -
    // and the refusal then lands on the CONSUMER (see the status line: "unsupported HRX node 2: SCALE"). Pinned as
    // the safe outcome it is today: the plan is refused, not silently computed through a wrong layout. If a consumer
    // ever accepts a strided alias, THIS assertion is where it should become visible.
    REQUIRE(!strided);
    return 0;
}
