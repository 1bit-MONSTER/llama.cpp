#include "dispatch-add.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kAddF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_add_f32");

// The HRX/loom runtime submits a dispatch through a fixed-size kernarg ring and
// fails with OUT_OF_RANGE when the grid does not fit (see
// amdgpu/host_queue_submission.c); that failed submission also leaves the device
// stream unusable for every later submission. Only claim shapes whose grid stays
// inside the ring. The add kernel launches one 256-wide workgroup per 256
// elements, so 4096 workgroups is ~1M elements - far above any activation this
// route is used for, and well below the ~1.9M-element point where the measured
// submission starts failing.
static constexpr uint64_t kAddMaxWorkgroups = 4096;

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool supports_add_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_ADD || node->inputs.size() != 2) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * a      = graph_value(graph, node->inputs[0]);
    const Value * b      = graph_value(graph, node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }
    return output->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           same_shape(*output, *a) && same_shape(*output, *b) && output->contiguous && a->contiguous && b->contiguous &&
           output->element_count > 0 &&
           static_cast<uint64_t>(output->element_count) <= std::numeric_limits<uint32_t>::max() &&
           (static_cast<uint64_t>(output->element_count) + 255) / 256 <= kAddMaxWorkgroups;
}

static bool match_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!supports_add_f32_dispatch(context.graph, context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    const Value * a      = graph_value(context.graph, context.root_node->inputs[0]);
    const Value * b      = graph_value(context.graph, context.root_node->inputs[1]);
    if (output == nullptr || a == nullptr || b == nullptr) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.bindings.push_back({ a->id, 0, a->byte_count });
    dispatch.bindings.push_back({ b->id, 0, b->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_add_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.add_f32",
        GGML_OP_ADD,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_add_f32_dispatch,
    });
}

}  // namespace ggml::hrx
