#include "dispatch-get-rows.h"

#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <utility>

namespace ggml::hrx {

static constexpr KernelCatalogRef kGetRowsF32Kernel = GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_get_rows_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

// GET_ROWS: output[nrows, width] = source[ids[nrows], width], ids is i32.
// The embedding lookup: source is the [vocab, hidden] weight, ids the token
// list.  Handles the unfused (dense) graphs that the fused Qwen3-MoE
// dispatches do not cover.
static bool supports_get_rows_f32_dispatch(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return false;
    }
    const Value * output = graph_value(graph, node->output);
    const Value * source = graph_value(graph, node->inputs[0]);
    const Value * ids    = graph_value(graph, node->inputs[1]);
    if (output == nullptr || source == nullptr || ids == nullptr) {
        return false;
    }
    // Source may be f32 or q8_0 (the kernel dequantizes q8_0 inline); any
    // other quantized embed type (q4k etc.) is covered by the fused dispatch.
    if (output->type != GGML_TYPE_F32) {
        return false;
    }
    const bool source_f32 = source->type == GGML_TYPE_F32;
    const bool source_q8 = source->type == GGML_TYPE_Q8_0;
    if (!source_f32 && !source_q8) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (!output->contiguous || !source->contiguous || !ids->contiguous) {
        return false;
    }
    // output is [nrows, width]; source is [nrows_src, width]; ids [nrows].
    if (output->ne[1] != ids->ne[0]) {
        return false;
    }
    const uint64_t width = static_cast<uint64_t>(output->ne[0]);
    const uint64_t output_rows = static_cast<uint64_t>(output->ne[1]);
    const uint64_t source_rows = static_cast<uint64_t>(source->ne[0]);
    if (width == 0 || output_rows == 0 || source_rows == 0) {
        return false;
    }
    if (width > std::numeric_limits<uint32_t>::max() ||
        output_rows > std::numeric_limits<uint32_t>::max() ||
        source_rows > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return true;
}

static bool match_get_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    if (!supports_get_rows_f32_dispatch(context.graph, context.root_node)) {
        return false;
    }
    const Value * output = graph_value(context.graph, context.root_node->output);
    const Value * source = graph_value(context.graph, context.root_node->inputs[0]);
    const Value * ids    = graph_value(context.graph, context.root_node->inputs[1]);
    if (output == nullptr || source == nullptr || ids == nullptr) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsF32Kernel);
    dispatch.kernel.integer_parameters.emplace("source_row_count", static_cast<int64_t>(source->ne[0]));
    dispatch.kernel.integer_parameters.emplace("output_row_count", static_cast<int64_t>(output->ne[1]));
    dispatch.kernel.integer_parameters.emplace("width", static_cast<int64_t>(output->ne[0]));
    // source_format: 0 = f32, 1 = q8_0 (mirrors the kernel's format branch).
    const int64_t source_format = source->type == GGML_TYPE_Q8_0 ? 1 : 0;
    dispatch.kernel.integer_parameters.emplace("source_format", source_format);
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_get_rows_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.get_rows_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_get_rows_f32_dispatch,
    });
}

}  // namespace ggml::hrx
