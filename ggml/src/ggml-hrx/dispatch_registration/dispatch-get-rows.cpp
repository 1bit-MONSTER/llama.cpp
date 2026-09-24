// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

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
//
// The kernel's domain is fully pinned by the Loom source and by the ABI, and the
// claim has to match it exactly - a claimed shape that the dispatcher cannot run
// either fails to compile or (worse) fails at submission time and leaves the device
// stream unusable:
//
//  - `index.assume` in get_rows_f32.loom caps width to [32, 16384], the row counts to
//    [1, 131072] / [1, 2048]; specialising a constant outside those ranges is a
//    compile error, not a fallback.
//  - The kernel indexes a flat 2D view with (output_row, column) and derives the
//    launch grid as (ceil(width/256), output_row_count, 1), so batched (ne[2]/ne[3] >
//    1) or higher-rank get_rows is not covered.
//  - The HRX runtime charges the dispatch's kernarg footprint against a fixed
//    262144-entry ring; measured on gfx1151 the footprint is about
//    16 * ceil(width/256) * (source_row_count + output_row_count) blocks. Crossing
//    the ring aborts the submission and corrupts the stream, so claim only shapes
//    that stay comfortably inside it.
static constexpr uint64_t kGetRowsMinWidth        = 32;
static constexpr uint64_t kGetRowsMaxWidth        = 16384;
static constexpr uint64_t kGetRowsMaxOutputRows   = 2048;
static constexpr uint64_t kGetRowsMaxSourceRows   = 131072;
// 0.75 x the measured 262144-entry limit, leaving headroom for the per-dispatch
// constant terms the estimate above does not model.
static constexpr uint64_t kGetRowsMaxKernargBlocks = 196608;

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
    const bool source_q8  = source->type == GGML_TYPE_Q8_0;
    if (!source_f32 && !source_q8) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32) {
        return false;
    }
    if (!output->contiguous || !source->contiguous || !ids->contiguous) {
        return false;
    }
    // Flat 2D gather only: the kernel has no batch dimensions.
    if (source->ne[2] != 1 || source->ne[3] != 1 || output->ne[2] != 1 || output->ne[3] != 1 || ids->ne[1] != 1 ||
        ids->ne[2] != 1) {
        return false;
    }
    // source is [width, source_rows]; output is [width, output_rows] == [ids->ne[0], width].
    if (source->ne[0] != output->ne[0] || output->ne[1] != ids->ne[0]) {
        return false;
    }
    const uint64_t width       = static_cast<uint64_t>(output->ne[0]);
    const uint64_t output_rows = static_cast<uint64_t>(output->ne[1]);
    const uint64_t source_rows = static_cast<uint64_t>(source->ne[1]);
    if (width < kGetRowsMinWidth || width > kGetRowsMaxWidth) {
        return false;
    }
    if (output_rows > kGetRowsMaxOutputRows || source_rows > kGetRowsMaxSourceRows) {
        return false;
    }
    // q8_0 rows are whole 32-value blocks.
    if (source_q8 && width % 32 != 0) {
        return false;
    }
    const uint64_t column_workgroups = (width + 255) / 256;
    const uint64_t kernarg_blocks     = 16 * column_workgroups * (source_rows + output_rows) + 256;
    if (kernarg_blocks > kGetRowsMaxKernargBlocks) {
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
    // source is [width, row_count]; the row count is ne[1] (ne[0] is the width).
    dispatch.kernel.integer_parameters.emplace("source_row_count", static_cast<int64_t>(source->ne[1]));
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
    // Registered for the shapes the standalone kernel can actually run. Anything
    // outside supports_get_rows_f32_dispatch() - batched/higher-rank gathers, widths
    // outside [32, 16384], vocabularies or row counts that would overflow the
    // runtime's kernarg ring - is left to the CPU backend, which needs no copies
    // because the HRX buffer type is host-visible.
    registry.add({
        "common.get_rows_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_get_rows_f32_dispatch,
    });
    GGML_UNUSED(registry);
}

}  // namespace ggml::hrx
