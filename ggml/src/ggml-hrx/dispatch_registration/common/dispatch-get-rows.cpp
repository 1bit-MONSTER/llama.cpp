#include "dispatch-get-rows.h"

#include "../qwen/dispatch-llm-profiles.h"
#include "dispatch-mul-mat-weight-format.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kGetRowsF32Kernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_f32");
static constexpr KernelCatalogRef kGetRowsScalar3dF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_scalar_3d_f32");
static constexpr KernelCatalogRef kRepeatScalar3dF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_repeat_scalar_3d_f32");
static constexpr KernelCatalogRef kGetRowsF32NextKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_f32_next");
static constexpr int64_t          kMaximumHiddenElements = int64_t{ 1 } << 30;
static constexpr int64_t          kQ1_0GetRowsFormat    = 10;
static constexpr int64_t          kQwenHiddenSize       = kQwen30BMoeDispatchProfile.hidden_size;
static constexpr int64_t          kQwenVocabularyCount  = 151936;
// 2^18 + 64 was the Qwen vocabulary bound; a per-layer token embedding table is 320,001,536 rows
// (Qwen3.8-Flash-Next), so the cap follows the kernel's own id range (engine #572). DeepSeek-V4.1's Engram
// table is 768,022,850 rows (256 x 768,022,850, IQ4_NL), which is past 2^29: the id range is 2^30 now. The ids
// are I32, so this keeps a 2x margin to INT32_MAX, and the row byte offset is computed in `offset` (64 bit):
// worst case here is 2^30 * 144 B = 155 GB, far past 32-bit.
static constexpr int64_t          kMaxGetRowsRowCount   = 1073741824;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_1d_or_2d_column(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_hidden_size(ggml_type type, int64_t hidden_size) {
    if (hidden_size < 4 || hidden_size > kMaximumHiddenElements || hidden_size % 4 != 0) {
        return false;
    }

    const int64_t block_size = ggml_blck_size(type);
    return block_size > 0 && hidden_size % block_size == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_row_count(int64_t row_count) {
    return row_count >= 1 && row_count <= kMaxGetRowsRowCount;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t row_byte_count(ggml_type type, int64_t token_count, int64_t hidden_size) {
    if (token_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(type, hidden_size);
}

static bool is_qwen_q6k_q8_consumer(const Graph & graph, const GraphNode * consumer, const Value & input) {
    if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
        consumer->inputs[1] != input.id) {
        return false;
    }

    const Value * weight = graph_value(graph, consumer->inputs[0]);
    const Value * output = graph_value(graph, consumer->output);
    if (weight == nullptr || output == nullptr || weight->type != GGML_TYPE_Q6_K || output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !output->contiguous) {
        return false;
    }

    return input.ne[0] == kQwenHiddenSize && input.ne[1] == 1 && input.ne[2] == 1 && input.ne[3] == 1 &&
           weight->ne[0] == kQwenHiddenSize && weight->ne[1] == kQwenVocabularyCount && weight->ne[2] == 1 &&
           weight->ne[3] == 1 && output->ne[0] == kQwenVocabularyCount && output->ne[1] == 1 && output->ne[2] == 1 &&
           output->ne[3] == 1;
}

static void append_unique_demand(std::vector<ggml_type> & demands, ggml_type type) {
    if (std::find(demands.begin(), demands.end(), type) == demands.end()) {
        demands.push_back(type);
    }
}

static std::vector<ggml_type> collect_alternate_demands(const Graph & graph, const Value & value) {
    std::vector<ggml_type> demands;
    if (!graph.has_index()) {
        return demands;
    }

    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value.id);
    for (const GraphNode * consumer : consumers) {
        if (is_qwen_q6k_q8_consumer(graph, consumer, value)) {
            append_unique_demand(demands, GGML_TYPE_Q8_1);
        }
    }
    return demands;
}

static const char * alternate_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_1:
            return "common.get_rows.q8_1_x4";
        case GGML_TYPE_F16:
            return "common.get_rows.f16";
        case GGML_TYPE_F32:
            return "common.get_rows.f32";
        default:
            return "common.get_rows.next";
    }
}

static bool get_rows_format_for_type(ggml_type type, int64_t & format) {
    if (type == GGML_TYPE_Q1_0) {
        format = kQ1_0GetRowsFormat;
        return true;
    }

    CommonMulMatWeightFormat common_format;
    if (!common_mul_mat_format_for_type(type, common_format)) {
        return false;
    }

    format = common_mul_mat_format_config_value(common_format);
    return true;
}

struct GetRowsMatch {
    const Value * ids                 = nullptr;
    const Value * weight              = nullptr;
    const Value * output              = nullptr;
    int64_t       weight_format_value = -1;
    int64_t       token_count         = 0;
    int64_t       row_count           = 0;
    int64_t       hidden_size         = 0;

    bool matched() const {
        return ids != nullptr && weight != nullptr && output != nullptr && weight_format_value >= 0;
    }
};

static void add_common_compile_parameters(Dispatch & dispatch, const GetRowsMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.hidden_capacity", to_config_value(match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.weight_format",
                                               to_config_value(match.weight_format_value));
}

static void add_common_integer_parameters(Dispatch & dispatch, const GetRowsMatch & match) {
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("row_count", match.row_count);
    dispatch.kernel.integer_parameters.emplace("hidden_size", match.hidden_size);
}

static void add_primary_bindings(Dispatch & dispatch, const GetRowsMatch & match) {
    dispatch.bindings.push_back({ match.ids->id, 0, match.ids->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
}

static bool match_repeat_scalar_3d_f32_input(const Graph & graph, const GraphNode * node,
                                             const Value *& source, const Value *& repeated) {
    const bool is_repeat = node != nullptr && node->op == GGML_OP_REPEAT;
    const bool has_one_input = node != nullptr && node->inputs.size() == 1;
    if (!is_repeat || !has_one_input) {
        return false;
    }

    source = graph_value(graph, node->inputs[0]);
    repeated = graph_value(graph, node->output);
    const bool has_values = source != nullptr && repeated != nullptr;
    if (!has_values) {
        return false;
    }

    const bool has_types = source->type == GGML_TYPE_F32 && repeated->type == GGML_TYPE_F32;
    const bool has_source_shape = source->ne[0] == 1 && source->ne[1] == 128 &&
                                  source->ne[2] == 1 && source->ne[3] == 1;
    const bool has_repeat_shape = repeated->ne[0] == 1 && repeated->ne[1] == 128 &&
                                  repeated->ne[3] == 1 && is_supported_token_count(repeated->ne[2]);
    const bool has_layout = source->contiguous && repeated->contiguous;
    return has_types && has_source_shape && has_repeat_shape && has_layout;
}

static bool match_repeat_scalar_3d_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const Value * source = nullptr;
    const Value * repeated = nullptr;
    if (!match_repeat_scalar_3d_f32_input(context.graph, context.root_node, source, repeated)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRepeatScalar3dF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("token_count", repeated->ne[2]);
    dispatch.bindings.push_back({ source->id, 0, source->byte_count });
    dispatch.bindings.push_back({ repeated->id, 0, repeated->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_get_rows_scalar_3d_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    const bool is_get_rows = node != nullptr && node->op == GGML_OP_GET_ROWS;
    const bool has_two_inputs = node != nullptr && node->inputs.size() == 2;
    if (!is_get_rows || !has_two_inputs) {
        return false;
    }

    const Value * scales = graph_value(context.graph, node->inputs[0]);
    const Value * ids = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    const bool has_values = scales != nullptr && ids != nullptr && output != nullptr;
    if (!has_values) {
        return false;
    }

    const int64_t token_count = output->ne[2];
    const bool has_supported_tokens = is_supported_token_count(token_count);
    const bool has_scale_shape = scales->ne[0] == 1 && scales->ne[1] == 128 &&
                                 scales->ne[2] == token_count && scales->ne[3] == 1;
    const bool has_id_shape = ids->ne[0] == 8 && ids->ne[1] == token_count &&
                              ids->ne[2] == 1 && ids->ne[3] == 1;
    const bool has_output_shape = output->ne[0] == 1 && output->ne[1] == 8 &&
                                  output->ne[3] == 1;
    const bool has_types = scales->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
                           output->type == GGML_TYPE_F32;
    const bool has_contiguous_values = scales->contiguous && output->contiguous;
    const bool has_id_columns = ids->nb[0] == sizeof(int32_t) && ids->nb[1] % sizeof(int32_t) == 0;
    const bool has_supported_shape = has_supported_tokens && has_scale_shape && has_id_shape &&
                                     has_output_shape && has_types && has_contiguous_values && has_id_columns;
    if (!has_supported_shape) {
        return false;
    }

    const size_t route_stride = ids->nb[1] / sizeof(int32_t);
    const bool has_supported_stride = route_stride >= 8 && route_stride <= 512;
    const size_t expected_id_bytes = (static_cast<size_t>(token_count) - 1) * ids->nb[1] + 8 * sizeof(int32_t);
    const bool has_exact_id_span = ids->byte_count == expected_id_bytes;
    if (!has_supported_stride || !has_exact_id_span) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsScalar3dF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("token_count", token_count);
    dispatch.kernel.integer_parameters.emplace("route_stride", static_cast<int64_t>(route_stride));
    dispatch.bindings.push_back({ ids->id, 0, ids->byte_count });
    dispatch.bindings.push_back({ scales->id, 0, scales->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static GetRowsMatch match_get_rows_f32(const Graph & graph, const GraphNode * node) {
    GetRowsMatch match;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return match;
    }

    const Value *            weight = graph_value(graph, node->inputs[0]);
    const Value *            ids    = graph_value(graph, node->inputs[1]);
    const Value *            output = graph_value(graph, node->output);
    if (weight == nullptr || ids == nullptr || output == nullptr || ids->type != GGML_TYPE_I32 ||
        output->type != GGML_TYPE_F32 || !weight->contiguous || !ids->contiguous || !output->contiguous ||
        !is_2d(*weight) || !is_1d_or_2d_column(*ids) || !is_2d(*output)) {
        return {};
    }

    int64_t weight_format_value = 0;
    if (!get_rows_format_for_type(weight->type, weight_format_value)) {
        return {};
    }

    const int64_t hidden_size = weight->ne[0];
    const int64_t row_count   = weight->ne[1];
    const int64_t token_count = ids->ne[0];
    if (output->ne[0] != hidden_size || output->ne[1] != token_count ||
        !is_supported_hidden_size(weight->type, hidden_size) || !is_supported_row_count(row_count) ||
        !is_supported_token_count(token_count)) {
        return {};
    }

    match.ids                 = ids;
    match.weight              = weight;
    match.output              = output;
    match.weight_format_value = weight_format_value;
    match.token_count         = token_count;
    match.row_count           = row_count;
    match.hidden_size         = hidden_size;
    return match;
}

static Dispatch make_get_rows_dispatch(const GetRowsMatch & match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsF32Kernel);
    add_common_integer_parameters(dispatch, match);
    add_common_compile_parameters(dispatch, match);
    add_primary_bindings(dispatch, match);
    return dispatch;
}

static Dispatch make_get_rows_next_dispatch(const GetRowsMatch &     match,
                                            CommonMulMatWeightFormat next_format,
                                            ValueId                  next_value,
                                            size_t                   next_byte_count) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsF32NextKernel);
    add_common_integer_parameters(dispatch, match);
    add_common_compile_parameters(dispatch, match);
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.next_format",
                                               to_config_value(common_mul_mat_format_config_value(next_format)));
    add_primary_bindings(dispatch, match);
    dispatch.bindings.push_back({ next_value, 0, next_byte_count });
    return dispatch;
}

static bool append_alternate_metadata(DispatchMatch & dispatch_match,
                                      const Value &   output,
                                      ValueId         alternate_value,
                                      ggml_type       type,
                                      size_t          byte_count) {
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { output.id, alternate_value, type, byte_count, alternate_name(type) }, metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }
    return true;
}

static bool match_get_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GetRowsMatch match = match_get_rows_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(make_get_rows_dispatch(match));
    return true;
}

static bool match_get_rows_f32_next_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GetRowsMatch match = match_get_rows_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const std::vector<ggml_type> demands = collect_alternate_demands(context.graph, *match.output);
    if (demands.empty()) {
        return false;
    }

    for (ggml_type type : demands) {
        CommonMulMatWeightFormat next_format;
        if (!common_mul_mat_alternate_format_for_type(type, next_format)) {
            return false;
        }

        const size_t next_byte_count = row_byte_count(type, match.token_count, match.hidden_size);
        if (next_byte_count == 0) {
            return false;
        }

        if (type == GGML_TYPE_F32 && next_byte_count == match.output->byte_count) {
            if (!append_alternate_metadata(dispatch_match, *match.output, match.output->id, type, next_byte_count)) {
                return false;
            }
            continue;
        }

        const ValueId next_value(context.next_plan_value.value +
                                 static_cast<int32_t>(dispatch_match.transients.size()));
        dispatch_match.dispatches.push_back(
            make_get_rows_next_dispatch(match, next_format, next_value, next_byte_count));
        dispatch_match.transients.push_back({ next_value, alternate_name(type), next_byte_count, 256 });
        if (!append_alternate_metadata(dispatch_match, *match.output, next_value, type, next_byte_count)) {
            return false;
        }
    }

    if (dispatch_match.dispatches.empty()) {
        dispatch_match.dispatches.push_back(make_get_rows_dispatch(match));
    }
    dispatch_match.covered_nodes.push_back(context.root_index);
    return dispatch_match.status.success();
}

}  // namespace

void register_get_rows_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.repeat.scalar_3d_f32",
        GGML_OP_REPEAT,
        DispatchMatchKind::SingleOp,
        125,
        DispatchSource::Common,
        match_repeat_scalar_3d_f32_dispatch,
    });
    registry.add({
        "common.get_rows.scalar_3d_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        125,
        DispatchSource::Common,
        match_get_rows_scalar_3d_f32_dispatch,
    });
    registry.add({
        "common.get_rows_scale.f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_get_rows_scale_f32_dispatch,
    });
    registry.add({
        "common.get_rows_rmsnorm_binary.q8_1_x4_f16",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        330,
        DispatchSource::Common,
        match_get_rows_rmsnorm_binary_dispatch,
    });
    registry.add({
        "common.get_rows.f32_next",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        150,
        DispatchSource::Common,
        match_get_rows_f32_next_dispatch,
    });
    registry.add({
        "common.get_rows.f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_get_rows_f32_dispatch,
    });
}

}  // namespace ggml::hrx
