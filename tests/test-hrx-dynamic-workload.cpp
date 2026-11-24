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

// engine#3382: a dynamic workload compiles one program for any token count and evaluates the launch program per
// dispatch. This is the host-tier arm: it plans the common generic ops, checks the workload specialization the
// matchers select, and compiles one kernel to show the per-count launch geometry. No device is opened.

#include "dispatch/dispatch-scheduler.h"
#include "graph/graph.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/loom-kernel-jit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

namespace {

constexpr const char * kScheduleTarget = "gfx1151";
constexpr const char * kCompileTarget  = "gfx1100";

struct PlannedKernel {
    ggml::hrx::KernelSpecialization specialization;
    ggml::hrx::Dispatch            dispatch;
};

static std::string kernel_name_for_id(uint64_t kernel_id) {
    const ggml::hrx::KernelResolveResult resolved =
        ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), kScheduleTarget, kernel_id);
    REQUIRE(resolved.found());
    return ggml::hrx::kernel_definition_name(*resolved.definition);
}

static ggml_context * new_context() {
    ggml_init_params params = {};
    params.mem_size         = 64 * 1024 * 1024;
    params.no_alloc         = true;
    ggml_context * ctx      = ggml_init(params);
    REQUIRE(ctx != nullptr);
    return ctx;
}

static ggml_cgraph * graph_of(ggml_context * ctx, ggml_tensor * output) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    REQUIRE(graph != nullptr);
    ggml_build_forward_expand(graph, output);
    return graph;
}

// Plans one multi-node graph and returns the dispatch that serves `kernel_name`.
static PlannedKernel plan_kernel(ggml_cgraph * graph, const char * kernel_name) {
    ggml::hrx::GraphImportResult imported = ggml::hrx::import_ggml_graph(*graph);
    REQUIRE(imported.valid());
    ggml::hrx::DispatchScheduler scheduler;
    REQUIRE(scheduler.schedule_graph(imported.graph, { kScheduleTarget }));
    REQUIRE(scheduler.plan().valid());
    for (const ggml::hrx::Dispatch & dispatch : scheduler.plan().dispatches) {
        if (kernel_name_for_id(dispatch.kernel.kernel_id) == kernel_name) {
            return { dispatch.kernel, dispatch };
        }
    }
    std::fprintf(stderr, "no dispatch for %s in the plan\n", kernel_name);
    std::abort();
}

static int64_t integer_parameter(const ggml::hrx::KernelSpecialization & kernel, const char * name) {
    const auto found = kernel.integer_parameters.find(name);
    REQUIRE(found != kernel.integer_parameters.end());
    return found->second;
}

static std::string compile_parameter(const ggml::hrx::KernelSpecialization & kernel, const char * name) {
    const auto found = kernel.compile_parameters.find(name);
    REQUIRE(found != kernel.compile_parameters.end());
    return found->second;
}

static void require_dynamic(const PlannedKernel & planned, const char * kernel_name) {
    REQUIRE(planned.specialization.workload_specialization == ggml::hrx::WorkloadSpecialization::Dynamic);
    std::printf("PASS dynamic %-28s workload=%lld\n", kernel_name,
                static_cast<long long>(planned.specialization.integer_parameters.begin()->second));
}

// One graph per op family, so nothing fuses and each plan row names the kernel the common matcher chose.
static void run_plan_checks() {
    for (const int64_t tokens : { 3, 18, 512 }) {
        {
            ggml_context * ctx   = new_context();
            ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, tokens);
            ggml_tensor *  out   = ggml_rms_norm(ctx, input, 1.0e-5f);
            REQUIRE(out != nullptr);
            const PlannedKernel planned = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_rmsnorm_f32");
            require_dynamic(planned, "loom_libs:ggml_rmsnorm_f32");
            REQUIRE(integer_parameter(planned.specialization, "token_count") == tokens);
            ggml_free(ctx);
        }
        {
            ggml_context * ctx   = new_context();
            ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, tokens);
            ggml_tensor *  out   = ggml_scale(ctx, input, 0.5f);
            REQUIRE(out != nullptr);
            const PlannedKernel planned = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_scale_f32");
            require_dynamic(planned, "loom_libs:ggml_scale_f32");
            REQUIRE(integer_parameter(planned.specialization, "element_count") == 512 * tokens);
            ggml_free(ctx);
        }
        {
            ggml_context * ctx   = new_context();
            ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, tokens);
            ggml_tensor *  out   = ggml_sqr(ctx, input);
            REQUIRE(out != nullptr);
            const PlannedKernel planned = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_unary_f32");
            require_dynamic(planned, "loom_libs:ggml_unary_f32");
            REQUIRE(integer_parameter(planned.specialization, "element_count") == 512 * tokens);
            ggml_free(ctx);
        }
        {
            ggml_context * ctx   = new_context();
            ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, tokens);
            ggml_tensor *  out   = ggml_cont(ctx, input);
            REQUIRE(out != nullptr);
            const PlannedKernel planned = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_copy_f32");
            require_dynamic(planned, "loom_libs:ggml_copy_f32");
            REQUIRE(integer_parameter(planned.specialization, "element_count") == 512 * tokens);
            ggml_free(ctx);
        }
    }

    // concat_dim0's compiled kernel reads row_count as a compile-time config (kernel-corpus/kernels/loom-libs/
    // ops/copy_f32.loom:318), so its dispatch stays Exact: a static form that is unchanged by the dynamic restore.
    for (const int64_t rows : { 7, 9 }) {
        ggml_context * ctx = new_context();
        ggml_tensor *  lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, rows);
        ggml_tensor *  rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, rows);
        ggml_tensor *  out = ggml_concat(ctx, lhs, rhs, 0);
        REQUIRE(out != nullptr);
        const PlannedKernel planned = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_concat_dim0_f32");
        REQUIRE(planned.specialization.workload_specialization == ggml::hrx::WorkloadSpecialization::Exact);
        REQUIRE(compile_parameter(planned.specialization, "ggml.concat_dim0_f32.row_count") == std::to_string(rows));
        std::printf("PASS exact   loom_libs:ggml_concat_dim0_f32 row_count=%lld\n", static_cast<long long>(rows));
        ggml_free(ctx);
    }
}

static ggml::hrx::KernelResolveResult resolve_kernel(const PlannedKernel & planned) {
    return ggml::hrx::resolve_kernel_definition(ggml::hrx::get_qwen_kernel_corpus(), kScheduleTarget,
                                                planned.specialization.kernel_id);
}

// One compiled program for every token count in the class: two dispatches that differ only in their workload
// values share a cache entry. The same shape under a static form does not: its compile config selects the entry.
static void run_compile_cache_checks() {
    ggml::hrx::KernelExecutableCache cache(ggml::hrx::LoomJitMode::Sync);
    const ggml::hrx::KernelExecutablePrepareContext context = { nullptr, kCompileTarget, nullptr };

    PlannedKernel        scale_18 = {};
    std::vector<uint8_t> constants_18;
    {
        ggml_context * ctx   = new_context();
        ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 18);
        ggml_tensor *  out   = ggml_scale(ctx, input, 0.5f);
        REQUIRE(out != nullptr);
        scale_18 = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_scale_f32");
        ggml_free(ctx);
    }
    PlannedKernel scale_512 = {};
    {
        ggml_context * ctx   = new_context();
        ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
        ggml_tensor *  out   = ggml_scale(ctx, input, 0.5f);
        REQUIRE(out != nullptr);
        scale_512 = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_scale_f32");
        ggml_free(ctx);
    }

    const ggml::hrx::KernelResolveResult scale_definition = resolve_kernel(scale_18);
    REQUIRE(scale_definition.found());
    ggml::hrx::KernelExecutableRef first =
        cache.get_or_compile(context, *scale_definition.definition, scale_18.dispatch, constants_18);
    REQUIRE(first.valid());
    std::vector<uint8_t> constants_512;
    ggml::hrx::KernelExecutableRef second =
        cache.get_or_compile(context, *scale_definition.definition, scale_512.dispatch, constants_512);
    REQUIRE(second.valid());
    REQUIRE(second.entry == first.entry);
    std::printf("PASS one compiled program serves element_count 9216 and 262144\n");

    // A static form keys on its compile config: two row counts are two entries, as before the restore.
    PlannedKernel concat_7 = {};
    {
        ggml_context * ctx = new_context();
        ggml_tensor *  lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 7);
        ggml_tensor *  rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 7);
        ggml_tensor *  out = ggml_concat(ctx, lhs, rhs, 0);
        REQUIRE(out != nullptr);
        concat_7 = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_concat_dim0_f32");
        ggml_free(ctx);
    }
    PlannedKernel concat_9 = {};
    {
        ggml_context * ctx = new_context();
        ggml_tensor *  lhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 9);
        ggml_tensor *  rhs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 9);
        ggml_tensor *  out = ggml_concat(ctx, lhs, rhs, 0);
        REQUIRE(out != nullptr);
        concat_9 = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_concat_dim0_f32");
        ggml_free(ctx);
    }
    const ggml::hrx::KernelResolveResult concat_definition = resolve_kernel(concat_7);
    REQUIRE(concat_definition.found());
    std::vector<uint8_t> concat_constants_7;
    ggml::hrx::KernelExecutableRef concat_first =
        cache.get_or_compile(context, *concat_definition.definition, concat_7.dispatch, concat_constants_7);
    REQUIRE(concat_first.valid());
    std::vector<uint8_t> concat_constants_9;
    ggml::hrx::KernelExecutableRef concat_second =
        cache.get_or_compile(context, *concat_definition.definition, concat_9.dispatch, concat_constants_9);
    REQUIRE(concat_second.valid());
    REQUIRE(concat_second.entry != concat_first.entry);
    std::printf("PASS static concat_dim0 at row_count 7 and 9 stays two entries\n");
}

static ggml::hrx::LoomKernelCompileRequest make_request(const PlannedKernel & planned, bool specialize) {
    const ggml::hrx::KernelResolveResult definition = resolve_kernel(planned);
    REQUIRE(definition.found());
    const ggml::hrx::KernelDefinition & kernel = *definition.definition;
    REQUIRE(!kernel.compile_recipe.primary_sources.empty());
    const ggml::hrx::KernelSourceRef & source = kernel.compile_recipe.primary_sources.front();
    REQUIRE(source.contents != nullptr);

    ggml::hrx::LoomKernelCompileRequest request;
    request.source_data          = source.contents->source.data;
    request.source_size          = source.contents->source.length;
    request.source_identifier    = source.path != nullptr ? source.path : "";
    request.symbol               = kernel.symbol != nullptr ? kernel.symbol : "";
    request.launch_config_symbol = kernel.name != nullptr ? kernel.name : "";
    request.specialize_workload  = specialize;
    for (const ggml::hrx::KernelCompileConfig & config : kernel.compile_config) {
        request.config_storage.emplace_back(config.key != nullptr ? config.key : "",
                                            config.value != nullptr ? config.value : "");
    }
    for (const auto & config : planned.specialization.compile_parameters) {
        request.config_storage.push_back(config);
    }
    return request;
}

static ggml_hrx_loom_jit_compile_result compile_once(const ggml::hrx::LoomKernelCompileRequest & request,
                                                     const char *                               key) {
    std::string error;
    std::unique_ptr<ggml::hrx::LoomJit> jit =
        ggml::hrx::create_loom_jit(kCompileTarget, ggml::hrx::LoomJitMode::Sync, error);
    REQUIRE(jit != nullptr);
    ggml::hrx::LoomCompiledKernelRef compiled = jit->compile(key, request);
    REQUIRE(compiled != nullptr);
    REQUIRE(compiled->resolve());
    ggml_hrx_loom_jit_compile_result result = compiled->take_result();
    REQUIRE(result.hsaco_data != nullptr);
    return result;
}

static bool require_hrx(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        return false;
    }
    return true;
}

// The compile of a dynamic workload keeps its launch program, and evaluating it with each dispatch's own workload
// values gives that dispatch's grid. This is the step the exact path replaces with one compile-time evaluation.
static void run_launch_program_checks() {
    ggml_context * ctx   = new_context();
    ggml_tensor *  input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 18);
    ggml_tensor *  out   = ggml_scale(ctx, input, 0.5f);
    REQUIRE(out != nullptr);
    PlannedKernel planned = plan_kernel(graph_of(ctx, out), "loom_libs:ggml_scale_f32");
    ggml_free(ctx);

    ggml_hrx_loom_jit_compile_result result = compile_once(make_request(planned, false), "dynamic-scale-launch");
    REQUIRE(result.launch_program != nullptr);

    const int64_t counts[] = { 512, 1024 };
    for (const int64_t count : counts) {
        ggml_hrx_loom_jit_launch_config launch = {};
        REQUIRE(require_hrx(
            ggml_hrx_loom_jit_launch_program_evaluate(result.launch_program, &count, 1, &launch)));
        REQUIRE(launch.workgroup_count[0] == static_cast<uint32_t>((count + 255) / 256));
        REQUIRE(launch.workgroup_size[0] == 256);
        std::printf("PASS launch program: element_count %lld -> %u workgroup(s)\n", static_cast<long long>(count),
                    launch.workgroup_count[0]);
    }
    result.reset();

    // A static form keeps the old path: launch evaluated at compile, no launch program retained.
    ggml_context * concat_ctx = new_context();
    ggml_tensor *  lhs        = ggml_new_tensor_2d(concat_ctx, GGML_TYPE_F32, 64, 7);
    ggml_tensor *  rhs        = ggml_new_tensor_2d(concat_ctx, GGML_TYPE_F32, 32, 7);
    ggml_tensor *  concat_out = ggml_concat(concat_ctx, lhs, rhs, 0);
    REQUIRE(concat_out != nullptr);
    const PlannedKernel concat_planned =
        plan_kernel(graph_of(concat_ctx, concat_out), "loom_libs:ggml_concat_dim0_f32");
    ggml_free(concat_ctx);

    ggml_hrx_loom_jit_compile_result exact = compile_once(make_request(concat_planned, true), "exact-concat-launch");
    REQUIRE(exact.launch_program == nullptr);
    REQUIRE(exact.launch_config.workgroup_count[0] == 3);
    REQUIRE(exact.launch_config.workgroup_size[0] == 256);
    std::printf("PASS exact path: concat_dim0 row_count 7 -> compile-time launch of %u workgroup(s)\n",
                exact.launch_config.workgroup_count[0]);
    exact.reset();
}

}  // namespace

int main() {
    run_plan_checks();
    run_compile_cache_checks();
    run_launch_program_checks();
    std::printf("test-hrx-dynamic-workload: ok\n");
    return 0;
}
