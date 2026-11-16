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

// engine#2704: these arms pin the GET_ROWS accept predicates that the placement path (ggml-hrx.cpp device_supports_op)
// and the matchers (dispatch-get-rows.cpp) both call. No device, no graph, no libhrx: the bug being prevented is two
// callers DISAGREEING, and a disagreement cannot be caught by a test of either caller alone.
//
// Each arm is a shape that history has already paid for, or the shape that a too-narrow conjunction would break:
//   engine#2639  a strided VIEW as the source  -> no form executes it, placement must decline
//   llama.cpp#407 a batched IQ3_S source       -> no form executes it, placement must decline
//   qwen3.6-35b  a routed GET_ROWS whose ids are a strided route table -> the ROUTE-TABLE form executes it, so a
//                conjunctive "ids contiguous" in the placement predicate would silently move it to the CPU

#include "dispatch_registration/common/dispatch-get-rows-support.h"

#include "ggml.h"

#include <cstdio>

using ggml::hrx::get_rows_contiguous_form_supported;
using ggml::hrx::get_rows_route_table_form_supported;
using ggml::hrx::get_rows_tensor_facts;
using ggml::hrx::get_rows_tensor_supported;
using ggml::hrx::kGetRowsIq3sType;

static int failures = 0;

static void expect(bool got, bool want, const char * what) {
    if (got != want) {
        std::printf("FAIL %s: got %s, want %s\n", what, got ? "execute" : "decline", want ? "execute" : "decline");
        ++failures;
    }
}

// The ordinary dense case: contiguous weight, contiguous ids, contiguous output, f32.
static get_rows_tensor_facts dense() {
    get_rows_tensor_facts facts;
    facts.src_contiguous   = true;
    facts.ids_contiguous   = true;
    facts.ids_column_table = true;
    facts.out_contiguous   = true;
    facts.src_type         = static_cast<int32_t>(GGML_TYPE_F32);
    return facts;
}

// The routed case the scalar_3d matcher serves: ids are an i32 column table with a row stride, so they are not
// contiguous, and that is exactly what the route kernel is written for.
static get_rows_tensor_facts route_table() {
    get_rows_tensor_facts facts = dense();
    facts.ids_contiguous = false;
    return facts;
}

int main() {
    // 0. The header repeats GGML_TYPE_IQ3_S as a plain integer to stay ggml-free; if upstream renumbers the enum this
    //    arm says so, instead of the predicates quietly gating the wrong type.
    if (kGetRowsIq3sType != static_cast<int32_t>(GGML_TYPE_IQ3_S)) {
        std::printf("FAIL kGetRowsIq3sType = %d but GGML_TYPE_IQ3_S = %d\n", kGetRowsIq3sType,
                    static_cast<int>(GGML_TYPE_IQ3_S));
        ++failures;
    }

    // 1. Dense: both forms apply, so placement claims it.
    expect(get_rows_contiguous_form_supported(dense()), true, "dense: contiguous form");
    expect(get_rows_tensor_supported(dense()), true, "dense: placement");

    // 2. Routed: only the route-table form applies. The contiguous form must refuse it -- that is what keeps the two
    //    forms from being confused -- while the placement question stays yes.
    expect(get_rows_route_table_form_supported(route_table()), true, "routed: route-table form");
    expect(get_rows_contiguous_form_supported(route_table()), false, "routed: contiguous form must refuse strided ids");
    expect(get_rows_tensor_supported(route_table()), true, "routed: placement must still claim it");

    // 3. engine#2639: a strided VIEW as the source. Neither form can read it, so placement declines and the CPU serves
    //    the node instead of the graph failing with "unsupported HRX node".
    {
        auto facts = dense();
        facts.src_contiguous = false;
        expect(get_rows_tensor_supported(facts), false, "strided source (engine#2639)");
    }
    // 4. A non-contiguous output is refused by both forms.
    {
        auto facts = dense();
        facts.out_contiguous = false;
        expect(get_rows_tensor_supported(facts), false, "non-contiguous output");
    }
    // 5. ids that are neither contiguous nor an i32 column table (a strided view of something wider than int32).
    {
        auto facts = dense();
        facts.ids_contiguous   = false;
        facts.ids_column_table = false;
        expect(get_rows_tensor_supported(facts), false, "ids neither contiguous nor a column table");
    }
    // 6. llama.cpp#407: a batched IQ3_S source returns wrong rows, so no form executes it.
    {
        auto facts = dense();
        facts.src_type = static_cast<int32_t>(GGML_TYPE_IQ3_S);
        facts.src_ne2  = 2;
        expect(get_rows_tensor_supported(facts), false, "batched IQ3_S source (llama.cpp#407)");
        expect(get_rows_contiguous_form_supported(facts), false, "batched IQ3_S source: contiguous form");
    }
    // 7. ... while a 2-D IQ3_S source is exactly what test-hrx-get-rows-iq covers, and stays claimable.
    {
        auto facts = dense();
        facts.src_type = static_cast<int32_t>(GGML_TYPE_IQ3_S);
        expect(get_rows_tensor_supported(facts), true, "2-D IQ3_S source");
    }
    // 8. Batching itself is not the limit: a batched f32 source has no #407 restriction. If a matcher ever stops
    //    accepting it, that condition belongs in these predicates with its own arm, not as a clause in the placement
    //    path where nobody can see it.
    {
        auto facts = dense();
        facts.src_ne2 = 2;
        expect(get_rows_tensor_supported(facts), true, "batched f32 source is not an IQ limitation");
    }

    if (failures == 0) {
        std::printf("ok: 9 arms\n");
        return 0;
    }
    std::printf("FAILURES: %d\n", failures);
    return 1;
}
