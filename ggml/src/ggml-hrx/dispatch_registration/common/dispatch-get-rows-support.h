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
//
// engine#2704: the GET_ROWS tensor conditions, written once, as data.
//
// Two decisions decide whether a GET_ROWS node runs on HRX, and they were made in two files that cannot see each
// other: `dispatch-get-rows.cpp` decides at compute time whether a matcher executes the node, and
// `device_supports_op` (ggml-hrx.cpp) decides at placement time whether the node stays on HRX. When they disagreed the
// node was placed on HRX with no dispatch and the graph failed with "unsupported HRX node" -- which is how engine#2639
// (a strided VIEW as the source) and llama.cpp#407 (batched IQ3_S sources) each cost a hand-written clause in the
// placement path, a clause that re-encodes a matcher's property in another file and drifts the next time a predicate
// tightens.
//
// The shape of the fix matters, because the family has TWO forms with different conditions and a single conjunction is
// wrong in a way that would cost throughput rather than fail loudly:
//
//   * the CONTIGUOUS form (match_get_rows_f32 and everything that calls it: f32, f32_next, scale_f32,
//     rmsnorm_binary) needs the weight, the ids and the output all contiguous;
//   * the ROUTE-TABLE form (match_get_rows_scalar_3d_f32, the qwen3.6-35b routed experts) reads the ids through a
//     row stride on purpose, so its ids are NOT contiguous and it is fine.
//
// So a placement claim is "SOME form can execute this node", and each form is its own named predicate that its own
// matcher calls. A conjunctive `ids contiguous` here would move every routed GET_ROWS to the CPU while every test that
// looks at one matcher at a time stayed green -- which is why this file has three predicates and an arm per form.
//
// The rule for adding a condition: it belongs HERE only if the placement path can see it too, i.e. it is a property of
// the node's own tensors. Conditions that need the graph -- a consumer's type, an exact id span, a hidden size -- stay
// in the matcher; the placement claim is then an over-approximation, which is the safe direction (it can only cost a CPU
// fallback for a node that no matcher claims, never a claimed node with no dispatch).
//
// No ggml dependency: the two adapters (one from ggml_tensor in ggml-hrx.cpp, one from a dispatch Value in
// dispatch-get-rows.cpp) build these facts wherever those types already live, so tests/test-hrx-get-rows-facts.cpp runs
// with no device, no graph and no libhrx.

#pragma once

#include <cstdint>

namespace ggml::hrx {

// What the placement path and the matchers can both see about one GET_ROWS node. Named fields, not positional, so a
// new condition cannot silently read its neighbour.
struct get_rows_tensor_facts {
    bool    src_contiguous  = false;  // the weight (or the scales tensor of the scalar_3d form)
    bool    ids_contiguous  = false;  // the row table, densely packed
    bool    ids_column_table = false; // i32 columns with a row stride: nb[0] == sizeof(int32_t), nb[1] % sizeof(int32_t) == 0
    bool    out_contiguous  = false;  // the destination
    int32_t src_type        = 0;      // a ggml_type value, kept as int32_t so this header stays ggml-free
    int64_t src_ne2         = 1;      // source batch dimensions: the IQ row kernels take a 2-D source only
    int64_t src_ne3         = 1;
};

// The contiguous form. This is the condition match_get_rows_f32 refuses a node on, and the derived forms (scale_f32,
// rmsnorm_binary) inherit it by calling that match first.
bool get_rows_contiguous_form_supported(const get_rows_tensor_facts & facts);

// The route-table form: match_get_rows_scalar_3d_f32 reads the ids through `route_stride`, so a strided ids tensor is
// expected here. It gates the scales and the output on contiguity and the ids on being i32 columns.
bool get_rows_route_table_form_supported(const get_rows_tensor_facts & facts);

// The placement question: can ANY form execute this node? false therefore means "place it on the CPU", and true means
// "claim it only if the fused/standalone question also passes", never "claim it".
bool get_rows_tensor_supported(const get_rows_tensor_facts & facts);

// GGML_TYPE_IQ3_S repeated as a plain integer so this header stays ggml-free. The test asserts it against the real enum,
// so an upstream renumbering fails an arm instead of silently declining the wrong type.
constexpr int32_t kGetRowsIq3sType = 21;

}  // namespace ggml::hrx
