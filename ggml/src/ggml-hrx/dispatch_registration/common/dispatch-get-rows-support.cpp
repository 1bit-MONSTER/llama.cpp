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

#include "dispatch-get-rows-support.h"

namespace ggml::hrx {

namespace {

// llama.cpp#407: the IQ3_S row kernels take a 2-D source. A batched IQ3_S source (ne[2] * ne[3] > 1) returns wrong
// rows in test-backend-ops GET_ROWS, so no form executes it. f32 sources have no such limit.
bool iq3s_batch_is_executable(const get_rows_tensor_facts & facts) {
    return !(facts.src_type == kGetRowsIq3sType && facts.src_ne2 * facts.src_ne3 > 1);
}

}  // namespace

bool get_rows_contiguous_form_supported(const get_rows_tensor_facts & facts) {
    return facts.src_contiguous && facts.ids_contiguous && facts.out_contiguous && iq3s_batch_is_executable(facts);
}

bool get_rows_route_table_form_supported(const get_rows_tensor_facts & facts) {
    return facts.src_contiguous && facts.out_contiguous && facts.ids_column_table && iq3s_batch_is_executable(facts);
}

bool get_rows_tensor_supported(const get_rows_tensor_facts & facts) {
    return get_rows_contiguous_form_supported(facts) || get_rows_route_table_form_supported(facts);
}

}  // namespace ggml::hrx
