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

#include "dispatch-i4-prefill-opt-in.h"

#include <cstdlib>

namespace ggml::hrx {

bool symmetric_i4_prefill_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_HRX_SYMMETRIC_I4_PREFILL");
        return env != nullptr && *env != '\0' && std::atoi(env) != 0;
    }();
    return enabled;
}

}  // namespace ggml::hrx
