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

#include "hip/hip-capabilities.h"

#include "dispatch_registration/dispatch-registry.h"

#include <array>
#include <atomic>

namespace ggml::hrx {
namespace {

std::array<std::atomic<bool>, GGML_OP_COUNT> g_declared_ops = {};

}  // namespace

void hip_declare_eager_op(enum ggml_op op) {
    if (op >= 0 && op < GGML_OP_COUNT) {
        g_declared_ops[op].store(true, std::memory_order_release);
    }
}

bool hip_eager_op_declared(enum ggml_op op) {
    // the registries are function-local statics: the first lookup builds them (and runs every
    // register_* function, which is where declarations happen) exactly once
    static const bool registries_built = [] {
        for (const char * architecture : { "gfx1151", "gfx1100" }) {
            DispatchTarget target;
            target.architecture = architecture;
            find_dispatch_registry(target);
        }
        return true;
    }();
    (void) registries_built;
    return op >= 0 && op < GGML_OP_COUNT && g_declared_ops[op].load(std::memory_order_acquire);
}

}  // namespace ggml::hrx
