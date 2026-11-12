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

// written only while the registries are built (once, under the function-local static in ensure_registries_built), read
// after; a fixed table keeps the read side lock-free
constexpr size_t                                     kMaxTensorClaims = 16;
std::array<std::atomic<HipTensorClaim>, kMaxTensorClaims> g_tensor_claims = {};
std::atomic<size_t>                                  g_tensor_claim_count{ 0 };

}  // namespace

void hip_declare_eager_op(enum ggml_op op) {
    if (op >= 0 && op < GGML_OP_COUNT) {
        g_declared_ops[op].store(true, std::memory_order_release);
    }
}

namespace {

void ensure_registries_built() {
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
}

}  // namespace

bool hip_eager_op_declared(enum ggml_op op) {
    ensure_registries_built();
    return op >= 0 && op < GGML_OP_COUNT && g_declared_ops[op].load(std::memory_order_acquire);
}

void hip_declare_tensor_claim(HipTensorClaim claim) {
    if (claim == nullptr) {
        return;
    }
    // one registry per architecture runs every register_* function, so the same predicate arrives more than once
    const size_t n = g_tensor_claim_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < n; ++i) {
        if (g_tensor_claims[i].load(std::memory_order_acquire) == claim) {
            return;
        }
    }
    GGML_ASSERT(n < kMaxTensorClaims && "raise kMaxTensorClaims");
    g_tensor_claims[n].store(claim, std::memory_order_release);
    g_tensor_claim_count.store(n + 1, std::memory_order_release);
}

bool hip_tensor_claimed(const struct ggml_tensor * op) {
    ensure_registries_built();
    const size_t n = g_tensor_claim_count.load(std::memory_order_acquire);
    for (size_t i = 0; i < n; ++i) {
        const HipTensorClaim claim = g_tensor_claims[i].load(std::memory_order_acquire);
        if (claim != nullptr && claim(op)) {
            return true;
        }
    }
    return false;
}

}  // namespace ggml::hrx
