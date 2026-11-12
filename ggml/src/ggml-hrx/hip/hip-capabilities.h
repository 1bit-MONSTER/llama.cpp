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

// Extra eager-capability ops declared by HIP matchers. ggml-hrx's supports_op only claims a node
// whose op is in its built-in capability list; a HIP matcher for an op outside that list (built in
// here or from an add-on, see README.md "Kernel add-ons") declares the op while it registers, and
// eager_capability_declared consults this set for every op it does not know. Empty without such a
// declaration, so a build without matchers for extra ops behaves as before.

#pragma once

#include "ggml.h"

namespace ggml::hrx {

// Called from a matcher's register_* function (so from register_hip_dispatches / the add-on's
// ggml_hrx_hip_addon_register) while the dispatch registry is being built.
void hip_declare_eager_op(enum ggml_op op);

// True when a matcher declared op. Builds the dispatch registries on first use so the declarations
// exist before the scheduler probes supports_op.
bool hip_eager_op_declared(enum ggml_op op);

// True when hip/dispatch-hip-top-k.cpp executes this GGML_OP_TOP_K node. device_supports_op asks this before
// claiming the op (declared above), so a TOP_K shape the matcher declines still goes to the CPU backend instead
// of failing the graph (engine #679).
bool hip_top_k_tensor_supported(const struct ggml_tensor * op);

// A per-tensor claim for a node only a fused HIP dispatch executes. The scheduler asks supports_op about each node
// alone, before any graph exists, and device_supports_op answers that from can_execute_standalone_op_as_graph: a
// one-node graph. A node whose only dispatch is a fusion rooted at it (the consumers are what make the pattern) then
// reads as unsupported, the scheduler puts it on the CPU, and the fusion never sees it in a model graph. engine #679:
// the add-on's hip.qsa_pool_blocks is rooted at the QSA indexer's f16 GET_ROWS, so in Flash-Next decode that gather
// ran on the CPU in all 12 QSA layers, with a 4.85 MB copy out and 9.7 MB back per layer per step.
//
// A matcher registers a predicate while it registers itself, and device_supports_op returns true when one holds.
// The predicate has to be no broader than its matcher: a claimed node no dispatch takes fails the whole graph
// (engine #389). Builds without such a registration claim exactly what they claimed before.
//
// GGML_HRX_HIP_TENSOR_CLAIM is 1 unless the build sets it: device_supports_op consults the predicates only when it is
// non-zero, so -DGGML_HRX_HIP_TENSOR_CLAIM=0 builds the fork as it answered before this hook (the path-taken mutant
// of tests/hrx-backend-test.cpp hip_tensor_claim, which then fails). An add-on tests `#ifdef` for the declaration,
// which exists either way.
#ifndef GGML_HRX_HIP_TENSOR_CLAIM
#    define GGML_HRX_HIP_TENSOR_CLAIM 1
#endif
using HipTensorClaim = bool (*)(const struct ggml_tensor * op);
void hip_declare_tensor_claim(HipTensorClaim claim);

// True when a registered predicate claims op. Builds the dispatch registries on first use, like
// hip_eager_op_declared, so the registrations exist before the scheduler probes supports_op.
bool hip_tensor_claimed(const struct ggml_tensor * op);

}  // namespace ggml::hrx
