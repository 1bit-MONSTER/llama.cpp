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

}  // namespace ggml::hrx
