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
// zha: per-layer attention q/k dump for the offline ZHA study (TZHA-BUILD stage 1).
// No-op unless LLAMA_ZHA_DUMP_DIR is set in the environment.

#pragma once

struct ggml_tensor;
struct llama_ubatch;

bool llama_zha_dump_active();
void llama_zha_dump_hook(ggml_tensor * t, const char * name, int il);
void llama_zha_dump_flush(const llama_ubatch & ubatch);
void llama_zha_dump_close();
