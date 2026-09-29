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

// Hadamard-folded weights (the prism.hadamard.* GGUF keys of PrismML's Ternary Bonsai models).
// Such a file stores most matmul weights in a Walsh-Hadamard-rotated basis: W' = W (S H)^T with
// H the normalized Sylvester-Walsh-Hadamard matrix of prism.hadamard.block_size and S an optional
// +/-1 sign vector per input width. The model is unchanged if every activation x feeding one of
// those weights is rotated the same way first, x' = H (S x) blockwise, and the token-embedding
// rows (stored rotated too) get the inverse after the lookup. Here the rotation is an ordinary
// MUL_MAT against the H matrix, so any backend with an F32 matmul (HRX among them) runs these
// files; with the weights themselves converted to Q4_0 (the engine's tools/ternary_to_q4_0.py,
// exact for ternary weights) no new weight type is needed.
//
// Follows PrismML-Eng/llama.cpp's implementation (MIT, the ggml authors and PrismML): same GGUF
// keys, same checks, same activation transform and the same graph-coverage check.

#pragma once

#include "llama-arch.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_model;
struct llama_model_loader;

struct llama_hadamard_transform {
    ggml_tensor * rot   = nullptr;  // [block, block] F32
    ggml_tensor * signs = nullptr;  // [width] F32, nullptr for the identity sign mode
    // > 1: the activation arrives with its features in tiled head order [hd, nk, rep] and the
    // fold was computed in grouped order [hd, rep, nk] (prism.hadamard.gdn_v_grouped, ssm_out)
    int64_t perm_hd = 0, perm_nk = 0, perm_rep = 0;
};

// per-graph cache: several weights reading the same activation share one transform
using llama_hadamard_memo = std::map<std::pair<const ggml_tensor *, const ggml_tensor *>, ggml_tensor *>;

struct llama_hadamard {
    // from the GGUF keys (load_keys)
    std::unordered_map<std::string, uint32_t> weight_blocks;   // weight name -> block size
    std::unordered_map<std::string, uint32_t> inverse_blocks;  // lookup tables stored rotated
    std::map<uint32_t, std::vector<int32_t>>  sign_data;       // input width -> signs
    bool gdn_v_grouped = false;
    bool tied_output   = false;

    // bound to the model's tensors (setup)
    std::unordered_map<const ggml_tensor *, llama_hadamard_transform> rotations;
    std::unordered_map<const ggml_tensor *, llama_hadamard_transform> inverses;

    bool empty() const { return rotations.empty() && inverses.empty(); }

    // reads prism.hadamard.*; throws on anything this implementation does not verify
    void load_keys(llama_model_loader & ml, llm_arch arch);
    // makes the rotation and sign tensors next to the weights, after the weights are loaded
    void setup(const llama_model & model);

    // x as the folded weight w expects it (x itself when w is not folded)
    ggml_tensor * forward(ggml_context * ctx, const ggml_tensor * w, ggml_tensor * x, llama_hadamard_memo & memo) const;
    // rows looked up from table: back to the plain basis (rows itself when table is not rotated)
    ggml_tensor * inverse(ggml_context * ctx, const ggml_tensor * table, ggml_tensor * rows) const;

    // throws if a folded weight is consumed without its transform (a matmul path that bypasses
    // build_lora_mm) or a rotated table is read without the inverse
    void verify_graph(ggml_cgraph * gf) const;

  private:
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;
};
