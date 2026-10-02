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
// Portions follow PrismML-Eng/llama.cpp (https://github.com/PrismML-Eng/llama.cpp), the llama.cpp
// fork of PrismML, who made the Ternary Bonsai models and the prism.hadamard.* format. Thanks to
// PrismML for publishing both. Those portions are used under the MIT License:
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Hadamard-folded weights (prism.hadamard.*): see llama-hadamard.h.

#include "llama-hadamard.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>

// keys straight from the GGUF: llama_model_loader instantiates its string-keyed get_key / get_arr
// for a few types only (bool and array ones fail to link with GCC)
static bool gguf_bool(const gguf_context * ctx, const char * key, bool & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return false;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(format("%s is not a bool", key));
    }
    out = gguf_get_val_bool(ctx, id);
    return true;
}

static bool gguf_strings(const gguf_context * ctx, const char * key, std::vector<std::string> & out, bool required) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key));
        }
        return false;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("%s is not an array of strings", key));
    }
    out.clear();
    for (size_t i = 0; i < gguf_get_arr_n(ctx, id); ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return true;
}

static void gguf_ints(const gguf_context * ctx, const char * key, std::vector<int32_t> & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(format("key not found in model: %s", key));
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_INT32) {
        throw std::runtime_error(format("%s is not an array of int32", key));
    }
    const auto * data = static_cast<const int32_t *>(gguf_get_arr_data(ctx, id));
    out.assign(data, data + gguf_get_arr_n(ctx, id));
}

static bool gguf_u32(const gguf_context * ctx, const char * key, uint32_t & out, bool required) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key));
        }
        return false;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx, id); return true;
        case GGUF_TYPE_INT32:  out = (uint32_t) gguf_get_val_i32(ctx, id); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx, id); return true;
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8(ctx, id); return true;
        default: throw std::runtime_error(format("%s is not an unsigned integer", key));
    }
}

static void gguf_str(const gguf_context * ctx, const char * key, std::string & out) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(format("%s: missing or not a string", key));
    }
    out = gguf_get_val_str(ctx, id);
}

void llama_hadamard::load_keys(llama_model_loader & ml, llm_arch arch) {
    const gguf_context * meta = ml.metadata;
    uint32_t version = 0;
    gguf_bool(meta, "prism.hadamard.tied_output", tied_output);
    // The engine's tools/hadamard_q4_0.py stamps onebit.hadamard_q4_0 = 32 instead: every Q4_0 weight of
    // the file is rotated by the normalized 32-point Sylvester Walsh-Hadamard matrix per block along
    // its input dimension, without signs, which is the prism.hadamard transform with block_size 32 and
    // sign_mode identity. It names no weights: the Q4_0 tensors are the rotated ones.
    bool onebit = false;
    if (!gguf_u32(meta, "prism.hadamard.version", version, false)) {
        if (tied_output) {
            throw std::runtime_error("prism.hadamard.tied_output without prism.hadamard.version");
        }
        const int64_t kid = gguf_find_key(meta, "onebit.hadamard_q4_0");
        if (kid < 0) {
            return;
        }
        const gguf_type kt = gguf_get_kv_type(meta, kid);
        const int64_t stamp = kt == GGUF_TYPE_INT32  ? gguf_get_val_i32(meta, kid) :
                              kt == GGUF_TYPE_UINT32 ? (int64_t) gguf_get_val_u32(meta, kid) : -1;
        if (stamp != 32) {
            throw std::runtime_error(format("unsupported onebit.hadamard_q4_0: %lld", (long long) stamp));
        }
        onebit = true;
        onebit_q4_0 = true;
    } else if (version != 1 && version != 2) {
        throw std::runtime_error(format("unsupported prism.hadamard.version: %u", version));
    }
    if ((version == 2) != tied_output) {
        throw std::runtime_error("prism.hadamard version 2 requires tied_output=true; version 1 forbids it");
    }
    if (!onebit && tied_output && ml.get_weight("output.weight")) {
        throw std::runtime_error("prism.hadamard.tied_output requires output.weight to be absent");
    }

    uint32_t block_size = 0;
    std::string transform, axis, sign_mode;
    std::vector<std::string> weight_names;
    if (onebit) {
        block_size = 32;
        transform  = "normalized-sylvester-walsh-hadamard";
        axis       = "input-last-dimension";
        sign_mode  = "identity";
        // the names are collected below, from the Q4_0 tensors on a verified matmul path
    } else {
        gguf_u32(meta, "prism.hadamard.block_size", block_size, true);
        gguf_str(meta, "prism.hadamard.transform", transform);
        gguf_str(meta, "prism.hadamard.axis", axis);
        gguf_str(meta, "prism.hadamard.sign_mode", sign_mode);
        gguf_strings(meta, "prism.hadamard.weight_names", weight_names, true);
    }

    if (block_size == 0 || (block_size & (block_size - 1)) != 0) {
        throw std::runtime_error(format("invalid prism.hadamard.block_size: %u", block_size));
    }
    if (transform != "normalized-sylvester-walsh-hadamard") {
        throw std::runtime_error(format("unsupported prism.hadamard.transform: %s", transform.c_str()));
    }
    if (axis != "input-last-dimension") {
        throw std::runtime_error(format("unsupported prism.hadamard.axis: %s", axis.c_str()));
    }
    if (sign_mode != "identity" && sign_mode != "explicit") {
        throw std::runtime_error(format("unsupported prism.hadamard.sign_mode: %s", sign_mode.c_str()));
    }
    if (!onebit && weight_names.empty()) {
        throw std::runtime_error("prism.hadamard.weight_names is empty");
    }

    if (sign_mode == "explicit") {
        std::vector<int32_t> widths, values;
        gguf_ints(meta, "prism.hadamard.sign_widths", widths);
        gguf_ints(meta, "prism.hadamard.sign_values", values);
        // explicit with no widths would read as identity later and silently change the model
        if (widths.empty()) {
            throw std::runtime_error("prism.hadamard.sign_mode is explicit but sign_widths is empty");
        }
        size_t off = 0;
        for (const int32_t width : widths) {
            if (width <= 0 || (uint32_t) width % block_size != 0 || off + width > values.size()) {
                throw std::runtime_error(format("invalid prism.hadamard sign width: %d", width));
            }
            auto & vec = sign_data[width];
            vec.assign(values.begin() + off, values.begin() + off + width);
            for (const int32_t v : vec) {
                if (v != 1 && v != -1) {
                    throw std::runtime_error("prism.hadamard sign values must be +/-1");
                }
            }
            off += width;
        }
        if (off != values.size()) {
            throw std::runtime_error("prism.hadamard.sign_values length mismatch");
        }
    }

    gguf_bool(meta, "prism.hadamard.gdn_v_grouped", gdn_v_grouped);

    // the activation transform is applied in build_lora_mm / build_lora_mm_id only: refuse
    // architectures and tensor kinds not verified to route every matmul through them
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN3NEXT:
            break;
        default:
            throw std::runtime_error(format(
                "prism.hadamard: arch '%s' is not verified to apply the activation transform to all folded weights",
                llm_arch_name(arch)));
    }

    const auto foldable = [](const std::string & name) {
        static const char * kinds[] = {
            "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
            "ffn_gate", "ffn_up", "ffn_down",
            "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
            "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
            "ssm_out", "ssm_alpha", "ssm_beta",
        };
        if (name == "output.weight") {
            return true;  // the output head goes through build_lora_mm in every arch
        }
        if (name.compare(0, 4, "blk.") != 0) {
            return false;
        }
        size_t pos = 4;
        while (pos < name.size() && isdigit((unsigned char) name[pos])) {
            pos++;
        }
        if (pos == 4 || pos >= name.size() || name[pos] != '.') {
            return false;
        }
        pos++;
        for (const char * kind : kinds) {
            if (name.compare(pos, std::string::npos, std::string(kind) + ".weight") == 0) {
                return true;
            }
        }
        return false;
    };
    if (onebit) {
        // tools/hadamard_q4_0.py rotates the Q4_0 matmul weights; a Q4_0 lookup table (token_embd)
        // is written in the plain basis and needs no transform
        for (const auto & [name, w] : ml.weights_map) {
            if (w.tensor->type == GGML_TYPE_Q4_0 && foldable(name)) {
                weight_names.push_back(name);
            }
        }
        if (weight_names.empty()) {
            throw std::runtime_error("onebit.hadamard_q4_0: the file has no Q4_0 matmul weights");
        }
    }
    for (const auto & name : weight_names) {
        if (!foldable(name)) {
            throw std::runtime_error(format("prism.hadamard: weight '%s' is not on a verified Hadamard-aware matmul path", name.c_str()));
        }
        if (!weight_blocks.emplace(name, block_size).second) {
            throw std::runtime_error(format("duplicate prism.hadamard weight: %s", name.c_str()));
        }
    }

    // tables read by row lookup store rotated rows: the lookup result gets the inverse
    std::vector<std::string> inverse_names;
    gguf_strings(meta, "prism.hadamard.inverse_weight_names", inverse_names, false);
    for (const auto & name : inverse_names) {
        // only the token-embedding lookup applies it; any other table would stay rotated
        if (name != "token_embd.weight") {
            throw std::runtime_error(format("prism.hadamard: weight '%s' is not a verified inverse-after-lookup table", name.c_str()));
        }
        if (weight_blocks.count(name) || !inverse_blocks.emplace(name, block_size).second) {
            throw std::runtime_error(format("duplicate prism.hadamard inverse weight: %s", name.c_str()));
        }
    }

    if (tied_output) {
        const auto it = inverse_blocks.find("token_embd.weight");
        if (it == inverse_blocks.end()) {
            throw std::runtime_error("prism.hadamard.tied_output requires a latent token embedding");
        }
        weight_blocks.emplace("token_embd.weight", it->second);
    } else if (inverse_blocks.count("token_embd.weight") && !ml.get_weight("output.weight")) {
        throw std::runtime_error("a tied Hadamard output requires version 2 and tied_output=true");
    }
}

// one [n] F32 tensor or [n, n] matrix in its own buffer of type buft, owned by this struct
static ggml_tensor * hadamard_const(std::vector<ggml_context_ptr> & ctxs, std::vector<ggml_backend_buffer_ptr> & bufs,
                                    ggml_backend_buffer_type_t buft, const char * name, int64_t ne0, int64_t ne1,
                                    const std::vector<float> & data) {
    ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error("failed to create a Hadamard context");
    }
    ggml_tensor * t = ne1 > 1 ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1)
                              : ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ne0);
    ggml_set_name(t, name);
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
    if (!buf) {
        throw std::runtime_error(format("unable to allocate a %s buffer for %s", ggml_backend_buft_name(buft), name));
    }
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    ctxs.emplace_back(std::move(ctx));
    bufs.emplace_back(std::move(buf));
    return t;
}

void llama_hadamard::setup(const llama_model & model) {
    if (weight_blocks.empty() && inverse_blocks.empty()) {
        return;
    }
    std::map<std::pair<uint32_t, ggml_backend_buffer_type_t>, ggml_tensor *> rots, signs;
    // the inverse (lookup side) must not inherit a host buffer from a CPU-mapped table, or every
    // token's transform crosses devices: it takes the forward rotations' buffer type
    ggml_backend_buffer_type_t preferred = nullptr;

    const std::pair<const std::unordered_map<std::string, uint32_t> *, decltype(rotations) *> groups[] = {
        { &weight_blocks, &rotations }, { &inverse_blocks, &inverses },
    };
    for (const auto & [blocks, target] : groups) {
        for (const auto & [name, block] : *blocks) {
            const ggml_tensor * w = model.get_tensor(name.c_str());
            if (tied_output && name == "token_embd.weight") {
                w = target == &rotations ? model.output : model.tok_embd;
                if (!w || strcmp(w->name, "token_embd.weight") != 0) {
                    throw std::runtime_error("prism.hadamard.tied_output is not bound to the token embedding");
                }
            }
            if (!w && onebit_q4_0) {
                continue;  // a weight of the file the model does not load (the MTP layer without --mtp)
            }
            if (!w) {
                throw std::runtime_error(format("prism.hadamard weight not found: %s", name.c_str()));
            }
            if (w->ne[0] % block != 0) {
                throw std::runtime_error(format("prism.hadamard block size %u does not divide input dimension %lld for %s",
                                                block, (long long) w->ne[0], name.c_str()));
            }
            if (!w->buffer) {
                throw std::runtime_error(format("prism.hadamard weight has no buffer: %s", name.c_str()));
            }
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(w->buffer);
            // CPU extra buffer types (CPU_REPACK) only take tensors they can repack
            if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft)) {
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    buft = ggml_backend_dev_buffer_type(dev);
                }
            }
            if (target == &rotations) {
                preferred = buft;
            } else if (preferred) {
                buft = preferred;
            }

            ggml_tensor *& rot = rots[{ block, buft }];
            if (!rot) {
                std::vector<float> h((size_t) block * block);
                const float scale = 1.0f / sqrtf((float) block);
                for (uint32_t r = 0; r < block; ++r) {
                    for (uint32_t c = 0; c < block; ++c) {
                        h[(size_t) r * block + c] = __builtin_parity(r & c) ? -scale : scale;
                    }
                }
                char n[GGML_MAX_NAME];
                snprintf(n, sizeof(n), "prism.hadamard.%u", block);
                rot = hadamard_const(ctxs, bufs, buft, n, block, block, h);
            }

            ggml_tensor * sign = nullptr;
            if (!sign_data.empty()) {
                const uint32_t width = (uint32_t) w->ne[0];
                const auto sd = sign_data.find(width);
                if (sd == sign_data.end()) {
                    throw std::runtime_error(format("prism.hadamard has no sign vector for width %u (%s)", width, name.c_str()));
                }
                ggml_tensor *& s = signs[{ width, buft }];
                if (!s) {
                    char n[GGML_MAX_NAME];
                    snprintf(n, sizeof(n), "prism.hadamard.signs.%u", width);
                    s = hadamard_const(ctxs, bufs, buft, n, width, 1, std::vector<float>(sd->second.begin(), sd->second.end()));
                }
                sign = s;
            }

            llama_hadamard_transform t { rot, sign };
            if (gdn_v_grouped && name.find(".ssm_out.") != std::string::npos) {
                const int64_t n_v = model.hparams.ssm_dt_rank;
                const int64_t n_k = model.hparams.ssm_n_group;
                if (n_k <= 0 || n_v <= 0 || n_v % n_k != 0 || w->ne[0] % n_v != 0) {
                    throw std::runtime_error(format("prism.hadamard: bad GDN head geometry for %s", name.c_str()));
                }
                t.perm_hd  = w->ne[0] / n_v;
                t.perm_nk  = n_k;
                t.perm_rep = n_v / n_k;
            }
            target->emplace(w, t);
        }
    }
    LLAMA_LOG_INFO("%s: %zu Hadamard-folded weight(s) (%zu inverse-lookup), %zu rotation(s), %zu sign vector(s)\n",
                   __func__, rotations.size() + inverses.size(), inverses.size(), rots.size(), signs.size());
}

ggml_tensor * llama_hadamard::forward(ggml_context * ctx, const ggml_tensor * w, ggml_tensor * x, llama_hadamard_memo & memo) const {
    const auto it = rotations.find(w);
    if (it == rotations.end()) {
        return x;
    }
    const auto & t = it->second;
    const auto key = std::make_pair((const ggml_tensor *) x, (const ggml_tensor *) t.rot);
    if (const auto m = memo.find(key); m != memo.end()) {
        return m->second;
    }
    ggml_tensor * cur = x;
    if (t.perm_rep > 1) {
        // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
        ggml_tensor * c = ggml_is_contiguous(cur) ? cur : ggml_cont(ctx, cur);
        const int64_t n = c->ne[1] * c->ne[2] * c->ne[3];
        const int64_t ne1 = c->ne[1], ne2 = c->ne[2], ne3 = c->ne[3];
        c = ggml_reshape_4d(ctx, c, t.perm_hd, t.perm_nk, t.perm_rep, n);
        c = ggml_cont(ctx, ggml_permute(ctx, c, 0, 2, 1, 3));
        cur = ggml_reshape_4d(ctx, c, t.perm_hd * t.perm_nk * t.perm_rep, ne1, ne2, ne3);
    }
    if (t.signs) {
        cur = ggml_mul(ctx, cur, t.signs);
    }
    cur = llama_mul_mat_hadamard(ctx, cur, t.rot);
    memo[key] = cur;
    return cur;
}

ggml_tensor * llama_hadamard::inverse(ggml_context * ctx, const ggml_tensor * table, ggml_tensor * rows) const {
    const auto it = inverses.find(table);
    if (it == inverses.end()) {
        return rows;
    }
    // rows hold z = H (s * h); the normalized Sylvester matrix is symmetric and its own
    // inverse, so h = s * (H z)
    ggml_tensor * cur = llama_mul_mat_hadamard(ctx, rows, it->second.rot);
    return it->second.signs ? ggml_mul(ctx, cur, it->second.signs) : cur;
}

void llama_hadamard::verify_graph(ggml_cgraph * gf) const {
    const auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };
    const auto is_rotation = [](const ggml_tensor * t) {
        return t && t->op == GGML_OP_MUL_MAT && ((const int32_t *) t->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD;
    };
    std::map<const ggml_tensor *, bool> lookups;  // lookups of rotated tables -> inverse applied

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        if (is_rotation(node)) {
            if (const auto lk = lookups.find(unwrap(node->src[1])); lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }
        const auto it = rotations.find(node->src[0]);
        if (it == rotations.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format("Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        if (!(is_rotation(src) && src->src[0] == it->second.rot)) {
            throw std::runtime_error(format("Hadamard-folded weight '%s' is consumed without its activation transform; "
                                            "this graph's matmul path does not support prism.hadamard folding", node->src[0]->name));
        }
    }
    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            throw std::runtime_error(format("Hadamard-latent table '%s' is read without the inverse transform", node->src[0]->name));
        }
    }
}
