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
// zha: dump the per-layer attention q/k of the full-attention layers (TZHA-BUILD stage 1).
//
// Off unless LLAMA_ZHA_DUMP_DIR is set. When set, every tensor the model builder
// reports through llm_graph_context::cb() whose name is one of the four below is
// marked as a graph output (so the scheduler keeps it alive), and after the
// ubatch computes it is appended to a file per (layer, name) as f16.
//
//   <dir>/L<il>_<name>.f16     row-major [n_tokens_total][ne1][ne0]
//   <dir>/tokens.i32           every dumped token id, in graph order
//   <dir>/ubatches.jsonl       one line per ubatch: token/pos/n so dumps can be aligned
//   <dir>/tensors.jsonl        one line per (layer, name): shape and dtype
//
// Names: Qcur/Kcur are post-RoPE (what attention uses); Qcur_normed/Kcur_normed are
// the same vectors before rotation, which is what a summary frame re-anchor needs.

#include "llama-zha-dump.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-impl.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct zha_entry {
    ggml_tensor * t;
    std::string   name;
    int           il;
};

bool  g_active = false;
bool  g_inited = false;
std::string g_dir;

std::vector<zha_entry>          g_pending;   // built this ubatch, flushed after compute
std::map<std::string, FILE *>   g_files;     // (layer, name) -> append handle
std::map<std::string, uint64_t> g_rows;      // (layer, name) -> rows written
std::vector<std::string>        g_ref;       // the (layer, name) set every ubatch must have
FILE * g_tokens  = nullptr;
FILE * g_ubatch  = nullptr;
FILE * g_tensors = nullptr;
long long g_ub_index = 0;

bool name_wanted(const char * name) {
    return strcmp(name, "Qcur")         == 0 ||
           strcmp(name, "Kcur")         == 0 ||
           strcmp(name, "Qcur_normed")  == 0 ||
           strcmp(name, "Kcur_normed")  == 0;
}

const char * short_name(const char * name) {
    if (strcmp(name, "Qcur")        == 0) return "q";
    if (strcmp(name, "Kcur")        == 0) return "k";
    if (strcmp(name, "Qcur_normed") == 0) return "qpre";
    return "kpre";
}

void init() {
    g_inited = true;
    const char * d = getenv("LLAMA_ZHA_DUMP_DIR");
    if (d == nullptr || *d == '\0') {
        return;
    }
    g_dir  = d;
    g_active = true;
    g_tokens  = fopen((g_dir + "/tokens.i32").c_str(),  "wb");
    g_ubatch  = fopen((g_dir + "/ubatches.jsonl").c_str(), "wb");
    g_tensors = fopen((g_dir + "/tensors.jsonl").c_str(),  "wb");
    if (!g_tokens || !g_ubatch || !g_tensors) {
        LLAMA_LOG_ERROR("%s: cannot open dump files in %s\n", __func__, g_dir.c_str());
        g_active = false;
    } else {
        LLAMA_LOG_INFO("%s: dumping q/k to %s\n", __func__, g_dir.c_str());
    }
}

} // namespace

bool llama_zha_dump_active() {
    if (!g_inited) {
        init();
    }
    return g_active;
}

void llama_zha_dump_hook(ggml_tensor * t, const char * name, int il) {
    if (!llama_zha_dump_active() || t == nullptr || name == nullptr) {
        return;
    }
    if (!name_wanted(name)) {
        return;
    }
    // keep it alive until we read it back after graph_compute
    ggml_set_output(t);
    // ...and say so in the name, because the output flag alone is not a request to *materialise* the value:
    // llama.cpp marks several tensors as graph outputs for API reasons (result_norm is t_embd), and ggml-hrx
    // would then decline a fusion for every one of them. The prefix is the explicit "someone will read this
    // back" marker that ggml-hrx looks for; nothing else in a run carries it, so a run without the dump is
    // planned exactly as before.
    {
        constexpr const char * k_readback_prefix = "zha-readback.";
        constexpr size_t       k_prefix_len      = sizeof("zha-readback.") - 1;
        if (strncmp(t->name, k_readback_prefix, k_prefix_len) != 0) {
            char renamed[GGML_MAX_NAME];
            snprintf(renamed, sizeof(renamed), "%s%s", k_readback_prefix, t->name);
            ggml_set_name(t, renamed);
        }
    }
    g_pending.push_back({ t, short_name(name), il });
}

void llama_zha_dump_flush(const llama_ubatch & ubatch) {
    if (!g_active || g_pending.empty()) {
        g_pending.clear();
        return;
    }

    // Pass 1: keep the first registration of each (layer, name) that this ubatch actually
    // allocated and sized. A tensor registered twice, or one the allocator dropped, would
    // otherwise append extra rows to a single file and silently shift every later row of it.
    std::vector<zha_entry> keep;
    for (const auto & e : g_pending) {
        const ggml_tensor * t = e.t;
        if (t->buffer == nullptr || (int) t->ne[2] != (int) ubatch.n_tokens || t->type != GGML_TYPE_F32) {
            continue;
        }
        bool dup = false;
        for (const auto & k : keep) {
            if (k.il == e.il && k.name == e.name) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            keep.push_back(e);
        }
    }
    std::vector<std::string> keys;
    for (const auto & e : keep) {
        keys.push_back("L" + std::to_string(e.il) + "_" + e.name);
    }
    std::sort(keys.begin(), keys.end());

    // Pass 2: all-or-nothing. The first complete set defines what every ubatch must have;
    // any other set is skipped entirely, so the per-tensor files stay row-aligned to each
    // other and to the manifest.
    const bool written = g_ref.empty() ? !keys.empty() : (keys == g_ref);
    if (written && g_ref.empty()) {
        g_ref = keys;
        LLAMA_LOG_INFO("%s: %zu tensors per ubatch\n", __func__, g_ref.size());
    } else if (!written) {
        static long long n_skipped = 0;
        if (++n_skipped <= 4) {
            LLAMA_LOG_WARN("%s: ubatch of %u tokens has %zu tensors, expected %zu - skipped\n",
                    __func__, ubatch.n_tokens, keys.size(), g_ref.size());
        }
    }

    if (written) {
        for (const auto & e : keep) {
            const ggml_tensor * t = e.t;
            const std::string key = "L" + std::to_string(e.il) + "_" + e.name;
            auto & f = g_files[key];
            if (f == nullptr) {
                f = fopen((g_dir + "/" + key + ".f16").c_str(), "wb");
                if (f == nullptr) {
                    LLAMA_LOG_ERROR("%s: cannot open %s\n", __func__, key.c_str());
                    continue;
                }
                fprintf(g_tensors, "{\"key\":\"%s\",\"il\":%d,\"name\":\"%s\",\"ne\":[%lld,%lld,%lld]}\n",
                        key.c_str(), e.il, e.name.c_str(),
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
                fflush(g_tensors);
            }
            const size_t n = (size_t) ggml_nelements(t);
            std::vector<float> f32(n);
            // A view can be strided: qwen35moe's Qcur is every other head_dim block of a
            // [q|gate] projection, so a flat read would return the interleaved rows.
            const size_t esz = ggml_element_size(t);
            const size_t row_bytes = (size_t) t->ne[0] * esz;
            const bool contiguous = t->nb[1] == row_bytes && t->nb[2] == t->nb[1] * (size_t) t->ne[1];
            if (contiguous) {
                ggml_backend_tensor_get(t, f32.data(), 0, n * sizeof(float));
            } else {
                float * dst = f32.data();
                for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                        ggml_backend_tensor_get(t, dst, i1 * t->nb[1] + i2 * t->nb[2], row_bytes);
                        dst += t->ne[0];
                    }
                }
            }
            std::vector<ggml_fp16_t> f16(n);
            ggml_fp32_to_fp16_row(f32.data(), f16.data(), (int64_t) n);
            fwrite(f16.data(), sizeof(ggml_fp16_t), n, f);
            g_rows[key] += (uint64_t) t->ne[2];
        }
    }

    if (written && ubatch.token) {
        fwrite(ubatch.token, sizeof(llama_token), ubatch.n_tokens, g_tokens);
    }
    fprintf(g_ubatch, "{\"ub\":%lld,\"n_tokens\":%u,\"written\":%d,\"tok0\":%d,\"pos0\":%d}\n",
            g_ub_index++, ubatch.n_tokens, written ? 1 : 0,
            ubatch.token ? (int) ubatch.token[0] : -1,
            ubatch.pos   ? (int) ubatch.pos[0]   : -1);
    fflush(g_ubatch);

    g_pending.clear();
}

void llama_zha_dump_close() {
    for (auto & kv : g_files) {
        if (kv.second) {
            fclose(kv.second);
        }
    }
    g_files.clear();
    if (g_tokens)  { fclose(g_tokens);  g_tokens  = nullptr; }
    if (g_ubatch)  { fflush(g_ubatch);  fclose(g_ubatch);  g_ubatch  = nullptr; }
    if (g_tensors) { fflush(g_tensors); fclose(g_tensors); g_tensors = nullptr; }
    if (g_active) {
        for (const auto & kv : g_rows) {
            LLAMA_LOG_INFO("%s: %s rows=%" PRIu64 "\n", __func__, kv.first.c_str(), kv.second);
        }
    }
    g_active = false;
}
