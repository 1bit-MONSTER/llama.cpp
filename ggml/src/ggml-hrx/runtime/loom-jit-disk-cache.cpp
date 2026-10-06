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

#include "loom-jit-disk-cache.h"

#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace ggml::hrx {

namespace {

constexpr uint32_t kMagic   = 0x434a4231;  // "1BJC"
constexpr uint32_t kVersion = 1;

std::mutex  g_mutex;
std::string g_target;

// Two FNV-1a lanes with different offset bases give a 128-bit key.
struct KeyHash {
    uint64_t a = 0xcbf29ce484222325ull;
    uint64_t b = 0x84222325cbf29ce4ull;

    void bytes(const void * data, size_t size) {
        const auto * p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            a = (a ^ p[i]) * 0x100000001b3ull;
            b = (b ^ p[i]) * 0x100000001b3ull;
            b ^= b >> 29;
        }
    }
    void field(const void * data, size_t size) {  // length-prefixed, so fields cannot run together
        const uint64_t n = size;
        bytes(&n, sizeof n);
        bytes(data, size);
    }
    void str(const std::string & s) { field(s.data(), s.size()); }
    std::string hex() const {
        char out[33];
        std::snprintf(out, sizeof out, "%016llx%016llx", (unsigned long long) a, (unsigned long long) b);
        return out;
    }
};

bool enabled() {
    static const bool on = [] {
        const char * v = std::getenv("GGML_HRX_JIT_CACHE");
        if (v != nullptr && (std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0)) {
            return false;
        }
        const char * s = std::getenv("GGML_HRX_LOOM_SANITIZER");
        return s == nullptr || s[0] == '\0';
    }();
    return on;
}

// The library holding this code (and the statically linked Loom compiler): path, size, mtime.
const std::string & library_identity() {
    static const std::string id = [] {
        Dl_info info = {};
        struct stat st = {};
        if (dladdr(reinterpret_cast<const void *>(&loom_jit_disk_cache_set_target), &info) == 0 ||
            info.dli_fname == nullptr || stat(info.dli_fname, &st) != 0) {
            return std::string();
        }
        return std::string(info.dli_fname) + ":" + std::to_string(st.st_size) + ":" + std::to_string(st.st_mtime);
    }();
    return id;
}

const std::filesystem::path & cache_dir() {
    static const std::filesystem::path dir = [] {
        if (const char * d = std::getenv("GGML_HRX_JIT_CACHE_DIR"); d != nullptr && d[0] != '\0') {
            return std::filesystem::path(d);
        }
        const char * home = std::getenv("HOME");
        return home ? std::filesystem::path(home) / ".cache" / "1bit" / "hrx-jit" : std::filesystem::path();
    }();
    return dir;
}

std::string key_for(const LoomKernelCompileRequest & request) {
    KeyHash h;
    const uint32_t version = kVersion;
    h.field(&version, sizeof version);
    h.str(library_identity());
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        h.str(g_target);
    }
    h.field(request.source_data, request.source_size);
    h.field(&request.source_format, sizeof request.source_format);
    h.str(request.source_identifier);
    h.str(request.symbol);
    h.str(request.launch_config_symbol);
    for (const auto & dep : request.dependencies) {
        h.field(dep.source_data, dep.source_size);
        h.field(&dep.source_format, sizeof dep.source_format);
        h.str(dep.source_identifier ? dep.source_identifier : "");
    }
    std::vector<std::pair<std::string, std::string>> configs = request.config_storage;
    std::sort(configs.begin(), configs.end());
    for (const auto & c : configs) {
        h.str(c.first);
        h.str(c.second);
    }
    h.field(request.workload.data(), request.workload.size() * sizeof(int64_t));
    return h.hex();
}

bool read_exact(FILE * f, void * out, size_t n) { return n == 0 || std::fread(out, 1, n, f) == n; }

bool read_blob(FILE * f, void ** out, size_t * out_size, bool nul_terminate) {
    uint64_t n = 0;
    if (!read_exact(f, &n, sizeof n) || n > (1ull << 30)) {
        return false;
    }
    if (n == 0) {
        *out = nullptr;
        *out_size = 0;
        return true;
    }
    void * p = nullptr;
    if (!hrx_status_is_ok(hrx_host_allocator_malloc(hrx_host_allocator_system(), n + (nul_terminate ? 1 : 0), &p))) {
        return false;
    }
    if (!read_exact(f, p, n)) {
        hrx_host_allocator_free(hrx_host_allocator_system(), p);
        return false;
    }
    if (nul_terminate) {
        static_cast<char *>(p)[n] = '\0';
    }
    *out = p;
    *out_size = n;
    return true;
}

}  // namespace

void loom_jit_disk_cache_set_target(const char * target) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_target = target ? target : "";
}

bool loom_jit_disk_cache_load(const LoomKernelCompileRequest & request, ggml_hrx_loom_jit_compile_result & compiled) {
    if (!enabled() || cache_dir().empty() || library_identity().empty()) {
        return false;
    }
    const std::filesystem::path path = cache_dir() / (key_for(request) + ".bin");
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    ggml_hrx_loom_jit_compile_result result;
    uint32_t magic = 0, version = 0;
    auto & lc = result.launch_config;
    uint64_t workload_argument_count = 0;
    bool ok = read_exact(f, &magic, sizeof magic) && read_exact(f, &version, sizeof version) && magic == kMagic &&
              version == kVersion && read_exact(f, lc.workgroup_count.data(), sizeof(uint32_t) * 3) &&
              read_exact(f, lc.workgroup_size.data(), sizeof(uint32_t) * 3) &&
              read_exact(f, &lc.subgroup_size, sizeof lc.subgroup_size) &&
              read_exact(f, &lc.workgroup_storage_bytes, sizeof lc.workgroup_storage_bytes) &&
              read_exact(f, &workload_argument_count, sizeof workload_argument_count) &&
              read_exact(f, &lc.fields, sizeof lc.fields);
    if (ok) {
        lc.workload_argument_count = static_cast<size_t>(workload_argument_count);
        size_t manifest_size = 0;
        ok = read_blob(f, &result.hsaco_data, &result.hsaco_size, false) &&
             read_blob(f, reinterpret_cast<void **>(&result.manifest_json), &manifest_size, true) &&
             result.hsaco_size > 0;
        result.manifest_json_size = manifest_size;
    }
    std::fclose(f);
    if (!ok) {
        return false;  // a damaged entry is recompiled and overwritten
    }
    compiled = std::move(result);
    return true;
}

void loom_jit_disk_cache_store(const LoomKernelCompileRequest &         request,
                               const ggml_hrx_loom_jit_compile_result & compiled) {
    if (!enabled() || cache_dir().empty() || library_identity().empty() || compiled.hsaco_data == nullptr ||
        compiled.hsaco_size == 0) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(cache_dir(), ec);
    const std::string key = key_for(request);
    const std::filesystem::path path = cache_dir() / (key + ".bin");
    const std::filesystem::path tmp =
        cache_dir() / (key + ".tmp." + std::to_string(reinterpret_cast<uintptr_t>(&compiled)));
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return;
    }
    const auto & lc = compiled.launch_config;
    const uint64_t workload_argument_count = lc.workload_argument_count;
    const uint64_t hsaco_size = compiled.hsaco_size;
    const uint64_t manifest_size = compiled.manifest_json ? compiled.manifest_json_size : 0;
    bool ok = std::fwrite(&kMagic, sizeof kMagic, 1, f) == 1 && std::fwrite(&kVersion, sizeof kVersion, 1, f) == 1 &&
              std::fwrite(lc.workgroup_count.data(), sizeof(uint32_t), 3, f) == 3 &&
              std::fwrite(lc.workgroup_size.data(), sizeof(uint32_t), 3, f) == 3 &&
              std::fwrite(&lc.subgroup_size, sizeof lc.subgroup_size, 1, f) == 1 &&
              std::fwrite(&lc.workgroup_storage_bytes, sizeof lc.workgroup_storage_bytes, 1, f) == 1 &&
              std::fwrite(&workload_argument_count, sizeof workload_argument_count, 1, f) == 1 &&
              std::fwrite(&lc.fields, sizeof lc.fields, 1, f) == 1 &&
              std::fwrite(&hsaco_size, sizeof hsaco_size, 1, f) == 1 &&
              std::fwrite(compiled.hsaco_data, 1, compiled.hsaco_size, f) == compiled.hsaco_size &&
              std::fwrite(&manifest_size, sizeof manifest_size, 1, f) == 1 &&
              (manifest_size == 0 || std::fwrite(compiled.manifest_json, 1, manifest_size, f) == manifest_size);
    ok = (std::fclose(f) == 0) && ok;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);  // atomic: readers see a whole entry or none
    }
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
    }
}

}  // namespace ggml::hrx
