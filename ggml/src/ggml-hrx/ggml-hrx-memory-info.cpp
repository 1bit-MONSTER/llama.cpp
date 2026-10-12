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

#include "ggml-hrx-memory-info.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace ggml::hrx {
namespace {

std::optional<uint64_t> read_u64_file(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "r");
    if (f == nullptr) {
        return std::nullopt;
    }
    unsigned long long value = 0;
    const int          n     = std::fscanf(f, "%llu", &value);
    std::fclose(f);
    if (n != 1) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(value);
}

std::optional<uint64_t> mem_available_bytes() {
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return std::nullopt;
    }
    char                    line[256];
    std::optional<uint64_t> result;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        unsigned long long kib = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            result = static_cast<uint64_t>(kib) * 1024ull;
            break;
        }
    }
    std::fclose(f);
    return result;
}

// Device directories (/sys/class/drm/cardN/device) of amdgpu cards that expose GTT counters.
std::vector<std::string> amdgpu_card_device_dirs() {
    std::vector<std::string> dirs;
    DIR *                    drm = opendir("/sys/class/drm");
    if (drm == nullptr) {
        return dirs;
    }
    while (dirent * entry = readdir(drm)) {
        const char * name = entry->d_name;
        if (std::strncmp(name, "card", 4) != 0 || name[4] == 0 || std::strchr(name, '-')) {
            continue;  // skip connectors such as card0-DP-1
        }
        const std::string dir = std::string("/sys/class/drm/") + name + "/device";
        if (read_u64_file(dir + "/mem_info_gtt_total")) {
            dirs.push_back(dir);
        }
    }
    closedir(drm);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

// Pick the card for an HRX device: the only amdgpu card, else the one whose GTT total equals the HRX total.
std::string resolve_card_dir(size_t device_total) {
    const std::vector<std::string> dirs = amdgpu_card_device_dirs();
    if (dirs.size() == 1) {
        return dirs.front();
    }
    for (const std::string & dir : dirs) {
        if (read_u64_file(dir + "/mem_info_gtt_total").value_or(0) == device_total) {
            return dir;
        }
    }
    return {};
}

}  // namespace

size_t device_free_memory(size_t device_total) {
    static std::once_flag once;
    static bool           legacy = false;
    static std::string    card_dir;
    std::call_once(once, [&]() {
        const char * mode = std::getenv("GGML_HRX_FREE_MEMORY");
        legacy            = mode != nullptr && std::strcmp(mode, "total") == 0;
        if (!legacy) {
            card_dir = resolve_card_dir(device_total);
            GGML_LOG_INFO("ggml_hrx: free memory from %s%s and MemAvailable\n",
                          card_dir.empty() ? "no amdgpu card" : card_dir.c_str(),
                          card_dir.empty() ? "" : " GTT");
        }
    });
    if (legacy) {
        return device_total;
    }
    std::optional<uint64_t> free_bytes;
    if (!card_dir.empty()) {
        const std::optional<uint64_t> gtt_total = read_u64_file(card_dir + "/mem_info_gtt_total");
        const std::optional<uint64_t> gtt_used  = read_u64_file(card_dir + "/mem_info_gtt_used");
        if (gtt_total && gtt_used) {
            free_bytes = *gtt_total > *gtt_used ? *gtt_total - *gtt_used : 0;
        }
    }
    if (const std::optional<uint64_t> available = mem_available_bytes()) {
        free_bytes = free_bytes ? std::min(*free_bytes, *available) : *available;
    }
    if (!free_bytes) {
        return device_total;
    }
    return static_cast<size_t>(std::min<uint64_t>(*free_bytes, device_total));
}

}  // namespace ggml::hrx
