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

#include "host-memory-available.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace ggml::hrx {
namespace {

constexpr size_t kMeminfoBytes = 16 * 1024;

}  // namespace

size_t mem_available_from_meminfo(const char * meminfo) {
    if (meminfo == nullptr) {
        return 0;
    }
    for (const char * line = meminfo; *line != '\0';) {
        if (std::strncmp(line, "MemAvailable:", sizeof("MemAvailable:") - 1) == 0) {
            unsigned long long kib = 0;
            if (std::sscanf(line + sizeof("MemAvailable:") - 1, " %llu", &kib) != 1) {
                return 0;
            }
            if (kib > std::numeric_limits<size_t>::max() / 1024) {
                return 0;
            }
            return static_cast<size_t>(kib) * 1024;
        }
        const char * next = std::strchr(line, '\n');
        if (next == nullptr) {
            return 0;
        }
        line = next + 1;
    }
    return 0;
}

size_t host_memory_available_bytes() {
    FILE * file = std::fopen("/proc/meminfo", "rb");
    if (file == nullptr) {
        return 0;
    }
    char   body[kMeminfoBytes] = {};
    const size_t read  = std::fread(body, 1, sizeof(body) - 1, file);
    std::fclose(file);
    body[read]         = '\0';
    return mem_available_from_meminfo(body);
}

}  // namespace ggml::hrx
