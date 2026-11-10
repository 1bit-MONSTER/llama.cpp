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

bool mem_available_from_meminfo(const char * meminfo, size_t & out) {
    if (meminfo == nullptr) {
        return false;
    }
    for (const char * line = meminfo; *line != '\0';) {
        if (std::strncmp(line, "MemAvailable:", sizeof("MemAvailable:") - 1) == 0) {
            unsigned long long kib = 0;
            if (std::sscanf(line + sizeof("MemAvailable:") - 1, " %llu", &kib) != 1) {
                return false;   // the field is there but unreadable: unknown, not zero
            }
            if (kib > std::numeric_limits<size_t>::max() / 1024) {
                return false;
            }
            out = static_cast<size_t>(kib) * 1024;
            return true;        // "MemAvailable: 0 kB" is a reading, not an unknown
        }
        const char * next = std::strchr(line, '\n');
        if (next == nullptr) {
            return false;
        }
        line = next + 1;
    }
    return false;
}

size_t mem_available_from_meminfo(const char * meminfo) {
    size_t out = 0;
    return mem_available_from_meminfo(meminfo, out) ? out : 0;
}

bool host_memory_available_bytes(size_t & out) {
    FILE * file = std::fopen("/proc/meminfo", "rb");
    if (file == nullptr) {
        return false;
    }
    char   body[kMeminfoBytes] = {};
    const size_t read  = std::fread(body, 1, sizeof(body) - 1, file);
    std::fclose(file);
    body[read]         = '\0';
    return mem_available_from_meminfo(body, out);
}

size_t host_memory_available_bytes() {
    size_t out = 0;
    return host_memory_available_bytes(out) ? out : 0;
}

}  // namespace ggml::hrx
