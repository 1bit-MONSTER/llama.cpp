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

#pragma once

#include <cstddef>

namespace ggml::hrx {

// Bytes of "MemAvailable" in a /proc/meminfo body; 0 when the field is absent, unparsable, or would overflow.
// Split from the read so the warning in host-memory.cpp can be tested without a device or a live /proc.
size_t mem_available_from_meminfo(const char * meminfo);

// The same parse, saying whether a figure was taken: false when the field is absent, the body is truncated or the
// number is unparsable. "MemAvailable: 0 kB" returns true with 0 - a fact, not an unknown (see the reader below).
bool mem_available_from_meminfo(const char * meminfo, size_t & out);

// Live MemAvailable on this host; 0 when /proc/meminfo cannot be read. 0 means "unknown": a caller that wants to warn
// about memory must stay quiet rather than guess a threshold (engine #669).
size_t host_memory_available_bytes();

// The same read, but saying whether a figure was actually taken: true and *out filled when MemAvailable was parsed,
// false (and *out untouched) when /proc/meminfo or the field is missing. A *measured* zero returns true with 0 in *out -
// a host with nothing left is a fact, not an unknown, and engine #1502's admission check must refuse on it rather than
// drop the host term (the size_t form above cannot tell the two apart).
bool host_memory_available_bytes(size_t & out);

}  // namespace ggml::hrx
