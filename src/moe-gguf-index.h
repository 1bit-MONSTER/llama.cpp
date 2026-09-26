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

// Where each routed expert's weights sit in a GGUF file (or the shards of a split file):
// the header is parsed for tensor names, shapes, types and offsets; nothing else is read.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::moe {

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    int file = 0;             // index into GgufIndex::files
    uint64_t offset = 0;      // absolute file offset of the tensor data
    uint64_t bytes = 0;
};

// One part (gate, up or down) of one layer's experts: expert e's bytes are
// [base + e * stride, base + e * stride + bytes) in files[file].
struct ExpertPart {
    std::string tensor;
    int file = 0;
    uint64_t base = 0, stride = 0, bytes = 0;
};

struct GgufIndex {
    std::vector<std::string> files;
    std::vector<GgufTensor> tensors;
    std::map<int, std::vector<ExpertPart>> experts;  // MoE layer -> parts, in file order
    int n_expert = 0;

    // Throws std::runtime_error on an unreadable file or an unknown tensor type.
    static GgufIndex open(const std::string& path);
    uint64_t expert_bytes(int layer) const;  // one expert, all parts
};

}  // namespace onebit::moe
