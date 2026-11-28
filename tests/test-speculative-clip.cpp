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

// engine#3613 / #3627: common_speculative_clip_to_vocab (common/speculative-clip.h) cuts a DFlash draft at its first id
// the target's tokenizer lacks, and the per-token candidate lists with it. Replaces test-dflash-logits (the in-graph
// -INF pad it tested is gone: the drafter logits are now as wide as llama_context copies by construction).
#include "speculative-clip.h"

#include <cstdio>
#include <vector>

static int fails = 0;
static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += ok ? 0 : 1;
}

int main() {
    const int32_t n_tgt = 200058;   // Inkling-2-Mini's real tokens; its padded rows run to 201024

    std::vector<llama_token> d = { 10, 200057, 200058, 5 };
    std::vector<std::vector<llama_token_data>> q(4, std::vector<llama_token_data>(2));
    common_speculative_clip_to_vocab(d, &q, n_tgt);
    check(d.size() == 2 && d[1] == 200057, "cut at the first id past the target's last token, keeping the last real id");
    check(q.size() == 2, "candidate lists cut in step");

    d = { 1, 2, 3 };
    q.assign(3, std::vector<llama_token_data>(1));
    common_speculative_clip_to_vocab(d, &q, n_tgt);
    check(d.size() == 3 && q.size() == 3, "an all-valid draft is unchanged");

    d = { 201023, 7 };
    common_speculative_clip_to_vocab(d, nullptr, n_tgt);
    check(d.empty(), "a first id past the vocab empties the draft (no candidate lists: greedy)");

    d = { 4, -1, 6 };
    common_speculative_clip_to_vocab(d, nullptr, n_tgt);
    check(d.size() == 1, "a negative id is cut too");

    d = { 9, 300000 };
    q.assign(1, std::vector<llama_token_data>(1));   // shorter candidate list than the draft: never grown
    common_speculative_clip_to_vocab(d, &q, n_tgt);
    check(d.size() == 1 && q.size() == 1, "a shorter candidate list is left as is");

    std::printf("test-speculative-clip: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
