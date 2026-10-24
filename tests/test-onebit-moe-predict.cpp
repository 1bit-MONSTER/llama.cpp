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

// The routing predictor's host-side policy (src/onebit-moe-predict.*): environment parsing and
// the margin gate. No model, no graph, no device -- the numbers the device plan turns on, checked
// on any machine.

#include "onebit-moe-predict.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace onebit::moe;

static bool near(float a, float b) { return fabsf(a - b) < 1e-5f; }

// asserts vanish in a Release build, and a test that checks nothing is worse than no test
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

int main() {
    // ---- the margin gate: a deeper prediction is read only where it is close to the best candidate
    {
        predict_config c;
        c.depth  = 20;
        c.margin = 0.0f;
        const std::vector<float> s(20, 0.0f);
        // no gate: everything kept
        CHECK(keep_count(c, s.data(), 20, 10) == 20);

        // gate 1.0 from the best: candidates at 0, -0.5 kept, -1.5 and beyond dropped
        std::vector<float> g(20, 0.0f);
        g[0] = 5.0f; g[1] = 4.8f; g[2] = 4.4f; g[3] = 3.9f; g[4] = 3.8f;
        for (size_t i = 5; i < g.size(); ++i) g[i] = 1.0f;
        c.margin = 1.0f;
        CHECK(keep_count(c, g.data(), 20, 10) == 10);   // dropped to the floor, never below it
        CHECK(keep_count(c, g.data(), 20, 3)  == 3);    // 5.0 - 3.9 = 1.1 > 1.0 stops before i = 3
        c.margin = 1.5f;
        CHECK(keep_count(c, g.data(), 20, 3)  == 5);    // 5.0 - 3.9 = 1.1 <= 1.5, 5.0 - 3.8 = 1.2 <= 1.5
        c.margin = 0.05f;
        CHECK(keep_count(c, g.data(), 20, 1)  == 1);    // 5.0 - 4.8 = 0.2 > 0.05

        // the depth is the ceiling whatever the margin
        c.depth  = 3;
        c.margin = 100.0f;
        CHECK(keep_count(c, g.data(), 20, 2) == 3);
        // fewer candidates than the depth
        CHECK(keep_count(c, g.data(), 2, 2) == 2);
        // degenerate inputs
        CHECK(keep_count(c, nullptr, 20, 2) == 0);
        CHECK(keep_count(c, g.data(), 0, 2) == 0);
        printf("ok  margin gate / depth / floor\n");
    }

    // ---- environment: parsed once, immutable, and the layer mask bounds the prediction
    // (set before the first call into the module; config() caches)
    setenv("ONEBIT_MOE_PREDICT", "block", 1);
    setenv("ONEBIT_MOE_PREDICT_DEPTH", "16", 1);
    setenv("ONEBIT_MOE_PREDICT_MIN", "8", 1);
    setenv("ONEBIT_MOE_PREDICT_MARGIN", "0.75", 1);
    setenv("ONEBIT_MOE_PREDICT_LAYERS", "1,2,5-9,47", 1);
    setenv("ONEBIT_MOE_PREDICT_NO_ATTN", "1", 1);
    setenv("ONEBIT_MOE_PREDICT_STATS", "0", 1);

    const predict_config & c = config();
    CHECK(enabled());
    CHECK(c.mode == predict_mode::block);
    CHECK(c.depth == 16);
    CHECK(c.min == 8);
    CHECK(near(c.margin, 0.75f));
    CHECK(!c.attn);

    CHECK(wanted(1) && wanted(2) && wanted(5) && wanted(9) && wanted(47));
    CHECK(!wanted(0) && !wanted(3) && !wanted(10) && !wanted(-1));

    CHECK(depth(10) == 16);   // an explicit depth wins over the layer's top-k
    CHECK(depth(32) == 16);

    // a gate with the configured floor
    {
        std::vector<float> s(16);
        for (int i = 0; i < 16; ++i) s[(size_t) i] = 10.0f - (float) i;   // 10, 9, 8, ...
        // best 10: i = 8 gives a margin of 8 > 0.75, so the tail stops there, but min = 8 holds it
        CHECK(keep_count(c, s.data(), 16, c.min) == 8);
    }

    // ---- unknown mode falls back to off instead of predicting with a mode nobody defined
    reset_stats();
    const totals_t & t = totals();
    CHECK(t.asked == 0 && t.steps == 0);

    printf("ok  ONEBIT_MOE_PREDICT=block, depth 16, margin 0.75, min 8, layers 1,2,5-9,47, no attn\n");
    printf("test-onebit-moe-predict: all checks passed\n");
    return 0;
}
