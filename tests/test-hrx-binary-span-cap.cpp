// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// engine#3521: the host refuses binary launches past the ranges the Loom kernels declare, and the host constants are
// the kernels' own numbers. No device, no ggml link: binary-limits.h plus the kernel sources as text.

#include "binary-limits.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

using ggml::hrx::binary_launch_facts;
using ggml::hrx::binary_launch_fits;
using ggml::hrx::kBinaryBcF32MaxElements;
using ggml::hrx::kBinaryF32MaxElements;

static int failures = 0;
static int arms     = 0;

static void expect(bool got, bool want, const char * what) {
    ++arms;
    if (got != want) {
        std::printf("FAIL %s: got %s, want %s\n", what, got ? "fits" : "declined", want ? "fits" : "declined");
        ++failures;
    }
}

static binary_launch_facts contiguous(uint64_t rows, uint64_t width) {
    binary_launch_facts f;
    f.element_count = rows * width;
    f.ne[0]         = width;
    f.ne[1]         = rows;
    for (int i = 0; i < 3; ++i) {
        f.src0_stride[i] = f.src1_stride[i] = (i == 0) ? width : rows * width;
    }
    f.src0_span = f.src1_span = rows * width;
    return f;
}

// every upper bound N of `range(..., lo, N)` on the lines of `path` that match `line_re`
static bool every_bound_is(const char * path, const char * line_re, uint64_t want, int & seen) {
    std::ifstream in(path);
    if (!in) {
        std::printf("FAIL cannot read %s\n", path);
        ++failures;
        return false;
    }
    const std::regex line(line_re);
    const std::regex bound(R"(range\([^,]+,\s*\d+,\s*(\d+)\))");
    std::string      text;
    bool             ok = true;
    while (std::getline(in, text)) {
        if (!std::regex_search(text, line)) {
            continue;
        }
        std::smatch m;
        if (std::regex_search(text, m, bound)) {
            ++seen;
            if (std::stoull(m[1].str()) != want) {
                std::printf("FAIL %s declares %s, host constant is %llu: %s\n", path, m[1].str().c_str(),
                            (unsigned long long) want, text.c_str());
                ok = false;
            }
        }
    }
    return ok;
}

int main() {
    const uint64_t cap = kBinaryF32MaxElements;
    const uint64_t bc  = kBinaryBcF32MaxElements;

    // 1. The #3521 shape: a 36,991-token prefill at width 4,096 spans 151.5M elements, past the old 2^27, inside 2^29.
    expect(binary_launch_fits(contiguous(36991, 4096), cap), true, "36,991 x 4,096 fits the binary_f32 range");
    expect(binary_launch_fits(contiguous(36991, 4096), bc), false, "the same shape is past the broadcast kernel's 2^27");
    // 2. Both edges of the cap: exactly 2^29 elements fits, one more does not.
    expect(binary_launch_fits(contiguous(1u << 16, 1u << 13), cap), true, "exactly 2^29 elements fits");
    {
        binary_launch_facts f = contiguous(1u << 16, 1u << 13);
        f.element_count += 1;
        expect(binary_launch_fits(f, cap), false, "2^29 + 1 elements is declined");
    }
    // 3. Any single ranged value past the cap declines, not only the element count.
    {
        binary_launch_facts f = contiguous(8, 128);
        f.src1_stride[2]      = cap + 1;
        expect(binary_launch_fits(f, cap), false, "a stride past the cap is declined");
    }
    {
        binary_launch_facts f = contiguous(8, 128);
        f.src0_span           = cap + 1;
        expect(binary_launch_fits(f, cap), false, "a source span past the cap is declined");
    }
    // 4. The ranges start at 1: a zero (or a negative shape cast to uint64) is never a valid launch.
    {
        binary_launch_facts f = contiguous(8, 128);
        f.ne[2]               = 0;
        expect(binary_launch_fits(f, cap), false, "a zero dimension is declined");
        f.ne[2] = static_cast<uint64_t>(int64_t{ -1 });
        expect(binary_launch_fits(f, cap), false, "a negative dimension is declined");
    }
    // 5. The broadcast kernel keeps 2^27.
    expect(binary_launch_fits(contiguous(1u << 14, 1u << 13), bc), true, "exactly 2^27 elements fits binary_bc");
    {
        binary_launch_facts f = contiguous(1u << 14, 1u << 13);
        f.element_count += 1;
        expect(binary_launch_fits(f, bc), false, "2^27 + 1 elements is declined by binary_bc");
    }

    // 6. Drift guard: the constants above are the kernels' declared upper bounds, in every copy the manifest builds.
    int seen = 0;
    const std::string dir = HRX_LOOM_OPS_DIR;
    for (const char * f : { "/binary_f32.loom", "/elementwise/binary_f32.loom" }) {
        const std::string p = dir + f;
        ++arms;
        if (!every_bound_is(p.c_str(), R"(config\.decl @ggml\.binary_f32\.(ne\d|src\d_stride\d|src\d_span) )", cap, seen) ||
            !every_bound_is(p.c_str(), R"(%count = index\.assume %element_count)", cap, seen)) {
            ++failures;
        }
    }
    for (const char * f : { "/binary_bc_f32.loom", "/elementwise/binary_bc_f32.loom" }) {
        const std::string p = dir + f;
        ++arms;
        if (!every_bound_is(p.c_str(), R"(%count = index\.assume %element_count)", bc, seen)) {
            ++failures;
        }
    }
    if (seen < 2 * 12 + 2) {
        std::printf("FAIL the drift guard read only %d declared bounds (want >= 26): the kernel text changed shape\n", seen);
        ++failures;
    }

    if (failures != 0) {
        std::printf("test-hrx-binary-span-cap: %d failure(s) in %d arms\n", failures, arms);
        return 1;
    }
    std::printf("test-hrx-binary-span-cap: all %d arms ok (%d declared bounds checked)\n", arms, seen);
    return 0;
}
