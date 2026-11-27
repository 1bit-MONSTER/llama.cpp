// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// engine#3521: the element ranges the binary Loom kernels DECLARE, and the one check both callers -- the placement
// predicate (supported_binary_f32_tensor) and the matchers (dispatch-binary.cpp, dispatch-glu.cpp) -- run before
// claiming a node. Kept free of ggml and HRX headers so tests/test-hrx-binary-span-cap.cpp needs neither a device nor
// a ggml link.
//
// Why the check exists: the kernels' limits are `config.decl ... range(...)` (compile parameters: a value outside
// fails the specialization at run time) and `index.assume ... range(...)` on the launch's element count (a value
// outside is undefined behaviour inside the kernel). Before this header nothing on the host compared a shape with
// either, so a long prefill whose binary span passed 2^27 was claimed for HRX and came back as 0 generated tokens
// (the 18,418- and 36,991-token Inkling references of #3521). A node past the cap must stay off HRX instead.

#pragma once

#include <cstdint>

namespace ggml::hrx {

// ggml_binary_f32 (ops/binary_f32.loom, ops/elementwise/binary_f32.loom; also the packed GLU form that reuses its
// configuration, dispatch-glu.cpp): ne0..ne2, every stride and both spans are declared range(1, 2^29), and the launch
// assumes element_count in the same range. 2^29 is also the largest count whose f32 BYTE offsets stay below 2^31
// (4 x (2^29 - 1) = 2,147,483,644), so it is the ceiling while the kernels address with signed 32-bit offsets: raise
// it only together with 64-bit addressing in the kernels.
inline constexpr uint64_t kBinaryF32MaxElements = uint64_t{ 1 } << 29;

// ggml_binary_bc_f32 (ops/binary_bc_f32.loom, ops/elementwise/binary_bc_f32.loom): the launch assumes element_count,
// ne0..ne2 and both source counts in range(1, 2^27). Unchanged by #3521.
inline constexpr uint64_t kBinaryBcF32MaxElements = uint64_t{ 1 } << 27;

// Every value a binary launch hands the kernel as a ranged parameter. Unused fields stay 1 (inside every range).
struct binary_launch_facts {
    uint64_t element_count  = 1;
    uint64_t ne[3]          = { 1, 1, 1 };
    uint64_t src0_stride[3] = { 1, 1, 1 };
    uint64_t src1_stride[3] = { 1, 1, 1 };
    uint64_t src0_span      = 1;
    uint64_t src1_span      = 1;
};

// true when every ranged value is in [1, cap]: the only launches the kernel declares itself correct for.
inline bool binary_launch_fits(const binary_launch_facts & f, uint64_t cap) {
    auto in = [cap](uint64_t v) { return v >= 1 && v <= cap; };
    if (!in(f.element_count) || !in(f.src0_span) || !in(f.src1_span)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!in(f.ne[i]) || !in(f.src0_stride[i]) || !in(f.src1_stride[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace ggml::hrx
