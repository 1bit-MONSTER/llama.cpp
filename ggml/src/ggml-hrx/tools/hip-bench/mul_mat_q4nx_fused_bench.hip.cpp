// Per-op correctness test for the HRX0 Q4NX fused dequant+matmul kernel.
//
// Builds a synthetic Q4NX weight blob (all 16 signed int4 values, BF16
// scales/zero-points, sanitize edge cases), a deterministic F32 activation
// matrix, runs hrx_mul_mat_q4nx_fused_f32 on the device and compares every
// element against an independent CPU reference (dequantize + matmul).
//
//   amdclang++ -x hip --offload-arch=gfx1151 -O2 mul_mat_q4nx_fused_bench.hip.cpp -lamdhip64 -o /tmp/q4nx_fused_bench
//   /tmp/q4nx_fused_bench

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../kernels/mul_mat_q4nx_fused_f32.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static constexpr int kTileBytes = 5120;

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<uint16_t>((u + 0x8000u) >> 16);
}

static float bf16_to_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Independent host reference: dequantize W(row, i) for the Q4NX tile layout.
static float reference_weight(const std::vector<uint8_t> & blob, int64_t row, int64_t i, int64_t n_tile_cols) {
    const int64_t tile_row = row / 32;
    const int64_t in_tile_row = row % 32;
    const int64_t tile_col = i / 256;
    const int64_t in_tile_col = i % 256;
    const int64_t tile_base = (tile_row * n_tile_cols + tile_col) * kTileBytes;
    const int64_t scale_idx = in_tile_row * 8 + (in_tile_col / 32);

    auto rd = [&](int64_t off) -> float {
        const uint16_t h = static_cast<uint16_t>(blob[off]) | (static_cast<uint16_t>(blob[off + 1]) << 8);
        return bf16_to_f32(h);
    };
    float scale = rd(tile_base + scale_idx * 2);
    float zp    = rd(tile_base + 512 + scale_idx * 2);
    if (!(std::fabs(scale) < 100.0f)) scale = 0.0f;
    if (!(std::fabs(zp) < 100.0f))    zp = 0.0f;

    const int64_t lane_row = in_tile_row % 16;
    const int64_t byte_off = tile_base + 1024 + (in_tile_row / 16) * 2048 + in_tile_col * 8 + (lane_row / 2);
    const unsigned int packed = blob[byte_off];
    const int q = (in_tile_row % 2) == 0 ? static_cast<int>(packed & 0x0Fu) : static_cast<int>(packed >> 4);
    const int val = q < 8 ? q : q - 16;
    return static_cast<float>(val) * scale + zp;
}

int main() {
    const int64_t n_tc = 2;                     // column tiles
    const int64_t k = n_tc * 256;               // 512
    const int64_t n_tr = 2;                     // row tiles
    const int64_t n_tiles = n_tr * n_tc;
    const int64_t rows = n_tr * 32;             // 64
    const int64_t cols = 3;

    std::vector<uint8_t> blob(static_cast<size_t>(n_tiles) * kTileBytes, 0);
    for (int64_t t = 0; t < n_tiles; ++t) {
        uint8_t * packed = blob.data() + t * kTileBytes + 1024;
        for (size_t i = 0; i < 4096; ++i) {
            packed[i] = static_cast<uint8_t>((i * 37u + 11u + static_cast<size_t>(t) * 5u) & 0xFFu);
        }
        for (int64_t s = 0; s < 256; ++s) {
            float sc = 0.25f + static_cast<float>((s + t) % 5) * 0.5f;
            float zp = -0.125f + static_cast<float>((s * 3 + t) % 7) * 0.0625f;
            if (s == 5) sc = 250.0f;
            if (s == 6) zp = std::nanf("");
            const uint16_t sc_h = f32_to_bf16(sc), zp_h = f32_to_bf16(zp);
            blob[t * kTileBytes + s * 2 + 0] = static_cast<uint8_t>(sc_h & 0xFF);
            blob[t * kTileBytes + s * 2 + 1] = static_cast<uint8_t>(sc_h >> 8);
            blob[t * kTileBytes + 512 + s * 2 + 0] = static_cast<uint8_t>(zp_h & 0xFF);
            blob[t * kTileBytes + 512 + s * 2 + 1] = static_cast<uint8_t>(zp_h >> 8);
        }
    }

    std::vector<float> src1(static_cast<size_t>(k * cols));
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t c = 0; c < cols; ++c) {
            src1[static_cast<size_t>(c * k + i)] = 0.01f * static_cast<float>((i * 7 + c * 13) % 23) - 0.11f;
        }
    }

    const unsigned char * d_src0 = nullptr;
    const float * d_src1 = nullptr;
    float * d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src0, blob.size()));
    HIP_CHECK(hipMalloc(&d_src1, src1.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst, static_cast<size_t>(rows * cols) * sizeof(float)));
    HIP_CHECK(hipMemcpy(const_cast<unsigned char *>(d_src0), blob.data(), blob.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(const_cast<float *>(d_src1), src1.data(), src1.size() * sizeof(float), hipMemcpyHostToDevice));

    hrx_mul_mat_q4nx_fused_f32_constants c = { k, rows, cols, n_tc };
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(cols), 1);
    hrx_mul_mat_q4nx_fused_f32<<<grid, 256>>>(d_src0, d_src1, d_dst, c);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got(static_cast<size_t>(rows * cols));
    HIP_CHECK(hipMemcpy(got.data(), d_dst, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    double max_abs = 0.0, max_rel = 0.0;
    int64_t bad = 0;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            double ref = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                ref += static_cast<double>(reference_weight(blob, row, i, n_tc)) * static_cast<double>(src1[col * k + i]);
            }
            const double have = got[static_cast<size_t>(row + col * rows)];
            const double d = std::fabs(ref - have);
            const double r = d / std::max(1.0, std::fabs(ref));
            if (d > max_abs) max_abs = d;
            if (r > max_rel) max_rel = r;
            if (!(d <= 1e-3 + 1e-4 * std::fabs(ref))) ++bad;
        }
    }

    std::printf("hrx_mul_mat_q4nx_fused_f32: k=%lld rows=%lld cols=%lld n_tc=%lld\n",
            (long long) k, (long long) rows, (long long) cols, (long long) n_tc);
    std::printf("max_abs_diff=%.3e max_rel_diff=%.3e mismatches=%lld\n", max_abs, max_rel, (long long) bad);

    HIP_CHECK(hipFree(const_cast<unsigned char *>(d_src0)));
    HIP_CHECK(hipFree(const_cast<float *>(d_src1)));
    HIP_CHECK(hipFree(d_dst));

    if (bad != 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
