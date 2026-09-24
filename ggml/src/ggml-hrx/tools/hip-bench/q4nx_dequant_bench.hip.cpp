// Per-op correctness test for the HRX0 Q4NX dequant kernel.
//
// Builds a synthetic Q4NX tile blob (covering all 16 signed int4 nibble values,
// BF16 scales/zero-points, and the sanitize cases), runs hrx_q4nx_dequant_f32
// on the device and compares every element against an independent CPU
// reference of the documented format.
//
//   amdclang++ -x hip --offload-arch=gfx1151 -O2 q4nx_dequant_bench.hip.cpp -lamdhip64 -o /tmp/q4nx_dequant_bench
//   /tmp/q4nx_dequant_bench
//
// The reference is transcribed from ggml/include/ggml.h (GGML_Q4NX_TILE_*) and
// ggml-hrx2/kernels/q4nx_dequant_f32.loom.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../kernels/q4nx_dequant_f32.hip.cpp"

#define HIP_CHECK(expr) do { \
    hipError_t _err = (expr); \
    if (_err != hipSuccess) { \
        std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        std::exit(2); \
    } \
} while (0)

static constexpr int kTileBytes = 5120;
static constexpr int kTileRows = 32;
static constexpr int kTileCols = 256;

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<uint16_t>((u + 0x8000u) >> 16);  // round-to-nearest-even approx
}

static float bf16_to_f32(uint16_t h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Independent CPU reference of the Q4NX tile decode.
static float reference_dequant(const std::vector<uint8_t> & blob, int64_t row, int64_t col, int64_t n_tile_cols) {
    const int64_t group     = col / 32;
    const int64_t g_in_tile = group % 8;
    const int64_t tile_col  = col / 256;
    const int64_t tile_row  = row / 32;
    const int64_t tile_base = (tile_row * n_tile_cols + tile_col) * kTileBytes;
    const int64_t in_tile_row = row % 32;
    const int64_t scale_idx = in_tile_row * 8 + g_in_tile;

    auto rd_bf16 = [&](int64_t off) -> float {
        const uint16_t h = static_cast<uint16_t>(blob[off]) | (static_cast<uint16_t>(blob[off + 1]) << 8);
        return bf16_to_f32(h);
    };

    float scale = rd_bf16(tile_base + scale_idx * 2);
    float zp    = rd_bf16(tile_base + 512 + scale_idx * 2);
    if (!(std::fabs(scale) < 100.0f)) scale = 0.0f;
    if (!(std::fabs(zp) < 100.0f))    zp = 0.0f;

    const int64_t lane_row = in_tile_row % 16;
    const int64_t byte_idx = lane_row / 2;
    const int64_t nib      = in_tile_row % 2;
    const int64_t byte_off = tile_base + 1024 + (in_tile_row / 16) * 2048 + (col % 256) * 8 + byte_idx;

    const unsigned int packed = blob[byte_off];
    const int q = nib == 0 ? static_cast<int>(packed & 0x0Fu) : static_cast<int>(packed >> 4);
    const int val = q < 8 ? q : q - 16;
    return static_cast<float>(val) * scale + zp;
}

int main() {
    const int64_t n_tile_cols = 3, n_tiles_row = 2;
    const int64_t ncols = n_tile_cols * kTileCols;      // 768
    const int64_t nrows = n_tiles_row * kTileRows;      // 64
    const int64_t n_tiles = n_tile_cols * n_tiles_row;
    const int64_t total = nrows * ncols;

    std::vector<uint8_t> blob(static_cast<size_t>(n_tiles) * kTileBytes, 0);

    // Packed int4: a pattern that walks all 16 nibble values.
    for (int64_t t = 0; t < n_tiles; ++t) {
        uint8_t * packed = blob.data() + t * kTileBytes + 1024;
        for (size_t i = 0; i < 4096; ++i) {
            packed[i] = static_cast<uint8_t>((i * 37u + 11u + static_cast<size_t>(t) * 5u) & 0xFFu);
        }
        // Scales / zero-points: normal values plus sanitize cases.
        for (int64_t s = 0; s < 256; ++s) {
            float sc = 0.25f + static_cast<float>((s + t) % 5) * 0.5f;
            float zp = -0.125f + static_cast<float>((s * 3 + t) % 7) * 0.0625f;
            if (s == 7)  sc = 250.0f;                       // |scale| > 100 -> 0
            if (s == 8)  zp = -1e30f;                       // |zp| > 100 -> 0
            if (s == 9)  sc = std::nanf("");                // NaN -> 0
            if (s == 10) zp = INFINITY;                     // inf -> 0
            const uint16_t sc_h = f32_to_bf16(sc), zp_h = f32_to_bf16(zp);
            blob[t * kTileBytes + s * 2 + 0] = static_cast<uint8_t>(sc_h & 0xFF);
            blob[t * kTileBytes + s * 2 + 1] = static_cast<uint8_t>(sc_h >> 8);
            blob[t * kTileBytes + 512 + s * 2 + 0] = static_cast<uint8_t>(zp_h & 0xFF);
            blob[t * kTileBytes + 512 + s * 2 + 1] = static_cast<uint8_t>(zp_h >> 8);
        }
    }

    const unsigned char * d_src = nullptr;
    float * d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, blob.size()));
    HIP_CHECK(hipMalloc(&d_dst, static_cast<size_t>(total) * sizeof(float)));
    HIP_CHECK(hipMemcpy(const_cast<unsigned char *>(d_src), blob.data(), blob.size(), hipMemcpyHostToDevice));

    hrx_q4nx_dequant_f32_constants c = { nrows, ncols, n_tile_cols };
    const unsigned int threads = 256;
    const unsigned int groups = static_cast<unsigned int>((total + threads - 1) / threads);
    hrx_q4nx_dequant_f32<<<groups, threads>>>(d_src, d_dst, c);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got(static_cast<size_t>(total));
    HIP_CHECK(hipMemcpy(got.data(), d_dst, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    double max_abs = 0.0;
    int64_t bad = 0, first_bad = -1;
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const float want = reference_dequant(blob, row, col, n_tile_cols);
            const float have = got[static_cast<size_t>(row * ncols + col)];
            const double d = std::fabs(static_cast<double>(want) - static_cast<double>(have));
            if (d > max_abs) max_abs = d;
            if (!(d <= 1e-5)) {  // bit-exact in practice; small tolerance for safety
                if (bad == 0) first_bad = row * ncols + col;
                ++bad;
            }
        }
    }

    std::printf("hrx_q4nx_dequant_f32: rows=%lld cols=%lld tiles=%lld elements=%lld\n",
            (long long) nrows, (long long) ncols, (long long) n_tiles, (long long) total);
    std::printf("max_abs_diff=%.3e mismatches=%lld", max_abs, (long long) bad);
    if (first_bad >= 0) {
        std::printf(" first_bad=%lld (want=%.6f have=%.6f)",
                (long long) first_bad,
                reference_dequant(blob, first_bad / ncols, first_bad % ncols, n_tile_cols),
                got[static_cast<size_t>(first_bad)]);
    }
    std::printf("\n");

    HIP_CHECK(hipFree(const_cast<unsigned char *>(d_src)));
    HIP_CHECK(hipFree(d_dst));

    if (bad != 0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
