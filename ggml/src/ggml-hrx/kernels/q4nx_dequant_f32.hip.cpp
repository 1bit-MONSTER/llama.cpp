#include <hip/hip_runtime.h>
#include <stdint.h>

// Q4NX int4 dequant, HRX0 port of ggml-hrx2's hrx2_q4nx_dequant_f32.
//
// 1bit-MONSTER Q4NX engine format: each 5120-byte tile covers
// [32 BF16 rows x 256 BF16 cols] and is laid out
//   [0..511]     256 BF16 scales, row-major: scales[row*8 + col/32]
//   [512..1023]  256 BF16 zero_points, same layout
//   [1024..5119] 4096 B packed INT4
//     lane = row/16, lane_row = row%16, byte_idx = lane_row/2, nib = row%2
//     byte = packed[lane*2048 + col*8 + byte_idx]
//     q = nib==0 ? byte&0xF : byte>>4 ; val = q<8 ? q : q-16 (signed int4)
//     out = val * scale + zp
//
// Weights are tile-major: 2-D [8192, n_tiles] or 3-D [8192, tpe, n_expert].
// The kernel takes ONE raw binding (the tile blob) and indexes the three
// sections at their in-tile offsets (0 / 512 / 1024), which is equivalent to
// HRX2 binding the same blob at offsets 0 / 512 / 1024 and lets a slice upload
// with a single stream copy.
//
// Non-finite or outlier (|x| > 100) scales/zero-points are zeroed, matching the
// engine dequant_q4nx.cpp: the last 6 bytes of the 512-byte zp section
// (zp indices 253..255) carry per-tile trailer data, not zero-points.

struct hrx_q4nx_dequant_f32_constants {
    long long nrows;
    long long ncols;
    long long n_tile_cols;
};

static __device__ __forceinline__ float hrx_q4nx_bf16_at(const unsigned char * p) {
    // BF16 is the top 16 bits of an f32; little-endian byte pair.
    const unsigned int lo = static_cast<unsigned int>(p[0]);
    const unsigned int hi = static_cast<unsigned int>(p[1]);
    const unsigned int bits = (lo | (hi << 8)) << 16;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ float hrx_q4nx_sanitize(float x) {
    // Matches the loom clamp: cmpf olt, |x|, 100.0 -- false for NaN, so NaN -> 0.
    return (fabsf(x) < 100.0f) ? x : 0.0f;
}

extern "C" __global__ void hrx_q4nx_dequant_f32(
        const unsigned char * src,
        float * dst,
        hrx_q4nx_dequant_f32_constants c) {
    const long long total = c.nrows * c.ncols;
    const long long linear =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * blockDim.x +
        static_cast<long long>(__builtin_amdgcn_workitem_id_x());
    if (linear >= total) {
        return;
    }

    const long long col = linear % c.ncols;
    const long long row = linear / c.ncols;

    const long long group     = col / 32;      // 32-col scale group within the tile
    const long long g_in_tile = group % 8;     // 8 groups per 256-col tile
    const long long tile_col  = col / 256;
    const long long tile_row  = row / 32;
    const long long tile_idx  = tile_row * c.n_tile_cols + tile_col;
    const long long tile_base = tile_idx * 5120;

    const long long in_tile_row = row % 32;
    const long long scale_idx   = in_tile_row * 8 + g_in_tile;
    const long long scale_off   = tile_base + scale_idx * 2;

    const float scale = hrx_q4nx_sanitize(hrx_q4nx_bf16_at(src + scale_off));
    const float zp    = hrx_q4nx_sanitize(hrx_q4nx_bf16_at(src + tile_base + 512 + scale_idx * 2));

    const long long lane_row   = in_tile_row % 16;
    const long long byte_idx   = lane_row / 2;
    const long long nib        = in_tile_row % 2;
    const long long in_tile_col = col % 256;
    const long long byte_off   = tile_base + 1024 + (in_tile_row / 16) * 2048 + in_tile_col * 8 + byte_idx;

    const unsigned int packed = static_cast<unsigned int>(src[byte_off]);
    const int q = nib == 0 ? static_cast<int>(packed & 0x0Fu) : static_cast<int>(packed >> 4);
    const int val = q < 8 ? q : q - 16;   // two's-complement signed int4

    dst[row * c.ncols + col] = static_cast<float>(val) * scale + zp;
}
