#include <hip/hip_runtime.h>
#include <stdint.h>

// Q4NX fused dequant + f32 matmul, HRX0 port of ggml-hrx2's
// hrx2_mul_mat_q4nx_fused_f32.
//
// src0 is a GGML_TYPE_Q4NX weight stored tile-major as [8192, n_tiles]
// (each 8192-element row = one 5120-byte tile = [32 rows x 256 cols]);
// tiles are in (tile_row, tile_col) order with tile_col = t % n_tc and
// n_tc = k/256. src1 is the F32 activation [k, cols]. dst is F32 [rows, cols]
// with rows = (n_tiles / n_tc) * 32.
//
// One workgroup per (row, col); the workgroup reduces over k, dequantizing the
// weight tile element inline (same decode as q4nx_dequant_f32.hip.cpp).

struct hrx_mul_mat_q4nx_fused_f32_constants {
    long long k;
    long long rows;
    long long cols;
    long long n_tile_cols;
};

static __device__ __forceinline__ float hrx_q4nx_bf16_at(const unsigned char * p) {
    const unsigned int lo = static_cast<unsigned int>(p[0]);
    const unsigned int hi = static_cast<unsigned int>(p[1]);
    return __uint_as_float((lo | (hi << 8)) << 16);
}

static __device__ __forceinline__ float hrx_q4nx_sanitize(float x) {
    return (fabsf(x) < 100.0f) ? x : 0.0f;
}

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_q4nx_reduce_add(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return sum;
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();
    sum = lane < (WG_SIZE / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_q4nx_fused_f32_impl(
        const unsigned char * src0, const float * src1, float * dst,
        hrx_mul_mat_q4nx_fused_f32_constants c) {
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    const long long col = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows || col >= c.cols) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];

    // Row geometry is fixed for the whole workgroup.
    const long long tile_row     = row / 32;
    const long long in_tile_row  = row % 32;
    const long long tr_x_ntc     = tile_row * c.n_tile_cols;
    const long long in_tile_row8 = in_tile_row * 8;
    const long long lane_row     = in_tile_row % 16;
    const long long byte_idx     = lane_row / 2;
    const long long nib          = in_tile_row % 2;
    const long long lane_base    = (in_tile_row / 16) * 2048;

    const float * src1_col = src1 + col * c.k;
    float sum = 0.0f;

    for (long long i = static_cast<long long>(tid); i < c.k; i += WG_SIZE) {
        const long long tile_col    = i / 256;
        const long long in_tile_col = i % 256;
        const long long tile_base   = (tr_x_ntc + tile_col) * 5120;

        const long long scale_idx = in_tile_row8 + (in_tile_col / 32);
        const float scale = hrx_q4nx_sanitize(hrx_q4nx_bf16_at(src0 + tile_base + scale_idx * 2));
        const float zp    = hrx_q4nx_sanitize(hrx_q4nx_bf16_at(src0 + tile_base + 512 + scale_idx * 2));

        const long long byte_off = tile_base + 1024 + lane_base + in_tile_col * 8 + byte_idx;
        const unsigned int packed = static_cast<unsigned int>(src0[byte_off]);
        const int q = nib == 0 ? static_cast<int>(packed & 0x0Fu) : static_cast<int>(packed >> 4);
        const int val = q < 8 ? q : q - 16;

        sum += (static_cast<float>(val) * scale + zp) * src1_col[i];
    }

    sum = hrx_q4nx_reduce_add<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[row + col * c.rows] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_q4nx_fused_f32(
        const unsigned char * src0,
        const float * src1,
        float * dst,
        hrx_mul_mat_q4nx_fused_f32_constants c) {
    hrx_mul_mat_q4nx_fused_f32_impl<256>(src0, src1, dst, c);
}
