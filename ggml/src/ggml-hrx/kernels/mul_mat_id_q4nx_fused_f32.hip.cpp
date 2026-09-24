#include <hip/hip_runtime.h>
#include <stdint.h>

// Q4NX MoE fused dequant + f32 matmul, HRX0 port of the ggml-hrx2
// MUL_MAT_ID_Q4NX route (which reuses the fused dense kernel per expert).
//
// src0 is a GGML_TYPE_Q4NX expert stack stored tile-major 3-D
// [8192, tiles_per_expert, n_expert]; one expert's tiles are contiguous, so
// expert e starts at byte offset e * tpe * 5120. src1 is the F32 activation
// [k, ntokens]. ids is I32 [nselected, ntokens]. dst is F32
// [rows, nselected, ntokens] with rows = (tpe / n_tc) * 32, n_tc = k/256.
//
// One workgroup per (row, slot, token); the workgroup reads the selected
// expert id and reduces over k with inline Q4NX dequant.

struct hrx_mul_mat_id_q4nx_fused_f32_constants {
    long long k;
    long long rows;
    long long tpe;
    long long n_tile_cols;
    long long nselected;
    long long ntokens;
    long long n_experts;
    long long ids_nb0;
    long long ids_nb1;
    long long src1_nb1;
    long long src1_nb2;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ float hrx_q4nx_id_bf16_at(const unsigned char * p) {
    const unsigned int lo = static_cast<unsigned int>(p[0]);
    const unsigned int hi = static_cast<unsigned int>(p[1]);
    return __uint_as_float((lo | (hi << 8)) << 16);
}

static __device__ __forceinline__ float hrx_q4nx_id_sanitize(float x) {
    return (fabsf(x) < 100.0f) ? x : 0.0f;
}

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_q4nx_id_reduce_add(float sum, float * shared) {
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
static __device__ __forceinline__ void hrx_mul_mat_id_q4nx_fused_f32_impl(
        const unsigned char * src0, const float * src1, const int32_t * ids, float * dst,
        hrx_mul_mat_id_q4nx_fused_f32_constants c) {
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    const long long sel = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    const long long tok = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows || sel >= c.nselected || tok >= c.ntokens) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];
    __shared__ long long expert_sh;

    if (tid == 0) {
        const long long ids_index = (c.ids_nb0 * sel + c.ids_nb1 * tok) / static_cast<long long>(sizeof(int32_t));
        expert_sh = static_cast<long long>(ids[ids_index]);
    }
    __syncthreads();

    const long long expert = expert_sh;
    const long long dst_index =
        row + sel * (c.dst_nb1 / static_cast<long long>(sizeof(float))) +
              tok * (c.dst_nb2 / static_cast<long long>(sizeof(float)));
    if (expert < 0 || expert >= c.n_experts) {
        if (tid == 0) {
            dst[dst_index] = 0.0f;
        }
        return;
    }

    const long long tile_row     = row / 32;
    const long long in_tile_row  = row % 32;
    const long long tr_x_ntc     = tile_row * c.n_tile_cols;
    const long long in_tile_row8 = in_tile_row * 8;
    const long long lane_row     = in_tile_row % 16;
    const long long byte_idx     = lane_row / 2;
    const long long nib          = in_tile_row % 2;
    const long long lane_base    = (in_tile_row / 16) * 2048;
    const long long expert_base  = expert * c.tpe * 5120;

    // src1 is [k, n_expert_used, ntokens] (ggml ne order). src1_nb1 is 0 when
    // n_expert_used == 1 (the activation is shared by every selected expert,
    // matching ggml_mul_mat_id's broadcast rule).
    const float * src1_tok = src1 +
        (sel * c.src1_nb1) / static_cast<long long>(sizeof(float)) +
        (tok * c.src1_nb2) / static_cast<long long>(sizeof(float));
    float sum = 0.0f;

    for (long long i = static_cast<long long>(tid); i < c.k; i += WG_SIZE) {
        const long long tile_col    = i / 256;
        const long long in_tile_col = i % 256;
        const long long tile_base   = expert_base + (tr_x_ntc + tile_col) * 5120;

        const long long scale_idx = in_tile_row8 + (in_tile_col / 32);
        const float scale = hrx_q4nx_id_sanitize(hrx_q4nx_id_bf16_at(src0 + tile_base + scale_idx * 2));
        const float zp    = hrx_q4nx_id_sanitize(hrx_q4nx_id_bf16_at(src0 + tile_base + 512 + scale_idx * 2));

        const long long byte_off = tile_base + 1024 + lane_base + in_tile_col * 8 + byte_idx;
        const unsigned int packed = static_cast<unsigned int>(src0[byte_off]);
        const int q = nib == 0 ? static_cast<int>(packed & 0x0Fu) : static_cast<int>(packed >> 4);
        const int val = q < 8 ? q : q - 16;

        sum += (static_cast<float>(val) * scale + zp) * src1_tok[i];
    }

    sum = hrx_q4nx_id_reduce_add<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[dst_index] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4nx_fused_f32(
        const unsigned char * src0,
        const float * src1,
        const int32_t * ids,
        float * dst,
        hrx_mul_mat_id_q4nx_fused_f32_constants c) {
    hrx_mul_mat_id_q4nx_fused_f32_impl<256>(src0, src1, ids, dst, c);
}
