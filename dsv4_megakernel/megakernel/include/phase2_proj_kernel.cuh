#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>

// ============================================================
// DeepSeek-V4 Phase 2: Small Projections (wq_a + wkv)
// Input: bf16 [num_pos, D=7168]
// Weights: bf16 [D, N] transposed (K-major, N-contiguous)
// Output: bf16 [num_pos, N_QA=1536] + [num_pos, N_KV=512]
// Launch: block=1024, grid=2*SM, grid-stride loop
// ============================================================

// Model dimensions (DeepSeek-V4 Pro)
constexpr int D_PROJ = 7168;        // model hidden_size = input dim
constexpr int N_QA = 1536;          // wq_a output dim (q_lora_rank)
constexpr int N_KV = 512;           // wkv output dim (head_dim)
constexpr int ROPE_DIM = 64;        // RoPE applied to last 64 dims
constexpr int BLOCK_PROJ = 1024;    // threads per block
constexpr int NUM_WARPS_PROJ = BLOCK_PROJ / 32;  // 32 warps

// ---- Device utilities (shared with hc_fused) ----

__device__ __forceinline__ float warp_reduce_sum_p2(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum_p2(float val, float* smem, int tid) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    val = warp_reduce_sum_p2(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum_p2(val);
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__ float bf16_to_float_p2(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_p2(float val) {
    return __float2bfloat16(val);
}

inline int get_num_sms_p2() {
    int device, num_sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    return num_sms;
}
