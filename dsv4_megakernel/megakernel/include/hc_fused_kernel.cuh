#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <float.h>

// ============================================================
// DeepSeek-V4 HC Fused Kernel - Header
// Input: bf16, Weight: bf16, Output: bf16 (matching origin)
// Launch: block=1024, grid=2*SM
// ============================================================

constexpr int HC_DEFAULT = 4;
constexpr int D_DEFAULT = 1792;
constexpr int N_OUT_DEFAULT = 24;
constexpr int BLOCK_SIZE_DEFAULT = 1024;
constexpr int SINKHORN_DEFAULT = 20;
constexpr int HC_D = HC_DEFAULT * D_DEFAULT;  // 7168

// ---- Device utilities ----

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem, int tid) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float val) {
    return __float2bfloat16(val);
}

inline int get_num_sms() {
    int device, num_sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    return num_sms;
}
