#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>

// ============================================================
// DeepSeek-V4 HC Fused Kernel - Tensor Core version Header
// Uses SM80 m16n8k16 MMA for Phase 2 (GEMV→GEMM with batch)
// Input: bf16 [BATCH, HC*DIM] = [16, 28672]
// Weight: bf16 [N_PAD, HC*DIM] = [32, 28672] (padded from 24)
// Output: bf16 [BATCH, DIM] = [16, 7168]
// ============================================================

// Model constants (same as CC version)
constexpr int HC_TC = 4;
constexpr int DIM_TC = 7168;
constexpr int N_OUT_TC = 24;          // actual output width
constexpr int N_PAD_TC = 32;          // padded to fit TC tile (8-aligned)
constexpr int HC_DIM_TC = HC_TC * DIM_TC;  // 28672 (K dimension)
constexpr int SINKHORN_TC = 20;

// Batched TC kernel configuration
constexpr int BATCH_DEFAULT = 16;     // positions per batch (M dim for MMA)
constexpr int BK_TC = 256;            // K tile size (fits 2 stages in 48KB smem)
constexpr int BLOCK_SIZE_TC = 1024;   // 32 warps: align with CC version
constexpr int NUM_WARPS_TC = BLOCK_SIZE_TC / 32;  // 32
constexpr int MMA_THREADS = 128;      // first 4 warps do MMA, others help load

// MMA atom: SM80_16x8x16_F32BF16BF16F32_TN
// Covers: M=16, N=8, K=16 per atom
// To cover N=32: repeat 4× along N → full 16×32 output per K=16 step
// K iterations per smem tile: BK/16 = 32

// Shared memory sizing
// Phase 2 GEMM (double buffered, Kstage=2, A padded):
//   smem_A per stage: [BATCH+PAD, BK] bf16 = 24×256×2 = 12288B = 12KB
//   smem_B per stage: [N_PAD, BK] bf16 = 32×256×2 = 16384B = 16KB
//   Per stage: 28KB, 2 stages: 56KB (needs cudaFuncSetAttribute)
// Phase 5 Collapse: [DIM/2] floats = 3584×4 = 14KB (reuse)
constexpr int PAD_A_TC = 8;  // smem A padding for bank conflict elimination
constexpr int SMEM_A_SIZE = (BATCH_DEFAULT + PAD_A_TC) * BK_TC;  // 6144 bf16 elements
constexpr int SMEM_B_SIZE = N_PAD_TC * BK_TC;                    // 8192 bf16 elements
constexpr int SMEM_STAGE_BYTES = (SMEM_A_SIZE + SMEM_B_SIZE) * 2;  // 1 stage = 28KB
constexpr int SMEM_GEMM_BYTES = SMEM_STAGE_BYTES * 2;              // 2 stages = 56KB

// ---- Device utilities (shared with CC version) ----

__device__ __forceinline__ float warp_reduce_sum_tc(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum_tc(float val, float* smem, int tid) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    val = warp_reduce_sum_tc(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum_tc(val);
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__ float fast_sigmoid_tc(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline int get_num_sms_tc() {
    int device, num_sms;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    return num_sms;
}
