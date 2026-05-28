#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// block_mma_eplogue_f32
//   fp32 -> fp32: 整 block 协同把 A[M, N] 搬到 B[*, ldb]
//   VPT=4 个 float = 16 字节 (uint4)
// ============================================================
template<int32_t TPB, int32_t M, int32_t N>
__device__
void block_mma_eplogue_f32(
    const float* __restrict__ A, // [M, N]
    float* __restrict__ B,       // [*, ldb]
    const int32_t strideA,
    const int32_t strideB
) {
    const int32_t tid = threadIdx.x;
    constexpr int32_t VPT = 4;

    static_assert(N % VPT == 0, "N must be a multiple of 4");
    for (int32_t i = tid * VPT; i < M * N; i += TPB * VPT) {
        const int32_t m = i / N;
        const int32_t n = i % N;
        uint4 v = *reinterpret_cast<const uint4*>(A + m * strideA + n);
        *reinterpret_cast<uint4*>(B + m * strideB + n) = v;
    }
}

// ============================================================
// block_mma_eplogue_f32_f16
//   fp32 -> fp16: 整 block 协同读 A[M, N] fp32，逐元素转 fp16 写到 B[*, ldb]
//   读: VPT=4 个 float = 16B (uint4)
//   写: VPT=4 个 half  = 8B  (uint2)
// ============================================================
template<int32_t TPB, int32_t M, int32_t N>
__device__
void block_mma_eplogue_f32_f16(
    const float* __restrict__ A, // [M, N]
    __half* __restrict__ B,      // [*, ldb]
    const int32_t ldb
) {
    const int32_t tid = threadIdx.x;
    constexpr int32_t VPT = 4;
    float  local32[VPT];
    __half local16[VPT];

    static_assert(N % VPT == 0, "N must be a multiple of 4");
    for (int32_t i = tid * VPT; i < M * N; i += TPB * VPT) {
        const int32_t m = i / N;
        const int32_t n = i % N;

        *reinterpret_cast<uint4*>(local32) =
            *reinterpret_cast<const uint4*>(A + i);

        #pragma unroll
        for (int32_t j = 0; j < VPT; j++) {
            local16[j] = __float2half(local32[j]);
        }

        *reinterpret_cast<uint2*>(B + m * ldb + n) =
            *reinterpret_cast<const uint2*>(local16);
    }
}

// ============================================================
// block_mma_eplogue_f16
//   fp16 -> fp16: 整 block 协同把 A[M, N] 搬到 B[*, ldb]
//   VPT=8 个 half = 16B (uint4)
// ============================================================
template<int32_t TPB, int32_t M, int32_t N>
__device__
void block_mma_eplogue_f16(
    const __half* __restrict__ A, // [M, N]
    __half* __restrict__ B,       // [*, ldb]
    const int32_t strideA,
    const int32_t strideB
) {
    const int32_t tid = threadIdx.x;
    constexpr int32_t VPT = 8;

    static_assert(N % VPT == 0, "N must be a multiple of 8");
    for (int32_t i = tid * VPT; i < M * N; i += TPB * VPT) {
        const int32_t m = i / N;
        const int32_t n = i % N;
        uint4 v = *reinterpret_cast<const uint4*>(A + m * strideA + n);
        *reinterpret_cast<uint4*>(B + m * strideB + n) = v;
    }
}
