#pragma once

#include "copy.cuh"
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// block_mma_prelogue_f16
//   一个 block 协同把 A[*, lda] 里的 [M, N] 子块搬到 B[M, ldb]
//   要求 N 是 VPT(=8) 的整数倍 (一次性 16B = uint4 向量化加载)
// ============================================================
template<int32_t TPB, int32_t M, int32_t N>
__device__
void block_mma_prelogue_f16(
    const __half* __restrict__ A, // [*, lda]
    __half* __restrict__ B,       // [M, ldb]
    const int32_t lda,            // 源矩阵列数
    const int32_t ldb             // 目的矩阵列数
) {
    const int32_t tid = threadIdx.x;
    constexpr int32_t VPT = 8;  // 8 个 fp16 = 16 字节

    static_assert(N % VPT == 0, "N must be a multiple of 8");
    for (int32_t i = tid * VPT; i < M * N; i += TPB * VPT) {
        const int32_t m = i / N;
        const int32_t n = i % N;
        // 16-byte 向量化拷贝: gmem -> reg -> smem
        uint4 v = *reinterpret_cast<const uint4*>(A + m * lda + n);
        *reinterpret_cast<uint4*>(B + m * ldb + n) = v;
    }
}

// ============================================================
// block_mma_prelogue_f16_async
//   cp.async 版本: GMEM → SMEM 绕过寄存器 (SM80+, 旧架构自动 fallback)
//   异步拷贝, 调用者需在读 smem 前 cp_async_wait_group<0>() + __syncthreads()
// ============================================================
template<int32_t TPB, int32_t M, int32_t N>
__device__
void block_mma_prelogue_f16_async(
    const __half* __restrict__ A,
    __half* __restrict__ B,
    const int32_t lda,
    const int32_t ldb
) {
    const int32_t tid = threadIdx.x;
    constexpr int32_t VPT = 8;

    static_assert(N % VPT == 0, "N must be a multiple of 8");
    for (int32_t i = tid * VPT; i < M * N; i += TPB * VPT) {
        const int32_t m = i / N;
        const int32_t n = i % N;
        cp_async<sizeof(__half) * VPT>(
            A + m * lda + n,
            B + m * ldb + n
        );
    }
    cp_async_commit();
}

// ============================================================
// block_mma_prelogue_f16_f32
//   fp16 -> fp32: 整 block 协同读 A[*, lda] fp16, 逐元素转 fp32 写到 B[M, N]
//   B 假设连续 (stride = N), 用于 FA1 从 HBM 加载 O 段做 fp32 在线累加
//   读: VPT=4 个 half = 8B (uint2)
//   写: VPT=4 个 float = 16B (uint4)
// ============================================================
template<int32_t TPB, int32_t M, int32_t N>
__device__
void block_mma_prelogue_f16_f32(
    const __half* __restrict__ A, // [*, lda]
    float* __restrict__ B,        // [M, N]  (连续)
    const int32_t lda
) {
    const int32_t tid = threadIdx.x;
    constexpr int32_t VPT = 4;
    __half local16[VPT];
    float  local32[VPT];

    static_assert(N % VPT == 0, "N must be a multiple of 4");
    for (int32_t i = tid * VPT; i < M * N; i += TPB * VPT) {
        const int32_t m = i / N;
        const int32_t n = i % N;

        *reinterpret_cast<uint2*>(local16) =
            *reinterpret_cast<const uint2*>(A + m * lda + n);

        #pragma unroll
        for (int32_t j = 0; j < VPT; j++) {
            local32[j] = __half2float(local16[j]);
        }

        *reinterpret_cast<uint4*>(B + i) =
            *reinterpret_cast<const uint4*>(local32);
    }
}
