#pragma once

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
    const int32_t lda,
    const int32_t ldb
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
