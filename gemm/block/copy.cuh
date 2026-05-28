#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// SM80+ cp.async 异步拷贝原语（直接 gmem -> smem，跳过寄存器）
// ============================================================
template<int32_t Bytes>
__device__ __forceinline__
void cp_async(const void* gmem_ptr, void* smem_ptr) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async only supports 4/8/16 bytes");
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
#if __CUDA_ARCH__ >= 800
    // ca = cache all (L1+L2); .L2::128B = 提示 L2 按 128B 粒度预取，提高复用率
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
                 :: "r"(smem_int), "l"(gmem_ptr), "n"(Bytes));
#else
    // 旧架构 fallback：同步拷贝
    if constexpr (Bytes == 16) {
        *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
    } else if constexpr (Bytes == 8) {
        *reinterpret_cast<uint2*>(smem_ptr) = *reinterpret_cast<const uint2*>(gmem_ptr);
    } else {
        *reinterpret_cast<uint32_t*>(smem_ptr) = *reinterpret_cast<const uint32_t*>(gmem_ptr);
    }
#endif
}

__device__ __forceinline__
void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int32_t N>
__device__ __forceinline__
void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// ============================================================
// Async_BlockMMA_GmemToSmem_f16
//   block 协同发出 cp.async：A、B 从 global 异步搬到 shared
//   smemA: [Kstage, BlockM, PaddedWarpK]
//   smemB: [Kstage, BlockN, PaddedWarpK]
// ============================================================
template<
    int32_t TPB, int32_t VPT,
    int32_t BlockM, int32_t BlockN, int32_t WarpK,
    int32_t Kstage, int32_t Padding
>
struct Async_BlockMMA_GmemToSmem_f16 {
    static_assert(WarpK % VPT == 0);
    static_assert(Kstage <= 5);

    __device__ __forceinline__
    void load_async(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        __half *smemA,
        __half *smemB,
        const int32_t strideA,
        const int32_t strideB
    ) {
        constexpr int32_t PaddedWarpK = WarpK + Padding;
        const int32_t tid = threadIdx.x;

        for (int32_t idx = tid * VPT; idx < BlockM * WarpK; idx += TPB * VPT) {
            const int32_t m = idx / WarpK;
            const int32_t k = idx % WarpK;
            cp_async<sizeof(__half) * VPT>(
                &A[m * strideA + k],
                &smemA[m * PaddedWarpK + k]
            );
        }

        for (int32_t idx = tid * VPT; idx < BlockN * WarpK; idx += TPB * VPT) {
            const int32_t n = idx / WarpK;
            const int32_t k = idx % WarpK;
            cp_async<sizeof(__half) * VPT>(
                &B[n * strideB + k],
                &smemB[n * PaddedWarpK + k]
            );
        }

        cp_async_commit();
    }

    __device__ __forceinline__
    void wait_sync(int32_t stage) {
        switch (stage) {
        case 0: cp_async_wait_group<0>(); break;
        case 1: cp_async_wait_group<1>(); break;
        case 2: cp_async_wait_group<2>(); break;
        case 3: cp_async_wait_group<3>(); break;
        case 4: cp_async_wait_group<4>(); break;
        default: break;
        }
        __syncthreads();
    }

    __device__ __forceinline__
    void wait_all() {
        cp_async_wait_group<0>();
        __syncthreads();
    }
};
