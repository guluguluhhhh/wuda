#include "warp/mma.cuh"
#include "block/prelogue.cuh"
#include "block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

// ============================================================
// bmma_tn_f16
//   block 级 GEMM: 一个 block 算出 C 上 [BlockM, BlockN] 的输出 tile
//   A: [*, strideA] row-major
//   B: [*, strideB] row-major (即 B^T, TN layout)
//   C: [*, strideC] row-major
// ============================================================
template<
    int32_t TPB, int32_t BlockM, int32_t BlockN,
    int32_t WarpM, int32_t WarpN, int32_t WarpK,
    int32_t WarpCountM, int32_t WarpCountN
>
__device__
void bmma_tn_f16(
    const __half *A,
    const __half *B,
    __half *C,
    const int32_t strideA,
    const int32_t strideB,
    const int32_t strideC
) {
    __shared__ __half smem_A[BlockM * WarpK];
    __shared__ __half smem_B[BlockN * WarpK];
    __shared__ __half smem_C[BlockM * BlockN];

    constexpr int32_t warp_size = 32;
    const int32_t tid = threadIdx.x;
    const int32_t warp_id = tid / warp_size;
    const int32_t warp_tile_idx_m = warp_id / WarpCountN;
    const int32_t warp_tile_idx_n = warp_id % WarpCountN;
    WarpHMMA_f16<WarpM, WarpN, WarpK> wmma;
    wmma.zero();

    const int32_t K = strideA;
    for (int32_t k = 0; k < K; k += WarpK) {
        // 整 block 协同把 A, B 的 K 切片搬到 smem
        block_mma_prelogue_f16<TPB, BlockM, WarpK>(A + k, smem_A, K, WarpK);
        block_mma_prelogue_f16<TPB, BlockN, WarpK>(B + k, smem_B, K, WarpK);
        __syncthreads();

        wmma.forward(
            smem_A + warp_tile_idx_m * WarpM * WarpK,
            smem_B + warp_tile_idx_n * WarpN * WarpK,
            WarpK,
            WarpK
        );

        __syncthreads();
    }
    __syncthreads();

    // 累加器 (fp32) → fp16 写到 smem_C，再整 block 合并写回 global
    wmma.stmatrix(
        smem_C + warp_tile_idx_m * WarpM * BlockN + warp_tile_idx_n * WarpN,
        BlockN
    );
    __syncthreads();

    block_mma_eplogue_f16<TPB, BlockM, BlockN>(smem_C, C, BlockN, strideC);
}

template<
    int32_t TPB, int32_t BlockM, int32_t BlockN,
    int32_t WarpM, int32_t WarpN, int32_t WarpK,
    int32_t WarpCountM, int32_t WarpCountN
>
__global__
void device_mma_tn_f16(
    const __half *A,  // [M, K]
    const __half *B,  // [N, K]
    __half *C,        // [M, N]
    const int32_t M,
    const int32_t N,
    const int32_t K
) {
    bmma_tn_f16<
        TPB,
        BlockM, BlockN,
        WarpM, WarpN, WarpK,
        WarpCountM, WarpCountN
    >(
        A + blockIdx.x * BlockM * K,
        B + blockIdx.y * BlockN * K,
        C + blockIdx.x * BlockM * N + blockIdx.y * BlockN,
        K, K, N
    );
}

// ============================================================
// Host launcher
//   A: [M, K] row-major (device ptr)
//   B: [N, K] row-major (device ptr, 即 B^T)
//   C: [M, N] row-major (device ptr)
//   要求 M、N 是 BlockM/BlockN 的整数倍
// ============================================================
void gemm2_f16(
    const __half* A,
    const __half* B,
    __half* C,
    int32_t M, int32_t N, int32_t K
) {
    constexpr int32_t TPB = 256;

    constexpr int32_t BlockM = 64;
    constexpr int32_t BlockN = 64;

    constexpr int32_t WarpM = 16;
    constexpr int32_t WarpN = 32;
    constexpr int32_t WarpK = 32;

    constexpr int32_t WarpCountM = 4;
    constexpr int32_t WarpCountN = 2;

    static_assert(TPB / WarpCountM / WarpCountN == 32);
    static_assert(BlockM == WarpM * WarpCountM);
    static_assert(BlockN == WarpN * WarpCountN);

    if (M % BlockM != 0 || N % BlockN != 0) {
        fprintf(stderr, "gemm2_f16: M(%d) and N(%d) must be multiples of BlockM(%d), BlockN(%d)\n",
                M, N, BlockM, BlockN);
        return;
    }

    const dim3 grid{(unsigned)(M / BlockM), (unsigned)(N / BlockN), 1};

    device_mma_tn_f16<
        TPB, BlockM, BlockN,
        WarpM, WarpN, WarpK,
        WarpCountM, WarpCountN>
    <<<grid, TPB>>>(A, B, C, M, N, K);
}
