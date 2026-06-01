#include "warp/mma.cuh"
#include "block/copy.cuh"
#include "block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

template<
    int32_t TPB, int32_t BlockM, int32_t BlockN,
    int32_t WarpM, int32_t WarpN, int32_t WarpK,
    int32_t WarpCountM, int32_t WarpCountN,
    int32_t Kstage
>
__device__
void bmma_tn_f16_db(
    const __half *A,
    const __half *B,
    __half *C,
    const int32_t strideA,
    const int32_t strideB,
    const int32_t strideC
) {
    constexpr int32_t PAD = 8;
    constexpr int32_t S = WarpK + PAD;
    __shared__ __half smem_A[Kstage * BlockM * S];
    __shared__ __half smem_B[Kstage * BlockN * S];
    __shared__ __half smem_C[BlockM * BlockN];

    Async_BlockMMA_GmemToSmem_f16<TPB, 8, BlockM, BlockN, WarpK, Kstage, PAD> g2s;

    constexpr int32_t warp_size = 32;
    const int32_t warp_id = threadIdx.x / warp_size;
    const int32_t warp_tile_idx_m = warp_id / WarpCountN;
    const int32_t warp_tile_idx_n = warp_id % WarpCountN;
    WarpHMMA_f16<WarpM, WarpN, WarpK> wmma;
    wmma.zero();

    const int32_t K = strideA;
    const int32_t t_tile_max = K / WarpK;

    int32_t load_kidx = 0;
    int32_t mma_kidx = load_kidx - Kstage + 1;

    while (true) {
        if (load_kidx < t_tile_max) {
            g2s.load_async(
                A + load_kidx * WarpK,
                B + load_kidx * WarpK,
                smem_A + (load_kidx % Kstage) * BlockM * S,
                smem_B + (load_kidx % Kstage) * BlockN * S,
                strideA, strideB
            );
        }

        if (mma_kidx >= 0) {
            const int32_t remain = t_tile_max - mma_kidx - 1;
            g2s.wait_sync(remain > (Kstage - 1) ? Kstage - 1 : remain);
            wmma.forward(
                smem_A + warp_tile_idx_m * WarpM * S + (mma_kidx % Kstage) * BlockM * S,
                smem_B + warp_tile_idx_n * WarpN * S + (mma_kidx % Kstage) * BlockN * S,
                S, S
            );
        }

        load_kidx++;
        if (++mma_kidx == t_tile_max) break;
    }

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
    int32_t WarpCountM, int32_t WarpCountN,
    int32_t Kstage
>
__global__
void device_mma_tn_f16_db(
    const __half *A,
    const __half *B,
    __half *C,
    const int32_t M,
    const int32_t N,
    const int32_t K
) {
    // block swizzle: 让相邻 block 访问相邻 B 列，提高 L2 命中率
    const int bx = blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;

    bmma_tn_f16_db<
        TPB, BlockM, BlockN,
        WarpM, WarpN, WarpK,
        WarpCountM, WarpCountN, Kstage
    >(
        A + by * BlockM * K,
        B + bx * BlockN * K,
        C + by * BlockM * N + bx * BlockN,
        K, K, N
    );
}

void gemm3_f16(
    const __half* A,
    const __half* B,
    __half* C,
    int32_t M, int32_t N, int32_t K
) {
    constexpr int32_t TPB = 128;

    constexpr int32_t BlockM = 128;
    constexpr int32_t BlockN = 64;

    constexpr int32_t WarpM = 64;
    constexpr int32_t WarpN = 32;
    constexpr int32_t WarpK = 32;
    constexpr int32_t Kstage = 2;

    constexpr int32_t WarpCountM = 2;
    constexpr int32_t WarpCountN = 2;

    static_assert(TPB / WarpCountM / WarpCountN == 32);
    static_assert(BlockM == WarpM * WarpCountM);
    static_assert(BlockN == WarpN * WarpCountN);

    if (M % BlockM != 0 || N % BlockN != 0) {
        fprintf(stderr, "gemm3_f16: M(%d) and N(%d) must be multiples of BlockM(%d), BlockN(%d)\n",
                M, N, BlockM, BlockN);
        return;
    }

    const int grid_m = M / BlockM;
    const int grid_n = N / BlockN;
    // swizzle: N 方向按 swizzle_stride 分段，每段内的 block 访问相邻 B 列
    const int swizzle_stride = (N >= 2048) ? N / (2 * BlockN) : grid_n;
    const int n_swizzle = (grid_n + swizzle_stride - 1) / swizzle_stride;
    const dim3 grid{(unsigned)((grid_n + n_swizzle - 1) / n_swizzle),
                    (unsigned)grid_m,
                    (unsigned)n_swizzle};

    device_mma_tn_f16_db<
        TPB, BlockM, BlockN,
        WarpM, WarpN, WarpK,
        WarpCountM, WarpCountN, Kstage>
    <<<grid, TPB>>>(A, B, C, M, N, K);
}
