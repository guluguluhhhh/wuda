#include "warp/mma_ptx.cuh"
#include "block/tma.cuh"
#include "block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>

// ============================================================
// device kernel: TMA + mbarrier 版本
//   单线程发起 TMA bulk tensor copy, mbarrier 同步
//   warp 消费侧用 ptx::WarpHMMA_f16_sw64 (sm_120 没有 wgmma)
// ============================================================
template<
    int32_t TPB, int32_t BlockM, int32_t BlockN,
    int32_t WarpM, int32_t WarpN, int32_t WarpK,
    int32_t WarpCountM, int32_t WarpCountN,
    int32_t Kstage
>
__global__
void device_mma_tn_f16_tma(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    __half *C,
    const int32_t M, const int32_t N, const int32_t K
) {
    constexpr int32_t stage_A_elems = BlockM * WarpK;
    constexpr int32_t stage_B_elems = BlockN * WarpK;
    constexpr int32_t mainloop_elems = Kstage * (stage_A_elems + stage_B_elems);
    constexpr int32_t epilogue_elems = BlockM * BlockN;
    constexpr int32_t storage_elems  = mainloop_elems > epilogue_elems
                                       ? mainloop_elems : epilogue_elems;

    __shared__ alignas(16) __half    smem_storage[storage_elems];
    __shared__ alignas(8)  uint64_t  mbar[Kstage];

    __half* smem_A = smem_storage;
    __half* smem_B = smem_storage + Kstage * stage_A_elems;
    __half* smem_C = smem_storage;  // 别名

    const int32_t bx = blockIdx.z * gridDim.x + blockIdx.x;
    const int32_t by = blockIdx.y;

    if (threadIdx.x < Kstage) {
        mbarrier_init(&mbar[threadIdx.x], 1);
    }
    __syncthreads();

    const int32_t t_tile_max = K / WarpK;
    constexpr int32_t bytes_per_stage =
        (stage_A_elems + stage_B_elems) * sizeof(__half);

    // Prologue
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int32_t s = 0; s < Kstage; ++s) {
            if (s < t_tile_max) {
                mbarrier_arrive_expect_tx(&mbar[s], bytes_per_stage);
                cp_async_bulk_tensor_2d(
                    smem_A + s * stage_A_elems, &desc_A,
                    s * WarpK, by * BlockM, &mbar[s]
                );
                cp_async_bulk_tensor_2d(
                    smem_B + s * stage_B_elems, &desc_B,
                    s * WarpK, bx * BlockN, &mbar[s]
                );
            }
        }
    }

    constexpr int32_t warp_size = 32;
    const int32_t warp_id          = threadIdx.x / warp_size;
    const int32_t warp_tile_idx_m  = warp_id / WarpCountN;
    const int32_t warp_tile_idx_n  = warp_id % WarpCountN;

    ptx::WarpHMMA_f16_sw64<WarpM, WarpN, WarpK> wmma;
    wmma.zero();

    uint32_t phase[Kstage] = {0};

    for (int32_t k = 0; k < t_tile_max; ++k) {
        const int32_t stage = k % Kstage;

        mbarrier_wait(&mbar[stage], phase[stage]);
        phase[stage] ^= 1u;

        wmma.forward(
            smem_A + stage * stage_A_elems + warp_tile_idx_m * WarpM * WarpK,
            smem_B + stage * stage_B_elems + warp_tile_idx_n * WarpN * WarpK,
            WarpK, WarpK
        );

        __syncthreads();

        const int32_t next_k = k + Kstage;
        if (next_k < t_tile_max && threadIdx.x == 0) {
            mbarrier_arrive_expect_tx(&mbar[stage], bytes_per_stage);
            cp_async_bulk_tensor_2d(
                smem_A + stage * stage_A_elems, &desc_A,
                next_k * WarpK, by * BlockM, &mbar[stage]
            );
            cp_async_bulk_tensor_2d(
                smem_B + stage * stage_B_elems, &desc_B,
                next_k * WarpK, bx * BlockN, &mbar[stage]
            );
        }
    }

    wmma.stmatrix(
        smem_C + warp_tile_idx_m * WarpM * BlockN + warp_tile_idx_n * WarpN,
        BlockN
    );
    __syncthreads();

    block_mma_eplogue_f16<TPB, BlockM, BlockN>(
        smem_C,
        C + by * BlockM * N + bx * BlockN,
        BlockN, N
    );
}

// ============================================================
// host: 启动入口
// ============================================================
void gemm_tma_f16(
    const __half* A,
    const __half* B,
    __half* C,
    int32_t M, int32_t N, int32_t K
) {
    constexpr int32_t TPB = 128;

    constexpr int32_t BlockM = 128;
    constexpr int32_t BlockN = 128;

    constexpr int32_t WarpM = 64;
    constexpr int32_t WarpN = 64;
    constexpr int32_t WarpK = 32;
    constexpr int32_t Kstage = 2;

    constexpr int32_t WarpCountM = 2;
    constexpr int32_t WarpCountN = 2;

    static_assert(TPB / WarpCountM / WarpCountN == 32);
    static_assert(BlockM == WarpM * WarpCountM);
    static_assert(BlockN == WarpN * WarpCountN);

    if (M % BlockM != 0 || N % BlockN != 0) {
        fprintf(stderr,
                "gemm_tma_f16: M(%d) N(%d) must be multiples of BlockM(%d) BlockN(%d)\n",
                M, N, BlockM, BlockN);
        return;
    }
    if (K % WarpK != 0) {
        fprintf(stderr, "gemm_tma_f16: K(%d) must be multiple of WarpK(%d)\n", K, WarpK);
        return;
    }

    // 构造 TMA descriptor
    // A: [M, K] row-major, box=[BlockM, WarpK]
    // B: [N, K] row-major, box=[BlockN, WarpK]  (TN layout)
    // SW64: 4 atoms × 16B 一行, atom_phys = atom_logical XOR (row & 3)
    CUtensorMap desc_A = make_tma_2d_desc(A, M, K, BlockM, WarpK,
                                          CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap desc_B = make_tma_2d_desc(B, N, K, BlockN, WarpK,
                                          CU_TENSOR_MAP_SWIZZLE_64B);

    // grid swizzle
    const int32_t grid_m = M / BlockM;
    const int32_t grid_n = N / BlockN;
    const int32_t swizzle_stride = (N >= 2048) ? N / (2 * BlockN) : grid_n;
    const int32_t n_swizzle = (grid_n + swizzle_stride - 1) / swizzle_stride;
    const dim3 grid{(unsigned)((grid_n + n_swizzle - 1) / n_swizzle),
                    (unsigned)grid_m,
                    (unsigned)n_swizzle};

    device_mma_tn_f16_tma<
        TPB, BlockM, BlockN,
        WarpM, WarpN, WarpK,
        WarpCountM, WarpCountN, Kstage>
    <<<grid, TPB>>>(desc_A, desc_B, C, M, N, K);
}
