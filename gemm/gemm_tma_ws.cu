#include "warp/mma_ptx.cuh"
#include "block/tma.cuh"
#include "block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>

// ============================================================
// Warp-specialized TMA GEMM
//   - 1 producer warp: 发 TMA + 等 empty barrier
//   - N consumer warps: 跑 mma + 等 full barrier
//   - 2 套 mbarrier (full / empty) 实现 producer-consumer 解耦
//   - smem 用 SW64 swizzle，warp mma 用 ptx::WarpHMMA_f16_sw64
// ============================================================
template<
    int32_t TPB, int32_t BlockM, int32_t BlockN,
    int32_t WarpM, int32_t WarpN, int32_t WarpK,
    int32_t WarpCountM, int32_t WarpCountN,
    int32_t Kstage
>
__global__
void device_mma_tn_f16_tma_ws(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    __half *C,
    const int32_t M, const int32_t N, const int32_t K
) {
    constexpr int32_t stage_A_elems  = BlockM * WarpK;
    constexpr int32_t stage_B_elems  = BlockN * WarpK;
    constexpr int32_t mainloop_elems = Kstage * (stage_A_elems + stage_B_elems);
    constexpr int32_t epilogue_elems = BlockM * BlockN;
    constexpr int32_t storage_elems  = mainloop_elems > epilogue_elems
                                       ? mainloop_elems : epilogue_elems;

    __shared__ alignas(16) __half    smem_storage[storage_elems];
    __shared__ alignas(8)  uint64_t  mbar_full [Kstage];  // producer arrive, consumer wait
    __shared__ alignas(8)  uint64_t  mbar_empty[Kstage];  // consumer arrive, producer wait

    __half* smem_A = smem_storage;
    __half* smem_B = smem_storage + Kstage * stage_A_elems;
    __half* smem_C = smem_storage;  // 别名

    const int32_t bx = blockIdx.z * gridDim.x + blockIdx.x;
    const int32_t by = blockIdx.y;

    constexpr int32_t warp_size          = 32;
    constexpr int32_t num_consumer_warps = WarpCountM * WarpCountN;
    constexpr int32_t producer_warp_id   = num_consumer_warps;  // 最后一个 warp

    const int32_t warp_id = threadIdx.x / warp_size;
    const int32_t lane_id = threadIdx.x & 31;
    const bool    is_producer = (warp_id == producer_warp_id);

    if (threadIdx.x < Kstage) {
        mbarrier_init(&mbar_full [threadIdx.x], 1);                    // producer 1 次 arrive_expect_tx
        mbarrier_init(&mbar_empty[threadIdx.x], num_consumer_warps);   // 每 consumer warp 1 次 arrive
    }
    __syncthreads();

    const int32_t t_tile_max = K / WarpK;
    constexpr int32_t bytes_per_stage =
        (stage_A_elems + stage_B_elems) * sizeof(__half);

    // consumer 用 - 放外面让 stmatrix 路径能用
    const int32_t warp_tile_idx_m = warp_id / WarpCountN;
    const int32_t warp_tile_idx_n = warp_id % WarpCountN;

    ptx::WarpHMMA_f16_sw64<WarpM, WarpN, WarpK> wmma;
    if (!is_producer) wmma.zero();

    if (is_producer) {
        // ===== Producer =====
        uint32_t phase_empty[Kstage] = {0};

        for (int32_t k = 0; k < t_tile_max; ++k) {
            const int32_t stage = k % Kstage;

            // 重用 stage 前等 consumer 用完 (Kstage 拍之后才会发生)
            if (k >= Kstage) {
                mbarrier_wait(&mbar_empty[stage], phase_empty[stage]);
                phase_empty[stage] ^= 1u;
            }

            if (lane_id == 0) {
                mbarrier_arrive_expect_tx(&mbar_full[stage], bytes_per_stage);
                cp_async_bulk_tensor_2d(
                    smem_A + stage * stage_A_elems, &desc_A,
                    k * WarpK, by * BlockM, &mbar_full[stage]
                );
                cp_async_bulk_tensor_2d(
                    smem_B + stage * stage_B_elems, &desc_B,
                    k * WarpK, bx * BlockN, &mbar_full[stage]
                );
            }
        }
    } else {
        // ===== Consumer =====
        uint32_t phase_full[Kstage] = {0};

        for (int32_t k = 0; k < t_tile_max; ++k) {
            const int32_t stage = k % Kstage;

            mbarrier_wait(&mbar_full[stage], phase_full[stage]);
            phase_full[stage] ^= 1u;

            wmma.forward(
                smem_A + stage * stage_A_elems + warp_tile_idx_m * WarpM * WarpK,
                smem_B + stage * stage_B_elems + warp_tile_idx_n * WarpN * WarpK,
                WarpK, WarpK
            );

            if (lane_id == 0) {
                mbarrier_arrive(&mbar_empty[stage]);
            }
        }
    }

    // 所有 warp (含 producer) 必须收口
    __syncthreads();

    // Epilogue: 只 consumer 写 smem_C
    if (!is_producer) {
        wmma.stmatrix(
            smem_C + warp_tile_idx_m * WarpM * BlockN + warp_tile_idx_n * WarpN,
            BlockN
        );
    }
    __syncthreads();

    // 全员参与 gmem 写
    block_mma_eplogue_f16<TPB, BlockM, BlockN>(
        smem_C,
        C + by * BlockM * N + bx * BlockN,
        BlockN, N
    );
}

// ============================================================
// host
// ============================================================
void gemm_tma_ws_f16(
    const __half* A,
    const __half* B,
    __half* C,
    int32_t M, int32_t N, int32_t K
) {
    constexpr int32_t WarpCountM = 2;
    constexpr int32_t WarpCountN = 2;
    constexpr int32_t num_consumer_warps = WarpCountM * WarpCountN;       // 4
    constexpr int32_t TPB = (num_consumer_warps + 1) * 32;                // 5*32 = 160

    constexpr int32_t BlockM = 128;
    constexpr int32_t BlockN = 128;

    constexpr int32_t WarpM = 64;
    constexpr int32_t WarpN = 64;
    constexpr int32_t WarpK = 32;
    constexpr int32_t Kstage = 2;

    static_assert(BlockM == WarpM * WarpCountM);
    static_assert(BlockN == WarpN * WarpCountN);

    if (M % BlockM != 0 || N % BlockN != 0) {
        fprintf(stderr,
                "gemm_tma_ws_f16: M(%d) N(%d) must be multiples of BlockM(%d) BlockN(%d)\n",
                M, N, BlockM, BlockN);
        return;
    }
    if (K % WarpK != 0) {
        fprintf(stderr, "gemm_tma_ws_f16: K(%d) must be multiple of WarpK(%d)\n", K, WarpK);
        return;
    }

    CUtensorMap desc_A = make_tma_2d_desc(A, M, K, BlockM, WarpK,
                                          CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap desc_B = make_tma_2d_desc(B, N, K, BlockN, WarpK,
                                          CU_TENSOR_MAP_SWIZZLE_64B);

    const int32_t grid_m = M / BlockM;
    const int32_t grid_n = N / BlockN;
    const int32_t swizzle_stride = (N >= 2048) ? N / (2 * BlockN) : grid_n;
    const int32_t n_swizzle = (grid_n + swizzle_stride - 1) / swizzle_stride;
    const dim3 grid{(unsigned)((grid_n + n_swizzle - 1) / n_swizzle),
                    (unsigned)grid_m,
                    (unsigned)n_swizzle};

    device_mma_tn_f16_tma_ws<
        TPB, BlockM, BlockN,
        WarpM, WarpN, WarpK,
        WarpCountM, WarpCountN, Kstage>
    <<<grid, TPB>>>(desc_A, desc_B, C, M, N, K);
}
