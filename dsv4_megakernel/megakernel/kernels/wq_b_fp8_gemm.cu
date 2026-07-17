// ============================================================
// wq_b_gemm.cu
// tcgen05 FP8 (e4m3) block-scale GEMM — Kernel + Host + PyTorch Binding
// Aligned with DeepGEMM sm100_fp8_gemm_1d1d / megakernel w1_merged_fp8_gemm.
//
// x_fp8[M,1536] @ w_fp8[65536,1536]^T -> y[M,65536] (FP32)
// M=32~256 (32-aligned), K=1536, N=65536, e4m3 inputs + per-32K UE8M0 scale.
// M<=128: swap-AB BM128xBN128. M>=160: non-swap BM128xBN224, cluster along M.
// UMMA_K=32, gran_k=32.
// 2SM MMA (cta_group::2), block_scale, Persistent, Warp-Specialized.
// SF pipeline: TMA -> smem -> warp2 transpose -> UTCCP -> TMEM (block_scale MMA).
// MMA (UTCCP + block_scale) delegated to cluster_mma_fp8.cuh.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdlib>   // getenv / atoi (WQ_B_CLUSTERS experiment knob)

#include "wq_b_fp8_gemm.cuh"
#include "cluster_mma_fp8.cuh"   // FP8 block-scale MMA engine (leader MMA warp)

using namespace wq_b;
using Barrier = mma_desc::Barrier;

// ======================== Shared Memory Layout ========================
template <int M_TPL>
struct SharedStorage {
    using D = SwapDims<M_TPL>;
    static constexpr int NS   = D::NUM_STAGES;
    static constexpr int SA   = D::SMEM_A_PER_STAGE;
    static constexpr int SSFA = D::SMEM_SFA_PER_STAGE;

    alignas(1024) uint8_t smem_cd[SMEM_CD_TOTAL];
    alignas(1024) uint8_t smem_a[NS * SA];                    // activation e4m3
    alignas(1024) uint8_t smem_b[NS * SMEM_B_PER_STAGE];      // weight     e4m3
    alignas(128)  uint8_t smem_sfa[NS * SSFA];                // activation SF (uint32 packed)
    alignas(128)  uint8_t smem_sfb[NS * SMEM_SFB_PER_STAGE];  // weight     SF

    // Barriers
    alignas(16) Barrier full_barriers[NS];            // A/B/SF TMA done (per-CTA, init 1)
    alignas(16) Barrier empty_barriers[NS];           // stage smem reusable
    alignas(16) Barrier with_sf_full_barriers[NS];    // SF transposed in smem, ready for UTCCP
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES];

    // TMEM base address
    alignas(16) uint32_t tmem_base;
};

// ======================== Kernel ========================
template <int M_TPL, bool kProfile>
__global__ void __launch_bounds__(TPB, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,    // activation [M,K] e4m3, K-major
    const __grid_constant__ CUtensorMap desc_B,    // weight     [N,K] e4m3, K-major
    const __grid_constant__ CUtensorMap desc_SFA,  // x_sf  [sf_k, sfa_mn] int32 (MN-major)
    const __grid_constant__ CUtensorMap desc_SFB,  // w_sf  [sf_k, sfb_mn] int32 (MN-major)
    const __grid_constant__ CUtensorMap desc_D,    // output [M,N] FP32 row-major
    int num_blocks,
    int64_t* prof)                                 // clock64 timing buffer (nullptr if disabled)
{
    using Dims = SwapDims<M_TPL>;
    constexpr int NS              = Dims::NUM_STAGES;
    constexpr int SA              = Dims::SMEM_A_PER_STAGE;
    constexpr int SSFA            = Dims::SMEM_SFA_PER_STAGE;
    constexpr int LOAD_BLOCK_M_T  = Dims::LOAD_BLOCK_M;    // 64 (fixed, BM/2)
    constexpr int UMMA_N_T        = Dims::UMMA_N;          // 128 (fixed, BM)
    constexpr int NUM_TMEM_COLS_T = Dims::NUM_TMEM_COLS;   // 512
    constexpr int TMEM_SFA        = Dims::TMEM_START_SFA;  // 256 (activation SF)
    constexpr int TMEM_SFB        = Dims::TMEM_START_SFB;  // 260 (weight SF)
    constexpr int NUM_M_SUB       = Dims::NUM_M_SUB;       // ceil(M/128) subtiles
    constexpr int BM_T            = BM;                    // 128 (subtile M)

    constexpr int NUM_TILES_TOTAL = NUM_N_TILES; // N tiles (inner loop over M subtiles)

    using Storage = SharedStorage<M_TPL>;
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    Storage& s = *reinterpret_cast<Storage*>(smem_buf);

    const uint32_t warp_id  = threadIdx.x / 32;
    const uint32_t lane_id  = ptx::get_lane_idx();
    const uint32_t cta_rank = ptx::block_rank_in_cluster();
    const bool is_leader    = (cta_rank == 0);

    // ================================================================
    // INITIALIZATION
    // ================================================================
    ptx::cluster_sync();

    if (warp_id == 0) {
        cute::prefetch_tma_descriptor(&desc_A);
        cute::prefetch_tma_descriptor(&desc_B);
        cute::prefetch_tma_descriptor(&desc_SFA);
        cute::prefetch_tma_descriptor(&desc_SFB);
        cute::prefetch_tma_descriptor(&desc_D);
    }

    if (warp_id == 1 && ptx::elect_one_sync()) {
        for (int i = 0; i < NS; ++i) {
            s.full_barriers[i].init(1);                        // per-CTA A/B/SF TMA
            s.empty_barriers[i].init(1);
            s.with_sf_full_barriers[i].init(NUM_MULTICAST * 32); // both CTAs' warp2 (32 lanes each)
        }
        for (int i = 0; i < NUM_EPI_STAGES; ++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(NUM_MULTICAST * NUM_STORE_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_id == 2) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_2sm(addr, NUM_TMEM_COLS_T);
    }

    ptx::cluster_sync();
    cudaGridDependencySynchronize();

    // SMEM bases (swap-AB: MMA A-operand = weight (smem_b), B-operand = activation (smem_a)).
    auto* smem_a_base = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a);   // activation
    auto* smem_b_base = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b);   // weight

    // ================================================================
    // Persistent tile scheduling (single M block; iterate N tiles)
    // ================================================================
    int num_clusters    = num_blocks / CLUSTER_SIZE;
    int cluster_id      = blockIdx.x / CLUSTER_SIZE;
    int num_tiles_total = NUM_TILES_TOTAL;

    // ======== WARP 0: TMA PRODUCER (both CTAs, plain per-CTA loads) ========
    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        auto advance = [&]() { stage = (stage + 1) % NS; if (stage == 0) phase ^= 1; };

        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            int n_base = tile_id * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N; // weight N (per CTA)
            int sfb_n  = n_base;                                             // weight SF (per CTA)
            // Inner loop over M subtiles: weight/SFB (n_base) reused -> HBM once, L2 after.
            for (int m_sub = 0; m_sub < NUM_M_SUB; ++m_sub) {
                int m_base = m_sub * BM_T + cta_rank * LOAD_BLOCK_M_T;  // this subtile's activation half
                int sfa_m  = m_sub * BM_T;                             // this subtile's SFA base

                // [PROFILE] Load (producer) window for this iteration's K-loop.
                long long prof_ld_t0 = 0;
                if (kProfile && cluster_id == 0 && cta_rank == 0)
                    prof_ld_t0 = ptx::rdclock();

                for (int k = 0; k < NUM_K_TILES; ++k) {
                    s.empty_barriers[stage].wait(phase ^ 1);
                    int k_off = k * BLOCK_K;

                    auto* sa  = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a + stage * SA);
                    auto* sb  = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b + stage * SMEM_B_PER_STAGE);
                    tma::copy_2d_fp8(&desc_A, &s.full_barriers[stage], sa, k_off, m_base);
                    tma::copy_2d_fp8(&desc_B, &s.full_barriers[stage], sb, k_off, n_base);

                    auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage * SSFA);
                    auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage * SMEM_SFB_PER_STAGE);
                    tma::copy_2d_sf(&desc_SFA, &s.full_barriers[stage], sfa, sfa_m, (uint32_t)k);
                    tma::copy_2d_sf(&desc_SFB, &s.full_barriers[stage], sfb, sfb_n, (uint32_t)k);

                    constexpr uint32_t kNumArrivalBytes =
                        SA + SMEM_B_PER_STAGE + BM * sizeof(uint32_t) + SF_BLOCK_N * sizeof(uint32_t);
                    s.full_barriers[stage].arrive_and_expect_tx(kNumArrivalBytes);
                    advance();
                }

                if (kProfile && cluster_id == 0 && cta_rank == 0) {
                    prof[persistent_iter * 7 + 0] = prof_ld_t0;
                    prof[persistent_iter * 7 + 1] = ptx::rdclock();
                }
                persistent_iter++;
            }
        }
    }

    // ======== WARP 2: SF TRANSPOSER (both CTAs) ========
    else if (warp_id == 2) {
        auto warp_transpose = [&](uint32_t* smem_ptr) {
            // read [4 x 32], write transposed [32 x 4] (DeepGEMM utccp_required_smem_warp_transpose)
            uint32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = ptx::ld_shared_u32(smem_ptr + i * 32 + lane_id);
            __syncwarp();
            ptx::st_shared_v4_u32(smem_ptr + lane_id * 4, v[0], v[1], v[2], v[3]);
        };
        uint32_t stage = 0, phase = 0;
        auto advance = [&]() { stage = (stage + 1) % NS; if (stage == 0) phase ^= 1; };

        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
          for (int m_sub = 0; m_sub < NUM_M_SUB; ++m_sub) {
            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.full_barriers[stage].wait(phase);
                auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage * SSFA);
                auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage * SMEM_SFB_PER_STAGE);
                #pragma unroll
                for (int i = 0; i < Dims::SF_BLOCK_M / NUM_UTCCP_ALIGNED; ++i)
                    warp_transpose(sfa + i * NUM_UTCCP_ALIGNED);
                #pragma unroll
                for (int i = 0; i < SF_BLOCK_N / NUM_UTCCP_ALIGNED; ++i)
                    warp_transpose(sfb + i * NUM_UTCCP_ALIGNED);
                cutlass::arch::fence_view_async_shared();
                s.with_sf_full_barriers[stage].arrive(0u);   // all 32 lanes arrive (x2 CTAs)
                advance();
            }
          }
        }
    }

    // ======== WARP 1: MMA CONSUMER (leader only) ========
    // Per-tile UTCCP + block_scale MMA delegated to the fixed-128 cluster_mma_fp8 engine.
    else if (warp_id == 1 && is_leader) {
        using CM = cluster_mma_fp8::ClusterMmaFP8BlockScale<BLOCK_K, NS>;
        static_assert(UMMA_N_T == CM::UMMA_N, "FIXED-BM kernel requires UMMA_N == 128");

        auto ds = CM::init_desc(smem_a_base, smem_b_base, lane_id);

        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
          for (int m_sub = 0; m_sub < NUM_M_SUB; ++m_sub) {
            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;

            // [PROFILE] MMA warp per-iteration window (includes the accumulator-empty
            // wait that run_tile does first).
            long long prof_mma_t0 = 0;
            if (kProfile && cluster_id == 0 && lane_id == 0)
                prof_mma_t0 = ptx::rdclock();

            // SF roles: smem_sfa/TMEM_SFA = activation SF ; smem_sfb/TMEM_SFB = weight SF.
            // kProfile: run_tile accumulates the MMA warp's WAIT cycles into prof col6,
            // so MMA_active = (mma_end - mma_start) - wait reveals compute vs stall.
            long long* wait_ptr = (kProfile && cluster_id == 0)
                ? reinterpret_cast<long long*>(&prof[persistent_iter * 7 + 6]) : nullptr;
            CM::template run_tile<kProfile>(ds, s.with_sf_full_barriers, s.empty_barriers,
                         s.tmem_full_barriers[accum_stage], s.tmem_empty_barriers[accum_stage],
                         s.smem_sfa, s.smem_sfb,
                         accum_stage * UMMA_N_T, TMEM_SFA, TMEM_SFB,
                         NUM_K_TILES, accum_phase, stage, phase, wait_ptr);

            if (kProfile && cluster_id == 0 && lane_id == 0) {
                prof[persistent_iter * 7 + 2] = prof_mma_t0;
                prof[persistent_iter * 7 + 3] = ptx::rdclock();
            }
            persistent_iter++;
          }
        }

        if (persistent_iter > 0) {
            uint32_t last_iter        = persistent_iter - 1;
            uint32_t last_accum_stage = last_iter % NUM_EPI_STAGES;
            uint32_t last_accum_phase = (last_iter / NUM_EPI_STAGES) & 1;
            s.tmem_empty_barriers[last_accum_stage].wait(last_accum_phase);
        }
    }

    // ======== EPILOGUE WARPS (both CTAs, 128 threads / 4 warps) — FP32 store, unchanged ========
    else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
             warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        uint32_t epi_warp_idx = warp_id - (NUM_NON_EPI_THREADS / 32);  // 0..3
        uint32_t tma_store_idx = 0;

        constexpr int NUM_STORES         = BM_T / STORE_BLOCK_M;               // 8 per subtile (TMA clips rows >= M)
        constexpr int NUM_TMEM_SUBROWS   = STORE_BLOCK_M / 8;                  // 2
        constexpr int NUM_N_STORE_ATOMS  = STORE_BLOCK_N / STORE_BLOCK_N_ATOM; // 4
        constexpr int SMEM_CD_PER_STAGE_T = SMEM_CD_PER_STAGE;                 // 8192

        uint32_t persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
          for (int m_sub = 0; m_sub < NUM_M_SUB; ++m_sub) {
            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;

            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_fence_after_sync();

            // [PROFILE] Epilogue (leader CTA): start of this iteration's readback+store.
            long long prof_epi_t0 = 0;
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0)
                prof_epi_t0 = ptx::rdclock();

            uint32_t tmem_base = accum_stage * UMMA_N_T;
            int base_n = tile_id * CLUSTER_BLOCK_N + cta_rank * BLOCK_N;
            int base_m = m_sub * BM_T;   // this subtile's output rows (TMA clips >= M)

            for (int st = 0; st < NUM_STORES; ++st,
                 tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES) {
                auto* smem_cd_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * SMEM_CD_PER_STAGE_T);

                if (epi_warp_idx == 0)
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                // ---- Read TMEM (FP32), transpose into SMEM (DeepGEMM swap FP32 path) ----
                #pragma unroll
                for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                    uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * 8;

                    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                    ptx::tmem_load_32dp32b8x(tmem_addr, v0, v1, v2, v3, v4, v5, v6, v7);
                    cutlass::arch::fence_view_async_tmem_load();
                    uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};

                    uint8_t* smem_base_ptr = smem_cd_ptr
                        + epi_warp_idx * (STORE_BLOCK_M * SWIZZLE_CD)
                        + i * (8 * SWIZZLE_CD);
                    uint32_t col = lane_id / 4;
                    #pragma unroll
                    for (uint32_t row = 0; row < 8; ++row) {
                        uint8_t* smem_ptr = smem_base_ptr
                            + row * (16 * 8)
                            + (col ^ row) * 16
                            + (lane_id % 4) * sizeof(float);
                        ptx::st_shared_u32(smem_ptr, vals[row]);
                    }
                }

                if (st == NUM_STORES - 1) {
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                if (epi_warp_idx == 0 && ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int i = 0; i < NUM_N_STORE_ATOMS; ++i) {
                        auto* smem_ptr = reinterpret_cast<float*>(smem_cd_ptr)
                            + i * (STORE_BLOCK_M * STORE_BLOCK_N_ATOM);
                        int n_idx = base_n + i * STORE_BLOCK_N_ATOM;
                        int m_idx = base_m + st * STORE_BLOCK_M;
                        tma::store_2d(&desc_D, smem_ptr, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

            // [PROFILE] Epilogue (leader CTA): end of this iteration's readback+store.
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0) {
                prof[persistent_iter * 7 + 4] = prof_epi_t0;
                prof[persistent_iter * 7 + 5] = ptx::rdclock();
            }

            persistent_iter++;
          }
        }
    }

    // ================================================================
    // CLEANUP
    // ================================================================
    ptx::cluster_sync();
    if (warp_id == 0) {
        ptx::tcgen05_dealloc_2sm(0, NUM_TMEM_COLS_T);
    }
}

// ======================== Non-swap specialization (M >= 160) ========================
// DeepGEMM's heuristic selects BM=128, BN=224 and a cluster along M for this
// problem. Consecutive CTA ranks own consecutive 128-row M blocks and share one
// 224-column N block through the 2SM MMA.
namespace nonswap {

static constexpr int BLOCK_M_NS        = 128;
static constexpr int BLOCK_N_NS        = 224;
static constexpr int LOAD_BLOCK_M_NS   = 128;
static constexpr int LOAD_BLOCK_N_NS   = BLOCK_N_NS / NUM_MULTICAST;  // 112
static constexpr int SF_BLOCK_M_NS     = 128;
static constexpr int SF_BLOCK_N_NS     = 256;  // UTCCP alignment of BLOCK_N_NS
static constexpr int NUM_STAGES_NS     = 6;
static constexpr int UMMA_N_NS         = BLOCK_N_NS;
static constexpr int NUM_TMEM_COLS_NS  = 512;
static constexpr int TMEM_SFA_NS       = UMMA_N_NS * NUM_EPI_STAGES;  // 448
static constexpr int TMEM_SFB_NS       = TMEM_SFA_NS + SF_BLOCK_M_NS / 32; // 452

static constexpr int SMEM_A_PER_STAGE_NS   = LOAD_BLOCK_M_NS * BLOCK_K; // 16384
static constexpr int SMEM_B_PER_STAGE_NS   = LOAD_BLOCK_N_NS * BLOCK_K; // 14336
static constexpr int SMEM_SFA_PER_STAGE_NS = SF_BLOCK_M_NS * sizeof(uint32_t); // 512
static constexpr int SMEM_SFB_PER_STAGE_NS = SF_BLOCK_N_NS * sizeof(uint32_t); // 1024

static constexpr int STORE_BLOCK_M_NS       = BLOCK_M_NS;
static constexpr int STORE_BLOCK_N_NS       = SWIZZLE_CD / sizeof(float); // 32
static constexpr int SMEM_CD_PER_STAGE_NS   = STORE_BLOCK_M_NS * STORE_BLOCK_N_NS * sizeof(float); // 16384
static constexpr int SMEM_CD_TOTAL_NS       = SMEM_CD_PER_STAGE_NS * NUM_TMA_STORE_STAGES; // 32768

static constexpr int NUM_M_BLOCKS_NS = 2; // M=160..256 => ceil(M/128)=2
static constexpr int NUM_N_BLOCKS_NS = (N_TOTAL + BLOCK_N_NS - 1) / BLOCK_N_NS; // 293
static constexpr int NUM_TASKS_NS    = NUM_M_BLOCKS_NS * NUM_N_BLOCKS_NS;       // 586 CTAs
static_assert(NUM_TASKS_NS % CLUSTER_SIZE == 0, "non-swap task tail must contain full clusters");

struct SharedStorage {
    alignas(1024) uint8_t smem_cd[SMEM_CD_TOTAL_NS];
    alignas(1024) uint8_t smem_a[NUM_STAGES_NS * SMEM_A_PER_STAGE_NS];
    alignas(1024) uint8_t smem_b[NUM_STAGES_NS * SMEM_B_PER_STAGE_NS];
    alignas(128)  uint8_t smem_sfa[NUM_STAGES_NS * SMEM_SFA_PER_STAGE_NS];
    alignas(128)  uint8_t smem_sfb[NUM_STAGES_NS * SMEM_SFB_PER_STAGE_NS];

    alignas(16) Barrier full_barriers[NUM_STAGES_NS];
    alignas(16) Barrier empty_barriers[NUM_STAGES_NS];
    alignas(16) Barrier with_sf_full_barriers[NUM_STAGES_NS];
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES];
    alignas(16) uint32_t tmem_base;
};

static_assert(sizeof(SharedStorage) <= SMEM_CAPACITY,
              "non-swap FP8 shared storage exceeds SM100 capacity");

// With an even physical grid, task pairs remain cluster-aligned for every
// persistent iteration. This is the target-shape reduction of DeepGEMM's L2
// swizzled scheduler (two M blocks are the primary group).
__device__ __forceinline__ bool get_task(
    uint32_t persistent_iter, int physical_grid,
    uint32_t& m_block_idx, uint32_t& n_block_idx) {
    uint32_t task_idx = persistent_iter * static_cast<uint32_t>(physical_grid) + blockIdx.x;
    if (task_idx >= NUM_TASKS_NS)
        return false;
    m_block_idx = task_idx & 1u;
    n_block_idx = task_idx >> 1u;
    return true;
}

template <bool kProfile>
__global__ void __launch_bounds__(TPB, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    const __grid_constant__ CUtensorMap desc_SFA,
    const __grid_constant__ CUtensorMap desc_SFB,
    const __grid_constant__ CUtensorMap desc_D,
    int physical_grid,
    int64_t* prof) {
    using CM = cluster_mma_fp8::ClusterMmaFP8BlockScale<
        BLOCK_K, NUM_STAGES_NS, BLOCK_M_NS, BLOCK_N_NS, false>;
    static_assert(CM::UMMA_N == UMMA_N_NS, "invalid non-swap UMMA N");
    static_assert(CM::SMEM_ACT_PER_STAGE == SMEM_A_PER_STAGE_NS, "invalid activation stage stride");
    static_assert(CM::SMEM_WGT_PER_STAGE == SMEM_B_PER_STAGE_NS, "invalid weight stage stride");
    static_assert(CM::SMEM_SF_ACT_PER_STAGE == SMEM_SFA_PER_STAGE_NS, "invalid activation SF stride");
    static_assert(CM::SMEM_SF_WGT_PER_STAGE == SMEM_SFB_PER_STAGE_NS, "invalid weight SF stride");

    extern __shared__ __align__(1024) uint8_t smem_buf[];
    SharedStorage& s = *reinterpret_cast<SharedStorage*>(smem_buf);

    const uint32_t warp_id  = threadIdx.x / 32;
    const uint32_t lane_id  = ptx::get_lane_idx();
    const uint32_t cta_rank = ptx::block_rank_in_cluster();
    const bool is_leader    = cta_rank == 0;

    ptx::cluster_sync();

    if (warp_id == 0) {
        cute::prefetch_tma_descriptor(&desc_A);
        cute::prefetch_tma_descriptor(&desc_B);
        cute::prefetch_tma_descriptor(&desc_SFA);
        cute::prefetch_tma_descriptor(&desc_SFB);
        cute::prefetch_tma_descriptor(&desc_D);
    }

    if (warp_id == 1 && ptx::elect_one_sync()) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES_NS; ++i) {
            s.full_barriers[i].init(1);
            s.empty_barriers[i].init(1);
            s.with_sf_full_barriers[i].init(NUM_MULTICAST * 32);
        }
        #pragma unroll
        for (int i = 0; i < NUM_EPI_STAGES; ++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(NUM_MULTICAST * NUM_STORE_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_id == 2) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_2sm(addr, NUM_TMEM_COLS_NS);
    }

    ptx::cluster_sync();
    cudaGridDependencySynchronize();

    auto* smem_a_base = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a);
    auto* smem_b_base = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b);

    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        auto advance = [&]() {
            stage = (stage + 1) % NUM_STAGES_NS;
            if (stage == 0) phase ^= 1;
        };

        uint32_t m_block_idx, n_block_idx;
        while (get_task(persistent_iter, physical_grid, m_block_idx, n_block_idx)) {
            const int m_base = m_block_idx * BLOCK_M_NS;
            const int n_base = n_block_idx * BLOCK_N_NS + cta_rank * LOAD_BLOCK_N_NS;
            const int sfa_m  = m_block_idx * BLOCK_M_NS;
            const int sfb_n  = n_block_idx * BLOCK_N_NS;

            long long prof_t0 = 0;
            if (kProfile && blockIdx.x == 0)
                prof_t0 = ptx::rdclock();

            #pragma unroll 4
            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.empty_barriers[stage].wait(phase ^ 1);
                const int k_off = k * BLOCK_K;

                auto* sa = reinterpret_cast<__nv_fp8_e4m3*>(
                    s.smem_a + stage * SMEM_A_PER_STAGE_NS);
                auto* sb = reinterpret_cast<__nv_fp8_e4m3*>(
                    s.smem_b + stage * SMEM_B_PER_STAGE_NS);
                tma::copy_2d_fp8(&desc_A, &s.full_barriers[stage], sa, k_off, m_base);
                tma::copy_2d_fp8(&desc_B, &s.full_barriers[stage], sb, k_off, n_base);

                auto* sfa = reinterpret_cast<uint32_t*>(
                    s.smem_sfa + stage * SMEM_SFA_PER_STAGE_NS);
                auto* sfb = reinterpret_cast<uint32_t*>(
                    s.smem_sfb + stage * SMEM_SFB_PER_STAGE_NS);
                tma::copy_2d_sf(&desc_SFA, &s.full_barriers[stage], sfa, sfa_m, k);
                tma::copy_2d_sf(&desc_SFB, &s.full_barriers[stage], sfb, sfb_n, k);

                constexpr uint32_t kArrivalBytes =
                    SMEM_A_PER_STAGE_NS + SMEM_B_PER_STAGE_NS +
                    BLOCK_M_NS * sizeof(uint32_t) + BLOCK_N_NS * sizeof(uint32_t);
                s.full_barriers[stage].arrive_and_expect_tx(kArrivalBytes);
                advance();
            }

            if (kProfile && blockIdx.x == 0) {
                prof[persistent_iter * 7 + 0] = prof_t0;
                prof[persistent_iter * 7 + 1] = ptx::rdclock();
            }
            ++persistent_iter;
        }
    } else if (warp_id == 2) {
        auto warp_transpose = [&](uint32_t* smem_ptr) {
            uint32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = ptx::ld_shared_u32(smem_ptr + i * 32 + lane_id);
            __syncwarp();
            ptx::st_shared_v4_u32(smem_ptr + lane_id * 4, v[0], v[1], v[2], v[3]);
        };

        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        auto advance = [&]() {
            stage = (stage + 1) % NUM_STAGES_NS;
            if (stage == 0) phase ^= 1;
        };

        uint32_t m_block_idx, n_block_idx;
        while (get_task(persistent_iter, physical_grid, m_block_idx, n_block_idx)) {
            #pragma unroll 4
            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.full_barriers[stage].wait(phase);

                auto* sfa = reinterpret_cast<uint32_t*>(
                    s.smem_sfa + stage * SMEM_SFA_PER_STAGE_NS);
                auto* sfb = reinterpret_cast<uint32_t*>(
                    s.smem_sfb + stage * SMEM_SFB_PER_STAGE_NS);
                warp_transpose(sfa);
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (int i = 0; i < SF_BLOCK_N_NS / NUM_UTCCP_ALIGNED; ++i)
                    warp_transpose(sfb + i * NUM_UTCCP_ALIGNED);
                cutlass::arch::fence_view_async_shared();

                s.with_sf_full_barriers[stage].arrive(0u);
                advance();
            }
            ++persistent_iter;
        }
    } else if (warp_id == 1 && is_leader) {
        auto ds = CM::init_desc(smem_a_base, smem_b_base, lane_id);
        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        uint32_t m_block_idx, n_block_idx;

        while (get_task(persistent_iter, physical_grid, m_block_idx, n_block_idx)) {
            const uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            const uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;

            long long prof_t0 = 0;
            if (kProfile && blockIdx.x == 0 && lane_id == 0)
                prof_t0 = ptx::rdclock();
            long long* wait_ptr = (kProfile && blockIdx.x == 0)
                ? reinterpret_cast<long long*>(&prof[persistent_iter * 7 + 6]) : nullptr;

            CM::template run_tile<kProfile>(
                ds, s.with_sf_full_barriers, s.empty_barriers,
                s.tmem_full_barriers[accum_stage], s.tmem_empty_barriers[accum_stage],
                s.smem_sfa, s.smem_sfb,
                accum_stage * UMMA_N_NS, TMEM_SFA_NS, TMEM_SFB_NS,
                NUM_K_TILES, accum_phase, stage, phase, wait_ptr);

            if (kProfile && blockIdx.x == 0 && lane_id == 0) {
                prof[persistent_iter * 7 + 2] = prof_t0;
                prof[persistent_iter * 7 + 3] = ptx::rdclock();
            }
            ++persistent_iter;
        }

        if (persistent_iter > 0) {
            const uint32_t last_iter = persistent_iter - 1;
            const uint32_t accum_stage = last_iter % NUM_EPI_STAGES;
            const uint32_t accum_phase = (last_iter / NUM_EPI_STAGES) & 1;
            s.tmem_empty_barriers[accum_stage].wait(accum_phase);
        }
    } else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
               warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        const uint32_t epi_warp_idx = warp_id - NUM_NON_EPI_THREADS / 32;
        uint32_t tma_store_idx = 0, persistent_iter = 0;
        uint32_t m_block_idx, n_block_idx;

        while (get_task(persistent_iter, physical_grid, m_block_idx, n_block_idx)) {
            const uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            const uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;
            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_fence_after_sync();

            long long prof_t0 = 0;
            if (kProfile && blockIdx.x == 0 && epi_warp_idx == 0 && lane_id == 0)
                prof_t0 = ptx::rdclock();

            const uint32_t tmem_base = accum_stage * UMMA_N_NS;
            const int base_m = m_block_idx * BLOCK_M_NS;
            const int base_n = n_block_idx * BLOCK_N_NS;

            #pragma unroll
            for (int st = 0; st < BLOCK_N_NS / STORE_BLOCK_N_NS; ++st) {
                auto* smem_cd_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * SMEM_CD_PER_STAGE_NS);

                if (epi_warp_idx == 0)
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                #pragma unroll
                for (int i = 0; i < STORE_BLOCK_N_NS / 4; ++i) {
                    const uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_N_NS + i * 4;
                    uint32_t v0, v1, v2, v3;
                    ptx::tmem_load_32dp32b4x(tmem_addr, v0, v1, v2, v3);
                    cutlass::arch::fence_view_async_tmem_load();

                    const uint32_t row = lane_id;
                    const uint32_t col = i ^ (row % 8);
                    auto* smem_ptr = smem_cd_ptr +
                        epi_warp_idx * 32 * SWIZZLE_CD +
                        row * SWIZZLE_CD + col * 16;
                    ptx::st_shared_v4_u32(smem_ptr, v0, v1, v2, v3);
                }

                if (st == BLOCK_N_NS / STORE_BLOCK_N_NS - 1) {
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);
                if (epi_warp_idx == 0 && ptx::elect_one_sync()) {
                    tma::store_2d(&desc_D, reinterpret_cast<float*>(smem_cd_ptr),
                                  base_n + st * STORE_BLOCK_N_NS, base_m);
                    cute::tma_store_arrive();
                }
                __syncwarp();
                tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES;
            }

            if (kProfile && blockIdx.x == 0 && epi_warp_idx == 0 && lane_id == 0) {
                prof[persistent_iter * 7 + 4] = prof_t0;
                prof[persistent_iter * 7 + 5] = ptx::rdclock();
            }
            ++persistent_iter;
        }
    }

    ptx::cluster_sync();
    if (warp_id == 0)
        ptx::tcgen05_dealloc_2sm(0, NUM_TMEM_COLS_NS);
}

} // namespace nonswap

// ======================== Host: TMA Descriptors ========================
static CUtensorMap make_tma_desc_fp8_2d(
    const __nv_fp8_e4m3* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(__nv_fp8_e4m3)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

// SF descriptor: [sf_k, mn] int32, MN-major, no swizzle. box = (box_mn, 1) at coords (mn, k).
static CUtensorMap make_tma_desc_sf_2d(const uint32_t* ptr, int mn, int sf_k, int box_mn)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)mn, (uint64_t)sf_k};
    uint64_t globalStride[1] = {(uint64_t)mn * sizeof(uint32_t)};
    uint32_t boxDim[2]       = {(uint32_t)box_mn, 1u};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_INT32,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

static CUtensorMap make_tma_desc_fp32_2d(
    const float* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(float)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

// ======================== Kernel / SMEM selectors ========================
// M<=128 keeps the established swap-AB path. M>=160 uses the DeepGEMM-selected
// non-swap BM=128, BN=224 specialization.
template <bool kProfile>
static void* get_kernel_ptr(int M) {
    switch (M) {
        case 32:  return (void*)&wq_b_proj_kernel<32,  kProfile>;
        case 64:  return (void*)&wq_b_proj_kernel<64,  kProfile>;
        case 96:  return (void*)&wq_b_proj_kernel<96,  kProfile>;
        case 128: return (void*)&wq_b_proj_kernel<128, kProfile>;
        case 160:
        case 192:
        case 224:
        case 256: return (void*)&nonswap::wq_b_proj_kernel<kProfile>;
        default:  return nullptr;
    }
}
static int get_smem_bytes(int M) {
    switch (M) {
        case 32:  return (int)sizeof(SharedStorage<32>);
        case 64:  return (int)sizeof(SharedStorage<64>);
        case 96:  return (int)sizeof(SharedStorage<96>);
        case 128: return (int)sizeof(SharedStorage<128>);
        case 160:
        case 192:
        case 224:
        case 256: return (int)sizeof(nonswap::SharedStorage);
        default:  return 0;
    }
}

// FP8 block-scale run. Inputs are pre-quantized:
//   x_fp8 [M,K] e4m3 ; x_sf [sf_k, sfa_mn] int32 (4 UE8M0/uint32, MN-major)
//   w_fp8 [N,K] e4m3 ; w_sf [sf_k, sfb_mn] int32
// Output FP32 [M,N]. (profile: also returns clock64 timing[max_iters,7].)
static std::vector<torch::Tensor> run_wq_b(
    torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w_fp8, torch::Tensor w_sf, bool profile)
{
    TORCH_CHECK(x_fp8.is_cuda() && x_fp8.is_contiguous() &&
                x_fp8.scalar_type() == torch::kFloat8_e4m3fn, "x_fp8 must be CUDA e4m3");
    TORCH_CHECK(w_fp8.is_cuda() && w_fp8.is_contiguous() &&
                w_fp8.scalar_type() == torch::kFloat8_e4m3fn, "w_fp8 must be CUDA e4m3");
    TORCH_CHECK(x_sf.is_cuda() && x_sf.scalar_type() == torch::kInt32, "x_sf must be CUDA int32");
    TORCH_CHECK(w_sf.is_cuda() && w_sf.scalar_type() == torch::kInt32, "w_sf must be CUDA int32");

    const int M = x_fp8.size(0);
    TORCH_CHECK(x_fp8.size(1) == K_DIM, "x_fp8 must be [M,", K_DIM, "]");
    TORCH_CHECK(w_fp8.size(0) == N_TOTAL && w_fp8.size(1) == K_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0,
                "kernel supports 32-aligned M in [32,256], got M=", M);
    const bool use_nonswap = M > 128;
    const int num_m_sub = (M + BM - 1) / BM;  // swap-path profiling only

    // SF shapes: 4 UE8M0/uint32 -> sf_k = K/(gran_k*4) = K/BLOCK_K. mn aligned to 16/4 = 4.
    const int sf_k   = K_DIM / (GRAN_K * SF_IDS_PER_UINT);        // 12
    const int sfa_mn = align_up(M, 16 / SF_ELEM_SIZE);           // align M to 4
    const int sfb_mn = align_up(N_TOTAL, 16 / SF_ELEM_SIZE);     // 65536
    TORCH_CHECK(x_sf.dim() == 2 && x_sf.size(0) == sf_k && x_sf.size(1) == sfa_mn,
                "x_sf must be [", sf_k, ",", sfa_mn, "] (MN-major, 4-scale/uint32)");
    TORCH_CHECK(w_sf.dim() == 2 && w_sf.size(0) == sf_k && w_sf.size(1) == sfb_mn,
                "w_sf must be [", sf_k, ",", sfb_mn, "]");

    auto out = torch::empty({M, N_TOTAL}, x_fp8.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_ptr   = reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8.data_ptr());
    auto w_ptr   = reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8.data_ptr());
    auto xsf_ptr = reinterpret_cast<const uint32_t*>(x_sf.data_ptr());
    auto wsf_ptr = reinterpret_cast<const uint32_t*>(w_sf.data_ptr());
    auto out_ptr = reinterpret_cast<float*>(out.data_ptr());

    // Both paths use K-major 128B-swizzled operands. TMA OOB handling covers the
    // partial second M block and the final 224-column N block in the non-swap path.
    const int load_block_m = use_nonswap ? nonswap::LOAD_BLOCK_M_NS : BM / NUM_MULTICAST;
    const int load_block_n = use_nonswap ? nonswap::LOAD_BLOCK_N_NS : LOAD_BLOCK_N;
    const int sfb_block_n  = use_nonswap ? nonswap::BLOCK_N_NS : SF_BLOCK_N;
    const int store_block_m = use_nonswap ? nonswap::STORE_BLOCK_M_NS : STORE_BLOCK_M;
    const int store_block_n = use_nonswap ? nonswap::STORE_BLOCK_N_NS : STORE_BLOCK_N_ATOM;
    CUtensorMap desc_A   = make_tma_desc_fp8_2d(x_ptr, M, K_DIM, load_block_m, BLOCK_K);
    CUtensorMap desc_B   = make_tma_desc_fp8_2d(w_ptr, N_TOTAL, K_DIM, load_block_n, BLOCK_K);
    CUtensorMap desc_SFA = make_tma_desc_sf_2d(xsf_ptr, sfa_mn, sf_k, BM);
    CUtensorMap desc_SFB = make_tma_desc_sf_2d(wsf_ptr, sfb_mn, sf_k, sfb_block_n);
    CUtensorMap desc_D   = make_tma_desc_fp32_2d(
        out_ptr, M, N_TOTAL, store_block_m, store_block_n);

    // Grid: persistent, cluster of 2 CTAs.
    static const int num_SMs = []() {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
        return n;
    }();
    TORCH_CHECK(num_SMs >= CLUSTER_SIZE,
                "wq_b FP8 requires at least ", CLUSTER_SIZE, " SMs, got ", num_SMs);
    int grid_size = 0, max_iters = 0;
    if (use_nonswap) {
        // DeepGEMM launches all physical SMs. Keep the grid even so every task
        // pair remains a valid 2-CTA cluster across persistent iterations.
        const int max_clusters = num_SMs / CLUSTER_SIZE;
        int num_clusters = max_clusters;
        if (const char* e = std::getenv("WQ_B_CLUSTERS")) {
            const int req = atoi(e);
            if (req > 0) num_clusters = req < max_clusters ? req : max_clusters;
        }
        grid_size = num_clusters * CLUSTER_SIZE;
        max_iters = (nonswap::NUM_TASKS_NS + grid_size - 1) / grid_size;
    } else {
        const int total_cta = NUM_N_TILES * CLUSTER_SIZE;
        const int max_clusters = min(num_SMs, total_cta) / CLUSTER_SIZE;
        int num_clusters = max_clusters > 0 ? max_clusters : 1;
        while (num_clusters > 1 && NUM_N_TILES % num_clusters != 0) --num_clusters;
        if (const char* e = std::getenv("WQ_B_CLUSTERS")) {
            const int req = atoi(e);
            if (req > 0) num_clusters = req < max_clusters ? req : max_clusters;
        }
        grid_size = num_clusters * CLUSTER_SIZE;
        max_iters = ((NUM_N_TILES + num_clusters - 1) / num_clusters) * num_m_sub;
    }

    int64_t* prof_dev = nullptr;
    torch::Tensor timing;
    if (profile) {
        // [max_iters, 7] per persistent iteration on cluster0/CTA0 (same SM, clock64):
        //   load_start/end | mma_start/end | epi_start/end | mma_wait_cycles
        timing = torch::zeros({max_iters, 7}, x_fp8.options().dtype(torch::kInt64));
        prof_dev = reinterpret_cast<int64_t*>(timing.data_ptr());
    }

    void* kernel_ptr = profile ? get_kernel_ptr<true>(M) : get_kernel_ptr<false>(M);
    int smem_bytes = get_smem_bytes(M);
    TORCH_CHECK(kernel_ptr != nullptr && smem_bytes > 0, "Unsupported M=", M);

    static bool smem_configured[2][9] = {{false}};
    const int m_idx = M / 32;
    const int p_idx = profile ? 1 : 0;
    if (!smem_configured[p_idx][m_idx]) {
        auto attr_err = cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(attr_err), " smem_bytes=", smem_bytes);
        smem_configured[p_idx][m_idx] = true;
    }

    {
        dim3 grid(grid_size, 1, 1);
        dim3 block(TPB, 1, 1);

        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem_bytes;
        config.stream = stream;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = CLUSTER_SIZE;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;

        void* ptr_args[] = { &desc_A, &desc_B, &desc_SFA, &desc_SFB, &desc_D, &grid_size, &prof_dev };
        auto err = cudaLaunchKernelExC(&config, kernel_ptr, ptr_args);
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    }

    if (profile) return {out, timing};
    return {out};
}

// ======================== PyTorch Binding ========================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x_fp8, torch::Tensor x_sf, torch::Tensor w_fp8, torch::Tensor w_sf)
{
    return run_wq_b(x_fp8, x_sf, w_fp8, w_sf, /*profile=*/false)[0];
}

std::vector<torch::Tensor> wq_b_proj_gemm_profiled(
    torch::Tensor x_fp8, torch::Tensor x_sf, torch::Tensor w_fp8, torch::Tensor w_sf)
{
    return run_wq_b(x_fp8, x_sf, w_fp8, w_sf, /*profile=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj FP8 block-scale (tcgen05 2SM, hybrid layout, Blackwell) -> FP32");
    m.def("wq_b_proj_gemm_profiled", &wq_b_proj_gemm_profiled,
          "wq_b proj FP8 + clock64 load/MMA/epilogue timing -> (out, timing[max_iters,7])");
}
