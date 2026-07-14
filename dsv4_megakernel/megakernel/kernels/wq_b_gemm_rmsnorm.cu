// ============================================================
// wq_b_gemm_rmsnorm.cu
// tcgen05 BF16 GEMM — Kernel + Host + PyTorch Binding (SWAP-AB)
// Aligned with DeepGEMM sm100_bf16_gemm.cuh (swap_ab path)
//
// M=32~256 (32-aligned), K=1536, N=65536, BF16 -> FP32 output
// swap_ab=1: UMMA_M=256 along N, UMMA_N=M along M.
// 2SM MMA (cta_group::2), Cluster=(2,1,1) [cluster_n=2],
// kIsMulticastOnA=true, Persistent, Warp-Specialized.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "wq_b_gemm_rmsnorm.cuh"

using namespace wq_b;
using Barrier = mma_desc::Barrier;

// ======================== Shared Memory Layout ========================
template <int M_TPL>
struct SharedStorage {
    using D = SwapDims<M_TPL>;
    static constexpr int NUM_STAGES_T       = D::NUM_STAGES;
    static constexpr int SMEM_A_PER_STAGE_T = D::SMEM_A_PER_STAGE;
    static constexpr int NUM_EPI_STAGES_T   = D::NUM_EPI_STAGES;

    alignas(1024) uint8_t smem_cd[SMEM_CD_TOTAL];
    alignas(1024) uint8_t smem_a[NUM_STAGES_T * SMEM_A_PER_STAGE_T];
    alignas(1024) uint8_t smem_b[NUM_STAGES_T * SMEM_B_PER_STAGE];

    // Fused RMSNorm reduction scratch (per M-row, over head_dim).
    //   smem_warp_sq : per-warp partial sum-of-squares (both paths; avoids pre-zero + atomics)
    //   smem_peer_sq : partial pushed by the peer CTA (double-buffered by epi stage)
    //   smem_rms     : rsqrt(mean + eps) scale applied in the store
    alignas(16) float smem_warp_sq[4][M_TPL];
    alignas(16) float smem_peer_sq[NUM_EPI_STAGES_T][M_TPL];
    alignas(16) float smem_rms[M_TPL];

    // Barriers
    alignas(16) Barrier full_barriers[NUM_STAGES_T];
    alignas(16) Barrier empty_barriers[NUM_STAGES_T];
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES_T];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES_T];
    alignas(16) Barrier dsmem_barriers[NUM_EPI_STAGES_T];   // cross-CTA sum-of-squares push

    // TMEM base address
    alignas(16) uint32_t tmem_base;
};

// ======================== Kernel ========================
template <int M_TPL, bool kProfile>
__global__ void __launch_bounds__(TPB, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,   // activation [M, K], K-major
    const __grid_constant__ CUtensorMap desc_B,   // weight     [N, K], K-major
    const __grid_constant__ CUtensorMap desc_D,   // output     [M, N], FP32 row-major
    int num_blocks,
    float eps,                                    // fused RMSNorm epsilon
    int64_t* prof)                                // clock64 timing buffer (nullptr if disabled)
{
    using Dims = SwapDims<M_TPL>;
    constexpr int NUM_STAGES_T       = Dims::NUM_STAGES;
    constexpr int SMEM_A_PER_STAGE_T = Dims::SMEM_A_PER_STAGE;
    constexpr int LOAD_BLOCK_M_T     = Dims::LOAD_BLOCK_M;   // M/2
    constexpr int UMMA_N_T           = Dims::UMMA_N;         // M
    constexpr int NUM_TMEM_COLS_T    = Dims::NUM_TMEM_COLS;  // NUM_EPI_STAGES*2*M (pow2)
    constexpr int NUM_EPI_STAGES_T   = Dims::NUM_EPI_STAGES; // 2 for M<=128 else 1
    constexpr int ACCUM_COLS_T       = SUBTILES_PER_HEAD * UMMA_N_T; // TMEM cols per epi stage

    // Fused: one cluster owns a whole head (HEAD_DIM of N) as SUBTILES_PER_HEAD sub-tiles,
    // so the head_dim reduction is cluster-local. Persistent loop iterates over heads.
    constexpr int NUM_TILES_TOTAL = NUM_HEAD_TILES;

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
        cute::prefetch_tma_descriptor(&desc_D);
    }

    if (warp_id == 1 && ptx::elect_one_sync()) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES_T; ++i) {
            s.full_barriers[i].init(NUM_MULTICAST);   // arrivals from both CTAs (+ tx to leader)
            s.empty_barriers[i].init(1);
        }
        #pragma unroll
        for (int i = 0; i < NUM_EPI_STAGES_T; ++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(NUM_MULTICAST * NUM_STORE_THREADS);
            // One local arrive (arrive_and_expect_tx) + transaction bytes pushed by the peer CTA.
            s.dsmem_barriers[i].init(1);
        }
        cutlass::arch::fence_barrier_init();
    }

    if (warp_id == 2) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_2sm(addr, NUM_TMEM_COLS_T);
    }

    ptx::cluster_sync();
    cudaGridDependencySynchronize();

    // ================================================================
    // SMEM DESCRIPTOR PRE-COMPUTATION (warp-shuffle trick)
    // swap-AB: MMA A-operand = weight (smem_b), B-operand = activation (smem_a)
    // ================================================================
    auto* smem_a_base = reinterpret_cast<nv_bfloat16*>(s.smem_a);   // activation
    auto* smem_b_base = reinterpret_cast<nv_bfloat16*>(s.smem_b);   // weight

    auto act_desc = mma_desc::make_smem_desc_k_major(smem_a_base);  // B-operand
    auto wgt_desc = mma_desc::make_smem_desc_k_major(smem_b_base);  // A-operand

    uint32_t act_desc_lo = (lane_id < NUM_STAGES_T)
        ? act_desc.lo + lane_id * (SMEM_A_PER_STAGE_T / 16) : 0u;
    uint32_t wgt_desc_lo = (lane_id < NUM_STAGES_T)
        ? wgt_desc.lo + lane_id * (SMEM_B_PER_STAGE / 16) : 0u;

    uint64_t runtime_idesc = mma_desc::make_runtime_instr_desc<UMMA_N_T>();

    // ================================================================
    // Persistent tile scheduling (single M block; iterate over heads)
    // Each cluster processes one head = HEAD_DIM (=512) columns of N, laid out as
    // SUBTILES_PER_HEAD (=2) sub-tiles of CLUSTER_BLOCK_N (=256). Within sub-tile j,
    // CTA rank r owns N[head*512 + j*256 + r*128 : +128].
    // ================================================================
    int num_clusters      = num_blocks / CLUSTER_SIZE;
    int cluster_id        = blockIdx.x / CLUSTER_SIZE;
    int num_tiles_total   = NUM_TILES_TOTAL;

    // ======== WARP 0: TMA PRODUCER (both CTAs) ========
    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = (stage_idx + 1) % NUM_STAGES_T;
            if (stage_idx == 0) phase ^= 1;
        };

        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            // Activation M base for this CTA (split across the 2 CTAs; single M block).
            int m_base = cta_rank * LOAD_BLOCK_M_T;

            // Stream the SUBTILES_PER_HEAD sub-tiles of this head through the same
            // SMEM load pipeline (activation A is re-loaded per sub-tile; it hits L2).
            for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                // Weight N base for this CTA within this head's sub-tile.
                int n_base = tile_id * HEAD_DIM + sub * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N;

                for (int k = 0; k < NUM_K_TILES; ++k) {
                    s.empty_barriers[stage_idx].wait(phase ^ 1);

                    int k_offset = k * BLOCK_K;
                    auto* smem_a_dst = reinterpret_cast<nv_bfloat16*>(
                        s.smem_a + stage_idx * SMEM_A_PER_STAGE_T);
                    auto* smem_b_dst = reinterpret_cast<nv_bfloat16*>(
                        s.smem_b + stage_idx * SMEM_B_PER_STAGE);

                    // activation (A): outer = m_base ; weight (B): outer = n_base
                    tma::copy_2sm_2d(&desc_A, &s.full_barriers[stage_idx],
                                     smem_a_dst, k_offset, m_base);
                    tma::copy_2sm_2d(&desc_B, &s.full_barriers[stage_idx],
                                     smem_b_dst, k_offset, n_base);

                    constexpr uint32_t kNumArrivalBytes = SMEM_A_PER_STAGE_T + SMEM_B_PER_STAGE;
                    if (is_leader) {
                        s.full_barriers[stage_idx].arrive_and_expect_tx(
                            kNumArrivalBytes * NUM_MULTICAST);
                    } else {
                        s.full_barriers[stage_idx].arrive(0u);
                    }

                    advance_pipeline();
                }
            }
        }
    }

    // ======== WARP 1: MMA CONSUMER (leader only) ========
    else if (warp_id == 1 && is_leader) {
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = (stage_idx + 1) % NUM_STAGES_T;
            if (stage_idx == 0) phase ^= 1;
        };

        uint32_t persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES_T;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES_T) & 1;

            s.tmem_empty_barriers[accum_stage].wait(accum_phase ^ 1);
            ptx::tcgen05_fence_after_sync();

            // [PROFILE] MMA warp: start of this head's issue window.
            long long prof_mma_t0 = 0;
            if (kProfile && cluster_id == 0 && lane_id == 0)
                prof_mma_t0 = ptx::rdclock();

            // Each head keeps SUBTILES_PER_HEAD independent accumulators resident.
            for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                uint32_t tmem_c = accum_stage * ACCUM_COLS_T + sub * UMMA_N_T;

                for (int k = 0; k < NUM_K_TILES; ++k) {
                    s.full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_fence_after_sync();

                    uint32_t w_base = __shfl_sync(0xffffffff, wgt_desc_lo, stage_idx);
                    uint32_t a_base = __shfl_sync(0xffffffff, act_desc_lo, stage_idx);

                    if (ptx::elect_one_sync()) {
                        #pragma unroll
                        for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                            uint32_t w_lo = mma_desc::advance_desc_lo_for_k(w_base, kk);
                            uint32_t a_lo = mma_desc::advance_desc_lo_for_k(a_base, kk);

                            uint64_t w_full = (static_cast<uint64_t>(wgt_desc.hi) << 32) | w_lo;
                            uint64_t a_full = (static_cast<uint64_t>(act_desc.hi) << 32) | a_lo;

                            // accum_flag resets per sub-tile (k resets to 0 each sub-tile).
                            uint32_t accum_flag = (k > 0 || kk > 0) ? 1 : 0;
                            // A-operand = weight (UMMA_M=256), B-operand = activation (UMMA_N=M)
                            ptx::tcgen05_mma_2sm(tmem_c, w_full, a_full,
                                                 runtime_idesc, accum_flag);
                        }
                    }
                    __syncwarp();

                    constexpr uint16_t CTA_MASK = (1 << NUM_MULTICAST) - 1;
                    ptx::umma_arrive_multicast_2sm(
                        reinterpret_cast<uint64_t*>(&s.empty_barriers[stage_idx]), CTA_MASK);
                    // Signal accumulators ready only after the head's LAST sub-tile's last k.
                    if (sub == SUBTILES_PER_HEAD - 1 && k == NUM_K_TILES - 1) {
                        ptx::umma_arrive_multicast_2sm(
                            reinterpret_cast<uint64_t*>(&s.tmem_full_barriers[accum_stage]),
                            CTA_MASK);
                    }
                    __syncwarp();

                    advance_pipeline();
                }
            }

            // [PROFILE] MMA warp: end of this head's issue window.
            if (kProfile && cluster_id == 0 && lane_id == 0) {
                prof[persistent_iter * 6 + 0] = prof_mma_t0;
                prof[persistent_iter * 6 + 1] = ptx::rdclock();
            }

            persistent_iter++;
        }

        if (persistent_iter > 0) {
            uint32_t last_iter        = persistent_iter - 1;
            uint32_t last_accum_stage = last_iter % NUM_EPI_STAGES_T;
            uint32_t last_accum_phase = (last_iter / NUM_EPI_STAGES_T) & 1;
            s.tmem_empty_barriers[last_accum_stage].wait(last_accum_phase);
        }
    }

    // ======== EPILOGUE WARPS (both CTAs, 128 threads / 4 warps) ========
    else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
             warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        uint32_t epi_warp_idx = warp_id - (NUM_NON_EPI_THREADS / 32);  // 0..3
        uint32_t tma_store_idx = 0;

        constexpr int NUM_STORES         = M_TPL / STORE_BLOCK_M;              // M/16
        constexpr int NUM_TMEM_SUBROWS   = STORE_BLOCK_M / 8;                  // 2
        constexpr int NUM_N_STORE_ATOMS  = STORE_BLOCK_N / STORE_BLOCK_N_ATOM; // 4
        constexpr int SMEM_CD_PER_STAGE_T = SMEM_CD_PER_STAGE;                 // 8192

        const uint32_t thread_in_wg = threadIdx.x - NUM_NON_EPI_THREADS;   // 0..127
        const uint32_t peer_rank    = cta_rank ^ 1u;

      // ================================================================
      // EPILOGUE (all M): 2-pass — PASS 1 reduces the head_dim sum-of-squares
      // (per-warp slots -> 4-warp combine -> cross-CTA DSMEM fold -> rsqrt),
      // PASS 2 re-reads TMEM, scales by rms, and does a merged store of both
      // sub-tiles under one barrier pair. (The former M<=32 single-read
      // register-staged fast path was removed: it benchmarked within noise of
      // 2-pass, so the extra code path wasn't worth keeping.)
      // ================================================================
      {
        uint32_t persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES_T;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES_T) & 1;

            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_fence_after_sync();

            // [PROFILE] Epilogue (leader CTA): start of this head's readback+store.
            long long prof_epi_t0 = 0;
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0)
                prof_epi_t0 = ptx::rdclock();

            // ============================================================
            // PASS 1: partial sum-of-squares over head_dim (per-warp slots, no
            // pre-zero / no atomicAdd) -> 4-warp combine -> cross-CTA fold -> rsqrt.
            // Only 2 CTA-wide barriers (was 4).
            // ============================================================
            for (int st = 0; st < NUM_STORES; ++st) {
                #pragma unroll
                for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                    float acc[8] = {0,0,0,0,0,0,0,0};
                    #pragma unroll
                    for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                        uint32_t tmem_addr = accum_stage * ACCUM_COLS_T + sub * UMMA_N_T
                                           + st * STORE_BLOCK_M + i * 8;
                        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                        ptx::tmem_load_32dp32b8x(tmem_addr, v0, v1, v2, v3, v4, v5, v6, v7);
                        cutlass::arch::fence_view_async_tmem_load();
                        uint32_t vv[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                        #pragma unroll
                        for (int row = 0; row < 8; ++row) {
                            float f = __uint_as_float(vv[row]);
                            acc[row] += f * f;
                        }
                    }
                    #pragma unroll
                    for (int row = 0; row < 8; ++row) {
                        float r = ptx::warp_reduce_sum32(acc[row]);      // over 32 N-lanes
                        if (lane_id == 0)
                            s.smem_warp_sq[epi_warp_idx][st * STORE_BLOCK_M + i * 8 + row] = r;
                    }
                }
            }
            cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);   // barrier #1: warp slots visible

            // [PROFILE] epilogue boundary: end of PASS 1 (TMEM read + sum-of-squares).
            long long prof_epi_t1 = 0;
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0)
                prof_epi_t1 = ptx::rdclock();

            // Cross-CTA fold via DSMEM (cute::store_shared_remote -> peer smem + txn barrier).
            // NOTE: no end-to-end DeepGEMM reference; validate with racecheck/synccheck.
            uint32_t bar_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&s.dsmem_barriers[accum_stage]));
            if (thread_in_wg == 0)
                s.dsmem_barriers[accum_stage].arrive_and_expect_tx(M_TPL * sizeof(float));
            for (uint32_t m = thread_in_wg; m < M_TPL; m += NUM_STORE_THREADS) {
                float local = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) local += s.smem_warp_sq[w][m];
                uint32_t dst = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&s.smem_peer_sq[accum_stage][m]));
                cute::store_shared_remote(__float_as_uint(local), dst, bar_addr, peer_rank);
            }
            s.dsmem_barriers[accum_stage].wait(accum_phase);
            for (uint32_t m = thread_in_wg; m < M_TPL; m += NUM_STORE_THREADS) {
                float local = 0.f;
                #pragma unroll
                for (int w = 0; w < 4; ++w) local += s.smem_warp_sq[w][m];
                float full = local + s.smem_peer_sq[accum_stage][m];
                s.smem_rms[m] = rsqrtf(full / float(HEAD_DIM) + eps);
            }
            cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);   // barrier #2: rms visible

            // [PROFILE] epilogue boundary: end of cross-CTA DSMEM fold + rms.
            long long prof_epi_t2 = 0;
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0)
                prof_epi_t2 = ptx::rdclock();

            // ============================================================
            // PASS 2: MERGED store — both sub-tiles written under ONE barrier pair
            // per st (store iters = NUM_STORES, not NUM_STORES*SUBTILES -> half the
            // epilogue barriers). Each store stage holds SUBTILES_PER_HEAD regions.
            // ============================================================
            for (int st = 0; st < NUM_STORES; ++st,
                 tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES) {
                auto* stage_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * (SUBTILES_PER_HEAD * SMEM_CD_PER_STAGE_T));

                if (epi_warp_idx == 0)
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                // Read both sub-tiles' TMEM, scale by rms, transpose into their SMEM regions.
                // BF16 store via stmatrix.trans (DeepGEMM sm100_store_cd_swap_ab bf16 path):
                // one hardware-transposed matrix store replaces 8 scattered st.shared.u16.
                #pragma unroll
                for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                    uint32_t tmem_base = accum_stage * ACCUM_COLS_T + sub * UMMA_N_T;
                    uint8_t* smem_cd_ptr = stage_ptr + sub * SMEM_CD_PER_STAGE_T;
                    #pragma unroll
                    for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                        uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * 8;

                        // 16dp256b layout required by stmatrix.trans: 2 loads -> 8 fp32/lane.
                        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                        ptx::tmem_load_16dp256b1x(tmem_addr,                v0, v1, v2, v3);
                        ptx::tmem_load_16dp256b1x(tmem_addr | 0x00100000u, v4, v5, v6, v7);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Per-M-row rms scale in fp32. Each lane's 8 values map to just
                        // two M-rows: m0 = mbase + 2*(lane%4), m1 = m0 + 1; even value
                        // indices -> m0, odd -> m1 (from the 16dp256b DstLayout).
                        int mbase   = st * STORE_BLOCK_M + i * 8;
                        float rms0  = s.smem_rms[mbase + 2 * (lane_id & 3u) + 0];
                        float rms1  = s.smem_rms[mbase + 2 * (lane_id & 3u) + 1];
                        __nv_bfloat162 b0 = __float22bfloat162_rn(
                            make_float2(__uint_as_float(v0) * rms0, __uint_as_float(v1) * rms1));
                        __nv_bfloat162 b1 = __float22bfloat162_rn(
                            make_float2(__uint_as_float(v2) * rms0, __uint_as_float(v3) * rms1));
                        __nv_bfloat162 b2 = __float22bfloat162_rn(
                            make_float2(__uint_as_float(v4) * rms0, __uint_as_float(v5) * rms1));
                        __nv_bfloat162 b3 = __float22bfloat162_rn(
                            make_float2(__uint_as_float(v6) * rms0, __uint_as_float(v7) * rms1));

                        // Destination in the 128B-swizzled bf16 SMEM tile (2 warps/atom,
                        // 2 atoms of 64 N per sub-tile). Matches the TMA store box below.
                        uint32_t row = lane_id & 7u;                          // 0..7 (M within 8)
                        uint32_t col = (epi_warp_idx & 1u) * 4 + (lane_id >> 3); // 0..7 (N group)
                        uint8_t* smem_base_ptr = smem_cd_ptr
                            + (epi_warp_idx >> 1) * (STORE_BLOCK_M * SWIZZLE_CD) // outer atom
                            + i * (8 * SWIZZLE_CD);                             // inner 8 rows
                        uint8_t* dst = smem_base_ptr
                            + row * (16 * 8)               // kNumBankGroupBytes(16) * 8
                            + ((col ^ row) * 16);
                        ptx::stmatrix_x4_trans(dst,
                            *reinterpret_cast<uint32_t*>(&b0),
                            *reinterpret_cast<uint32_t*>(&b1),
                            *reinterpret_cast<uint32_t*>(&b2),
                            *reinterpret_cast<uint32_t*>(&b3));
                    }
                }

                // Free accumulators after the last st (all sub-tiles now read out).
                if (st == NUM_STORES - 1) {
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                if (epi_warp_idx == 0 && ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                        uint8_t* smem_cd_ptr = stage_ptr + sub * SMEM_CD_PER_STAGE_T;
                        int base_n = tile_id * HEAD_DIM + sub * CLUSTER_BLOCK_N + cta_rank * BLOCK_N;
                        #pragma unroll
                        for (int i = 0; i < NUM_N_STORE_ATOMS; ++i) {
                            auto* smem_ptr = reinterpret_cast<nv_bfloat16*>(smem_cd_ptr)
                                + i * (STORE_BLOCK_M * STORE_BLOCK_N_ATOM);
                            int n_idx = base_n + i * STORE_BLOCK_N_ATOM;
                            int m_idx = st * STORE_BLOCK_M;
                            tma::store_2d(&desc_D, smem_ptr, n_idx, m_idx);
                        }
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

            // [PROFILE] Epilogue (leader CTA): sub-phase timeline of this head.
            //   col2 epi_start | col3 after PASS1 | col4 after DSMEM fold+rms | col5 epi end
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0) {
                prof[persistent_iter * 6 + 2] = prof_epi_t0;
                prof[persistent_iter * 6 + 3] = prof_epi_t1;
                prof[persistent_iter * 6 + 4] = prof_epi_t2;
                prof[persistent_iter * 6 + 5] = ptx::rdclock();
            }

            persistent_iter++;
        }
      }  // end epilogue
    }

    // ================================================================
    // CLEANUP
    // ================================================================
    ptx::cluster_sync();
    if (warp_id == 0) {
        ptx::tcgen05_dealloc_2sm(0, NUM_TMEM_COLS_T);
    }
}

// ======================== Host: TMA Descriptors ========================
static CUtensorMap make_tma_desc_bf16_2d(
    const nv_bfloat16* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(nv_bfloat16)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
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
template <bool kProfile>
static void* get_kernel_ptr(int M) {
    switch (M) {
        case 32:  return (void*)&wq_b_proj_kernel<32,  kProfile>;
        case 64:  return (void*)&wq_b_proj_kernel<64,  kProfile>;
        case 96:  return (void*)&wq_b_proj_kernel<96,  kProfile>;
        case 128: return (void*)&wq_b_proj_kernel<128, kProfile>;
        case 160: return (void*)&wq_b_proj_kernel<160, kProfile>;
        case 192: return (void*)&wq_b_proj_kernel<192, kProfile>;
        case 224: return (void*)&wq_b_proj_kernel<224, kProfile>;
        case 256: return (void*)&wq_b_proj_kernel<256, kProfile>;
        default:  return nullptr;
    }
}

static int get_smem_bytes(int M) {
    switch (M) {
        case 32:  return (int)sizeof(SharedStorage<32>);
        case 64:  return (int)sizeof(SharedStorage<64>);
        case 96:  return (int)sizeof(SharedStorage<96>);
        case 128: return (int)sizeof(SharedStorage<128>);
        case 160: return (int)sizeof(SharedStorage<160>);
        case 192: return (int)sizeof(SharedStorage<192>);
        case 224: return (int)sizeof(SharedStorage<224>);
        case 256: return (int)sizeof(SharedStorage<256>);
        default:  return 0;
    }
}

// Common launch. When `profile` is true, allocates a [max_iters, 4] int64 timing
// tensor and returns it as the 2nd element. Each row is one persistent head-
// iteration of cluster 0's leader CTA:
//   col 0 = MMA warp start   (clock64)
//   col 1 = MMA warp end
//   col 2 = epilogue start
//   col 3 = epilogue end
// MMA and epilogue here are on the SAME CTA (same SM) -> clock64 directly comparable.
static std::vector<torch::Tensor> run_wq_b(
    torch::Tensor x, torch::Tensor w, float eps, bool profile)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);

    const int M = x.size(0);
    TORCH_CHECK(x.size(1) == K_DIM);
    TORCH_CHECK(w.size(0) == N_TOTAL && w.size(1) == K_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0);

    // swap-AB: BLOCK_M = M exactly (single M block), no padding.
    // Output is BF16 (model-faithful: DSV4 wq_b is a bf16 Linear -> q is bf16).
    auto out = torch::empty({M, N_TOTAL}, x.options().dtype(torch::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_ptr   = reinterpret_cast<const nv_bfloat16*>(x.data_ptr());
    auto w_ptr   = reinterpret_cast<const nv_bfloat16*>(w.data_ptr());
    auto out_ptr = reinterpret_cast<nv_bfloat16*>(out.data_ptr());

    // TMA descriptors
    const int load_block_m = M / NUM_MULTICAST; // activation rows per CTA
    CUtensorMap desc_A = make_tma_desc_bf16_2d(x_ptr, M, K_DIM, load_block_m, BLOCK_K);
    CUtensorMap desc_B = make_tma_desc_bf16_2d(w_ptr, N_TOTAL, K_DIM, LOAD_BLOCK_N, BLOCK_K);
    // Output D (BF16): box = STORE_BLOCK_M (M rows) x STORE_BLOCK_N_ATOM (=64 bf16), 128B swizzle
    CUtensorMap desc_D = make_tma_desc_bf16_2d(out_ptr, M, N_TOTAL, STORE_BLOCK_M, STORE_BLOCK_N_ATOM);

    // Grid: persistent, cluster of 2 CTAs.
    static const int num_SMs = []() {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
        return n;
    }();
    // One cluster processes a whole head (fused RMSNorm), so scheduling unit = head.
    int total_cta = NUM_HEAD_TILES * CLUSTER_SIZE;   // max useful CTAs
    int grid_size = min(num_SMs, total_cta);
    grid_size = (grid_size / CLUSTER_SIZE) * CLUSTER_SIZE;  // multiple of cluster size
    if (grid_size < CLUSTER_SIZE) grid_size = CLUSTER_SIZE;

    const int num_clusters = grid_size / CLUSTER_SIZE;
    const int max_iters = (NUM_HEAD_TILES + num_clusters - 1) / num_clusters;

    // Profiling timing buffer (only cluster 0's leader CTA writes into it).
    int64_t* prof_dev = nullptr;
    torch::Tensor timing;
    if (profile) {
        timing = torch::zeros({max_iters, 6}, x.options().dtype(torch::kInt64));
        prof_dev = reinterpret_cast<int64_t*>(timing.data_ptr());
    }

    void* kernel_ptr = profile ? get_kernel_ptr<true>(M) : get_kernel_ptr<false>(M);
    int smem_bytes = get_smem_bytes(M);
    TORCH_CHECK(kernel_ptr != nullptr && smem_bytes > 0, "Unsupported M=", M);

    // Configure max dynamic SMEM once per (profile, M) variant.
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

        void* ptr_args[] = { &desc_A, &desc_B, &desc_D, &grid_size, &eps, &prof_dev };
        auto err = cudaLaunchKernelExC(&config, kernel_ptr, ptr_args);
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    }

    if (profile) return {out, timing};
    return {out};
}

// ======================== PyTorch Binding ========================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps)
{
    (void)rms_w;   // weightless RMSNorm: rms_w kept only for API parity
    return run_wq_b(x, w, static_cast<float>(eps), /*profile=*/false)[0];
}

// Profiled variant: returns (out, timing[max_iters, 4]) with clock64 timestamps
// of the MMA / epilogue windows on cluster 0's leader CTA.
std::vector<torch::Tensor> wq_b_proj_gemm_profiled(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps)
{
    (void)rms_w;
    return run_wq_b(x, w, static_cast<float>(eps), /*profile=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj + fused per-head RMSNorm (tcgen05 2SM MMA, swap-AB, Blackwell)");
    m.def("wq_b_proj_gemm_profiled", &wq_b_proj_gemm_profiled,
          "wq_b proj + fused RMSNorm + clock64 MMA/epilogue overlap timing -> (out, timing[max_iters,4])");
}
