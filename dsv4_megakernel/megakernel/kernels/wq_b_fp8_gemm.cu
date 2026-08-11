// ============================================================
// wq_b_fp8_gemm.cu — MERGED wq_b projection (tcgen05 FP8 block-scale GEMM)
//
// x_fp8[M,K_DIM] @ w_fp8[N_MERGED,K_DIM]^T   (indexer rows before main Q)
//   -> y [M,N_TOTAL] bf16 + ssq [M,NUM_HEADS_OUT] fp32
//   -> indexer q: by default DRAINED as bf16 iq_ws [M,64,128] (finish it with
//      idx_postprocess); mock_post=False fuses either RTP RoPE+FP8+weight-fold
//      or the legacy RoPE+Hadamard+MXFP4 transform in-kernel.
//
// Usage (default: 256 threads, GEMM + ssq):
//   y, iq_fp4, iq_sf, ssq, iq_ws = wq_b_proj_gemm_merged(
//       x_fp8, x_sf [M,12] ue8m0, w_fp8, w_sf [N_MERGED/128,12] ue8m0,
//       q_pos [M] i32, rope_cos, rope_sin [max_pos,32] f32)
//   iq_fp4/iq_sf are GARBAGE in this mode -- run idx_postprocess(iq_ws, ...).
//   M in [1,128] dispatches to a 32-row template; TMA handles OOB rows. Optional
//   kwargs: head_ssq (caller zero-init buffer), enable_ssq, mock_post. Return order:
//   [y, iq_fp4, iq_sf, ssq?, iq_ws?, idx_q4?, idx_s4?, timing?] per the flags.
//
// Fused indexer (winkv) COMPRESSOR (optional, delivery op-B-tail port, NO
// split-K): pass cmp_pos/idx_norm/cos_tab/sin_tab/idx_state (fresh state
// row arrives from front's FRONT-EMIT epilogue, +idx_ape)
// together and warps 8..15 run the state read + compress chain fully decoupled
// from the GEMM. In RTP FP8 mode it writes 128 E4M3 bytes + one FP32 scale;
// the explicit legacy mode retains FWHT + FP4.
// LOCAL KV WINDOW (optional, CSA stage 4 FULL chain): pass win_y2 [M,512] +
// win_norm [512] (+ cmp_pos/cos_tab/sin_tab) -> RMSNorm + RoPE + per-64 fp8;
// appends win_q8 [M,448] u8 + win_s8 [M,7] f32 + win_rope [M,64] bf16. Chains
// in idx_comp_fp4.cuh (warp-level, one row per warp on warps 8..15).
//
// Swap-AB (UMMA_N = M_pad, BN=128), 2SM MMA, persistent, warp-specialized;
// mock_post=False adds 256 async transform workers (512-thread CTA). Config in
// wq_b_fp8_gemm.cuh; MMA engine in cluster_mma_fp8.cuh; indexer chain in
// idx_post_fp4.cuh.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdlib>   // getenv / atoi (WQ_B_CLUSTERS experiment knob)

#include "wq_b_fp8_gemm.cuh"
#include "idx_post_fp4.cuh"   // shared indexer post-processing chain + standalone kernel
#include "idx_comp_fp4.cuh"   // fused indexer compressor chain (delivery port, 128-thread)
#include "cluster_mma_fp8.cuh"   // FP8 block-scale MMA engine (leader MMA warp)

using namespace wq_b;
using Barrier = mma_desc::Barrier;

// Devices we keep an iq ready-flag page for (see the handoff block in run_wq_b).
static constexpr int kMaxDevices = 16;

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
    alignas(128)  uint8_t smem_sfa[NS * SSFA];                // expanded K32 activation SF
    alignas(128)  uint8_t smem_sfb[NS * SMEM_SFB_PER_STAGE];  // expanded K32 weight SF

    // Barriers
    alignas(16) Barrier full_barriers[NS];            // A/B TMA done (per-CTA, init 1)
    alignas(16) Barrier empty_barriers[NS];           // stage smem reusable
    alignas(16) Barrier with_sf_full_barriers[NS];    // expanded SF ready for UTCCP
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES];

    // head_ssq per-warp scratch (RMSNorm scale folding): warp w's partial
    // sum-of-squares of row r over its 32 columns; folded + RED'd at tile end.
    alignas(16) float ssq_scratch[4][BM];

    // TMEM base address
    alignas(16) uint32_t tmem_base;
};

// Compile-time guard: SwapDims::NUM_STAGES comes from an OVERHEAD ESTIMATE in
// wq_b_fp8_gemm.cuh that must track the members above (incl. the 2KB ssq_scratch:
// the `+ 4 * BM * sizeof(float)` term). Any skew -- e.g. a stale header on the
// build box -- otherwise only surfaces at RUNTIME as cudaFuncSetAttribute
// "invalid argument" (observed: headerless-ssq NS=11 + scratch = 233472 > 232448).
static_assert(sizeof(SharedStorage<32>)  <= wq_b::SMEM_CAPACITY &&
              sizeof(SharedStorage<64>)  <= wq_b::SMEM_CAPACITY &&
              sizeof(SharedStorage<96>)  <= wq_b::SMEM_CAPACITY &&
              sizeof(SharedStorage<128>) <= wq_b::SMEM_CAPACITY,
              "SharedStorage exceeds SM100 smem capacity: SwapDims::SMEM_OVERHEAD "
              "(wq_b_fp8_gemm.cuh) is out of sync with the SharedStorage members -- "
              "is the header stale (missing the ssq_scratch overhead term)?");

// ======================== Standalone head-ssq kernel ========================
// The separate-kernel fallback for callers that disable fused head_ssq. The
// materialized main-Q output is BF16, so this necessarily measures the rounded
// values; the fused path below retains the more accurate FP32 accumulators.
__global__ void __launch_bounds__(256, 1)
head_ssq_kernel(
    const __nv_bfloat16* __restrict__ y, // [M, N_TOTAL] bf16
    float* __restrict__ ssq,             // [M, NUM_HEADS_OUT]
    int total_heads)                     // M * NUM_HEADS_OUT
{
    const int h = blockIdx.x * 8 + (int)(threadIdx.x >> 5);
    if (h >= total_heads)
        return;                                                // warp-uniform
    const uint32_t lane = threadIdx.x & 31;
    const __nv_bfloat16* p = y + (int64_t)h * HEAD_DIM_OUT + lane * 16;
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const float v = __bfloat162float(p[k]);
        s = fmaf(v, v, s);
    }
    #pragma unroll
    for (int x = 16; x >= 1; x >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, x);
    if (lane == 0)
        ssq[h] = s;
}

// ======================== Activation quant kernel (PDL producer) ========
// rmsnorm(gamma) + 1x128 quant (or plain quant when gamma == nullptr) of the
// bf16 activation, launched on the SAME stream right before the merged GEMM
// with PDL (DeepGEMM discipline). The GEMM's prologue (barrier init / TMEM
// alloc / descriptor prefetch) overlaps this kernel; its
// cudaGridDependencySynchronize -- already after that prologue -- provides the
// cross-kernel ordering the old in-kernel grid ticket used to (whose sync
// floor was ~1.5-2us on the activation-TMA critical path).
// 192 threads = 6 warps, 2 K128 blocks each per 1536-wide row.
//
// One warp per row was tried and is much WORSE (d_q+norm 2.8 -> 7.9us at M=128,
// 18us at M=1): a warp then owns all 12 blocks, so its 12 five-step shfl trees
// serialise instead of running 2-per-warp across 6 warps. See
// qnorm_quant_row_cta.
__global__ void __launch_bounds__(192, 1)
qnorm_quant_kernel(const __nv_bfloat16* __restrict__ y, int64_t lda,
                   const float* __restrict__ gamma, float eps,
                   __nv_fp8_e4m3* __restrict__ x_fp8,
                   uint8_t* __restrict__ x_sf, int m_total,
                   float* __restrict__ ssq_zero,
                   uint32_t* __restrict__ iq_drain_ready_reset) {
    // Let WQ_B run its independent prologue while this grid produces inputs;
    // its GDS precedes every global-memory access to those inputs.
    asm volatile("griddepcontrol.launch_dependents;");
    __shared__ float ssq_smem[NUM_K_TILES];
    const int warp_id = (int)(threadIdx.x / 32);
    const int lane_id = (int)(threadIdx.x % 32);
    // Fold the head_ssq RED-buffer zeroing in here (glue removal): the
    // dependent GEMM's GDS orders its RED atomics after our completion.
    if (ssq_zero != nullptr) {
        const int tot = m_total * NUM_HEADS_OUT;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tot;
             i += gridDim.x * blockDim.x)
            ssq_zero[i] = 0.0f;
    }
    if (iq_drain_ready_reset != nullptr && blockIdx.x == 0) {
        for (int head = threadIdx.x; head < IDX_NUM_HEADS; head += blockDim.x)
            iq_drain_ready_reset[head * IQ_FLAG_STRIDE] = 0u;
    }
    if (gamma != nullptr) {
        for (int r = blockIdx.x; r < m_total; r += gridDim.x)
            qnorm_quant_row_cta(y + (size_t)r * lda, gamma, eps, ssq_smem,
                                warp_id, lane_id, /*nwarps=*/6,
                                /*bar_threads=*/192, /*bar_id=*/0,
                                x_fp8 + (size_t)r * K_DIM,
                                x_sf + r * NUM_K_TILES);
    } else {
        const int gwarp = blockIdx.x * 6 + warp_id;
        const int gwarps = gridDim.x * 6;
        const int nqb = m_total * NUM_K_TILES;
        for (int qb = gwarp; qb < nqb; qb += gwarps) {
            const int m = qb / NUM_K_TILES, b = qb % NUM_K_TILES;
            quant_k128_ue8m0(y + (size_t)m * lda + b * BLOCK_K + lane_id * 4,
                             lane_id,
                             x_fp8 + (size_t)m * K_DIM + b * BLOCK_K,
                             x_sf + m * NUM_K_TILES + b);
        }
    }
}

// ======================== Kernel ========================
// The indexer weight precedes main Q, so its transform overlaps the remaining
// main-Q weight stream. Store warps drain indexer tiles before async workers
// post-process them.
template <int M_TPL, bool kProfile, bool kSsq, bool kIdxFp8>
__global__ void __launch_bounds__(TPB_IDX, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,    // activation [M,K] e4m3, K-major
    const __grid_constant__ CUtensorMap desc_B,    // weight     [N,K] e4m3, K-major
    const uint8_t* __restrict__ x_sf,               // [M,K/128] UE8M0
    const uint8_t* __restrict__ w_sf,               // [N/128,K/128] UE8M0
    const __grid_constant__ CUtensorMap desc_D,    // output [M,N] BF16 row-major
    int problem_m,
    int num_blocks,                                // physical grid size (CTAs)
    int num_gemm_clusters,                         // clusters assigned GEMM tiles
    float* __restrict__ head_ssq,                  // [M,NUM_HEADS_OUT] FP32; nullptr disables
    // fused indexer projection outputs + rotary metadata
    const __grid_constant__ IqDest iqd,            // indexer-q destination geometry
    __nv_bfloat16* __restrict__ iq_scratch,        // [M, 64, 128] bf16 drain buffer
    uint32_t* __restrict__ iq_drain_ready,         // [IQ_FLAG_SLOTS] per-head ready flags
    uint32_t iq_drain_seq,                         // host-monotonic launch tag for them
    const int* __restrict__ q_pos,                 // [M] rotary positions
    const float* __restrict__ rope_cos,           // [max_pos, 32]
    const float* __restrict__ rope_sin,           // [max_pos, 32]
    int mock_post,                                 // 1 = drain only, skip the transform (baseline)
    const __grid_constant__ idx_comp::Args comp,   // fused indexer compressor (kv==nullptr => off)
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

    constexpr int NUM_TILES_TOTAL = NUM_N_TILES_MERGED; // N tiles (inner loop over M subtiles)

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
        for (int i = 0; i < NS; ++i) {
            s.full_barriers[i].init(1);                        // per-CTA A/B TMA
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
    int num_clusters    = num_gemm_clusters;
    int cluster_id      = blockIdx.x / CLUSTER_SIZE;
    int num_tiles_total = NUM_TILES_TOTAL;
    // The full physical grid gives the CUDA-core transform one CTA per SM. Only
    // the first num_gemm_clusters clusters own GEMM tiles; the rest skip ahead
    // to the transform phase.
    const int gemm_begin =
        cluster_id < num_gemm_clusters ? cluster_id : num_tiles_total;

    // ======== WARP 0: TMA PRODUCER (both CTAs, plain per-CTA loads) ========
    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        auto advance = [&]() { stage = (stage + 1) % NS; if (stage == 0) phase ^= 1; };

        // The iq segment is the FIRST N_IDX columns of the weight, so a plain
        // forward tile walk already puts the tiles that need post-processing in
        // iteration 0 -- their drain + async transform then overlap the remaining
        // main tiles' weight stream. All roles share this mapping.
        for (int it_t = gemm_begin; it_t < num_tiles_total; it_t += num_clusters) {
            int n_base = it_t * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N; // weight N (per CTA)
            // Inner loop over M subtiles: weight (n_base) is reused from L2.
            for (int m_sub = 0; m_sub < NUM_M_SUB; ++m_sub) {
                int m_base = m_sub * BM_T + cta_rank * LOAD_BLOCK_M_T;  // this subtile's activation half

                // [PROFILE] Load (producer) window for this iteration's K-loop.
                long long prof_ld_t0 = 0;
                if (kProfile && cluster_id == 0 && cta_rank == 0)
                    prof_ld_t0 = ptx::rdclock();

                for (int k = 0; k < NUM_K_TILES; ++k) {
                    s.empty_barriers[stage].wait(phase ^ 1);
                    int k_off = k * BLOCK_K;

                    auto* sa  = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a + stage * SA);
                    auto* sb  = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b + stage * SMEM_B_PER_STAGE);
                    tma::copy_2d_fp8(&desc_B, &s.full_barriers[stage], sb, k_off, n_base);
                    tma::copy_2d_fp8(&desc_A, &s.full_barriers[stage], sa, k_off, m_base);

                    constexpr uint32_t kNumArrivalBytes = SA + SMEM_B_PER_STAGE;
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

    // ======== WARP 2: NATIVE SF PACKER (both CTAs) ========
    // DeepGEMM SF amortization: the native layouts x_sf[M, K/128] / w_sf[N/128,
    // K/128] keep a row's K scales CONTIGUOUS, so the 4 bytes of one SF group are
    // ONE naturally-aligned u32 (row stride 12 % 4 == 0) -- a single coalesced LDG
    // replaces the old per-stage byte read + x0x01010101 replication. Only the
    // GROUP-START stage (k % 4 == 0) fills smem + UTCCPs; the other stages just
    // relay the barrier. sf_id in the MMA selects the byte within the group.
    else if (warp_id == 2) {
        uint32_t stage = 0, phase = 0;
        auto advance = [&]() { stage = (stage + 1) % NS; if (stage == 0) phase ^= 1; };

        // iq segment first (see the producer): plain forward walk.
        for (int it_t = gemm_begin; it_t < num_tiles_total; it_t += num_clusters) {
          const int n_base = it_t * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N;
          for (int m_sub = 0; m_sub < NUM_M_SUB; ++m_sub) {
            const int sfa_m = m_sub * BM_T;
            for (int k = 0; k < NUM_K_TILES; ++k) {
                if (k % 4 == 0) {
                    // One u32 per row = this group's 4 consecutive K128 exponents.
                    uint32_t va[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = sfa_m + i * 32 + lane_id;
                        va[i] = row < problem_m
                            ? *reinterpret_cast<const uint32_t*>(x_sf + row * NUM_K_TILES + k)
                            : UE8M0_ONE * 0x01010101u;
                    }
                    uint32_t vb = UE8M0_ONE * 0x01010101u;
                    if (lane_id == 0)
                        vb = *reinterpret_cast<const uint32_t*>(
                            w_sf + (n_base / WEIGHT_QUANT_BLOCK_N) * NUM_K_TILES + k);
                    vb = __shfl_sync(0xffffffffu, vb, 0);

                    s.full_barriers[stage].wait(phase);
                    auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage * SSFA);
                    auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage * SMEM_SFB_PER_STAGE);
                    ptx::st_shared_v4_u32(sfa + lane_id * 4, va[0], va[1], va[2], va[3]);
                    ptx::st_shared_v4_u32(sfb + lane_id * 4, vb, vb, vb, vb);
                    cutlass::arch::fence_view_async_shared();
                } else {
                    s.full_barriers[stage].wait(phase);   // operands only this stage
                }
                s.with_sf_full_barriers[stage].arrive(0u);
                advance();
            }
          }
        }
    }

    // ======== WARP 1: MMA CONSUMER (leader only) ========
    // Per-tile UTCCP + block_scale MMA delegated to cluster_mma_fp8 (UMMA_N = M).
    else if (warp_id == 1 && is_leader) {
        using CM = cluster_mma_fp8::ClusterMmaFP8BlockScale<BLOCK_K, NS, UMMA_N_T, BLOCK_N, true>;
        static_assert(UMMA_N_T == CM::UMMA_N, "swap-AB requires UMMA_N == problem M");

        auto ds = CM::init_desc(smem_a_base, smem_b_base, lane_id);

        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        // Same forward mapping as the other roles; the MMA warp itself is tile-id
        // agnostic -- it only paces stages/accums.
        for (int it_t = gemm_begin; it_t < num_tiles_total; it_t += num_clusters) {
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

    // ======== EPILOGUE WARPS (both CTAs, 128 threads / 4 warps) ========
    else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
             warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        uint32_t epi_warp_idx = warp_id - (NUM_NON_EPI_THREADS / 32);  // 0..3
        uint32_t tma_store_idx = 0;

        constexpr int NUM_STORES         = UMMA_N_T / STORE_BLOCK_M;          // M/16 (no padding rows)
        constexpr int NUM_TMEM_SUBROWS   = STORE_BLOCK_M / 8;                  // 2
        constexpr int NUM_N_STORE_ATOMS  = STORE_BLOCK_N / STORE_BLOCK_N_ATOM; // 4
        constexpr int SMEM_CD_PER_STAGE_T = SMEM_CD_PER_STAGE;                 // 4096

        uint32_t persistent_iter = 0;
        // iq segment first (see the producer): plain forward walk.
        for (int it_t = gemm_begin; it_t < num_tiles_total; it_t += num_clusters) {
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
            int base_n = it_t * CLUSTER_BLOCK_N + cta_rank * BLOCK_N;
            int base_m = m_sub * BM_T;   // this subtile's output rows (TMA clips >= M)
            // Indexer tile: whole CTA tile is ONE indexer head row-block
            // (BLOCK_N == IDX_HEAD_DIM); drained + post-processed async, no fp32
            // TMA store, no ssq, smem_cd untouched. The iq segment leads the N
            // range, so main-q columns are offset by N_IDX in the output.
            const bool is_idx = base_n < N_IDX;
            const int  out_n  = base_n - N_IDX;   // main-q column in y / ssq

            if (is_idx) {
                    // DRAIN-FIRST: nothing heavy on the TMEM-drain path (the accum
                    // stage gates the MMA). The transform immediately rounds every
                    // value to bf16, so draining bf16 is bitwise equivalent while
                    // halving the L2 scratch round trip.
                    // tmem_empty, hand off to the async transform warpgroup.
                    const int head = base_n / IDX_HEAD_DIM;
                    __nv_bfloat16* dst = iq_scratch
                        + ((int64_t)base_m * IDX_NUM_HEADS + head) * IDX_HEAD_DIM
                        + epi_warp_idx * 32 + lane_id;
                    #pragma unroll
                    for (int st = 0; st < NUM_STORES; ++st) {
                        #pragma unroll
                        for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                            uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * 8;
                            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                            ptx::tmem_load_32dp32b8x(tmem_addr, v0, v1, v2, v3, v4, v5, v6, v7);
                            cutlass::arch::fence_view_async_tmem_load();
                            uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                            __nv_bfloat16* drow =
                                dst + (int64_t)(st * STORE_BLOCK_M + i * 8) * N_IDX;
                            #pragma unroll
                            for (int row = 0; row < 8; ++row)
                                if (st * STORE_BLOCK_M + i * 8 + row < problem_m)
                                    drow[(int64_t)row * N_IDX] =
                                        __float2bfloat16_rn(__uint_as_float(vals[row]));
                        }
                    }
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);

                    // Handoff: bar.sync (id 1, store + xform threads) is both the
                    // ready signal and the CTA-scope memory fence for the scratch
                    // writes. mock_post keeps the drain but skips the handoff
                    // (the "merged GEMM only" baseline).
                    if (!mock_post)
                        cutlass::arch::NamedBarrier::sync(
                            NUM_STORE_THREADS + NUM_XFORM_THREADS, 1);
            } else {

            for (int st = 0; st < NUM_STORES; ++st,
                 tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES) {
                auto* smem_cd_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * SMEM_CD_PER_STAGE_T);

                if (epi_warp_idx == 0)
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                // Read FP32 accumulators from TMEM. The output store rounds them
                // once to BF16; SSQ, when enabled, still uses these FP32 values.
                #pragma unroll
                for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                    uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * 8;

                    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                    ptx::tmem_load_16dp256b1x(tmem_addr, v0, v1, v2, v3);
                    ptx::tmem_load_16dp256b1x(
                        tmem_addr | 0x00100000u, v4, v5, v6, v7);
                    cutlass::arch::fence_view_async_tmem_load();

                    const float f0 = __uint_as_float(v0);
                    const float f1 = __uint_as_float(v1);
                    const float f2 = __uint_as_float(v2);
                    const float f3 = __uint_as_float(v3);
                    const float f4 = __uint_as_float(v4);
                    const float f5 = __uint_as_float(v5);
                    const float f6 = __uint_as_float(v6);
                    const float f7 = __uint_as_float(v7);

                    const __nv_bfloat162 b0 = __floats2bfloat162_rn(f0, f1);
                    const __nv_bfloat162 b1 = __floats2bfloat162_rn(f2, f3);
                    const __nv_bfloat162 b2 = __floats2bfloat162_rn(f4, f5);
                    const __nv_bfloat162 b3 = __floats2bfloat162_rn(f6, f7);
                    const uint32_t row = lane_id & 7u;
                    const uint32_t col = (epi_warp_idx & 1u) * 4u + (lane_id >> 3);
                    uint8_t* smem_base_ptr = smem_cd_ptr
                        + (epi_warp_idx >> 1) * (STORE_BLOCK_M * SWIZZLE_CD)
                        + i * (8 * SWIZZLE_CD);
                    uint8_t* dst = smem_base_ptr
                        + row * (16 * 8) + ((col ^ row) * 16);
                    ptx::stmatrix_x4_trans(
                        dst,
                        *reinterpret_cast<const uint32_t*>(&b0),
                        *reinterpret_cast<const uint32_t*>(&b1),
                        *reinterpret_cast<const uint32_t*>(&b2),
                        *reinterpret_cast<const uint32_t*>(&b3));

                    if constexpr (kSsq) {
                        // 16dp256b maps lane (a + 4*b) to rows (2*a,2*a+1)
                        // and columns (b,b+8,b+16,b+24).
                        const uint32_t row0 = (lane_id & 3u) * 2u;
                        const uint32_t row1 = row0 + 1u;

                        // Each lane owns four columns for each of two rows. Reduce
                        // across the eight lanes with the same `a`: 6 shuffles for
                        // 8 row sums, versus 16 in the 32dp32b transpose reduction.
                        float sq0 = fmaf(f6, f6, fmaf(f4, f4, fmaf(f2, f2, f0 * f0)));
                        float sq1 = fmaf(f7, f7, fmaf(f5, f5, fmaf(f3, f3, f1 * f1)));
                        #pragma unroll
                        for (int off = 4; off <= 16; off *= 2) {
                            sq0 += __shfl_xor_sync(0xffffffffu, sq0, off);
                            sq1 += __shfl_xor_sync(0xffffffffu, sq1, off);
                        }
                        if (lane_id < 4) {
                            s.ssq_scratch[epi_warp_idx]
                                [st * STORE_BLOCK_M + i * 8 + row0] = sq0;
                            s.ssq_scratch[epi_warp_idx]
                                [st * STORE_BLOCK_M + i * 8 + row1] = sq1;
                        }
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
                        auto* smem_ptr = reinterpret_cast<__nv_bfloat16*>(smem_cd_ptr)
                            + i * (STORE_BLOCK_M * STORE_BLOCK_N_ATOM);
                        int n_idx = out_n + i * STORE_BLOCK_N_ATOM;
                        int m_idx = base_m + st * STORE_BLOCK_M;
                        tma::store_2d(&desc_D, smem_ptr, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

            // Tile-end fold: thread r sums the 4 warp partials of row r and issues
            // ONE fire-and-forget RED (128 per CTA-tile, no intra-CTA same-address
            // collisions). Ordering is free: the last st's NamedBarrier above made
            // all scratch writes visible, and the next tile's first NamedBarrier
            // orders scratch reuse.
            if constexpr (kSsq) {
                const int r = (int)threadIdx.x - NUM_NON_EPI_THREADS;   // 0..127
                const int m = base_m + r;
                const float v = s.ssq_scratch[0][r] + s.ssq_scratch[1][r]
                              + s.ssq_scratch[2][r] + s.ssq_scratch[3][r];
                if (m < problem_m) {
                    float* dst = head_ssq
                        + (size_t)m * NUM_HEADS_OUT + out_n / HEAD_DIM_OUT;
                    // No return value is needed; RED avoids waiting for the global
                    // atomic round trip. Kernel completion provides visibility.
                    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;"
                                 :: "l"(dst), "f"(v) : "memory");
                }
            }

            }  // !is_idx (main-q tile store path)

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

    // ======== ASYNC TRANSFORM WORKERS (threads 256..511) ========
    // Three sections, ordered by EARLIEST POSSIBLE START:
    //   1 HANDSHAKE  serve barrier 1 (the store warps are BLOCKED on it right
    //                after each iq drain) and release that head's ready flag.
    //   2 CHAINS     winkv + compressor -- no in-kernel dependency at all, so as
    //                early as possible.
    //   3 SPREAD     the rope/hadamard/fp4 transform. The only section that can
    //                block on ANOTHER CTA's flag, hence last.
    // Only index-tile CTAs drain scratch; SPREAD distributes the transform over all
    // resident CTAs.
    else if (warp_id >= (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        // The iq segment LEADS the N range, so its tiles are exactly it_t <
        // NUM_IQ_TILES (N_IDX being whole cluster tiles is static_assert'd in the
        // header). head = it_t * CLUSTER_SIZE + cta_rank, which at the production
        // grid equals blockIdx.x -- so SPREAD's first task is this CTA's own head
        // and its flag is already set.
        constexpr int NUM_IQ_TILES = N_IDX / CLUSTER_BLOCK_N;
        // Draining CTAs vs the rest; CHAINS ranks the draining ones last.
        const int busy  = min(num_clusters, NUM_IQ_TILES) * CLUSTER_SIZE;
        const int nfree = num_blocks - busy;

        // ---- 1 HANDSHAKE ----
        // Barrier arrivals must stay in lockstep with the store warps', so the loop
        // shape is fixed; the body is only a release store, which is what keeps
        // these CTAs from becoming the long pole.
        if (!mock_post)
          for (int it_t = cluster_id; it_t < NUM_IQ_TILES; it_t += num_clusters) {
            const int head = it_t * CLUSTER_SIZE + cta_rank;
            cutlass::arch::NamedBarrier::sync(
                NUM_STORE_THREADS + NUM_XFORM_THREADS, 1);   // drain done + CTA fence
            // The CTA barrier orders every store warp's scratch writes before this
            // release; the release publishes that happens-before chain at GPU scope.
            // Paired with ld.acquire in SPREAD.
            if (threadIdx.x == NUM_NON_EPI_THREADS + NUM_STORE_THREADS) {
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(iq_drain_ready + head * IQ_FLAG_STRIDE),
                                "r"(iq_drain_seq)
                             : "memory");
            }
          }

        // ---- 2 CHAINS ----
        // Ahead of SPREAD on purpose: these can run from t=0 while SPREAD cannot
        // start before a drain lands. Running SPREAD first parked them behind a
        // mid-kernel flag wait on exactly the CTAs this pool prefers -- measured
        // d_all 0.26 -> 5.2us at M=31. WARP-LEVEL task pool, ONE task per token row
        // (separate win/cmp loops keep the two chains' register live ranges
        // disjoint), no atomics and no barriers -- purely static. Worker order = all
        // transform warp levels of the iq-FREE CTAs first, iq CTAs only as a last
        // resort. Only exit sync: trailing cluster_sync.
        {
            const bool win_on = comp.win_y2 != nullptr;
            const bool cmp_on = comp.state != nullptr;
            if (win_on || cmp_on) {
                const int wlocal = (int)warp_id - (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32;
                const int rank = ((int)blockIdx.x >= busy)
                    ? ((int)blockIdx.x - busy)                  // iq-free CTAs: rank 0..nfree-1
                    : (nfree + (int)blockIdx.x);                // iq CTAs after them
                constexpr int XFORM_WARPS = NUM_XFORM_THREADS / 32;
                const int wid = (rank < nfree)
                    ? wlocal * nfree + rank                     // free CTAs, levels 0..3 first
                    : XFORM_WARPS * nfree + wlocal * busy + (rank - nfree);
                const int stride = num_blocks * XFORM_WARPS;
                if (win_on)
                    for (int m = wid; m < problem_m; m += stride)
                        idx_comp::process_win_row(comp, m, (int)lane_id);
                if (cmp_on)
                    for (int m = wid; m < problem_m; m += stride)
                        idx_comp::process_row<kIdxFp8>(comp, m, (int)lane_id);
            }
        }

        // ---- 3 SPREAD ----
        // Flat (head, row-batch) task space over the whole grid. One task uses
        // all transform workers: 8 lanes per row, XFORM_ROWS_PER_TASK rows.
        if (!mock_post) {
            // [PROFILE] SPREAD entry/exit (cluster0/CTA0).
            long long iq_t0 = 0;
            const bool iq_prof = iqd.prof != nullptr && blockIdx.x == 0 &&
                threadIdx.x == NUM_NON_EPI_THREADS + NUM_STORE_THREADS;
            if (iq_prof) iq_t0 = ptx::rdclock();
            const uint32_t xtid = threadIdx.x -
                (NUM_NON_EPI_THREADS + NUM_STORE_THREADS);
            const uint32_t r = xtid >> 3, e8 = xtid & 7;
            // Batches per head from problem_m, NOT M_TPL: padding rows run the full
            // converged compute (only their stores are suppressed), so an
            // all-padding batch is pure waste. Trip counts stay CTA-uniform.
            const int nbph =
                (problem_m + XFORM_ROWS_PER_TASK - 1) / XFORM_ROWS_PER_TASK;
            const int nb   = IDX_NUM_HEADS * nbph;
            for (int task = (int)blockIdx.x; task < nb; task += num_blocks) {
                const int head = task % IDX_NUM_HEADS;
                const int m0   = (task / IDX_NUM_HEADS) * XFORM_ROWS_PER_TASK;
                uint32_t v;
                const unsigned long long spin_t0 = ptx::rdtimer_ns();
                do {   // whole warp polls ONE address -> a single broadcast read;
                       // relaxed while spinning, ONE acquire once it passes
                    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                                 : "=r"(v) : "l"(iq_drain_ready + head * IQ_FLAG_STRIDE)
                                 : "memory");
                    if (v == iq_drain_seq) break;
                    if (ptx::rdtimer_ns() - spin_t0 >= IQ_SPIN_TIMEOUT_NS) {
                        printf("wq_b SPREAD drain timeout: cta=%d head=%d flag=%u "
                               "want=%u\n", (int)blockIdx.x, head, v, iq_drain_seq);
                        __trap();
                    }
                    __nanosleep(128);
                } while (true);
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(v) : "l"(iq_drain_ready + head * IQ_FLAG_STRIDE)
                             : "memory");
                // padding rows CLAMP to the last valid row: warps stay converged for
                // the full-mask shuffles, stores are suppressed
                const int m = m0 + (int)r;
                const int mc = m < problem_m ? m : problem_m - 1;
                IdxRowIn d;
                idx_row_load(
                    iq_scratch + ((int64_t)mc * IDX_NUM_HEADS + head) * IDX_HEAD_DIM,
                    e8, mc, q_pos, rope_cos, rope_sin, d);
                if constexpr (kIdxFp8) {
                    idx_row_compute_fp8<true>(
                        d, e8, mc, iqd.head_base + head, iqd.weights,
                        iqd.fp4, reinterpret_cast<float*>(iqd.sf),
                        iqd.num_heads, m < problem_m);
                } else {
                    idx_row_compute<true>(d, e8, mc, iqd.head_base + head,
                                          iqd.fp4, iqd.sf,
                                          iqd.num_heads, m < problem_m);
                }
            }
            if (iq_prof) {
                iqd.prof[0] = iq_t0;
                iqd.prof[1] = ptx::rdclock();
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

static CUtensorMap make_tma_desc_bf16_2d(
    const __nv_bfloat16* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(__nv_bfloat16)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

// ======================== Kernel / SMEM selectors ========================
// Swap-AB path only (M <= 128, decode). kSsq is a COMPILE-TIME dispatch
// dimension (DeepGEMM JIT-config discipline): the ssq-off binary carries no
// dead branch in the hot loops.
template <bool kProfile, bool kSsq, bool kIdxFp8>
static void* get_kernel_ptr(int M) {
    switch (M) {
        case 32:  return (void*)&wq_b_proj_kernel<32,  kProfile, kSsq, kIdxFp8>;
        case 64:  return (void*)&wq_b_proj_kernel<64,  kProfile, kSsq, kIdxFp8>;
        case 96:  return (void*)&wq_b_proj_kernel<96,  kProfile, kSsq, kIdxFp8>;
        case 128: return (void*)&wq_b_proj_kernel<128, kProfile, kSsq, kIdxFp8>;
        default:  return nullptr;
    }
}
static int get_smem_bytes(int M) {
    switch (M) {
        case 32:  return (int)sizeof(SharedStorage<32>);
        case 64:  return (int)sizeof(SharedStorage<64>);
        case 96:  return (int)sizeof(SharedStorage<96>);
        case 128: return (int)sizeof(SharedStorage<128>);
        default:  return 0;
    }
}

// Merged FP8 block-scale run with the indexer segment leading N. Inputs use the
// native DSV4 checkpoint/runtime layout:
//   x_fp8 [M,K] e4m3 ; x_sf [M,K/128] UE8M0 (activation 1x128)
//   w_fp8 [N_MERGED,K] e4m3 ; w_sf [N_MERGED/128,K/128] UE8M0 (weight 128x128)
// Returns [y bf16 [M,N_TOTAL], iq_fp4 i8 [M,64,64], iq_sf i32 [M,64], (timing)]:
// the first N_IDX weight rows are the indexer wq_b; their tiles are drained and
// rope+hadamard+fp4quant runs in the async transform warpgroup. Requires
// q_pos [M] i32 and rope_cos/rope_sin [max_pos,32] f32 (CPU-precomputed).
// head_ssq (optional): ZERO-INITIALIZED fp32 [M,NUM_HEADS_OUT]; the epilogue
// RED-accumulates per-(row, head) sum-of-squares of the MAIN q segment
// (RMSNorm scale folding). mock_post: drain-only baseline (no transform).
static std::vector<torch::Tensor> run_wq_b(
    torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w_fp8, torch::Tensor w_sf, bool profile,
    c10::optional<torch::Tensor> head_ssq,
    torch::Tensor q_pos, torch::Tensor rope_cos, torch::Tensor rope_sin,
    bool mock_post = false, bool enable_ssq = true,
    // fused indexer compressor bundle (all-or-none; see idx_comp_fp4.cuh).
    // FRONT-EMIT: the fresh state row (+idx_ape) is published by front's
    // epilogue; no idx_y4/idx_ape inputs here anymore.
    c10::optional<torch::Tensor> cmp_pos = c10::nullopt,
    c10::optional<torch::Tensor> idx_norm = c10::nullopt,
    c10::optional<torch::Tensor> cos_tab = c10::nullopt,
    c10::optional<torch::Tensor> sin_tab = c10::nullopt,
    c10::optional<torch::Tensor> idx_state = c10::nullopt,
    c10::optional<torch::Tensor> idx_state_row = c10::nullopt,
    int64_t state_ring_entries = 0,
    // local kv window bundle (winkv, CSA stage 4; needs cmp_pos + rope tables too)
    c10::optional<torch::Tensor> win_y2   = c10::nullopt,
    c10::optional<torch::Tensor> win_norm = c10::nullopt,
    // fused activation-quant prologue: q_y bf16 [M,1536] (front y[:, :1536]
    // view OK, stride(1)==1). q_norm_w fp32 [1536] additionally fuses the
    // rmsnorm (CTA-wide, deterministic); without it: plain 1x128 quant.
    // When q_y is given, x_fp8/x_sf are OUTPUT buffers written by the kernel.
    c10::optional<torch::Tensor> q_y      = c10::nullopt,
    c10::optional<torch::Tensor> q_norm_w = c10::nullopt,
    double q_eps = 1e-6,
    // RTP paged-pool writes. Destination values are flat logical slots; each
    // pool supplies its own entries-per-block and physical byte stride.
    c10::optional<torch::Tensor> idx_cache = c10::nullopt,
    c10::optional<torch::Tensor> idx_dst   = c10::nullopt,
    int64_t idx_entries_per_block = 0,
    int64_t idx_block_stride_bytes = 0,
    c10::optional<torch::Tensor> swa_cache = c10::nullopt,
    c10::optional<torch::Tensor> swa_dst   = c10::nullopt,
    int64_t swa_entries_per_block = 0,
    int64_t swa_block_stride_bytes = 0,
    // Optional caller-owned local MQA input buffers. By default the outputs use
    // the plain [M, IDX_NUM_HEADS] layout.
    c10::optional<torch::Tensor> iq_dst         = c10::nullopt,
    c10::optional<torch::Tensor> iq_dst_sf      = c10::nullopt,
    bool indexer_fp8 = false,
    c10::optional<torch::Tensor> iq_weights = c10::nullopt,
    int64_t iq_head_base = 0,
    c10::optional<torch::Tensor> iq_prof = c10::nullopt)
{
    TORCH_CHECK(x_fp8.is_cuda() && x_fp8.is_contiguous() &&
                x_fp8.scalar_type() == torch::kFloat8_e4m3fn, "x_fp8 must be CUDA e4m3");
    TORCH_CHECK(w_fp8.is_cuda() && w_fp8.is_contiguous() &&
                w_fp8.scalar_type() == torch::kFloat8_e4m3fn, "w_fp8 must be CUDA e4m3");
    const auto valid_sf_dtype = [](const torch::Tensor& t) {
        return t.scalar_type() == torch::kFloat8_e8m0fnu ||
               t.scalar_type() == torch::kUInt8;
    };
    TORCH_CHECK(x_sf.is_cuda() && x_sf.is_contiguous() && valid_sf_dtype(x_sf),
                "x_sf must be contiguous CUDA UE8M0 (float8_e8m0fnu or raw uint8)");
    TORCH_CHECK(w_sf.is_cuda() && w_sf.is_contiguous() && valid_sf_dtype(w_sf),
                "w_sf must be contiguous CUDA UE8M0 (float8_e8m0fnu or raw uint8)");

    int M = x_fp8.size(0);
    TORCH_CHECK(x_fp8.size(1) == K_DIM, "x_fp8 must be [M,", K_DIM, "]");
    TORCH_CHECK(w_fp8.size(0) == N_MERGED && w_fp8.size(1) == K_DIM,
                "w_fp8 must be the MERGED weight [", N_MERGED, ",", K_DIM, "]");
    TORCH_CHECK(M >= 1 && M <= 128,
                "kernel supports M in [1,128] (swap path), got M=", M);
    // Arbitrary batch: pick the 32-aligned template; TMA zero-fills the padded
    // activation rows (OOB reads) and clips the padded output rows (OOB stores);
    // the SF packer / ssq / drain / xform paths guard on problem_m == M.
    const int M_pad = (M + 31) / 32 * 32;
    const int num_m_sub = (M_pad + BM - 1) / BM;  // profiling only (== 1)

    const int sf_k = K_DIM / QUANT_BLOCK_K;                       // 12
    TORCH_CHECK(x_sf.dim() == 2 && x_sf.size(0) == M && x_sf.size(1) == sf_k,
                "x_sf must be [M,K/128]=[", M, ",", sf_k, "]");
    TORCH_CHECK(w_sf.dim() == 2 && w_sf.size(0) == NUM_WEIGHT_SF_ROWS_MERGED
                && w_sf.size(1) == sf_k,
                "w_sf must be [N/128,K/128]=[", NUM_WEIGHT_SF_ROWS_MERGED, ",", sf_k, "]");

    // Fused-epilogue outputs + rotary metadata
    TORCH_CHECK(q_pos.is_cuda() && q_pos.scalar_type() == torch::kInt32
                && q_pos.numel() == M && q_pos.is_contiguous(), "q_pos [M] i32");
    TORCH_CHECK(rope_cos.is_cuda() && rope_cos.scalar_type() == torch::kFloat
                && rope_cos.dim() == 2 && rope_cos.size(1) == 32
                && rope_cos.is_contiguous(), "rope_cos [max_pos,32] f32");
    TORCH_CHECK(rope_sin.is_cuda() && rope_sin.scalar_type() == torch::kFloat
                && rope_sin.sizes() == rope_cos.sizes()
                && rope_sin.is_contiguous(), "rope_sin [max_pos,32] f32");
    // Indexer-q destination (see IqDest). Default allocation degenerates the
    // store math to the plain non-TP [M, IDX_NUM_HEADS] layout.
    torch::Tensor iq_fp4, iq_sf;
    if (iq_dst.has_value()) {
        TORCH_CHECK(iq_dst_sf.has_value(), "iq_dst also needs iq_dst_sf");
        iq_fp4 = iq_dst.value();
        iq_sf  = iq_dst_sf.value();
    } else {
        TORCH_CHECK(!iq_dst_sf.has_value() && iq_head_base == 0,
                    "iq_dst_sf / iq_head_base require iq_dst");
        iq_fp4 = torch::empty(
            {M, IDX_NUM_HEADS, indexer_fp8 ? IDX_HEAD_DIM : IDX_HEAD_DIM / 2},
            x_fp8.options().dtype(indexer_fp8 ? torch::kFloat8_e4m3fn
                                              : torch::kInt8));
        iq_sf = torch::empty({M, IDX_NUM_HEADS},
                             x_fp8.options().dtype(indexer_fp8 ? torch::kFloat32
                                                               : torch::kInt32));
    }
    const int iq_rows  = (int)iq_fp4.size(0);
    const int iq_heads = (int)iq_fp4.size(1);
    TORCH_CHECK(iq_fp4.is_cuda() && iq_fp4.dim() == 3 && iq_fp4.is_contiguous()
                && iq_fp4.scalar_type() == (indexer_fp8 ? torch::kFloat8_e4m3fn
                                                        : torch::kInt8)
                && iq_fp4.size(2) == (indexer_fp8 ? IDX_HEAD_DIM : IDX_HEAD_DIM / 2),
                "iq_dst has the wrong dtype/shape for the selected Indexer format");
    TORCH_CHECK(iq_sf.is_cuda() && iq_sf.is_contiguous()
                && iq_sf.scalar_type() == (indexer_fp8 ? torch::kFloat32
                                                       : torch::kInt32)
                && iq_sf.sizes() == torch::IntArrayRef({iq_rows, iq_heads}),
                "iq_dst_sf must match the selected Indexer format and iq_dst");
    if (indexer_fp8 && !mock_post) {
        TORCH_CHECK(iq_weights.has_value() && iq_weights->is_cuda()
                    && iq_weights->scalar_type() == torch::kFloat32
                    && iq_weights->dim() == 2 && iq_weights->is_contiguous()
                    && iq_weights->sizes() == torch::IntArrayRef({iq_rows, iq_heads}),
                    "FP8 Indexer needs contiguous CUDA fp32 iq_weights [rows,heads]");
    }
    TORCH_CHECK(iq_head_base >= 0 && iq_head_base + IDX_NUM_HEADS <= iq_heads,
                "iq_head_base + IDX_NUM_HEADS(", IDX_NUM_HEADS, ") exceeds the "
                "destination head count ", iq_heads);
    TORCH_CHECK(iq_rows >= M, "iq_dst needs >= M rows, got ", iq_rows, " < ", M);
    // drain-first workspace: the epilogue dumps the iq accum here (L2-resident)
    // to release the TMEM stage before the transform pass
    auto iq_ws = torch::empty({M, IDX_NUM_HEADS, IDX_HEAD_DIM},
                              x_fp8.options().dtype(torch::kBFloat16));
    const int* q_pos_ptr = q_pos.data_ptr<int>();
    const float* cos_ptr = rope_cos.data_ptr<float>();
    const float* sin_ptr = rope_sin.data_ptr<float>();
    uint8_t* iq_fp4_ptr = reinterpret_cast<uint8_t*>(iq_fp4.data_ptr());
    int* iq_sf_ptr = reinterpret_cast<int*>(iq_sf.data_ptr());
    __nv_bfloat16* iq_ws_ptr =
        reinterpret_cast<__nv_bfloat16*>(iq_ws.data_ptr());
    IqDest iqd{};
    iqd.fp4       = iq_fp4_ptr;
    iqd.sf        = iq_sf_ptr;
    iqd.weights   = iq_weights.has_value() ? iq_weights->data_ptr<float>() : nullptr;
    iqd.head_base = (int)iq_head_base;
    iqd.num_heads = iq_heads;
    if (iq_prof.has_value()) {
        TORCH_CHECK(iq_prof.value().is_cuda() && iq_prof.value().is_contiguous()
                    && iq_prof.value().numel() >= 2
                    && iq_prof.value().scalar_type() == torch::kInt64,
                    "iq_prof must be a CUDA i64 [>=2]");
        iqd.prof = reinterpret_cast<long long*>(iq_prof.value().data_ptr());
    }

    const bool quant_on = q_y.has_value() && q_y->numel() > 0;
    auto out = torch::empty({M, N_TOTAL}, x_fp8.options().dtype(torch::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();

    // Ready flags for the cross-CTA idx-post handoff (one 128B line per indexer
    // head), allocated ONCE per device.
    // EAGER: no reset. Producers publish a host-monotonic launch tag and
    // consumers wait for that exact value, so last launch's flags can never be
    // mistaken for this launch's -- that is what keeps a memset off the path.
    // CAPTURE: the host tag is frozen into the graph. The fused quant producer
    // resets the valid flag words before GDS releases WQ_B; capture without that
    // producer retains a memset fallback. Graph and eager launches sharing a
    // device must not run concurrently on different streams.
    static uint32_t* iq_drain_ready_dev[kMaxDevices] = {};
    static uint32_t  iq_drain_ready_tag[kMaxDevices] = {};
    const int dev_id = x_fp8.device().index();
    TORCH_CHECK(dev_id >= 0 && dev_id < kMaxDevices,
                "wq_b: device index ", dev_id, " outside the flag table");
    constexpr size_t kFlagBytes = IQ_FLAG_SLOTS * sizeof(uint32_t);
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    TORCH_CHECK(cudaStreamIsCapturing(stream, &cap) == cudaSuccess,
                "wq_b: cudaStreamIsCapturing failed");
    if (iq_drain_ready_dev[dev_id] == nullptr) {
        TORCH_CHECK(cap == cudaStreamCaptureStatusNone,
                    "wq_b: run one eager launch before capturing (the iq "
                    "ready-flag page cannot be allocated during capture)");
        TORCH_CHECK(cudaMalloc(&iq_drain_ready_dev[dev_id], kFlagBytes) == cudaSuccess,
                    "wq_b: iq ready-flag alloc failed");
        TORCH_CHECK(cudaMemsetAsync(iq_drain_ready_dev[dev_id], 0, kFlagBytes, stream)
                        == cudaSuccess, "wq_b: iq ready-flag zero failed");
    }
    uint32_t* iq_drain_ready_ptr = iq_drain_ready_dev[dev_id];
    uint32_t* iq_drain_ready_reset_ptr = nullptr;
    uint32_t  iq_drain_seq;
    if (cap == cudaStreamCaptureStatusNone) {
        iq_drain_seq = ++iq_drain_ready_tag[dev_id];
    } else {
        iq_drain_seq = 1u;
        if (quant_on) {
            iq_drain_ready_reset_ptr = iq_drain_ready_ptr;
        } else {
            TORCH_CHECK(cudaMemsetAsync(iq_drain_ready_ptr, 0, kFlagBytes, stream)
                            == cudaSuccess,
                        "wq_b: iq ready-flag reset node failed");
        }
    }

    auto x_ptr   = reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8.data_ptr());
    auto w_ptr   = reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8.data_ptr());
    auto xsf_ptr = reinterpret_cast<const uint8_t*>(x_sf.data_ptr());
    auto wsf_ptr = reinterpret_cast<const uint8_t*>(w_sf.data_ptr());
    auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    // K-major 128B-swizzled operands; desc_D covers the main q segment only
    // (iq tiles never TMA-store). desc_A/desc_D use the REAL M (globalDim) with
    // the PADDED box: TMA zero-fills OOB loads and clips OOB stores.
    CUtensorMap desc_A   = make_tma_desc_fp8_2d(x_ptr, M, K_DIM, M_pad / NUM_MULTICAST, BLOCK_K);
    CUtensorMap desc_B   = make_tma_desc_fp8_2d(w_ptr, N_MERGED, K_DIM, LOAD_BLOCK_N, BLOCK_K);
    CUtensorMap desc_D   = make_tma_desc_bf16_2d(
        out_ptr, M, N_TOTAL, STORE_BLOCK_M, STORE_BLOCK_N_ATOM);

    // Grid: persistent, cluster of 2 CTAs. PER DEVICE: the SM count is a device
    // property, and caching device 0's for the whole process burns its grid size
    // into every other card -- wrong the moment one process drives two GPUs, or
    // one process drives a GPU that is not 0.
    static int num_SMs_dev[kMaxDevices] = {};
    if (num_SMs_dev[dev_id] == 0)
        cudaDeviceGetAttribute(&num_SMs_dev[dev_id],
                               cudaDevAttrMultiProcessorCount, dev_id);
    const int num_SMs = num_SMs_dev[dev_id];
    TORCH_CHECK(num_SMs >= CLUSTER_SIZE,
                "wq_b FP8 requires at least ", CLUSTER_SIZE, " SMs, got ", num_SMs);
    constexpr int num_tiles = NUM_N_TILES_MERGED;
    const int total_cta = num_tiles * CLUSTER_SIZE;
    const int max_clusters = min(num_SMs, total_cta) / CLUSTER_SIZE;
    int num_clusters = max_clusters > 0 ? max_clusters : 1;
    // Minimize wave count, then use the fewest clusters that achieve it; exact
    // divisibility would unnecessarily reduce the TPDP 160-tile grid.
    const int waves = (num_tiles + num_clusters - 1) / num_clusters;
    num_clusters = (num_tiles + waves - 1) / waves;
    if (const char* e = std::getenv("WQ_B_CLUSTERS")) {
        const int req = atoi(e);
        if (req > 0) num_clusters = req < max_clusters ? req : max_clusters;
    }
    int grid_size = max_clusters * CLUSTER_SIZE;   // full grid serves transform work
    const int max_iters = ((num_tiles + num_clusters - 1) / num_clusters) * num_m_sub;

    int64_t* prof_dev = nullptr;
    torch::Tensor timing;
    if (profile) {
        // [max_iters, 7] per persistent iteration on cluster0/CTA0 (same SM, clock64):
        //   load_start/end | mma_start/end | epi_start/end | mma_wait_cycles
        timing = torch::zeros({max_iters, 7}, x_fp8.options().dtype(torch::kInt64));
        prof_dev = reinterpret_cast<int64_t*>(timing.data_ptr());
    }

    // head_ssq: ON by default. Callers may pass their own ZERO-INITIALIZED
    // [M,NUM_HEADS_OUT] buffer (reuse across layers); otherwise allocate one.
    // enable_ssq=false (and no buffer) -> bit-identical plain GEMM.
    torch::Tensor ssq_t;
    float* ssq_ptr = nullptr;
    if (head_ssq.has_value()) {
        auto& t = head_ssq.value();
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat && t.is_contiguous()
                    && t.numel() >= (int64_t)M * NUM_HEADS_OUT,
                    "head_ssq must be CUDA fp32 contiguous [M,", NUM_HEADS_OUT,
                    "], ZERO-INITIALIZED (the kernel accumulates via f32 RED atomics)");
        ssq_t = t;
        ssq_ptr = t.data_ptr<float>();
    } else if (enable_ssq) {
        ssq_t = torch::zeros({M, NUM_HEADS_OUT}, x_fp8.options().dtype(torch::kFloat32));
        ssq_ptr = ssq_t.data_ptr<float>();
    }

    // ---- fused indexer compressor + local kv window: validate the bundles ----
    const bool comp_on = idx_state.has_value();
    const bool win_on  = win_y2.has_value();
    idx_comp::Args comp{};
    torch::Tensor idx_q4_t, idx_s4_t, win_q8_t, win_s8_t, win_rope_t;
    auto ck = [](const torch::Tensor& t, torch::ScalarType ty, const char* n) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == ty,
                    n, " must be contiguous CUDA ", ty);
    };
    auto i32m = [&](const torch::Tensor& t, const char* n) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                    t.scalar_type() == torch::kInt32 && t.numel() >= M,
                    n, " must be CUDA i32 [>=M]");
        return t.data_ptr<int>();
    };
    if (comp_on || win_on) {
        TORCH_CHECK(cmp_pos && cos_tab && sin_tab,
                    "comp/win bundles need cmp_pos + cos_tab + sin_tab");
        ck(*cmp_pos, torch::kInt64, "cmp_pos");
        ck(*cos_tab, torch::kFloat, "cos_tab");
        ck(*sin_tab, torch::kFloat, "sin_tab");
        TORCH_CHECK(cmp_pos->numel() == M, "cmp_pos must be [M]");
        TORCH_CHECK(cos_tab->dim() == 2 && cos_tab->size(1) == idx_comp::RD / 2
                    && sin_tab->sizes() == cos_tab->sizes(),
                    "cos_tab/sin_tab must be [S,32]");
        comp.pos = reinterpret_cast<const long long*>(cmp_pos->data_ptr<int64_t>());
        comp.cos_tab = cos_tab->data_ptr<float>();
        comp.sin_tab = sin_tab->data_ptr<float>();
    }
    if (comp_on) {
        // FRONT-EMIT: the fresh idx state row (fp32 + idx_ape on the score
        // half) is published by front_mixed's epilogue BEFORE this kernel;
        // the chain here only aggregates -- no y4/ape inputs anymore.
        TORCH_CHECK(idx_norm,
                    "indexer compressor needs idx_norm + idx_state");
        ck(*idx_norm, torch::kFloat, "idx_norm");
        ck(*idx_state, torch::kFloat, "idx_state");
        TORCH_CHECK(idx_norm->numel() == idx_comp::D_I, "idx_norm must be [128]");
        TORCH_CHECK(idx_state->dim() >= 2 &&
                    idx_state->size(-1) == 2 * idx_comp::WK_I,
                    "idx_state last dim must be ", 2 * idx_comp::WK_I,
                    " ([kv|score] halves)");
        // Compact q4/s4 SKIPPED in cache mode (double-write bandwidth saver);
        // the paged direct write below is the production output. PURE outputs
        // -> empty, not zeros (a zeros fill kernel lands in wall timings;
        // non-compress rows are garbage by contract, consumers gate on pos).
        if (!(idx_cache.has_value() && idx_cache->numel() > 0)) {
            idx_q4_t = torch::empty({M, indexer_fp8 ? idx_comp::D_I
                                                     : idx_comp::D_I / 2},
                                    x_fp8.options().dtype(torch::kUInt8));
            idx_s4_t = torch::empty(
                {M, indexer_fp8 ? 1 : idx_comp::D_I / 32},
                x_fp8.options().dtype(indexer_fp8 ? torch::kFloat32
                                                  : torch::kUInt8));
            comp.q4 = idx_q4_t.data_ptr<uint8_t>();
            comp.s4 = reinterpret_cast<uint8_t*>(idx_s4_t.data_ptr());
        }
        comp.norm_w = idx_norm->data_ptr<float>();
        comp.state = idx_state->data_ptr<float>();
        TORCH_CHECK(idx_state_row.has_value(),
                    "idx_state requires idx_state_row");
        TORCH_CHECK(state_ring_entries >= idx_comp::SROWS,
                    "state_ring_entries must be >= ", idx_comp::SROWS);
        comp.state_row = i32m(*idx_state_row, "idx_state_row");
        comp.state_ring_entries = (int)state_ring_entries;
    }
    if (win_on) {
        TORCH_CHECK(win_norm, "local kv window needs win_y2 AND win_norm");
        ck(*win_y2,   torch::kFloat, "win_y2");
        ck(*win_norm, torch::kFloat, "win_norm");
        TORCH_CHECK(win_y2->dim() == 2 && win_y2->size(0) == M
                    && win_y2->size(1) == idx_comp::D_W, "win_y2 must be [M,512]");
        TORCH_CHECK(win_norm->numel() == idx_comp::D_W, "win_norm must be [512]");
        // Compact win outputs SKIPPED in cache mode (see idx note above).
        if (!(swa_cache.has_value() && swa_cache->numel() > 0)) {
            win_q8_t = torch::empty({M, idx_comp::NF8_W},
                                    x_fp8.options().dtype(torch::kUInt8));
            win_s8_t = torch::empty({M, idx_comp::NF8_W / 64},
                                    x_fp8.options().dtype(torch::kFloat32));
            win_rope_t = torch::empty({M, idx_comp::RD},
                                      x_fp8.options().dtype(torch::kBFloat16));
            comp.win_q8 = win_q8_t.data_ptr<uint8_t>();
            comp.win_s8 = win_s8_t.data_ptr<float>();
            comp.win_rope = reinterpret_cast<__nv_bfloat16*>(win_rope_t.data_ptr());
        }
        comp.win_y2 = win_y2->data_ptr<float>();
        comp.win_norm = win_norm->data_ptr<float>();
    }
    // ---- RTP cache geometry + direct pool writes ----
    if (idx_cache.has_value() && idx_cache->numel() > 0) {
        TORCH_CHECK(idx_dst.has_value(), "idx_cache requires idx_dst");
        TORCH_CHECK(idx_entries_per_block > 0,
                    "idx_cache requires idx_entries_per_block");
        const int64_t payload = idx_entries_per_block
            * ((indexer_fp8 ? idx_comp::D_I : idx_comp::D_I / 2) + 4);
        TORCH_CHECK(idx_block_stride_bytes >= payload,
                    "idx_block_stride_bytes must cover page payload");
        TORCH_CHECK(!indexer_fp8 || idx_block_stride_bytes % 4 == 0,
                    "FP8 indexer block stride must preserve FP32 scale alignment");
        TORCH_CHECK(idx_cache->is_cuda() && idx_cache->is_contiguous() &&
                    idx_cache->scalar_type() == torch::kUInt8 &&
                    idx_cache->numel() % idx_block_stride_bytes == 0,
                    "idx_cache must be uint8 physical pages with the supplied stride");
        comp.idx_cache = reinterpret_cast<uint8_t*>(idx_cache->data_ptr());
        comp.idx_dst = i32m(*idx_dst, "idx_dst");
        comp.idx_entries_per_block = (int)idx_entries_per_block;
        comp.idx_block_stride_bytes = (size_t)idx_block_stride_bytes;
    }
    if (swa_cache.has_value() && swa_cache->numel() > 0) {
        TORCH_CHECK(swa_dst.has_value(), "swa_cache requires swa_dst");
        TORCH_CHECK(swa_entries_per_block > 0,
                    "swa_cache requires swa_entries_per_block");
        const int64_t payload = swa_entries_per_block * (idx_comp::M1_TOK_BODY + 8);
        TORCH_CHECK(swa_block_stride_bytes >= payload,
                    "swa_block_stride_bytes must cover page payload");
        TORCH_CHECK(swa_cache->is_cuda() && swa_cache->is_contiguous() &&
                    swa_cache->scalar_type() == torch::kUInt8 &&
                    swa_cache->numel() % swa_block_stride_bytes == 0,
                    "swa_cache must be uint8 physical pages with the supplied stride");
        comp.swa_cache = reinterpret_cast<uint8_t*>(swa_cache->data_ptr());
        comp.swa_dst = i32m(*swa_dst, "swa_dst");
        comp.swa_entries_per_block = (int)swa_entries_per_block;
        comp.swa_block_stride_bytes = (size_t)swa_block_stride_bytes;
    }

    // ---- activation quant producer (PDL; replaces the in-kernel grid ticket) ----
    const __nv_bfloat16* qy_ptr = nullptr;
    const float* qgamma_ptr = nullptr;
    int64_t qy_lda = 0;
    if (quant_on) {
        TORCH_CHECK(q_y->is_cuda() && q_y->scalar_type() == torch::kBFloat16 &&
                    q_y->dim() == 2 && q_y->size(0) == M &&
                    q_y->size(1) == K_DIM && q_y->stride(1) == 1 &&
                    q_y->stride(0) % 4 == 0,
                    "q_y must be bf16 [M,", K_DIM, "] with stride(1)==1 and "
                    "16B-aligned rows (strided views of front y are fine)");
        qy_ptr = reinterpret_cast<const __nv_bfloat16*>(q_y->data_ptr());
        qy_lda = q_y->stride(0);
        if (q_norm_w.has_value() && q_norm_w->numel() > 0) {
            TORCH_CHECK(q_norm_w->is_cuda() && q_norm_w->is_contiguous() &&
                        q_norm_w->scalar_type() == torch::kFloat32 &&
                        q_norm_w->numel() == K_DIM,
                        "q_norm_w must be fp32 [", K_DIM, "]");
            qgamma_ptr = q_norm_w->data_ptr<float>();
        }
    }

    const bool want_ssq = ssq_ptr != nullptr;
    void* kernel_ptr = nullptr;
    if (indexer_fp8) {
        kernel_ptr = profile
            ? (want_ssq ? get_kernel_ptr<true, true, true>(M_pad)
                        : get_kernel_ptr<true, false, true>(M_pad))
            : (want_ssq ? get_kernel_ptr<false, true, true>(M_pad)
                        : get_kernel_ptr<false, false, true>(M_pad));
    } else {
        kernel_ptr = profile
            ? (want_ssq ? get_kernel_ptr<true, true, false>(M_pad)
                        : get_kernel_ptr<true, false, false>(M_pad))
            : (want_ssq ? get_kernel_ptr<false, true, false>(M_pad)
                        : get_kernel_ptr<false, false, false>(M_pad));
    }
    int smem_bytes = get_smem_bytes(M_pad);
    TORCH_CHECK(kernel_ptr != nullptr && smem_bytes > 0, "Unsupported M=", M);

    // cudaFuncSetAttribute is PER DEVICE state, so this cache needs the device
    // ordinal: without it the second card in a process never gets the opt-in
    // dynamic smem raised and every launch on it fails.
    static bool smem_configured[kMaxDevices][2][2][2][9] = {};
    const int m_idx = M_pad / 32;
    const int p_idx = profile ? 1 : 0;
    const int s_idx = want_ssq ? 1 : 0;
    const int f_idx = indexer_fp8 ? 1 : 0;
    if (!smem_configured[dev_id][p_idx][s_idx][f_idx][m_idx]) {
        auto attr_err = cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(attr_err), " smem_bytes=", smem_bytes);
        smem_configured[dev_id][p_idx][s_idx][f_idx][m_idx] = true;
    }

    {
        dim3 grid(grid_size, 1, 1);
        // Fused post-processing OFF by default (mock_post=true): launch the plain
        // 256 threads, keeping warps 8..15 absent when no CUDA-core chain needs them.
        // work. Enabling the fused path launches the async transform warpgroup
        // too (512) -- the blockDim is TIED to mock_post so the drain handoff
        // barrier can never wait on threads that were not launched. The fused
        // indexer compressor / winkv chains also run on warps 8..15.
        dim3 block((mock_post && !comp_on && !win_on) ? TPB : TPB_IDX, 1, 1);
        int mock_i = mock_post ? 1 : 0;

        // Activation quant producer: launched FIRST on the same stream; the
        // GEMM below is PDL-linked (programmatic stream serialization), so its
        // prologue overlaps the quant and its GDS provides the ordering.
        if (quant_on) {
            const int qgrid = qgamma_ptr
                ? M                                             // one CTA per row
                : (M * NUM_K_TILES + 5) / 6;                    // one warp per K128 block
            qnorm_quant_kernel<<<qgrid, 192, 0, stream>>>(
                qy_ptr, qy_lda, qgamma_ptr, static_cast<float>(q_eps),
                const_cast<__nv_fp8_e4m3*>(x_ptr),
                const_cast<uint8_t*>(xsf_ptr), M, ssq_ptr,
                iq_drain_ready_reset_ptr);
        }

        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem_bytes;
        config.stream = stream;

        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = CLUSTER_SIZE;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        if (quant_on) {
            // PDL only when a producer is in flight: an unconditional flag
            // would let the GEMM overlap whatever ran before it (e.g. the
            // benchmark's L2-flush kernel), skewing timings.
            attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[config.numAttrs].val.programmaticStreamSerializationAllowed = 1;
            config.numAttrs++;
        }

        void* ptr_args[] = {
            &desc_A, &desc_B, &xsf_ptr, &wsf_ptr, &desc_D,
            &M, &grid_size, &num_clusters, &ssq_ptr,
            &iqd, &iq_ws_ptr, &iq_drain_ready_ptr, &iq_drain_seq,
            &q_pos_ptr, &cos_ptr, &sin_ptr,
            &mock_i, &comp, &prof_dev
        };
        auto err = cudaLaunchKernelExC(&config, kernel_ptr, ptr_args);
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    }

    std::vector<torch::Tensor> ret = {out, iq_fp4, iq_sf};
    if (ssq_ptr) ret.push_back(ssq_t);     // present unless enable_ssq=false w/o buffer
    if (mock_post) ret.push_back(iq_ws);   // export drained bf16 iq (bitwise test hook)
    if (comp_on && idx_q4_t.defined()) { ret.push_back(idx_q4_t); ret.push_back(idx_s4_t); }
    if (win_on && win_q8_t.defined()) {
        ret.push_back(win_q8_t); ret.push_back(win_s8_t); ret.push_back(win_rope_t);
    }
    if (profile) ret.push_back(timing);
    return ret;
}

// ======================== PyTorch Binding ========================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("n_main") = N_TOTAL;
    m.attr("n_index") = N_IDX;
    m.attr("n_merged") = N_MERGED;
    m.attr("num_main_heads") = NUM_HEADS_OUT;
    m.attr("num_index_heads") = IDX_NUM_HEADS;
    m.def("wq_b_proj_gemm_merged",
          [](torch::Tensor x_fp8, torch::Tensor x_sf,
             torch::Tensor w_fp8, torch::Tensor w_sf,
             torch::Tensor q_pos, torch::Tensor rope_cos, torch::Tensor rope_sin,
             c10::optional<torch::Tensor> head_ssq, bool mock_post, bool enable_ssq,
             c10::optional<torch::Tensor> cmp_pos, c10::optional<torch::Tensor> idx_norm,
             c10::optional<torch::Tensor> cos_tab, c10::optional<torch::Tensor> sin_tab,
             c10::optional<torch::Tensor> idx_state,
             c10::optional<torch::Tensor> idx_state_row,
             int64_t state_ring_entries,
             c10::optional<torch::Tensor> win_y2, c10::optional<torch::Tensor> win_norm,
             c10::optional<torch::Tensor> q_y, c10::optional<torch::Tensor> q_norm_w,
             double q_eps,
             c10::optional<torch::Tensor> idx_cache, c10::optional<torch::Tensor> idx_dst,
             int64_t idx_entries_per_block, int64_t idx_block_stride_bytes,
             c10::optional<torch::Tensor> swa_cache, c10::optional<torch::Tensor> swa_dst,
             int64_t swa_entries_per_block, int64_t swa_block_stride_bytes,
             c10::optional<torch::Tensor> iq_dst, c10::optional<torch::Tensor> iq_dst_sf,
             bool indexer_fp8, c10::optional<torch::Tensor> iq_weights,
             int64_t iq_head_base,
             c10::optional<torch::Tensor> iq_prof) {
              return run_wq_b(x_fp8, x_sf, w_fp8, w_sf, /*profile=*/false, head_ssq,
                              q_pos, rope_cos, rope_sin, mock_post, enable_ssq,
                              cmp_pos, idx_norm,
                              cos_tab, sin_tab, idx_state, idx_state_row,
                              state_ring_entries, win_y2, win_norm,
                              q_y, q_norm_w, q_eps,
                              idx_cache, idx_dst, idx_entries_per_block,
                              idx_block_stride_bytes, swa_cache, swa_dst,
                              swa_entries_per_block, swa_block_stride_bytes,
                              iq_dst, iq_dst_sf, indexer_fp8, iq_weights,
                              iq_head_base,
                              iq_prof);
          },
          "MERGED wq_b + indexer wq_b (w [8192+N,K]: indexer rows FIRST, then "
          "main q), swap path, M in [1,128]. "
          "Returns [y bf16 [M,n_main], iq_fp4 i8 [M,64,64], iq_sf i32 [M,64], "
          "ssq fp32 [M,num_main_heads] (unless enable_ssq=False), iq_ws (if mock_post), "
          "idx_q4 u8 [M,64] + idx_s4 u8 [M,4] (if the compressor bundle is given)]. "
          "head_ssq: optional caller-owned ZERO-INIT buffer; default None allocates. "
          "DEFAULT mock_post=True: 256 threads, GEMM+ssq only -- iq_fp4/iq_sf stay "
          "garbage; run idx_postprocess_standalone over iq_ws. mock_post=False: "
          "512 threads, fuses rope+hadamard+fp4quant in-kernel. Fused indexer "
          "COMPRESSOR (no split-K): pass cmp_pos [M] i64 / idx_norm [128] / "
          "cos_tab+sin_tab [S,32] / idx_state [..,512] + idx_state_row + "
          "state_ring_entries (framework state ring; the "
          "fresh row is published by front's FRONT-EMIT epilogue, +idx_ape) "
          "-- warps 8..15 run it beside the GEMM. "
          "LOCAL KV WINDOW (CSA stage 4 full chain): pass win_y2 [M,512] + win_norm "
          "[512] (+ cmp_pos/cos_tab/sin_tab) -> appends win_q8 [M,448] u8, win_s8 "
          "[M,7] f32, win_rope [M,64] bf16. "
          "IQ DESTINATION: optional iq_dst [rows,heads,64] + iq_dst_sf "
          "[rows,heads] write directly into caller-owned local MQA input buffers; "
          "omit both for the plain [M,IDX_NUM_HEADS] layout.",
          py::arg("x_fp8"), py::arg("x_sf"), py::arg("w_fp8"), py::arg("w_sf"),
          py::arg("q_pos"), py::arg("rope_cos"), py::arg("rope_sin"),
          py::arg("head_ssq") = c10::nullopt,
          py::arg("mock_post") = true,
          py::arg("enable_ssq") = true,
          py::arg("cmp_pos") = c10::nullopt,
          py::arg("idx_norm") = c10::nullopt,
          py::arg("cos_tab") = c10::nullopt,
          py::arg("sin_tab") = c10::nullopt,
          py::arg("idx_state") = c10::nullopt,
          py::arg("idx_state_row") = c10::nullopt,
          py::arg("state_ring_entries") = 0,
          py::arg("win_y2") = c10::nullopt,
          py::arg("win_norm") = c10::nullopt,
          py::arg("q_y") = c10::nullopt,
          py::arg("q_norm_w") = c10::nullopt,
          py::arg("q_eps") = 1e-6,
          py::arg("idx_cache") = c10::nullopt,
          py::arg("idx_dst") = c10::nullopt,
          py::arg("idx_entries_per_block") = 0,
          py::arg("idx_block_stride_bytes") = 0,
          py::arg("swa_cache") = c10::nullopt,
          py::arg("swa_dst") = c10::nullopt,
          py::arg("swa_entries_per_block") = 0,
          py::arg("swa_block_stride_bytes") = 0,
          py::arg("iq_dst") = c10::nullopt,
          py::arg("iq_dst_sf") = c10::nullopt,
          py::arg("indexer_fp8") = false,
          py::arg("iq_weights") = c10::nullopt,
          py::arg("iq_head_base") = 0,
          py::arg("iq_prof") = c10::nullopt);
    m.def("wq_b_proj_gemm_merged_profiled",
          [](torch::Tensor x_fp8, torch::Tensor x_sf,
             torch::Tensor w_fp8, torch::Tensor w_sf,
             torch::Tensor q_pos, torch::Tensor rope_cos, torch::Tensor rope_sin,
             bool mock_post) {
              return run_wq_b(x_fp8, x_sf, w_fp8, w_sf, /*profile=*/true, c10::nullopt,
                              q_pos, rope_cos, rope_sin, mock_post);
          },
          "MERGED path + clock64 load/MMA/epilogue timing -> "
          "[y, iq_fp4, iq_sf, (iq_ws), timing[max_iters,7]]",
          py::arg("x_fp8"), py::arg("x_sf"), py::arg("w_fp8"), py::arg("w_sf"),
          py::arg("q_pos"), py::arg("rope_cos"), py::arg("rope_sin"),
          py::arg("mock_post") = true);
    m.def("idx_postprocess_standalone",
          [](torch::Tensor iq_f32, torch::Tensor q_pos,
             torch::Tensor rope_cos, torch::Tensor rope_sin) {
              TORCH_CHECK(iq_f32.is_cuda() && iq_f32.is_contiguous()
                          && (iq_f32.scalar_type() == torch::kFloat ||
                              iq_f32.scalar_type() == torch::kBFloat16)
                          && iq_f32.dim() == 3 && iq_f32.size(1) == IDX_NUM_HEADS
                          && iq_f32.size(2) == IDX_HEAD_DIM,
                          "iq_f32 must be contiguous CUDA fp32/bf16 [M,64,128]");
              const int M = iq_f32.size(0);
              TORCH_CHECK(M >= 1, "empty batch");
              auto iq_fp4 = torch::empty({M, IDX_NUM_HEADS, IDX_HEAD_DIM / 2},
                                         iq_f32.options().dtype(torch::kInt8));
              auto iq_sf = torch::empty({M, IDX_NUM_HEADS},
                                        iq_f32.options().dtype(torch::kInt32));
              const int rows = M * IDX_NUM_HEADS;
              auto stream = at::cuda::getCurrentCUDAStream();
              if (iq_f32.scalar_type() == torch::kBFloat16)
                  idx_post_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(iq_f32.data_ptr()),
                      q_pos.data_ptr<int>(), rope_cos.data_ptr<float>(),
                      rope_sin.data_ptr<float>(),
                      reinterpret_cast<uint8_t*>(iq_fp4.data_ptr()),
                      iq_sf.data_ptr<int>(), rows, IDX_NUM_HEADS);
              else
                  idx_post_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
                      iq_f32.data_ptr<float>(), q_pos.data_ptr<int>(),
                      rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
                      reinterpret_cast<uint8_t*>(iq_fp4.data_ptr()),
                      iq_sf.data_ptr<int>(), rows, IDX_NUM_HEADS);
              return std::vector<torch::Tensor>{iq_fp4, iq_sf};
          },
          "Standalone rope+hadamard+fp4quant over fp32/bf16 iq [M,64,128] -> "
          "(iq_fp4 [M,64,64] i8, iq_sf [M,64] i32) -- the separate-kernel baseline",
          py::arg("iq_f32"), py::arg("q_pos"), py::arg("rope_cos"), py::arg("rope_sin"));
    m.def("idx_postprocess_fp8_standalone",
          [](torch::Tensor iq, torch::Tensor weights, torch::Tensor q_pos,
             torch::Tensor rope_cos, torch::Tensor rope_sin) {
              TORCH_CHECK(iq.is_cuda() && iq.is_contiguous()
                          && (iq.scalar_type() == torch::kFloat
                              || iq.scalar_type() == torch::kBFloat16)
                          && iq.dim() == 3 && iq.size(1) == IDX_NUM_HEADS
                          && iq.size(2) == IDX_HEAD_DIM,
                          "iq must be contiguous CUDA fp32/bf16 [M,64,128]");
              const int M = iq.size(0);
              TORCH_CHECK(M >= 1 && weights.is_cuda() && weights.is_contiguous()
                          && weights.scalar_type() == torch::kFloat32
                          && weights.sizes() == torch::IntArrayRef({M, IDX_NUM_HEADS}),
                          "weights must be contiguous CUDA fp32 [M,64]");
              auto q8 = torch::empty({M, IDX_NUM_HEADS, IDX_HEAD_DIM},
                                     iq.options().dtype(torch::kFloat8_e4m3fn));
              auto wf = torch::empty({M, IDX_NUM_HEADS},
                                     iq.options().dtype(torch::kFloat32));
              const int rows = M * IDX_NUM_HEADS;
              auto stream = at::cuda::getCurrentCUDAStream();
              if (iq.scalar_type() == torch::kBFloat16)
                  idx_post_fp8_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
                      reinterpret_cast<const __nv_bfloat16*>(iq.data_ptr()),
                      weights.data_ptr<float>(), q_pos.data_ptr<int>(),
                      rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
                      reinterpret_cast<uint8_t*>(q8.data_ptr()), wf.data_ptr<float>(),
                      rows, IDX_NUM_HEADS);
              else
                  idx_post_fp8_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
                      iq.data_ptr<float>(), weights.data_ptr<float>(),
                      q_pos.data_ptr<int>(), rope_cos.data_ptr<float>(),
                      rope_sin.data_ptr<float>(),
                      reinterpret_cast<uint8_t*>(q8.data_ptr()), wf.data_ptr<float>(),
                      rows, IDX_NUM_HEADS);
              return std::vector<torch::Tensor>{q8, wf};
          },
          "Standalone RTP Indexer-Q BF16/RoPE/E4M3 quant with scale-folded weights",
          py::arg("iq"), py::arg("weights"), py::arg("q_pos"),
          py::arg("rope_cos"), py::arg("rope_sin"));
    m.def("head_ssq_standalone",
          [](torch::Tensor y) {
              TORCH_CHECK(y.is_cuda() && y.is_contiguous()
                          && y.scalar_type() == torch::kBFloat16
                          && y.dim() == 2 && y.size(1) == N_TOTAL,
                          "y must be contiguous CUDA bf16 [M,N_MAIN]");
              const int M = y.size(0);
              auto ssq = torch::empty({M, NUM_HEADS_OUT},
                                      y.options().dtype(torch::kFloat32));
              const int total = M * NUM_HEADS_OUT;
              auto stream = at::cuda::getCurrentCUDAStream();
              head_ssq_kernel<<<(total + 7) / 8, 256, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
                  ssq.data_ptr<float>(), total);
              return ssq;
          },
          "Standalone per-(row, head) sum-of-squares over the materialized bf16 "
          "main q [M,N_MAIN] -> ssq [M,num_heads]",
          py::arg("y"));
}
