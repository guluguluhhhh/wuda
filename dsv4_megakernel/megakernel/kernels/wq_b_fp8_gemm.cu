// ============================================================
// wq_b_fp8_gemm.cu
// MERGED wq_b projection (tcgen05 FP8 block-scale) — Kernel + Host + Binding
//
// x_fp8[M,1536] @ w_fp8[73728,1536]^T (main q 65536 + indexer 8192), M in
// [1,128] (host pads to 32-aligned), e4m3 inputs + native DSV4
// activation-1x128 / weight-128x128 UE8M0 scales. Outputs:
//   y[M,65536] fp32 (+ optional fused head_ssq [M,128], RMSNorm scale folding)
//   iq_fp4[M,64,64] + iq_sf[M,64]  (indexer q: fused rope/hadamard/MXFP4)
// Swap-AB (UMMA_N = M_pad), BN=128, 2SM MMA (cta_group::2), block_scale,
// Persistent, Warp-Specialized + async transform warpgroup (threads 256..383).
// SF pipeline: native global SF -> warp2 broadcast/layout -> UTCCP -> TMEM.
// MMA (UTCCP + block_scale) delegated to cluster_mma_fp8.cuh.
// ============================================================
//
// ======================== USAGE ========================
// Build (JIT): see test/test_wq_b_fp8_gemm.py::load_module (sm_100a, CUTLASS inc).
//
// DEFAULT path (256 threads; GEMM + fused head_ssq; indexer post-processing OFF):
//
//   y, iq_fp4, iq_sf, ssq, iq_ws = module.wq_b_proj_gemm_merged(
//       x_fp8,     # [M,1536]  e4m3, M in [1,128] (host pads to 32-aligned)
//       x_sf,      # [M,12]    UE8M0 (float8_e8m0fnu or raw uint8), act 1x128
//       w_fp8,     # [73728,1536] e4m3, main-q 65536 rows ++ indexer 8192 rows
//       w_sf,      # [576,12]  UE8M0, weight 128x128
//       q_pos,     # [M] int32 rotary positions
//       rope_cos,  # [max_pos,32] fp32   rope_sin,  # [max_pos,32] fp32
//   )
//   y      [M,65536] fp32   : main q
//   ssq    [M,128]   fp32   : per-head sum-of-squares (RMSNorm scale folding:
//                             consumer applies rsqrt(ssq/512 + eps))
//   iq_ws  [M,64,128] fp32  : drained indexer q; finish it with
//       iq_fp4, iq_sf = module.idx_postprocess_standalone(iq_ws, q_pos, rope_cos, rope_sin)
//   (iq_fp4/iq_sf from the merged call are GARBAGE in this mode)
//
// Optional kwargs:
//   head_ssq   = None : pass your own ZERO-INITIALIZED fp32 [M,128] buffer to
//                       reuse across layers (skips the internal allocation)
//   enable_ssq = True : False (w/o buffer) -> bit-identical plain GEMM; the ssq
//                       slot disappears from the returned list
//   mock_post  = True : False -> 384 threads, the async transform warpgroup
//                       fuses rope+hadamard+fp4quant in-kernel (iq_fp4/iq_sf
//                       become valid, no iq_ws in the returned list)
// Return order: [y, iq_fp4, iq_sf, ssq?, iq_ws?, timing?]  (? = per the flags).
//
// Non-Python callers: the kernel is a plain __global__ template; replicate the
// host duties in run_wq_b() -- TMA descs via cuTensorMapEncodeTiled (exact box/
// swizzle params), ZERO-INIT ssq buffer, iq scratch [M,64,128] fp32, blockDim
// 256/384 tied to mock_post, cluster (2,1,1) launch attr, dynamic smem
// cudaFuncSetAttribute, and the 16-entry ptr_args order.
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
// The SEPARATE-KERNEL baseline for the fused head_ssq accumulation: reads the
// fp32 main-q output once and reduces per-(row, head) sum-of-squares. One warp
// per 512-col head (float4 loads, 16 values per lane, 5-level shuffle tree) --
// memory-bound at the read of y, i.e. as fast as this op can be standalone.
__global__ void __launch_bounds__(256, 1)
head_ssq_kernel(
    const float* __restrict__ y,        // [M, 65536] fp32
    float* __restrict__ ssq,            // [M, 128]
    int total_heads)                    // M * 128
{
    const int h = blockIdx.x * 8 + (int)(threadIdx.x >> 5);   // row*128 + head
    if (h >= total_heads)
        return;                                                // warp-uniform
    const uint32_t lane = threadIdx.x & 31;
    const float* p = y + (int64_t)h * HEAD_DIM_OUT + lane * 4;
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float4 v = *reinterpret_cast<const float4*>(p + k * 128);
        s += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    #pragma unroll
    for (int x = 16; x >= 1; x >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, x);
    if (lane == 0)
        ssq[h] = s;
}

// ======================== Kernel ========================
// MERGED-ONLY (N = 73728): the indexer wq_b weight is concatenated after the
// main q weight. iq tiles are DRAINED by the store warps (TMEM -> L2 scratch ->
// tmem_empty) and post-processed by the dedicated ASYNC TRANSFORM WARPGROUP
// (threads 256..383), fully overlapped with the remaining weight streaming.
template <int M_TPL, bool kProfile, bool kSsq>
__global__ void __launch_bounds__(TPB_IDX, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,    // activation [M,K] e4m3, K-major
    const __grid_constant__ CUtensorMap desc_B,    // weight     [N,K] e4m3, K-major
    const uint8_t* __restrict__ x_sf,               // [M,K/128] UE8M0
    const uint8_t* __restrict__ w_sf,               // [N/128,K/128] UE8M0
    const __grid_constant__ CUtensorMap desc_D,    // output [M,N] FP32 row-major
    int problem_m,
    int num_blocks,
    float* __restrict__ head_ssq,                  // [M,128] per-head sum-of-squares (RED-accumulated; nullptr off)
    // fused indexer projection outputs + rotary metadata
    uint8_t* __restrict__ iq_fp4,                  // [M, 64, 64] packed fp4
    int* __restrict__ iq_sf,                       // [M, 64] packed-ue8m0
    float* __restrict__ iq_scratch,                // [M, 64, 128] fp32 drain buffer
    const int* __restrict__ q_pos,                 // [M] rotary positions
    const float* __restrict__ rope_cos,           // [max_pos, 32]
    const float* __restrict__ rope_sin,           // [max_pos, 32]
    int mock_post,                                 // 1 = drain only, skip the transform (baseline)
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
    int num_clusters    = num_blocks / CLUSTER_SIZE;
    int cluster_id      = blockIdx.x / CLUSTER_SIZE;
    int num_tiles_total = NUM_TILES_TOTAL;

    // ======== WARP 0: TMA PRODUCER (both CTAs, plain per-CTA loads) ========
    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage = 0, phase = 0, persistent_iter = 0;
        auto advance = [&]() { stage = (stage + 1) % NS; if (stage == 0) phase ^= 1; };

        // REVERSED tile order: iq tiles (the LAST 32) run FIRST so their drain +
        // async transform overlap the remaining main tiles' weight stream; all
        // roles share the mapping (pipeline pairing unchanged).
        for (int it_t = cluster_id; it_t < num_tiles_total; it_t += num_clusters) {
            const int tile_id = num_tiles_total - 1 - it_t;
            int n_base = tile_id * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N; // weight N (per CTA)
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
                    tma::copy_2d_fp8(&desc_A, &s.full_barriers[stage], sa, k_off, m_base);
                    tma::copy_2d_fp8(&desc_B, &s.full_barriers[stage], sb, k_off, n_base);

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

        // REVERSED tile order: iq tiles (the LAST 32) run FIRST so their drain +
        // async transform overlap the remaining main tiles' weight stream; all
        // roles share the mapping (pipeline pairing unchanged).
        for (int it_t = cluster_id; it_t < num_tiles_total; it_t += num_clusters) {
            const int tile_id = num_tiles_total - 1 - it_t;
          const int n_base = tile_id * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N;
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
        // REVERSED tile order (same mapping as the other roles); the MMA warp
        // itself is tile-id agnostic -- it only paces stages/accums.
        for (int it_t = cluster_id; it_t < num_tiles_total; it_t += num_clusters) {
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

        constexpr int NUM_STORES         = UMMA_N_T / STORE_BLOCK_M;          // M/16 (no padding rows)
        constexpr int NUM_TMEM_SUBROWS   = STORE_BLOCK_M / 8;                  // 2
        constexpr int NUM_N_STORE_ATOMS  = STORE_BLOCK_N / STORE_BLOCK_N_ATOM; // 4
        constexpr int SMEM_CD_PER_STAGE_T = SMEM_CD_PER_STAGE;                 // 8192

        uint32_t persistent_iter = 0;
        // REVERSED tile order: iq tiles (the LAST 32) run FIRST so their drain +
        // async transform overlap the remaining main tiles' weight stream; all
        // roles share the mapping (pipeline pairing unchanged).
        for (int it_t = cluster_id; it_t < num_tiles_total; it_t += num_clusters) {
            const int tile_id = num_tiles_total - 1 - it_t;
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
            // Indexer tile: whole CTA tile is ONE indexer head row-block
            // (BLOCK_N == IDX_HEAD_DIM); drained + post-processed async, no fp32
            // TMA store, no ssq, smem_cd untouched.
            const bool is_idx = base_n >= N_TOTAL;

            if (is_idx) {
                    // DRAIN-FIRST: nothing heavy on the TMEM-drain path (the accum
                    // stage gates the MMA). Dump the accum to the fp32 scratch
                    // (lane = col -> 128B-coalesced rows, L2-resident), release
                    // tmem_empty, hand off to the async transform warpgroup.
                    const int head = (base_n - N_TOTAL) / IDX_HEAD_DIM;
                    float* dst = iq_scratch
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
                            float* drow = dst + (int64_t)(st * STORE_BLOCK_M + i * 8) * N_IDX;
                            #pragma unroll
                            for (int row = 0; row < 8; ++row)
                                if (st * STORE_BLOCK_M + i * 8 + row < problem_m)
                                    drow[(int64_t)row * N_IDX] = __uint_as_float(vals[row]);
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

                    // ---- optional per-head sum-of-squares (RMSNorm scale folding) ----
                    // vals[8] = 8 M rows; the warp's 32 lanes cover 32 cols of ONE
                    // 512-col head. Multi-row butterfly co-reduction: 16 shuffles
                    // for all 8 row sums (vs 8 x 5-level trees); after the xor-4
                    // step lane l holds row (l>>2)&7. The consumer RED is order-free
                    // (redux.sync f32 is min/max-only on sm_100a, so shuffles).
                    if constexpr (kSsq) {
                        float v[8];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float f = __uint_as_float(vals[j]);
                            v[j] = f * f;
                        }
                        float t8[8];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            t8[j] = __shfl_xor_sync(0xffffffffu, v[j], 16);
                        const bool hi16 = (lane_id & 16) != 0;   // keep rows 4..7
                        float w4[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            w4[j] = hi16 ? (v[j + 4] + t8[j + 4]) : (v[j] + t8[j]);
                        float t4[4];
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            t4[j] = __shfl_xor_sync(0xffffffffu, w4[j], 8);
                        const bool hi8 = (lane_id & 8) != 0;     // keep back half
                        float w2[2];
                        #pragma unroll
                        for (int j = 0; j < 2; ++j)
                            w2[j] = hi8 ? (w4[j + 2] + t4[j + 2]) : (w4[j] + t4[j]);
                        const float ta = __shfl_xor_sync(0xffffffffu, w2[0], 4);
                        const float tb = __shfl_xor_sync(0xffffffffu, w2[1], 4);
                        float sq = (lane_id & 4) ? (w2[1] + tb) : (w2[0] + ta);
                        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
                        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
                        if ((lane_id & 3) == 0)
                            s.ssq_scratch[epi_warp_idx]
                                [st * STORE_BLOCK_M + i * 8 + (int)((lane_id >> 2) & 7)] = sq;
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
                if (m < problem_m)
                    atomicAdd(head_ssq + (size_t)m * NUM_HEADS_OUT + base_n / HEAD_DIM_OUT, v);
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

    // ======== ASYNC TRANSFORM WARPGROUP (threads 256..383) ========
    // Replays the reversed tile schedule, picks the iq tiles, waits on barrier 1
    // (store warps arrive right after the drain) and runs the transform from the
    // L2 scratch -- off the GEMM's critical path; only has to finish before the
    // trailing cluster_sync.
    else if (warp_id >= (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        if (!mock_post)
        for (int it_t = cluster_id; it_t < num_tiles_total; it_t += num_clusters) {
            const int tile_id = NUM_TILES_TOTAL - 1 - it_t;   // reversed order
            const int base_n = tile_id * CLUSTER_BLOCK_N + cta_rank * BLOCK_N;
            if (base_n < N_TOTAL)
                continue;                                     // main-q tile: not ours
            const int head = (base_n - N_TOTAL) / IDX_HEAD_DIM;

            cutlass::arch::NamedBarrier::sync(
                NUM_STORE_THREADS + NUM_XFORM_THREADS, 1);    // drain done + mem fence

            const uint32_t t = threadIdx.x - (NUM_NON_EPI_THREADS + NUM_STORE_THREADS);
            const uint32_t r = t >> 3, e8 = t & 7;            // 16 rows x 8 lanes
            const int num_batches = M_TPL / STORE_BLOCK_M;    // whole-tile M rows
            // Software-pipelined: prefetch batch b+1's L2 reads while computing
            // batch b (see IdxRowIn)
            auto row_ptr = [&](int m) {
                return iq_scratch + ((int64_t)m * IDX_NUM_HEADS + head) * IDX_HEAD_DIM;
            };
            IdxRowIn cur, nxt;
            // padding rows (m >= problem_m) CLAMP to the last valid row: warps
            // stay converged for the full-mask shuffles, stores are suppressed
            const auto clamp_m = [&](int m) {
                return m < problem_m ? m : problem_m - 1;
            };
            idx_row_load(row_ptr(clamp_m((int)r)), e8, clamp_m((int)r),
                         q_pos, rope_cos, rope_sin, cur);
            for (int b = 0; b < num_batches; ++b) {
                const int m = b * STORE_BLOCK_M + (int)r;     // base_m == 0 (swap)
                if (b + 1 < num_batches)
                    idx_row_load(row_ptr(clamp_m(m + STORE_BLOCK_M)), e8,
                                 clamp_m(m + STORE_BLOCK_M),
                                 q_pos, rope_cos, rope_sin, nxt);
                idx_row_compute(cur, e8, clamp_m(m), head, iq_fp4, iq_sf,
                                m < problem_m);
                cur = nxt;
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
// Swap-AB path only (M <= 128, decode). kSsq is a COMPILE-TIME dispatch
// dimension (DeepGEMM JIT-config discipline): the ssq-off binary carries no
// dead branch in the hot loops.
template <bool kProfile, bool kSsq>
static void* get_kernel_ptr(int M) {
    switch (M) {
        case 32:  return (void*)&wq_b_proj_kernel<32,  kProfile, kSsq>;
        case 64:  return (void*)&wq_b_proj_kernel<64,  kProfile, kSsq>;
        case 96:  return (void*)&wq_b_proj_kernel<96,  kProfile, kSsq>;
        case 128: return (void*)&wq_b_proj_kernel<128, kProfile, kSsq>;
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

// MERGED FP8 block-scale run (N = 73728 = main q 65536 + indexer 8192). Inputs
// use the native DSV4 checkpoint/runtime layout:
//   x_fp8 [M,K] e4m3 ; x_sf [M,K/128] UE8M0 (activation 1x128)
//   w_fp8 [N_MERGED,K] e4m3 ; w_sf [N_MERGED/128,K/128] UE8M0 (weight 128x128)
// Returns [y fp32 [M,65536], iq_fp4 i8 [M,64,64], iq_sf i32 [M,64], (timing)]:
// the last N_IDX weight rows are the indexer wq_b; their tiles are drained and
// rope+hadamard+fp4quant runs in the async transform warpgroup. Requires
// q_pos [M] i32 and rope_cos/rope_sin [max_pos,32] f32 (CPU-precomputed).
// head_ssq (optional): ZERO-INITIALIZED fp32 [M,128]; the epilogue
// RED-accumulates per-(row, head) sum-of-squares of the MAIN q segment
// (RMSNorm scale folding). mock_post: drain-only baseline (no transform).
static std::vector<torch::Tensor> run_wq_b(
    torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w_fp8, torch::Tensor w_sf, bool profile,
    c10::optional<torch::Tensor> head_ssq,
    torch::Tensor q_pos, torch::Tensor rope_cos, torch::Tensor rope_sin,
    bool mock_post = false, bool enable_ssq = true)
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
    auto iq_fp4 = torch::empty({M, IDX_NUM_HEADS, IDX_HEAD_DIM / 2},
                               x_fp8.options().dtype(torch::kInt8));
    auto iq_sf = torch::empty({M, IDX_NUM_HEADS}, x_fp8.options().dtype(torch::kInt32));
    // drain-first workspace: the epilogue dumps the iq accum here (L2-resident)
    // to release the TMEM stage before the transform pass
    auto iq_ws = torch::empty({M, IDX_NUM_HEADS, IDX_HEAD_DIM},
                              x_fp8.options().dtype(torch::kFloat32));
    const int* q_pos_ptr = q_pos.data_ptr<int>();
    const float* cos_ptr = rope_cos.data_ptr<float>();
    const float* sin_ptr = rope_sin.data_ptr<float>();
    uint8_t* iq_fp4_ptr = reinterpret_cast<uint8_t*>(iq_fp4.data_ptr());
    int* iq_sf_ptr = iq_sf.data_ptr<int>();
    float* iq_ws_ptr = iq_ws.data_ptr<float>();

    auto out = torch::empty({M, N_TOTAL}, x_fp8.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_ptr   = reinterpret_cast<const __nv_fp8_e4m3*>(x_fp8.data_ptr());
    auto w_ptr   = reinterpret_cast<const __nv_fp8_e4m3*>(w_fp8.data_ptr());
    auto xsf_ptr = reinterpret_cast<const uint8_t*>(x_sf.data_ptr());
    auto wsf_ptr = reinterpret_cast<const uint8_t*>(w_sf.data_ptr());
    auto out_ptr = reinterpret_cast<float*>(out.data_ptr());

    // K-major 128B-swizzled operands; desc_D covers the main q segment only
    // (iq tiles never TMA-store). desc_A/desc_D use the REAL M (globalDim) with
    // the PADDED box: TMA zero-fills OOB loads and clips OOB stores.
    CUtensorMap desc_A   = make_tma_desc_fp8_2d(x_ptr, M, K_DIM, M_pad / NUM_MULTICAST, BLOCK_K);
    CUtensorMap desc_B   = make_tma_desc_fp8_2d(w_ptr, N_MERGED, K_DIM, LOAD_BLOCK_N, BLOCK_K);
    CUtensorMap desc_D   = make_tma_desc_fp32_2d(
        out_ptr, M, N_TOTAL, STORE_BLOCK_M, STORE_BLOCK_N_ATOM);

    // Grid: persistent, cluster of 2 CTAs.
    static const int num_SMs = []() {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
        return n;
    }();
    TORCH_CHECK(num_SMs >= CLUSTER_SIZE,
                "wq_b FP8 requires at least ", CLUSTER_SIZE, " SMs, got ", num_SMs);
    constexpr int num_tiles = NUM_N_TILES_MERGED;
    const int total_cta = num_tiles * CLUSTER_SIZE;
    const int max_clusters = min(num_SMs, total_cta) / CLUSTER_SIZE;
    int num_clusters = max_clusters > 0 ? max_clusters : 1;
    while (num_clusters > 1 && num_tiles % num_clusters != 0) --num_clusters;
    if (const char* e = std::getenv("WQ_B_CLUSTERS")) {
        const int req = atoi(e);
        if (req > 0) num_clusters = req < max_clusters ? req : max_clusters;
    }
    int grid_size = num_clusters * CLUSTER_SIZE;   // non-const: passed via ptr_args
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
    // [M,128] buffer (reuse across layers); otherwise one is allocated here.
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

    const bool want_ssq = ssq_ptr != nullptr;
    void* kernel_ptr = profile
        ? (want_ssq ? get_kernel_ptr<true, true>(M_pad)  : get_kernel_ptr<true, false>(M_pad))
        : (want_ssq ? get_kernel_ptr<false, true>(M_pad) : get_kernel_ptr<false, false>(M_pad));
    int smem_bytes = get_smem_bytes(M_pad);
    TORCH_CHECK(kernel_ptr != nullptr && smem_bytes > 0, "Unsupported M=", M);

    static bool smem_configured[2][2][9] = {{{false}}};
    const int m_idx = M_pad / 32;
    const int p_idx = profile ? 1 : 0;
    const int s_idx = want_ssq ? 1 : 0;
    if (!smem_configured[p_idx][s_idx][m_idx]) {
        auto attr_err = cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(attr_err), " smem_bytes=", smem_bytes);
        smem_configured[p_idx][s_idx][m_idx] = true;
    }

    {
        dim3 grid(grid_size, 1, 1);
        // Fused post-processing OFF by default (mock_post=true): launch the plain
        // 256 threads, keeping warps 8..11 free for future in-kernel cuda-core
        // work. Enabling the fused path launches the async transform warpgroup
        // too (384) -- the blockDim is TIED to mock_post so the drain handoff
        // barrier can never wait on threads that were not launched.
        dim3 block(mock_post ? TPB : TPB_IDX, 1, 1);
        int mock_i = mock_post ? 1 : 0;

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

        void* ptr_args[] = {
            &desc_A, &desc_B, &xsf_ptr, &wsf_ptr, &desc_D,
            &M, &grid_size, &ssq_ptr,
            &iq_fp4_ptr, &iq_sf_ptr, &iq_ws_ptr, &q_pos_ptr, &cos_ptr, &sin_ptr,
            &mock_i, &prof_dev
        };
        auto err = cudaLaunchKernelExC(&config, kernel_ptr, ptr_args);
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    }

    std::vector<torch::Tensor> ret = {out, iq_fp4, iq_sf};
    if (ssq_ptr) ret.push_back(ssq_t);     // present unless enable_ssq=false w/o buffer
    if (mock_post) ret.push_back(iq_ws);   // export the drained fp32 iq (bitwise test hook)
    if (profile) ret.push_back(timing);
    return ret;
}

// ======================== PyTorch Binding ========================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm_merged",
          [](torch::Tensor x_fp8, torch::Tensor x_sf,
             torch::Tensor w_fp8, torch::Tensor w_sf,
             torch::Tensor q_pos, torch::Tensor rope_cos, torch::Tensor rope_sin,
             c10::optional<torch::Tensor> head_ssq, bool mock_post, bool enable_ssq) {
              return run_wq_b(x_fp8, x_sf, w_fp8, w_sf, /*profile=*/false, head_ssq,
                              q_pos, rope_cos, rope_sin, mock_post, enable_ssq);
          },
          "MERGED wq_b + indexer wq_b (w [N+8192,K]), swap path, M in [1,128]. "
          "Returns [y fp32 [M,65536], iq_fp4 i8 [M,64,64], iq_sf i32 [M,64], "
          "ssq fp32 [M,128] (unless enable_ssq=False), iq_ws (if mock_post)]. "
          "head_ssq: optional caller-owned ZERO-INIT buffer; default None allocates. "
          "DEFAULT mock_post=True: 256 threads, GEMM+ssq only -- iq_fp4/iq_sf stay "
          "garbage; run idx_postprocess_standalone over iq_ws. mock_post=False: "
          "384 threads, fuses rope+hadamard+fp4quant in-kernel",
          py::arg("x_fp8"), py::arg("x_sf"), py::arg("w_fp8"), py::arg("w_sf"),
          py::arg("q_pos"), py::arg("rope_cos"), py::arg("rope_sin"),
          py::arg("head_ssq") = c10::nullopt,
          py::arg("mock_post") = true,
          py::arg("enable_ssq") = true);
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
                          && iq_f32.scalar_type() == torch::kFloat
                          && iq_f32.dim() == 3 && iq_f32.size(1) == IDX_NUM_HEADS
                          && iq_f32.size(2) == IDX_HEAD_DIM,
                          "iq_f32 must be contiguous CUDA fp32 [M,64,128]");
              const int M = iq_f32.size(0);
              TORCH_CHECK(M >= 1, "empty batch");
              auto iq_fp4 = torch::empty({M, IDX_NUM_HEADS, IDX_HEAD_DIM / 2},
                                         iq_f32.options().dtype(torch::kInt8));
              auto iq_sf = torch::empty({M, IDX_NUM_HEADS},
                                        iq_f32.options().dtype(torch::kInt32));
              const int rows = M * IDX_NUM_HEADS;
              auto stream = at::cuda::getCurrentCUDAStream();
              idx_post_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
                  iq_f32.data_ptr<float>(), q_pos.data_ptr<int>(),
                  rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
                  reinterpret_cast<uint8_t*>(iq_fp4.data_ptr()),
                  iq_sf.data_ptr<int>(), rows);
              return std::vector<torch::Tensor>{iq_fp4, iq_sf};
          },
          "Standalone rope+hadamard+fp4quant over fp32 iq [M,64,128] -> "
          "(iq_fp4 [M,64,64] i8, iq_sf [M,64] i32) -- the separate-kernel baseline",
          py::arg("iq_f32"), py::arg("q_pos"), py::arg("rope_cos"), py::arg("rope_sin"));
    m.def("head_ssq_standalone",
          [](torch::Tensor y) {
              TORCH_CHECK(y.is_cuda() && y.is_contiguous()
                          && y.scalar_type() == torch::kFloat
                          && y.dim() == 2 && y.size(1) == N_TOTAL,
                          "y must be contiguous CUDA fp32 [M,65536]");
              const int M = y.size(0);
              auto ssq = torch::empty({M, NUM_HEADS_OUT},
                                      y.options().dtype(torch::kFloat32));
              const int total = M * NUM_HEADS_OUT;
              auto stream = at::cuda::getCurrentCUDAStream();
              head_ssq_kernel<<<(total + 7) / 8, 256, 0, stream>>>(
                  y.data_ptr<float>(), ssq.data_ptr<float>(), total);
              return ssq;
          },
          "Standalone per-(row, head) sum-of-squares over the fp32 main q "
          "[M,65536] -> ssq [M,128] -- the separate-kernel baseline for the "
          "fused head_ssq accumulation",
          py::arg("y"));
}
