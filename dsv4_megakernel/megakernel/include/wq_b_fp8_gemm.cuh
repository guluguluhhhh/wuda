#pragma once
// ============================================================
// wq_b_fp8_gemm.cuh — config + PTX/TMA helpers for the MERGED wq_b projection
// (kernel/host/usage: kernels/wq_b_fp8_gemm.cu).
//
// Shape: x[M,1536] e4m3 @ w[73728,1536]^T (main q ++ indexer), M_pad in
// {32,64,96,128}. Swap-AB: MMA A = weight (UMMA_M=256 along N), B = activation
// (UMMA_N = M_pad along M), both K-major 128B-swizzled. Native DSV4 scales:
// activation 1x128, weight 128x128, UE8M0; warp2 expands each K128 byte to four
// K32 scale IDs (tcgen05 mxf8f6f4 granularity). Block-scale MMA engine lives in
// cluster_mma_fp8.cuh.
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// CUTLASS/CuTe headers
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>

// ======================== Configuration ========================
namespace wq_b {

// ---- Problem dimensions (fixed for wq_b projection) ----
static constexpr int K_DIM    = 1536;
static constexpr int N_TOTAL  = 65536;  // 128 heads x 512 dim
// ---- Output head geometry (per-head RMSNorm scale folding) ----
// The epilogue can ACCUMULATE per-(row, head) sum-of-squares into an optional
// zero-initialized head_ssq[M, NUM_HEADS_OUT] fp32 buffer (fire-and-forget f32
// RED atomics; each 128-col N block lies inside ONE 512-col head). The consumer
// finalizes s_h = rsqrt(ssq/512 + eps) and folds it into its own q pass -- the
// normalized y never round-trips HBM and TMEM never has to hold a whole head.
static constexpr int HEAD_DIM_OUT  = 512;
static constexpr int NUM_HEADS_OUT = N_TOTAL / HEAD_DIM_OUT;   // 128
static constexpr int BLOCK_K  = 128;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K; // 12
static constexpr int QUANT_BLOCK_K = 128; // native DSV4 FP8 quantization granularity
static constexpr int UMMA_SF_GRAN_K = 32; // hardware block-scale consumption granularity
static constexpr int WEIGHT_QUANT_BLOCK_N = 128;
static constexpr int NUM_WEIGHT_SF_ROWS =
    (N_TOTAL + WEIGHT_QUANT_BLOCK_N - 1) / WEIGHT_QUANT_BLOCK_N; // 512

// ---- Merged indexer projection (CSA stage 7 Idx_WProj fused into this GEMM) ----
// The indexer wq_b shares A (= qr) and K, so its weight is CONCATENATED along N:
// w[N_TOTAL + N_IDX, K]. The iq segment is tile-aligned for the swap path
// (N_IDX % CLUSTER_BLOCK_N == 0) and each 128-col CTA tile is exactly ONE indexer
// head row -- the epilogue post-processes it in place (rope + hadamard + fp4
// quant) instead of storing fp32. Swap path (M <= 128) only.
static constexpr int IDX_NUM_HEADS = 64;
static constexpr int IDX_HEAD_DIM  = 128;
static constexpr int N_IDX         = IDX_NUM_HEADS * IDX_HEAD_DIM;              // 8192
static constexpr int N_MERGED      = N_TOTAL + N_IDX;                           // 73728
static constexpr int NUM_WEIGHT_SF_ROWS_MERGED =
    N_MERGED / WEIGHT_QUANT_BLOCK_N;                                            // 576

// ---- Element sizes (bytes) ----
static constexpr int FP8_ELEM_SIZE = 1;   // e4m3

// ---- Cluster / multicast (swap-AB: cluster_n = 2) ----
static constexpr int CLUSTER_SIZE  = 2;
static constexpr int NUM_MULTICAST = CLUSTER_SIZE; // 2
static constexpr bool IS_MULTICAST_ON_A = true;

// ---- MMA instruction shape (2SM) ----
// UMMA_M = LAYOUT_AD_M(128) * kNumMulticast(2) = 256 (along problem-N)
// UMMA_N = M (along problem-M; per-SwapDims template, no padding).
static constexpr int LAYOUT_AD_M = 128;
static constexpr int UMMA_M = LAYOUT_AD_M * NUM_MULTICAST; // 256
static constexpr int UMMA_K = 32;                          // FP8 block-scale UMMA K
static constexpr int BM       = 128;                       // max swap-path M (scratch sizing)

// ---- N tiling ----
static constexpr int BLOCK_N        = 128;
static constexpr int CLUSTER_BLOCK_N = BLOCK_N * NUM_MULTICAST;  // 256
static constexpr int LOAD_BLOCK_N   = BLOCK_N;                    // 128
static constexpr int NUM_N_TILES    = N_TOTAL / CLUSTER_BLOCK_N;  // 256
static constexpr int NUM_N_TILES_MERGED = N_MERGED / CLUSTER_BLOCK_N;  // 288 (swap/kIdx only)
static_assert(N_IDX % CLUSTER_BLOCK_N == 0, "iq segment must be whole cluster tiles");
static_assert(BLOCK_N == IDX_HEAD_DIM, "one CTA tile must equal one indexer head row");

// ---- Layout: both A(weight) and B(activation) are K-major ----
static constexpr auto MAJOR_A = cute::UMMA::Major::K;
static constexpr auto MAJOR_B = cute::UMMA::Major::K;

// ---- Swizzle (128B). K-major fp8: BLOCK_K*1 = 128 bytes -> 128B swizzle. ----
static constexpr int SWIZZLE_A  = 128;
static constexpr int SWIZZLE_B  = 128;
static constexpr int SWIZZLE_CD = 128;

// ---- Scale-factor (block-scale) layout ----
static constexpr int SF_IDS_PER_QUANT_BLOCK = QUANT_BLOCK_K / UMMA_SF_GRAN_K; // 4
static constexpr uint32_t UE8M0_ONE = 0x7fu;
static_assert(BLOCK_K == QUANT_BLOCK_K && SF_IDS_PER_QUANT_BLOCK == 4,
              "one pipeline stage must expand one K128 scale to four K32 IDs");
static constexpr int NUM_UTCCP_ALIGNED = 128;
static constexpr int SF_BLOCK_N        = ((BLOCK_N + NUM_UTCCP_ALIGNED - 1) / NUM_UTCCP_ALIGNED) * NUM_UTCCP_ALIGNED; // 128
static constexpr int NUM_SFB_TMEM_COLS = SF_BLOCK_N / 32;    // 4
static constexpr int SMEM_SFB_PER_STAGE = SF_BLOCK_N * (int)sizeof(uint32_t); // 512

// ---- Pipeline ----
static constexpr int NUM_EPI_STAGES      = 2;  // TMEM accumulator double buffer
static constexpr int NUM_TMA_STORE_STAGES = 2;

// ---- Threads: warp0 TMA, warp1 MMA(leader), warp2 SF expand/layout, warps4-7 epilogue ----
static constexpr int TPB                 = 256;
static constexpr int NUM_NON_EPI_THREADS = 128;
static constexpr int NUM_EPI_THREADS     = 128;
static constexpr int NUM_STORE_THREADS   = 128;
// [kIdx] async transform warpgroup (warps 8..11): merged-indexer instances launch
// TPB_IDX threads; the extra 128 run the rope/hadamard/fp4 chain off the GEMM's
// critical path (the epilogue warps are near-saturated by the TMEM-load train).
static constexpr int NUM_XFORM_THREADS   = 128;
static constexpr int TPB_IDX             = TPB + NUM_XFORM_THREADS;   // 384

// ---- Epilogue store tile (swap-AB, FP32 output) ----
static constexpr int STORE_BLOCK_M      = 16;                             // M-rows per store stage
static constexpr int STORE_BLOCK_N      = BLOCK_N;                        // 128
static constexpr int STORE_BLOCK_N_ATOM = SWIZZLE_CD / (int)sizeof(float); // 128/4 = 32

// ---- Per-stage SMEM for weight B (constant); A depends on M via SwapDims ----
static constexpr int SMEM_B_PER_STAGE  = LOAD_BLOCK_N * BLOCK_K * FP8_ELEM_SIZE;        // 128*128*1 = 16384
static constexpr int SMEM_CD_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(float); // 16*128*4 = 8192
static constexpr int SMEM_CD_TOTAL     = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES;      // 16384

// SMEM capacity budget (SM100)
static constexpr int SMEM_CAPACITY = 232448;

static constexpr int div_up(int a, int b)  { return (a + b - 1) / b; }
static constexpr int align_up(int a, int b) { return div_up(a, b) * b; }
static constexpr int num_aligned_tmem_cols(int c) {
    if (c <= 32)  return 32;
    if (c <= 64)  return 64;
    if (c <= 128) return 128;
    if (c <= 256) return 256;
    return 512;
}

// ---- Compile-time helpers parameterised on M (UMMA_N = M, no padding) ----
// One tile covers the WHOLE problem M: UMMA_N = M, LOAD_BLOCK_M = M/2 per CTA.
// Smaller M shrinks SMEM_A and the TMEM accumulator footprint, so the pipeline
// deepens automatically (M=32 -> ~11 stages, matching DeepGEMM's pick).
template <int M_> struct SwapDims {
    static constexpr int BLOCK_M          = M_;                    // true M (<= 128)
    static constexpr int NUM_M_SUB        = div_up(M_, BM);        // 1 for all swap Ms
    static constexpr int UMMA_N           = M_;                    // MMA-N = problem M
    static constexpr int LOAD_BLOCK_M     = M_ / NUM_MULTICAST;    // M/2 per CTA
    static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * FP8_ELEM_SIZE;

    // SFA stays UTCCP-aligned to 128 rows (rows >= M carry UE8M0_ONE padding;
    // the MMA only consumes the first M).
    static constexpr int SF_BLOCK_M         = align_up(M_, NUM_UTCCP_ALIGNED);        // 128
    static constexpr int SMEM_SFA_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);     // 512
    static constexpr int NUM_SFA_TMEM_COLS  = SF_BLOCK_M / 32;                        // 4

    static constexpr int SMEM_PER_STAGE = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE +
                                          SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE;

    // TMEM layout: [accum (UMMA_N*NUM_EPI_STAGES)] [SFA cols] [SFB cols]
    static constexpr int NUM_ACCUM_TMEM_COLS = UMMA_N * NUM_EPI_STAGES;               // 2M
    static constexpr int TMEM_START_SFA      = NUM_ACCUM_TMEM_COLS;
    static constexpr int TMEM_START_SFB      = NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS;
    static constexpr int NUM_TMEM_COLS = num_aligned_tmem_cols(
        NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS);

    // Number of pipeline stages fitting the SMEM budget.
    // Overhead: smem_cd + barriers(full/empty/with_sf per stage + tmem full/empty) + tmem_ptr
    //           + the head_ssq per-warp scratch (4 x BM floats, RMSNorm scale folding)
    //           + the fused-quant grid-barrier target + per-row block-partial
    //             ssq scratch (alignas(16) u32 + 12 floats -> 64B).
    // (The fused indexer compressor is fully register-resident -- no smem term.)
    static constexpr int SMEM_BARRIERS = (16 * 3 + NUM_EPI_STAGES * 2) * 8;
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_BARRIERS + 8
                                         + 4 * BM * (int)sizeof(float) + 64;
    static constexpr int STAGES_RAW    = (SMEM_CAPACITY - SMEM_OVERHEAD) / SMEM_PER_STAGE;
    static constexpr int NUM_STAGES    = STAGES_RAW > 12 ? 12 : STAGES_RAW;
};

} // namespace wq_b

// ======================== Barrier alias ========================
namespace mma_desc {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
} // namespace mma_desc

// ======================== PTX Wrappers ========================
// Block-scale MMA + descriptors live in cluster_mma_fp8.cuh. Here we keep only the
// helpers used by the producer / SF layout / epilogue / init in this kernel.
namespace ptx {

// TMEM alloc/dealloc for 2SM
__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t smem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(num_cols));
}
__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(num_cols));
}

// Fences
__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// TMEM load: 32dp32b, x4 (4 FP32 per lane) — used by the non-swap FP32 store
__device__ __forceinline__ void tmem_load_32dp32b4x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
        : "r"(tmem_addr));
}

// TMEM load: 32dp32b, x8 (8 FP32 per lane) — used by the swap-AB FP32 store
__device__ __forceinline__ void tmem_load_32dp32b8x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
    uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
          "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
        : "r"(tmem_addr));
}
__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Shared memory load/store (used by the SF layout warp and the epilogue)
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t v;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}
__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void st_shared_v4_u32(void* ptr, uint32_t v0, uint32_t v1,
                                                 uint32_t v2, uint32_t v3) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

// Cluster utilities
__device__ __forceinline__ uint32_t block_rank_in_cluster() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(rank));
    return rank;
}
__device__ __forceinline__ void cluster_sync() {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}
__device__ __forceinline__ uint32_t get_lane_idx() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
    return lane;
}
__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}" : "=r"(pred));
    return pred != 0;
}

// gpu-scope acquire load (the fused-quant grid barrier's spin read)
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* ptr) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ long long rdclock() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

} // namespace ptx

// ======================== TMA Copy Helpers ========================
namespace tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// Plain (per-CTA) TMA 2D load for FP8 operands. Each CTA loads its own box;
// the cta_group::2 MMA combines the pair across SMs. (Aligned with w1_merged.)
__device__ __forceinline__
void copy_2d_fp8(void const* desc_ptr, Barrier* barrier_ptr,
                 __nv_fp8_e4m3* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, k_idx, mn_idx);
}

// TMA store 2D (FP32 output)
__device__ __forceinline__
void store_2d(void const* desc_ptr, void* smem_ptr,
              uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

} // namespace tma

// ==================== Fused activation-quant prologue (delivery opB port) ====
// PLAIN 1x128 block quant ONLY (no rmsnorm -- isolation experiment): one warp
// quantizes one K128 block, single memory wave, M*12 blocks spread over the
// whole grid. sync is a MONOTONIC ticket (never reset across launches).
namespace wq_b {

struct QuantProArgs {
    const __nv_bfloat16* y = nullptr;   // [M, lda] bf16 (front y[:, :1536] view)
    int64_t lda            = 0;         // row stride in ELEMENTS (lda % 4 == 0)
    const float* gamma     = nullptr;   // [K_DIM] q_norm weight (null => plain quant)
    float eps              = 1e-6f;
    __nv_fp8_e4m3* x_fp8   = nullptr;   // [M, K_DIM] e4m3 out (desc_A's buffer)
    uint8_t* x_sf          = nullptr;   // [M, NUM_K_TILES] UE8M0 out
    uint32_t* sync         = nullptr;   // monotonic grid ticket counter
};

// fp8(e4m3) + native-UE8M0 quant of ONE K128 block, one WARP per block
// (byte-exact port of delivery tile::quant_k128_ue8m0): 4 bf16 elems/lane ->
// warp-shfl amax -> E=ceil(log2(amax/448)) clamped -> v*2^-E as uchar4;
// lane 0 writes the UE8M0 exponent byte. xrow is pre-offset by lane*4.
__device__ __forceinline__ void quant_k128_ue8m0(
    const __nv_bfloat16* __restrict__ xrow, int lane_id,
    __nv_fp8_e4m3* __restrict__ q_dst, uint8_t* __restrict__ sf_dst) {
    const float v0 = __bfloat162float(xrow[0]);
    const float v1 = __bfloat162float(xrow[1]);
    const float v2 = __bfloat162float(xrow[2]);
    const float v3 = __bfloat162float(xrow[3]);
    float block_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        block_amax = fmaxf(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, o));
    int scale_exp = (int)ceilf(log2f(fmaxf(block_amax, 1e-30f) / 448.0f));
    scale_exp = max(-127, min(127, scale_exp));
    const float scale_factor = exp2f(-(float)scale_exp);
    uchar4 packed;
    packed.x = __nv_fp8_e4m3(v0 * scale_factor).__x;
    packed.y = __nv_fp8_e4m3(v1 * scale_factor).__x;
    packed.z = __nv_fp8_e4m3(v2 * scale_factor).__x;
    packed.w = __nv_fp8_e4m3(v3 * scale_factor).__x;
    *reinterpret_cast<uchar4*>(q_dst + lane_id * 4) = packed;
    if (lane_id == 0)
        *sf_dst = (uint8_t)(scale_exp + 127);
}

// rmsnorm(gamma) + 1x128 quant of ONE row, whole-CTA wide (the delivery
// block-parallel shape, NOT the slow one-warp-per-row chain): the CTA's
// warps split the 12 K128 blocks (<= 2 each); ONE memory wave loads v AND
// gamma together; block partial ssq (fixed shfl tree) -> ssq_smem[12] ->
// syncthreads -> every warp sums in FIXED block order (bitwise identical)
// -> r -> qr = bf16((v*r)*gamma) materialized round -> quant_k128 tail.
// MUST be called by ALL threads of the CTA (contains __syncthreads).
__device__ __forceinline__ void qnorm_quant_row_cta(
    const __nv_bfloat16* __restrict__ yrow,      // [1536] strided row base
    const float* __restrict__ gamma,             // [1536]
    float eps, float* __restrict__ ssq_smem,     // [NUM_K_TILES] CTA scratch
    int warp_id, int lane_id, int nwarps,
    __nv_fp8_e4m3* __restrict__ qf_row,          // [1536] e4m3 out
    uint8_t* __restrict__ qsf_row) {             // [12] UE8M0 out
    float v[2][4], g[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int b = warp_id + i * nwarps;
        if (b >= NUM_K_TILES) continue;
        const uint2 raw = *reinterpret_cast<const uint2*>(
            yrow + b * BLOCK_K + lane_id * 4);
        const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const float2 f0 = __bfloat1622float2(p0);
        const float2 f1 = __bfloat1622float2(p1);
        v[i][0] = f0.x; v[i][1] = f0.y; v[i][2] = f1.x; v[i][3] = f1.y;
        const float4 gg = *reinterpret_cast<const float4*>(
            gamma + b * BLOCK_K + lane_id * 4);
        g[i][0] = gg.x; g[i][1] = gg.y; g[i][2] = gg.z; g[i][3] = gg.w;
        float ps = v[i][0] * v[i][0] + v[i][1] * v[i][1]
                 + v[i][2] * v[i][2] + v[i][3] * v[i][3];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            ps += __shfl_xor_sync(0xffffffffu, ps, o);
        if (lane_id == 0)
            ssq_smem[b] = ps;
    }
    __syncthreads();                             // block partials visible
    float ssq = 0.f;
    #pragma unroll
    for (int j = 0; j < NUM_K_TILES; ++j)        // FIXED order -> bitwise
        ssq += ssq_smem[j];
    const float rinv = rsqrtf(ssq * (1.f / (float)K_DIM) + eps);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int b = warp_id + i * nwarps;
        if (b >= NUM_K_TILES) continue;
        float q[4], amax = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __bfloat162float(__float2bfloat16_rn((v[i][j] * rinv) * g[i][j]));
            amax = fmaxf(amax, fabsf(q[j]));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        int e = (int)ceilf(log2f(fmaxf(amax, 1e-30f) / 448.0f));
        e = max(-127, min(127, e));
        const float inv = exp2f(-(float)e);
        uchar4 packed;
        packed.x = __nv_fp8_e4m3(q[0] * inv).__x;
        packed.y = __nv_fp8_e4m3(q[1] * inv).__x;
        packed.z = __nv_fp8_e4m3(q[2] * inv).__x;
        packed.w = __nv_fp8_e4m3(q[3] * inv).__x;
        *reinterpret_cast<uchar4*>(
            reinterpret_cast<uint8_t*>(qf_row) + b * BLOCK_K + lane_id * 4) = packed;
        if (lane_id == 0)
            qsf_row[b] = (uint8_t)(e + 127);
    }
    __syncthreads();                             // ssq_smem reuse guard
}

} // namespace wq_b
