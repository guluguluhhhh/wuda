#pragma once
// ============================================================
// wq_b_gemm.cuh
// tcgen05 FP8 (e4m3) block-scale GEMM — shared config and PTX/TMA helpers
// Aligned with DeepGEMM sm100_fp8_gemm_1d1d + megakernel/w1_merged_fp8_gemm.
//
// Swap-path target: M=32~128 (32-aligned), K=1536, N=65536, e4m3 -> FP32 output.
//   x_fp8[M,1536] @ w_fp8[65536,1536]^T -> y[M,65536] (FP32)
//   A = weight (MMA A-operand, UMMA_M=256 along N), B = activation (UMMA_N=128 along M).
//   Both K-major, 128B swizzle. Native DSV4 quantization: activation 1x128,
//   weight 128x128, scale = UE8M0.
//
// swap_ab=1, 2SM MMA (cta_group::2), Cluster=(2,1,1) -> cluster_n=2, Persistent,
// Warp-Specialized. UMMA_N is pinned to 128 (BM=128); M<128 is zero-padded (TMA OOB
// fill) and the padded output columns are dropped in the epilogue. The block_scale
// MMA is issued by cluster_mma_fp8.cuh; this header carries the config + the SF
// pipeline plumbing (SMEM/TMA helpers) that feeds it.
//
// Native SF layout:
//   x_sf[M, K/128] and w_sf[ceil(N/128), K/128], one UE8M0 byte per K128
//   quantization block. Warp2 expands each byte to four identical K32 scale IDs
//   because tcgen05.mma.kind::mxf8f6f4 consumes scale factors at K32 granularity.
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
static constexpr int BLOCK_K  = 128;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K; // 12
static constexpr int QUANT_BLOCK_K = 128; // native DSV4 FP8 quantization granularity
static constexpr int UMMA_SF_GRAN_K = 32; // hardware block-scale consumption granularity
static constexpr int WEIGHT_QUANT_BLOCK_N = 128;
static constexpr int NUM_WEIGHT_SF_ROWS =
    (N_TOTAL + WEIGHT_QUANT_BLOCK_N - 1) / WEIGHT_QUANT_BLOCK_N; // 512

// ---- Element sizes (bytes) ----
static constexpr int FP8_ELEM_SIZE = 1;   // e4m3

// ---- Cluster / multicast (swap-AB: cluster_n = 2) ----
static constexpr int CLUSTER_SIZE  = 2;
static constexpr int NUM_MULTICAST = CLUSTER_SIZE; // 2
static constexpr bool IS_MULTICAST_ON_A = true;

// ---- MMA instruction shape (2SM) ----
// UMMA_M = LAYOUT_AD_M(128) * kNumMulticast(2) = 256 (along problem-N)
// UMMA_N = 128 FIXED (along problem-M = BM). M<128 padded to 128.
static constexpr int LAYOUT_AD_M = 128;
static constexpr int UMMA_M = LAYOUT_AD_M * NUM_MULTICAST; // 256
static constexpr int UMMA_K = 32;                          // FP8 block-scale UMMA K
static constexpr int BM       = 128;                       // problem-M tile = UMMA_N
static constexpr int UMMA_N_FIXED = BM;                    // 128
static constexpr int LOAD_BLOCK_M_FIXED = BM / NUM_MULTICAST; // 64 activation rows per CTA

// ---- N tiling ----
static constexpr int BLOCK_N        = 128;
static constexpr int CLUSTER_BLOCK_N = BLOCK_N * NUM_MULTICAST;  // 256
static constexpr int LOAD_BLOCK_N   = BLOCK_N;                    // 128
static constexpr int NUM_N_TILES    = N_TOTAL / CLUSTER_BLOCK_N;  // 256

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

// ---- Compile-time helpers parameterised on M (fixed BM=128, M split into subtiles) ----
// Each subtile is a fixed 128-row (BM) tile: UMMA_N=128, LOAD_BLOCK_M=64. M>128 is
// handled by NUM_M_SUB = ceil(M/128) subtiles processed back-to-back per N-tile
// (weight stays L2-resident across subtiles -> HBM weight read once). All per-stage
// SMEM / TMEM sizes are per-subtile (128), so they are identical for every M.
template <int M_> struct SwapDims {
    static constexpr int BLOCK_M          = M_;                    // true M (total)
    static constexpr int NUM_M_SUB        = div_up(M_, BM);        // ceil(M/128) subtiles
    static constexpr int UMMA_N           = UMMA_N_FIXED;          // 128 (per subtile)
    static constexpr int LOAD_BLOCK_M     = LOAD_BLOCK_M_FIXED;    // 64  (per subtile, BM/2)
    static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * FP8_ELEM_SIZE; // 64*128*1 = 8192

    // SFA covers the FULL BLOCK_M(=128) tokens (loaded on each CTA; leader UTCCP
    // broadcasts into both SMs' TMEM). Aligned up to the UTCCP granularity.
    static constexpr int SF_BLOCK_M         = align_up(BM, NUM_UTCCP_ALIGNED);        // 128
    static constexpr int SMEM_SFA_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);     // 512
    static constexpr int NUM_SFA_TMEM_COLS  = SF_BLOCK_M / 32;                        // 4

    static constexpr int SMEM_PER_STAGE = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE +
                                          SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE;     // 25600

    // TMEM layout: [accum (UMMA_N*NUM_EPI_STAGES)] [SFA cols] [SFB cols]
    static constexpr int NUM_ACCUM_TMEM_COLS = UMMA_N * NUM_EPI_STAGES;               // 256
    static constexpr int TMEM_START_SFA      = NUM_ACCUM_TMEM_COLS;                   // 256
    static constexpr int TMEM_START_SFB      = NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS; // 260
    static constexpr int NUM_TMEM_COLS = num_aligned_tmem_cols(
        NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS);                 // 512

    // Number of pipeline stages fitting the SMEM budget.
    // Overhead: smem_cd + barriers(full/empty/with_sf per stage + tmem full/empty) + tmem_ptr.
    static constexpr int SMEM_BARRIERS = (16 * 3 + NUM_EPI_STAGES * 2) * 8;
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_BARRIERS + 8;
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
