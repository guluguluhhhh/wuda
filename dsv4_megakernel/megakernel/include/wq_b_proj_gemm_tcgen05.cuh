#pragma once
// ============================================================
// wq_b_proj_gemm_tcgen05.cuh
// tcgen05 BF16 GEMM — Header (SWAP-AB variant)
// Aligned with DeepGEMM sm100_bf16_gemm.cuh + sm100_store_cd_swap_ab.cuh
//
// Target: M=32~256 (32-aligned), K=1536, N=65536, BF16 -> FP32 output
// swap_ab=1, 2SM MMA (cta_group::2), Cluster=(1,2,1) -> cluster_n=2,
// kIsMulticastOnA=true, Persistent, Warp-Specialized.
//
// Rationale (from NCU profile of the previous non-swap kernel):
//   the previous kernel was Tensor-pipe / latency bound (DRAM only ~41%,
//   Tensor pipe 66% top, occupancy 12%) because UMMA_M=256 x UMMA_N=256
//   wasted 4~8x tensor work on padded M rows. swap_ab puts N on the 256
//   axis and M (=UMMA_N) on the small axis, so tensor work drops to the
//   real M and the kernel becomes HBM-bound like DeepGEMM.
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// CUTLASS/CuTe headers (same as DeepGEMM)
#include <cutlass/arch/barrier.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>   // set_block_rank / store_shared_remote (DSMEM)

// ======================== Configuration ========================
namespace wq_b {

// ---- Problem dimensions (fixed for wq_b projection) ----
static constexpr int K_DIM    = 1536;
static constexpr int N_TOTAL  = 65536;  // 128 heads x 512 dim
static constexpr int BLOCK_K  = 64;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K; // 24

// ---- Cluster / multicast (swap-AB: cluster_n = 2) ----
// DeepGEMM mapping: kNumMulticast = cluster_size, kIsMulticastOnA = (cluster_n > 1)
static constexpr int CLUSTER_SIZE  = 2;
static constexpr int NUM_MULTICAST = CLUSTER_SIZE; // 2
static constexpr bool IS_MULTICAST_ON_A = true;    // cluster_n = 2

// ---- MMA instruction shape (2SM) ----
// UMMA_M = LAYOUT_AD_M(128) * kNumMulticast(2) = 256  (along problem-N)
// UMMA_N = BLOCK_M = M                                 (along problem-M, per-kernel template)
static constexpr int LAYOUT_AD_M = 128;
static constexpr int UMMA_M = LAYOUT_AD_M * NUM_MULTICAST; // 256
static constexpr int UMMA_K = 16;                          // BF16: 16 elements

// ---- N tiling ----
// Each 2SM cluster produces UMMA_M = 256 columns of N (128 per CTA).
static constexpr int BLOCK_N        = 128;          // per-CTA weight rows (N)
static constexpr int CLUSTER_BLOCK_N = BLOCK_N * NUM_MULTICAST; // 256 (per-cluster N tile)
static constexpr int LOAD_BLOCK_N   = BLOCK_N;      // 128 (weight rows loaded per CTA)
static constexpr int NUM_N_TILES    = N_TOTAL / CLUSTER_BLOCK_N; // 65536/256 = 256

// ---- Fused weightless RMSNorm over head_dim (aligns model.py wq_b: rsqrt(mean(q^2))) ----
// A whole head must be resident in one cluster so the head_dim reduction is cluster-local.
// One 2SM MMA covers CLUSTER_BLOCK_N(256) of N; a head (512) = SUBTILES_PER_HEAD sub-tiles.
static constexpr int HEAD_DIM          = 512;                        // RMSNorm reduction length
static constexpr int SUBTILES_PER_HEAD = HEAD_DIM / CLUSTER_BLOCK_N; // 512/256 = 2
static constexpr int NUM_HEAD_TILES    = N_TOTAL / HEAD_DIM;         // 65536/512 = 128
static_assert(HEAD_DIM % CLUSTER_BLOCK_N == 0, "head must be an integer number of cluster N-tiles");
static_assert(N_TOTAL  % HEAD_DIM == 0,        "N must be an integer number of heads");

// ---- Layout: both A(activation) and B(weight) are K-major ----
static constexpr auto MAJOR_A = cute::UMMA::Major::K;
static constexpr auto MAJOR_B = cute::UMMA::Major::K;

// ---- Swizzle (128B) ----
static constexpr int SWIZZLE_A  = 128;   // bytes (K-major, BLOCK_K*2 = 128)
static constexpr int SWIZZLE_B  = 128;   // bytes
static constexpr int SWIZZLE_CD = 128;   // bytes

// ---- Pipeline ----
// NUM_EPI_STAGES (TMEM accumulator buffering) is now M-dependent and lives in SwapDims:
// the fused kernel keeps SUBTILES_PER_HEAD accumulators resident per head, so TMEM
// (512 cols) only fits double-buffering (=2) for M<=128; larger M falls back to 1.
static constexpr int NUM_TMA_STORE_STAGES = 2;

// ---- Threads: 128 non-epilogue + 128 epilogue = 256 ----
static constexpr int TPB                 = 256;
static constexpr int NUM_NON_EPI_THREADS = 128;
static constexpr int NUM_EPI_THREADS     = 128;
// swap-AB uses a full warpgroup (128 threads / 4 warps) for the store
static constexpr int NUM_STORE_THREADS   = 128;

// ---- Epilogue store tile (swap-AB, DeepGEMM sm100.hpp L157-158) ----
// store_block_m = umma_step_n = 16 ; store_block_n = block_n = 128
static constexpr int STORE_BLOCK_M      = 16;                          // M-rows per store stage
static constexpr int STORE_BLOCK_N      = BLOCK_N;                     // 128 (N cols in smem tile)
static constexpr int STORE_BLOCK_N_ATOM = SWIZZLE_CD / (int)sizeof(float); // 128/4 = 32 (TMA store atom on N)

// ---- Per-stage SMEM for weight B (constant); A depends on M ----
static constexpr int SMEM_B_PER_STAGE  = LOAD_BLOCK_N * BLOCK_K * sizeof(nv_bfloat16); // 128*64*2 = 16384
static constexpr int SMEM_CD_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(float); // 16*128*4 = 8192
// Each TMA-store stage holds SUBTILES_PER_HEAD sub-tiles side by side so both are
// written under one barrier pair (merged store -> halves epilogue barriers).
static constexpr int SMEM_CD_TOTAL     = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES * SUBTILES_PER_HEAD; // 32768

// SMEM capacity budget (SM100)
static constexpr int SMEM_CAPACITY = 232448;

// Round up to the next power of two (tcgen05.alloc requires a power-of-2 column count).
static constexpr int ceil_pow2(int v) {
    int p = 32;
    while (p < v) p <<= 1;
    return p;
}

// ---- Compile-time helpers parameterised on M ----
// LOAD_BLOCK_M: activation rows loaded per CTA (split across the 2 CTAs on M).
template <int M_> struct SwapDims {
    static constexpr int BLOCK_M          = M_;              // = problem M (single M block)
    static constexpr int UMMA_N           = M_;              // MMA-N = problem M
    static constexpr int LOAD_BLOCK_M     = M_ / NUM_MULTICAST; // M/2 per CTA
    static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    static constexpr int SMEM_PER_STAGE   = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE;

    // TMEM accumulator buffering. Fused RMSNorm keeps SUBTILES_PER_HEAD accumulators
    // (a whole head) resident, each UMMA_N columns wide. TMEM has 512 columns, so
    // double-buffer (=2) only fits for M<=128; larger M uses a single buffer.
    static constexpr int NUM_EPI_STAGES   = (SUBTILES_PER_HEAD * 2 * UMMA_N <= 512) ? 2 : 1;
    // Allocate NUM_EPI_STAGES * SUBTILES_PER_HEAD * UMMA_N columns, rounded up to a power
    // of two for tcgen05.alloc. Stage s / sub-tile j is addressed at
    // (s * SUBTILES_PER_HEAD + j) * UMMA_N.
    static constexpr int NUM_TMEM_COLS    = ceil_pow2(NUM_EPI_STAGES * SUBTILES_PER_HEAD * UMMA_N);

    // Solve number of stages fitting the SMEM budget.
    // Overhead: smem_cd + reduction scratch (warp_sq[4][M] + rms[M] + peer_sq[EPI][M])
    //           + barriers (full+empty per stage, tmem full/empty + dsmem per epi stage) + tmem_ptr.
    // Barrier bytes conservatively budgeted for up to 16 stages (8B each).
    // warp_sq[4][M] + peer_sq[EPI][M] + rms[M]  (per-warp reduction slots used by both paths).
    static constexpr int SMEM_REDUCE   = (5 + NUM_EPI_STAGES) * M_ * (int)sizeof(float);
    static constexpr int SMEM_BARRIERS = (16 * 2 + NUM_EPI_STAGES * 3) * 8; // <= actual for <=16 stages
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_REDUCE + SMEM_BARRIERS + 8;
    static constexpr int STAGES_RAW    = (SMEM_CAPACITY - SMEM_OVERHEAD) / SMEM_PER_STAGE;
    static constexpr int NUM_STAGES    = STAGES_RAW > 12 ? 12 : STAGES_RAW;
};

} // namespace wq_b

// ======================== Descriptor Helpers ========================
namespace mma_desc {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// K-major 128B-swizzle SMEM descriptor (DeepGEMM mma/sm100.cuh::make_umma_desc, K-major branch).
// For K-major the SBO depends only on BLOCK_K (num_non_contiguous = 128/16 = 8), so the same
// construction works for both the weight tile (128 rows) and the activation tile (M/2 rows).
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_smem_desc_k_major(void* smem_ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;
    desc.lbo_mode_    = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
    // num_non_contiguous(8) * BLOCK_K(64) * sizeof(bf16)(2) = 1024
    constexpr uint32_t SBO = 8 * wq_b::BLOCK_K * sizeof(nv_bfloat16); // 1024
    desc.stride_byte_offset_  = SBO >> 4;  // 64
    desc.leading_byte_offset_ = 0;
    desc.base_offset_         = 0;
    return desc;
}

// Advance descriptor .lo by a whole pipeline stage (offset = stage_bytes / 16).
__device__ __forceinline__
uint32_t advance_desc_lo_for_stage(uint32_t base_lo, uint32_t stage_idx, uint32_t stage_bytes) {
    return base_lo + stage_idx * (stage_bytes / 16);
}

// Advance descriptor .lo by one UMMA_K step within a stage.
// K-major, stride_k = 1 element; UMMA_K=16 elems -> 32 bytes -> 32/16 = 2 units.
__device__ __forceinline__
uint32_t advance_desc_lo_for_k(uint32_t base_lo, uint32_t k_idx) {
    return base_lo + k_idx * (wq_b::UMMA_K * sizeof(nv_bfloat16) / 16); // k_idx * 2
}

// Runtime instruction descriptor for 2SM BF16 GEMM, templated on UMMA_N.
// swap-AB: make_instr_desc<bf16, bf16, float, UMMA_M, UMMA_N, MajorB, MajorA>()
// (both operands K-major here so the major order is symmetric).
template <int UMMA_N_T>
__device__ __forceinline__
uint64_t make_runtime_instr_desc() {
    auto idesc = cute::UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        wq_b::UMMA_M, UMMA_N_T,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    return cute::UMMA::make_runtime_instr_desc(idesc);
}

} // namespace mma_desc

// ======================== PTX Wrappers ========================
namespace ptx {

// tcgen05.mma.cta_group::2.kind::f16 (DeepGEMM ptx/tcgen05.cuh SM100_MMA_F16BF16_2x1SM_SS)
__device__ __forceinline__ void tcgen05_mma_2sm(
    uint32_t tmem_addr, uint64_t a_desc, uint64_t b_desc,
    uint64_t runtime_idesc, uint32_t accum) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(static_cast<uint32_t>(runtime_idesc >> 32)), "r"(accum));
}

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

// umma_arrive for 2SM multicast (tcgen05.commit)
__device__ __forceinline__ void umma_arrive_multicast_2sm(uint64_t* bar, uint16_t mask) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    if (cute::elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :: "r"(addr), "h"(mask) : "memory");
    }
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

// Store a single 32-bit word to shared memory
__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory");
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

// Full-warp (32-lane) sum reduction; result valid in lane 0.
__device__ __forceinline__ float warp_reduce_sum32(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

} // namespace ptx

// ======================== TMA Copy Helpers ========================
namespace tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// 2SM TMA 2D load (K-major, 128B swizzle -> single atom per tile).
// Each CTA independently loads its own box at (k_idx, mn_idx); the 2SM op only
// redirects the transaction bytes to the leader's mbarrier (peer-bit mask).
__device__ __forceinline__
void copy_2sm_2d(void const* desc_ptr, Barrier* barrier_ptr,
                 nv_bfloat16* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM100_TMA_2SM_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr,
        k_idx, mn_idx);
}

// TMA store 2D
__device__ __forceinline__
void store_2d(void const* desc_ptr, void* smem_ptr,
              uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

} // namespace tma
