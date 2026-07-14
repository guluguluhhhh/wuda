#pragma once
// ============================================================
// w1_merged_fp8_gemm.cuh
// tcgen05 FP8 (E4M3) block-scale GEMM — Device Header (SWAP-AB, 2SM)
//
// Skeleton: the repo's proven wq_b_proj_gemm_tcgen05 2SM swap-AB kernel
//           (split-both geometry: activation split on M, weight split on N,
//            cta_group::2 MMA combines the pair).
// FP8/SF mechanics referenced from DeepGEMM sm100_fp8_fp4_gemm_1d1d_impl:
//   - block-scale MMA  : tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale
//   - scale factors     : TMA load -> smem staging -> warp transpose -> UTCCP -> TMEM
//   - BF16 epilogue     : TMEM(FP32) -> STSM transpose -> swizzled smem -> TMA store
//
//   x_fp8[M,7168] @ w1_fp8[4352,7168].T -> y_all_bf16[M,4352]
//   A = x (activation, K-major e4m3), SFA = x_sf ; B = w1 (weight, K-major e4m3), SFB = w1_sf
//   gran_k_a = gran_k_b = 32, BLOCK_K = 128.
//
// Self-contained: depends only on CUTLASS/CuTe arch headers (no deep_gemm/*).
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>

// ======================== Configuration ========================
namespace w1fp8 {

// ---- Problem dimensions ----
static constexpr int K_DIM      = 7168;
static constexpr int N_TOTAL    = 4352;      // 1536 + 512 + 2048 + 256
static constexpr int BLOCK_K    = 128;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K;   // 56
static constexpr int GRAN_K     = 32;        // SF granularity (one UE8M0 per 32 K)

// ---- Element sizes (bytes) ----
static constexpr int FP8_ELEM_SIZE  = 1;     // e4m3
static constexpr int BF16_ELEM_SIZE = 2;
static constexpr int SF_ELEM_SIZE   = 4;     // packed uint32 (4 UE8M0)

// ---- Cluster / multicast (swap-AB: cluster_n = 2, cta_group::2) ----
static constexpr int CLUSTER_SIZE  = 2;
static constexpr int NUM_MULTICAST = CLUSTER_SIZE;   // 2
static constexpr bool IS_MULTICAST_ON_A = true;

// ---- MMA instruction shape (2SM) ----
static constexpr int LAYOUT_AD_M = 128;
static constexpr int UMMA_M      = LAYOUT_AD_M * NUM_MULTICAST;  // 256 (along problem-N)
static constexpr int UMMA_K      = 32;                           // FP8 block-scale UMMA K

// ---- N tiling: each 2SM cluster produces UMMA_M = 256 columns of N (128/CTA) ----
static constexpr int BLOCK_N         = 128;                        // per-CTA weight rows (N)
static constexpr int CLUSTER_BLOCK_N = BLOCK_N * NUM_MULTICAST;     // 256
static constexpr int LOAD_BLOCK_N    = BLOCK_N;                     // 128
static constexpr int NUM_N_TILES     = N_TOTAL / CLUSTER_BLOCK_N;   // 4352/256 = 17

// ---- Both A(activation) and B(weight) are K-major ----
static constexpr auto MAJOR_A = cute::UMMA::Major::K;
static constexpr auto MAJOR_B = cute::UMMA::Major::K;

// ---- Swizzle (128B). K-major requires swizzle == BLOCK_K * sizeof(dtype) = 128. ----
static constexpr int SWIZZLE_A  = 128;
static constexpr int SWIZZLE_B  = 128;
static constexpr int SWIZZLE_CD = 128;

// ---- Pipeline ----
static constexpr int NUM_EPI_STAGES       = 2;   // TMEM accumulator double buffer
static constexpr int NUM_TMA_STORE_STAGES = 2;

// ---- Threads: warp0 TMA, warp1 MMA(leader), warp2 UTCCP-transpose, warps4-7 epilogue ----
static constexpr int TPB                 = 256;
static constexpr int NUM_NON_EPI_THREADS = 128;   // warps 0..3
static constexpr int NUM_STORE_THREADS   = 128;   // warps 4..7 (full warpgroup)

// ---- Scale-factor (block-scale) layout ----
static constexpr int SF_GRAN_K         = 32;
static constexpr int SF_IDS_PER_UINT   = BLOCK_K / SF_GRAN_K;   // 4
static constexpr int NUM_UTCCP_ALIGNED = 128;
static constexpr int SF_BLOCK_N        = ((BLOCK_N + NUM_UTCCP_ALIGNED - 1) / NUM_UTCCP_ALIGNED) * NUM_UTCCP_ALIGNED; // 128
static constexpr int NUM_SFB_TMEM_COLS = SF_BLOCK_N / 32;       // 4

// ---- Epilogue store tile (swap-AB) ----
static constexpr int STORE_BLOCK_M      = 16;                              // M rows per store stage
static constexpr int STORE_BLOCK_N      = BLOCK_N;                         // 128
static constexpr int STORE_BLOCK_N_ATOM = SWIZZLE_CD / (int)sizeof(nv_bfloat16); // 128/2 = 64

// ---- Per-stage SMEM byte sizes ----
static constexpr int SMEM_B_PER_STAGE   = LOAD_BLOCK_N * BLOCK_K * (int)sizeof(__nv_fp8_e4m3); // 16384
static constexpr int SMEM_SFB_PER_STAGE = SF_BLOCK_N * (int)sizeof(uint32_t);                  // 512
static constexpr int SMEM_CD_PER_STAGE  = STORE_BLOCK_M * STORE_BLOCK_N * (int)sizeof(nv_bfloat16); // 4096
static constexpr int SMEM_CD_TOTAL      = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES;            // 8192

static constexpr int SMEM_CAPACITY = 232448;

static constexpr int div_up(int a, int b) { return (a + b - 1) / b; }
static constexpr int align_up(int a, int b) { return div_up(a, b) * b; }
static constexpr int num_aligned_tmem_cols(int c) {
    if (c <= 32) return 32;
    if (c <= 64) return 64;
    if (c <= 128) return 128;
    if (c <= 256) return 256;
    return 512;
}

// ---- Compile-time helpers parameterised on BLOCK_M (= UMMA_N = per-block token count) ----
// wq_b split-both geometry: LOAD_BLOCK_M = BLOCK_M/2 activation rows per CTA.
template <int BLOCK_M_> struct SwapDims {
    static constexpr int BLOCK_M      = BLOCK_M_;
    static constexpr int UMMA_N       = BLOCK_M_;
    static constexpr int LOAD_BLOCK_M = BLOCK_M_ / NUM_MULTICAST;   // BLOCK_M/2 per CTA

    static constexpr int SMEM_A_PER_STAGE   = LOAD_BLOCK_M * BLOCK_K * (int)sizeof(__nv_fp8_e4m3);
    // DeepGEMM verbatim: SFA covers the FULL BLOCK_M tokens (loaded on each CTA);
    // the leader UTCCP (2cta variant) broadcasts it into both SMs' TMEM.
    static constexpr int SF_BLOCK_M         = align_up(BLOCK_M_, NUM_UTCCP_ALIGNED); // >=128
    static constexpr int SMEM_SFA_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);
    static constexpr int NUM_SFA_TMEM_COLS  = SF_BLOCK_M / 32;

    static constexpr int SMEM_PER_STAGE = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE +
                                          SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE;

    // TMEM layout: [accum (UMMA_N*NUM_EPI_STAGES)] [SFA cols] [SFB cols]
    static constexpr int NUM_ACCUM_TMEM_COLS = UMMA_N * NUM_EPI_STAGES;
    static constexpr int TMEM_START_SFA      = NUM_ACCUM_TMEM_COLS;
    static constexpr int TMEM_START_SFB      = NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS;
    static constexpr int NUM_TMEM_COLS = num_aligned_tmem_cols(
        NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS);

    static constexpr int SMEM_BARRIERS = (16 * 3 + NUM_EPI_STAGES * 2) * 8 + 8;
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_BARRIERS + 8;
    static constexpr int STAGES_RAW    = (SMEM_CAPACITY - SMEM_OVERHEAD) / SMEM_PER_STAGE;
    static constexpr int NUM_STAGES    = STAGES_RAW > 12 ? 12 : STAGES_RAW;
};

} // namespace w1fp8

// ======================== Descriptor Helpers ========================
namespace w1_mma_desc {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// K-major 128B-swizzle SMEM descriptor for FP8 operands.
// num_non_contiguous = 128/16 = 8 ; SBO = 8 * BLOCK_K * 1 = 1024.
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_smem_desc_k_major_fp8(void* smem_ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;
    desc.lbo_mode_    = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
    constexpr uint32_t SBO = 8 * w1fp8::BLOCK_K * (uint32_t)sizeof(__nv_fp8_e4m3); // 1024
    desc.stride_byte_offset_  = SBO >> 4;   // 64
    desc.leading_byte_offset_ = 0;
    desc.base_offset_         = 0;
    return desc;
}

// SF (UTCCP source) descriptor: SWIZZLE_NONE, one 8x128b atom. SBO=8*16=128, LBO=0.
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;
    desc.lbo_mode_    = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_NONE);
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
    desc.stride_byte_offset_  = (8 * 16) >> 4;   // 8
    desc.leading_byte_offset_ = 0;
    desc.base_offset_         = 0;
    return desc;
}

__device__ __forceinline__
void replace_sf_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(const_cast<void*>(smem_ptr));
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

// Advance descriptor .lo by a whole pipeline stage (stage_bytes / 16).
__device__ __forceinline__
uint32_t advance_desc_lo_for_stage(uint32_t base_lo, uint32_t stage_idx, uint32_t stage_bytes) {
    return base_lo + stage_idx * (stage_bytes / 16);
}

// Advance descriptor .lo by one UMMA_K step (K-major fp8: 32 elems -> 32B -> 2 units).
__device__ __forceinline__
uint32_t advance_desc_lo_for_k(uint32_t base_lo, uint32_t k_idx) {
    return base_lo + k_idx * (w1fp8::UMMA_K * (uint32_t)sizeof(__nv_fp8_e4m3) / 16); // k_idx*2
}

// Base InstrDescriptorBlockScaled (swap-AB: operand-A = weight (UMMA_M), operand-B = activation).
template <int UMMA_N_T>
__device__ __forceinline__
cute::UMMA::InstrDescriptorBlockScaled make_block_scaled_idesc() {
    return cute::UMMA::make_instr_desc_block_scaled<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, cutlass::float_ue8m0_t,
        w1fp8::UMMA_M, UMMA_N_T,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
}

// Runtime descriptor (hi 32 bits) with per-sub-block SF ids applied.
__device__ __forceinline__
uint64_t make_runtime_idesc_with_sf_id(cute::UMMA::InstrDescriptorBlockScaled desc,
                                       uint32_t a_sf_id, uint32_t b_sf_id) {
    desc.a_sf_id_ = a_sf_id;
    desc.b_sf_id_ = b_sf_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

} // namespace w1_mma_desc

// ======================== PTX Wrappers ========================
namespace w1ptx {

// tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale (DeepGEMM SM100_MMA_MXF8F6F4_2x1SM_SS)
__device__ __forceinline__ void tcgen05_mma_2sm_block_scale(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint64_t runtime_idesc, uint32_t accum,
    uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
           "r"(static_cast<uint32_t>(runtime_idesc >> 32)), "r"(accum),
           "r"(tmem_sfa), "r"(tmem_sfb));
}

__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t smem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(num_cols));
}
__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(num_cols));
}
__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}
__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// tcgen05.commit for 2SM multicast (leader MMA warp signals the mbarrier).
__device__ __forceinline__ void umma_arrive_multicast_2sm(uint64_t* bar, uint16_t mask) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    if (cute::elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :: "r"(addr), "h"(mask) : "memory");
    }
}

// TMEM load: 16dp256b x1 (4 regs) — BF16 STSM transpose store path.
__device__ __forceinline__ void tmem_load_16dp256b1x(
    uint32_t tmem_addr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(tmem_addr));
}
__device__ __forceinline__ void tmem_load_wait() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// stmatrix.trans.x4 (BF16) — store transposed BF16 tile into swizzled smem.
__device__ __forceinline__ void stmatrix_x4_trans_b16(
    void* smem_dst, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1,%2,%3,%4};\n"
        :: "l"(__cvta_generic_to_shared(smem_dst)), "r"(s0), "r"(s1), "r"(s2), "r"(s3));
}

// Cast (bit-reinterpreted) FP32 pair -> packed bf16x2.
__device__ __forceinline__ uint32_t cast_pack_bf16(uint32_t x, uint32_t y) {
    float fx = *reinterpret_cast<float*>(&x);
    float fy = *reinterpret_cast<float*>(&y);
    nv_bfloat162 p = __float22bfloat162_rn(make_float2(fx, fy));
    return *reinterpret_cast<uint32_t*>(&p);
}

// Shared memory helpers
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "l"(__cvta_generic_to_shared(ptr)));
    return v;
}
__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t v) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(v) : "memory");
}
__device__ __forceinline__ void st_shared_v4_u32(void* ptr, uint32_t v0, uint32_t v1,
                                                 uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"
        :: "l"(__cvta_generic_to_shared(ptr)), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

} // namespace w1ptx

// ======================== TMA Copy Helpers ========================
namespace w1tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// Plain (num_multicast=1) TMA 2D load for FP8 operands — DeepGEMM's A/B load path.
// Each CTA loads its own box; the cta_group::2 MMA combines the pair across SMs.
__device__ __forceinline__
void copy_2d_fp8(void const* desc_ptr, Barrier* barrier_ptr,
                 __nv_fp8_e4m3* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, k_idx, mn_idx);
}

// 2SM TMA 2D load (K-major, 128B swizzle -> single atom per tile) for FP8 operands.
__device__ __forceinline__
void copy_2sm_2d_fp8(void const* desc_ptr, Barrier* barrier_ptr,
                     __nv_fp8_e4m3* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM100_TMA_2SM_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, k_idx, mn_idx);
}

// Plain (non-multicast) TMA 2D load for scale-factors (no swizzle). mn inner, k outer.
__device__ __forceinline__
void copy_2d_sf(void const* desc_ptr, Barrier* barrier_ptr,
                uint32_t* smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, mn_idx, k_idx);
}

// TMA store 2D (BF16 output)
__device__ __forceinline__
void store_2d(void const* desc_ptr, void* smem_ptr, uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

// TMA reduce-add 2D (split-K accumulation into BF16 output).
__device__ __forceinline__
void reduce_add_2d(void const* desc_ptr, void* smem_ptr, uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_REDUCE_ADD_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

} // namespace w1tma
