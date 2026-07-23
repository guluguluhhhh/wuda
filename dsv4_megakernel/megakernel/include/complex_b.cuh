#pragma once
#include "cluster_mma_fp8.cuh"
// ============================================================
// gemm_fuse_norm_b.cuh  —  complex_gemm STEP 3 (initial version): y1's second GEMM
// fused with per-head RMSNorm. tcgen05 FP8 (e4m3) block-scale GEMM, SWAP-AB variant.
// Aligned with DeepGEMM sm100_fp8_gemm_1d1d + megakernel w1_merged_fp8_gemm.
// Origin: example.cu (wq_b_proj_gemm_tcgen05). OPERATOR-ONLY header (torch binding
// removed -> plain C++ launcher wq_b_proj_run for the test harness).
// TODO: fold in y2/y3/y4 post-processing (reduce+rmsnorm+rope) later; NOT done yet.
// NOTE: the fused RMSNorm here is WEIGHTLESS (rsqrt(mean(q^2)+eps)); complex_gemm
// step-3 uses rms_w2[512] -- to be added when wiring the real model path.
//
// Native DSV4 FP8 quantization: activation 1x128 / weight 128x128 UE8M0 scales.
//   x_sf[M, K/128] uint8 (per token per K128 block); w_sf[N/128, K/128] uint8.
//   The bf16->fp8 ACTIVATION quant is FUSED into the kernel prologue (all CTAs,
//   grid-strided over M*12 blocks) with a software grid barrier (monotonic ticket
//   counter) ordering the writes before any TMA/SF read -- no separate quant kernel.
//   Weight quant (128x128 blocks) is offline-static (quant_weights_block128, cached).
//   SF is never TMA-loaded: warp2 reads the native bytes and replicates each K128
//   byte into four identical K32 scale IDs (mxf8f6f4 consumes SF at K32).
//
// Target: M=16~256 (16-aligned), K=1536, N=65536, e4m3 in -> BF16 out
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
#include <cuda_fp8.h>          // fused compressor: real e4m3 quant
#include <cuda_fp4.h>          // fused compressor: real e2m1 quant (CUDA >= 12.8)
#include <cstdint>

// CUTLASS/CuTe headers (same as DeepGEMM)
#include <cutlass/arch/barrier.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>   // set_block_rank / store_shared_remote (DSMEM)

// gfnb: single wrapping namespace so this operator can co-exist in ONE translation
// unit with complex_a.cuh (both otherwise declare a GLOBAL `SharedStorage` and
// tcgen05 machinery). The pipeline test includes both -> wrap to avoid collisions.
namespace gfnb {

// ======================== Configuration ========================
namespace wq_b {

// ---- Problem dimensions (fixed for wq_b projection) ----
static constexpr int K_DIM    = 1536;
static constexpr int N_TOTAL  = 65536;  // 128 heads x 512 dim
static constexpr int BLOCK_K  = 128;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K; // 12

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
static constexpr int UMMA_K = 32;                          // BF16: 16 elements

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
// ---- FP8 block-scale scale-factor: NATIVE DSV4 granularity (UE8M0) ----
// activation 1x128 : x_sf[M, K/128]      one uint8 per row per K128 block
// weight   128x128 : w_sf[N/128, K/128]  one uint8 per 128x128 block
// tcgen05.mma.kind::mxf8f6f4 consumes SF at K32 granularity; warp2 replicates each
// native K128 byte into four identical K32 scale IDs (e * 0x01010101). SF is NOT
// TMA-loaded: warp2 reads the native bytes straight from global memory.
static constexpr int QUANT_BLOCK_K          = 128;  // native DSV4 activation/weight K granularity
static constexpr int UMMA_SF_GRAN_K         = 32;   // hardware block-scale consumption granularity
static constexpr int SF_IDS_PER_QUANT_BLOCK = QUANT_BLOCK_K / UMMA_SF_GRAN_K; // 4
static constexpr uint32_t UE8M0_ONE         = 0x7fu; // exponent bias 127 => scale 1.0 (OOB rows)
static constexpr int WEIGHT_QUANT_BLOCK_N   = 128;
static constexpr int NUM_WEIGHT_SF_ROWS     =
    (N_TOTAL + WEIGHT_QUANT_BLOCK_N - 1) / WEIGHT_QUANT_BLOCK_N; // 512
static_assert(BLOCK_K == QUANT_BLOCK_K && SF_IDS_PER_QUANT_BLOCK == 4,
              "one pipeline stage must expand one K128 scale to four K32 IDs");
static constexpr int NUM_UTCCP_ALIGNED = 128;
static constexpr int SF_BLOCK_M        = 128;
static constexpr int SF_BLOCK_N        = 128;
static constexpr int SMEM_SFA_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);
static constexpr int SMEM_SFB_PER_STAGE = SF_BLOCK_N * (int)sizeof(uint32_t);
static constexpr int NUM_SFA_TMEM_COLS = SF_BLOCK_M / 32;
static constexpr int NUM_SFB_TMEM_COLS = SF_BLOCK_N / 32;

// ---- Threads: 128 non-epilogue (warps 0-3) + 128 epilogue (warps 4-7) = 256 GEMM
//      threads, PLUS 256 CUDA-core tail threads (warps 8-15) = 512 total. The tail
//      warps run op A's y2/y3/y4 reduce + y2 RMSNorm concurrently with, and FULLY
//      DECOUPLED from, the GEMM (own NamedBarrier id 1; no __syncthreads; no TMEM). ----
static constexpr int NUM_NON_EPI_THREADS = 128;
static constexpr int NUM_EPI_THREADS     = 128;
// swap-AB uses a full warpgroup (128 threads / 4 warps) for the store
static constexpr int NUM_STORE_THREADS   = 128;
// CUDA-core tail group: 8 warps (warps 8-15), 256 threads.
static constexpr int NUM_TAIL_WARPS      = 8;
static constexpr int NUM_TAIL_THREADS    = NUM_TAIL_WARPS * 32;                             // 256
static constexpr int TAIL_WARP_LO        = (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32;  // 8
static constexpr int TPB                 = NUM_NON_EPI_THREADS + NUM_STORE_THREADS + NUM_TAIL_THREADS; // 512

// ---- Epilogue store tile (swap-AB, DeepGEMM sm100.hpp L157-158) ----
// store_block_m = umma_step_n = 16 ; store_block_n = block_n = 128
static constexpr int STORE_BLOCK_M      = 16;                          // M-rows per store stage
static constexpr int STORE_BLOCK_N      = BLOCK_N;                     // 128 (N cols in smem tile)
// Output is BF16 (model-faithful: q is bf16 in DSV4). 128B swizzle = 64 bf16 per atom.
static constexpr int STORE_BLOCK_N_ATOM = SWIZZLE_CD / (int)sizeof(nv_bfloat16); // 128/2 = 64

// ---- Per-stage SMEM for weight B (constant); A depends on M ----
static constexpr int SMEM_B_PER_STAGE  = LOAD_BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3); // 128*64*2 = 16384
static constexpr int SMEM_CD_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(nv_bfloat16); // 16*128*2 = 4096
// Each TMA-store stage holds SUBTILES_PER_HEAD sub-tiles side by side so both are
// written under one barrier pair (merged store -> halves epilogue barriers).
static constexpr int SMEM_CD_TOTAL     = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES * SUBTILES_PER_HEAD; // 4096*2*2 = 16384

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
    static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr int SMEM_SFA_PER_STAGE = wq_b::SMEM_SFA_PER_STAGE;
    static constexpr int SMEM_SFB_PER_STAGE = wq_b::SMEM_SFB_PER_STAGE;
    static constexpr int SMEM_PER_STAGE   = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE + SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE;

    // TMEM accumulator buffering. Fused RMSNorm keeps SUBTILES_PER_HEAD accumulators
    // (a whole head) resident, each UMMA_N columns wide. TMEM has 512 columns, so
    // double-buffer (=2) only fits for M<=128; larger M uses a single buffer.
    static constexpr int NUM_EPI_STAGES   = (SUBTILES_PER_HEAD * 2 * UMMA_N + 8 <= 512) ? 2 : 1;
    // Allocate NUM_EPI_STAGES * SUBTILES_PER_HEAD * UMMA_N columns, rounded up to a power
    // of two for tcgen05.alloc. Stage s / sub-tile j is addressed at
    // (s * SUBTILES_PER_HEAD + j) * UMMA_N.
    static constexpr int NUM_ACCUM_TMEM_COLS = NUM_EPI_STAGES * SUBTILES_PER_HEAD * UMMA_N;
    static constexpr int TMEM_START_SFA      = NUM_ACCUM_TMEM_COLS;
    static constexpr int TMEM_START_SFB      = NUM_ACCUM_TMEM_COLS + wq_b::NUM_SFA_TMEM_COLS;
    static constexpr int NUM_TMEM_COLS       = ceil_pow2(NUM_ACCUM_TMEM_COLS + wq_b::NUM_SFA_TMEM_COLS + wq_b::NUM_SFB_TMEM_COLS);

    // Solve number of stages fitting the SMEM budget.
    // Overhead: smem_cd + reduction scratch (warp_sq[4][M] + rms[M] + peer_sq[EPI][M])
    //           + barriers (full+empty per stage, tmem full/empty + dsmem per epi stage) + tmem_ptr.
    // Barrier bytes conservatively budgeted for up to 16 stages (8B each).
    // warp_sq[4][M] + peer_sq[EPI][M] + rms[M]  (per-warp reduction slots used by both paths).
    static constexpr int SMEM_REDUCE   = (5 + NUM_EPI_STAGES) * M_ * (int)sizeof(float);
    static constexpr int SMEM_BARRIERS = (16 * 3 + NUM_EPI_STAGES * 3) * 8; // <= actual for <=16 stages
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_REDUCE + SMEM_BARRIERS + 8 + 4096; // + alignas(1024/128) padding margin
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
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* p) {
    uint32_t r; uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(a)); return r;
}
__device__ __forceinline__ void st_shared_v4_u32(uint32_t* p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t ad = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(ad), "r"(a), "r"(b), "r"(c), "r"(d));
}
__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(num_cols));
}
// MANDATORY after the last tcgen05.alloc in a CTA: release the allocation permit so
// the HW allocator can serve the CTAs that reuse this SM (incl. the next launch).
// Omitting it makes each retired CTA keep the lock -> back-to-back launches stall in
// periodic bursts while the allocator reclaims permits (the ~2ms wq_b_proj spikes).
__device__ __forceinline__ void tcgen05_relinquish_alloc_permit_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
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

// Store a single 16-bit half-word to shared memory (bf16 output store)
__device__ __forceinline__ void st_shared_u16(void* ptr, uint16_t v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.u16 [%0], %1;" :: "r"(addr), "h"(v) : "memory");
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

// Acquire-load a global u32 (software grid-barrier spin of the fused quant prologue).
__device__ __forceinline__ uint32_t ld_acquire_gpu_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Full-warp (32-lane) sum reduction; result valid in lane 0.
__device__ __forceinline__ float warp_reduce_sum32(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// setmaxnreg (SM90+/Blackwell): warpgroup-wide (128 threads, converged) register-cap
// reallocation. `inc` RAISES the per-thread max to N, drawing from the CTA pool that
// other warpgroups released; `dec` LOWERS it to N, returning the surplus to the pool.
// N is the ABSOLUTE target (not a delta), an immediate in [24,256], multiple of 8, and
// (inc: N>=current, dec: N<=current). Two opcodes because the DIRECTION drives the pool
// accounting. Must be hit by all 128 threads of the warpgroup, converged.
template <int N>
__device__ __forceinline__ void warpgroup_reg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(N));
}
template <int N>
__device__ __forceinline__ void warpgroup_reg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(N));
}

// Device-wide nanosecond timer (%globaltimer): ONE clock for the whole GPU, so stamps
// taken by different CTAs/warps are directly comparable -- used to see when the
// tensor-core path (warps 0-7) and the CUDA-core tail (warps 8-15) each start/finish
// (they are warp-specialized and run concurrently).
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
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

// Plain per-CTA TMA 2D load for FP8 operands (each CTA loads its own box).
__device__ __forceinline__
void copy_2d_fp8(void const* desc_ptr, Barrier* barrier_ptr,
                 __nv_fp8_e4m3* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, k_idx, mn_idx);
}

} // namespace tma

// ======================== Kernel + Host (merged; plain CUDA, no torch binding) ========================
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
    alignas(128)  uint8_t smem_sfa[NUM_STAGES_T * D::SMEM_SFA_PER_STAGE];
    alignas(128)  uint8_t smem_sfb[NUM_STAGES_T * D::SMEM_SFB_PER_STAGE];

    // Fused RMSNorm reduction scratch (per M-row, over head_dim).
    //   smem_warp_sq : per-warp partial sum-of-squares (both paths; avoids pre-zero + atomics)
    //   smem_peer_sq : partial pushed by the peer CTA (double-buffered by epi stage)
    //   smem_rms     : rsqrt(mean + eps) scale applied in the store
    alignas(16) float smem_warp_sq[4][M_TPL];
    alignas(16) float smem_peer_sq[NUM_EPI_STAGES_T][M_TPL];
    alignas(16) float smem_rms[M_TPL];

    // CUDA-core tail-reduce scratch (warps 8-15): 8-warp partial for y2's RMSNorm.
    alignas(16) float tail_sq[NUM_TAIL_WARPS];

    // Fused-compressor tail scratch (warps 8-15, compress rows only). Reused across
    // the main(512) then indexer(128) passes: aggregated->normed->roped c and the
    // per-block scale/exponent. block-amax reads |cmp| inline, so no separate |.| buffer.
    alignas(16) float cmp[512];
    alignas(16) float cmp_blk[8];

    // Barriers
    alignas(16) Barrier full_barriers[NUM_STAGES_T];
    alignas(16) Barrier empty_barriers[NUM_STAGES_T];
    alignas(16) Barrier with_sf_full_barriers[NUM_STAGES_T];
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES_T];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES_T];
    alignas(16) Barrier dsmem_barriers[NUM_EPI_STAGES_T];   // cross-CTA sum-of-squares push

    // TMEM base address
    alignas(16) uint32_t tmem_base;

    // Fused activation-quant prologue: this launch's software grid-barrier target
    // (monotonic ticket counter value all CTAs spin to).
    alignas(16) uint32_t quant_target;
};

// ============ Fused compressor: per-row post-processing (tail warps) ============
// Called by ALL 256 tail threads on a COMPRESS row (do_c uniform across the CTA),
// after pass2 has already written this row's current slot into the state buffers.
// Mirrors cublas_q_winkv_compressor_baseline.cuh (== V4-pro model.py Compressor):
//   overlap-cat 8 rows -> per-col softmax weighted sum -> shift state ->
//   (bf16) RMSNorm -> interleaved RoPE(last64) -> [indexer: 128-pt hadamard] ->
//   REAL fp8(main, block64)/fp4(indexer, block32) quant. All 256 threads MUST hit
//   every NamedBarrier (id 1, tail-group only) -> no early return inside.
// Uses s.cmp[512]/s.cmp_blk[8]/s.tail_sq[8] as scratch (reused main-then-indexer).
// eps == the kernel's fused-RMSNorm epsilon (1e-6 to match golden).
__device__ __forceinline__ int cmp_flog2_ceil(float x) {
    unsigned b = __float_as_uint(x); int e = (int)((b >> 23) & 0xFF); unsigned m = b & 0x7FFFFFu;
    return e - 127 + (m != 0u ? 1 : 0);
}
__device__ __forceinline__ float cmp_fpow2(int k) { return __uint_as_float((unsigned)(k + 127) << 23); }

template <int M_TPL>
__device__ void compressor_process_row(
    SharedStorage<M_TPL>& s, int t, int m, float eps, long long p,
    const float* __restrict__ comp_norm, const float* __restrict__ idx_norm,
    const float* __restrict__ cos_tab,  const float* __restrict__ sin_tab,
    float* __restrict__ comp_kv, float* __restrict__ comp_sc,
    float* __restrict__ idx_kv,  float* __restrict__ idx_sc,
    uint8_t* __restrict__ comp_q8, float* __restrict__ comp_s8, nv_bfloat16* __restrict__ comp_rope,
    uint8_t* __restrict__ idx_q4,  uint8_t* __restrict__ idx_s4,
    unsigned long long* __restrict__ prof)   // [num_blocks*8]: intra-compressor phase stamps [5..7], t==0 only
{
    constexpr int RATIO = 4, RD = 64, SROWS = 8;
    constexpr int D_M = 512, WK_M = 1024, D_I = 128, WK_I = 256;
    // Tail-group-only barrier (NamedBarrier id 1): synchronizes just these 256 tail
    // threads, never the GEMM warps (which use id 0 / __syncthreads elsewhere).
    using TailBarrier = cutlass::arch::NamedBarrier;
    const int t_warp = t >> 5;                          // 0..7 tail-warp index
    const long long rope_pos = p + 1 - RATIO;           // compressed-token RoPE position
    const float* cos_row = cos_tab + (size_t)rope_pos * (RD / 2);
    const float* sin_row = sin_tab + (size_t)rope_pos * (RD / 2);

    // ================= MAIN compressor (d=512) =================
    // The aggregate (-> s.cmp) was done by the CALLER (register-prefetched, hidden under the
    // GEMM-covered reduce); here we only shift the state window. s.cmp already holds the row.
    {   // shift state[:RATIO] <- state[RATIO:] (drop the oldest RATIO rows)
        // Copy in CHUNK-sized waves through a small REUSED register array: CHUNK loads in flight
        // hide HBM latency while the staging footprint stays CHUNK regs (not NSHIFT=16), so the
        // tail's register peak stays <= the hard 128 cap of a 512-thread block -> no spill.
        // Pure copy (same src->dst mapping, wave order irrelevant) => bit-identical.
        float* state_kv = comp_kv + (size_t)m * SROWS * WK_M;
        float* state_sc = comp_sc + (size_t)m * SROWS * WK_M;
        constexpr int NSHIFT = (RATIO * WK_M) / NUM_TAIL_THREADS;   // 4096/256 = 16 elems/thread
        constexpr int CHUNK  = 4;                                   // 4 loads in flight; peak 4 staging regs
        float stage[CHUNK];
        #pragma unroll
        for (int base = 0; base < NSHIFT; base += CHUNK) {
            #pragma unroll
            for (int j = 0; j < CHUNK; ++j)
                stage[j] = state_kv[RATIO * WK_M + t + (base + j) * NUM_TAIL_THREADS];
            #pragma unroll
            for (int j = 0; j < CHUNK; ++j)
                state_kv[t + (base + j) * NUM_TAIL_THREADS] = stage[j];
        }
        #pragma unroll
        for (int base = 0; base < NSHIFT; base += CHUNK) {
            #pragma unroll
            for (int j = 0; j < CHUNK; ++j)
                stage[j] = state_sc[RATIO * WK_M + t + (base + j) * NUM_TAIL_THREADS];
            #pragma unroll
            for (int j = 0; j < CHUNK; ++j)
                state_sc[t + (base + j) * NUM_TAIL_THREADS] = stage[j];
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }
    // [DIAG] prof[5] moved to the caller (right after the aggregate) so mainAgg isolates the
    // aggregate and mainNQ now includes this shift. Restore after this measurement.

    {   // bf16 RMSNorm over d=512: normalize s.cmp in place (bf16-rounded, matches golden)
        float sumsq = 0.f;
        for (int c = t; c < D_M; c += NUM_TAIL_THREADS) {
            const float v = (float)(nv_bfloat16)s.cmp[c];
            sumsq += v * v;
        }
        sumsq = ptx::warp_reduce_sum32(sumsq);
        if ((t & 31) == 0) s.tail_sq[t_warp] = sumsq;
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
        float sumsq_all = 0.f;
        #pragma unroll
        for (int w = 0; w < NUM_TAIL_WARPS; ++w) sumsq_all += s.tail_sq[w];
        const float rms = rsqrtf(sumsq_all / float(D_M) + eps);     // tail_sq read-only hereafter; no barrier
        for (int c = t; c < D_M; c += NUM_TAIL_THREADS) {
            const float v = (float)(nv_bfloat16)s.cmp[c];
            s.cmp[c] = (float)(nv_bfloat16)(v * rms * comp_norm[c]);
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }

    // interleaved RoPE on the last RD(64) dims: pair j=t rotates (even,odd) by (cos_j,sin_j)
    if (t < RD / 2) {
        const int   j         = t;
        const int   rope_base = D_M - RD;
        const float even = s.cmp[rope_base + 2 * j];
        const float odd  = s.cmp[rope_base + 2 * j + 1];
        const float cos_j = cos_row[j], sin_j = sin_row[j];
        s.cmp[rope_base + 2 * j]     = (float)(nv_bfloat16)(even * cos_j - odd * sin_j);
        s.cmp[rope_base + 2 * j + 1] = (float)(nv_bfloat16)(even * sin_j + odd * cos_j);
    }
    TailBarrier::sync(NUM_TAIL_THREADS, 1);

    {   // fp8(e4m3) quant of [0,448) in block64 + copy the rope tail [448,512) -> bf16
        constexpr int NF8 = D_M - RD;                   // 448
        // per-64-block amax (abs fused into the block-amax -> drops an abs pass + a barrier)
        if (t < NF8 / 64) {
            float blk_amax = 0.f;
            for (int i = 0; i < 64; ++i) blk_amax = fmaxf(blk_amax, fabsf(s.cmp[t * 64 + i]));
            s.cmp_blk[t] = fmaxf(blk_amax, 1e-4f);
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
        for (int c = t; c < NF8; c += NUM_TAIL_THREADS) {
            const int   blk   = c >> 6;
            const float scale = s.cmp_blk[blk] * (1.0f / 448.0f);
            const __nv_fp8_e4m3 q = __nv_fp8_e4m3(s.cmp[c] / scale);
            comp_q8[(size_t)m * NF8 + c] = q.__x;
            if ((c & 63) == 0) comp_s8[(size_t)m * (NF8 / 64) + blk] = scale;
        }
        for (int c = NF8 + t; c < D_M; c += NUM_TAIL_THREADS)
            comp_rope[(size_t)m * RD + (c - NF8)] = (nv_bfloat16)s.cmp[c];
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }
    if (prof && t == 0) prof[blockIdx.x * 16 + 6] = ptx::globaltimer();   // [phase] main norm+rope+fp8 quant done

    // ================= INDEXER compressor (d=128, rotate=True) =================
    {   // aggregate: per-col 8-row softmax weighted sum (overlap-cat rows<RATIO->[0,128) else [128,256))
        float* state_kv = idx_kv + (size_t)m * SROWS * WK_I;
        float* state_sc = idx_sc + (size_t)m * SROWS * WK_I;
        for (int c = t; c < D_I; c += NUM_TAIL_THREADS) {
            float sc[8], kv[8];
            #pragma unroll
            for (int rr = 0; rr < 8; ++rr) {
                const int col = (rr < RATIO) ? c : (D_I + c);
                sc[rr] = state_sc[(size_t)rr * WK_I + col];
                kv[rr] = state_kv[(size_t)rr * WK_I + col];
            }
            float max_logit = sc[0];
            #pragma unroll
            for (int rr = 1; rr < 8; ++rr) max_logit = fmaxf(max_logit, sc[rr]);
            float softmax_denom = 0.f, weighted_sum = 0.f;
            #pragma unroll
            for (int rr = 0; rr < 8; ++rr) {
                const float w = expf(sc[rr] - max_logit);
                softmax_denom += w;
                weighted_sum  += w * kv[rr];
            }
            s.cmp[c] = weighted_sum / softmax_denom;
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
        // shift indexer state[:RATIO] <- state[RATIO:]
        for (int i = t; i < RATIO * WK_I; i += NUM_TAIL_THREADS) {
            state_kv[i] = state_kv[RATIO * WK_I + i];
            state_sc[i] = state_sc[RATIO * WK_I + i];
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }
    {   // bf16 RMSNorm over d=128
        float sumsq = 0.f;
        for (int c = t; c < D_I; c += NUM_TAIL_THREADS) {
            const float v = (float)(nv_bfloat16)s.cmp[c];
            sumsq += v * v;
        }
        sumsq = ptx::warp_reduce_sum32(sumsq);
        if ((t & 31) == 0) s.tail_sq[t_warp] = sumsq;
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
        float sumsq_all = 0.f;
        #pragma unroll
        for (int w = 0; w < NUM_TAIL_WARPS; ++w) sumsq_all += s.tail_sq[w];
        const float rms = rsqrtf(sumsq_all / float(D_I) + eps);     // tail_sq read-only hereafter; no barrier
        for (int c = t; c < D_I; c += NUM_TAIL_THREADS) {
            const float v = (float)(nv_bfloat16)s.cmp[c];
            s.cmp[c] = (float)(nv_bfloat16)(v * rms * idx_norm[c]);
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }
    if (t < RD / 2) {   // interleaved RoPE on the last RD(64) dims
        const int   j         = t;
        const int   rope_base = D_I - RD;
        const float even = s.cmp[rope_base + 2 * j];
        const float odd  = s.cmp[rope_base + 2 * j + 1];
        const float cos_j = cos_row[j], sin_j = sin_row[j];
        s.cmp[rope_base + 2 * j]     = (float)(nv_bfloat16)(even * cos_j - odd * sin_j);
        s.cmp[rope_base + 2 * j + 1] = (float)(nv_bfloat16)(even * sin_j + odd * cos_j);
    }
    TailBarrier::sync(NUM_TAIL_THREADS, 1);
    {   // 128-pt natural-order FWHT (fp32): 5 intra-warp butterflies via shuffle (no barrier) +
        // one 4-warp cross-combine matching the h=32,h=64 grouping -> 3 barriers instead of 8, bit-exact.
        float v = (t < D_I) ? s.cmp[t] : 0.f;
        const unsigned lane = t & 31;
        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {                          // stages h=1,2,4,8,16 stay inside the warp
            const float partner = __shfl_xor_sync(0xffffffffu, v, h);
            v = ((lane & h) == 0) ? (v + partner) : (partner - v);
        }
        if (t < D_I) s.cmp[t] = v;                                  // publish per-warp 32-pt results
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
        float fwht_out = 0.f;
        if (t < D_I) {                                              // cross-warp: h=32 then h=64 (Sylvester grouping)
            const int   l  = t & 31, w = t >> 5;
            const float a0 = s.cmp[l], a1 = s.cmp[l + 32], a2 = s.cmp[l + 64], a3 = s.cmp[l + 96];
            const float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
            fwht_out = (w == 0) ? (b0 + b2) : (w == 1) ? (b1 + b3) : (w == 2) ? (b0 - b2) : (b1 - b3);
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);                               // all s.cmp reads done before overwrite
        if (t < D_I) s.cmp[t] = (float)(nv_bfloat16)(fwht_out * rsqrtf((float)D_I));
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }
    {   // fp4(e2m1) quant of 128 in block32, packed 2/byte + e8m0 block scale
        if (t < D_I / 32) {
            float blk_amax = 0.f;
            for (int i = 0; i < 32; ++i) blk_amax = fmaxf(blk_amax, fabsf(s.cmp[t * 32 + i]));
            blk_amax = fmaxf(blk_amax, 6.0f * 1.1754944e-38f);
            const int k = cmp_flog2_ceil(blk_amax * (1.0f / 6.0f));
            s.cmp_blk[t] = cmp_fpow2(k);
            idx_s4[(size_t)m * (D_I / 32) + t] = (uint8_t)(k + 127);
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
        if (t < D_I / 2) {
            const int    c     = 2 * t;
            const int    blk   = c >> 5;
            const float  scale = s.cmp_blk[blk];
            const float2 f = make_float2(s.cmp[c] / scale, s.cmp[c + 1] / scale);
            const __nv_fp4x2_e2m1 packed = __nv_fp4x2_e2m1(f);
            idx_q4[(size_t)m * (D_I / 2) + t] = (uint8_t)packed.__x;
        }
        TailBarrier::sync(NUM_TAIL_THREADS, 1);
    }
    if (prof && t == 0) prof[blockIdx.x * 16 + 7] = ptx::globaltimer();   // [phase] indexer done == compressor end
}

// ======================== Kernel ========================
template <int M_TPL>
__global__ void __launch_bounds__(TPB, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,   // activation [M, K] e4m3 (x_fp8 below), K-major
    const __grid_constant__ CUtensorMap desc_B,   // weight     [N, K] e4m3, K-major
    const nv_bfloat16* __restrict__ x_bf16,       // activation y1 bf16 [M, lda_x] (fused-quant source)
    int lda_x,                                    // x_bf16 row pitch in elements
    __nv_fp8_e4m3* __restrict__ x_fp8,            // fused-quant output [M, K_DIM] (desc_A reads this)
    uint8_t* __restrict__ x_sf,                   // [M, K/128] native UE8M0 activation SF (written here)
    const uint8_t* __restrict__ w_sf,             // [N/128, K/128] native UE8M0 weight SF (128x128 blocks)
    uint32_t* __restrict__ quant_sync,            // monotonic ticket counter (software grid barrier)
    const __grid_constant__ CUtensorMap desc_D,   // output     [M, N], BF16 row-major
    int num_blocks,
    float eps,                                    // fused RMSNorm epsilon (head + y2 tail)
    const float* __restrict__ ws_tail,            // op A split-K partials [ks_tail, M, ws_stride]
    int ks_tail,                                  // #split-K slabs (0 => tail work disabled)
    const float* __restrict__ rms_w2,             // y2 RMSNorm weight (len 512)
    nv_bfloat16* __restrict__ dtail,              // write target [M, ws_stride] (== op A's D)
    int ws_stride,                                // physical row stride of ws/dtail (4352 legacy / 4608 fused)
    const float2* __restrict__ rope_cs,           // per-row cos/sin for y2's last-64 RoPE:
                                                  // [M,32], rope_cs[m*32+j]=(cos_j,sin_j); null=>no rope
    unsigned long long* __restrict__ prof,        // [num_blocks*8] u64 TC/CC + intra-compressor phase stamps (null=>off)
    // ---- fused compressor (y3/y4) inputs/outputs; comp_kv==null => disabled (old y3/y4 reduce-write) ----
    const long long* __restrict__ cmp_pos,        // [M] int64 absolute token positions
    const float* __restrict__ comp_ape, const float* __restrict__ idx_ape,     // [4,1024],[4,256]
    const float* __restrict__ comp_norm, const float* __restrict__ idx_norm,   // [512],[128]
    const float* __restrict__ cos_tab,  const float* __restrict__ sin_tab,     // [SEQ,32]
    float* __restrict__ comp_kv, float* __restrict__ comp_sc,                  // [M,8,1024] state in/out
    float* __restrict__ idx_kv,  float* __restrict__ idx_sc,                   // [M,8,256]  state in/out
    uint8_t* __restrict__ comp_q8, float* __restrict__ comp_s8, nv_bfloat16* __restrict__ comp_rope, // [M,448],[M,7],[M,64]
    uint8_t* __restrict__ idx_q4,  uint8_t* __restrict__ idx_s4)               // [M,64] packed,[M,4] e8m0
{
    using Dims = SwapDims<M_TPL>;
    constexpr int NUM_STAGES_T       = Dims::NUM_STAGES;
    constexpr int SMEM_A_PER_STAGE_T = Dims::SMEM_A_PER_STAGE;
    constexpr int LOAD_BLOCK_M_T     = Dims::LOAD_BLOCK_M;   // M/2
    constexpr int UMMA_N_T           = Dims::UMMA_N;         // M
    constexpr int NUM_TMEM_COLS_T    = Dims::NUM_TMEM_COLS;  // NUM_EPI_STAGES*2*M (pow2)
    constexpr int NUM_EPI_STAGES_T   = Dims::NUM_EPI_STAGES; // 2 for M<=128 else 1
    constexpr int ACCUM_COLS_T       = SUBTILES_PER_HEAD * UMMA_N_T; // TMEM cols per epi stage
    constexpr int SMEM_SFA_PER_STAGE_T = Dims::SMEM_SFA_PER_STAGE;
    constexpr int SMEM_SFB_PER_STAGE_T = Dims::SMEM_SFB_PER_STAGE;
    constexpr int TMEM_SFA_T           = Dims::TMEM_START_SFA;
    constexpr int TMEM_SFB_T           = Dims::TMEM_START_SFB;

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

    // [profile] TC/CC path START stamps -- taken at KERNEL ENTRY, BEFORE the register-budget
    // / barrier-init / TMEM-alloc / fused-quant startup, so both spans cover op B's FULL
    // lifetime (init + fused quant + grid-sync + GEMM/tail), not just steady-state. All
    // threads are converged here (no branch yet), so elect_one_sync is safe.
    //   TC = MMA leader (warp1, leader CTA);  CC = tail group's first warp (both CTAs).
    if (prof && ptx::elect_one_sync()) {
        if (warp_id == 1 && is_leader)          prof[blockIdx.x * 16 + 0] = ptx::globaltimer();
        else if (warp_id == wq_b::TAIL_WARP_LO) prof[blockIdx.x * 16 + 2] = ptx::globaltimer();
    }

    // ================================================================
    // PER-WARPGROUP REGISTER BUDGET  (setmaxnreg scaffold)
    // ================================================================
    // 16 warps = 4 warpgroups (128 threads each):
    //   WG0 = warps 0-3  : TMA producer / MMA issue / TMEM alloc / idle   (light)
    //   WG1 = warps 4-7  : epilogue -- TMEM read + head RMSNorm + TMA store (HEAVY)
    //   WG2 = warps 8-11 : CUDA-core tail  (y2/y3/y4 reduce + y2 RMSNorm + y2 RoPE)
    //   WG3 = warps 12-15: CUDA-core tail  (same)
    // MECHANISM + EXPERIMENTAL VERDICT (PTX ISA 9.7.20.5 setmaxnreg + 11.4.1 .maxnreg):
    //   __launch_bounds__(TPB,1) lowers to .maxntid 512 + .minnctapersm 1 => the COMPILE-TIME
    //   architectural register cap ptxas prints as "Used N" is 65536/(maxThreads*minBlocks) =
    //   65536/512 = 128. setmaxnreg does NOT change this: it only re-distributes PHYSICAL registers
    //   at RUNTIME from a per-CTA pool (dec returns, inc claims). The SASS still references only 128
    //   architectural regs, so inc<160> hands out physical regs that NO instruction can use -> spill
    //   is unchanged. PROVEN on B300: WG0 dec<32> + WG2/WG3 inc<160> -> ptxas still "Used 128" and
    //   spill WORSENED (104B -> 192B, tailTot 45.7 -> 48.6). CUTLASS gets consumers to 160/232 only
    //   because its blocks are <=384 threads (65536/384=170 >= 160): the BASELINE is already high
    //   enough. Our 512-thread block caps the baseline at 128 -- a hard architectural limit that
    //   setmaxnreg cannot lift. And we gain nothing from register donation either: smem (~200KB)
    //   pins us to 1 CTA/SM, so freeing registers buys no occupancy.
    // => Keep a MILD split (WG0 light, WG1/2/3 at the 128 baseline). To actually kill the tail spill
    //   we must fit the tail into 128 regs (trim pf[28]/shift staging) or use fewer threads/block --
    //   NOT setmaxnreg. WG0=64; WG1/WG2/WG3=128. sum = 64+128*3 = 448 <= 512 => 1 CTA/SM.
    const uint32_t wg_id = warp_id / 4;   // 0..3 (warpgroup-uniform => setmaxnreg issued converged)
    switch (wg_id) {
        case 0:  ptx::warpgroup_reg_dec<64>();  break;   // WG0 producer/mma (light)
        default: ptx::warpgroup_reg_inc<128>(); break;   // WG1 epilogue + WG2/WG3 tail (baseline 128)
    }

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
            s.full_barriers[i].init(1);               // per-CTA A/B TMA (fp8, plain SM90 loads)
            s.with_sf_full_barriers[i].init(NUM_MULTICAST * 32); // both CTAs warp2 (32 lanes each)
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
        // Give up the alloc permit immediately (this CTA allocates exactly once).
        ptx::tcgen05_relinquish_alloc_permit_2sm();
    }

    ptx::cluster_sync();
    // NOTE: op B is launched WITHOUT cudaLaunchAttributeProgrammaticStreamSerialization,
    // so it is not a PDL consumer. A cudaGridDependencySynchronize() here is a no-op in
    // isolation, but when op B directly follows op A's ProgrammaticStreamSerialization
    // reduce on the same stream it accumulates a back-to-back streaming stall. Removed:
    // op B still respects normal stream ordering (it starts only after op A completes).

    // ================================================================
    // FUSED ACTIVATION-QUANT PROLOGUE (all 512 threads, all CTAs)
    // bf16 x [M, lda_x] -> e4m3 x_fp8 [M, K_DIM] + native UE8M0 x_sf [M, K/128]
    // (1x128 per-row granularity == DSV4). One warp quantizes one K128 block
    // (4 elems/lane -> warp-shfl amax -> E=ceil(log2(amax/448)) -> v*2^-E).
    // Every CTA TMA-reads ALL M rows of x_fp8 (and warp2 reads x_sf), so a grid-wide
    // barrier must order the writes before any read. The persistent grid is capped at
    // #SMs with 1 CTA/SM (~200KB smem), so ALL CTAs are co-resident and a software
    // spin barrier on a monotonic global ticket counter is deadlock-free. The counter
    // is never reset: launch L spins to (L+1)*num_blocks (stream order keeps launches
    // disjoint). Only warps 0 (TMA) and 2 (SF reader) block on it; the epilogue is
    // gated transitively via tmem barriers and the CUDA-core tail never touches x_fp8.
    // ================================================================
    {
        constexpr int NUM_QBLOCKS = M_TPL * NUM_K_TILES;
        const int gwarp = blockIdx.x * (TPB / 32) + (int)warp_id;
        const int gwarps = num_blocks * (TPB / 32);
        for (int qb = gwarp; qb < NUM_QBLOCKS; qb += gwarps) {
            const int m = qb / NUM_K_TILES, b = qb % NUM_K_TILES;
            const nv_bfloat16* xrow = x_bf16 + (size_t)m * lda_x + b * QUANT_BLOCK_K + lane_id * 4;
            const float v0 = __bfloat162float(xrow[0]);
            const float v1 = __bfloat162float(xrow[1]);
            const float v2 = __bfloat162float(xrow[2]);
            const float v3 = __bfloat162float(xrow[3]);
            float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            int E = (int)ceilf(log2f(fmaxf(a, 1e-30f) / 448.0f));
            E = max(-127, min(127, E));
            const float sc = exp2f(-(float)E);
            uchar4 q;
            q.x = __nv_fp8_e4m3(v0 * sc).__x;
            q.y = __nv_fp8_e4m3(v1 * sc).__x;
            q.z = __nv_fp8_e4m3(v2 * sc).__x;
            q.w = __nv_fp8_e4m3(v3 * sc).__x;
            *reinterpret_cast<uchar4*>(x_fp8 + (size_t)m * K_DIM + b * QUANT_BLOCK_K + lane_id * 4) = q;
            if (lane_id == 0) x_sf[m * NUM_K_TILES + b] = (uint8_t)(E + 127);
        }
        __syncthreads();                       // CTA-local: all quant stores issued
        if (threadIdx.x == 0) {
            __threadfence();                   // release the quant writes to L2 (TMA-visible)
            const uint32_t old = atomicAdd(quant_sync, 1u);
            s.quant_target = (old / (uint32_t)num_blocks + 1u) * (uint32_t)num_blocks;
        }
        __syncthreads();                       // publish s.quant_target
        // Only the SF reader (warp2, reads x_sf) must block on the grid barrier here. The
        // TMA producer (warp0) does NOT spin here: its WEIGHT load is quant-independent and
        // should overlap the grid-sync -- warp0 gates ONLY its activation load, lazily, in
        // the producer loop below (see "gate the activation load").
        if (warp_id == 2) {
            if (ptx::elect_one_sync()) {
                const uint32_t tgt = s.quant_target;
                while (ptx::ld_acquire_gpu_u32(quant_sync) < tgt) {}
            }
            __syncwarp();
        }
    }

    // ================================================================
    // SMEM DESCRIPTOR PRE-COMPUTATION (warp-shuffle trick)
    // swap-AB: MMA A-operand = weight (smem_b), B-operand = activation (smem_a)
    // ================================================================
    auto* smem_a_base = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a);   // activation
    auto* smem_b_base = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b);   // weight

    auto act_desc = cluster_mma_fp8::detail::make_smem_desc_k_major<wq_b::BLOCK_K>(smem_a_base);  // B-operand
    auto wgt_desc = cluster_mma_fp8::detail::make_smem_desc_k_major<wq_b::BLOCK_K>(smem_b_base);  // A-operand

    uint32_t act_desc_lo = (lane_id < NUM_STAGES_T)
        ? act_desc.lo + lane_id * (SMEM_A_PER_STAGE_T / 16) : 0u;
    uint32_t wgt_desc_lo = (lane_id < NUM_STAGES_T)
        ? wgt_desc.lo + lane_id * (SMEM_B_PER_STAGE / 16) : 0u;

    auto fp8_instr_desc = cluster_mma_fp8::detail::make_block_scaled_idesc<UMMA_N_T>();

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
        // Weight (B) is offline-static -> independent of the fused activation quant, so its
        // TMA is issued WITHOUT waiting on the quant grid-barrier (it overlaps the quant +
        // grid-sync). Only the activation (A) load reads x_fp8, so gate JUST that on the
        // barrier, lazily on the first issue (quant_target was published by the prologue).
        const uint32_t quant_tgt = s.quant_target;
        bool act_synced = false;

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
                    auto* smem_a_dst = reinterpret_cast<__nv_fp8_e4m3*>(
                        s.smem_a + stage_idx * SMEM_A_PER_STAGE_T);
                    auto* smem_b_dst = reinterpret_cast<__nv_fp8_e4m3*>(
                        s.smem_b + stage_idx * SMEM_B_PER_STAGE);

                    // Per-CTA fp8 operand loads (SM90 plain; 2SM MMA gathers across cluster).
                    // WEIGHT first (no quant dependency -> overlaps the quant grid-sync).
                    tma::copy_2d_fp8(&desc_B, &s.full_barriers[stage_idx],
                                     smem_b_dst, k_offset, n_base);
                    // Gate the activation load on the quant grid-barrier (once): x_fp8 is
                    // written by ALL CTAs' prologue, so it must be globally visible first.
                    if (!act_synced) {
                        while (ptx::ld_acquire_gpu_u32(quant_sync) < quant_tgt) {}
                        act_synced = true;
                    }
                    // activation (A): outer = m_base. SF is NOT TMA-loaded (warp2 reads global).
                    tma::copy_2d_fp8(&desc_A, &s.full_barriers[stage_idx],
                                     smem_a_dst, k_offset, m_base);

                    constexpr uint32_t kNumArrivalBytes = SMEM_A_PER_STAGE_T + SMEM_B_PER_STAGE;
                    s.full_barriers[stage_idx].arrive_and_expect_tx(kNumArrivalBytes);

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

            // Each head keeps SUBTILES_PER_HEAD independent accumulators resident.
            for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                uint32_t tmem_c = accum_stage * ACCUM_COLS_T + sub * UMMA_N_T;

                for (int k = 0; k < NUM_K_TILES; ++k) {
                    s.with_sf_full_barriers[stage_idx].wait(phase);
                    ptx::tcgen05_fence_after_sync();

                    uint32_t w_base = __shfl_sync(0xffffffff, wgt_desc_lo, stage_idx);
                    uint32_t a_base = __shfl_sync(0xffffffff, act_desc_lo, stage_idx);

                    if (ptx::elect_one_sync()) {
                        // UTCCP this stage SF (smem -> TMEM), then block-scale MMA.
                        auto sf_desc = cluster_mma_fp8::detail::make_sf_desc();
                        const uint32_t* sf_act = reinterpret_cast<const uint32_t*>(
                            s.smem_sfa + stage_idx * SMEM_SFA_PER_STAGE_T);
                        const uint32_t* sf_wgt = reinterpret_cast<const uint32_t*>(
                            s.smem_sfb + stage_idx * SMEM_SFB_PER_STAGE_T);
                        #pragma unroll
                        for (int i = 0; i < wq_b::SF_BLOCK_M / wq_b::NUM_UTCCP_ALIGNED; ++i) {
                            cluster_mma_fp8::detail::replace_sf_desc_addr(sf_desc, sf_act + i * wq_b::NUM_UTCCP_ALIGNED);
                            cluster_mma_fp8::detail::utccp_4x32_2cta(TMEM_SFA_T + i * 4, cluster_mma_fp8::detail::sf_desc_bits(sf_desc));
                        }
                        #pragma unroll
                        for (int i = 0; i < wq_b::SF_BLOCK_N / wq_b::NUM_UTCCP_ALIGNED; ++i) {
                            cluster_mma_fp8::detail::replace_sf_desc_addr(sf_desc, sf_wgt + i * wq_b::NUM_UTCCP_ALIGNED);
                            cluster_mma_fp8::detail::utccp_4x32_2cta(TMEM_SFB_T + i * 4, cluster_mma_fp8::detail::sf_desc_bits(sf_desc));
                        }
                        #pragma unroll
                        for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                            uint32_t sf_id = kk;
                            uint64_t rdesc = cluster_mma_fp8::detail::make_runtime_idesc_with_sf_id(fp8_instr_desc, sf_id, sf_id);
                            uint32_t w_lo = cluster_mma_fp8::detail::advance_lo_k(w_base, kk);
                            uint32_t a_lo = cluster_mma_fp8::detail::advance_lo_k(a_base, kk);
                            uint64_t w_full = (static_cast<uint64_t>(wgt_desc.hi) << 32) | w_lo;
                            uint64_t a_full = (static_cast<uint64_t>(act_desc.hi) << 32) | a_lo;
                            uint32_t accum_flag = (k > 0 || kk > 0) ? 1 : 0;
                            // A-op=weight, B-op=activation (swap-AB); hw SFA=weight, SFB=act.
                            cluster_mma_fp8::detail::mma_2sm_block_scale(tmem_c, w_full, a_full, rdesc, accum_flag, TMEM_SFB_T, TMEM_SFA_T);
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

            persistent_iter++;
        }

        if (persistent_iter > 0) {
            uint32_t last_iter        = persistent_iter - 1;
            uint32_t last_accum_stage = last_iter % NUM_EPI_STAGES_T;
            uint32_t last_accum_phase = (last_iter / NUM_EPI_STAGES_T) & 1;
            s.tmem_empty_barriers[last_accum_stage].wait(last_accum_phase);
        }
    }

    // ======== WARP 2: NATIVE K128 SF EXPANDER (both CTAs) ========
    // Native DSV4 scales: x_sf[M, K/128] (1x128 rows), w_sf[N/128, K/128] (128x128
    // blocks), one UE8M0 byte each. Each byte is replicated into four identical K32
    // scale IDs (mxf8f6f4 consumes SF at K32) and stored directly in UTCCP's
    // [32 lanes][4 row groups] smem layout -- no SF TMA, no transpose pass.
    // NOTE: activation SF covers rows 0..127 (SF_BLOCK_M=128, full-M block, base 0 on
    // both CTAs) -- same coverage as the previous packed-SF path.
    else if (warp_id == 2) {
        auto pack_k128_sf = [](uint32_t e) { return e * 0x01010101u; };
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = (stage_idx + 1) % NUM_STAGES_T;
            if (stage_idx == 0) phase ^= 1;
        };
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                const int n_base = tile_id * HEAD_DIM + sub * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N;
                for (int k = 0; k < NUM_K_TILES; ++k) {
                    // Prefetch native scales while the operand TMA is in flight. One K128
                    // scale per activation row; one per aligned 128x128 weight block.
                    uint32_t va[4];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = i * 32 + (int)lane_id;
                        const uint32_t e = row < M_TPL
                            ? static_cast<uint32_t>(x_sf[row * NUM_K_TILES + k])
                            : UE8M0_ONE;
                        va[i] = pack_k128_sf(e);
                    }
                    uint32_t eb = UE8M0_ONE;
                    if (lane_id == 0)
                        eb = static_cast<uint32_t>(
                            w_sf[(n_base / WEIGHT_QUANT_BLOCK_N) * NUM_K_TILES + k]);
                    eb = __shfl_sync(0xffffffffu, eb, 0);
                    const uint32_t vb = pack_k128_sf(eb);

                    s.full_barriers[stage_idx].wait(phase);
                    auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage_idx * SMEM_SFA_PER_STAGE_T);
                    auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage_idx * SMEM_SFB_PER_STAGE_T);
                    // Directly produce UTCCP's [32 lanes][4 row groups] layout. Replicating
                    // the byte makes sf_id 0..3 share the official K128 scale.
                    ptx::st_shared_v4_u32(sfa + lane_id * 4, va[0], va[1], va[2], va[3]);
                    ptx::st_shared_v4_u32(sfb + lane_id * 4, vb, vb, vb, vb);
                    cutlass::arch::fence_view_async_shared();
                    s.with_sf_full_barriers[stage_idx].arrive(0u);
                    advance_pipeline();
                }
            }
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
                #pragma unroll
                for (int sub = 0; sub < SUBTILES_PER_HEAD; ++sub) {
                    uint32_t tmem_base = accum_stage * ACCUM_COLS_T + sub * UMMA_N_T;
                    uint8_t* smem_cd_ptr = stage_ptr + sub * SMEM_CD_PER_STAGE_T;
                    #pragma unroll
                    for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                        uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * 8;
                        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                        ptx::tmem_load_32dp32b8x(tmem_addr, v0, v1, v2, v3, v4, v5, v6, v7);
                        cutlass::arch::fence_view_async_tmem_load();
                        uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};

                        // BF16 store: 128 N per sub-tile = 2 atoms of 64 (2 warps/atom).
                        // Swizzle derived from the fp32 path (bank group = 16B = 8 bf16).
                        uint32_t atom         = epi_warp_idx >> 1;         // 0..1
                        uint32_t warp_in_atom = epi_warp_idx & 1u;         // 0/1
                        uint32_t n_in_atom    = warp_in_atom * 32 + lane_id; // 0..63 logical N
                        uint32_t bg           = n_in_atom >> 3;            // bank group 0..7
                        uint32_t intra        = n_in_atom & 7u;            // 0..7 within group
                        uint8_t* smem_base_ptr = smem_cd_ptr
                            + atom * (STORE_BLOCK_M * SWIZZLE_CD)
                            + i * (8 * SWIZZLE_CD);
                        #pragma unroll
                        for (uint32_t row = 0; row < 8; ++row) {
                            int m = st * STORE_BLOCK_M + i * 8 + row;
                            float scaled = __uint_as_float(vals[row]) * s.smem_rms[m];
                            nv_bfloat16 bscaled = __float2bfloat16(scaled);
                            uint8_t* smem_ptr = smem_base_ptr
                                + row * (16 * 8)
                                + ((bg ^ row) * 16)
                                + intra * (int)sizeof(nv_bfloat16);
                            ptx::st_shared_u16(smem_ptr, *reinterpret_cast<uint16_t*>(&bscaled));
                        }
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

            persistent_iter++;
        }
      }  // end epilogue

      // [profile] tensor-core path END (this CTA issued all epilogue stores).
      if (prof && epi_warp_idx == 0 && ptx::elect_one_sync())
          prof[blockIdx.x * 16 + 1] = ptx::globaltimer();
    }

    // ================================================================
    // CUDA-CORE TAIL WARPS (warps 8-15; 256 threads) -- run CONCURRENTLY with, and
    // fully decoupled from, the tensor-core GEMM path (warps 0-7).
    // ================================================================
    // Decoupling: touches NO TMEM / full-empty / tmem barriers / NamedBarrier id 0 and
    // NO __syncthreads. Its only cross-thread sync is a tail-local NamedBarrier (id 1)
    // over just these 256 threads. Per row (goal: hide this latency under the GEMM):
    //   STEP 1  split-K reduce : sum op A's ks_tail partials for cols [1536,4608) (y2|y3|y4)
    //   STEP 2  y2 RMSNorm+RoPE: weighted RMSNorm over the 512-wide y2, RoPE on its last 64
    //   STEP 3  y3/y4 state    : write the reduced y3/y4 into the compressor/indexer state
    //   STEP 4  compressor     : on compress rows, aggregate+shift+norm+rope+quant (fused)
    // (Init/cleanup cluster_sync are the ONLY points shared with the GEMM warps.)
    else if (warp_id >= TAIL_WARP_LO && warp_id < TAIL_WARP_LO + NUM_TAIL_WARPS &&
             ws_tail != nullptr && ks_tail > 0) {
        // ---- tail-group thread identity ----
        const int    tid        = threadIdx.x - (TAIL_WARP_LO * 32);  // 0..255 within the tail group
        const int    tail_warp  = tid >> 5;                           // 0..7 tail-warp index
        const size_t slab_elems = (size_t)ws_stride * M_TPL;          // elements in one split-K slab

        // ---- row layout: op A's [M, ws_stride] partial row is [y1 | y2 | y3 | y4] ----
        constexpr int Y2_LO = 1536, Y2_HI = 2048;      // y2 = cols [1536,2048), 512-wide
        constexpr int ROW_HI = 4608;                   // widest (fused) row; legacy 4352 clamped at runtime
        constexpr int Y2_W  = Y2_HI - Y2_LO;           // 512  (y2 RMSNorm reduction length)
        constexpr int ROPE_DIM   = 64;                 // RoPE acts on y2's LAST 64 dims
        constexpr int ROPE_PAIRS = ROPE_DIM / 2;       // 32 interleaved (2j,2j+1) pairs
        constexpr int ROPE_LO    = Y2_HI - ROPE_DIM;   // 1984 (global col of first rope dim)
        const bool do_rope = (rope_cs != nullptr);

        // ---- y3/y4 -> compressor/indexer state column layout within the row ----
        //   [2048,3072) comp wkv | [3072,4096) comp wgate | [4096,4352) idx wkv | [4352,4608) idx wgate
        constexpr int Y3_LO = 2048, WK_M = 1024, Y4_LO = 4096, WK_I = 256;
        const bool comp_active = (comp_kv != nullptr);
        const int  row_hi = ws_stride;                 // this row's real width (4352 legacy / 4608 fused)

        // Each thread owns a FIXED set of columns (stride NUM_TAIL_THREADS). The whole
        // [Y2_LO,ROW_HI) tail is read in ONE k-loop (max memory-level parallelism), so
        // COLS_PER_THREAD partials are in flight per split-K step.
        constexpr int Y2_COLS_PER_THREAD = Y2_W / NUM_TAIL_THREADS;              // 512 /256 = 2
        constexpr int COLS_PER_THREAD    = (ROW_HI - Y2_LO) / NUM_TAIL_THREADS;  // 3072/256 = 12

        // One row per CTA (grid-stride over the global block index -> disjoint rows, no
        // double-write). All 256 tail threads cooperate on each row.
        // (CC_beg is stamped at kernel entry, before the fused-quant prologue.)
        for (int m = blockIdx.x; m < M_TPL; m += num_blocks) {
            const size_t    row_base        = (size_t)m * ws_stride;
            const long long token_pos       = comp_active ? cmp_pos[m] : 0;      // this row's token position
            const int       pos_mod_ratio   = (int)(token_pos & 3);             // pos % RATIO(4)
            const size_t    state_slot      = (size_t)(m * 8 + 4 + pos_mod_ratio); // RATIO + pos%RATIO
            const bool      is_compress_row = comp_active && (((token_pos + 1) & 3) == 0); // (pos+1)%RATIO==0

            // ========================================================
            // STEP 1: split-K reduce -- sum ks_tail partials for cols [Y2_LO,ROW_HI).
            //   part[j] = reduced value of this thread's owned column j.
            // ========================================================
            float part[COLS_PER_THREAD];
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) part[j] = 0.f;
            for (int k = 0; k < ks_tail; ++k) {
                const size_t split_off = (size_t)k * slab_elems + row_base;
                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j) {
                    const int col = Y2_LO + tid + j * NUM_TAIL_THREADS;
                    if (col < row_hi) part[j] += ws_tail[split_off + col];       // clamp to real row width
                }
            }
            // [profile] combined load done (covers BOTH y2 and comp/idx loads)
            if (prof && is_compress_row && tid == 0) prof[blockIdx.x * 16 + 8] = ptx::globaltimer();

            // ========================================================
            // STEP 2a: y2 RMSNorm scale -- sum-of-squares over the 512-wide y2, reduced
            // across all 256 tail threads (per-warp partial -> tail_sq[8] -> rsqrt).
            // ========================================================
            float y2_sumsq = 0.f;
            #pragma unroll
            for (int j = 0; j < Y2_COLS_PER_THREAD; ++j) y2_sumsq += part[j] * part[j];
            y2_sumsq = ptx::warp_reduce_sum32(y2_sumsq);
            if ((tid & 31) == 0) s.tail_sq[tail_warp] = y2_sumsq;
            cutlass::arch::NamedBarrier::sync(NUM_TAIL_THREADS, 1);          // tail-only barrier (id 1)
            float y2_sumsq_all = 0.f;
            #pragma unroll
            for (int w = 0; w < NUM_TAIL_WARPS; ++w) y2_sumsq_all += s.tail_sq[w];
            const float y2_rms = rsqrtf(y2_sumsq_all / float(Y2_W) + eps);
            cutlass::arch::NamedBarrier::sync(NUM_TAIL_THREADS, 1);          // protect tail_sq before next row
            if (prof && is_compress_row && tid == 0) prof[blockIdx.x * 16 + 9] = ptx::globaltimer();
            // [profile] no separate pass2 load anymore -> p2_load collapses to ~0 (stamp kept for the table)
            if (prof && is_compress_row && tid == 0) prof[blockIdx.x * 16 + 10] = ptx::globaltimer();

            // ========================================================
            // STEP 2b/3: store this thread's owned columns.
            //   y2   (col < Y2_HI) -> weighted RMSNorm (its last-64 rope pairs deferred to STEP 2c)
            //   y3/y4 (comp path)  -> write the reduced value into the compressor/indexer state (+ape)
            //   y3/y4 (legacy)     -> raw bf16 reduce-write
            // ========================================================
            #pragma unroll
            for (int j = 0; j < COLS_PER_THREAD; ++j) {
                const int col = Y2_LO + tid + j * NUM_TAIL_THREADS;
                if (col >= row_hi) continue;                                // past this row's real width
                const float val = part[j];
                if (col < Y2_HI) {                                          // ---- y2: weighted RMSNorm ----
                    if (do_rope && col >= ROPE_LO) continue;                // last-64 rope pairs handled in STEP 2c
                    dtail[row_base + col] = __float2bfloat16(val * y2_rms * rms_w2[col - Y2_LO]);
                } else if (comp_active) {                                   // ---- y3/y4: write into state (+ape) ----
                    if (col < Y3_LO + WK_M) {                               // comp wkv
                        comp_kv[state_slot * WK_M + (col - Y3_LO)] = val;
                    } else if (col < Y4_LO) {                               // comp wgate (+ape)
                        const int i = col - (Y3_LO + WK_M);
                        comp_sc[state_slot * WK_M + i] = val + comp_ape[(size_t)pos_mod_ratio * WK_M + i];
                    } else if (col < Y4_LO + WK_I) {                        // idx wkv
                        idx_kv[state_slot * WK_I + (col - Y4_LO)] = val;
                    } else {                                               // idx wgate (+ape)
                        const int i = col - (Y4_LO + WK_I);
                        idx_sc[state_slot * WK_I + i] = val + idx_ape[(size_t)pos_mod_ratio * WK_I + i];
                    }
                } else {                                                   // ---- legacy: raw y3/y4 reduce-write ----
                    dtail[row_base + col] = __float2bfloat16(val);
                }
            }
            // [profile] reduce done HERE (split-K numeric reduce + state writes), BEFORE the
            // prefetch below (which is aggregate prep, charged to mainAgg = prof[5]-prof[4]).
            if (prof && is_compress_row && tid == 0) prof[blockIdx.x * 16 + 4] = ptx::globaltimer();

            // ========================================================
            // STEP 4 prep [prefetch]: load the main compressor's 7 HISTORICAL overlap rows
            // (0-6) into registers NOW (after the reduce). part[] is dead here, so this no
            // longer inflates the reduce-phase register peak (avoids the spill). Rows 0-6 are
            // final for this token (this row only wrote slot 7; the shift runs later) =>
            // byte-exact. The 28 loads fly under the RoPE + id-1 barrier + aggregate below.
            //   hist_sc/hist_kv[col-half ci in {tid, tid+256}][historical overlap row 0..6]
            // ========================================================
            float hist_sc[2][7], hist_kv[2][7];
            if (is_compress_row) {
                const float* state_sc_base = comp_sc + (size_t)m * 8 * WK_M;
                const float* state_kv_base = comp_kv + (size_t)m * 8 * WK_M;
                #pragma unroll
                for (int ci = 0; ci < 2; ++ci) {
                    const int c = tid + ci * NUM_TAIL_THREADS;
                    #pragma unroll
                    for (int rr = 0; rr < 7; ++rr) {
                        const int col = (rr < 4) ? c : (512 + c);           // overlap-cat: rows<4 -> [0,512), rows>=4 -> [512,1024)
                        hist_sc[ci][rr] = state_sc_base[(size_t)rr * WK_M + col];
                        hist_kv[ci][rr] = state_kv_base[(size_t)rr * WK_M + col];
                    }
                }
            }

            // ========================================================
            // STEP 2c: interleaved RoPE on the NORMALIZED last-64 y2 dims. Thread tid<32
            // owns pair j=tid and rotates (even,odd) by this row's (cos_j,sin_j); pairs are
            // adjacent (2j,2j+1) per model.py apply_rotary_emb (view_as_complex).
            // ========================================================
            if (do_rope && tid < ROPE_PAIRS) {
                const int j        = tid;
                const int col_even = ROPE_LO + 2 * j;        // even global col
                const int col_odd  = col_even + 1;           // odd  global col
                float val_even = 0.f, val_odd = 0.f;
                for (int k = 0; k < ks_tail; ++k) {
                    val_even += ws_tail[(size_t)k * slab_elems + row_base + col_even];
                    val_odd  += ws_tail[(size_t)k * slab_elems + row_base + col_odd];
                }
                val_even = val_even * y2_rms * rms_w2[col_even - Y2_LO];   // weighted-normalize BEFORE rotating
                val_odd  = val_odd  * y2_rms * rms_w2[col_odd  - Y2_LO];
                const float2 cossin = rope_cs[(size_t)m * ROPE_PAIRS + j]; // (cos_j, sin_j)
                dtail[row_base + col_even] = __float2bfloat16(val_even * cossin.x - val_odd * cossin.y);
                dtail[row_base + col_odd]  = __float2bfloat16(val_even * cossin.y + val_odd * cossin.x);
            }

            // ========================================================
            // STEP 4: fused compressor post-processing for this row.
            // ========================================================
            if (comp_active) {
                cutlass::arch::NamedBarrier::sync(NUM_TAIL_THREADS, 1);      // all y3/y4 state slots for this row written
                if (is_compress_row) {
                    // [4a: aggregate] softmax over the register-prefetched historical rows 0-6
                    // (hist_*) + fresh row 7. Accumulation order (rows 0..6 then 7) matches the
                    // in-compressor version => byte-exact. Direct-index hist_* (no sc/kv copy).
                    {
                        const float* row7_sc = comp_sc + (size_t)(m * 8 + 7) * WK_M;   // slot 7 = this token's fresh row
                        const float* row7_kv = comp_kv + (size_t)(m * 8 + 7) * WK_M;
                        #pragma unroll
                        for (int ci = 0; ci < 2; ++ci) {
                            const int   c      = tid + ci * NUM_TAIL_THREADS;
                            const float sc_cur = row7_sc[512 + c];          // row 7 (>=RATIO) uses col 512+c
                            const float kv_cur = row7_kv[512 + c];
                            float max_logit = sc_cur;
                            #pragma unroll
                            for (int rr = 0; rr < 7; ++rr) max_logit = fmaxf(max_logit, hist_sc[ci][rr]);
                            float softmax_denom = 0.f, weighted_sum = 0.f;
                            #pragma unroll
                            for (int rr = 0; rr < 7; ++rr) {
                                const float w = expf(hist_sc[ci][rr] - max_logit);
                                softmax_denom += w;
                                weighted_sum  += w * hist_kv[ci][rr];
                            }
                            const float w_cur = expf(sc_cur - max_logit);   // fresh row 7 contribution
                            softmax_denom += w_cur;
                            weighted_sum  += w_cur * kv_cur;
                            s.cmp[c] = weighted_sum / softmax_denom;
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(NUM_TAIL_THREADS, 1);  // aggregated s.cmp ready for the compressor
                    if (prof && tid == 0) prof[blockIdx.x * 16 + 5] = ptx::globaltimer(); // [phase] aggregate done
                    // [4b: shift + RMSNorm + RoPE + real fp8(main)/fp4(indexer) quant]
                    compressor_process_row<M_TPL>(s, tid, m, eps, token_pos,
                        comp_norm, idx_norm, cos_tab, sin_tab,
                        comp_kv, comp_sc, idx_kv, idx_sc,
                        comp_q8, comp_s8, comp_rope, idx_q4, idx_s4, prof);
                }
            }
        }
        // [profile] cuda-core tail END (all tail rows reduced/normalized/stored).
        if (prof && tid == 0) prof[blockIdx.x * 16 + 3] = ptx::globaltimer();
    }

    // ================================================================
    // CLEANUP
    // ================================================================
    ptx::cluster_sync();
    if (warp_id == 0) {
        ptx::tcgen05_dealloc_2sm(s.tmem_base, NUM_TMEM_COLS_T);
    }
}

// ======================== Host: TMA Descriptors ========================
static CUtensorMap make_tma_desc_bf16_2d(
    const nv_bfloat16* ptr, int rows, int cols, int box_rows, int box_cols, int ld_cols = -1)
{
    CUtensorMap desc{};
    const int ld = (ld_cols < 0) ? cols : ld_cols;   // row pitch in ELEMENTS (>= cols); lets op B
                                                     // read a [rows,cols] view out of a WIDER buffer
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)ld * sizeof(nv_bfloat16)};
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

// ---- Standalone activation quant (REFERENCE / debug): bf16 [M,K] (pitch lda) -> e4m3
//      + native UE8M0 SF [M, K/128] (1x128 per-row, DSV4 granularity). The production
//      path FUSES this into wq_b_proj_kernel's prologue; this kernel stays for the
//      harness's standalone quant-latency reference. grid = (M, NUM_K_TILES), 128 thr.
__global__ void quant_act_gran128(const nv_bfloat16* __restrict__ x, int M, int lda,
                                  __nv_fp8_e4m3* __restrict__ qout, uint8_t* __restrict__ sf) {
    const int r = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int k = b * wq_b::QUANT_BLOCK_K + tid;
    float v = (r < M) ? __bfloat162float(x[(size_t)r * lda + k]) : 0.0f;
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wmax[4];
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) wmax[wid] = a;
    __syncthreads();
    float amax = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
    int E = (int)ceilf(log2f(fmaxf(amax, 1e-30f) / 448.0f));
    E = max(-127, min(127, E));
    if (r < M) {
        qout[(size_t)r * wq_b::K_DIM + k] = __nv_fp8_e4m3(v * exp2f(-(float)E));
        if (tid == 0) sf[(size_t)r * wq_b::NUM_K_TILES + b] = (uint8_t)(E + 127);
    }
}

// ---- Offline-static weight quant: bf16 [N,K] -> e4m3 + native UE8M0 SF [N/128, K/128]
//      (128x128 block granularity == DSV4). One 128-thread block per 128x128 tile;
//      thread t owns tile row t and walks it twice (amax pass, then quantize+store;
//      the row is L2-hot on the second pass). Requires lda % 8 == 0 (16B vectors).
//      grid = (NUM_WEIGHT_SF_ROWS, NUM_K_TILES).
__global__ void quant_weights_block128(const nv_bfloat16* __restrict__ w, int lda,
                                       __nv_fp8_e4m3* __restrict__ qout, uint8_t* __restrict__ sf) {
    const int br = blockIdx.x, bk = blockIdx.y;
    const int tid = threadIdx.x;
    const int r = br * wq_b::WEIGHT_QUANT_BLOCK_N + tid;
    const nv_bfloat16* wrow = w + (size_t)r * lda + bk * wq_b::QUANT_BLOCK_K;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < wq_b::QUANT_BLOCK_K / 8; ++i) {
        const uint4 pk = *reinterpret_cast<const uint4*>(wrow + i * 8);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&pk);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float wmax[4];
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) wmax[wid] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
    int E = (int)ceilf(log2f(fmaxf(amax, 1e-30f) / 448.0f));
    E = max(-127, min(127, E));
    const float sc = exp2f(-(float)E);
    __nv_fp8_e4m3* orow = qout + (size_t)r * wq_b::K_DIM + bk * wq_b::QUANT_BLOCK_K;
    #pragma unroll
    for (int i = 0; i < wq_b::QUANT_BLOCK_K / 8; ++i) {
        const uint4 pk = *reinterpret_cast<const uint4*>(wrow + i * 8);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&pk);
        const float2 f0 = __bfloat1622float2(h[0]);
        const float2 f1 = __bfloat1622float2(h[1]);
        const float2 f2 = __bfloat1622float2(h[2]);
        const float2 f3 = __bfloat1622float2(h[3]);
        uchar4 qa, qb;
        qa.x = __nv_fp8_e4m3(f0.x * sc).__x; qa.y = __nv_fp8_e4m3(f0.y * sc).__x;
        qa.z = __nv_fp8_e4m3(f1.x * sc).__x; qa.w = __nv_fp8_e4m3(f1.y * sc).__x;
        qb.x = __nv_fp8_e4m3(f2.x * sc).__x; qb.y = __nv_fp8_e4m3(f2.y * sc).__x;
        qb.z = __nv_fp8_e4m3(f3.x * sc).__x; qb.w = __nv_fp8_e4m3(f3.y * sc).__x;
        *reinterpret_cast<uchar4*>(orow + i * 8)     = qa;
        *reinterpret_cast<uchar4*>(orow + i * 8 + 4) = qb;
    }
    if (tid == 0) sf[(size_t)br * wq_b::NUM_K_TILES + bk] = (uint8_t)(E + 127);
}

static CUtensorMap make_tma_desc_fp8_2d(
    const __nv_fp8_e4m3* ptr, int rows, int cols, int box_rows, int box_cols) {
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

// ======================== Host launcher (plain CUDA; no torch) ========================
// x     : activation y1, bf16 [M, K_DIM=1536]. Row pitch = `lda` ELEMENTS (>= K_DIM).
//         Pass lda=4608 to read y1 DIRECTLY from complex_a's [M,4608] D buffer
//         (first 1536 cols, zero-copy); pass lda=K_DIM (default) for a packed input.
//         The bf16 -> e4m3 + UE8M0 activation quant is FUSED into the kernel prologue.
// w_fp8 : weight w2, e4m3 [N_TOTAL=65536, K_DIM=1536], row-major (offline quantized).
// w_sf  : native DSV4 weight SF, uint8 [NUM_WEIGHT_SF_ROWS=512, K/128=12] (128x128 blocks).
// out   : bf16 [M, N_TOTAL], row-major (contiguous). head_dim(512) RMSNorm fused (weightless).
// M     : 16-aligned, in [16,256]. NOTE: activation-SF coverage is rows 0..127 (SF_BLOCK_M).
static cudaError_t wq_b_proj_run(
    nv_bfloat16* out, const nv_bfloat16* x, const __nv_fp8_e4m3* w_fp8, const uint8_t* w_sf,
    int M, float eps, int lda = K_DIM, cudaStream_t stream = 0,
    const float* ws_tail = nullptr, int ks_tail = 0,
    const float* rms_w2 = nullptr, nv_bfloat16* dtail = nullptr,
    int ws_stride = 4352, const float2* rope_cs = nullptr,
    unsigned long long* prof = nullptr,
    // ---- fused compressor (y3/y4); comp_kv==nullptr => disabled (legacy y3/y4 reduce-write) ----
    const long long* cmp_pos = nullptr,
    const float* comp_ape = nullptr, const float* idx_ape = nullptr,
    const float* comp_norm = nullptr, const float* idx_norm = nullptr,
    const float* cos_tab = nullptr,  const float* sin_tab = nullptr,
    float* comp_kv = nullptr, float* comp_sc = nullptr,
    float* idx_kv = nullptr,  float* idx_sc = nullptr,
    uint8_t* comp_q8 = nullptr, float* comp_s8 = nullptr, nv_bfloat16* comp_rope = nullptr,
    uint8_t* idx_q4 = nullptr,  uint8_t* idx_s4 = nullptr)
{
    if (!(M >= 16 && M <= 256 && M % 16 == 0)) return cudaErrorInvalidValue;
    if (lda % 4 != 0) return cudaErrorInvalidValue;   // fused quant reads 8B vectors

    // TMA descriptors. Activation (x=y1) uses a pitched row stride (lda) so a
    // [M,1536] view can be read out of a wider (e.g. 4608) buffer with no copy.
    const int load_block_m = M / NUM_MULTICAST;   // activation rows per CTA
    // ---- Fused-quant buffers: the kernel's prologue writes x_fp8 / x_sf and TMA/warp2
    //      read them after the in-kernel grid barrier (s_quant_sync ticket counter).
    static __nv_fp8_e4m3* s_x_fp8      = nullptr;
    static uint8_t*       s_x_sf       = nullptr;
    static uint32_t*      s_quant_sync = nullptr;
    if (s_x_fp8 == nullptr) {
        cudaMalloc(&s_x_fp8, (size_t)256 * K_DIM * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&s_x_sf,  (size_t)256 * wq_b::NUM_K_TILES * sizeof(uint8_t));
        cudaMalloc(&s_quant_sync, sizeof(uint32_t));
        cudaMemset(s_quant_sync, 0, sizeof(uint32_t));
    }
    CUtensorMap desc_A   = make_tma_desc_fp8_2d(s_x_fp8, M,       K_DIM,   load_block_m, BLOCK_K);
    CUtensorMap desc_B   = make_tma_desc_fp8_2d(w_fp8,   N_TOTAL, K_DIM,   LOAD_BLOCK_N, BLOCK_K);
    CUtensorMap desc_D   = make_tma_desc_bf16_2d(out, M,       N_TOTAL, STORE_BLOCK_M, STORE_BLOCK_N_ATOM);

    // Grid: persistent, cluster of 2 CTAs. Cache SM count once per process.
    static const int num_SMs = []() {
        int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n;
    }();
    // One cluster processes a whole head (fused RMSNorm), so scheduling unit = head.
    int total_cta = NUM_HEAD_TILES * CLUSTER_SIZE;
    int grid_size = (num_SMs < total_cta) ? num_SMs : total_cta;
    grid_size = (grid_size / CLUSTER_SIZE) * CLUSTER_SIZE;   // multiple of cluster size
    if (grid_size < CLUSTER_SIZE) grid_size = CLUSTER_SIZE;

    // Select kernel + SMEM by M (compile-time).
    void* kernel_ptr = nullptr;
    int smem_bytes = 0;
    switch (M) {
        case 16:  kernel_ptr = (void*)&wq_b_proj_kernel<16>;  smem_bytes = sizeof(SharedStorage<16>);  break;
        case 32:  kernel_ptr = (void*)&wq_b_proj_kernel<32>;  smem_bytes = sizeof(SharedStorage<32>);  break;
        case 48:  kernel_ptr = (void*)&wq_b_proj_kernel<48>;  smem_bytes = sizeof(SharedStorage<48>);  break;
        case 64:  kernel_ptr = (void*)&wq_b_proj_kernel<64>;  smem_bytes = sizeof(SharedStorage<64>);  break;
        case 80:  kernel_ptr = (void*)&wq_b_proj_kernel<80>;  smem_bytes = sizeof(SharedStorage<80>);  break;
        case 96:  kernel_ptr = (void*)&wq_b_proj_kernel<96>;  smem_bytes = sizeof(SharedStorage<96>);  break;
        case 112:  kernel_ptr = (void*)&wq_b_proj_kernel<112>;  smem_bytes = sizeof(SharedStorage<112>);  break;
        case 128:  kernel_ptr = (void*)&wq_b_proj_kernel<128>;  smem_bytes = sizeof(SharedStorage<128>);  break;
        case 144:  kernel_ptr = (void*)&wq_b_proj_kernel<144>;  smem_bytes = sizeof(SharedStorage<144>);  break;
        case 160:  kernel_ptr = (void*)&wq_b_proj_kernel<160>;  smem_bytes = sizeof(SharedStorage<160>);  break;
        case 176:  kernel_ptr = (void*)&wq_b_proj_kernel<176>;  smem_bytes = sizeof(SharedStorage<176>);  break;
        case 192:  kernel_ptr = (void*)&wq_b_proj_kernel<192>;  smem_bytes = sizeof(SharedStorage<192>);  break;
        case 208:  kernel_ptr = (void*)&wq_b_proj_kernel<208>;  smem_bytes = sizeof(SharedStorage<208>);  break;
        case 224:  kernel_ptr = (void*)&wq_b_proj_kernel<224>;  smem_bytes = sizeof(SharedStorage<224>);  break;
        case 240:  kernel_ptr = (void*)&wq_b_proj_kernel<240>;  smem_bytes = sizeof(SharedStorage<240>);  break;
        case 256:  kernel_ptr = (void*)&wq_b_proj_kernel<256>;  smem_bytes = sizeof(SharedStorage<256>);  break;
        default: return cudaErrorInvalidValue;
    }

    // Configure max dynamic SMEM once per kernel variant (M/32 -> 1..8).
    static bool smem_configured[17] = {false};
    const int m_idx = M / 16;
    if (!smem_configured[m_idx]) {
        cudaError_t attr_err = cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if (attr_err != cudaSuccess) return attr_err;
        smem_configured[m_idx] = true;
    }

    dim3 grid(grid_size, 1, 1);
    dim3 block(TPB, 1, 1);
    cudaLaunchConfig_t config = {};
    config.gridDim         = grid;
    config.blockDim        = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream          = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    void* ptr_args[] = { &desc_A, &desc_B, &x, &lda, &s_x_fp8, &s_x_sf, &w_sf, &s_quant_sync,
                         &desc_D, &grid_size, &eps,
                         &ws_tail, &ks_tail, &rms_w2, &dtail, &ws_stride, &rope_cs, &prof,
                         // ---- fused compressor (order MUST match wq_b_proj_kernel signature) ----
                         &cmp_pos, &comp_ape, &idx_ape, &comp_norm, &idx_norm,
                         &cos_tab, &sin_tab, &comp_kv, &comp_sc, &idx_kv, &idx_sc,
                         &comp_q8, &comp_s8, &comp_rope, &idx_q4, &idx_s4 };
    return cudaLaunchKernelExC(&config, kernel_ptr, ptr_args);
}

// ---- Convenience overload: bf16 weight in (offline-static-quantized ONCE, cached). ----
// Lets existing bf16 test harnesses call op B unchanged; weight fp8-quant is done a single
// time per weight buffer (NOT per call), i.e. equivalent to offline static quantization.
// Weight granularity = NATIVE DSV4 128x128 blocks (quant_weights_block128).
static cudaError_t wq_b_proj_run(
    nv_bfloat16* out, const nv_bfloat16* x, const nv_bfloat16* w_bf16,
    int M, float eps, int lda = K_DIM, cudaStream_t stream = 0,
    const float* ws_tail = nullptr, int ks_tail = 0,
    const float* rms_w2 = nullptr, nv_bfloat16* dtail = nullptr,
    int ws_stride = 4352, const float2* rope_cs = nullptr,
    unsigned long long* prof = nullptr,
    const long long* cmp_pos = nullptr,
    const float* comp_ape = nullptr, const float* idx_ape = nullptr,
    const float* comp_norm = nullptr, const float* idx_norm = nullptr,
    const float* cos_tab = nullptr,  const float* sin_tab = nullptr,
    float* comp_kv = nullptr, float* comp_sc = nullptr,
    float* idx_kv = nullptr,  float* idx_sc = nullptr,
    uint8_t* comp_q8 = nullptr, float* comp_s8 = nullptr, nv_bfloat16* comp_rope = nullptr,
    uint8_t* idx_q4 = nullptr,  uint8_t* idx_s4 = nullptr)
{
    static __nv_fp8_e4m3* s_w_fp8 = nullptr;
    static uint8_t*       s_w_sf  = nullptr;
    static const nv_bfloat16* s_w_src = nullptr;
    if (s_w_fp8 == nullptr) {
        cudaMalloc(&s_w_fp8, (size_t)N_TOTAL * K_DIM * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&s_w_sf,  (size_t)wq_b::NUM_WEIGHT_SF_ROWS * wq_b::NUM_K_TILES * sizeof(uint8_t));
    }
    if (s_w_src != w_bf16) {
        quant_weights_block128<<<dim3(wq_b::NUM_WEIGHT_SF_ROWS, wq_b::NUM_K_TILES), 128, 0, stream>>>(
            w_bf16, K_DIM, s_w_fp8, s_w_sf);
        s_w_src = w_bf16;
    }
    return wq_b_proj_run(out, x, s_w_fp8, s_w_sf, M, eps, lda, stream,
        ws_tail, ks_tail, rms_w2, dtail, ws_stride, rope_cs, prof,
        cmp_pos, comp_ape, idx_ape, comp_norm, idx_norm, cos_tab, sin_tab,
        comp_kv, comp_sc, idx_kv, idx_sc, comp_q8, comp_s8, comp_rope, idx_q4, idx_s4);
}

} // namespace gfnb