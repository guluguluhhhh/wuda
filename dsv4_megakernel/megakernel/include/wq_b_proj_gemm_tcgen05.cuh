#pragma once
// ============================================================
// wq_b_proj_gemm_tcgen05.cuh
// tcgen05 BF16 GEMM — Header
// Uses same CUTLASS components as DeepGEMM (minimal subset)
//
// Target: M=32~256, K=1536, N=65536, BF16, 2SM MMA, Cluster=(2,1,1)
// Dependencies: cutlass/arch/barrier.h, cute/arch/mma_sm100_desc.hpp,
//               cute/arch/copy_sm90_tma.hpp, cute/arch/copy_sm100_tma.hpp
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

// ======================== Configuration ========================
namespace wq_b {

// Tile dimensions (cluster-level)
// DeepGEMM heuristic for M<=32: BLOCK_M=32, for M<=64: BLOCK_M=64, else: 128
// We use BLOCK_M=64 as default (covers M=32~64 with padding)
static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 256;
static constexpr int BLOCK_K = 64;

// Cluster: (2,1,1) → 2SM MMA, multicast on A (along M)
static constexpr int CLUSTER_M = 2;
static constexpr int CLUSTER_N = 1;

// Per-CTA load dimensions
// DeepGEMM: kIsMulticastOnA = (cluster_n > 1) = false for cluster=(2,1,1)
// So multicast is on B side (cluster_m splits N)
static constexpr bool IS_MULTICAST_ON_A = false;  // cluster_n == 1
static constexpr int LOAD_BLOCK_M = BLOCK_M;                                        // 64 (full, no split)
static constexpr int LOAD_BLOCK_N = BLOCK_N / (IS_MULTICAST_ON_A ? 1 : CLUSTER_M);  // 256/2 = 128

// MMA instruction shape (2SM: LAYOUT_AD_M=128 × kNumMulticast=2 = 256)
static constexpr int UMMA_M = 256;
static constexpr int UMMA_N = BLOCK_N;  // 256
static constexpr int UMMA_K = 16;       // BF16: 16 elements = 32 bytes

// Layout: both A and B are K-major
static constexpr auto MAJOR_A = cute::UMMA::Major::K;
static constexpr auto MAJOR_B = cute::UMMA::Major::K;

// Swizzle: 128B (= BLOCK_K * sizeof(bf16) = 64*2 = 128)
static constexpr int SWIZZLE_A = 128;    // bytes
static constexpr int SWIZZLE_B = 128;    // bytes
static constexpr int SWIZZLE_CD = 128;   // bytes

// Pipeline (DeepGEMM: auto-calculated from SMEM capacity)
// smem_per_stage = LOAD_BLOCK_M*BK*2 + LOAD_BLOCK_N*BK*2 = 64*64*2 + 128*64*2 = 8192+16384 = 24576
// smem_available ≈ 228KB - CD - barriers ≈ 220KB
// max_stages = 220KB / 24KB ≈ 9 → use 8 (conservative)
static constexpr int NUM_STAGES = 8;
static constexpr int NUM_EPI_STAGES = 2; // TMEM double buffer
static constexpr int NUM_TMA_STORE_STAGES = 2;

// Threads: 128 non-epilogue + 128 epilogue = 256
static constexpr int TPB = 256;
static constexpr int NUM_NON_EPI_THREADS = 128;
static constexpr int NUM_EPI_THREADS = 128;

// Epilogue store tile (DeepGEMM L79: store_block_n = kSwizzleCDMode / sizeof(cd_dtype_t))
static constexpr int STORE_BLOCK_M = BLOCK_M;                        // 64
static constexpr int STORE_BLOCK_N = SWIZZLE_CD / (int)sizeof(float); // 128/4 = 32
static constexpr int NUM_STORE_THREADS = STORE_BLOCK_M;               // 64 (warps 4-5)

// SMEM sizes per stage (bytes)
static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(nv_bfloat16); // 64*64*2 = 8192
static constexpr int SMEM_B_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(nv_bfloat16); // 128*64*2 = 16384
static constexpr int SMEM_CD_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(float); // 64*32*4 = 8192
static constexpr int SMEM_CD_TOTAL = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES;          // 16384

// Total TMA bytes per K-tile per multicast group
static constexpr int TMA_BYTES_PER_STAGE = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE; // 40960

// TMEM columns: 2 epilogue stages × UMMA_N
static constexpr int NUM_TMEM_COLS = NUM_EPI_STAGES * UMMA_N; // 512

// Problem dimensions (fixed for wq_b projection)
static constexpr int K_DIM = 1536;
static constexpr int N_TOTAL = 65536;  // 128 heads × 512 dim
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K; // 24

// Multicast
static constexpr int NUM_MULTICAST = CLUSTER_M;  // 2

} // namespace wq_b

// ======================== Descriptor Helpers ========================
// Directly from DeepGEMM mma/sm100.cuh, simplified for our fixed config
namespace mma_desc {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// Make SMEM descriptor using CUTLASS SmemDescriptor bitfield
// (from DeepGEMM: deep_gemm/mma/sm100.cuh::make_smem_desc + make_umma_desc)
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_smem_desc_k_major(void* smem_ptr) {
    // Aligned with DeepGEMM mma/sm100.cuh::make_smem_desc
    cute::UMMA::SmemDescriptor desc;

    // Set the version for SM100
    desc.version_ = 1;

    // Legacy mode
    desc.lbo_mode_ = 0;

    // Layout type: SWIZZLE_128B = 2
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);

    // Start address
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // For K-major with 128B swizzle:
    //   num_non_contiguous = 128 / atom_base = 128 / 16 = 8
    //   SBO = num_non_contiguous * BLOCK_K * sizeof(bf16) = 8 * 64 * 2 = 1024
    //   LBO = 0 (only 1 K-atom)
    constexpr uint32_t SBO = 8 * wq_b::BLOCK_K * sizeof(nv_bfloat16); // 1024
    desc.stride_byte_offset_ = SBO >> 4;   // 64
    desc.leading_byte_offset_ = 0;

    // Base offset
    desc.base_offset_ = 0;

    return desc;
}

// Advance descriptor's .lo for the next stage (offset by stage_bytes / 16)
__device__ __forceinline__
uint32_t advance_desc_lo_for_stage(uint32_t base_lo, uint32_t stage_idx, uint32_t stage_bytes) {
    return base_lo + stage_idx * (stage_bytes / 16);
}

// Advance descriptor's .lo for K-step within a stage
// For K-major: stride_k = 1 element → sizeof(bf16) bytes per step
// Each UMMA_K = 16 elements → 32 bytes → 32/16 = 2 units
__device__ __forceinline__
uint32_t advance_desc_lo_for_k(uint32_t base_lo, uint32_t k_idx) {
    return base_lo + k_idx * (wq_b::UMMA_K * sizeof(nv_bfloat16) / 16); // k_idx * 2
}

// Make instruction descriptor for 2SM BF16 GEMM
// From DeepGEMM: cute::UMMA::make_instr_desc<bf16, bf16, float, 256, 256, K, K>()
__device__ __forceinline__
uint64_t make_runtime_instr_desc() {
    auto idesc = cute::UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        wq_b::UMMA_M, wq_b::UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    return cute::UMMA::make_runtime_instr_desc(idesc);
}

} // namespace mma_desc

// ======================== PTX Wrappers ========================
// Only for instructions not covered by CUTLASS headers
namespace ptx {

// tcgen05.mma.cta_group::2.kind::f16 (from DeepGEMM ptx/tcgen05.cuh)
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

// TMEM allocation for 2SM (from DeepGEMM, same as cute::TMEM::Allocator2Sm)
__device__ __forceinline__ void tcgen05_alloc_2sm(uint32_t smem_addr, uint32_t num_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(num_cols));
}

__device__ __forceinline__ void tcgen05_dealloc_2sm(uint32_t taddr, uint32_t num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(num_cols));
}

// Fences (from DeepGEMM ptx/tcgen05.cuh)
__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// umma_arrive for 2SM multicast (from cutlass::arch)
__device__ __forceinline__ void umma_arrive_multicast_2sm(uint64_t* bar, uint16_t mask) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    if (cute::elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :: "r"(addr), "h"(mask) : "memory");
    }
}

// TMEM load: 32dp32b4x (FP32, 4 values per lane)
__device__ __forceinline__ void tmem_load_32dp32b4x(
    uint32_t tmem_addr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Store 4×32-bit to shared memory
__device__ __forceinline__ void st_shared_v4(void* ptr, uint32_t v0, uint32_t v1,
                                              uint32_t v2, uint32_t v3) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

// BF16 pack (2 floats → 1 uint32 packed bf16x2)
__device__ __forceinline__ uint32_t bf16x2_pack(float a, float b) {
    uint32_t result;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(a), "f"(b));
    return result;
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

} // namespace ptx

// ======================== TMA Copy Helpers ========================
// Aligned with DeepGEMM common/tma_copy.cuh — uses CuTe API directly
namespace tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// TMA 2D copy for 2SM mode (DeepGEMM tma_copy.cuh L37-44)
// For K-major BF16, BLOCK_K=64, SWIZZLE_128B: atom_size = 128/2 = 64 elements
// So BLOCK_K / atom_size = 64/64 = 1 → single TMA per tile (no loop needed)
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

// TMA store 2D (DeepGEMM sm100_store_cd.cuh L128)
__device__ __forceinline__
void store_2d(void const* desc_ptr, void* smem_ptr,
              uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

} // namespace tma

// ======================== Pipeline State ========================
namespace wq_b {

struct PipeState {
    uint32_t index = 0;
    uint32_t phase = 0;

    __device__ __forceinline__ void advance() {
        if (++index >= NUM_STAGES) { index = 0; phase ^= 1; }
    }
};

} // namespace wq_b
