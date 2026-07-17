#pragma once
// ============================================================
// hc_fused_kernel_tc.cuh
// Reusable tcgen05 (Blackwell sm_100+) tf32 split-K GEMM engine for the MHC
// projection:  X[M, K_DIM] @ W[N_OUT, K_DIM]^T -> mix[M, N_OUT], K_DIM huge,
// N_OUT tiny (24).  A one-SM tcgen05 tile avoids the 8-10x N padding a 2-SM
// swap-AB tile would need.  Activation X is bf16 (cast to tf32 on-chip), weight
// W is fp32 (read as tf32) -- matching the official DeepSeek-V4 hc precision.
// Split-K partials stay FP32 in a workspace; the caller fuses the reduction with
// RMSNorm/gates/Sinkhorn/collapse.
//
// This header owns ONLY the GEMM (tensor core issue + TMA staging + tile/split
// scheduling + launch); the epilogue lives in the .cu. clock64 profiling: pass a
// device int64 `prof` buffer (nullptr = disabled); block 0 stamps boundaries.
// ============================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100.hpp>       // SM100_TMEM_STORE_16dp256b1x (A cast -> TMEM)
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>   // SM100_MMA_TF32_TS (A from TMEM, B from smem)

namespace hc_tc {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// ---- problem + tile constants (shared with the epilogue) ----
static constexpr int HC = 4;
static constexpr int DIM = 7168;
static constexpr int K_DIM = HC * DIM;          // 28672
static constexpr int N_OUT = 24;
static constexpr int SINKHORN_ITERS = 20;

static constexpr int BLOCK_K = 64;
static constexpr int UMMA_K = 8;                 // tf32 MMA K step = 32 / sizeof(float)
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K;
// tf32 TS path (aligned to DeepGEMM sm100_tf32_hc_prenorm): A is cast bf16->tf32 into
// TMEM (double-buffered), C accumulator sits after the A cast tiles.
//   TMEM cols = BLOCK_K * kNumCastStages (A) + N_TILE (C), rounded up to a pow2.
//   BLOCK_K*2 = 128 for A; + N_TILE(<=32) -> 160 -> 256 aligned.  256/512 -> only
//   2 blocks/SM can hold TMEM (was 32 cols -> 16 blocks); occupancy cost of tf32.
static constexpr int kNumCastStages = 2;         // TMEM A double-buffer (DeepGEMM)
static constexpr int TMEM_C_OFFSET = BLOCK_K * kNumCastStages;   // 128: C after A tiles
static constexpr int NUM_TMEM_COLS = 256;        // aligned(128 + N_TILE<=32)
static constexpr int GEMM_THREADS = 256;
static constexpr int EPILOGUE_THREADS = 256;
// cast + Σx² group = warps 4..7 (128 threads): during the K-loop they cast the bf16 x
// tile to tf32 into TMEM for the MMA and accumulate the input RMSNorm Σx² as a
// by-product; after the K-loop they run the TMEM->workspace epilogue.
static constexpr int kNumReduceThreads = 128;
static constexpr int kReduceWarpBase = 4;   // first warp of the cast group

// clock64 profiling buffer layout (int64), filled on block 0 only:
//   [0]=gemm start   [1]=gemm end                       (GEMM kernel)
//   [2]=epi start    [3]=after load+rms  [4]=after reduce
//   [5]=after act    [6]=after sinkhorn   [7]=after collapse  (epilogue)
static constexpr int PROF_SLOTS = 8;

static_assert(K_DIM % BLOCK_K == 0, "K must be tiled exactly");

// ============================================================
// PTX helpers (tcgen05 alloc/mma/commit/load, warp reduce, clock64)
// ============================================================
namespace ptx {

__device__ __forceinline__ long long rdclock() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void tcgen05_alloc_1sm(uint32_t smem_addr, uint32_t cols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(cols));
}

__device__ __forceinline__ void tcgen05_dealloc_1sm(uint32_t tmem_addr, uint32_t cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_addr), "r"(cols));
}

__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// tf32 TS MMA (cute::SM100_MMA_TF32_TS): A operand is a TMEM address (%1 = [tmem_a]),
// B operand is an smem descriptor (%2 = desc_b). Mirrors the bf16 f16-kind helper but
// A comes from TMEM instead of an smem descriptor. Caller guards with elect_one_sync().
__device__ __forceinline__ void tcgen05_mma_tf32_ts_1sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint64_t runtime_idesc, uint32_t accumulate) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 "
        "[%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t}"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
           "r"(static_cast<uint32_t>(runtime_idesc >> 32)), "r"(accumulate),
           "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

__device__ __forceinline__ void tcgen05_commit_1sm(Barrier* barrier) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    if (elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"(addr) : "memory");
    }
}

__device__ __forceinline__ void tmem_load_32dp32b8x(
    uint32_t addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
    uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
          "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
        : "r"(addr));
}

}  // namespace ptx

// ============================================================
// SMEM descriptors + instruction descriptors
// ============================================================
namespace mma_desc {

// B (weight) is fp32 K-major with 128B swizzle. Since BLOCK_K*sizeof(float)=256B > 128,
// the K axis spans BLOCK_SWIZZLED_BK = 128/4 = 32 elems per swizzle atom -> 2 atoms per
// BLOCK_K=64. Descriptor + advance ported from DeepGEMM mma::sm100 (K-major branch).
static constexpr int BLOCK_SWIZZLED_BK = 128 / (int)sizeof(float);   // 32

__device__ __forceinline__ cute::UMMA::SmemDescriptor make_b_desc_fp32(void* ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_ = 1;
    desc.lbo_mode_ = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto smem_ptr = cute::cast_smem_ptr_to_uint(ptr);
    desc.start_address_ = static_cast<uint16_t>(smem_ptr >> 4);
    // stride_byte_offset = num_non_contiguous(8) * BLOCK_SWIZZLED_BK(32) * sizeof(float)
    constexpr uint32_t stride_bytes = 8u * BLOCK_SWIZZLED_BK * sizeof(float);   // 1024
    desc.stride_byte_offset_ = stride_bytes >> 4;
    desc.leading_byte_offset_ = 0;
    desc.base_offset_ = 0;
    return desc;
}

template <int M_TILE, int N_TILE>
__device__ __forceinline__ uint64_t make_runtime_idesc() {
    auto idesc = cute::UMMA::make_instr_desc<
        cutlass::tfloat32_t, cutlass::tfloat32_t, float,
        M_TILE, N_TILE, cute::UMMA::Major::K, cute::UMMA::Major::K>();
    return cute::UMMA::make_runtime_instr_desc(idesc);
}

// advance B descriptor low bits for the k-th UMMA_K step (fp32, K-major, 2 atoms/BLOCK_K).
// offset jumps to the next 128B swizzle atom (N_TILE * BLOCK_SWIZZLED_BK floats apart).
template <int N_TILE>
__device__ __forceinline__ uint32_t advance_b_k(uint32_t base_lo, int k) {
    const int atom_idx = (k * UMMA_K) / BLOCK_SWIZZLED_BK;
    const int in_atom_idx = (k * UMMA_K) % BLOCK_SWIZZLED_BK;
    const int offset = atom_idx * N_TILE * BLOCK_SWIZZLED_BK;
    return base_lo + static_cast<uint32_t>(((offset + in_atom_idx) * (int)sizeof(float)) >> 4);
}

}  // namespace mma_desc

// ============================================================
// Helpers for the Σx² reduce warp group (ported verbatim from DeepGEMM
// sm100_tf32_hc_prenorm_gemm: swizzled-smem read of the bf16 x tile + FP32
// fused-square accumulate + 4-lane warp reduction). Our config matches
// DeepGEMM's assumptions: BLOCK_M(tile)=64, BLOCK_K=64, kSwizzleAMode=128.
// ============================================================
// SW128 swizzled smem byte-offset (DeepGEMM impl file, get_swizzled_smem_offset).
template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
__device__ __forceinline__ uint32_t hc_swizzled_offset(uint32_t offset, uint32_t lane_idx) {
    const auto bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);
    return row * 128 + col * kSwizzleBase;
}

// ldmatrix.x4.m8n8.b16 (DeepGEMM ptx::SM90_U32x4_LDSM_N).
__device__ __forceinline__ void hc_ldmatrix_x4(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, const void* smem_src) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "l"(__cvta_generic_to_shared(smem_src)));
}

// FP32 fused multiply-add on float2 (DeepGEMM math::fma2).
__device__ __forceinline__ float2 hc_fma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);
#else
    return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
#endif
}

// Sum across each group of 4 consecutive lanes (DeepGEMM warp_reduce_sum<4>).
__device__ __forceinline__ float hc_warp_reduce_sum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

template <int M_TILE, int N_TILE, int STAGES>
struct GemmSharedStorage {
    static constexpr int A_STAGE_ELEMS = M_TILE * BLOCK_K;
    static constexpr int B_STAGE_ELEMS = N_TILE * BLOCK_K;

    // A stays bf16 in smem (TMA load); the cast group reads it, casts bf16->tf32, and
    // stores it into TMEM for the MMA. B (weight) is fp32 in smem, read as tf32 by the MMA.
    alignas(1024) __nv_bfloat16 smem_a[STAGES * A_STAGE_ELEMS];
    alignas(1024) float smem_b[STAGES * B_STAGE_ELEMS];
    // Producer/consumer barriers (DeepGEMM tf32 topology):
    //   TMA --full--> cast group --full_cast--> MMA --empty_cast--> cast (TMEM A reuse)
    //                                            MMA --empty------> TMA (smem reuse)
    // empty[stage] is arrived only by the MMA; since the MMA runs after full_cast (i.e.
    // after the cast group finished reading smem_a), that also releases smem_a safely.
    alignas(16) Barrier full[STAGES];                    // TMA -> cast (smem A/B ready)
    alignas(16) Barrier empty[STAGES];                   // MMA -> TMA (smem consumed)
    alignas(16) Barrier full_cast[kNumCastStages];       // cast -> MMA (A cast in TMEM)
    alignas(16) Barrier empty_cast[kNumCastStages];      // MMA -> cast (TMEM A slot free)
    alignas(16) Barrier tmem_full;                       // MMA -> epilogue (C ready)
    alignas(16) uint32_t tmem_base;
};

__device__ __forceinline__ void tma_load_2d(
    const void* desc, Barrier* barrier, void* smem,
    int coord0, int coord1) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc, reinterpret_cast<uint64_t*>(barrier),
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
        smem, coord0, coord1);
}

// ============================================================
// Split-K GEMM kernel: mix_partial[split, M, N_OUT] (FP32).
//   grid = num_m_tiles * num_n_tiles * num_splits;  block = GEMM_THREADS.
//   prof != nullptr: block 0 stamps prof[0]=start, prof[1]=end (clock64).
// ============================================================
template <int M_TILE, int N_TILE, int STAGES>
__global__ void __launch_bounds__(GEMM_THREADS, 1)
hc_gemm_splitk_kernel(
    const __grid_constant__ CUtensorMap desc_x,
    const __grid_constant__ CUtensorMap desc_w,
    int num_positions,
    int num_m_tiles,
    int num_n_tiles,
    int num_splits,
    int k_tiles_per_split,
    float* __restrict__ workspace,
    float* __restrict__ sqr_sum,   // [num_splits, M] Σx² partials (nullptr = skip write)
    int64_t* __restrict__ prof) {
    using Storage = GemmSharedStorage<M_TILE, N_TILE, STAGES>;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Storage& s = *reinterpret_cast<Storage*>(smem_raw);

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const bool prof_block = (prof != nullptr && blockIdx.x == 0);
    if (prof_block && threadIdx.x == 0) prof[0] = ptx::rdclock();

    int task = static_cast<int>(blockIdx.x);
    const int m_tile = task % num_m_tiles;
    task /= num_m_tiles;
    const int n_tile = task % num_n_tiles;
    const int split = task / num_n_tiles;
    if (split >= num_splits) return;

    const int k_begin = split * k_tiles_per_split;
    const int k_end = min(k_begin + k_tiles_per_split, NUM_K_TILES);
    const int k_count = k_end - k_begin;
    const int m_base = m_tile * M_TILE;
    const int n_base = n_tile * N_TILE;

    if (warp_id == 0 && ptx::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&desc_x);
        cute::prefetch_tma_descriptor(&desc_w);
    }
    if (warp_id == 1 && ptx::elect_one_sync()) {
        #pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
            s.full[stage].init(1);     // TMA producer arrives (1 elected thread)
            s.empty[stage].init(1);    // MMA arrives via umma_arrive
        }
        #pragma unroll
        for (int cs = 0; cs < kNumCastStages; ++cs) {
            s.full_cast[cs].init(kNumReduceThreads);  // all 128 cast threads arrive
            s.empty_cast[cs].init(1);                 // MMA arrives via umma_arrive
        }
        s.tmem_full.init(1);
        cutlass::arch::fence_barrier_init();
    }
    if (warp_id == 2) {
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_1sm(addr, NUM_TMEM_COLS);
    }
    __syncthreads();

    if (warp_id == 0 && ptx::elect_one_sync()) {
        int stage = 0;
        int phase = 0;
        #pragma unroll 1
        for (int kt = k_begin; kt < k_end; ++kt) {
            s.empty[stage].wait(phase ^ 1);          // MMA released the smem (covers cast too)
            auto* a_dst = s.smem_a + stage * Storage::A_STAGE_ELEMS;
            auto* b_dst = s.smem_b + stage * Storage::B_STAGE_ELEMS;
            const int k_offset = kt * BLOCK_K;
            // A bf16: BLOCK_K=64 = 128B = 1 swizzle atom -> single TMA load.
            tma_load_2d(&desc_x, &s.full[stage], a_dst, k_offset, m_base);
            // B fp32: BLOCK_K=64 = 256B = 2 swizzle atoms of 32 elems -> 2 TMA loads,
            // atom i into smem_b + i*(N_TILE*32) at K offset k+i*32 (DeepGEMM tma::copy).
            constexpr int kAtomK = mma_desc::BLOCK_SWIZZLED_BK;   // 32
            #pragma unroll
            for (int atom = 0; atom < BLOCK_K / kAtomK; ++atom) {
                tma_load_2d(&desc_w, &s.full[stage],
                            b_dst + atom * (N_TILE * kAtomK),
                            k_offset + atom * kAtomK, n_base);
            }
            constexpr uint32_t tx_bytes =
                Storage::A_STAGE_ELEMS * sizeof(__nv_bfloat16)   // A bf16
                + Storage::B_STAGE_ELEMS * sizeof(float);        // B fp32 (both atoms)
            s.full[stage].arrive_and_expect_tx(tx_bytes);
            if (++stage == STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }
    } else if (warp_id == 1) {
        // tf32 TS MMA: A operand read from TMEM (cast group filled it), B from smem (fp32).
        // A tile for stage's cast_stage sits at TMEM columns [cast_stage*BLOCK_K, +BLOCK_K);
        // C accumulator sits at TMEM_C_OFFSET (after the A cast tiles).
        const auto b_desc = mma_desc::make_b_desc_fp32(s.smem_b);
        const uint32_t b_stage_lo = (lane_id < STAGES)
            ? b_desc.lo + lane_id * static_cast<uint32_t>(Storage::B_STAGE_ELEMS * sizeof(float) / 16)
            : 0u;
        const uint64_t runtime_idesc = mma_desc::make_runtime_idesc<M_TILE, N_TILE>();
        const uint32_t tmem_c = s.tmem_base + TMEM_C_OFFSET;

        #pragma unroll 1
        for (int ki = 0; ki < k_count; ++ki) {
            const int stage_idx = ki % STAGES;
            const int cast_stage = ki % kNumCastStages;
            const int cast_phase = (ki / kNumCastStages) & 1;
            s.full_cast[cast_stage].wait(cast_phase);   // A cast in TMEM (implies B smem ready)
            ptx::tcgen05_fence_after_sync();
            const uint32_t b_base = __shfl_sync(0xffffffffu, b_stage_lo, stage_idx);
            const uint32_t a_tmem_stage = s.tmem_base + cast_stage * BLOCK_K;
            if (ptx::elect_one_sync()) {
                #pragma unroll
                for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                    const uint32_t a_tmem = a_tmem_stage + kk * UMMA_K;
                    const uint32_t b_lo = mma_desc::advance_b_k<N_TILE>(b_base, kk);
                    const uint64_t b = (static_cast<uint64_t>(b_desc.hi) << 32) | b_lo;
                    const uint32_t accumulate = (ki != 0 || kk != 0) ? 1u : 0u;
                    ptx::tcgen05_mma_tf32_ts_1sm(tmem_c, a_tmem, b, runtime_idesc, accumulate);
                }
            }
            __syncwarp();
            ptx::tcgen05_commit_1sm(&s.empty_cast[cast_stage]);  // release TMEM A slot -> cast
            ptx::tcgen05_commit_1sm(&s.empty[stage_idx]);        // release smem -> TMA
            if (ki == k_count - 1) {
                ptx::tcgen05_commit_1sm(&s.tmem_full);           // C ready -> epilogue
            }
            __syncwarp();
        }
    } else if (warp_id >= 4) {
        // ===== cast + Σx² group (warps 4-7), concurrent with MMA (DeepGEMM tf32) =====
        // Read the bf16 x tile the TMA staged, cast bf16->tf32 and store it into TMEM for
        // the MMA (SM100_TMEM_STORE_16dp256b1x), AND accumulate the input RMSNorm Σx² as a
        // by-product. Ported verbatim from DeepGEMM sm100_tf32_hc_prenorm else-branch.
        // Config matches DeepGEMM: BLOCK_M=64, BLOCK_K=64, SW128, 4 sub-warps x 16 rows.
        {
            const int sub_warp_idx = warp_id - kReduceWarpBase;   // 0..3
            constexpr int BLOCK_M_PER_WARP = M_TILE / 4;          // 16
            constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(__nv_bfloat16);  // 128
            constexpr int kNumElemsPerBankGroup = 16 / (int)sizeof(__nv_bfloat16);  // 8
            constexpr int kNumLoads = BLOCK_K / kNumElemsPerBankGroup;              // 8
            float2 sqacc[2] = { {0.f, 0.f}, {0.f, 0.f} };
            #pragma unroll 1
            for (int ki = 0; ki < k_count; ++ki) {
                const int stage_idx = ki % STAGES;
                const int cast_stage = ki % kNumCastStages;
                const int cast_phase = (ki / kNumCastStages) & 1;
                s.full[stage_idx].wait((ki / STAGES) & 1);   // smem A ready from TMA
                const uint8_t* smem_base =
                    reinterpret_cast<const uint8_t*>(s.smem_a + stage_idx * Storage::A_STAGE_ELEMS)
                    + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;
                uint32_t uv[2][kNumLoads];
                #pragma unroll
                for (int i = 0; i < kNumLoads; i += 2) {
                    const void* sp = smem_base +
                        hc_swizzled_offset<kSwizzleAMode>(i + lane_id / 16, lane_id % 16);
                    hc_ldmatrix_x4(uv[0][i], uv[1][i], uv[0][i + 1], uv[1][i + 1], sp);
                }
                // Wait until the MMA has consumed this TMEM A slot before overwriting it.
                s.empty_cast[cast_stage].wait(cast_phase ^ 1);
                // Cast bf16->tf32(fp32 storage), accumulate Σx², store into TMEM.
                float2 fv[2][kNumLoads];
                uint32_t* upper = reinterpret_cast<uint32_t*>(&fv[0]);
                uint32_t* lower = reinterpret_cast<uint32_t*>(&fv[1]);
                const uint32_t a_tmem_col = s.tmem_base + cast_stage * BLOCK_K;
                #pragma unroll
                for (int i = 0; i < kNumLoads; ++i) {
                    #pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        fv[u][i] = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&uv[u][i]));
                        sqacc[u] = hc_fma2(fv[u][i], fv[u][i], sqacc[u]);
                    }
                    const int idx0 = i * 2, idx1 = i * 2 + 1;
                    cute::SM100::TMEM::STORE::SM100_TMEM_STORE_16dp256b1x::copy(
                        upper[idx0], upper[idx1], lower[idx0], lower[idx1],
                        a_tmem_col + i * 8);
                }
                cutlass::arch::fence_view_async_tmem_store();
                ptx::tcgen05_fence_before_sync();
                s.full_cast[cast_stage].arrive();   // A cast in TMEM -> MMA (all 128 threads)
            }
            // 4 lanes share a row -> reduce, then write sqr_sum (only n_tile==0).
            if (sqr_sum != nullptr && n_tile == 0) {
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const float r = hc_warp_reduce_sum4(sqacc[u].x + sqacc[u].y);
                    const int m_idx = m_base + sub_warp_idx * BLOCK_M_PER_WARP + lane_id / 4 + u * 8;
                    if ((lane_id & 3) == 0 && m_idx < num_positions)
                        sqr_sum[static_cast<int64_t>(split) * num_positions + m_idx] = r;
                }
            }
        }
        // ============ epilogue: TMEM C (mix) -> workspace ============
        // M_TILE=64 Layout F: four 16-row groups (one per epilogue warp), 16 lanes each.
        const int epi_warp = warp_id - 4;
        const int row = m_base + epi_warp * 16 + lane_id;
        s.tmem_full.wait(0);
        ptx::tcgen05_fence_after_sync();

        #pragma unroll
        for (int ng = 0; ng < N_TILE / 8; ++ng) {
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
            // C accumulator lives after the A cast tiles in TMEM (TMEM_C_OFFSET).
            ptx::tmem_load_32dp32b8x(
                s.tmem_base + TMEM_C_OFFSET + ng * 8, v0, v1, v2, v3, v4, v5, v6, v7);
            cutlass::arch::fence_view_async_tmem_load();
            if (lane_id < 16 && row < num_positions) {
                const uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                const int col0 = n_base + ng * 8;
                float* dst = workspace
                    + (static_cast<int64_t>(split) * num_positions + row) * N_OUT + col0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (col0 + j < N_OUT) dst[j] = __uint_as_float(vals[j]);
                }
            }
        }
        ptx::tcgen05_fence_before_sync();
        if (prof_block && epi_warp == 0 && lane_id == 0) prof[1] = ptx::rdclock();
    }

    __syncthreads();
    if (warp_id == 2) {
        ptx::tcgen05_dealloc_1sm(s.tmem_base, NUM_TMEM_COLS);
    }
}

// ============================================================
// Host: split/tile scheduling + TMA descriptors + launch
// ============================================================
struct SplitConfig {
    int m_tile;
    int n_tile;
    int num_m_tiles;
    int num_n_tiles;
    int num_splits;
    int k_tiles_per_split;
    int grid;
    int num_sms;
};

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

inline int choose_n_tile(int m) {
    // M<=128: N_TILE=8 -> N_OUT=24 tiles into 3 (zero padding); with num_m_tiles<=2 the
    // 3 n-tiles give the grid enough blocks. M in (128,256]: N_TILE=32 (one n-tile) --
    // NT=8 gets slow at M=160..224 (grid balloons, 2 blocks/SM by TMEM -> multi-wave).
    if (m <= 128) return 8;
    return 32;
}

inline int choose_m_tile(int) {
    // Fixed at 64: the Σx² reduce warp group's LDSM/row layout (ported from
    // DeepGEMM hc_prenorm, which also pins block_m=64) is 64-row-specific.
    return 64;
}

inline int cached_num_sms() {
    // Queried once (device count is fixed). Avoids a cudaGetDevice +
    // cudaDeviceGetAttribute on every forward, which is pure host overhead in
    // the small-M decode regime.
    static const int n = [] {
        int device = 0, num_sms = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
        return num_sms;
    }();
    return n;
}

inline SplitConfig make_split_config(int m) {
    const int num_sms = cached_num_sms();
    TORCH_CHECK(num_sms > 0, "cached_num_sms() returned ", num_sms);

    const int m_tile = choose_m_tile(m);
    const int n_tile = choose_n_tile(m);
    const int num_m_tiles = ceil_div(m, m_tile);
    const int num_n_tiles = ceil_div(N_OUT, n_tile);
    const int mn_tiles = num_m_tiles * num_n_tiles;

    // Two fixed configs, tuned for the decode range M<=256 (M>256 is out of scope):
    //   M<=128 : NT=8,  splitK=18  (3 n-tiles carry grid parallelism when M is tiny)
    //   M<=256 : NT=32, splitK=35  (1 n-tile; more splits was slower -- bigger reduce)
    const int target_splits = (m <= 128) ? 18 : 35;
    const int k_tiles_per_split = ceil_div(NUM_K_TILES, target_splits);
    const int num_splits = ceil_div(NUM_K_TILES, k_tiles_per_split);
    return {m_tile, n_tile, num_m_tiles, num_n_tiles, num_splits,
            k_tiles_per_split, mn_tiles * num_splits, num_sms};
}

inline CUtensorMap make_tma_bf16_2d(
    const char* name,
    const __nv_bfloat16* ptr,
    int rows,
    int cols,
    int box_rows) {
    CUtensorMap desc{};
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
    cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(cols) * sizeof(__nv_bfloat16)};
    cuuint32_t box_dims[2] = {
        static_cast<cuuint32_t>(BLOCK_K), static_cast<cuuint32_t>(box_rows)};
    cuuint32_t elem_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16*>(ptr), global_dims, global_strides,
        box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    message ? message : "unknown", " rows=", rows,
                    " cols=", cols, " box_rows=", box_rows);
    }
    return desc;
}

// FP32 weight TMA (tf32 path): B is loaded fp32 and read as tf32 by the MMA.
// Swizzle stays 128B (kSwizzleBMode = min(BLOCK_K*4,128) = 128); a BLOCK_K=64 fp32 row
// (256B) spans 2 swizzle atoms, handled by make_b_desc_fp32 / advance_b_k in the kernel.
inline CUtensorMap make_tma_fp32_2d(
    const char* name,
    const float* ptr,
    int rows,
    int cols,
    int box_rows) {
    CUtensorMap desc{};
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
    cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(cols) * sizeof(float)};
    // SW128 with fp32: the box inner dim must be one swizzle atom = 128B / 4 = 32 elems
    // (BLOCK_K=64 fp32 = 256B spans 2 atoms). The producer issues 2 TMA loads per K-tile.
    constexpr int kInnerAtom = 128 / (int)sizeof(float);   // 32
    cuuint32_t box_dims[2] = {
        static_cast<cuuint32_t>(kInnerAtom), static_cast<cuuint32_t>(box_rows)};
    cuuint32_t elem_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
        const_cast<float*>(ptr), global_dims, global_strides,
        box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    message ? message : "unknown", " rows=", rows,
                    " cols=", cols, " box_rows=", box_rows);
    }
    return desc;
}

struct TmaCache {
    const void* x_ptr = nullptr;
    const void* w_ptr = nullptr;
    int m = -1;
    int m_tile = -1;
    int n_tile = -1;
    CUtensorMap x{};
    CUtensorMap w{};
};

template <int M_TILE, int N_TILE, int STAGES>
inline void launch_gemm(
    const CUtensorMap& desc_x,
    const CUtensorMap& desc_w,
    const SplitConfig& cfg,
    int m,
    float* workspace,
    float* sqr_sum,
    int64_t* prof,
    cudaStream_t stream) {
    using Storage = GemmSharedStorage<M_TILE, N_TILE, STAGES>;
    void* kernel = reinterpret_cast<void*>(&hc_gemm_splitk_kernel<M_TILE, N_TILE, STAGES>);
    const int smem_bytes = static_cast<int>(sizeof(Storage));

    static bool configured = false;
    if (!configured) {
        const auto attr_err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess,
                    "cudaFuncSetAttribute(gemm) failed: ", cudaGetErrorString(attr_err),
                    " smem=", smem_bytes, " tile=", M_TILE, "x", N_TILE,
                    " stages=", STAGES);
        configured = true;
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(cfg.grid, 1, 1);
    config.blockDim = dim3(GEMM_THREADS, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    void* args[] = {
        const_cast<CUtensorMap*>(&desc_x), const_cast<CUtensorMap*>(&desc_w),
        &m, const_cast<int*>(&cfg.num_m_tiles), const_cast<int*>(&cfg.num_n_tiles),
        const_cast<int*>(&cfg.num_splits), const_cast<int*>(&cfg.k_tiles_per_split),
        &workspace, &sqr_sum, &prof};
    const auto launch_err = cudaLaunchKernelExC(&config, kernel, args);
    TORCH_CHECK(launch_err == cudaSuccess,
                "hc tcgen05 GEMM launch failed: ", cudaGetErrorString(launch_err));
}

// Dispatch launch_gemm on the runtime (M_TILE, N_TILE, STAGES) chosen by cfg.
inline void launch_gemm_dispatch(
    const CUtensorMap& desc_x, const CUtensorMap& desc_w,
    const SplitConfig& cfg, int m, float* workspace, float* sqr_sum, int64_t* prof,
    cudaStream_t stream) {
    // Only two configs (M_TILE fixed at 64; the cast group's 4-sub-warp x 16-row layout
    // is 64-specific): NT=8/STAGES=4 for M<=128, NT=32/STAGES=4 for M in (128,256].
    TORCH_CHECK(cfg.m_tile == 64, "M tile must be 64");
    switch (cfg.n_tile) {
        case 8:
            launch_gemm<64, 8, 4>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
            break;
        case 32:
            launch_gemm<64, 32, 4>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
            break;
        default:
            TORCH_CHECK(false, "unsupported N tile: ", cfg.n_tile);
    }
}

}  // namespace hc_tc
