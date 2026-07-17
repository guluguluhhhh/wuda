#pragma once
// ============================================================
// hc_fused_kernel_tc.cuh
// Reusable tcgen05 (Blackwell sm_100+) BF16 split-K GEMM engine for the MHC
// projection:  X[M, K_DIM] @ W[N_OUT, K_DIM]^T -> mix[M, N_OUT], K_DIM huge,
// N_OUT tiny (24).  A one-SM tcgen05 tile avoids the 8-10x N padding a 2-SM
// swap-AB tile would need.  Split-K partials stay FP32 in a workspace; the
// caller fuses the reduction with RMSNorm/gates/Sinkhorn/collapse.
//
// Layout mirror of cluster_mma*.cuh: this header owns ONLY the GEMM (tensor
// core issue + TMA staging + tile/split scheduling + launch).  The epilogue
// lives in the .cu.  clock64 profiling follows wq_b_fp8_gemm.cu: pass a device
// int64 `prof` buffer (nullptr = disabled); block 0 stamps phase boundaries.
// ============================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

namespace hc_tc {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// ---- problem + tile constants (shared with the epilogue) ----
static constexpr int HC = 4;
static constexpr int DIM = 7168;
static constexpr int K_DIM = HC * DIM;          // 28672
static constexpr int N_OUT = 24;
static constexpr int SINKHORN_ITERS = 20;

static constexpr int BLOCK_K = 64;
static constexpr int UMMA_K = 16;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K;
static constexpr int NUM_TMEM_COLS = 32;
static constexpr int GEMM_THREADS = 256;
static constexpr int EPILOGUE_THREADS = 256;
// Σx² reduce group = warps 4..7 of the GEMM block (128 threads). They are idle
// during the K-loop (they only do the TMEM->workspace epilogue afterwards), so we
// reuse them to accumulate the input RMSNorm sum-of-squares while the MMA runs.
static constexpr int kNumReduceThreads = 128;
static constexpr int kReduceWarpBase = 4;   // first warp of the reduce group
static constexpr int MIN_K_TILES_PER_SPLIT = 8;

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

__device__ __forceinline__ void tcgen05_mma_1sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint64_t runtime_idesc, uint32_t accumulate) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
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

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace ptx

// ============================================================
// SMEM descriptors + instruction descriptors
// ============================================================
namespace mma_desc {

__device__ __forceinline__ cute::UMMA::SmemDescriptor make_k_major(void* ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_ = 1;
    desc.lbo_mode_ = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto smem_ptr = cute::cast_smem_ptr_to_uint(ptr);
    desc.start_address_ = static_cast<uint16_t>(smem_ptr >> 4);
    constexpr uint32_t stride_bytes = 8u * BLOCK_K * sizeof(__nv_bfloat16);
    desc.stride_byte_offset_ = stride_bytes >> 4;
    desc.leading_byte_offset_ = 0;
    desc.base_offset_ = 0;
    return desc;
}

template <int M_TILE, int N_TILE>
__device__ __forceinline__ uint64_t make_runtime_idesc() {
    auto idesc = cute::UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        M_TILE, N_TILE, cute::UMMA::Major::K, cute::UMMA::Major::K>();
    return cute::UMMA::make_runtime_instr_desc(idesc);
}

__device__ __forceinline__ uint32_t advance_k(uint32_t lo, int kk) {
    return lo + kk * (UMMA_K * sizeof(__nv_bfloat16) / 16);
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

    alignas(1024) __nv_bfloat16 smem_a[STAGES * A_STAGE_ELEMS];
    alignas(1024) __nv_bfloat16 smem_b[STAGES * B_STAGE_ELEMS];
    alignas(16) Barrier full[STAGES];
    alignas(16) Barrier empty[STAGES];        // MMA -> TMA (smem consumed by MMA)
    alignas(16) Barrier empty_reduce[STAGES]; // Σx² reduce group -> TMA (smem consumed by reduce)
    alignas(16) Barrier tmem_full;
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
            s.full[stage].init(1);
            s.empty[stage].init(1);
            s.empty_reduce[stage].init(kNumReduceThreads);  // all reduce threads arrive
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
            s.empty[stage].wait(phase ^ 1);          // MMA released the smem
            s.empty_reduce[stage].wait(phase ^ 1);   // Σx² reduce group released the smem
            auto* a_dst = s.smem_a + stage * Storage::A_STAGE_ELEMS;
            auto* b_dst = s.smem_b + stage * Storage::B_STAGE_ELEMS;
            const int k_offset = kt * BLOCK_K;
            tma_load_2d(&desc_x, &s.full[stage], a_dst, k_offset, m_base);
            tma_load_2d(&desc_w, &s.full[stage], b_dst, k_offset, n_base);
            constexpr uint32_t tx_bytes =
                (Storage::A_STAGE_ELEMS + Storage::B_STAGE_ELEMS) * sizeof(__nv_bfloat16);
            s.full[stage].arrive_and_expect_tx(tx_bytes);
            if (++stage == STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }
    } else if (warp_id == 1) {
        const auto a_desc = mma_desc::make_k_major(s.smem_a);
        const auto b_desc = mma_desc::make_k_major(s.smem_b);
        const uint32_t a_stage_lo = (lane_id < STAGES)
            ? a_desc.lo + lane_id * (Storage::A_STAGE_ELEMS * sizeof(__nv_bfloat16) / 16)
            : 0u;
        const uint32_t b_stage_lo = (lane_id < STAGES)
            ? b_desc.lo + lane_id * (Storage::B_STAGE_ELEMS * sizeof(__nv_bfloat16) / 16)
            : 0u;
        const uint64_t runtime_idesc = mma_desc::make_runtime_idesc<M_TILE, N_TILE>();
        const uint32_t tmem_c = s.tmem_base;

        int stage = 0;
        int phase = 0;
        #pragma unroll 1
        for (int ki = 0; ki < k_count; ++ki) {
            s.full[stage].wait(phase);
            ptx::tcgen05_fence_after_sync();
            const uint32_t a_base = __shfl_sync(0xffffffffu, a_stage_lo, stage);
            const uint32_t b_base = __shfl_sync(0xffffffffu, b_stage_lo, stage);
            if (ptx::elect_one_sync()) {
                #pragma unroll
                for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                    const uint32_t a_lo = mma_desc::advance_k(a_base, kk);
                    const uint32_t b_lo = mma_desc::advance_k(b_base, kk);
                    const uint64_t a = (static_cast<uint64_t>(a_desc.hi) << 32) | a_lo;
                    const uint64_t b = (static_cast<uint64_t>(b_desc.hi) << 32) | b_lo;
                    const uint32_t accumulate = (ki != 0 || kk != 0) ? 1u : 0u;
                    ptx::tcgen05_mma_1sm(tmem_c, a, b, runtime_idesc, accumulate);
                }
            }
            __syncwarp();
            ptx::tcgen05_commit_1sm(&s.empty[stage]);
            if (ki == k_count - 1) {
                ptx::tcgen05_commit_1sm(&s.tmem_full);
            }
            __syncwarp();
            if (++stage == STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }
    } else if (warp_id >= 4) {
        // ============ Σx² reduce group (warps 4-7), concurrent with MMA ============
        // Accumulate the input RMSNorm sum-of-squares from the SAME x tiles the MMA
        // consumes (no extra HBM read), write sqr_sum[split, m]. Ported from DeepGEMM
        // hc_prenorm else-branch, minus the TF32 cast-to-TMEM (our MMA reads bf16 x
        // from smem directly). Config matches DeepGEMM: BLOCK_M=64, BLOCK_K=64, SW128.
        {
            const int sub_warp_idx = warp_id - kReduceWarpBase;   // 0..3
            constexpr int BLOCK_M_PER_WARP = M_TILE / 4;          // 16
            constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(__nv_bfloat16);  // 128
            constexpr int kNumElemsPerBankGroup = 16 / (int)sizeof(__nv_bfloat16);  // 8
            constexpr int kNumLoads = BLOCK_K / kNumElemsPerBankGroup;              // 8
            float2 sqacc[2] = { {0.f, 0.f}, {0.f, 0.f} };
            int rstage = 0, rphase = 0;
            #pragma unroll 1
            for (int ki = 0; ki < k_count; ++ki) {
                s.full[rstage].wait(rphase);
                const uint8_t* smem_base =
                    reinterpret_cast<const uint8_t*>(s.smem_a + rstage * Storage::A_STAGE_ELEMS)
                    + sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;
                uint32_t uv[2][kNumLoads];
                #pragma unroll
                for (int i = 0; i < kNumLoads; i += 2) {
                    const void* sp = smem_base +
                        hc_swizzled_offset<kSwizzleAMode>(i + lane_id / 16, lane_id % 16);
                    hc_ldmatrix_x4(uv[0][i], uv[1][i], uv[0][i + 1], uv[1][i + 1], sp);
                }
                #pragma unroll
                for (int i = 0; i < kNumLoads; ++i) {
                    #pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        float2 fv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&uv[u][i]));
                        sqacc[u] = hc_fma2(fv, fv, sqacc[u]);
                    }
                }
                s.empty_reduce[rstage].arrive();
                if (++rstage == STAGES) { rstage = 0; rphase ^= 1; }
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
        // ============ epilogue: TMEM mix -> workspace (existing) ===================
        const int epi_warp = warp_id - 4;
        // M64 Layout F has four 16-row groups at DP 0/32/64/96; M128
        // Layout D uses all 32 lanes of each datapath partition.
        const int rows_per_warp = M_TILE == 64 ? 16 : 32;
        const int row_local = epi_warp * rows_per_warp + lane_id;
        const int row = m_base + row_local;
        s.tmem_full.wait(0);
        ptx::tcgen05_fence_after_sync();

        #pragma unroll
        for (int ng = 0; ng < N_TILE / 8; ++ng) {
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
            ptx::tmem_load_32dp32b8x(
                s.tmem_base + ng * 8, v0, v1, v2, v3, v4, v5, v6, v7);
            cutlass::arch::fence_view_async_tmem_load();
            if ((M_TILE == 128 || lane_id < 16) && row < num_positions) {
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
    // cuBLAS (nvjet) uses tile 8x64 for M<256 -> N_OUT-axis tile = 8. Align.
    if (m < 256) return 8;
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

    int k_tiles_per_split, num_splits;
    if (m < 256) {
        // cuBLAS alignment (M<256): numSplitsK = 18 (from cublasLt heuristic on
        // this shape). Fewer splits than our SM-fill default (~56) -> 3x smaller
        // split-K workspace -> much cheaper reduce; parallelism kept via the 3
        // N_OUT-tiles (n_tile=8).
        constexpr int CUBLAS_SPLITS = 18;
        k_tiles_per_split = ceil_div(NUM_K_TILES, CUBLAS_SPLITS);
        num_splits = ceil_div(NUM_K_TILES, k_tiles_per_split);
    } else {
        const int target_splits = std::max(ceil_div(num_sms, mn_tiles), 1);
        k_tiles_per_split = std::max(
            ceil_div(NUM_K_TILES, target_splits), MIN_K_TILES_PER_SPLIT);
        num_splits = ceil_div(NUM_K_TILES, k_tiles_per_split);
    }
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
    if (cfg.m_tile == 128) {
        TORCH_CHECK(cfg.n_tile == 32, "M128 path requires N32");
        launch_gemm<128, 32, 8>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
        return;
    }
    switch (cfg.n_tile) {
        case 8:
            launch_gemm<64, 8, 4>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
            break;
        case 16:
            launch_gemm<64, 16, 4>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
            break;
        case 32:
            if (cfg.k_tiles_per_split <= 16) {
                launch_gemm<64, 32, 4>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
            } else {
                launch_gemm<64, 32, 12>(desc_x, desc_w, cfg, m, workspace, sqr_sum, prof, stream);
            }
            break;
        default:
            TORCH_CHECK(false, "unsupported N tile: ", cfg.n_tile);
    }
}

}  // namespace hc_tc
