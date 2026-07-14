// ============================================================
// w1_merged_fp8_gemm.cu
// tcgen05 FP8 (E4M3) block-scale GEMM — Kernel + Host + PyTorch binding.
// Faithful hand-written port of DeepGEMM sm100_fp8_fp4_gemm_1d1d_impl
// (2SM, swap-AB, block-scale) for the FIXED shape:
//   x_fp8[M,7168] @ w1_fp8[4352,7168].T -> y_all_bf16[M,4352]
//   M in {32,64,96,128,160,192,224,256}, K=7168, N=4352, BLOCK_K=128, gran_k=32.
//
// Geometry (from DeepGEMM scheduler get_swizzled_block_idx, kIsMulticastOnA):
//   A cluster's 2 CTAs share m_block and take ADJACENT n_blocks (n, n+1).
//   Each CTA loads its own 128-N weight + SFB, its own activation half
//   (m_idx += rank*LOAD_BLOCK_M) + full-BLOCK_M SFA. The cta_group::2 MMA
//   combines the pair into UMMA_M=256 (= 2 n_blocks) x UMMA_N=BLOCK_M.
//   Leader issues one 2cta-UTCCP; each SM copies its local smem SF into its
//   local TMEM. Each CTA's epilogue stores its own n_block.
//
// NOTE: untested on this host (no Blackwell). Validate on sm_100+.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

#include <cute/arch/copy_sm100.hpp>   // SM100_UTCCP_4x32dp128bit_2cta

#include "w1_merged_fp8_gemm.cuh"

using namespace w1fp8;
using Barrier = w1_mma_desc::Barrier;

// ======================== Shared Memory Layout ========================
template <int M_TPL>
struct SharedStorage {
    using D = SwapDims<M_TPL>;
    static constexpr int NS   = D::NUM_STAGES;
    static constexpr int SA   = D::SMEM_A_PER_STAGE;
    static constexpr int SSFA = D::SMEM_SFA_PER_STAGE;

    alignas(1024) uint8_t smem_cd[SMEM_CD_TOTAL];
    alignas(1024) uint8_t smem_a[NS * SA];
    alignas(1024) uint8_t smem_b[NS * SMEM_B_PER_STAGE];
    alignas(128)  uint8_t smem_sfa[NS * SSFA];
    alignas(128)  uint8_t smem_sfb[NS * SMEM_SFB_PER_STAGE];

    alignas(16) Barrier full_barriers[NS];
    alignas(16) Barrier empty_barriers[NS];
    alignas(16) Barrier with_sf_full_barriers[NS];
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES];

    alignas(16) uint32_t tmem_base;
};

// UTCCP (2cta): copy one 128-wide SF atom from smem descriptor into TMEM column.
__device__ __forceinline__ void utccp_4x32_2cta(uint32_t tmem_col, uint64_t sf_desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(tmem_col), "l"(sf_desc) : "memory");
}
__device__ __forceinline__ uint64_t sf_desc_bits(const cute::UMMA::SmemDescriptor& d) {
    return *reinterpret_cast<const uint64_t*>(&d);
}

// ======================== Kernel ========================
template <int M_TPL>
__global__ void __launch_bounds__(TPB, 1)
w1_merged_fp8_kernel(
    const __grid_constant__ CUtensorMap desc_A,    // activation [M,K] K-major fp8
    const __grid_constant__ CUtensorMap desc_B,    // weight     [N,K] K-major fp8
    const __grid_constant__ CUtensorMap desc_SFA,  // x_sf   (MN-major uint32)
    const __grid_constant__ CUtensorMap desc_SFB,  // w1_sf  (MN-major uint32)
    const __grid_constant__ CUtensorMap desc_D,    // output [M,N] BF16 row-major
    int shape_m, int num_sms, int num_1d_blocks_per_group,
    int num_k_splits, int k_tiles_per_split)
{
    using Dims = SwapDims<M_TPL>;
    constexpr int NS            = Dims::NUM_STAGES;
    constexpr int SA            = Dims::SMEM_A_PER_STAGE;
    constexpr int SSFA          = Dims::SMEM_SFA_PER_STAGE;
    constexpr int LOAD_BLOCK_M  = Dims::LOAD_BLOCK_M;   // M/2
    constexpr int UMMA_N        = Dims::UMMA_N;         // M
    constexpr int NUM_TMEM_COLS = Dims::NUM_TMEM_COLS;
    constexpr int SF_BLOCK_M    = Dims::SF_BLOCK_M;
    constexpr int TMEM_SFA      = Dims::TMEM_START_SFA;
    constexpr int TMEM_SFB      = Dims::TMEM_START_SFB;
    constexpr uint16_t CTA_MASK = (1 << NUM_MULTICAST) - 1;

    extern __shared__ __align__(1024) uint8_t smem_buf[];
    auto& s = *reinterpret_cast<SharedStorage<M_TPL>*>(smem_buf);

    const uint32_t warp_id  = threadIdx.x / 32;
    const uint32_t lane_id  = w1ptx::get_lane_idx();
    const uint32_t cta_rank = w1ptx::block_rank_in_cluster();
    const bool     is_leader = (cta_rank == 0);

    // ---- Init ----
    w1ptx::cluster_sync();
    if (warp_id == 0) {
        cute::prefetch_tma_descriptor(&desc_A);
        cute::prefetch_tma_descriptor(&desc_B);
        cute::prefetch_tma_descriptor(&desc_SFA);
        cute::prefetch_tma_descriptor(&desc_SFB);
        cute::prefetch_tma_descriptor(&desc_D);
    }
    if (warp_id == 1 && w1ptx::elect_one_sync()) {
        for (int i = 0; i < NS; ++i) {
            s.full_barriers[i].init(1);
            s.empty_barriers[i].init(1);
            s.with_sf_full_barriers[i].init(NUM_MULTICAST * 32);
        }
        for (int i = 0; i < NUM_EPI_STAGES; ++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(NUM_MULTICAST * NUM_STORE_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_id == 2) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        w1ptx::tcgen05_alloc_2sm(addr, NUM_TMEM_COLS);
    }
    w1ptx::cluster_sync();
    cudaGridDependencySynchronize();

    // ---- Scheduler state (Normal GEMM, grouping on N) ----
    const int num_m_blocks = (shape_m + Dims::BLOCK_M - 1) / Dims::BLOCK_M;
    const int num_n_blocks = N_TOTAL / BLOCK_N;              // 34
    const int num_mn_blocks = num_m_blocks * num_n_blocks;
    const int num_blocks   = num_mn_blocks * num_k_splits;

    auto get_block = [&](int iter, uint32_t& m_block, uint32_t& n_block, uint32_t& k_split) -> bool {
        const int block_idx = iter * num_sms + blockIdx.x;
        if (block_idx >= num_blocks) return false;
        k_split = block_idx / num_mn_blocks;
        const int mn_block_idx = block_idx - k_split * num_mn_blocks;
        // get_swizzled_block_idx, kIsMulticastOnA => grouping on N
        const int primary   = num_n_blocks;   // secondary axis blocks group
        const int secondary = num_m_blocks;
        const int per_group = secondary * num_1d_blocks_per_group;
        const int group_idx = mn_block_idx / per_group;
        const int first     = group_idx * num_1d_blocks_per_group;
        const int in_group  = mn_block_idx % per_group;
        int nbig = num_1d_blocks_per_group;
        if (primary - first < nbig) nbig = primary - first;
        m_block = in_group / nbig;
        n_block = first + in_group % nbig;
        return true;
    };

    // ---- Pipeline advance ----
    uint32_t stage = 0, phase = 0;
    auto advance = [&]() { stage = (stage + 1) % NS; if (stage == 0) phase ^= 1; };

    // ======== WARP 0: TMA producer (both CTAs, plain per-CTA loads) ========
    if (warp_id == 0 && w1ptx::elect_one_sync()) {
        int iter = 0; uint32_t m_block, n_block, k_split;
        while (get_block(iter, m_block, n_block, k_split)) {
            const uint32_t m_idx = m_block * Dims::BLOCK_M + cta_rank * LOAD_BLOCK_M; // activation half
            const uint32_t n_idx = n_block * BLOCK_N;                                  // this CTA's 128 N
            const uint32_t sfa_m = m_block * Dims::BLOCK_M;                            // full BLOCK_M SFA
            const uint32_t sfb_n = n_block * BLOCK_N;
            const int k_begin = k_split * k_tiles_per_split;
            const int k_end = (k_begin + k_tiles_per_split < NUM_K_TILES) ?
                              (k_begin + k_tiles_per_split) : NUM_K_TILES;

            for (int k = k_begin; k < k_end; ++k) {
                s.empty_barriers[stage].wait(phase ^ 1);
                const uint32_t k_idx = k * BLOCK_K;

                auto* sa = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a + stage * SA);
                auto* sb = reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b + stage * SMEM_B_PER_STAGE);
                w1tma::copy_2d_fp8(&desc_A, &s.full_barriers[stage], sa, k_idx, m_idx);
                w1tma::copy_2d_fp8(&desc_B, &s.full_barriers[stage], sb, k_idx, n_idx);

                uint32_t bytes = SA + SMEM_B_PER_STAGE;

                // gran_k=32 => one SF TMA per BLOCK_K (kNumSFStagesPerLoad = 1)
                auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage * SSFA);
                auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage * SMEM_SFB_PER_STAGE);
                w1tma::copy_2d_sf(&desc_SFA, &s.full_barriers[stage], sfa, sfa_m, (uint32_t)k);
                w1tma::copy_2d_sf(&desc_SFB, &s.full_barriers[stage], sfb, sfb_n, (uint32_t)k);
                bytes += Dims::BLOCK_M * sizeof(uint32_t) + BLOCK_N * sizeof(uint32_t);

                s.full_barriers[stage].arrive_and_expect_tx(bytes);
                advance();
            }
            ++iter;
        }
    }

    // ======== WARP 2: UTCCP transposer (both CTAs) ========
    else if (warp_id == 2) {
        auto warp_transpose = [&](uint32_t* smem_ptr) {
            // read [4 x 32], write transposed [32 x 4] (DeepGEMM utccp_required_smem_warp_transpose)
            uint32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = w1ptx::ld_shared_u32(smem_ptr + i * 32 + lane_id);
            __syncwarp();
            w1ptx::st_shared_v4_u32(smem_ptr + lane_id * 4, v[0], v[1], v[2], v[3]);
        };
        int iter = 0; uint32_t m_block, n_block, k_split;
        while (get_block(iter, m_block, n_block, k_split)) {
            const int k_begin = k_split * k_tiles_per_split;
            const int k_end = (k_begin + k_tiles_per_split < NUM_K_TILES) ?
                              (k_begin + k_tiles_per_split) : NUM_K_TILES;
            for (int k = k_begin; k < k_end; ++k) {
                s.full_barriers[stage].wait(phase);
                auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage * SSFA);
                auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage * SMEM_SFB_PER_STAGE);
                #pragma unroll
                for (int i = 0; i < SF_BLOCK_M / NUM_UTCCP_ALIGNED; ++i)
                    warp_transpose(sfa + i * NUM_UTCCP_ALIGNED);
                #pragma unroll
                for (int i = 0; i < SF_BLOCK_N / NUM_UTCCP_ALIGNED; ++i)
                    warp_transpose(sfb + i * NUM_UTCCP_ALIGNED);
                cutlass::arch::fence_view_async_shared();
                // arrive at the LEADER CTA's barrier (both CTAs -> NUM_MULTICAST*32)
                s.with_sf_full_barriers[stage].arrive(0u);
                advance();
            }
            ++iter;
        }
    }

    // ======== WARP 1: MMA (leader only) ========
    else if (warp_id == 1 && is_leader) {
        auto instr_desc = w1_mma_desc::make_block_scaled_idesc<UMMA_N>();
        auto a_desc = w1_mma_desc::make_smem_desc_k_major_fp8(
            reinterpret_cast<__nv_fp8_e4m3*>(s.smem_a));    // activation (B-operand)
        auto b_desc = w1_mma_desc::make_smem_desc_k_major_fp8(
            reinterpret_cast<__nv_fp8_e4m3*>(s.smem_b));    // weight     (A-operand)
        uint32_t a_desc_lo = (lane_id < NS) ? a_desc.lo + lane_id * (SA / 16) : 0u;
        uint32_t b_desc_lo = (lane_id < NS) ? b_desc.lo + lane_id * (SMEM_B_PER_STAGE / 16) : 0u;

        int iter = 0; uint32_t m_block, n_block, k_split; uint32_t pit = 0;
        while (get_block(iter, m_block, n_block, k_split)) {
            const uint32_t accum_stage = pit % NUM_EPI_STAGES;
            const uint32_t accum_phase = (pit / NUM_EPI_STAGES) & 1;
            s.tmem_empty_barriers[accum_stage].wait(accum_phase ^ 1);
            w1ptx::tcgen05_fence_after_sync();
            const uint32_t tmem_c = accum_stage * UMMA_N;
            const int k_begin = k_split * k_tiles_per_split;
            const int k_end = (k_begin + k_tiles_per_split < NUM_K_TILES) ?
                              (k_begin + k_tiles_per_split) : NUM_K_TILES;

            for (int k = k_begin; k < k_end; ++k) {
                s.with_sf_full_barriers[stage].wait(phase);
                w1ptx::tcgen05_fence_after_sync();

                const uint32_t a_base = __shfl_sync(0xffffffff, a_desc_lo, stage);
                const uint32_t b_base = __shfl_sync(0xffffffff, b_desc_lo, stage);

                if (w1ptx::elect_one_sync()) {
                    // UTCCP scale factors into TMEM (2cta: each SM copies its local smem)
                    auto sf_desc = w1_mma_desc::make_sf_desc(nullptr);
                    auto* sfa = reinterpret_cast<uint32_t*>(s.smem_sfa + stage * SSFA);
                    auto* sfb = reinterpret_cast<uint32_t*>(s.smem_sfb + stage * SMEM_SFB_PER_STAGE);
                    #pragma unroll
                    for (int i = 0; i < SF_BLOCK_M / NUM_UTCCP_ALIGNED; ++i) {
                        w1_mma_desc::replace_sf_desc_addr(sf_desc, sfa + i * NUM_UTCCP_ALIGNED);
                        utccp_4x32_2cta(TMEM_SFA + i * 4, sf_desc_bits(sf_desc));
                    }
                    #pragma unroll
                    for (int i = 0; i < SF_BLOCK_N / NUM_UTCCP_ALIGNED; ++i) {
                        w1_mma_desc::replace_sf_desc_addr(sf_desc, sfb + i * NUM_UTCCP_ALIGNED);
                        utccp_4x32_2cta(TMEM_SFB + i * 4, sf_desc_bits(sf_desc));
                    }

                    // block-scale MMA over UMMA_K sub-blocks (gran_k=32 => sf id = kk)
                    #pragma unroll
                    for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                        const uint32_t sfa_id = kk, sfb_id = kk;
                        // swap-AB: A-operand=weight(b_desc), B-operand=activation(a_desc),
                        //          tmem_sfa=weight SF (TMEM_SFB), tmem_sfb=activation SF (TMEM_SFA)
                        const uint64_t rdesc = w1_mma_desc::make_runtime_idesc_with_sf_id(
                            instr_desc, sfb_id, sfa_id);
                        const uint32_t a_lo = w1_mma_desc::advance_desc_lo_for_k(a_base, kk);
                        const uint32_t b_lo = w1_mma_desc::advance_desc_lo_for_k(b_base, kk);
                        const uint64_t a_full = (static_cast<uint64_t>(a_desc.hi) << 32) | a_lo;
                        const uint64_t b_full = (static_cast<uint64_t>(b_desc.hi) << 32) | b_lo;
                        const uint32_t flag = (k > k_begin || kk > 0) ? 1u : 0u;
                        w1ptx::tcgen05_mma_2sm_block_scale(
                            tmem_c, b_full, a_full, rdesc, flag, TMEM_SFB, TMEM_SFA);
                    }
                }
                __syncwarp();

                w1ptx::umma_arrive_multicast_2sm(
                    reinterpret_cast<uint64_t*>(&s.empty_barriers[stage]), CTA_MASK);
                if (k == k_end - 1)
                    w1ptx::umma_arrive_multicast_2sm(
                        reinterpret_cast<uint64_t*>(&s.tmem_full_barriers[accum_stage]), CTA_MASK);
                __syncwarp();
                advance();
            }
            ++iter; ++pit;
        }
        if (pit > 0) {
            const uint32_t li = pit - 1;
            s.tmem_empty_barriers[li % NUM_EPI_STAGES].wait((li / NUM_EPI_STAGES) & 1);
        }
    }

    // ======== EPILOGUE WARPS (both CTAs, warps 4..7) ========
    else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
             warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        const uint32_t epi_warp = warp_id - (NUM_NON_EPI_THREADS / 32);   // 0..3
        constexpr int kNumBankGroupBytes = 16;
        constexpr int kNumSwizzleAtomRows = 8;
        constexpr int STORE_N_ATOM = STORE_BLOCK_N_ATOM;                  // 64 (bf16)
        constexpr int kNumWarpsPerAtom = STORE_N_ATOM / 32;              // 2

        uint32_t tma_stage = 0;
        int iter = 0; uint32_t m_block, n_block, k_split; uint32_t pit = 0;
        while (get_block(iter, m_block, n_block, k_split)) {
            const uint32_t accum_stage = pit % NUM_EPI_STAGES;
            const uint32_t accum_phase = (pit / NUM_EPI_STAGES) & 1;
            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            w1ptx::tcgen05_fence_after_sync();

            const uint32_t tmem_base = accum_stage * UMMA_N;
            const uint32_t base_m = m_block * Dims::BLOCK_M;
            const uint32_t base_n = n_block * BLOCK_N;

            const int num_stores = Dims::BLOCK_M / STORE_BLOCK_M;   // token / 16
            for (int st = 0; st < num_stores; ++st, tma_stage = (tma_stage + 1) % NUM_TMA_STORE_STAGES) {
                auto* smem_cd = s.smem_cd + tma_stage * SMEM_CD_PER_STAGE;
                if (epi_warp == 0) cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                // Read TMEM (BF16 path: 16dp256b x2) and STSM-transpose into swizzled smem.
                #pragma unroll
                for (int i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++i) {
                    const uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * kNumSwizzleAtomRows;
                    uint32_t v[8];
                    w1ptx::tmem_load_16dp256b1x(tmem_addr,          v[0], v[1], v[2], v[3]);
                    w1ptx::tmem_load_16dp256b1x(tmem_addr | 0x00100000u, v[4], v[5], v[6], v[7]);
                    cutlass::arch::fence_view_async_tmem_load();

                    uint8_t* base = smem_cd
                        + (epi_warp / kNumWarpsPerAtom) * (STORE_BLOCK_M * SWIZZLE_CD)
                        + i * (kNumSwizzleAtomRows * SWIZZLE_CD);
                    const uint32_t row = lane_id % 8;
                    const uint32_t col = (epi_warp % 2) * 4 + lane_id / 8;
                    uint8_t* dst = base + row * (kNumBankGroupBytes * 8) + (col ^ row) * kNumBankGroupBytes;
                    w1ptx::stmatrix_x4_trans_b16(dst,
                        w1ptx::cast_pack_bf16(v[0], v[1]), w1ptx::cast_pack_bf16(v[2], v[3]),
                        w1ptx::cast_pack_bf16(v[4], v[5]), w1ptx::cast_pack_bf16(v[6], v[7]));
                }

                if (st == num_stores - 1) {
                    w1ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);   // leader CTA
                }
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                if (epi_warp == 0 && w1ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int i = 0; i < STORE_BLOCK_N / STORE_N_ATOM; ++i) {
                        auto* sp = reinterpret_cast<nv_bfloat16*>(smem_cd) + i * (STORE_BLOCK_M * STORE_N_ATOM);
                        const uint32_t n_idx = base_n + i * STORE_N_ATOM;
                        const uint32_t m_idx = base_m + st * STORE_BLOCK_M;
                        if (num_k_splits > 1)
                            w1tma::reduce_add_2d(&desc_D, sp, n_idx, m_idx);
                        else
                            w1tma::store_2d(&desc_D, sp, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
            ++iter; ++pit;
        }
    }

    // ---- Cleanup ----
    w1ptx::cluster_sync();
    if (warp_id == 0)
        w1ptx::tcgen05_dealloc_2sm(0, NUM_TMEM_COLS);
}

// ======================== Host: TMA descriptors ========================
static CUtensorMapSwizzle to_swizzle(int mode) {
    switch (mode) {
        case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default:  return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

// make_tma_2d: gmem [inner, outer], smem box [inner, outer]. swizzle!=0 => smem_inner = swizzle/elem.
static CUtensorMap make_tma_2d(const char* name, void* ptr, CUtensorMapDataType dtype, int elem_size,
                               int gmem_inner, int gmem_outer, int smem_inner, int smem_outer,
                               int gmem_outer_stride_elems, int swizzle_mode) {
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
    CUtensorMap tm{};
    cuuint64_t gdims[2]  = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    cuuint32_t sdims[2]  = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gstr[1]   = {(cuuint64_t)gmem_outer_stride_elems * elem_size};
    cuuint32_t estr[2]   = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(&tm, dtype, 2, ptr, gdims, gstr, sdims, estr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, to_swizzle(swizzle_mode),
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        const char* estr_msg = nullptr; cuGetErrorString(res, &estr_msg);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    (estr_msg ? estr_msg : "unknown"),
                    " [gmem=", gmem_inner, "x", gmem_outer, " smem=", smem_inner, "x", smem_outer,
                    " stride=", gmem_outer_stride_elems, " swizzle=", swizzle_mode, "]");
    }
    return tm;
}

// TMEM budget check for a swap-AB block_m (2*bm accum + SFA + SFB cols <= 512).
static int host_ceil_div(int a, int b) { return (a + b - 1) / b; }
static int host_align_up(int a, int b) { return (a + b - 1) / b * b; }
static bool tmem_fits(int block_m) {
    const int sfa = host_align_up(block_m, 128) / 32;
    return 2 * block_m + sfa + w1fp8::NUM_SFB_TMEM_COLS <= 512;
}

// Experiment: use BM=M for M<=224, and BM=128 for M=256 to stay within TMEM.
static int choose_block_m_b300(int M) {
    switch (M) {
        case 32:  return 32;
        case 64:  return 64;
        case 96:  return 96;
        case 128: return 128;
        case 160: return 160;
        case 192: return 192;
        case 224: return 224;
        case 256: return 128;
        default:  return -1;
    }
}

static void set_kernel_smem_attr_once(void* kptr, int block_m, int smem) {
    const int idx = block_m / 16;
    TORCH_CHECK(idx >= 1 && idx < 17, "invalid block_m for smem attribute cache: ", block_m);

    static bool configured[17] = {};
    if (!configured[idx]) {
        auto e = cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute: ", cudaGetErrorString(e), " smem=", smem);
        configured[idx] = true;
    }
}

// num_1d_blocks_per_group (DeepGEMM get_num_1d_blocks_per_group, grouping on N).
static int compute_num_1d_blocks_per_group(int block_m, int block_n, int num_sms) {
    int best = 0; long best_usage = -1;
    for (int cand : {8, 16}) {
        long usage = (long)cand * block_n + (long)((num_sms + cand - 1) / cand) * block_m; // grouping on N
        if (best_usage < 0 || usage < best_usage) { best_usage = usage; best = cand; }
    }
    return best;
}

// ======================== PyTorch binding ========================
static void w1_merged_fp8_gemm_launch(
    torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w1_fp8, torch::Tensor w1_sf,
    torch::Tensor out)
{
    // ---- Input validation ----
    TORCH_CHECK(x_fp8.is_cuda() && x_fp8.scalar_type() == torch::kFloat8_e4m3fn, "x_fp8 must be CUDA e4m3");
    TORCH_CHECK(w1_fp8.is_cuda() && w1_fp8.scalar_type() == torch::kFloat8_e4m3fn, "w1_fp8 must be CUDA e4m3");
    TORCH_CHECK(x_sf.is_cuda() && x_sf.scalar_type() == torch::kInt32, "x_sf must be CUDA int32");
    TORCH_CHECK(w1_sf.is_cuda() && w1_sf.scalar_type() == torch::kInt32, "w1_sf must be CUDA int32");

    const int M = x_fp8.size(0);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0, "M must be a multiple of 32 in [32,256], got ", M);
    TORCH_CHECK(x_fp8.dim() == 2 && x_fp8.size(1) == K_DIM, "x_fp8 must be [M,", K_DIM, "]");
    TORCH_CHECK(w1_fp8.dim() == 2 && w1_fp8.size(0) == N_TOTAL && w1_fp8.size(1) == K_DIM,
                "w1_fp8 must be [", N_TOTAL, ",", K_DIM, "]");
    // Operands are read as K-contiguous [*,K] (stride: inner K==1, outer==K).
    TORCH_CHECK(x_fp8.stride(1) == 1 && x_fp8.stride(0) == K_DIM, "x_fp8 must be K-contiguous [M,K]");
    TORCH_CHECK(w1_fp8.stride(1) == 1 && w1_fp8.stride(0) == K_DIM, "w1_fp8 must be K-contiguous [N,K]");

    // SF physical layout must match DeepGEMM MN-major packed: shape [sf_k, aligned_mn],
    // mn contiguous (stride 1), sf_k stride == aligned_mn.
    const int sf_k   = (K_DIM + GRAN_K * 4 - 1) / (GRAN_K * 4);          // 56
    const int sfa_mn = host_align_up(M, 16 / SF_ELEM_SIZE);             // align M to 4
    const int sfb_mn = host_align_up(N_TOTAL, 16 / SF_ELEM_SIZE);
    TORCH_CHECK(x_sf.dim() == 2 && x_sf.size(0) == sf_k && x_sf.size(1) == sfa_mn,
                "x_sf must be [", sf_k, ",", sfa_mn, "] (DeepGEMM MN-major packed), got [",
                x_sf.size(0), ",", x_sf.dim() > 1 ? x_sf.size(1) : -1, "]");
    TORCH_CHECK(w1_sf.dim() == 2 && w1_sf.size(0) == sf_k && w1_sf.size(1) == sfb_mn,
                "w1_sf must be [", sf_k, ",", sfb_mn, "] (DeepGEMM MN-major packed)");
    TORCH_CHECK(x_sf.stride(1) == 1 && x_sf.stride(0) == sfa_mn, "x_sf must be [sf_k, aligned_mn] contiguous");
    TORCH_CHECK(w1_sf.stride(1) == 1 && w1_sf.stride(0) == sfb_mn, "w1_sf must be [sf_k, aligned_mn] contiguous");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kBFloat16, "out must be CUDA bf16");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N_TOTAL,
                "out must be [", M, ",", N_TOTAL, "]");
    TORCH_CHECK(out.stride(1) == 1 && out.stride(0) == N_TOTAL, "out must be row-major contiguous [M,N]");

    auto stream = at::cuda::getCurrentCUDAStream();

    static const int num_SMs = []() {
        int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n;
    }();
    const int physical_grid = (num_SMs / NUM_MULTICAST) * NUM_MULTICAST;
    TORCH_CHECK(physical_grid >= NUM_MULTICAST, "invalid SM count for 2SM launch: ", num_SMs);

    const int block_m = choose_block_m_b300(M);
    TORCH_CHECK(block_m > 0, "no valid block_m for M=", M);
    TORCH_CHECK(tmem_fits(block_m), "B300 block_m table violates TMEM budget for M=", M, " block_m=", block_m);
    const int block_n = BLOCK_N;
    const int num_m_blocks = host_ceil_div(M, block_m);
    const int num_n_blocks = N_TOTAL / block_n;
    const int num_mn_blocks = num_m_blocks * num_n_blocks;
    const int max_k_splits_in_one_wave = physical_grid / num_mn_blocks;
    const int target_k_splits = max_k_splits_in_one_wave > 0 ? max_k_splits_in_one_wave : 1;
    const int k_tiles_per_split = host_ceil_div(NUM_K_TILES, target_k_splits);
    const int num_k_splits = host_ceil_div(NUM_K_TILES, k_tiles_per_split);
    const int total_blocks = num_mn_blocks * num_k_splits;
    int grid = total_blocks < physical_grid ? host_align_up(total_blocks, NUM_MULTICAST) : physical_grid;
    if (grid < NUM_MULTICAST) grid = NUM_MULTICAST;
    const int n1d = compute_num_1d_blocks_per_group(block_m, block_n, physical_grid);

    void* xp = x_fp8.data_ptr(); void* wp = w1_fp8.data_ptr();
    void* xsf = x_sf.data_ptr(); void* wsf = w1_sf.data_ptr(); void* op = out.data_ptr();

    CUtensorMap dA = make_tma_2d("A", xp, CU_TENSOR_MAP_DATA_TYPE_UINT8, FP8_ELEM_SIZE,
                                 K_DIM, M, BLOCK_K, block_m / NUM_MULTICAST, K_DIM, SWIZZLE_A);
    CUtensorMap dB = make_tma_2d("B", wp, CU_TENSOR_MAP_DATA_TYPE_UINT8, FP8_ELEM_SIZE,
                                 K_DIM, N_TOTAL, BLOCK_K, block_n, K_DIM, SWIZZLE_B);
    CUtensorMap dSFA = make_tma_2d("SFA", xsf, CU_TENSOR_MAP_DATA_TYPE_INT32, SF_ELEM_SIZE,
                                   sfa_mn, sf_k, block_m, 1, sfa_mn, 0);
    CUtensorMap dSFB = make_tma_2d("SFB", wsf, CU_TENSOR_MAP_DATA_TYPE_INT32, SF_ELEM_SIZE,
                                   sfb_mn, sf_k, block_n, 1, sfb_mn, 0);
    CUtensorMap dD = make_tma_2d("D", op, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, BF16_ELEM_SIZE,
                                 N_TOTAL, M, STORE_BLOCK_N, STORE_BLOCK_M, N_TOTAL, SWIZZLE_CD);

    // Select kernel + smem by block_m (compile-time template).
    void* kptr = nullptr; int smem = 0;
    auto pick = [&](auto tag) {
        constexpr int BM = decltype(tag)::value;
        kptr = (void*)&w1_merged_fp8_kernel<BM>;
        smem = (int)sizeof(SharedStorage<BM>);
    };
    switch (block_m) {
        case 16:  pick(std::integral_constant<int,16>{});  break;
        case 32:  pick(std::integral_constant<int,32>{});  break;
        case 48:  pick(std::integral_constant<int,48>{});  break;
        case 64:  pick(std::integral_constant<int,64>{});  break;
        case 80:  pick(std::integral_constant<int,80>{});  break;
        case 96:  pick(std::integral_constant<int,96>{});  break;
        case 112: pick(std::integral_constant<int,112>{}); break;
        case 128: pick(std::integral_constant<int,128>{}); break;
        case 160: pick(std::integral_constant<int,160>{}); break;
        case 192: pick(std::integral_constant<int,192>{}); break;
        case 224: pick(std::integral_constant<int,224>{}); break;
        default: TORCH_CHECK(false, "Unsupported block_m=", block_m);
    }

    set_kernel_smem_attr_once(kptr, block_m, smem);

    if (num_k_splits > 1) {
        const size_t out_bytes = static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
        auto e = cudaMemsetAsync(op, 0, out_bytes, stream);
        TORCH_CHECK(e == cudaSuccess, "cudaMemsetAsync(out) failed: ", cudaGetErrorString(e));
    }

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid, 1, 1);
    config.blockDim = dim3(TPB, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {(unsigned)NUM_MULTICAST, 1, 1};
    config.attrs = attrs; config.numAttrs = 1;

    int shape_m = M;
    // NOTE: scheduler stride = launched grid (not physical num_SMs).
    void* args[] = {&dA, &dB, &dSFA, &dSFB, &dD, &shape_m, &grid, (void*)&n1d,
                    (void*)&num_k_splits, (void*)&k_tiles_per_split};
    auto err = cudaLaunchKernelExC(&config, kptr, args);
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

torch::Tensor w1_merged_fp8_gemm(
    torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w1_fp8, torch::Tensor w1_sf)
{
    const int M = x_fp8.size(0);
    auto out = torch::empty({M, N_TOTAL}, x_fp8.options().dtype(torch::kBFloat16));
    w1_merged_fp8_gemm_launch(x_fp8, x_sf, w1_fp8, w1_sf, out);
    return out;
}

torch::Tensor w1_merged_fp8_gemm_out(
    torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w1_fp8, torch::Tensor w1_sf,
    torch::Tensor out)
{
    w1_merged_fp8_gemm_launch(x_fp8, x_sf, w1_fp8, w1_sf, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w1_merged_fp8_gemm", &w1_merged_fp8_gemm,
          "w1 merged FP8 block-scale GEMM (tcgen05 2SM swap-AB, DeepGEMM-aligned)");
    m.def("w1_merged_fp8_gemm_out", &w1_merged_fp8_gemm_out,
          "w1 merged FP8 block-scale GEMM with preallocated output");
}
