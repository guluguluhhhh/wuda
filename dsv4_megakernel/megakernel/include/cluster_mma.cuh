#pragma once
// ============================================================
// cluster_mma.cuh
// Single-cluster BF16 swap-AB MMA engine (tcgen05, 2SM) — reusable building
// block for the dsv4 decode megakernel.
//
// Extracted from megakernel/kernels/wq_b_gemm.cu (leader MMA warp, warp1):
// it does ONLY the tensor-core part of one cluster's tile — no TMA loads,
// no epilogue, no persistent scheduling. Those stay with the caller.
//
// Fixed tile shape (swap-AB, cta_group::2):
//   BM = 128 (problem-M tile) = UMMA_N = 128
//   BN = 128 (per-CTA problem-N); cluster of 2 CTAs -> cluster-N = 256 = UMMA_M
//   UMMA_K = 16 (BF16)
//   one tcgen05.mma instruction = 256 x 128 x 16
//
// The A-operand is the WEIGHT tile (spans problem-N via UMMA_M), the B-operand
// is the ACTIVATION tile (spans problem-M via UMMA_N). Both K-major, 128B swizzle.
//
// Self-contained: depends only on CUTLASS/CuTe arch headers. All PTX/descriptor
// helpers live in cluster_mma::detail so this header can coexist in the same TU
// as wq_b_gemm.cuh (whose helpers sit in the global ptx/mma_desc namespaces).
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

namespace cluster_mma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// ======================== Internal helpers ========================
namespace detail {

// K-major 128B-swizzle SMEM descriptor. For K-major, num_non_contiguous is
// fixed at 128/16 = 8, so SBO depends only on BLOCK_K. Works for both the
// weight tile (128 rows) and the activation tile (BM/2 rows per CTA).
template <int BLOCK_K>
__device__ __forceinline__
cute::UMMA::SmemDescriptor make_smem_desc_k_major(void* smem_ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;
    desc.lbo_mode_    = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
    // num_non_contiguous(8) * BLOCK_K * sizeof(bf16)
    constexpr uint32_t SBO = 8u * BLOCK_K * sizeof(nv_bfloat16);
    desc.stride_byte_offset_  = SBO >> 4;
    desc.leading_byte_offset_ = 0;
    desc.base_offset_         = 0;
    return desc;
}

// Runtime instruction descriptor for the fixed 256x128 BF16 2SM MMA.
// swap-AB: both operands K-major so the major order is symmetric.
__device__ __forceinline__
uint64_t make_runtime_instr_desc() {
    auto idesc = cute::UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        /*UMMA_M=*/256, /*UMMA_N=*/128,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    return cute::UMMA::make_runtime_instr_desc(idesc);
}

// Advance descriptor .lo by one UMMA_K step within a stage.
// K-major bf16: UMMA_K=16 elems -> 32 bytes -> 32/16 = 2 units.
__device__ __forceinline__
uint32_t advance_lo_k(uint32_t base_lo, uint32_t kk) {
    return base_lo + kk * (16u * sizeof(nv_bfloat16) / 16u); // kk * 2
}

// tcgen05.mma.cta_group::2.kind::f16 (only the high 32 bits of idesc are used).
__device__ __forceinline__ void mma_2sm(
    uint32_t tmem_addr, uint64_t a_desc, uint64_t b_desc,
    uint64_t runtime_idesc, uint32_t accum) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_addr), "l"(a_desc), "l"(b_desc),
           "r"(static_cast<uint32_t>(runtime_idesc >> 32)), "r"(accum));
}

// tcgen05.commit — arrive an mbarrier for 2SM multicast.
__device__ __forceinline__ void arrive_multicast_2sm(Barrier* bar, uint16_t mask) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    if (cute::elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
            :: "r"(addr), "h"(mask) : "memory");
    }
}

__device__ __forceinline__ void fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}" : "=r"(pred));
    return pred != 0;
}

} // namespace detail

// ======================== Cluster MMA engine ========================
// BLOCK_K   : K per pipeline stage (e.g. 64). Must be a multiple of UMMA_K(16).
// NUM_STAGES: number of SMEM pipeline stages (bounds the per-lane descriptor table).
template <int BLOCK_K, int NUM_STAGES>
struct ClusterMmaBF16 {
    // ---- Fixed geometry ----
    static constexpr int BM = 128, BN = 128, NUM_MULTICAST = 2;
    static constexpr int UMMA_M = 256, UMMA_N = 128, UMMA_K = 16;
    static constexpr int STEPS_PER_STAGE = BLOCK_K / UMMA_K;          // UMMA_K steps per stage
    static constexpr uint16_t CTA_MASK = (1 << NUM_MULTICAST) - 1;    // 0b11

    // Per-stage SMEM byte strides for the standard tight K-major layout.
    // Activation is split across the 2 CTAs on M -> BM/2 rows per CTA.
    static constexpr int SMEM_A_PER_STAGE = (BM / NUM_MULTICAST) * BLOCK_K * (int)sizeof(nv_bfloat16); // 64*BK*2
    static constexpr int SMEM_B_PER_STAGE = BN * BLOCK_K * (int)sizeof(nv_bfloat16);                    // 128*BK*2

    static_assert(BLOCK_K % UMMA_K == 0, "BLOCK_K must be a multiple of UMMA_K(16)");

    // Descriptor state precomputed once per MMA warp (warp-shuffle table trick).
    struct DescState {
        uint32_t act_lo, wgt_lo;   // per-lane, stage-indexed .lo (valid for lane < NUM_STAGES)
        uint32_t act_hi, wgt_hi;   // constant .hi
        uint64_t idesc;            // runtime instruction descriptor
    };

    // Build the per-stage descriptor tables. Call once at MMA-warp entry, by all
    // 32 lanes of the leader CTA's MMA warp.
    //   smem_a_base = activation SMEM base (B-operand)
    //   smem_b_base = weight     SMEM base (A-operand)
    static __device__ __forceinline__
    DescState init_desc(nv_bfloat16* smem_a_base, nv_bfloat16* smem_b_base, uint32_t lane_id) {
        auto act_desc = detail::make_smem_desc_k_major<BLOCK_K>(smem_a_base);
        auto wgt_desc = detail::make_smem_desc_k_major<BLOCK_K>(smem_b_base);

        DescState ds;
        ds.act_lo = (lane_id < NUM_STAGES)
            ? act_desc.lo + lane_id * (SMEM_A_PER_STAGE / 16) : 0u;
        ds.wgt_lo = (lane_id < NUM_STAGES)
            ? wgt_desc.lo + lane_id * (SMEM_B_PER_STAGE / 16) : 0u;
        ds.act_hi = act_desc.hi;
        ds.wgt_hi = wgt_desc.hi;
        ds.idesc  = detail::make_runtime_instr_desc();
        return ds;
    }

    // Run the full K-loop for one cluster tile, accumulating into TMEM at column
    // tmem_c. Entered by all 32 lanes of the leader CTA's MMA warp.
    //
    // Sequence (faithful to wq_b_gemm.cu warp1):
    //   wait tmem_empty(accum_phase ^ 1)  -- previous epilogue has drained the accumulator
    //   for k in [0, num_k_tiles):
    //       wait full_barriers[stage] ; issue STEPS_PER_STAGE MMAs (accum after the first)
    //       arrive empty_barriers[stage] (multicast)
    //       on last k: arrive tmem_full (multicast)  -- result ready for the epilogue
    // Advances stage_idx / phase in place across the num_k_tiles stages.
    //
    // NOTE: the caller still owns the outer tile-scheduling loop and the final
    // drain wait on tmem_empty after the last tile.
    static __device__ __forceinline__
    void run_tile(const DescState& ds,
                  Barrier* full_barriers, Barrier* empty_barriers,
                  Barrier& tmem_full, Barrier& tmem_empty,
                  uint32_t tmem_c, int num_k_tiles, uint32_t accum_phase,
                  uint32_t& stage_idx, uint32_t& phase) {
        tmem_empty.wait(accum_phase ^ 1);
        detail::fence_after_sync();

        for (int k = 0; k < num_k_tiles; ++k) {
            full_barriers[stage_idx].wait(phase);
            detail::fence_after_sync();

            uint32_t w_base = __shfl_sync(0xffffffff, ds.wgt_lo, stage_idx);
            uint32_t a_base = __shfl_sync(0xffffffff, ds.act_lo, stage_idx);

            if (detail::elect_one_sync()) {
                #pragma unroll
                for (int kk = 0; kk < STEPS_PER_STAGE; ++kk) {
                    uint32_t w_lo = detail::advance_lo_k(w_base, kk);
                    uint32_t a_lo = detail::advance_lo_k(a_base, kk);
                    uint64_t w_full = (static_cast<uint64_t>(ds.wgt_hi) << 32) | w_lo;
                    uint64_t a_full = (static_cast<uint64_t>(ds.act_hi) << 32) | a_lo;

                    uint32_t accum_flag = (k > 0 || kk > 0) ? 1 : 0;
                    // A-operand = weight (UMMA_M=256), B-operand = activation (UMMA_N=128)
                    detail::mma_2sm(tmem_c, w_full, a_full, ds.idesc, accum_flag);
                }
            }
            __syncwarp();

            detail::arrive_multicast_2sm(&empty_barriers[stage_idx], CTA_MASK);
            if (k == num_k_tiles - 1) {
                detail::arrive_multicast_2sm(&tmem_full, CTA_MASK);
            }
            __syncwarp();

            stage_idx = (stage_idx + 1) % NUM_STAGES;
            if (stage_idx == 0) phase ^= 1;
        }
    }
};

} // namespace cluster_mma
