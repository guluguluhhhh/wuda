#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

#include "query_rms_rope.cuh"
#include "tp2_symmetric.cuh"

// FP8 MQA uses DeepGEMM's paged scheduler and device math. This local copy keeps
// that attention path isolated from the existing FP4 implementation and adds a
// fourth warpgroup which runs query RMSNorm+RoPE and the DSV4 MAIN compressor.

namespace wuda_fp8_mqa {

struct MainCompressorArgs {
    const long long* pos = nullptr;
    const float* norm = nullptr;
    const float* cos_tab = nullptr;
    const float* sin_tab = nullptr;
    const float* state = nullptr;
    const int64_t* state_row = nullptr;
    int state_ring_entries = 0;
    uint8_t* q8 = nullptr;
    float* s8 = nullptr;
    nv_bfloat16* rope = nullptr;
    uint8_t* cmp_cache = nullptr;
    const int64_t* cmp_dst = nullptr;
    int cmp_entries_per_block = 0;
    size_t cmp_block_stride_bytes = 0;
    uint32_t seq_len = 0;
};

struct QueryRmsRopeArgs {
    const nv_bfloat16* x = nullptr;
    const long long* positions = nullptr;
    const float* cos_tab = nullptr;
    const float* sin_tab = nullptr;
    nv_bfloat16* out = nullptr;
    const int* work_flag = nullptr;
    int input_heads = 0;
    uint32_t output_rows = 0;
    float eps = 1e-6f;
    wuda::tp2::SymmetricView symmetric_output;
    nv_bfloat16* local_second_out = nullptr;
    // 0: local payload/no handshake, 1: local payload/handshake,
    // 2: peer payload/handshake. nullptr is the production peer mode.
    // Ablation-only extensions: 3 dense-order local control (row r -> slot r
    // of local_second_out, no handshake), 4 clustered local (one warp owns 16
    // consecutive heads, no handshake), 5 clustered peer payload/handshake.
    const uint32_t* comm_mode = nullptr;
    uint32_t batch_total = 0;
    uint32_t rank0_batch = 0;
};

// Any warp that has finished its primary role claims one query row from a
// per-CTA shared queue. A full warp owns one complete 512-wide RMS reduction.
template <bool kAblation>
__device__ __forceinline__ void run_query_rms_rope(
        const QueryRmsRopeArgs& query, uint32_t* next, uint32_t lane) {
    using namespace wuda_query_rms_rope;
    const bool do_work = !kAblation || *query.work_flag != 0;
    const bool tp2 = static_cast<bool>(query.symmetric_output);
    // The mode is stable across one launch; read it once per claiming warp.
    const uint32_t mode = query.comm_mode == nullptr ? 2u : *query.comm_mode;
    const bool clustered = tp2 && mode >= 4u;
    const bool remote_payload = mode == 2u || mode == 5u;

    auto process_row = [&](uint32_t input_row, uint32_t batch, uint4* output) {
        const uint4* input = reinterpret_cast<const uint4*>(
            query.x + static_cast<uint64_t>(input_row) * kHidden) + lane;
        uint4 values[kVecsPerThread];
        #pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i)
            values[i] = input[i * kWarp];

        float sumsq = 0.0f;
        #pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i)
            sumsq += vec_sumsq(values[i]);
        const float scale = rsqrtf(
            warp_sum(sumsq)
                / static_cast<float>(kHidden) + query.eps);

        const long long position = query.positions[batch];
        const float* cos = query.cos_tab
            + static_cast<uint64_t>(position) * (kRope / 2);
        const float* sin = query.sin_tab
            + static_cast<uint64_t>(position) * (kRope / 2);
        output[0] = scale_vec(values[0], scale);
        if (lane >= 24) {
            const int pair_base = (lane - 24) * (kElemsPerVec / 2);
            output[kWarp] = scale_rope_vec(
                values[1], scale, cos, sin, pair_base);
        } else
            output[kWarp] = scale_vec(values[1], scale);
    };

    while (true) {
        uint32_t item = 0;
        if (lane == 0)
            item = atomicAdd(next, 1u);
        item = __shfl_sync(0xffffffffu, item, 0);
        if (clustered) {
            // Clustered claim: one warp owns 16 consecutive heads of one
            // batch, so its reads and its writes each form one contiguous
            // 16KB run instead of 16 scattered 1KB rows.
            constexpr uint32_t kClusterHeads = 16;
            const uint32_t tasks_per_batch =
                static_cast<uint32_t>(query.input_heads) / kClusterHeads;
            const uint32_t task = blockIdx.x + item * gridDim.x;
            if (task >= query.output_rows / kClusterHeads)
                break;
            if (!do_work)
                continue;
            const uint32_t batch = task / tasks_per_batch;
            const uint32_t head_base = (task % tasks_per_batch) * kClusterHeads;
            const uint32_t owner = batch >= query.rank0_batch;
            const uint32_t owner_row =
                owner == 0 ? batch : batch - query.rank0_batch;
            nv_bfloat16* destination;
            if (owner == query.symmetric_output.rank) {
                destination = query.out;
            } else {
                destination = remote_payload
                    ? query.symmetric_output.peer_base<nv_bfloat16>()
                    : query.local_second_out;
            }
            const uint32_t full_head_base =
                query.symmetric_output.rank * 64 + head_base;
            #pragma unroll 1
            for (uint32_t h = 0; h < kClusterHeads; ++h) {
                const uint32_t input_row =
                    batch * query.input_heads + head_base + h;
                uint4* output = reinterpret_cast<uint4*>(
                    destination
                    + (static_cast<uint64_t>(owner_row) * 128
                       + full_head_base + h) * kHidden) + lane;
                process_row(input_row, batch, output);
            }
            continue;
        }
        const uint32_t output_row = blockIdx.x + item * gridDim.x;
        if (output_row >= query.output_rows)
            break;
        // MODEL1 FlashMLA always consumes 128 query heads. Both production
        // input geometries (TP=64 and full-rank=128) are powers of two, so avoid
        // repeating runtime integer division/modulo in every lane of every row.
        const uint32_t batch = tp2
            ? output_row / static_cast<uint32_t>(query.input_heads)
            : output_row >> 7;
        const uint32_t output_head = tp2
            ? output_row % static_cast<uint32_t>(query.input_heads)
            : output_row & 127u;
        const uint32_t input_head = output_head & (query.input_heads - 1);
        const uint32_t input_row = batch * query.input_heads + input_head;

        if (!do_work)
            continue;
        uint4* output;
        if (tp2) {
            if (mode == 3u) {
                // Dense-order control: identical reads and volume, but row r
                // lands in slot r of the plain benchmark buffer.
                output = reinterpret_cast<uint4*>(
                    query.local_second_out
                    + static_cast<uint64_t>(output_row) * kHidden) + lane;
            } else {
                const uint32_t owner = batch >= query.rank0_batch;
                const uint32_t owner_row =
                    owner == 0 ? batch : batch - query.rank0_batch;
                const uint32_t full_head =
                    query.symmetric_output.rank * 64 + input_head;
                nv_bfloat16* destination;
                if (owner == query.symmetric_output.rank) {
                    destination = query.out;
                } else {
                    destination = remote_payload
                        ? query.symmetric_output.peer_base<nv_bfloat16>()
                        : query.local_second_out;
                }
                output = reinterpret_cast<uint4*>(
                    destination
                    + (static_cast<uint64_t>(owner_row) * 128 + full_head)
                        * kHidden) + lane;
            }
        } else {
            output = reinterpret_cast<uint4*>(
                query.out + static_cast<uint64_t>(output_row) * kHidden) + lane;
        }
        process_row(input_row, batch, output);
    }
}

constexpr int kM1TokenBodyBytes = 576;
constexpr int kM1ScaleRecordBytes = 8;

__device__ __forceinline__ int m1_flog2_ceil(float x) {
    const unsigned bits = __float_as_uint(x);
    const int exp_field = static_cast<int>((bits >> 23) & 0xff);
    return exp_field - 127 + ((bits & 0x7fffffu) != 0u ? 1 : 0);
}

__device__ __forceinline__ float m1_fpow2(int e) {
    return __uint_as_float(static_cast<unsigned>(e + 127) << 23);
}

// Four warps cooperate on one 512-wide compressor row. This is intentionally
// independent of the attention shared-memory/TMEM pipelines.
__device__ __forceinline__ void run_main_compressor_row(
    const MainCompressorArgs& comp, uint32_t m, long long p,
    uint32_t group, uint32_t lane, float rms_eps, uint32_t barrier_id) {
    constexpr uint32_t kDim = 512;
    constexpr uint32_t kStateWidth = 1024;
    constexpr uint32_t kRopeDim = 64;
    constexpr uint32_t kRatio = 4;
    __shared__ float ssq_smem[4];

    const int64_t state_row = comp.state_row[m];
    if (state_row < 0)
        return;
    const int ring = comp.state_ring_entries;
    const int current = state_row % ring;
    const int first = current >= 7 ? current - 7 : current - 7 + ring;
    const float* state_base = comp.state
        + static_cast<uint64_t>(state_row - current) * (2 * kStateWidth);

    float values[4];
    {
        const uint32_t float4_col = group * 32 + lane;
        float4 scores[8], states[8];
        #pragma unroll
        for (uint32_t row = 0; row < 8; ++row) {
            int physical_row = first + row;
            if (physical_row >= ring)
                physical_row -= ring;
            const uint32_t col = row < kRatio
                ? float4_col : kDim / 4 + float4_col;
            const float4* entry = reinterpret_cast<const float4*>(
                state_base + static_cast<uint64_t>(physical_row) * (2 * kStateWidth));
            scores[row] = entry[kStateWidth / 4 + col];
            states[row] = entry[col];
        }
        #pragma unroll
        for (uint32_t e = 0; e < 4; ++e) {
            float maximum = (&scores[0].x)[e];
            #pragma unroll
            for (uint32_t row = 1; row < 8; ++row)
                maximum = fmaxf(maximum, (&scores[row].x)[e]);
            float denominator = 0.0f, numerator = 0.0f;
            #pragma unroll
            for (uint32_t row = 0; row < 8; ++row) {
                const float exponent = expf((&scores[row].x)[e] - maximum);
                denominator += exponent;
                numerator += exponent * (&states[row].x)[e];
            }
            values[e] = numerator / denominator;
        }
    }

    float partial_ssq = 0.0f;
    #pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
        const float rounded = __bfloat162float(__float2bfloat16(values[e]));
        partial_ssq += rounded * rounded;
    }
    #pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1)
        partial_ssq += __shfl_xor_sync(0xffffffffu, partial_ssq, offset);
    if (lane == 0)
        ssq_smem[group] = partial_ssq;
    cutlass::arch::NamedBarrier(128, barrier_id).sync();

    const float total_ssq = ssq_smem[0] + ssq_smem[1] + ssq_smem[2] + ssq_smem[3];
    const float rms = rsqrtf(total_ssq / static_cast<float>(kDim) + rms_eps);
    const float4 norm = reinterpret_cast<const float4*>(comp.norm)[group * 32 + lane];
    #pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
        const float rounded = __bfloat162float(__float2bfloat16(values[e]));
        values[e] = __bfloat162float(__float2bfloat16(rounded * rms * (&norm.x)[e]));
    }

    const long long compressed_position = p + 1 - kRatio;
    if (group == 3 && lane >= 16) {
        const float* cos_row = comp.cos_tab
            + static_cast<uint64_t>(compressed_position) * (kRopeDim / 2);
        const float* sin_row = comp.sin_tab
            + static_cast<uint64_t>(compressed_position) * (kRopeDim / 2);
        #pragma unroll
        for (uint32_t e = 0; e < 4; e += 2) {
            const uint32_t pair = (lane - 16) * 2 + e / 2;
            const float even = values[e], odd = values[e + 1];
            values[e] = __bfloat162float(__float2bfloat16(
                even * cos_row[pair] - odd * sin_row[pair]));
            values[e + 1] = __bfloat162float(__float2bfloat16(
                even * sin_row[pair] + odd * cos_row[pair]));
        }
    }

    float maximum = 0.0f;
    #pragma unroll
    for (uint32_t e = 0; e < 4; ++e)
        maximum = fmaxf(maximum, fabsf(values[e]));
    #pragma unroll
    for (uint32_t offset = 1; offset < 16; offset <<= 1)
        maximum = fmaxf(maximum,
                        __shfl_xor_sync(0xffffffffu, maximum, offset, 16));

    const uint32_t col = group * 128 + lane * 4;
    uint8_t* page = nullptr;
    int page_offset = 0;
    if (comp.cmp_cache != nullptr && comp.cmp_dst != nullptr
        && comp.cmp_dst[m] >= 0) {
        const int64_t destination = comp.cmp_dst[m];
        page = comp.cmp_cache
            + static_cast<uint64_t>(destination / comp.cmp_entries_per_block)
                * comp.cmp_block_stride_bytes;
        page_offset = destination % comp.cmp_entries_per_block;
    }

    if (col < kDim - kRopeDim) {
        const int scale_exponent = m1_flog2_ceil(
            fmaxf(maximum * (1.0f / 448.0f), 1e-4f));
        const float scale = m1_fpow2(scale_exponent);
        uint32_t packed = 0;
        #pragma unroll
        for (uint32_t e = 0; e < 4; ++e)
            packed |= static_cast<uint32_t>(__nv_fp8_e4m3(values[e] / scale).__x)
                      << (8 * e);
        if (comp.q8 != nullptr)
            *reinterpret_cast<uint32_t*>(comp.q8
                + static_cast<uint64_t>(m) * (kDim - kRopeDim) + col) = packed;
        if ((lane & 15) == 0 && comp.s8 != nullptr)
            comp.s8[static_cast<uint64_t>(m) * 7 + (col >> 6)] = scale;
        if (page != nullptr) {
            *reinterpret_cast<uint32_t*>(page
                + page_offset * kM1TokenBodyBytes + col) = packed;
            if ((lane & 15) == 0)
                page[static_cast<size_t>(comp.cmp_entries_per_block) * kM1TokenBodyBytes
                     + page_offset * kM1ScaleRecordBytes + (col >> 6)] =
                    static_cast<uint8_t>(scale_exponent + 127);
        }
    } else {
        #pragma unroll
        for (uint32_t e = 0; e < 4; e += 2) {
            const auto pair = __floats2bfloat162_rn(values[e], values[e + 1]);
            if (comp.rope != nullptr)
                *reinterpret_cast<uint32_t*>(comp.rope
                    + static_cast<uint64_t>(m) * kRopeDim
                    + (col - (kDim - kRopeDim)) + e) =
                    *reinterpret_cast<const uint32_t*>(&pair);
            if (page != nullptr)
                *reinterpret_cast<uint32_t*>(page
                    + page_offset * kM1TokenBodyBytes + (kDim - kRopeDim)
                    + ((col - (kDim - kRopeDim)) + e) * 2) =
                    *reinterpret_cast<const uint32_t*>(&pair);
        }
    }

    cutlass::arch::NamedBarrier(128, barrier_id).sync();
}

} // namespace wuda_fp8_mqa

// Shared SM100 MQA logits core plus contiguous-KV and paged entries.

namespace deep_gemm {

// Ring-buffer counter avoiding `% kNumStages`, which ptxas can lower poorly for TMEM paths
template <uint32_t kNumStages>
struct RingPipeline {
    uint32_t stage_idx = 0, phase = 0;

    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t& step = 1) {
        const uint32_t current_stage_idx = stage_idx, current_phase = phase;
        stage_idx += step;
        if (stage_idx >= kNumStages) {
            stage_idx -= kNumStages;
            phase ^= 1;
        }
        return {current_stage_idx, current_phase};
    }
};

// Convert runtime valid-token count to `cute::Int` so token loops stay compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(cute::Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(cute::Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// Shared device core parameterized by dtype and scheduler geometry/addressing
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kWaitPrimary = true>
CUTLASS_DEVICE void sm100_mqa_logits_core_impl(const uint32_t logits_stride,
                                               logits_dtype_t* logits,
                                               const cute::TmaDescriptor& tensor_map_q,
                                               const cute::TmaDescriptor& tensor_map_sf_q,
                                               const cute::TmaDescriptor& tensor_map_kv,
                                               const cute::TmaDescriptor& tensor_map_sf_kv,
                                               const cute::TmaDescriptor& tensor_map_weights,
                                               const MakeScheduler& make_scheduler) {
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = kIsMXSF ? math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kNumSFKV = kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kIsMXSF ? (kRealNumSFQ * sizeof(int)) : 0;
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(reduce_dtype_t);

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(kIsMXSF ? 1 : kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    // The fused kernel has an independent 128-thread compressor warpgroup.
    // Only the original 384 attention threads participate in this rendezvous.
    cutlass::arch::NamedBarrier(kNumSpecializedThreads + kNumMathThreads, 1).sync();

    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    // A 512-thread fused CTA has a hard 128-register/thread block budget. Keep
    // weights in shared memory and consume the 64-head accumulator in two passes
    // below, matching the register-diet strategy used by the FP4 fused kernel.
    constexpr uint32_t kNumMathRegisters = 128;

    if constexpr (kWaitPrimary)
        cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
            // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kNumQKBytesPerToken, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                if constexpr (kIsMXSF)
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                if constexpr (decltype(scheduler)::kIsPaged) {
                    constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
                    constexpr uint32_t kNumPagesPerSplit = decltype(scheduler)::kNumPagesPerSplit;
                    DG_STATIC_ASSERT(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

                    const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
                    if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
                        cached_kv_page_base = (kv_page_base / 32) * 32;
                        cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
                    }

                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    if (cute::elect_one_sync())
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    __syncwarp();

                    int page_coords[kNumPagesPerSplit];
                    #pragma unroll
                    for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                        const auto src_lane = static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
                        page_coords[page_idx] = __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
                    }

                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                            tma::copy<kNumQKBytesPerToken, kPageKV, 0, qk_dtype_t, true>(
                                &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                                smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                                0, 0, 1, page_coords[page_idx]);
                            tma::copy<kPageKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                     smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                                     0, page_coords[page_idx]);
                        }
                        smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                } else if (cute::elect_one_sync()) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

                    const uint32_t kv_tma_offset = scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
                    tma::copy<kNumQKBytesPerToken, SPLIT_KV, 0>(
                        &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                        smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
                    tma::copy<SPLIT_KV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            if constexpr (kIsMXSF) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                    if (cute::elect_one_sync())
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                    __syncwarp();
                }
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                if constexpr (kIsMXSF) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        utccp_required_smem_warp_transpose(smem_ptr);
                    }
                    cutlass::arch::fence_view_async_shared();
                }

                if (cute::elect_one_sync()) {
                    if constexpr (kIsMXSF) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        if constexpr (kIsMXSF) {
                            DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or kHeadDim == 64 or kHeadDim == 128, "Invalid head dim");

                            constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
                            constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

                            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
                            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t,
                                                                                       UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * kPackFactor, k * kPackFactor);
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                mma_op_t::fma(
                                    a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                    kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                            }
                        } else {
                            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                                            UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc);
                            }
                        }

                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(&smem.full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                __syncwarp();
                if constexpr (kIsMXSF)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_kv_barriers[kv_stage_idx]));
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
        DG_STATIC_ASSERT(not kIsReduceBF16,
                         "the fused FP8 path uses FP32 weights/reduction");
        DG_STATIC_ASSERT(kNumHeads == 4 or kNumHeads == 8 or kNumHeads == 16 or kNumHeads == 32 or kNumHeads == 64,
                         "Unsupported TMEM load size");
        float accum[kNumHeads / 2];

        auto tmem_load_no_fence = [](auto num_elems_t, const uint32_t& addr, float* load_dst) {
            constexpr uint32_t N = decltype(num_elems_t)::value;
            using Loader = cute::conditional_t<N == 2,  cute::SM100_TMEM_LOAD_32dp32b2x,
                           cute::conditional_t<N == 4,  cute::SM100_TMEM_LOAD_32dp32b4x,
                           cute::conditional_t<N == 8,  cute::SM100_TMEM_LOAD_32dp32b8x,
                           cute::conditional_t<N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                           cute::conditional_t<N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                                        cute::SM100_TMEM_LOAD_32dp32b64x>>>>>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(addr, reinterpret_cast<uint32_t*>(load_dst)[Is]...);
            }(cute::make_index_sequence<N>{});
        };

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

                    // FP8 consumes the KV pipeline directly to read per-KV scales;
                    // MXFP4 / MXFP8 folds its scale into the MX SF UMMA, no extra scale
                    reduce_dtype_t scale_kv = static_cast<reduce_dtype_t>(0.0f);
                    uint32_t kv_stage_idx = 0;
                    if constexpr (not kIsMXSF) {
                        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx_local, kv_phase);
                        kv_stage_idx = kv_stage_idx_local;
                        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                        scale_kv = static_cast<reduce_dtype_t>(ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx));
                    }

                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    // Release KV smem only after UMMA commits TMEM; earlier release races TMA overwrite
                    if constexpr (not kIsMXSF)
                        smem.empty_kv_barriers[kv_stage_idx].arrive();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        const auto weights = reinterpret_cast<const float2*>(
                            smem.smem_weights[q_stage_idx] + i * kNumHeads);
                        auto sum_0 = make_float2(0.0f, 0.0f);
                        auto sum_1 = make_float2(0.0f, 0.0f);

                        #pragma unroll
                        for (uint32_t half = 0; half < 2; ++half) {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{},
                                               tmem_addr + half * (kNumHeads / 2),
                                               accum);
                            cutlass::arch::fence_view_async_tmem_load();

                            if (half == 1 && i == kNumValidTokens - 1) {
                                ptx::tcgen05_before_thread_sync();
                                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                            }

                            const uint32_t head_base = half * (kNumHeads / 2);
                            const auto transform = [&](const uint32_t& j,
                                                       const float2& sum) {
                                const auto activation = make_float2(
                                    fmaxf(accum[j], 0.0f),
                                    fmaxf(accum[j + 1], 0.0f));
                                const auto weight = ptx::ld_shared(
                                    weights + ((head_base + j) >> 1));
                                return __ffma2_rn(activation, weight, sum);
                            };
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads / 2; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }
                        }

                        const auto sum = __fadd2_rn(sum_0, sum_1);
                        const reduce_dtype_t reduced = sum.x + sum.y;
                        auto result = static_cast<logits_dtype_t>(kIsMXSF ? reduced : reduced * scale_kv);
                        const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = kv_offset - seq_k_start[i];
                            const uint32_t len = seq_k_end[i] - seq_k_start[i];
                            if (rel_kv < len)
                                logits[q_offset + rel_kv] = result;
                        } else {
                            logits[q_offset + kv_offset] = result;
                        }
                    }
                }
            };

            if constexpr (decltype(scheduler)::kHasPartialBlock)
                dispatch_num_block_tokens<BLOCK_Q>(scheduler.get_num_block_tokens(q_block_idx), process_q_block);
            else
                process_q_block(cute::Int<BLOCK_Q>{});

            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

// Unified contiguous-KV entry for FP8 / MXFP4 / MXFP8.
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                      const uint32_t logits_stride,
                      const uint32_t* cu_seq_len_k_start,
                      const uint32_t* cu_seq_len_k_end,
                      logits_dtype_t* logits,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                      const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, seq_k_start, seq_k_end);
    };

    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim, kIsMXSF, kIsCompressedLogits, BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, kNumSMs,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// Unified paged entry for FP8 / MXFP4 / MXFP8.
// Paged scheduler walks (Q-block, chunk) tasks; BLOCK_Q = 128 / kNumHeads
template <uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          bool kIsMXSF, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_paged_mqa_logits(const uint32_t num_q_tokens_total,
                            const uint32_t logits_stride, const uint32_t block_table_stride,
                            const uint32_t* context_lens, logits_dtype_t* logits,
                            const uint32_t* block_table, const uint32_t* indices,
                            const uint32_t* schedule_meta,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit, "Invalid split/page size");

    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* /*seq_k_start*/, uint32_t* /*seq_k_end*/) {
        return sched::SM100PagedMQALogitsScheduler<kTokensPerRequest, kIsContextLens2D, kIsVarlen,
                                                   kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
            sm_idx, context_lens, schedule_meta, indices,
            block_table, block_table_stride, num_q_tokens_total);
    };

    // Paged uses `kNumSMs = 0`; schedule meta drives the grid stride
    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim, kIsMXSF, false, BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages, 0,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t, logits_dtype_t,
                               reduce_dtype_t, decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// Wuda entry: DeepGEMM's paged FP8 attention plus an independent compressor
// warpgroup. The attention core is unchanged apart from its 128-register
// reduction form and the role-scoped initialization barrier above.
template <uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumTailThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kHasQuery = false, bool kWaitPrimary = false,
          bool kQueryAblation = false>
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads + kNumTailThreads, 1)
void sm100_fp8_paged_mqa_logits_fused(
    const uint32_t num_q_tokens_total,
    const uint32_t logits_stride, const uint32_t block_table_stride,
    const uint32_t* context_lens, logits_dtype_t* logits,
    const uint32_t* block_table, const uint32_t* indices,
    const uint32_t* schedule_meta,
    const wuda_fp8_mqa::MainCompressorArgs compressor,
    const float compressor_eps,
    const wuda_fp8_mqa::QueryRmsRopeArgs query,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    static constexpr uint32_t kBlockQ = 128 / kNumHeads;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t kFirstTailWarp =
        kNumMathWarpGroups * 4 + kNumSpecializedThreads / 32;
    DG_STATIC_ASSERT(kNumTailThreads == 128,
                     "MAIN compressor requires four tail warps");
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit,
                     "Invalid split/page size");

    __shared__ uint32_t query_next;
    if constexpr (kHasQuery) {
        if (threadIdx.x == 0)
            query_next = 0;
        __syncthreads();
    }

    const uint32_t warp = cutlass::canonical_warp_idx_sync();
    const uint32_t lane = ptx::get_lane_idx();
    if (warp >= kFirstTailWarp) {
        const uint32_t group = warp - kFirstTailWarp;
        if (compressor.state != nullptr) {
            // DeepGEMM assigns short schedules from low to high SM indices.
            // Reverse the independent compressor mapping so short decode steps
            // use otherwise-idle CTAs instead of contending with attention.
            for (uint32_t row = gridDim.x - 1 - blockIdx.x;
                 row < compressor.seq_len;
                 row += gridDim.x) {
                const long long position = compressor.pos[row];
                if (((position + 1) & 3) == 0)
                    wuda_fp8_mqa::run_main_compressor_row(
                        compressor, row, position, group, lane,
                        compressor_eps, /*barrier_id=*/2);
            }
        }
        if constexpr (kHasQuery) {
            if constexpr (kWaitPrimary)
                cudaGridDependencySynchronize();
            wuda_fp8_mqa::run_query_rms_rope<kQueryAblation>(
                query, &query_next, lane);
        }
        return;
    }

    const auto make_scheduler = [&](const uint32_t& sm_idx,
                                    uint32_t*, uint32_t*) {
        return sched::SM100PagedMQALogitsScheduler<
            kTokensPerRequest, kIsContextLens2D, kIsVarlen,
            kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
                sm_idx, context_lens, schedule_meta, indices,
                block_table, block_table_stride, num_q_tokens_total);
    };

    sm100_mqa_logits_core_impl<
        kNumHeads, kHeadDim,
        false, false, kBlockQ, SPLIT_KV,
        kNumQStages, kNumKVStages, 0,
        kNumSpecializedThreads, kNumMathThreads,
        cutlass::float_e4m3_t, logits_dtype_t, float,
        decltype(make_scheduler), kNumMathWarpGroups, kWaitPrimary>(
            logits_stride, logits,
            tensor_map_q, tensor_map_sf_q, tensor_map_kv,
            tensor_map_sf_kv, tensor_map_weights, make_scheduler);

    if constexpr (kHasQuery) {
        // Attention warps that finished their primary role join the query
        // queue in every geometry. The TP2 gather was previously excluded and
        // left all 8192 rows to the four tail warps, which exposed the tail
        // as the short-context critical path.
        wuda_fp8_mqa::run_query_rms_rope<kQueryAblation>(
            query, &query_next, lane);
    }
}

} // namespace deep_gemm
