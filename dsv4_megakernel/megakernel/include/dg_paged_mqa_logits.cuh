#pragma once
// ============================================================
// dg_paged_mqa_logits.cuh — DeepGEMM's SM100 paged MQA-logits, vendored for the
// DSV4 decode path and PRUNED to this build's single instantiation
// (paged / mxfp4 / fp32 reduce / non-varlen / 2D context_lens / raw logits):
// layout::MQALogitsSharedStorage,
// sched::{RequestInfo, sm100_paged_mqa_logits_metadata, SM100PagedMQALogitsScheduler},
// RingPipeline / dispatch_num_block_tokens / sm100_mqa_logits_core_impl and the
// sm100_paged_mqa_logits entry. The helper closure (math/utils/ptx/tma/mma) is
// shared with mqa_logits_fp4.cuh (itself vendored from the same DeepGEMM tree).
//
// Edits vs upstream, marked `// [MEGAKERNEL EDIT]` (everything kept is otherwise
// byte-identical to DeepGEMM @ workspace snapshot):
//   1. metadata kernel: `kNumSMs` template param -> runtime `num_sms` argument
//      (JIT baked it; AOT cannot).
//   2. `cudaGridDependencySynchronize()` neutralized (PDL-only; standalone launch).
//   3. fused MAIN-compressor tail warpgroup (TPB=512): math register diet
//      (224 -> 128, bit-exact) + prologue __syncthreads -> role-scoped NamedBarrier.
//   4. dead template branches REMOVED (varlen, fp8 per-KV scale, bf16 reduce,
//      contiguous-KV producer, compressed-logits store) — none are instantiated
//      here; restore from upstream if ever needed.
//
// Host launcher + PyTorch binding: kernels/mqa_logits_fp4.cu
// ============================================================

#include "mqa_logits_fp4.cuh"   // shared vendored helper closure + compressor

// ============================================================
// inlined from deep_gemm/layout/mqa_logits.cuh (verbatim)
// ============================================================

namespace deep_gemm::layout {

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTmemStages,
          typename qk_dtype_t, typename reduce_dtype_t = float>
struct MQALogitsSharedStorage {
    static constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using sf_dtype_t = cute::conditional_t<kIsMXSF, uint32_t, float>;

    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t kQKBytesPerElem = sizeof(qk_dtype_t);
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    // Align to one 8-row Q/K swizzle tile: FP4 uses head_dim / 2 bytes, FP8 use head_dim bytes per token
    static constexpr uint32_t kSwizzleAlignment = 8 * kNumQKBytesPerToken;
    static constexpr uint32_t kNumSFQ = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumQBytesPerStage = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t kNumKVBytesPerStage = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t kNumQElementsPerStage = kNumQBytesPerStage / kQKBytesPerElem;
    static constexpr uint32_t kNumKVElementsPerStage = kNumKVBytesPerStage / kQKBytesPerElem;
    // MX SF formats store per-block scale factors; FP8 stores one per-KV scale and no Q scale
    static constexpr uint32_t kNumScaleQ = kIsMXSF ? kNumSFQ : 1;
    static constexpr uint32_t kNumScaleKV = kIsMXSF ? kNumSFKV : SPLIT_KV;
    // TMA destinations in shared memory must be 128-byte aligned.
    static constexpr uint32_t kTmaAlignment = 128;

    DG_STATIC_ASSERT(kNumQBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kNumKVBytesPerStage % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(kSwizzleAlignment % 128 == 0, "TMA destination must be 128-byte aligned");
    DG_STATIC_ASSERT(kTmaAlignment % 128 == 0, "TMA destination must be 128-byte aligned");

    alignas(kSwizzleAlignment) qk_dtype_t smem_q[kNumQStages][kNumQElementsPerStage];
    alignas(kSwizzleAlignment) qk_dtype_t smem_kv[kNumKVStages][kNumKVElementsPerStage];
    alignas(kTmaAlignment) sf_dtype_t smem_sf_q[kNumQStages][kNumScaleQ];
    alignas(kTmaAlignment) sf_dtype_t smem_sf_kv[kNumKVStages][kNumScaleKV];
    alignas(kTmaAlignment) reduce_dtype_t smem_weights[kNumQStages][BLOCK_Q * kNumHeads];
    // Barriers require 8-byte alignment, already guaranteed by the preceding TMA-aligned arrays.
    Barrier full_q_barriers[kNumQStages];
    Barrier empty_q_barriers[kNumQStages];
    Barrier full_kv_barriers[kNumKVStages];
    Barrier empty_kv_barriers[kNumKVStages];
    Barrier full_tmem_barriers[kNumTmemStages];
    Barrier empty_tmem_barriers[kNumTmemStages];
    uint32_t tmem_ptr_in_smem;
};

} // namespace deep_gemm::layout

// ============================================================
// inlined from deep_gemm/scheduler/sm100_paged_mqa_logits.cuh ([MEGAKERNEL EDIT]:
// metadata `kNumSMs` template param -> runtime argument; varlen and 1D
// context_lens branches removed — non-varlen / 2D only)
// ============================================================

// SM100 paged scheduler: metadata emits per-SM (q_token_idx, kv_split_idx) starts
// Device traversal walks chunk-outer / Q-block-inner tasks

namespace deep_gemm::sched {

// Per-request geometry accessor (non-varlen, 2D context_lens: next_n tokens per
// request, context length taken at the request's LAST token)
template <uint32_t kNextN, uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV>
struct RequestInfo {
    uint32_t q_token_start;       // request_q_token_start
    uint32_t num_q_tokens;        // request_num_q_tokens
    uint32_t num_q_blocks;        // request_num_q_blocks
    uint32_t num_kv_splits;       // request_num_kv_splits  = ceil(context_len / SPLIT_KV)
    uint32_t num_kv_pages;        // = ceil(context_len / PAGE_KV); page-level bound for the last partial split

    // Resolve the request that starts at `q_token_idx`
    CUTLASS_DEVICE static RequestInfo from_q_token(const uint32_t& q_token_idx,
                                                   const uint32_t* context_lens) {
        RequestInfo info;
        info.q_token_start = q_token_idx;
        // Regular grid: request = q_token_idx / next_n, next_n tokens each
        const uint32_t request_id = q_token_idx / kNextN;
        info.num_q_tokens = kNextN;
        const uint32_t lens_idx = request_id * kNextN + kNextN - 1;   // 2D context_lens
        const uint32_t context_len = context_lens[lens_idx];
        info.num_q_blocks = math::ceil_div(info.num_q_tokens, BLOCK_Q);
        info.num_kv_splits = math::ceil_div(context_len, SPLIT_KV);
        info.num_kv_pages = math::ceil_div(context_len, PAGE_KV);
        return info;
    }

    // Average q-token partition across Q-blocks; returns both offset and count
    CUTLASS_DEVICE void get_q_block_span(const uint32_t& q, uint32_t& token_offset, uint32_t& num_tokens) const {
        const uint32_t base = num_q_tokens / num_q_blocks, rem = num_q_tokens % num_q_blocks;
        token_offset = q * base + (q < rem ? q : rem);
        num_tokens = base + (q < rem ? 1 : 0);
    }

    // block_table row for this request
    CUTLASS_DEVICE uint32_t get_block_table_row() const {
        return q_token_start / kNextN;
    }
};

// Metadata kernel balances work across SMs via a prefix sum over request work
// [MEGAKERNEL EDIT] `kNumSMs` was a template param (JIT-baked); AOT passes it at runtime.
template <uint32_t kNextN, uint32_t BLOCK_Q, uint32_t SPLIT_KV>
CUTLASS_GLOBAL __launch_bounds__(256, 1)
void sm100_paged_mqa_logits_metadata(const uint32_t num_requests,
                                     const uint32_t num_q_tokens_total,
                                     const uint32_t num_sms,
                                     const uint32_t* context_lens,
                                     uint32_t* schedule_meta) {
    // PAGE_KV is unused for metadata; pass SPLIT_KV as a placeholder
    using Info = RequestInfo<kNextN, BLOCK_Q, SPLIT_KV, SPLIT_KV>;

    // [MEGAKERNEL EDIT] PDL-only; neutralized for standalone launch.
    // cudaGridDependencySynchronize();  // wait for the primary kernel (CDP launch)

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t num_threads = blockDim.x;

    // smem: per-request work prefix sum + request start token
    extern __shared__ uint32_t smem[];
    uint32_t* prefix_work = smem;                       // [num_requests]
    uint32_t* request_q_token_start = smem + num_requests;  // [num_requests]

    const uint32_t num_logical_requests = num_requests;
    for (uint32_t r = thread_idx; r < num_logical_requests; r += num_threads)
        request_q_token_start[r] = r * kNextN;
    __syncthreads();

    // Work per request before prefix sum
    for (uint32_t r = thread_idx; r < num_logical_requests; r += num_threads) {
        const auto info = Info::from_q_token(request_q_token_start[r], context_lens);
        prefix_work[r] = info.num_kv_splits * info.num_q_tokens;
    }
    __syncthreads();

    // Inclusive prefix sum by one warp
    if (warp_idx == 0) {
        uint32_t carry = 0;
        for (uint32_t base = 0; base < num_logical_requests; base += 32) {
            const uint32_t r = base + lane_idx;
            const uint32_t v = (r < num_logical_requests) ? prefix_work[r] : 0u;
            const uint32_t scanned = math::warp_inclusive_sum(v, lane_idx) + carry;
            if (r < num_logical_requests)
                prefix_work[r] = scanned;
            carry = __shfl_sync(0xffffffff, scanned, 31);
        }
    }
    __syncthreads();

    const uint32_t num_total_work = num_logical_requests > 0 ? prefix_work[num_logical_requests - 1] : 0u;

    // Each thread emits one SM start; remainder is assigned to earlier SMs
    const uint32_t q = num_total_work / num_sms, rem = num_total_work % num_sms;
    for (uint32_t sm_idx = thread_idx; sm_idx <= num_sms; sm_idx += num_threads) {
        const uint32_t w = sm_idx * q + (sm_idx < rem ? sm_idx : rem);
        // First request whose prefix_work owns work unit `w`
        uint32_t lo = 0, hi = num_logical_requests;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (prefix_work[mid] <= w) lo = mid + 1; else hi = mid;
        }
        const uint32_t request_idx = lo;
        uint32_t q_token_idx, kv_split_idx;
        if (request_idx < num_logical_requests) {
            const uint32_t work_before = (request_idx == 0) ? 0u : prefix_work[request_idx - 1];
            const uint32_t w_in_request = w - work_before;
            const auto info = Info::from_q_token(request_q_token_start[request_idx], context_lens);
            // Align SM starts to request/split boundaries
            q_token_idx = info.q_token_start;
            kv_split_idx = w_in_request / info.num_q_tokens;
        } else {
            // Tail sentinel: one-past-the-end
            q_token_idx = num_q_tokens_total;
            kv_split_idx = 0;
        }
        schedule_meta[sm_idx * 2] = q_token_idx;
        schedule_meta[sm_idx * 2 + 1] = kv_split_idx;
    }
}

// Device scheduler walks this SM's schedule range and implements SchedulerConcept
// All specialized warps instantiate it and advance through the same task sequence
template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kSplitsPerChunk>
struct SM100PagedMQALogitsScheduler {
    // SchedulerConcept descriptors
    static constexpr bool kIsPaged = true;
    static constexpr bool kHasPartialBlock = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;

    using Info = RequestInfo<kNextN, BLOCK_Q, SPLIT_KV, PAGE_KV>;

    const uint32_t* context_lens;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t num_q_tokens_total;

    // Walk state
    Info cur;                           // current request geometry
    uint32_t cur_kv_split_base;         // current chunk start (request-internal split)
    uint32_t cur_q_block_in_request;    // current Q-block within the request
    uint32_t end_q_token_idx, end_kv_split_idx;
    bool done;

    // Geometry stashed by `next_q_block` for the accessors below
    uint32_t cur_block_table_row;       // request's block-table row
    uint32_t cur_q_block_token_base;    // global first-token row of this Q-block
    uint32_t cur_num_block_tokens;      // valid tokens in this Q-block
    uint32_t cur_request_num_kv_pages;  // ceil(context_len / PAGE_KV); bound for last partial split

    CUTLASS_DEVICE SM100PagedMQALogitsScheduler(const uint32_t& sm_idx,
                                                const uint32_t* context_lens,
                                                const uint32_t* schedule_meta,
                                                const uint32_t* block_table,
                                                const uint32_t& block_table_stride,
                                                const uint32_t& num_q_tokens_total) {
        this->context_lens = context_lens;
        this->block_table = block_table;
        this->block_table_stride = block_table_stride;
        this->num_q_tokens_total = num_q_tokens_total;

        const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;

        cur_kv_split_base = start.y;
        cur_q_block_in_request = 0;
        done = (start.x >= num_q_tokens_total) or
               (start.x == end_q_token_idx and start.y >= end_kv_split_idx);
        if (not done)
            cur = Info::from_q_token(start.x, context_lens);

        cur_block_table_row = 0;
        cur_q_block_token_base = 0;
        cur_num_block_tokens = 1;
        cur_request_num_kv_pages = 0;
    }

    // Exclusive split bound for the current request, clamped at the next SM start
    CUTLASS_DEVICE uint32_t get_cur_kv_split_upper() const {
        return (cur.q_token_start == end_q_token_idx) ? end_kv_split_idx : cur.num_kv_splits;
    }

    // Emit the next (Q-block, chunk) task and stash its addressing geometry
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits) {
        q_block_idx = 0;  // addressing uses stashed state
        if (done)
            return false;

        // Capture emitted task geometry before advancing state
        const uint32_t upper = get_cur_kv_split_upper();
        cur_block_table_row = cur.get_block_table_row();
        uint32_t q_block_token_offset, q_block_num_tokens;
        cur.get_q_block_span(cur_q_block_in_request, q_block_token_offset, q_block_num_tokens);
        cur_q_block_token_base = cur.q_token_start + q_block_token_offset;
        cur_num_block_tokens = q_block_num_tokens;
        cur_request_num_kv_pages = cur.num_kv_pages;
        kv_split_base = cur_kv_split_base;
        const uint32_t remaining = upper - cur_kv_split_base;   // upper > cur_kv_split_base (guarded by `done`)
        num_kv_splits = (cur.num_q_blocks == 1) ? remaining
                                               : (remaining < kSplitsPerChunk ? remaining : kSplitsPerChunk);

        // Advance in Q-block, chunk, request order
        ++ cur_q_block_in_request;
        if (cur_q_block_in_request == cur.num_q_blocks) {
            cur_q_block_in_request = 0;
            cur_kv_split_base += num_kv_splits;
            if (cur_kv_split_base >= upper) {
                if (cur.q_token_start == end_q_token_idx) {
                    // Reached the next SM's start
                    done = true;
                } else {
                    // Move to next request owned from split 0
                    const uint32_t next_q_token = cur.q_token_start + cur.num_q_tokens;
                    cur_kv_split_base = 0;
                    if (next_q_token >= num_q_tokens_total)
                        done = true;
                    else {
                        cur = Info::from_q_token(next_q_token, context_lens);
                        // The new request may already be this SM's end
                        if (cur.q_token_start == end_q_token_idx and end_kv_split_idx == 0)
                            done = true;
                    }
                }
            }
        }
        return true;
    }

    CUTLASS_DEVICE uint32_t get_num_block_tokens(const uint32_t&) const {
        return cur_num_block_tokens;
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t&) const {
        return cur_q_block_token_base;
    }

    CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
        if (page_offset >= cur_request_num_kv_pages)
            return 0;
        const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
        return block_table[block_table_offset + page_offset];
    }

    CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t&, const uint32_t& token_idx) const {
        return cur_q_block_token_base + token_idx;
    }

    CUTLASS_DEVICE uint32_t get_logits_col(const uint32_t& kv_split_base,
                                           const uint32_t& kv_split_idx,
                                           const uint32_t& math_thread_idx) const {
        return (kv_split_base + kv_split_idx) * SPLIT_KV + math_thread_idx;
    }
};

} // namespace deep_gemm::sched

// ============================================================
// inlined from deep_gemm/impls/sm100_mqa_logits.cuh ([MEGAKERNEL EDIT]:
// cudaGridDependencySynchronize neutralized; PRUNED to the mxfp4 paged fp32
// instantiation — the contiguous-KV entry/producer, the fp8 per-KV-scale path,
// the bf16 reduce path and the compressed-logits store are removed)
// ============================================================

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
// (mxfp4 block-scale only; logits stored RAW at absolute kv columns)
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
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
    static constexpr uint32_t kNumSFQ  = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kRealNumSFQ * sizeof(int);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kNumSFKV * sizeof(int);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, /*kIsMXSF=*/true, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t, float>;
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
            smem.empty_kv_barriers[i].init(1);
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
    // [MEGAKERNEL EDIT] prologue publish for the ATTENTION threads only (was
    // __syncthreads): with the fused tail warpgroup (TPB 384 -> 512) a block-wide
    // barrier would deadlock -- the tail never enters this core and must not be
    // waited on. NamedBarrier id 1; id 0 is the math epilogue, id 2 the compressor.
    cutlass::arch::NamedBarrier(kNumSpecializedThreads + kNumMathThreads, 1).sync();

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    constexpr uint32_t kNumSpecializedRegisters = 56;
    // [MEGAKERNEL EDIT] 224 -> 128 (register diet, see the math role below): the
    // fused tail warpgroup (TPB=512) leaves no budget for 224-reg math warpgroups
    // (128*56 + 256*224 = 64512 of 65536). setmaxnreg targets must respect the
    // per-CTA file; the diet keeps the fp32 reduce BIT-EXACT.
    constexpr uint32_t kNumMathRegisters = 128;

    // [MEGAKERNEL EDIT] PDL-only; neutralized for standalone launch.
    // cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx);
            // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kNumQKBytesPerToken, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
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

        auto scheduler = make_scheduler(sm_idx);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

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

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();

                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

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

                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(&smem.full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                __syncwarp();
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

        DG_STATIC_ASSERT(kNumHeads == 4 or kNumHeads == 8 or kNumHeads == 16 or kNumHeads == 32 or kNumHeads == 64,
                         "Unsupported TMEM load size");
        // [MEGAKERNEL EDIT] register diet (enables the fused tail): no
        // register-cached weights (read as float2 straight from smem -- the Q
        // stage stays valid until the empty_q arrive) and TMEM consumed in two
        // half-head passes reusing accum[kNumHeads/2].
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

        auto scheduler = make_scheduler(sm_idx);
        // NOTES: split index for paged scheduler, token offset for contiguous-KV scheduler.
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        float reduced;
                        {
                            // [MEGAKERNEL EDIT] fp32 register diet: TMEM in two
                            // kNumHeads/2 passes reusing the same accum registers;
                            // weights read as float2 from smem. Per-(sum_0, sum_1)
                            // chain the fp32 accumulation order is IDENTICAL to the
                            // upstream single-pass form -> bit-exact results.
                            const auto w2 = reinterpret_cast<const float2*>(
                                smem.smem_weights[q_stage_idx] + i * kNumHeads);
                            auto sum_0 = make_float2(0, 0);
                            auto sum_1 = make_float2(0, 0);
                            #pragma unroll
                            for (uint32_t half = 0; half < 2; ++ half) {
                                tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + half * (kNumHeads / 2), accum);
                                cutlass::arch::fence_view_async_tmem_load();

                                // Release TMEM only after ALL reads of this stage
                                if (half == 1 and i == kNumValidTokens - 1) {
                                    ptx::tcgen05_before_thread_sync();
                                    smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                                }

                                const uint32_t jb = half * (kNumHeads / 2);
                                const auto transform = [&](const uint32_t& j, const float2& sum) {
                                    auto a_0 = make_float2(accum[j], accum[j + 1]);
                                    auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                                    const auto b = ptx::ld_shared(w2 + ((jb + j) >> 1));
                                    return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
                                };

                                #pragma unroll
                                for (uint32_t j = 0; j < kNumHeads / 2; j += 4) {
                                    sum_0 = transform(j, sum_0);
                                    sum_1 = transform(j + 2, sum_1);
                                }
                            }

                            auto sum = __fadd2_rn(sum_0, sum_1);
                            reduced = (sum.x + sum.y) / 2;
                        }
                        auto result = static_cast<logits_dtype_t>(reduced);
                        const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                        logits[q_offset + kv_offset] = result;
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

// Paged entry (mxfp4 / non-varlen / 2D context_lens only).
// Paged scheduler walks (Q-block, chunk) tasks; BLOCK_Q = 128 / kNumHeads
template <uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          // [MEGAKERNEL EDIT] CUDA-core tail warpgroup (0 disables the branch entirely)
          uint32_t kNumTailThreads,
          typename qk_dtype_t, typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads + kNumTailThreads, 1)
void sm100_paged_mqa_logits(const uint32_t num_q_tokens_total,
                            const uint32_t logits_stride, const uint32_t block_table_stride,
                            const uint32_t* context_lens, logits_dtype_t* logits,
                            const uint32_t* block_table,
                            const uint32_t* schedule_meta,
                            // [MEGAKERNEL EDIT] fused MAIN-indexer compressor (tail
                            // warpgroup; see MainCompressorArgs). comp.kv == nullptr
                            // -> the tail exits immediately.
                            const float comp_eps,
                            const MainCompressorArgs comp,
                            // [MEGAKERNEL EDIT] benchmark tail_us: true -> the 384
                            // attention threads exit at entry, leaving the tail
                            // warpgroup running ALONE in its in-situ launch shape.
                            const bool attn_mock,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit, "Invalid split/page size");

    // [MEGAKERNEL EDIT] CUDA-core tail warpgroup (warps 12-15), split off BEFORE the
    // core impl -- gemm_fuse_norm_b discipline: the attention path and the tail
    // NEVER share a barrier (the core's prologue publish is a role-scoped
    // NamedBarrier(spec+math, 1), not __syncthreads). The tail touches no smem /
    // mbarrier / TMEM state, so it starts the instant the CTA does.
    // Work: MAIN-indexer compressor rows (all FOUR tail warps cooperate on ONE row,
    // one 128-col group per warp; rows stride by CTA).
    if constexpr (kNumTailThreads > 0) {
        DG_STATIC_ASSERT(kNumTailThreads == 128, "coop compressor assumes 4 tail warps");
        const auto warp_idx = cutlass::canonical_warp_idx_sync();
        constexpr uint32_t kTailWarpStart = (kNumSpecializedThreads + kNumMathThreads) / 32;
        if (warp_idx >= kTailWarpStart) {
            if (comp.kv != nullptr) {
                const uint32_t lane_idx = ptx::get_lane_idx();
                const uint32_t tail_warp = warp_idx - kTailWarpStart;
                for (uint32_t m = blockIdx.x; m < num_q_tokens_total; m += gridDim.x) {
                    const long long p = comp.pos[m];
                    if (((p + 1) & 3) != 0)
                        continue;                              // not a compress row
                    run_main_compressor_row(comp, m, p, tail_warp, lane_idx, comp_eps,
                                            /*barrier_id=*/2);
                }
            }
            return;
        }
        // attention mocked out for tail_us: the remaining 384 threads leave before
        // any prologue state (mbarrier init / TMEM alloc), tail runs alone in situ.
        if (attn_mock)
            return;
    }

    const auto make_scheduler = [&](const uint32_t& sm_idx) {
        return sched::SM100PagedMQALogitsScheduler<kTokensPerRequest,
                                                   kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
            sm_idx, context_lens, schedule_meta,
            block_table, block_table_stride, num_q_tokens_total);
    };

    // Schedule meta drives the grid stride
    sm100_mqa_logits_core_impl<kNumHeads, kHeadDim, BLOCK_Q, SPLIT_KV,
                               kNumQStages, kNumKVStages,
                               kNumSpecializedThreads, kNumMathThreads, qk_dtype_t, logits_dtype_t,
                               decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

} // namespace deep_gemm

// ============================================================
// DSV4 fixed configuration for the DG paged entry (decode-only, RTP-LLM shapes):
// H=64, D=128, mxfp4, PAGE_KV=64, SPLIT_KV=256, next_n=1, context_lens 2D,
// non-varlen; DeepGEMM host defaults: q/kv stages 3/10 (fp4), splits_per_chunk 16,
// 128 specialized + 256 math threads, BLOCK_Q = 128/64 = 2; plus the DSV4 fused
// MAIN-compressor tail warpgroup (128 threads -> TPB=512).
// ============================================================
namespace dg_paged_mqa_logits {
static constexpr int NUM_HEADS = 64;
static constexpr int HEAD_DIM  = 128;
static constexpr int PAGE_KV   = 64;
static constexpr int SPLIT_KV  = 256;
static constexpr int NEXT_N    = 1;
static constexpr int NUM_Q_STAGES  = 3;
static constexpr int NUM_KV_STAGES = 10;   // DeepGEMM: is_fp4 ? 10 : 5
static constexpr int SPLITS_PER_CHUNK = 16;
static constexpr int NUM_SPECIALIZED_THREADS = 128;
static constexpr int NUM_MATH_THREADS        = 2 * 128;
// CUDA-core tail warpgroup (warps 12-15): hides the MAIN-indexer compressor rows
// under the KV stream; idle when comp.kv == nullptr. TPB=512 caps the architectural
// register budget at 65536/512 = 128 -- the math register diet (224 -> 128 in the
// core impl, bit-exact) is the prerequisite.
static constexpr int NUM_TAIL_THREADS        = 128;
static constexpr int TPB = NUM_SPECIALIZED_THREADS + NUM_MATH_THREADS + NUM_TAIL_THREADS;  // 512
static constexpr int BLOCK_Q = 128 / NUM_HEADS;                          // 2

using SharedStorage = deep_gemm::layout::MQALogitsSharedStorage<
    NUM_HEADS, HEAD_DIM, /*kIsMXSF=*/true, BLOCK_Q, SPLIT_KV,
    NUM_Q_STAGES, NUM_KV_STAGES, /*kNumTmemStages=*/3, cutlass::float_e2m1_t, float>;
}  // namespace dg_paged_mqa_logits
