#pragma once
// ============================================================
// wq_b_fp8_gemm.cuh — config + PTX/TMA helpers for the MERGED wq_b projection
// (kernel/host/usage: kernels/wq_b_fp8_gemm.cu).
//
// Shape: x[M,1536] e4m3 @ w[N_MERGED,1536]^T (main q ++ indexer), M_pad in
// {32,64,96,128}. Swap-AB: MMA A = weight (UMMA_M=256 along N), B = activation
// (UMMA_N = M_pad along M), both K-major 128B-swizzled. Native DSV4 scales:
// activation 1x128, weight 128x128, UE8M0; warp2 expands each K128 byte to four
// K32 scale IDs (tcgen05 mxf8f6f4 granularity). Block-scale MMA engine lives in
// cluster_mma_fp8.cuh.
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// CUTLASS/CuTe headers
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>

// ======================== Configuration ========================
namespace wq_b {

// ---- Problem dimensions (fixed for wq_b projection) ----
static constexpr int K_DIM    = 1536;
#if defined(WQ_B_TPDP)
// TPDP keeps all index-Q heads replicated while sharding main-Q heads over TP2.
static constexpr int N_TOTAL  = 32768;  // 64 main-Q heads x 512 dim
#else
static constexpr int N_TOTAL  = 65536;  // 128 main-Q heads x 512 dim
#endif
// ---- Output head geometry (per-head RMSNorm scale folding) ----
// The epilogue can ACCUMULATE per-(row, head) sum-of-squares into an optional
// zero-initialized head_ssq[M, NUM_HEADS_OUT] fp32 buffer (fire-and-forget f32
// RED atomics; each 128-col N block lies inside ONE 512-col head). The consumer
// finalizes s_h = rsqrt(ssq/512 + eps) and folds it into its own q pass -- the
// normalized y never round-trips HBM and TMEM never has to hold a whole head.
static constexpr int HEAD_DIM_OUT  = 512;
static constexpr int NUM_HEADS_OUT = N_TOTAL / HEAD_DIM_OUT;
static constexpr int BLOCK_K  = 128;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K; // 12
static constexpr int QUANT_BLOCK_K = 128; // native DSV4 FP8 quantization granularity
static constexpr int UMMA_SF_GRAN_K = 32; // hardware block-scale consumption granularity
static constexpr int WEIGHT_QUANT_BLOCK_N = 128;
static constexpr int NUM_WEIGHT_SF_ROWS =
    (N_TOTAL + WEIGHT_QUANT_BLOCK_N - 1) / WEIGHT_QUANT_BLOCK_N;

// ---- Merged indexer projection (CSA stage 7 Idx_WProj fused into this GEMM) ----
// The indexer wq_b shares A (= qr) and K, so its weight is CONCATENATED along N.
// It goes FIRST: w[N_IDX + N_TOTAL, K], i.e. rows [0, N_IDX) are the indexer and
// [N_IDX, N_MERGED) are the main q. Leading it means a plain forward tile walk
// already schedules the tiles that need CUDA-core post-processing in iteration 0,
// so the remaining main-q weight stream is their shadow -- no reversed schedule.
// The iq segment is tile-aligned for the swap path (N_IDX % CLUSTER_BLOCK_N == 0)
// and each 128-col CTA tile is exactly ONE indexer head row -- the epilogue drains
// it to a BF16 L2 scratch (rope + hadamard + fp4 quant run async). Swap path
// (M <= 128) only.
static constexpr int IDX_NUM_HEADS = 64;
static constexpr int IDX_HEAD_DIM  = 128;
static constexpr int N_IDX         = IDX_NUM_HEADS * IDX_HEAD_DIM;              // 8192
static constexpr int N_MERGED      = N_TOTAL + N_IDX;
// Index-tile CTAs release per-head drain flags; every CTA pulls transform row
// batches from a flat task space. Each flag occupies its own cache line and uses
// a launch tag, avoiding false sharing and per-launch clearing.
static constexpr int IQ_FLAG_STRIDE = 32;                                // 128B / u32
static constexpr int IQ_FLAG_SLOTS  = IDX_NUM_HEADS * IQ_FLAG_STRIDE;

// Every spin in the iq handoff is bounded, because an unbounded one wedges the
// whole process and says nothing: the __syncthreads that could not be reached by
// the sibling warp roles, the `ready` page pointed at ourselves instead of the
// peer, and the host tag frozen into a CUDA graph all presented identically as a
// hang. DeepGEMM's nvlink_barrier does the same (60s, then printf + assert).
// %globaltimer ticks ns, so this is 10s: far past any legitimate peer skew.
static constexpr unsigned long long IQ_SPIN_TIMEOUT_NS = 10000000000ull;

// ---- Indexer-q DESTINATION (decoupled from what this rank COMPUTES) ----
// This rank computes IDX_NUM_HEADS heads for every problem_m token, but the
// transform stores straight into the mqa_logits input buffer, whose geometry is
// the FULL head set (`num_heads`) over the whole batch. Row m goes to whichever
// rank owns it -- [row_lo, row_hi) is ours, everything else is the peer's mirror
// -- at the SAME row index: both ranks size the buffer for the whole batch and
// leave the rows they do not own untouched, which is what a DP consumer wants
// since it only reads its own range. With peer-mapped (symmetric) memory that
// store IS the cross-rank exchange: no collective, no staging pack, no
// rank-major -> head-major repack. Corollary: iq is deliberately NOT
// bitwise-identical across ranks.
//   head_dst = head_base + local_head        (local_head in [0, IDX_NUM_HEADS))
// Single-GPU default (peer == nullptr, [row_lo, row_hi) covering every row,
// head_base == 0, num_heads == IDX_NUM_HEADS) degenerates to the plain
// [problem_m, IDX_NUM_HEADS] layout, bit-identical to the non-TP path.
//
// The completion handoff is IN this kernel, on purpose. Moving it to a 1-thread
// launch did remove it from wq_b (measured 2.8us -> 0.0 at every batch size) and
// made the end-to-end graph SLOWER by 0..7us: an extra node, and it breaks the
// PDL pair, since programmatic stream serialization couples a kernel to its
// IMMEDIATE stream predecessor. The 2.8us was already covered by the other CTAs'
// work; a launch is not.
//
// Arrive with `red`, not `atom`: red returns nothing, so a CTA never stalls on
// the round trip (`atom` cost ~7us across 144 CTAs, unchanged by spreading the
// counter over 8 cache lines -- which is how we know it was the round trip and
// not contention). Exactly one CTA then polls, like MoK's barrier_arrive/wait.
//
// `ready` is the PEER's flag page and `ready_self` ours; the publish writes BOTH,
// because under PDL the consumer starts before this rank's own SPREAD is done and
// so must wait on this rank's iq too. The published value is the device-side
// `gen` counter (++ per launch), never a host tag: a host tag freezes into a CUDA
// graph, so on replay the flag would still hold the capture-time value and the
// consumer would sail past before this replay's data landed. Consumers wait for
// `>= *gen`, which re-arms itself and needs no per-step memset.
//
// KNOWN COST, and the one worth attacking next. `gen` is published only once the
// LAST CTA has arrived, i.e. at the very end of this kernel, so mqa's Q TMA warp
// cannot use its PDL head start. Paired measurement at B=128 (self-loop: one GPU
// writing its own ready_self, so a real fence/publish/wait path but no inter-rank
// skew):
//   no crossing, mqa without PDL      span 72.0   (baseline)
//   no crossing, mqa with PDL         span 69.1   (-2.9 -- the head start is real)
//   full crossing, mqa waits          span 74.5   (+2.5 vs baseline)
// wq_b itself grew 3.6us and only 1.1us of that hid under mqa's KV prefetch, so
// 2.5us is exposed -- and the 2.9us of PDL overlap is forfeited on top, since the
// Q warp now blocks until the last arrival. Measured against the PDL-but-no-
// crossing pipeline the crossing therefore costs 5.4us, not 2.5.
// The lever is GRANULARITY, not a cheaper fence: this kernel already keeps
// per-head drain flags for its own SPREAD handshake, so publishing per head as
// each head's rows land would let the Q warp start on head 0 while the last head
// is still in flight, instead of waiting on one all-CTAs barrier.
static constexpr int IQ_DONE_WORDS = 32;      // 128B; only word 0 is used
struct IqDest {
    uint8_t* fp4;       int* sf;           // this rank's mqa-input buffer
    uint8_t* fp4_peer;  int* sf_peer;      // peer DP rank's mirror (nullptr = none)
    const float* weights;                  // [rows, num_heads], FP8 scale fold input
    int row_lo, row_hi;                    // rows THIS rank owns; others -> peer
    int head_base;                         // global index of this rank's head 0
    int num_heads;                         // heads in the DESTINATION (row stride)
    unsigned int* done;                    // [IQ_DONE_WORDS] arrival counter
    unsigned int* ready;                   // PEER's [world] flag page
    unsigned int* ready_self;              // OUR [world] flag page
    unsigned int* gen;                     // [1] local launch generation
    int rank;                              // which ready[] slot is ours
    // Optional [2] clock64 stamps, SPREAD entry/exit from CTA 0. In-kernel
    // because no external meter resolves it: a graph-diff over a ~170us wall
    // carries +-3us, comparable to the whole quantity.
    long long* prof;
};
static constexpr int NUM_WEIGHT_SF_ROWS_MERGED =
    N_MERGED / WEIGHT_QUANT_BLOCK_N;                                            // 576

// ---- Element sizes (bytes) ----
static constexpr int FP8_ELEM_SIZE = 1;   // e4m3

// ---- Cluster / multicast (swap-AB: cluster_n = 2) ----
static constexpr int CLUSTER_SIZE  = 2;
static constexpr int NUM_MULTICAST = CLUSTER_SIZE; // 2
static constexpr bool IS_MULTICAST_ON_A = true;

// ---- MMA instruction shape (2SM) ----
// UMMA_M = LAYOUT_AD_M(128) * kNumMulticast(2) = 256 (along problem-N)
// UMMA_N = M (along problem-M; per-SwapDims template, no padding).
static constexpr int LAYOUT_AD_M = 128;
static constexpr int UMMA_M = LAYOUT_AD_M * NUM_MULTICAST; // 256
static constexpr int UMMA_K = 32;                          // FP8 block-scale UMMA K
static constexpr int BM       = 128;                       // max swap-path M (scratch sizing)

// ---- N tiling ----
static constexpr int BLOCK_N        = 128;
static constexpr int CLUSTER_BLOCK_N = BLOCK_N * NUM_MULTICAST;  // 256
static constexpr int LOAD_BLOCK_N   = BLOCK_N;                    // 128
static constexpr int NUM_N_TILES    = N_TOTAL / CLUSTER_BLOCK_N;  // 256
static constexpr int NUM_N_TILES_MERGED = N_MERGED / CLUSTER_BLOCK_N;  // 288 (swap/kIdx only)
static_assert(N_IDX % CLUSTER_BLOCK_N == 0, "iq segment must be whole cluster tiles");
static_assert(BLOCK_N == IDX_HEAD_DIM, "one CTA tile must equal one indexer head row");

// ---- Layout: both A(weight) and B(activation) are K-major ----
static constexpr auto MAJOR_A = cute::UMMA::Major::K;
static constexpr auto MAJOR_B = cute::UMMA::Major::K;

// ---- Swizzle (128B). K-major fp8: BLOCK_K*1 = 128 bytes -> 128B swizzle. ----
static constexpr int SWIZZLE_A  = 128;
static constexpr int SWIZZLE_B  = 128;
static constexpr int SWIZZLE_CD = 128;

// ---- Scale-factor (block-scale) layout ----
static constexpr int SF_IDS_PER_QUANT_BLOCK = QUANT_BLOCK_K / UMMA_SF_GRAN_K; // 4
static constexpr uint32_t UE8M0_ONE = 0x7fu;
static_assert(BLOCK_K == QUANT_BLOCK_K && SF_IDS_PER_QUANT_BLOCK == 4,
              "one pipeline stage must expand one K128 scale to four K32 IDs");
static constexpr int NUM_UTCCP_ALIGNED = 128;
static constexpr int SF_BLOCK_N        = ((BLOCK_N + NUM_UTCCP_ALIGNED - 1) / NUM_UTCCP_ALIGNED) * NUM_UTCCP_ALIGNED; // 128
static constexpr int NUM_SFB_TMEM_COLS = SF_BLOCK_N / 32;    // 4
static constexpr int SMEM_SFB_PER_STAGE = SF_BLOCK_N * (int)sizeof(uint32_t); // 512

// ---- Pipeline ----
static constexpr int NUM_EPI_STAGES      = 2;  // TMEM accumulator double buffer
static constexpr int NUM_TMA_STORE_STAGES = 2;

// ---- Threads: warp0 TMA, warp1 MMA(leader), warp2 SF expand/layout, warps4-7 epilogue ----
static constexpr int TPB                 = 256;
static constexpr int NUM_NON_EPI_THREADS = 128;
static constexpr int NUM_EPI_THREADS     = 128;
static constexpr int NUM_STORE_THREADS   = 128;
// [kIdx] async transform workers (warps 8..15): occupancy is already fixed at
// one CTA/SM by dynamic smem, so use the remaining thread slots to shorten the
// rope/hadamard/fp4 tail without reducing residency.
static constexpr int NUM_XFORM_THREADS   = 256;
static constexpr int TPB_IDX             = TPB + NUM_XFORM_THREADS;   // 512
static constexpr int XFORM_ROWS_PER_TASK = NUM_XFORM_THREADS / 8;     // 32 rows/task

// ---- Epilogue store tile (swap-AB, BF16 output) ----
static constexpr int STORE_BLOCK_M      = 16;  // M-rows per store stage
static constexpr int STORE_BLOCK_N      = BLOCK_N;
static constexpr int STORE_BLOCK_N_ATOM =
    SWIZZLE_CD / (int)sizeof(__nv_bfloat16);    // 128B / 2B = 64 columns

// ---- Per-stage SMEM for weight B (constant); A depends on M via SwapDims ----
static constexpr int SMEM_B_PER_STAGE  = LOAD_BLOCK_N * BLOCK_K * FP8_ELEM_SIZE; // 16384
static constexpr int SMEM_CD_PER_STAGE =
    STORE_BLOCK_M * STORE_BLOCK_N * sizeof(__nv_bfloat16);                 // 4096
static constexpr int SMEM_CD_TOTAL = SMEM_CD_PER_STAGE * NUM_TMA_STORE_STAGES; // 8192

// SMEM capacity budget (SM100)
static constexpr int SMEM_CAPACITY = 232448;

static constexpr int div_up(int a, int b)  { return (a + b - 1) / b; }
static constexpr int align_up(int a, int b) { return div_up(a, b) * b; }
static constexpr int num_aligned_tmem_cols(int c) {
    if (c <= 32)  return 32;
    if (c <= 64)  return 64;
    if (c <= 128) return 128;
    if (c <= 256) return 256;
    return 512;
}

// ---- Compile-time helpers parameterised on M (UMMA_N = M, no padding) ----
// One tile covers the WHOLE problem M: UMMA_N = M, LOAD_BLOCK_M = M/2 per CTA.
// Smaller M shrinks SMEM_A and the TMEM accumulator footprint, so the pipeline
// deepens automatically (M=32 -> ~11 stages, matching DeepGEMM's pick).
template <int M_> struct SwapDims {
    static constexpr int BLOCK_M          = M_;                    // true M (<= 128)
    static constexpr int NUM_M_SUB        = div_up(M_, BM);        // 1 for all swap Ms
    static constexpr int UMMA_N           = M_;                    // MMA-N = problem M
    static constexpr int LOAD_BLOCK_M     = M_ / NUM_MULTICAST;    // M/2 per CTA
    static constexpr int SMEM_A_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * FP8_ELEM_SIZE;

    // SFA stays UTCCP-aligned to 128 rows (rows >= M carry UE8M0_ONE padding;
    // the MMA only consumes the first M).
    static constexpr int SF_BLOCK_M         = align_up(M_, NUM_UTCCP_ALIGNED);        // 128
    static constexpr int SMEM_SFA_PER_STAGE = SF_BLOCK_M * (int)sizeof(uint32_t);     // 512
    static constexpr int NUM_SFA_TMEM_COLS  = SF_BLOCK_M / 32;                        // 4

    static constexpr int SMEM_PER_STAGE = SMEM_A_PER_STAGE + SMEM_B_PER_STAGE +
                                          SMEM_SFA_PER_STAGE + SMEM_SFB_PER_STAGE;

    // TMEM layout: [accum (UMMA_N*NUM_EPI_STAGES)] [SFA cols] [SFB cols]
    static constexpr int NUM_ACCUM_TMEM_COLS = UMMA_N * NUM_EPI_STAGES;               // 2M
    static constexpr int TMEM_START_SFA      = NUM_ACCUM_TMEM_COLS;
    static constexpr int TMEM_START_SFB      = NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS;
    static constexpr int NUM_TMEM_COLS = num_aligned_tmem_cols(
        NUM_ACCUM_TMEM_COLS + NUM_SFA_TMEM_COLS + NUM_SFB_TMEM_COLS);

    // Number of pipeline stages fitting the SMEM budget.
    // Overhead: smem_cd + barriers(full/empty/with_sf per stage + tmem full/empty) + tmem_ptr
    //           + the head_ssq per-warp scratch (4 x BM floats, RMSNorm scale folding).
    // (The fused indexer compressor is fully register-resident -- no smem term.)
    static constexpr int SMEM_BARRIERS = (16 * 3 + NUM_EPI_STAGES * 2) * 8;
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL + SMEM_BARRIERS + 8
                                         + 4 * BM * (int)sizeof(float);
    static constexpr int STAGES_RAW    = (SMEM_CAPACITY - SMEM_OVERHEAD) / SMEM_PER_STAGE;
    static constexpr int NUM_STAGES    = STAGES_RAW > 12 ? 12 : STAGES_RAW;
};

} // namespace wq_b

// ======================== Barrier alias ========================
namespace mma_desc {
using Barrier = cutlass::arch::ClusterTransactionBarrier;
} // namespace mma_desc

// ======================== PTX Wrappers ========================
// Block-scale MMA + descriptors live in cluster_mma_fp8.cuh. Here we keep only the
// helpers used by the producer / SF layout / epilogue / init in this kernel.
namespace ptx {

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

// TMEM load: 32dp32b, x4 (4 FP32 per lane) — used by the non-swap FP32 store
__device__ __forceinline__ void tmem_load_32dp32b4x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
        : "r"(tmem_addr));
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

// TMEM load: 16dp256b, x1 (4 FP32 per lane). Two loads, with datapaths 0/16,
// expose an 8-row x 32-col warp tile as four same-row values per lane. That
// layout cuts the SSQ warp reduction from 16 shuffles to 6 per 8 rows.
__device__ __forceinline__ void tmem_load_16dp256b1x(
    uint32_t tmem_addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
        : "r"(tmem_addr));
}
__device__ __forceinline__ void tmem_load_fence() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Shared memory load/store (used by the SF layout warp and the epilogue)
__device__ __forceinline__ uint32_t ld_shared_u32(const uint32_t* ptr) {
    uint32_t v;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}
__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t v) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(v) : "memory");
}

// Warp-cooperative BF16 transpose into a 128B-swizzled TMA store tile.
__device__ __forceinline__ void stmatrix_x4_trans(
    void* smem_ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1,%2,%3,%4};"
        :: "l"(__cvta_generic_to_shared(smem_ptr)),
           "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}
__device__ __forceinline__ void st_shared_v4_u32(void* ptr, uint32_t v0, uint32_t v1,
                                                 uint32_t v2, uint32_t v3) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ long long rdclock() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

// %globaltimer, NOT %clock64: device-wide and in NANOSECONDS, so a timeout can be
// stated in real time. clock64 is per-SM and its rate is not architectural, which
// makes it fine for the relative prof stamps above and wrong for a deadline.
__device__ __forceinline__ unsigned long long rdtimer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

} // namespace ptx

// ======================== TMA Copy Helpers ========================
namespace tma {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// Plain (per-CTA) TMA 2D load for FP8 operands. Each CTA loads its own box;
// the cta_group::2 MMA combines the pair across SMs. (Aligned with w1_merged.)
__device__ __forceinline__
void copy_2d_fp8(void const* desc_ptr, Barrier* barrier_ptr,
                 __nv_fp8_e4m3* smem_ptr, uint32_t k_idx, uint32_t mn_idx) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc_ptr,
        reinterpret_cast<uint64_t*>(barrier_ptr),
        static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
        smem_ptr, k_idx, mn_idx);
}

__device__ __forceinline__
void store_2d(void const* desc_ptr, void* smem_ptr,
              uint32_t col_idx, uint32_t row_idx) {
    cute::SM90_TMA_STORE_2D::copy(desc_ptr, smem_ptr, col_idx, row_idx);
}

} // namespace tma

// ==================== Activation rmsnorm+quant chain (PDL producer) =========
// Device chain for the STANDALONE qnorm_quant kernel launched right before the
// merged GEMM with PDL (DeepGEMM discipline: the producer is its own kernel;
// the GEMM's cudaGridDependencySynchronize -- placed AFTER its prologue --
// replaces the old in-kernel grid ticket, whose sync floor was ~1.5-2us).
namespace wq_b {

// fp8(e4m3) + native-UE8M0 quant of ONE K128 block, one WARP per block
// (byte-exact port of delivery tile::quant_k128_ue8m0): 4 bf16 elems/lane ->
// warp-shfl amax -> E=ceil(log2(amax/448)) clamped -> v*2^-E as uchar4;
// lane 0 writes the UE8M0 exponent byte. xrow is pre-offset by lane*4.
__device__ __forceinline__ void quant_k128_ue8m0(
    const __nv_bfloat16* __restrict__ xrow, int lane_id,
    __nv_fp8_e4m3* __restrict__ q_dst, uint8_t* __restrict__ sf_dst) {
    const float v0 = __bfloat162float(xrow[0]);
    const float v1 = __bfloat162float(xrow[1]);
    const float v2 = __bfloat162float(xrow[2]);
    const float v3 = __bfloat162float(xrow[3]);
    float block_amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        block_amax = fmaxf(block_amax, __shfl_xor_sync(0xffffffffu, block_amax, o));
    int scale_exp = (int)ceilf(log2f(fmaxf(block_amax, 1e-30f) / 448.0f));
    scale_exp = max(-127, min(127, scale_exp));
    const float scale_factor = exp2f(-(float)scale_exp);
    uchar4 packed;
    packed.x = __nv_fp8_e4m3(v0 * scale_factor).__x;
    packed.y = __nv_fp8_e4m3(v1 * scale_factor).__x;
    packed.z = __nv_fp8_e4m3(v2 * scale_factor).__x;
    packed.w = __nv_fp8_e4m3(v3 * scale_factor).__x;
    *reinterpret_cast<uchar4*>(q_dst + lane_id * 4) = packed;
    if (lane_id == 0)
        *sf_dst = (uint8_t)(scale_exp + 127);
}

// rmsnorm(gamma) + 1x128 quant of ONE row, QUANT-GROUP wide (warps 2..7, 192
// threads -- warp0/1 stay OFF the quant path so the weight TMA stream starts
// at t=0): the group's warps split the 12 K128 blocks (2 each); ONE memory
// wave loads v AND gamma together; block partial ssq (fixed shfl tree) ->
// ssq_smem[12] -> group barrier -> every warp sums in FIXED block order
// (bitwise identical) -> r -> qr = bf16((v*r)*gamma) materialized round ->
// quant_k128 tail. MUST be called by ALL bar_threads of barrier bar_id.
//
// The two barriers are NOT the thing to remove. Collapsing this to one warp per
// row (partials in registers, zero barriers) made d_q+norm go 2.8 -> 7.9us at
// M=128 and 18us at M=1: a warp then owns all 12 blocks, so its 12 five-step
// shfl trees serialise into ~60 dependent shuffles (and the same again for the
// amax pass), where this split runs 2 trees per warp on 6 warps at once. The
// barrier pair is cheap next to that.
__device__ __forceinline__ void qnorm_quant_row_cta(
    const __nv_bfloat16* __restrict__ yrow,      // [1536] strided row base
    const float* __restrict__ gamma,             // [1536]
    float eps, float* __restrict__ ssq_smem,     // [NUM_K_TILES] CTA scratch
    int warp_id, int lane_id, int nwarps,
    int bar_threads, int bar_id,
    __nv_fp8_e4m3* __restrict__ qf_row,          // [1536] e4m3 out
    uint8_t* __restrict__ qsf_row) {             // [12] UE8M0 out
    float v[2][4], g[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int b = warp_id + i * nwarps;
        if (b >= NUM_K_TILES) continue;
        const uint2 raw = *reinterpret_cast<const uint2*>(
            yrow + b * BLOCK_K + lane_id * 4);
        const __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const float2 f0 = __bfloat1622float2(p0);
        const float2 f1 = __bfloat1622float2(p1);
        v[i][0] = f0.x; v[i][1] = f0.y; v[i][2] = f1.x; v[i][3] = f1.y;
        const float4 gg = *reinterpret_cast<const float4*>(
            gamma + b * BLOCK_K + lane_id * 4);
        g[i][0] = gg.x; g[i][1] = gg.y; g[i][2] = gg.z; g[i][3] = gg.w;
        float ps = v[i][0] * v[i][0] + v[i][1] * v[i][1]
                 + v[i][2] * v[i][2] + v[i][3] * v[i][3];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            ps += __shfl_xor_sync(0xffffffffu, ps, o);
        if (lane_id == 0)
            ssq_smem[b] = ps;
    }
    cutlass::arch::NamedBarrier::sync(bar_threads, bar_id);   // partials visible
    float ssq = 0.f;
    #pragma unroll
    for (int j = 0; j < NUM_K_TILES; ++j)        // FIXED order -> bitwise
        ssq += ssq_smem[j];
    const float rinv = rsqrtf(ssq * (1.f / (float)K_DIM) + eps);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int b = warp_id + i * nwarps;
        if (b >= NUM_K_TILES) continue;
        float q[4], amax = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            q[j] = __bfloat162float(__float2bfloat16_rn((v[i][j] * rinv) * g[i][j]));
            amax = fmaxf(amax, fabsf(q[j]));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        int e = (int)ceilf(log2f(fmaxf(amax, 1e-30f) / 448.0f));
        e = max(-127, min(127, e));
        const float inv = exp2f(-(float)e);
        uchar4 packed;
        packed.x = __nv_fp8_e4m3(q[0] * inv).__x;
        packed.y = __nv_fp8_e4m3(q[1] * inv).__x;
        packed.z = __nv_fp8_e4m3(q[2] * inv).__x;
        packed.w = __nv_fp8_e4m3(q[3] * inv).__x;
        *reinterpret_cast<uchar4*>(
            reinterpret_cast<uint8_t*>(qf_row) + b * BLOCK_K + lane_id * 4) = packed;
        if (lane_id == 0)
            qsf_row[b] = (uint8_t)(e + 127);
    }
    cutlass::arch::NamedBarrier::sync(bar_threads, bar_id);   // ssq_smem reuse guard
}

} // namespace wq_b
