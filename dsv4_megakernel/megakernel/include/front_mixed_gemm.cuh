#pragma once
// ============================================================
// front_mixed_gemm.cuh — MIXED-PRECISION front projection GEMM (SM100/SM103).
//
//   y[M,4672] = x[M,7168] @ W^T, checkpoint-accurate precision per segment:
//     cols [0   ,2048) FP8  E4M3 (wq_a 1536 | wkv 512), weight 128x128 UE8M0
//                      scale + activation 1x128 UE8M0 scale (DeepGEMM 1d2d)
//     cols [2048,4672) BF16 (main_comp 2048 | idx_comp 512 | w_proj 64)
//
// WHY MIXED-IN-ONE-KERNEL: cuBLAS measurements show a 2-GEMM split loses to the
// all-BF16 merge (narrow-N GEMMs starve the 148-SM machine: 2048-col fp8 runs
// at ~1.7TB/s vs ~3.2TB/s merged), while an all-FP8 merge is not checkpoint-
// accurate (56% of the cols are deliberately unquantized). One kernel keeps
// the merged tile space (full parallelism) and picks the MMA per tile:
//   n_tile <  32 -> tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale
//   n_tile >= 32 -> tcgen05.mma.cta_group::2.kind::f16      (verbatim bf16 path)
// Weight bytes: 52.3MB vs 67MB all-BF16 -> expected ~1.25x on the weight-bound
// decode shapes.
//
// Derived from front_bf16_standalone_tcgen05.cuh (n64 config, non-persistent:
// one (m_tile, n_tile) task per 2-CTA cluster, M128xN64xK128 tiles, 7-stage
// pipeline, 8-warp epilogue + TMA store). The fp8 additions:
//   * second operand pair (x_fp8 / w_fp8) with their own TMA descriptors
//     (fp8 K-major 128B swizzle: TMA_K = 128 elems -> ONE load per stage);
//   * scale factors are PRELOADED ONCE into resident smem by warp3 during the
//     prologue (non-persistent kernel = single tile: no per-stage SF pipeline,
//     no with_sf barriers -- unlike wq_b's persistent design);
//   * UTCCP once per 4-stage SF group + sf_id byte select, exactly the
//     cluster_mma_fp8 discipline (helpers reused from its detail namespace).
// ============================================================

#include <algorithm>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cluster_mma_fp8.cuh"   // detail:: UTCCP / block-scale MMA / SF descriptors

namespace front_mixed {

namespace fp8d = cluster_mma_fp8::detail;

constexpr int K = 7168;
constexpr int N = 4672;
constexpr int N_FP8 = 2048;              // wq_a 1536 | wkv 512 (reordered layout)
constexpr int BLOCK_M = 128;
constexpr int CTA_M = 64;
constexpr int BLOCK_K = 128;
constexpr int TMA_K_BF16 = 64;           // bf16: 128B swizzle atom = 64 elems
constexpr int TMA_K_FP8 = 128;           // fp8 : 128B swizzle atom = 128 elems
// SYMMETRIC A/B pipeline depth, smem filled to capacity (DeepGEMM discipline:
// stages = (smem - extras) / (A+B per stage), single ring, equal depth).
// Asymmetric A5/B14 was tried and REGRESSED (19.1 -> 20-22us, worse with
// larger M): A's traffic grows with M (OOB rows are free) and the shallow A
// ring became the new bottleneck -- L2 hits shorten A's latency but do not
// remove its in-flight requirement on a K=7168 stream. B beyond 7 stages
// showed no gain even at M=16 (A-traffic ~zero), so B is not smem-depth
// limited either. The rings stay decoupled in code (equal depths == coupled).
constexpr int NUM_STAGES_A = 8;
constexpr int NUM_STAGES_B = 8;
constexpr int NUM_K_TILES = K / BLOCK_K;                    // 56 (bf16 stages)
// fp8 tiles keep the SAME 64-col width (73 balanced tiles fill the machine)
// but use BLOCK_K=256 stages. WHY: the per-CTA speed limit is the B-stream's
// in-flight bytes / HBM RTT (bf16: 7 x 8KB = 56KB in flight). fp8 at K128
// half-fills the B slot -> in-flight halves and cancels the byte saving
// (measured x1.00). K256 fills BOTH slots exactly (A8 16KB, B8 8KB), keeping
// 56KB in flight over HALF the B bytes -> fp8 tiles run ~2x faster.
constexpr int BLOCK_K_FP8 = 256;
constexpr int NUM_K_TILES_FP8 = K / BLOCK_K_FP8;            // 28
constexpr int BLOCK_N_FP8 = 64;
constexpr int BLOCK_N_BF16 = 64;
constexpr int CTA_N_FP8 = 32;
constexpr int CTA_N_BF16 = 32;
constexpr int N_TILES_FP8 = N_FP8 / BLOCK_N_FP8;                    // 32
constexpr int N_TILES_BF16 = (N - N_FP8) / BLOCK_N_BF16;            // 41
constexpr int NUM_N_TILES = N_TILES_FP8 + N_TILES_BF16;             // 73
// ssq fold: fp8 tiles 0..23 are the wq_a segment (q_norm, 1536), 24..31 the
// wkv segment (kv_norm, 512).
constexpr int N_TILES_Q = 1536 / BLOCK_N_FP8;                       // 24
constexpr int CLUSTER_SIZE = 2;
constexpr int TPB = 256;

// Stage slots are IDENTICAL for both dtypes (fp8 K256 == bf16 K128 in bytes),
// so the pipeline geometry, barrier TX and smem layout are shared.
constexpr int A_STAGE_BYTES = CTA_M * BLOCK_K * (int)sizeof(__nv_bfloat16);   // 16384
constexpr int B_STAGE_BYTES = CTA_N_BF16 * BLOCK_K * (int)sizeof(__nv_bfloat16); // 8192
constexpr int A8_STAGE_BYTES = CTA_M * BLOCK_K_FP8;                           // 16384
constexpr int B8_STAGE_BYTES = CTA_N_FP8 * BLOCK_K_FP8;                       // 8192

// ---- resident scale-factor staging (fp8 tiles only) ----
// One packed u32 per row per SF GROUP: byte g of group u32 = UE8M0 exponent of
// K128 stage 4*group+g (DeepGEMM amortization; sf_id selects the byte).
constexpr int SF_GROUPS = NUM_K_TILES / 4;                  // 14
constexpr int SF_ROWS = 128;                                // UTCCP atom rows
constexpr int SF_BYTES = SF_GROUPS * SF_ROWS * (int)sizeof(uint32_t);   // 7168
constexpr uint32_t UE8M0_ONE4 = 0x7f7f7f7fu;

// TMEM: [accum 64][SFA 4][SFB 4] -> allocate 128 for both tile kinds.
constexpr int NUM_TMEM_COLS = 128;
constexpr int TMEM_SFA = 64;
constexpr int TMEM_SFB = 68;

constexpr uint64_t TMA_CACHE_HINT = 0x1000000000000000ull;
constexpr uint32_t TMA_2SM_PEER_MASK = 0xfeffffffu;
constexpr int EPI_ATOM_M = 64;
constexpr int EPI_ATOM_N = 64;
constexpr int EPI_ATOM_BYTES = EPI_ATOM_M * EPI_ATOM_N * (int)sizeof(__nv_bfloat16);

static_assert(N - N_FP8 == 2624 && (N - N_FP8) % BLOCK_N_BF16 == 0,
              "bf16 segment must tile exactly (41 x 64)");
static_assert(A_STAGE_BYTES == A8_STAGE_BYTES && B_STAGE_BYTES == B8_STAGE_BYTES,
              "fp8 K256 stages must exactly fill the bf16 K128 slots");
static_assert(NUM_K_TILES % 4 == 0, "SF groups pack 4 K128 stages per u32");

struct Barrier {
  alignas(8) uint64_t value;

  __device__ __forceinline__ void init(uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&value));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(addr), "r"(count) : "memory");
  }

  __device__ __forceinline__ bool try_wait(uint32_t phase) const {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&value));
    uint32_t ready;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(ready) : "r"(addr), "r"(phase) : "memory");
    return ready != 0;
  }

  __device__ __forceinline__ void wait(uint32_t phase) const {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&value));
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "L_wait_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
        "@!p bra L_wait_%=;\n\t}"
        :: "r"(addr), "r"(phase), "r"(ticks) : "memory");
  }

  __device__ __forceinline__ void arrive() {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&value));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(addr) : "memory");
  }

  __device__ __forceinline__ void arrive_and_expect_tx(uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&value));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(bytes) : "memory");
  }
};

struct SharedStorage {
  alignas(1024) uint8_t smem_a[NUM_STAGES_A * A_STAGE_BYTES];   //  80 KB
  alignas(1024) uint8_t smem_b[NUM_STAGES_B * B_STAGE_BYTES];   // 112 KB
  alignas(128) uint8_t sf_a[SF_BYTES];                          //   7 KB (resident)
  alignas(128) uint8_t sf_b[SF_BYTES];                          //   7 KB (resident)
  alignas(16) Barrier full_a[NUM_STAGES_A];
  alignas(16) Barrier empty_a[NUM_STAGES_A];
  alignas(16) Barrier full_b[NUM_STAGES_B];
  alignas(16) Barrier empty_b[NUM_STAGES_B];
  alignas(16) Barrier tmem_full;
  alignas(8) uint32_t tmem_base;
};

static_assert(sizeof(SharedStorage) <= 232448,
              "mixed pipeline exceeds the SM100/103 shared-memory budget");

__device__ __forceinline__ uint32_t lane_id() {
  uint32_t value;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t value;
  asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(value));
  return value;
}

__device__ __forceinline__ bool elect_one() {
  uint32_t value;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "elect.sync _|p, 0xffffffff;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t}"
      : "=r"(value));
  return value != 0;
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void fence_view_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* desc) {
  uint64_t addr = reinterpret_cast<uint64_t>(desc);
  asm volatile("prefetch.tensormap [%0];" :: "l"(addr) : "memory");
}

__device__ __forceinline__ void tma_load_2sm(
    const CUtensorMap* desc, Barrier* barrier, void* smem,
    uint16_t multicast_mask, int32_t coord_k, int32_t coord_mn) {
  uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
  uint32_t barrier_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&barrier->value)) &
      TMA_2SM_PEER_MASK;
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
      "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
      "[%0], [%1, {%4, %5}], [%2], %3, %6;"
      :: "r"(smem_addr), "l"(desc_addr), "r"(barrier_addr),
         "h"(multicast_mask), "r"(coord_k), "r"(coord_mn),
         "l"(TMA_CACHE_HINT)
      : "memory");
}

__device__ __forceinline__ void tma_store_2d(
    const CUtensorMap* desc, const void* smem,
    int32_t coord_n, int32_t coord_m) {
  uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%2, %3}], [%1];"
      :: "l"(desc_addr), "r"(smem_addr), "r"(coord_n), "r"(coord_m)
      : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t value) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(value) : "memory");
}

struct SmemDescriptor {
  uint32_t lo;
  uint32_t hi;
};

// K-major SWIZZLE_128B descriptor. SBO = 8 rows * 128B = 1024B for BOTH dtypes
// (a swizzle row is 128 BYTES: 64 bf16 or 128 fp8 elements), so .hi is shared.
__device__ __forceinline__ SmemDescriptor make_smem_desc(void* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  return {addr >> 4, 0x40004040u};
}

// Dense BF16xBF16->FP32, M128, N64, both operands K-major.
__device__ __forceinline__ uint64_t runtime_instr_desc_bf16() {
  constexpr uint32_t desc = 0x08100490u;
  return static_cast<uint64_t>(desc) << 32;
}

// Block-scaled E4M3xE4M3->FP32 + UE8M0, M128, N64, K-major (fp8 tiles).
__device__ __forceinline__
cute::UMMA::InstrDescriptorBlockScaled make_idesc_fp8_m128n64() {
  return cute::UMMA::make_instr_desc_block_scaled<
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
      cutlass::float_ue8m0_t,
      /*UMMA_M=*/128, /*UMMA_N=*/64,
      cute::UMMA::Major::K, cute::UMMA::Major::K>();
}

__device__ __forceinline__ void tmem_alloc_2sm(
    uint32_t smem_addr, uint32_t columns) {
  asm volatile(
      "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
      :: "r"(smem_addr), "r"(columns));
}

__device__ __forceinline__ void tmem_dealloc_2sm(
    uint32_t tmem_addr, uint32_t columns) {
  asm volatile(
      "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
      :: "r"(tmem_addr), "r"(columns));
}

__device__ __forceinline__ void mma_2sm_bf16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint64_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 "
      "[%0], %1, %2, %3, p;\n\t}"
      :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
         "r"(static_cast<uint32_t>(idesc >> 32)), "r"(accumulate));
}

__device__ __forceinline__ void commit_2sm(
    Barrier* barrier, uint16_t mask) {
  uint32_t addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&barrier->value));
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;"
      :: "r"(addr), "h"(mask) : "memory");
}

__device__ __forceinline__ void tmem_load_8x(
    uint32_t addr, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
    uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
        "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
      : "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tmem_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

// Squared value after bf16 rounding: the ssq fold must sum EXACTLY what the
// downstream consumer reads back from the bf16 output tensor.
__device__ __forceinline__ float bf16_round_sq(uint32_t v) {
  const float b = __bfloat162float(__float2bfloat16_rn(__uint_as_float(v)));
  return b * b;
}

__device__ __forceinline__ uint4 pack_bf16x8(
    uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3,
    uint32_t v4, uint32_t v5, uint32_t v6, uint32_t v7) {
  __nv_bfloat16 b0 = __float2bfloat16_rn(__uint_as_float(v0));
  __nv_bfloat16 b1 = __float2bfloat16_rn(__uint_as_float(v1));
  __nv_bfloat16 b2 = __float2bfloat16_rn(__uint_as_float(v2));
  __nv_bfloat16 b3 = __float2bfloat16_rn(__uint_as_float(v3));
  __nv_bfloat16 b4 = __float2bfloat16_rn(__uint_as_float(v4));
  __nv_bfloat16 b5 = __float2bfloat16_rn(__uint_as_float(v5));
  __nv_bfloat16 b6 = __float2bfloat16_rn(__uint_as_float(v6));
  __nv_bfloat16 b7 = __float2bfloat16_rn(__uint_as_float(v7));
  uint32_t p0 = __bfloat16_as_ushort(b0) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(b1)) << 16);
  uint32_t p1 = __bfloat16_as_ushort(b2) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(b3)) << 16);
  uint32_t p2 = __bfloat16_as_ushort(b4) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(b5)) << 16);
  uint32_t p3 = __bfloat16_as_ushort(b6) |
                (static_cast<uint32_t>(__bfloat16_as_ushort(b7)) << 16);
  return uint4{p0, p1, p2, p3};
}

// ============================================================
// [TC/CC DUAL-PATH] MHC post + Sinkhorn-comb tail (fp32), lifted VERBATIM from
// complex_a.cuh::hc_tail (which folded it from hc_fused_kernel_tc's post/comb
// branches). The tail consumes the ALREADY-REDUCED + rms-FOLDED mix [m,24]
// (upstream hc epilogue exports it), so it is pure CUDA-core compute --
// ~80B in / 80B out per position, zero DRAM contention with the GEMM's TMA
// stream. It runs on warp 5, which is otherwise IDLE for the whole mainloop
// (this kernel's epilogue only uses warpgroup 0), and finishes far earlier
// than the ~18us GEMM, then falls through to the existing __syncthreads /
// cluster_sync -- so unlike complex_a no role-scoped barrier is needed.
// Disabled when HcTailArgs.mix == nullptr.
// ============================================================
namespace hc_tail {
constexpr int HC             = 4;              // hc_mult
constexpr int N_OUT          = (2 + HC) * HC;  // 24 mix cols: pre[0,4) | post[4,8) | comb[8,24)
constexpr int SINKHORN_ITERS = 20;
}  // namespace hc_tail

struct HcTailArgs {
  const float* mix      = nullptr;   // [m, 24] REDUCED + rms-folded mix (fp32)
  const float* base     = nullptr;   // [24] hc base
  const float* scale    = nullptr;   // [3]  hc scale (scale[1]=post, scale[2]=comb)
  float  hc_eps   = 1e-6f;
  int    m        = 0;               // HC positions
  float* post_out = nullptr;         // [m, 4]    fp32
  float* comb_out = nullptr;         // [m, 4, 4] fp32
};

__device__ __forceinline__ float hc_tail_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// One warp handles one HC position (grid-strided): ONE coalesced 80B load of
// the reduced mix cols [4,24) -> post gate (lanes 0..3) -> warp-level Sinkhorn
// on the 4x4 comb (lanes 0..15; all xor offsets < 16 keep them closed, upper
// lanes carry masked garbage).
__device__ static void hc_tail_run(HcTailArgs const& a) {
  using namespace hc_tail;
  const int lane = threadIdx.x & 31;
  const float s_post = a.scale[1];   // uniform broadcast loads
  const float s_comb = a.scale[2];
  for (int pos = blockIdx.x; pos < a.m; pos += gridDim.x) {
    // lane c<20 owns the reduced+folded mix col 4+c (one coalesced load)
    const float mix = (lane < N_OUT - HC)
        ? __ldg(a.mix + (size_t)pos * N_OUT + (HC + lane)) : 0.f;

    // ---- post gate: lanes 0..3 hold mix cols 4..7 ----
    if (lane < HC)
      a.post_out[(size_t)pos * HC + lane] =
          2.0f * hc_tail_sigmoid(mix * s_post + a.base[HC + lane]);

    // ---- comb: shift cols 8..23 (lanes 4..19) down to lanes 0..15, activate ----
    float v = __shfl_sync(0xffffffffu, mix, (lane + HC) & 31);
    v = (lane < HC * HC) ? v * s_comb + a.base[2 * HC + lane] : 0.f;

    // ---- Sinkhorn (softmax row-norm, then SINKHORN_ITERS-1 row/col rounds) ----
    float max_v = v;
    #pragma unroll
    for (int off = 1; off < HC; off <<= 1)
      max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffffu, max_v, off));
    const float e = __expf(v - max_v);
    float row_sum = e;
    #pragma unroll
    for (int off = 1; off < HC; off <<= 1)
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off);
    v = e / row_sum + a.hc_eps;
    float col_sum = v;
    #pragma unroll
    for (int off = HC; off < HC * HC; off <<= 1)
      col_sum += __shfl_xor_sync(0xffffffffu, col_sum, off);
    v /= col_sum + a.hc_eps;
    #pragma unroll 1
    for (int iter = 0; iter < SINKHORN_ITERS - 1; ++iter) {
      row_sum = v;
      #pragma unroll
      for (int off = 1; off < HC; off <<= 1)
        row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off);
      v /= row_sum + a.hc_eps;
      col_sum = v;
      #pragma unroll
      for (int off = HC; off < HC * HC; off <<= 1)
        col_sum += __shfl_xor_sync(0xffffffffu, col_sum, off);
      v /= col_sum + a.hc_eps;
    }
    if (lane < HC * HC)
      a.comb_out[(size_t)pos * HC * HC + lane] = v;   // final fp32 gate
  }
}

// ============================================================
// Kernel: one (m_tile, n_tile) task per 2-CTA cluster (non-persistent).
//   x_sf : [M, 56] u8   activation 1x128 UE8M0 exponents (fp8 tiles)
//   w_sf : [16, 56] u8  weight 128x128 UE8M0 exponents (fp8 segment)
// ============================================================
static __global__ void __launch_bounds__(TPB, 1) front_mixed_kernel(
    const __grid_constant__ CUtensorMap desc_a,      // x_bf16 [M,7168]
    const __grid_constant__ CUtensorMap desc_b,      // w_bf16 [2624,7168]
    const __grid_constant__ CUtensorMap desc_a8,     // x_fp8  [M,7168]
    const __grid_constant__ CUtensorMap desc_b8,     // w_fp8  [2048,7168]
    const __grid_constant__ CUtensorMap desc_d,      // y      [M,4672]
    const uint8_t* __restrict__ x_sf,
    const uint8_t* __restrict__ w_sf,
    float* __restrict__ q_ssq,       // [M] fp32, RED-accumulated (null=>off)
    float* __restrict__ kv_ssq,      // [M] fp32, RED-accumulated (null=>off)
    const float* __restrict__ attn_ssq,  // [M] fp32 attn_norm ssq from MHC (null=>off)
    float attn_eps,
    HcTailArgs hc,                   // [TC/CC] MHC post+comb tail (mix=null=>off)
    int problem_m) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& s = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int warp = threadIdx.x >> 5;
  const int lane = lane_id();
  const int rank = cluster_rank();
  const int cluster = blockIdx.x / CLUSTER_SIZE;
  const int n_tile = cluster % NUM_N_TILES;
  const int m_tile = cluster / NUM_N_TILES;
  const bool is_fp8 = n_tile < N_TILES_FP8;
  // Output column base of this tile (fp8 tiles are 128 wide, bf16 64 wide).
  const int n_col_base = is_fp8
      ? n_tile * BLOCK_N_FP8
      : N_FP8 + (n_tile - N_TILES_FP8) * BLOCK_N_BF16;
  const bool elected = elect_one();

  if (warp == 0 && elected) {
    prefetch_tensormap(is_fp8 ? &desc_a8 : &desc_a);
    prefetch_tensormap(is_fp8 ? &desc_b8 : &desc_b);
  }
  cluster_sync();

  if (warp == 1 && elected) {
    #pragma unroll
    for (int i = 0; i < NUM_STAGES_A; ++i) {
      s.full_a[i].init(1);
      s.empty_a[i].init(1);
    }
    #pragma unroll
    for (int i = 0; i < NUM_STAGES_B; ++i) {
      s.full_b[i].init(1);
      s.empty_b[i].init(1);
    }
    s.tmem_full.init(1);
    fence_barrier_init();
  }
  if (warp == 2) {
    uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&s.tmem_base));
    tmem_alloc_2sm(addr, NUM_TMEM_COLS);
  }
  // ---- warp3: resident SF preload (fp8 tiles only; one shot, no pipeline).
  // Runs on BOTH CTAs before the prologue cluster_sync. TWO layout rules
  // learned the hard way (first cut: row-major smem, global 128 rows on both
  // CTAs -> row-permuted scales, M=1 accidentally passing as the only fixed
  // point of the permutation):
  //  1. UTCCP 32x128b expects the INTERLEAVED slot order: smem word
  //     (r%32)*4 + r/32 holds row r (wq_b warp2's st_shared_v4 pattern);
  //  2. 2SM block_scale consumes PER-CTA LOCAL rows: each CTA's SFA holds
  //     ITS OWN CTA_M=64 A-rows at local indices 0..63 (wq_b precedent);
  //     slots 64..127 pad with 1.0.
  // sf_b: a 64-col fp8 tile never straddles a 128x128 weight scale block
  // (128 % 64 == 0), so every slot broadcasts the tile's single block u32.
  if (warp == 3 && is_fp8) {
    const int sf_blk_row = n_col_base / 128;              // w 128x128 block row
    uint32_t wv[SF_GROUPS];
    #pragma unroll
    for (int g = 0; g < SF_GROUPS; ++g)
      wv[g] = *reinterpret_cast<const uint32_t*>(
          w_sf + sf_blk_row * NUM_K_TILES + g * 4);
    for (int l = lane; l < SF_ROWS; l += 32) {            // local SFA row slot
      const int row = m_tile * BLOCK_M + rank * CTA_M + l;
      const bool valid = l < CTA_M && row < problem_m;
      const int slot = (l % 32) * 4 + l / 32;             // UTCCP interleave
      #pragma unroll
      for (int g = 0; g < SF_GROUPS; ++g) {
        const uint32_t av = valid
            ? *reinterpret_cast<const uint32_t*>(
                  x_sf + (size_t)row * NUM_K_TILES + g * 4)
            : UE8M0_ONE4;
        st_shared_u32(s.sf_a + (g * SF_ROWS + slot) * 4, av);
        st_shared_u32(s.sf_b + (g * SF_ROWS + slot) * 4, wv[g]);
      }
    }
    fence_view_async_shared();   // publish to the async proxy (UTCCP source)
  }
  cluster_sync();

  SmemDescriptor desc_a_smem = make_smem_desc(s.smem_a);
  SmemDescriptor desc_b_smem = make_smem_desc(s.smem_b);
  constexpr uint32_t TX_BYTES_A = CLUSTER_SIZE * A_STAGE_BYTES;
  constexpr uint32_t TX_BYTES_B = CLUSTER_SIZE * B_STAGE_BYTES;
  const int num_k_tiles = is_fp8 ? NUM_K_TILES_FP8 : NUM_K_TILES;
  constexpr uint32_t task_phase = 0;

  if (warp == 0 && elected) {
    // ---- A producer (independent ring; both dtypes issue two loads/stage:
    // bf16 two K64 halves, fp8 two K128 swizzle atoms of the K256 stage).
    const int coord_a_row = m_tile * BLOCK_M + rank * CTA_M;
    const uint16_t self_mask = static_cast<uint16_t>(1u << rank);
    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      const uint32_t stage = (uint32_t)k_tile % NUM_STAGES_A;
      const uint32_t phase = ((uint32_t)k_tile / NUM_STAGES_A) & 1;
      s.empty_a[stage].wait(phase ^ 1);
      if (rank == 0) {
        s.full_a[stage].arrive_and_expect_tx(TX_BYTES_A);
      }
      const CUtensorMap* da = is_fp8 ? &desc_a8 : &desc_a;
      const int coord_k = k_tile * (is_fp8 ? BLOCK_K_FP8 : BLOCK_K);
      const int half_k = is_fp8 ? TMA_K_FP8 : TMA_K_BF16;
      tma_load_2sm(da, &s.full_a[stage],
                   s.smem_a + stage * A_STAGE_BYTES,
                   self_mask, coord_k, coord_a_row);
      tma_load_2sm(da, &s.full_a[stage],
                   s.smem_a + stage * A_STAGE_BYTES + A_STAGE_BYTES / 2,
                   self_mask, coord_k + half_k, coord_a_row);
    }
  } else if (warp == 4 && elected) {
    // ---- B producer: SEPARATE warp so the deep weight ring can run ahead
    // of the shallow A ring (a single sequential producer would clamp B's
    // effective depth to A's).
    const uint16_t self_mask = static_cast<uint16_t>(1u << rank);
    const int coord_b_row = is_fp8
        ? n_col_base + rank * CTA_N_FP8                       // w_fp8 rows
        : (n_col_base - N_FP8) + rank * CTA_N_BF16;           // w_bf16 rows
    #pragma unroll 1
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
      const uint32_t stage = (uint32_t)k_tile % NUM_STAGES_B;
      const uint32_t phase = ((uint32_t)k_tile / NUM_STAGES_B) & 1;
      s.empty_b[stage].wait(phase ^ 1);
      if (rank == 0) {
        s.full_b[stage].arrive_and_expect_tx(TX_BYTES_B);
      }
      const CUtensorMap* db = is_fp8 ? &desc_b8 : &desc_b;
      const int coord_k = k_tile * (is_fp8 ? BLOCK_K_FP8 : BLOCK_K);
      const int half_k = is_fp8 ? TMA_K_FP8 : TMA_K_BF16;
      tma_load_2sm(db, &s.full_b[stage],
                   s.smem_b + stage * B_STAGE_BYTES,
                   self_mask, coord_k, coord_b_row);
      tma_load_2sm(db, &s.full_b[stage],
                   s.smem_b + stage * B_STAGE_BYTES + B_STAGE_BYTES / 2,
                   self_mask, coord_k + half_k, coord_b_row);
    }
  } else if (warp == 5) {
    // ---- [TC/CC] warp 5: MHC post + Sinkhorn comb (otherwise idle until the
    // epilogue __syncthreads). Finishes in ~1us (<= 1 row per CTA at M<=128)
    // under the ~18us GEMM window, then falls through and rejoins the normal
    // block-wide synchronization.
    if (hc.mix != nullptr) {
      hc_tail_run(hc);
    }
  } else if (warp == 1 && rank == 0 && elected) {
    // ---- MMA (leader CTA): waits on BOTH rings, releases them separately.
    constexpr uint16_t CTA_MASK = 0x3;
    if (is_fp8) {
      const auto idesc8 = make_idesc_fp8_m128n64();
      tmem_fence_after_sync();
      #pragma unroll 1
      for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const uint32_t st_a = (uint32_t)k_tile % NUM_STAGES_A;
        const uint32_t ph_a = ((uint32_t)k_tile / NUM_STAGES_A) & 1;
        const uint32_t st_b = (uint32_t)k_tile % NUM_STAGES_B;
        const uint32_t ph_b = ((uint32_t)k_tile / NUM_STAGES_B) & 1;
        s.full_a[st_a].wait(ph_a);
        s.full_b[st_b].wait(ph_b);

        // One K256 stage spans TWO K128 scale sub-blocks. UTCCP refreshes the
        // SF group every 4 K128 blocks == every 2 fp8 stages; sf_id selects
        // the byte per K128 half.
        const int k128 = k_tile * 2;
        if ((k128 % 4) == 0) {
          const int g = k128 / 4;
          auto sfd = fp8d::make_sf_desc();
          fp8d::replace_sf_desc_addr(sfd, s.sf_a + g * SF_ROWS * 4);
          fp8d::utccp_4x32_2cta(TMEM_SFA, fp8d::sf_desc_bits(sfd));
          fp8d::replace_sf_desc_addr(sfd, s.sf_b + g * SF_ROWS * 4);
          fp8d::utccp_4x32_2cta(TMEM_SFB, fp8d::sf_desc_bits(sfd));
        }
        const uint32_t a_base =
            desc_a_smem.lo + st_a * (A_STAGE_BYTES / 16);
        const uint32_t b_base =
            desc_b_smem.lo + st_b * (B_STAGE_BYTES / 16);
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K_FP8 / 32; ++kk) {
          const uint32_t k_half = kk / 4;              // K128 half of the stage
          const uint64_t rdesc = fp8d::make_runtime_idesc_with_sf_id(
              idesc8, (k128 + k_half) % 4, (k128 + k_half) % 4);
          const uint32_t a_lo = a_base + k_half * (A_STAGE_BYTES / 2 / 16) +
                                (kk % 4) * 2;
          const uint32_t b_lo = b_base + k_half * (B_STAGE_BYTES / 2 / 16) +
                                (kk % 4) * 2;
          const uint64_t a = (static_cast<uint64_t>(desc_a_smem.hi) << 32) | a_lo;
          const uint64_t b = (static_cast<uint64_t>(desc_b_smem.hi) << 32) | b_lo;
          // A-operand = activation (M rows), B-operand = weight (N cols).
          fp8d::mma_2sm_block_scale(0, a, b, rdesc,
                                    (k_tile != 0 || kk != 0) ? 1u : 0u,
                                    TMEM_SFA, TMEM_SFB);
        }
        commit_2sm(&s.empty_a[st_a], CTA_MASK);
        commit_2sm(&s.empty_b[st_b], CTA_MASK);
        if (k_tile == num_k_tiles - 1) {
          commit_2sm(&s.tmem_full, CTA_MASK);
        }
      }
    } else {
      const uint64_t idesc = runtime_instr_desc_bf16();
      #pragma unroll 1
      for (int k_tile = 0; k_tile < NUM_K_TILES; ++k_tile) {
        const uint32_t st_a = (uint32_t)k_tile % NUM_STAGES_A;
        const uint32_t ph_a = ((uint32_t)k_tile / NUM_STAGES_A) & 1;
        const uint32_t st_b = (uint32_t)k_tile % NUM_STAGES_B;
        const uint32_t ph_b = ((uint32_t)k_tile / NUM_STAGES_B) & 1;
        s.full_a[st_a].wait(ph_a);
        s.full_b[st_b].wait(ph_b);

        uint32_t a_base = desc_a_smem.lo + st_a * (A_STAGE_BYTES / 16);
        uint32_t b_base = desc_b_smem.lo + st_b * (B_STAGE_BYTES / 16);
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K / 16; ++kk) {
          uint32_t k_half = kk / (TMA_K_BF16 / 16);
          uint32_t a_k_half_offset = k_half * (A_STAGE_BYTES / 2 / 16);
          uint32_t b_k_half_offset = k_half * (B_STAGE_BYTES / 2 / 16);
          uint32_t k_block_offset = (kk % (TMA_K_BF16 / 16)) * 2;
          uint64_t a = (static_cast<uint64_t>(desc_a_smem.hi) << 32) |
                       (a_base + a_k_half_offset + k_block_offset);
          uint64_t b = (static_cast<uint64_t>(desc_b_smem.hi) << 32) |
                       (b_base + b_k_half_offset + k_block_offset);
          mma_2sm_bf16(0, a, b, idesc, (k_tile != 0 || kk != 0) ? 1u : 0u);
        }
        commit_2sm(&s.empty_a[st_a], CTA_MASK);
        commit_2sm(&s.empty_b[st_b], CTA_MASK);
        if (k_tile == NUM_K_TILES - 1) {
          commit_2sm(&s.tmem_full, CTA_MASK);
        }
      }
    }
  }

  // ---- Epilogue: SAME shape for both tile kinds (M128xN64, fp32 accum ->
  // bf16 -> swizzled smem -> one TMA store atom). Warpgroup 0 only: the four
  // M32xN32 quadrants would duplicate on warpgroup 1.
  {
    const int epi_warp = warp;
    if (warp == 0 && elected) {
      s.tmem_full.wait(task_phase);
    }
    __syncthreads();
    tmem_fence_after_sync();

    const int epi_warpgroup = epi_warp >> 2;
    const int epi_local_warp = epi_warp & 3;
    const int row_half = epi_local_warp & 1;
    const int col_group = epi_local_warp >> 1;
    constexpr int EPI_STEPS = 2;
    constexpr int EPI_COLS_PER_WARP = 32;

    // q_norm / kv_norm ssq fold (fp8 segment only): sum the squares of the
    // BF16-ROUNDED outputs -- exactly the values the downstream norm+quant
    // kernel reads back. Each lane owns ONE row's 32-col slice across its 4
    // iterations (row is st/sub-invariant), so a single fp32 RED per lane
    // finishes the fold. Accumulation order is nondeterministic -- same
    // zero-init caller contract as wq_b's head_ssq.
    float* const ssq_dst = !is_fp8 ? nullptr
        : (n_tile < N_TILES_Q ? q_ssq : kv_ssq);
    float row_ssq = 0.f;

    // attn_norm fold: the MHC producer already accumulated attn_ssq[m] =
    // sum(x_bf16^2) over the 7168-dim row; its weight is burned into THIS
    // op's weights offline (W'[n,k] = W[n,k]*w_attn[k]), so all that remains
    // is the per-row scalar r = rsqrt(ssq/7168 + eps), applied to the fp32
    // accum BEFORE the bf16 round. Each lane owns one row -> one ld + rsqrt.
    // (q_ssq/kv_ssq then fold r^2 automatically -- same semantics as the
    // original chain where q_norm consumes attn_norm'ed values.)
    float r_attn = 1.f;
    if (attn_ssq != nullptr) {
      const int orow = m_tile * BLOCK_M + rank * CTA_M + row_half * 32 + lane;
      if (orow < problem_m) {
        r_attn = rsqrtf(attn_ssq[orow] * (1.f / (float)K) + attn_eps);
      }
    }

    if (epi_warpgroup == 0) {
      #pragma unroll
      for (int st = 0; st < EPI_STEPS; ++st) {
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          uint32_t addr = st * 16 + sub * 8;
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
          tmem_load_8x(addr, v0, v1, v2, v3, v4, v5, v6, v7);
          tmem_load_fence();
          if (attn_ssq != nullptr) {
            v0 = __float_as_uint(__uint_as_float(v0) * r_attn);
            v1 = __float_as_uint(__uint_as_float(v1) * r_attn);
            v2 = __float_as_uint(__uint_as_float(v2) * r_attn);
            v3 = __float_as_uint(__uint_as_float(v3) * r_attn);
            v4 = __float_as_uint(__uint_as_float(v4) * r_attn);
            v5 = __float_as_uint(__uint_as_float(v5) * r_attn);
            v6 = __float_as_uint(__uint_as_float(v6) * r_attn);
            v7 = __float_as_uint(__uint_as_float(v7) * r_attn);
          }
          const int row = row_half * 32 + lane;
          const int col = col_group * EPI_COLS_PER_WARP + st * 16 + sub * 8;
          const int out_row = m_tile * BLOCK_M + rank * CTA_M + row;
          const int out_col = n_col_base + col;
          if (out_row < problem_m && out_col + 7 < N) {
            uint4 packed = pack_bf16x8(v0, v1, v2, v3, v4, v5, v6, v7);
            if (ssq_dst != nullptr) {
              row_ssq += bf16_round_sq(v0) + bf16_round_sq(v1) +
                         bf16_round_sq(v2) + bf16_round_sq(v3) +
                         bf16_round_sq(v4) + bf16_round_sq(v5) +
                         bf16_round_sq(v6) + bf16_round_sq(v7);
            }
            const int atom_col = col % EPI_ATOM_N;
            const int swizzle_group = (atom_col / 8) ^ (row & 7);
            uint8_t* smem_row = s.smem_a +
                                row * (EPI_ATOM_N * sizeof(__nv_bfloat16)) +
                                swizzle_group * 16 +
                                (atom_col % 8) * sizeof(__nv_bfloat16);
            st_shared_u32(smem_row + 0, packed.x);
            st_shared_u32(smem_row + 4, packed.y);
            st_shared_u32(smem_row + 8, packed.z);
            st_shared_u32(smem_row + 12, packed.w);
          }
        }
      }
      if (ssq_dst != nullptr) {
        const int out_row =
            m_tile * BLOCK_M + rank * CTA_M + row_half * 32 + lane;
        if (out_row < problem_m) {
          atomicAdd(ssq_dst + out_row, row_ssq);
        }
      }
    }
    tmem_fence_before_sync();
  }

  __syncthreads();
  if (warp == 0 && elected) {
    tma_store_fence();
    const int out_m = m_tile * BLOCK_M + rank * CTA_M;
    tma_store_2d(&desc_d, s.smem_a, n_col_base, out_m);
    tma_store_arrive();
  }
  cluster_sync();

  if (warp == 0) {
    tmem_dealloc_2sm(0, NUM_TMEM_COLS);
  }
  if (warp == 0 && elected) {
    tma_store_wait();
  }
}

// ============================================================
// Host-side tensor maps + launch
// ============================================================
inline CUresult make_bf16_map(
    CUtensorMap* desc, const void* ptr, int rows, int box_rows) {
  uint64_t global_dim[2] = {
      static_cast<uint64_t>(K), static_cast<uint64_t>(rows)};
  uint64_t global_stride[1] = {
      static_cast<uint64_t>(K) * sizeof(__nv_bfloat16)};
  uint32_t box_dim[2] = {TMA_K_BF16, static_cast<uint32_t>(box_rows)};
  uint32_t elem_stride[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      const_cast<void*>(ptr), global_dim, global_stride,
      box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult make_fp8_map(
    CUtensorMap* desc, const void* ptr, int rows, int box_rows) {
  uint64_t global_dim[2] = {
      static_cast<uint64_t>(K), static_cast<uint64_t>(rows)};
  uint64_t global_stride[1] = {static_cast<uint64_t>(K)};
  uint32_t box_dim[2] = {TMA_K_FP8, static_cast<uint32_t>(box_rows)};
  uint32_t elem_stride[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<void*>(ptr), global_dim, global_stride,
      box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline CUresult make_bf16_output_map(
    CUtensorMap* desc, void* ptr, int rows) {
  uint64_t global_dim[2] = {
      static_cast<uint64_t>(N), static_cast<uint64_t>(rows)};
  uint64_t global_stride[1] = {
      static_cast<uint64_t>(N) * sizeof(__nv_bfloat16)};
  uint32_t box_dim[2] = {EPI_ATOM_N, EPI_ATOM_M};
  uint32_t elem_stride[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr,
      global_dim, global_stride, box_dim, elem_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

inline cudaError_t configure_front_mixed_kernel() {
  return cudaFuncSetAttribute(
      front_mixed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SharedStorage));
}

inline cudaError_t launch_front_mixed(
    const CUtensorMap& desc_a, const CUtensorMap& desc_b,
    const CUtensorMap& desc_a8, const CUtensorMap& desc_b8,
    const CUtensorMap& desc_d,
    const uint8_t* x_sf, const uint8_t* w_sf,
    float* q_ssq, float* kv_ssq,
    const float* attn_ssq, float attn_eps,
    const HcTailArgs& hc,
    int m, cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  const int m_tiles = (m + BLOCK_M - 1) / BLOCK_M;
  const int clusters = NUM_N_TILES * m_tiles;
  config.gridDim = dim3(clusters * CLUSTER_SIZE, 1, 1);
  config.blockDim = dim3(TPB, 1, 1);
  config.dynamicSmemBytes = sizeof(SharedStorage);
  config.stream = stream;
  cudaLaunchAttribute cluster_attr{};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = CLUSTER_SIZE;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;
  void* args[] = {
      const_cast<CUtensorMap*>(&desc_a),
      const_cast<CUtensorMap*>(&desc_b),
      const_cast<CUtensorMap*>(&desc_a8),
      const_cast<CUtensorMap*>(&desc_b8),
      const_cast<CUtensorMap*>(&desc_d),
      &x_sf, &w_sf, &q_ssq, &kv_ssq, &attn_ssq, &attn_eps,
      const_cast<HcTailArgs*>(&hc), &m};
  return cudaLaunchKernelExC(
      &config, reinterpret_cast<void*>(front_mixed_kernel), args);
}

}  // namespace front_mixed
