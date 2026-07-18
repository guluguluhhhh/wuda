// ============================================================
//  cgemm_norm_a.cuh — PURE OPERATOR (no main / no benchmark)
//  complex_gemm STEP 1 (gemm + norm): the wq_a projection y = x @ w1.T followed
//  by per-token RMSNorm, for the DECODE regime (small M activation, large N).
//  Target shape: M=1..64, N=4352 (=1536+512+2048+256, the four fused w1
//  projections), K=7168. Blackwell SM100 1-SM UMMA (single-CTA, cta_group::1) bf16.
//  Logic mirrors gemm_fuse_norm_a.cuh verbatim: the epilogue is ws-ALWAYS (every
//  ks writes fp32 partials) and the separate reduce is replaced by
//  gemm_rmsnorm_kernel, which sums the split-K slabs, computes each row's RMS
//  over its N features, and writes
//      D[m,:] = (Σ_k partials) * rsqrt(mean(row^2) + eps) * rms_w
//  as bf16 in one pass -- no extra global round-trip vs a plain reduce.
//
//  Public API (fusenorm_* names retained from gemm_fuse_norm_a.cuh):
//      using bf16_t = __nv_bfloat16;
//      void fusenorm_setup(FuseNormCtx&, const bf16_t* A, const bf16_t* B, M,N,K);
//      void fusenorm_run  (FuseNormCtx&, bf16_t* out, const float* rms_w,
//                          float eps = 1e-6f, cudaStream_t stream = 0);
//      void fusenorm_free (FuseNormCtx&);
//  A=(M,K) row-major activation, B=(N,K) row-major weight, out=(M,N) row-major,
//  rms_w length N (fp32 weight), D = rmsnorm(A * B, rms_w, eps).
//
//  SWAP-AB: weight(N) -> A-operand (MMA-M, one CTA loads the full BM weight rows);
//  activation(M) -> B-operand (MMA-N, one CTA loads the full BN activation rows).
//  D is written through a transposed view mD=(N,M) stride(1,N) so acc[n,m] lands
//  at physical D[m,n]. BM in {64,128} must divide N (4352%128==0); BN a multiple
//  of 16 (small M -> more zero-padded M-tiles, col<M predicated); K a mult of 64.
//
//  REQUIRES: CUDA >= 12.8, CUTLASS >= 3.8 (SM100), GPU sm_100f/sm_103a (B300).
//  Build (from a .cu driver that #includes this header):
//    nvcc -std=c++17 -gencode arch=compute_100f,code=sm_100 -O3 \
//         --expt-relaxed-constexpr --expt-extended-lambda \
//         -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//         -o test_cgemm_norm test_cgemm_norm.cu -lcublas -lcublasLt -lcuda
// ============================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>

#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/fast_math.h>          // cutlass::FastDivmod (grid-unpack div/mod)
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>          // block_rank_in_cluster / cluster_sync
#include <cute/numeric/integral_constant.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>  // TMEM allocator

using namespace cute;

using bf16_t = __nv_bfloat16;

#ifndef FUSENORM_CUDA_CHECK
#define FUSENORM_CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)
#endif

constexpr int FUSENORM_NUM_SMS = 148;   // B300 SM count (one CTA per SM slot, single-wave cap)

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// ---- element / accumulator config ----
using ElementA   = cutlass::bfloat16_t;
using ElementB   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;   // output bf16 (matches our decode GEMM)
using ElementAcc = float;

constexpr int MMA_K = 64;   // one 64-wide K-atom (TMA load + swizzle granularity)

// Merge-stage (super-stage) K-pipeline. KMERGE consecutive 64-wide K-atoms fold
// into ONE mbarrier round-trip (producer issues KMERGE A/B TMA loads under a
// single full_barrier; consumer waits once then fires KMERGE*(MMA_K/atomK)
// UMMAs before one empty_barrier arrive), cutting the K-loop barrier round-trips
// ~KMERGE x. KMERGE is a PER-LAUNCH template dim (dispatched over {1,2} via
// dispatch_km); this constant is only the production default (TileConfig.km).
#ifndef FUSENORM_KMERGE
#define FUSENORM_KMERGE 2
#endif
constexpr int KMERGE = FUSENORM_KMERGE;   // production-default 64-wide K-atoms per super-stage

// complex_gemm STEP 2 only reduce+RMSNorms y1 = y[:, 0:1536] (w1's first 1536 rows
// = the projection that feeds step-3's GEMM). The remaining N-1536 columns
// (y2/y3/y4) are left as split-K partials in ws for step-3 to reduce -- this op
// does NOT touch them. FUSENORM_NORM_DIM is that leading norm width; it is clamped
// to N at launch (so shapes with N<1536 fall back to full-row norm).
#ifndef FUSENORM_NORM_DIM
#define FUSENORM_NORM_DIM 1536
#endif

// SMEM budget for the A/B ring. The kernel is PERSISTENT (1 block/SM), so a larger
// ring never costs occupancy -> spend all available SMEM on pipeline depth. SM100
// opt-in max is ~227KB; reserve headroom for barriers/alignment.
constexpr int FUSENORM_PIPE_SMEM_BUDGET = 224 * 1024;   // bytes available for A/B slots

// PIPE_STAGES (super-stages) is chosen PER (MmaM,MmaN) as the largest depth whose
// A/B ring fits FUSENORM_PIPE_SMEM_BUDGET. In the 1-SM variant a single CTA holds
// the FULL A/B tile (no operand split across CTAs), so each 64-wide atom slot
// costs A: MmaM*MMA_K*2 + B: MmaN*MMA_K*2 = (MmaM+MmaN)*MMA_K*2 bytes. A
// super-stage holds KMERGE slots, and
// PIPE_SLOTS = PIPE_STAGES*KMERGE.
constexpr int fusenorm_pipe_stages(int MmaM, int MmaN, int KM) {
  const int per_slot  = (MmaM + MmaN) * MMA_K * 2;   // A+B bytes for one 64-wide slot (full tile)
  const int per_super = KM * per_slot;           // one super-stage (KM slots)
  int st = FUSENORM_PIPE_SMEM_BUDGET / per_super;
  if (st < 2) st = 2;                            // need >=2 to overlap producer/consumer
  return st;
}

// ------------------------------------------------------------
// Shared memory: multi-stage A/B ring buffer + per-stage full/empty mbarriers
// + TMEM ptr. ASmemLayout/BSmemLayout carry a trailing PIPE (=Stages) mode, so
// cosize_v already accounts for all stages.
// ------------------------------------------------------------
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout, int Stages>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;   // (...,PIPE)
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;   // (...,PIPE)
  alignas(16) cute::uint64_t full_barrier[Stages];   // per-stage: TMA data ready -> MMA may read
  alignas(16) cute::uint64_t empty_barrier[Stages];  // per-stage: MMA consumed -> slot free to refill
  alignas(16) cute::uint64_t acc_barrier;            // per-tile: UMMA accumulator complete -> epilogue may read TMEM
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

// ------------------------------------------------------------
// Device kernel (1-SM UMMA, multi-stage software pipeline, split-K, bf16 epilogue).
// Operand-agnostic: swap-AB (which matrix is A/B + the transposed D view) is set
// up on the host. Each CTA runs its own independent UMMA -- no cross-CTA sync.
// ------------------------------------------------------------
template <class SharedStorage,
          class ATensor, class BTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB, class Alpha, int KMERGE_>
__global__ static void
gemm_device(ATensor mA, BTensor mB, DTensor mD,
            MmaTiler_MNK mma_tiler, TiledMMA tiled_mma, ClusterShape_MNK cluster_shape,
            CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
            CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
            Alpha alpha,
            float* __restrict__ ws,               // ks*N*M fp32 partials (null if ks==1; reduced by a separate kernel)
            int ks, int ktiles_per_split, int gM, int gN,
            cutlass::FastDivmod ks_divmod, cutlass::FastDivmod gN_divmod)
{
  const int Mv = size<0>(mD);   // == N (weight rows, MMA-M dim)
  const int Nv = size<1>(mD);   // == M (activation rows, MMA-N dim, padded)

  // Super-stage fold factor is a template parameter (swept 1 vs 2); this local
  // alias lets the mainloop below keep referring to it as KMERGE. It SHADOWS the
  // file-scope KMERGE (which is only the production default for choose_tile_config).
  constexpr int KMERGE = KMERGE_;

  extern __shared__ char shared_memory[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor tCsA = smem.tensor_sA();
  Tensor tCsB = smem.tensor_sB();

  // ---- TMA partition coordinate. The cluster is a trivial 1x1x1; this layout is
  // kept only because make_tma_atom / tma_partition consume it (no peer CTA). ----
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  ThrMMA cta_mma = tiled_mma.get_slice(0);   // single CTA (cta_group::1) => slice 0

  Tensor tCrA = cta_mma.make_fragment_A(tCsA);   // SMEM descriptors (MMA,MMA_M,MMA_K,PIPE_SLOTS)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);   //                  (MMA,MMA_N,MMA_K,PIPE_SLOTS)

  // Pipeline depth is baked into the SMEM layout (last mode = PIPE_SLOTS); recover
  // the super-stage count (== barrier-array size) from it. Static -> compile-time.
  constexpr int kPipeSlots  = decltype(size<3>(tCrA))::value;   // total 64-wide atom slots
  constexpr int kPipeStages = kPipeSlots / KMERGE;              // super-stages (barrier count)

  const int warp_id       = threadIdx.x >> 5;
  uint32_t  elect_one_thr = cute::elect_one_sync();   // one lane per warp
  bool      elect_one_warp = (warp_id == 0);

  // ---- TMEM allocation (ONCE; reused by every tile this persistent CTA runs) ----
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  }
  __syncthreads();

  // Accumulator TMEM tensor: its LAYOUT (bM x bN partition) is tile-independent,
  // so build it once from a representative tile-0 D partition; data = TMEM base.
  Tensor gD0    = local_tile(mD, mma_tiler, make_coord(0, 0, _), Step<_1,_1, X>{});
  Tensor tCtAcc = cta_mma.make_fragment_C(cta_mma.partition_C(gD0));
  tCtAcc.data() = smem.tmem_base_ptr;

  // ---- TMA / MMA setup (constant across tiles) ----
  auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));

  // 1-SM: no multicast. Each CTA loads its own full A/B tile and runs its own UMMA,
  // so full_barrier (TMA data ready) and empty_barrier (slot free) / acc_barrier
  // (accumulator done) all have exactly ONE participant -- this very CTA.
  // Parallel barrier init: warp 0's lanes each initialize a lane-strided subset of
  // the per-stage barriers (one thread per barrier -- distinct SMEM addresses, so
  // no conflict). __syncthreads below is the correctness barrier (every warp must
  // observe barrier init + TMEM alloc before the producer issues TMA / the consumer
  // waits); there is no cross-CTA sync anymore.
  if (elect_one_warp) {
    const int lane = threadIdx.x & 31;   // warp 0 -> lane == threadIdx.x
    if (lane < kPipeStages) {
      cute::initialize_barrier(smem.full_barrier[lane],  /* num_threads */ 1);
      cute::initialize_barrier(smem.empty_barrier[lane], /* num_ctas    */ 1);
    }
    if (lane == 0)
      cute::initialize_barrier(smem.acc_barrier, /* num_ctas */ 1);
  }
  __syncthreads();  // observe barrier init + TMEM alloc within this CTA

  // ---- Persistent schedule (1-D grid; each CTA strides over its share of tiles) ----
  const int total_tiles = gM * gN * ks;                      // (N-tiles) x (M-tiles) x split-K

  // ==== Warp-specialized, persistent software-pipelined mainloop ====
  // Both warps keep INDEPENDENT persistent stage counters that advance +1 (mod
  // PIPE_STAGES) and flip phase at wrap.
  //   warp 0 : TMA producer  -> fills full_barrier[s] (tx bytes)
  //   warp 1 : UMMA consumer -> drains, signals empty_barrier[s]
  // full_barrier: producer -> consumer (data ready).  empty_barrier: consumer ->
  // producer (slot free).  acc_barrier: consumer -> all (accumulator complete).
  int tma_stage = 0, mma_stage = 0;   // persistent (NEVER reset per tile)
  int mma_phase = 1;                  // producer's empty-wait phase (1 => first fill is free)
  int tma_phase = 0;                  // consumer's full-wait phase
  int acc_phase = 0;                  // epilogue rendezvous phase

  // Epilogue reads the accumulator directly from TMEM with a raw tcgen05.ld
  // (32dp .x8), NOT CuTe make_tmem_copy: the SM100_TMEM_LOAD_32dp32b1x copy atom is
  // hard-wired for a 128-datapath accumulator and rejects MMA-M<128 (make_tmem_copy
  // AtomTVLayout static_assert). The raw load instead just activates BM/32 warps,
  // each owning its own 32 datapath rows, so BM in {32,64,96,128} all work. tCtAcc
  // above still defines the accumulator TMEM layout consumed by the MMA; here we
  // only need its base address (== smem.tmem_base_ptr) for the ld addressing.
  const uint32_t tmem_acc_base = smem.tmem_base_ptr;
  constexpr int  BMt = decltype(size<0>(mma_tiler))::value;   // MMA-M (weight-N tile)
  constexpr int  BNt = decltype(size<1>(mma_tiler))::value;   // MMA-N (activation-M tile)

  for (int work = blockIdx.x; work < total_tiles; work += gridDim.x) {
    int kc, mt, nt;
    // FastDivmod (magic mul+shift) replaces the native 32-bit IDIV/IMOD, which
    // has no HW instruction and expands to tens of cycles. For small shapes the
    // work loop runs ~1 iteration, so this unpack sits on the critical path.
    if (ks > 1) { int rest = ks_divmod.divmod(kc, work); mt = gN_divmod.divmod(nt, rest); }
    else        { kc = 0;         mt = gN_divmod.divmod(nt, work); }

    auto mma_coord = make_coord(mt, nt, _);
    Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X,_1>{});  // (MmaTile_M, MmaTile_K, Tiles_K)
    Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X,_1,_1>{});  // (MmaTile_N, MmaTile_K, Tiles_K)
    Tensor tCgA = cta_mma.partition_A(gA);
    Tensor tCgB = cta_mma.partition_B(gB);
    // (No CuTe D partition: the epilogue stores through raw pointers on mD, below.)

    auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                      get<2>(cta_in_cluster_coord_vmnk),
                                      make_layout(size<2>(cluster_layout_vmnk)),
                                      group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
    auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                      get<1>(cta_in_cluster_coord_vmnk),
                                      make_layout(size<1>(cluster_layout_vmnk)),
                                      group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));

    // 1SM TMA transaction bytes: this CTA loads one full A + one full B tile, single stage.
    int tma_transaction_bytes = sizeof(make_tensor_like(tAsA(_, Int<0>{})))
                              + sizeof(make_tensor_like(tBsB(_, Int<0>{})));

    const int ktiles_total = size<3>(tCgA);          // == K / MmaTile_K
    int k_beg = kc * ktiles_per_split;
    int k_end = k_beg + ktiles_per_split;
    if (k_end > ktiles_total) k_end = ktiles_total;

    // ---- Producer warp (warp 0, one thread): issue TMA loads ----
    // One super-stage folds KMERGE consecutive 64-wide K-atoms into ONE
    // full_barrier: set its transaction bytes once (KMERGE * per-atom), then issue
    // KMERGE A/B TMA loads into slots [stage*KMERGE, stage*KMERGE+KMERGE). A final
    // tail super-stage (0<tail<KMERGE) covers a split whose K-tile count is not a
    // multiple of KMERGE; for KMERGE==1 the tail folds away and this is the
    // original one-atom-per-stage loop.
    if (warp_id == 0 && elect_one_thr) {
      int t = k_beg;
      for (; t + KMERGE <= k_end; t += KMERGE) {
        cute::wait_barrier(smem.empty_barrier[tma_stage], mma_phase);   // super-stage slot free?
        cute::set_barrier_transaction_bytes(smem.full_barrier[tma_stage], KMERGE * tma_transaction_bytes);
        CUTE_UNROLL
        for (int a = 0; a < KMERGE; ++a) {
          int slot = tma_stage * KMERGE + a;
          copy(tma_atom_A.with(smem.full_barrier[tma_stage]), tAgA(_, t + a), tAsA(_, slot));
          copy(tma_atom_B.with(smem.full_barrier[tma_stage]), tBgB(_, t + a), tBsB(_, slot));
        }
        ++tma_stage;
        if (tma_stage == kPipeStages) { tma_stage = 0; mma_phase ^= 1; }
      }
      if (KMERGE > 1 && t < k_end) {
        int tail = k_end - t;
        cute::wait_barrier(smem.empty_barrier[tma_stage], mma_phase);
        cute::set_barrier_transaction_bytes(smem.full_barrier[tma_stage], tail * tma_transaction_bytes);
        for (int a = 0; a < tail; ++a) {
          int slot = tma_stage * KMERGE + a;
          copy(tma_atom_A.with(smem.full_barrier[tma_stage]), tAgA(_, t + a), tAsA(_, slot));
          copy(tma_atom_B.with(smem.full_barrier[tma_stage]), tBgB(_, t + a), tBsB(_, slot));
        }
        ++tma_stage;
        if (tma_stage == kPipeStages) { tma_stage = 0; mma_phase ^= 1; }
      }
    }

    // ---- Consumer warp (warp 1): issue UMMA back-to-back ----
    // Mirror the producer's super-stages: ONE full_barrier wait, then KMERGE atoms
    // x (MMA_K/atomK) UMMAs into TMEM, then ONE empty_barrier arrive. The tail
    // super-stage mirrors the producer for a K-tile count not divisible by KMERGE.
    if (warp_id == 1) {
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;   // first MMA of this tile clears TMEM
      int t = k_beg;
      for (; t + KMERGE <= k_end; t += KMERGE) {
        cute::wait_barrier(smem.full_barrier[mma_stage], tma_phase);    // super-stage data ready?
        CUTE_UNROLL
        for (int a = 0; a < KMERGE; ++a) {
          int slot = mma_stage * KMERGE + a;
          CUTE_UNROLL
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_,_,k_block,slot), tCrB(_,_,k_block,slot), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
        }
        cutlass::arch::umma_arrive(&smem.empty_barrier[mma_stage]);
        ++mma_stage;
        if (mma_stage == kPipeStages) { mma_stage = 0; tma_phase ^= 1; }
      }
      if (KMERGE > 1 && t < k_end) {
        int tail = k_end - t;
        cute::wait_barrier(smem.full_barrier[mma_stage], tma_phase);
        for (int a = 0; a < tail; ++a) {
          int slot = mma_stage * KMERGE + a;
          CUTE_UNROLL
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            gemm(tiled_mma, tCrA(_,_,k_block,slot), tCrB(_,_,k_block,slot), tCtAcc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
        }
        cutlass::arch::umma_arrive(&smem.empty_barrier[mma_stage]);
        ++mma_stage;
        if (mma_stage == kPipeStages) { mma_stage = 0; tma_phase ^= 1; }
      }
      cutlass::arch::umma_arrive(&smem.acc_barrier);  // accumulator complete
    }

    // ---- Rendezvous: MMA accumulator complete -> all warps enter the epilogue ----
    __syncthreads();
    cute::wait_barrier(smem.acc_barrier, acc_phase);
    acc_phase ^= 1;

    // ---- Epilogue: raw-PTX TMEM->RMEM (fp32) + scattered store through the
    // transposed D view. TMEM has 4 sub-partitions of 32 physical lanes (4*32=128).
    // tcgen05 packs the MMA-M (weight-N) rows as (rows_per_sp INNER, 4 OUTER=sub-
    // partition): physical lane = warp*32 + laneid, logical weight-N row =
    // warp*(BM/4) + laneid, VALID only for laneid < BM/4.
    //   BM=128 -> 32 rows/subpart: every lane valid, physical lane == logical row.
    //   BM=64  -> 16 rows/subpart: upper 16 lanes of each sub-partition are unused
    //             (skip their store); logical row = warp*16 + laneid.
    // ALL 4 warps are always active (one per sub-partition). Each lane reads 8
    // consecutive columns (activation-M) per tcgen05.ld.32x32b.x8. Accumulator
    // (MMA-M=weight-N=dim0, MMA-N=act-M=dim1):
    //   n_global = mt*BM + warp*(BM/4) + laneid   (weight-N, dim0; < Mv, unpadded)
    //   m        = nt*BN + mc*8 + e               (activation-M, dim1; padded -> m<Nv)
    // Transposed view mD stride (1, Mv): element (n_global,m) -> ptr[m*Mv + n_global].
    constexpr int kRowsPerSp = BMt / 4;                       // 32 (BM=128) / 16 (BM=64)
    if (warp_id < 4) {
      const int lane       = threadIdx.x & 31;
      const int tmem_row   = warp_id << 5;                    // sub-partition physical lane base
      const int n_global   = mt * BMt + warp_id * kRowsPerSp + lane;  // weight-N (dim0)
      const bool row_valid = (lane < kRowsPerSp);             // upper lanes unused when BM<128
      // FUSED-NORM: ws-always. The GEMM ALWAYS writes fp32 partials to ws (even
      // for ks==1); the fused reduce+RMSNorm kernel then produces the final bf16
      // D. This unifies the ks==1 / ks>1 paths so RMSNorm has one entry point.
      float* ws_slab = ws + (size_t)kc * Mv * Nv;
      CUTE_UNROLL
      for (int mc = 0; mc < (BNt >> 3); ++mc) {             // BN/8 chunks of 8 columns
        float t[8];
        uint32_t addr = tmem_acc_base + (uint32_t(tmem_row) << 16) + uint32_t(mc << 3);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(t[0]), "=f"(t[1]), "=f"(t[2]), "=f"(t[3]),
              "=f"(t[4]), "=f"(t[5]), "=f"(t[6]), "=f"(t[7])
            : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        if (!row_valid) continue;                            // ld is warp-collective; skip STORE only
        const int m_base = nt * BNt + (mc << 3);            // activation-M (dim1)
        CUTE_UNROLL
        for (int e = 0; e < 8; ++e) {
          int m = m_base + e;
          if (m < Nv) ws_slab[(size_t)m * Mv + n_global] = alpha * t[e];
        }
      }
    }

    __syncthreads();   // tile-end: all warps finished the epilogue before the
                       // next persistent tile re-inits/overwrites this TMEM/SMEM.
  }  // ===== end persistent tile loop =====

  // ---- TMEM dealloc (ONCE, after all tiles) ----
  __syncthreads();
  // Split-K partials (ws) are summed + RMS-normalized by gemm_rmsnorm_kernel in
  // a SEPARATE launch: the kernel boundary supplies the cross-CTA sync, so
  // NO in-kernel grid barrier / self-resetting counter is needed here. This
  // removes the grid-wide atomic spin that dominated the small-shape floor.
  //
  // PDL: fire the programmatic-launch trigger HERE -- BEFORE the TMEM dealloc.
  // The dependent gemm_rmsnorm_kernel only touches global ws (already fully
  // written and made block-visible by the tile-end __syncthreads above); it has
  // NO dependency on this block's TMEM lifecycle. Firing before free() lets the
  // reduce prologue (grid schedule / index math) overlap the TMEM release +
  // block teardown latency. Correctness still holds: the reduce gates its ws
  // READS behind cudaGridDependencySynchronize(), which only releases after ALL
  // GEMM blocks trigger (all partials visible). Fires UNCONDITIONALLY: ws-always
  // means the fused reduce+RMSNorm kernel always follows (including ks==1).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

// ------------------------------------------------------------
// Fused split-K reduce + RMSNorm (SEPARATE kernel; ws-always so ks in [1,8]).
// Each block owns ONE output row = one activation token m; row_len == N weight
// features. Steps:
//   (1) sum the KS fp32 partial slabs in ws -> the row's N accumulated values,
//   (2) block-reduce the row's sum-of-squares,
//   (3) rms = rsqrt(mean(y^2) + eps),
//   (4) y[n] = y[n] * rms * rms_w[n]  (rms_w length == N), write bf16 D.
// Launched as EXACTLY num_rows blocks; blockDim = round_up(row_len/8, 32) so the
// warp-shuffle block reduction sees only FULL warps -- threads past row_len/8 are
// inactive (contribute 0, no ws read / D write). Each ACTIVE thread == exactly
// one 8-wide column group (2x float4 read / 1x uint4 write). KS is a template
// parameter so the accumulation loop fully unrolls. REQUIRES row_len % 8 == 0
// (N % 128 == 0 => holds) and row_len/8 <= 1024 (N <= 8192).
// ------------------------------------------------------------
__device__ __forceinline__ float fusenorm_warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}
// Block-wide sum. blockDim is a multiple of 32 (launch rounds up), so every warp
// is full -> the 0xffffffff-mask shuffles are valid. Result broadcast to all.
__device__ __forceinline__ float fusenorm_block_sum(float v) {
  __shared__ float partial[32];   // one slot per warp (blockDim <= 1024 => <= 32 warps)
  const int lane = threadIdx.x & 31;
  const int wid  = threadIdx.x >> 5;
  v = fusenorm_warp_sum(v);
  if (lane == 0) partial[wid] = v;
  __syncthreads();
  const int nwarps = (blockDim.x + 31) >> 5;
  float t = (threadIdx.x < nwarps) ? partial[threadIdx.x] : 0.f;
  if (wid == 0) t = fusenorm_warp_sum(t);
  if (threadIdx.x == 0) partial[0] = t;
  __syncthreads();
  return partial[0];
}

template <int KS>
__global__ static void
gemm_rmsnorm_kernel(ElementC* __restrict__ out, const float* __restrict__ ws,
                    const float* __restrict__ rms_w, float eps,
                    int norm_len, int stride, int num_rows) {
  // norm_len = leading columns that get reduce+RMSNorm (y1 width, <= stride);
  // stride   = physical ws/out row stride (== full N). Columns [norm_len, stride)
  //            are NOT read/written here -- their split-K partials stay in ws.
  const size_t total = (size_t)stride * num_rows;    // per-slab element count (full N rows)
  // PDL prologue: index math is independent of the GEMM output; run it while the
  // producer GEMM drains, then wait for ALL GEMM blocks (ws partials visible).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
  const int    col    = threadIdx.x << 3;                    // this thread's 8-col group (feature n)
  const bool   active = (col < norm_len);                    // threads beyond y1 width idle
  const size_t base   = (size_t)blockIdx.x * stride + col;   // m*N + n (== ws / D physical layout)
  // Prefetch the RMSNorm weight now: it depends only on `col`, not on the reduce
  // result, so issue its global load here and let the latency overlap the split-K
  // reduce + block-wide sum-of-squares barrier below. (The compiler will not hoist
  // it on its own -- the __syncthreads() inside fusenorm_block_sum blocks that.)
  float4 w0, w1;
  if (active) {
    w0 = reinterpret_cast<const float4*>(rms_w + col)[0];
    w1 = reinterpret_cast<const float4*>(rms_w + col)[1];
  }
  float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (active) {
    #pragma unroll
    for (int k = 0; k < KS; ++k) {
      const float* src = ws + (size_t)k * total + base;
      float4 v0 = reinterpret_cast<const float4*>(src)[0];
      float4 v1 = reinterpret_cast<const float4*>(src)[1];
      s[0]+=v0.x; s[1]+=v0.y; s[2]+=v0.z; s[3]+=v0.w;
      s[4]+=v1.x; s[5]+=v1.y; s[6]+=v1.z; s[7]+=v1.w;
    }
  }
  // Row sum-of-squares (block-wide; inactive threads contribute 0) -> RMS scale.
  // Group size is norm_len (y1 width), NOT the full N.
  float ss = 0.f;
  #pragma unroll
  for (int e = 0; e < 8; ++e) ss += s[e] * s[e];
  ss = fusenorm_block_sum(ss);                               // ALL threads must reach this
  const float rms = rsqrtf(ss / (float)norm_len + eps);
  if (!active) return;                                       // beyond-y1 threads: no rms_w read / D write
  // y[n] = y[n] * rms * rms_w[n]   (w0/w1 prefetched above -> load latency hidden)
  __nv_bfloat162 h2[4];
  h2[0] = __floats2bfloat162_rn(s[0] * rms * w0.x, s[1] * rms * w0.y);
  h2[1] = __floats2bfloat162_rn(s[2] * rms * w0.z, s[3] * rms * w0.w);
  h2[2] = __floats2bfloat162_rn(s[4] * rms * w1.x, s[5] * rms * w1.y);
  h2[3] = __floats2bfloat162_rn(s[6] * rms * w1.z, s[7] * rms * w1.w);
  uint4* dst = reinterpret_cast<uint4*>(out + base);
  *dst = *reinterpret_cast<uint4*>(h2);
}

// Launch gemm_rmsnorm_kernel with a compile-time KS chosen from the runtime
// split-K depth (ks in [1, 8]; ws-always so ks==1 is a valid 1-slab pass).
static cudaError_t fusenorm_launch_reduce(int ks, cudaLaunchConfig_t const& cfg,
                                          ElementC* out, const float* ws,
                                          const float* rms_w, float eps,
                                          int norm_len, int stride, int num_rows) {
  switch (ks) {
    case 1: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<1>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 2: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<2>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 3: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<3>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 4: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<4>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 5: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<5>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 6: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<6>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 7: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<7>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    case 8: return cudaLaunchKernelEx(&cfg, gemm_rmsnorm_kernel<8>, out, ws, rms_w, eps, norm_len, stride, num_rows);
    default: return cudaErrorInvalidValue;   // ks>8 never produced (hard cap 8)
  }
}

// ------------------------------------------------------------
// Split-K plan + persistent workspace (ws allocated once; reduced by a
// separate kernel, so no sync counter is needed).
// ------------------------------------------------------------
struct SplitKPlan {
  float* d_ws       = nullptr;
  int    ks         = 1;
  int    ktiles_per_split = 0;
  int    base_tiles = 1;
};

// SWAP-AB tile config. ONE function (choose_tile_config) decides BM (MMA-M), BN
// (MMA-N) and the split-K depth (ks) together, plus the derived
// ktiles_per_split / base_tiles. The selection rule is a formula fitted to the
// B300 forced-tile sweep -- see the detailed findings comment inside
// choose_tile_config below. M not a multiple of BN is zero-padded (col<M
// predicated); BM divides N by construction (N%128==0 => N%64==0 too).
struct TileConfig {
  int  mma_m = 0;              // BM (0 => invalid / unsupported N)
  int  mma_n = 16;             // BN
  int  ks = 1;                 // split-K depth
  int  ktiles_per_split = 0;
  int  base_tiles = 1;
  int  km = KMERGE;            // super-stage fold factor (production default; sweep forces 1/2)
  bool valid = false;
};

// (ks, ktiles_per_split) for a base_tiles (PURE). ks capped at 8. base_tiles
// uses ceil on BOTH dims (Md=N divides MmaM exactly; Nd=M may need padding).
static void fusenorm_split_depth(int base_tiles, int K, int& ks, int& ktiles_per_split) {
  int ktiles_total  = K / MMA_K;
  int sm_slots      = FUSENORM_NUM_SMS;       // 148 (one CTA per SM slot, single wave)
  int ks_max        = 8;                    // hard cap (ks must not exceed 8)
  int ks_target     = sm_slots / base_tiles;
  if (ks_target > ks_max)       ks_target = ks_max;
  if (ks_target < 1)            ks_target = 1;
  if (ks_target > ktiles_total) ks_target = ktiles_total;
  if (ks_target == 1) { ks = 1; ktiles_per_split = ktiles_total; return; }
  ktiles_per_split = (ktiles_total + ks_target - 1) / ks_target;   // ceil
  ks               = (ktiles_total + ktiles_per_split - 1) / ktiles_per_split;
}

// The single BM/BN/ks selection entry point (see the fitted-rule comment inside
// the function). BN is always a multiple of 16 (the epilogue TMEM-load atom +
// SW128 swizzle require MMA-N%16), capped at 128 for the decode regime (the
// activation M-tile never needs to exceed 128).
static constexpr int FUSENORM_MAX_BN = 128;

static TileConfig choose_tile_config(int M, int N, int K) {
  TileConfig c;
  if (N % 128 != 0 || K % MMA_K != 0) return c;    // unsupported -> valid stays false

  // BN MUST be a multiple of 16: the K-major SW128 SMEM swizzle atom + the epilogue
  // TMEM-load atom require the MMA-N (activation-M) tile to align to 16 -- non-16
  // multiples fail cute's tile_to_shape divisibility static_asserts on B300.
  auto cover16 = [](int x) { return ((x < 1 ? 1 : x) + 15) / 16 * 16; };   // smallest mult-of-16 >= x

  // ---- Tile rule RE-TUNED from the B300 1SM forced-tile sweep at the real
  // complex_gemm step-1 shape (N=4352,K=7168,M=1..64):
  //  (1) BM=128 wins for every M (BM=64 reachable via fusenorm_setup_forced only).
  //  (2) latency-optimal BN = cover16(M): the SMALLEST multiple of 16 that covers
  //      M in ONE M-tile (BN>=M). Measured best BN: M<=16 ->16, M=24 ->32,
  //      M=48 ->48, M=64 ->64 (M=32 measured 48 but 32 is within 0.02us noise).
  //      The old ceil(M/2) (two M-tiles) UNDERSHOOTS for M>=24 and lost ~1us there.
  //  (3) ks = deepest split-K filling the single 148-CTA wave (cap 8), via
  //      fusenorm_split_depth. KMERGE stays at the production default (=2).
  // NOTE: the 2-SM/BM=256 variant (trash/cgemm_norm_2sm_a.cuh) was measured on the
  // real complex_gemm step-1 shape (N=4352,K=7168) and NEVER wins here -- this op is
  // access-bound on the full weight read (~62 MB -> ~12.4us floor), and 2-SM does
  // NOT cut that HBM traffic (multicast only saves the tiny activation operand),
  // while its coarser 17 N-tiles + cluster-barrier cost hurt the 148-SM wave fill
  // (1-SM hits 11.1us at M=8/16 vs 2-SM's 12.4; 2-SM is 10-16% slower at M>=48).
  // Keep 1-SM for decode; 2-SM would only pay off in a compute-bound / large-M path.
  const int bm = 128;
  int bn = cover16(M);                             // smallest mult-of-16 >= M (ONE M-tile)
  if (bn < 16)            bn = 16;
  if (bn > FUSENORM_MAX_BN) bn = FUSENORM_MAX_BN;

  const int ntiles = N / bm;                       // weight N-tiles (exact; N%128==0)
  const int mtiles = (M + bn - 1) / bn;            // ceil(M/BN); M padded to BN
  const int base   = ntiles * mtiles;
  int ks, kps;
  fusenorm_split_depth(base, K, ks, kps);            // deepest ks filling the 148 wave (cap 8)

  c.mma_m            = bm;
  c.mma_n            = bn;
  c.ks               = ks;
  c.ktiles_per_split = kps;
  c.base_tiles       = base;
  c.valid            = true;
  return c;                                        // c.km stays at its KMERGE default
}

// Allocate the split-K workspace for an already-decided TileConfig (Md=N weight
// rows, Nd=M activation rows). FUSED-NORM: ws-always -- even ks==1 allocates a
// 1-slab fp32 workspace, since the GEMM never writes bf16 D directly (the fused
// reduce+RMSNorm kernel is the sole D producer).
static SplitKPlan alloc_splitk(int Md, int Nd, TileConfig const& c) {
  SplitKPlan p;
  p.base_tiles       = c.base_tiles;
  p.ks               = c.ks;
  p.ktiles_per_split = c.ktiles_per_split;
  FUSENORM_CUDA_CHECK(cudaMalloc(&p.d_ws, (size_t)p.ks * Md * Nd * sizeof(float)));
  return p;
}

// Type-erased launch closure. The expensive TMA-descriptor encode
// (cuTensorMapEncodeTiled inside make_tma_atom_*_sm100) is done ONCE when the
// closure is built; each invocation only rebuilds the cheap transposed D view
// (no encode) and issues the cluster launch, then the fused reduce+RMSNorm.
// rms_w (length N == weight features) + eps are per-call RMSNorm parameters.
using FuseNormLaunchFn =
    std::function<cutlass::Status(ElementC* dD, const float* rms_w, float eps,
                                  SplitKPlan const& plan, float alpha,
                                  cudaStream_t stream)>;

// ------------------------------------------------------------
// Build the launch closure ONCE (called from fusenorm_setup). SWAP-AB: A-operand =
// WEIGHT (dB, N rows -> MMA-M, one CTA loads the full BM rows), B-operand =
// ACTIVATION (dA, M rows -> MMA-N, one CTA loads the full BN rows). D is written
// through a transposed view: mD_swap[n,m] -> physical D[m,n] (stride(1,N)).
// Template <MmaM=N-tier, MmaN=M-tier>. The strongly-typed TMA atoms (with their
// encoded descriptors) are captured by value into the returned closure, so the
// descriptor encode happens only here, not on every fusenorm_run.
// ------------------------------------------------------------
template <int MmaM, int MmaN, int KM>
static FuseNormLaunchFn build_launch_fn(ElementA const* dA, ElementB const* dB,
                                        int M, int N, int K, int ks)
{
  const int Md = N;   // weight rows -> MMA-M (dim0, 1SM A-operand, full BM tile)
  const int Nd = M;   // activation rows -> MMA-N (dim1, 1SM B-operand, full BN tile, padded)

  if (Md % MmaM != 0 || K % MMA_K != 0) {
    return FuseNormLaunchFn{};   // invalid; caller keeps ctx.valid=false
  }

  // A-operand = weight (N,K) K-major;  B-operand = activation (M,K) K-major.
  Layout layout_A = make_layout(make_shape(Md, K), make_stride(K, Int<1>{}));
  Layout layout_B = make_layout(make_shape(Nd, K), make_stride(K, Int<1>{}));

  Tensor mA = make_tensor(make_gmem_ptr(dB), layout_A);   // A-operand := weight
  Tensor mB = make_tensor(make_gmem_ptr(dA), layout_B);   // B-operand := activation

  // 1-SM MMA: MmaM x MmaN, AtomThrID=_1 (one CTA computes the full MmaM x MmaN tile).
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<ElementA, ElementB, ElementAcc,
                           MmaM, MmaN, UMMA::Major::K, UMMA::Major::K>{});

  auto bM = tile_size<0>(tiled_mma);              // MmaM (128 or 64) == weight-N tile
  auto bN = tile_size<1>(tiled_mma);              // MmaN == M-tile
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};   // 16 * 4 = 64
  auto mma_tiler = make_shape(bM, bN, bK);

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
  // Pipeline depth: PIPE_STAGES super-stages, chosen as the largest that fits SMEM
  // for THIS (MmaM,MmaN); PIPE_SLOTS = PIPE_STAGES*KMERGE 64-wide atom slots.
  constexpr int PIPE_STAGES = fusenorm_pipe_stages(MmaM, MmaN, KM);
  constexpr int PIPE_SLOTS  = PIPE_STAGES * KM;
  // Guard: fusenorm_pipe_stages() clamps the super-stage count to >=2, which
  // silently ASSUMES two super-stages fit FUSENORM_PIPE_SMEM_BUDGET. When that is
  // false (e.g. BM=256/BN=256 with KM=4 -> 2*128KB = 256KB > opt-in), the
  // clamp overflows and only fails at RUNTIME (cudaFuncSetAttribute ->
  // "invalid argument"). Catch that class of config at COMPILE time instead.
  static_assert(2 * KM * (MmaM + MmaN) * MMA_K * 2 <= FUSENORM_PIPE_SMEM_BUDGET,
                "KM too large for this (MmaM,MmaN): two super-stages exceed "
                "the SMEM budget. Lower KMERGE or cap BN for this tile.");
  // Append a trailing PIPE (=PIPE_SLOTS) mode so the SMEM ring buffer holds every
  // 64-wide atom slot (PIPE_STAGES super-stages x KMERGE atoms); the resulting
  // layout is (MMA,MMA_M,MMA_K,PIPE_SLOTS) (A) / (MMA,MMA_N,MMA_K,PIPE_SLOTS) (B).
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<ElementA>{},
                                           append(mma_shape_A, Int<PIPE_SLOTS>{}));
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<ElementB>{},
                                           append(mma_shape_B, Int<PIPE_SLOTS>{}));

  // Barrier arrays are per SUPER-stage (PIPE_STAGES), not per atom slot.
  using SMEMStorage = SharedStorage<ElementA, ElementB, decltype(sA_layout), decltype(sB_layout), PIPE_STAGES>;

  // Trivial 1-CTA cluster: M=1, N=1, K=1 (no pairing, no multicast).
  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  // ---- The one-time expensive part: encode the TMA descriptors ----
  // The TMA descriptor describes a SINGLE stage's tile, so slice off the PIPE mode.
  Copy_Atom tma_atom_A = make_tma_atom_A_sm100(
      SM90_TMA_LOAD{}, mA, sA_layout(_,_,_,Int<0>{}), mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));

  Copy_Atom tma_atom_B = make_tma_atom_B_sm100(
      SM90_TMA_LOAD{}, mB, sB_layout(_,_,_,Int<0>{}), mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape), size<2>(cluster_shape));
  // Persistent 1-D grid: a single linear axis of num_ctas CTAs (one CTA per tile
  // slot). Each CTA strides over its share of the (gM x gN x ks) tile space via
  // the kernel's persistent `for (work ...)` loop. gM = MMA-M tiles (weight N rows
  // / bM), gN = MMA-N tiles (activation M rows / bN), ks = split-K depth. The tile
  // config guarantees gM*gN*ks (== base_tiles*ks) <= 148 CTAs, so this is a single
  // wave; the loop structure stays correct if that ever exceeds the SM cap.
  const int gM       = (Md + int(bM) - 1) / int(bM);   // weight   N-row tiles
  const int gN       = (Nd + int(bN) - 1) / int(bN);   // activation M-row tiles
  const int num_ctas = gM * gN * ks;                   // total tiles == CTA count
  dim3 dimGrid(num_ctas, 1, 1);
  int  smemBytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &gemm_device<SMEMStorage,
                                  decltype(mA_tma), decltype(mB_tma),
                                  decltype(make_tensor(make_gmem_ptr((ElementC*)nullptr),
                                           make_layout(make_shape(Md, Nd), make_stride(Int<1>{}, Md)))),
                                  decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                  decltype(tma_atom_A), decltype(tma_atom_B), float, KM>;

  static bool attr_set = false;   // per-instantiation; smem is compile-time constant
  if (!attr_set) {
    FUSENORM_CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    attr_set = true;
  }

  // Capture the encoded atoms + launch config by value. Only the output-pointer-
  // dependent D view is rebuilt per call (cheap: no descriptor encode). ws-always:
  // the GEMM ALWAYS writes fp32 partials; the fused reduce+RMSNorm kernel then
  // sums the ks slabs, RMS-normalizes each row (y*rms*rms_w), and writes bf16 D.
  return [=](ElementC* dD, const float* rms_w, float eps, SplitKPlan const& plan,
             float alpha, cudaStream_t stream) -> cutlass::Status {
    // D transposed view: logical (N,M), element (n,m) -> physical D[m,n] = m*N + n.
    Layout layout_D = make_layout(make_shape(Md, Nd), make_stride(Int<1>{}, Md));
    Tensor mD = make_tensor(make_gmem_ptr(dD), layout_D);
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
    // Precompute the grid-unpack divisors (magic numbers) ONCE per launch; the
    // kernel uses them for the hot work->(kc,nt,mt) unpack instead of IDIV/IMOD.
    cutlass::FastDivmod ks_divmod(plan.ks < 1 ? 1 : plan.ks);
    cutlass::FastDivmod gN_divmod(gN   < 1 ? 1 : gN);
    cutlass::Status st = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                             mA_tma, mB_tma, mD,
                                             mma_tiler, tiled_mma, cluster_shape,
                                             tma_atom_A, tma_atom_B, alpha,
                                             plan.d_ws,
                                             plan.ks, plan.ktiles_per_split, gM, gN,
                                             ks_divmod, gN_divmod);
    if (st != cutlass::Status::kSuccess) return st;

    // Fused split-K reduce + RMSNorm (ALWAYS runs; ws-always). ROW-PER-BLOCK:
    // one block per output row (activation token). complex_gemm STEP 2 only
    // reduce+norms y1 = the leading FUSENORM_NORM_DIM (1536) columns; the rest of
    // the row (y2/y3/y4) stays as split-K partials in ws for step-3. blockDim
    // rounds norm_len/8 up to a full warp for the block-wide sum-of-squares.
    const int stride   = Md;                   // N: full physical ws/out row stride
    const int norm_len = (Md < FUSENORM_NORM_DIM) ? Md : FUSENORM_NORM_DIM;  // y1 width (clamp to N)
    const int num_rows = Nd;                   // M activation rows (ONE block per row)
    const int tpb      = ((norm_len >> 3) + 31) & ~31;   // round norm_len/8 up to a multiple of 32
    // PDL launch: declare a programmatic dependency on the just-launched GEMM
    // (same stream) so this kernel can prologue-overlap the GEMM tail.
    cudaLaunchAttribute rattr[1];
    rattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    rattr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t rcfg = {};
    rcfg.gridDim          = dim3(num_rows, 1, 1);   // exactly one block per output row
    rcfg.blockDim         = dim3(tpb, 1, 1);
    rcfg.dynamicSmemBytes = 0;
    rcfg.stream           = stream;
    rcfg.attrs            = rattr;
    rcfg.numAttrs         = 1;
    cudaError_t rerr = fusenorm_launch_reduce(plan.ks, rcfg, dD, plan.d_ws,
                                              rms_w, eps, norm_len, stride, num_rows);
    if (rerr != cudaSuccess) return cutlass::Status::kErrorInternal;
    return cutlass::Status::kSuccess;
  };
}

// Compile-time BN dispatch: instantiate build_launch_fn<MmaM, BN> for every
// BN = 16,32,...,FUSENORM_MAX_BN (step 16) and pick the one matching mma_n. The
// recursion terminates via `if constexpr` at BN > FUSENORM_MAX_BN.
template <int MmaM, int BN, int KM>
static FuseNormLaunchFn dispatch_bn(int mma_n, ElementA const* dA, ElementB const* dB,
                                  int M, int N, int K, int ks) {
  // BN is ALWAYS a multiple of 16: the epilogue TMEM-load atom + K-major SW128
  // SMEM swizzle atom require MMA-N%16 (non-16 multiples fail cute's compile-time
  // divisibility/AtomTVLayout static_asserts), independent of the 1SM/2SM split.
  constexpr int kBnStep = 16;
  if constexpr (BN > FUSENORM_MAX_BN) {
    return FuseNormLaunchFn{};
  } else {
    if (mma_n == BN) return build_launch_fn<MmaM, BN, KM>(dA, dB, M, N, K, ks);
    return dispatch_bn<MmaM, BN + kBnStep, KM>(mma_n, dA, dB, M, N, K, ks);
  }
}

// Dispatch to the right compile-time (MmaM=N-tier, MmaN=M-tier) instantiation,
// building (encoding) the launch closure exactly once. MmaN may be ANY multiple
// of 16 up to FUSENORM_MAX_BN; MmaM is 64 or 128 (1SM MMA-M). The raw-PTX epilogue
// (tcgen05.ld.32x32b.x8, active warps = BM/32) supports MMA-M<128, so BM=64 is now
// a live tile choice -- it doubles the weight-N tile count (finer split for better
// 148-SM wave fill on small decode shapes).
// Compile-time KMERGE dispatch: pick the KM=1 or KM=2 instantiation at runtime
// (the only super-stage folds worth sweeping -- KM>=4 overflows the SMEM budget
// at BN=128 per the build_launch_fn static_assert, and past sweeps show no gain).
template <int MmaM>
static FuseNormLaunchFn dispatch_km(int km, int mma_n, ElementA const* dA, ElementB const* dB,
                                  int M, int N, int K, int ks) {
  if (km == 1) return dispatch_bn<MmaM, 16, 1>(mma_n, dA, dB, M, N, K, ks);
  if (km == 2) return dispatch_bn<MmaM, 16, 2>(mma_n, dA, dB, M, N, K, ks);
  return FuseNormLaunchFn{};   // KMERGE outside {1,2} not instantiated
}

static FuseNormLaunchFn dispatch_build(int mma_m, int mma_n, int km,
                                     ElementA const* dA, ElementB const* dB,
                                     int M, int N, int K, int ks) {
  if (mma_m == 128) return dispatch_km<128>(km, mma_n, dA, dB, M, N, K, ks);
  if (mma_m == 64)  return dispatch_km< 64>(km, mma_n, dA, dB, M, N, K, ks);
  return FuseNormLaunchFn{};
}

// ============================================================
//  Public operator API (mirrors gemm_kernel.cuh: setup / run / free)
//  FUSED GEMM + RMSNorm: run(ctx, out, rms_w, eps) computes
//      D = A * B                              (bf16 GEMM, fp32 accumulate)
//      D[m,:] = D[m,:] * rsqrt(mean(D[m,:]^2)+eps) * rms_w   (per-token RMSNorm)
//  rms_w has length N (weight features); eps default 1e-6 (caller-supplied).
// ============================================================
struct FuseNormCtx {
  const ElementA* dA = nullptr;   // activation (M,K) row-major
  const ElementB* dB = nullptr;   // weight     (N,K) row-major
  int M = 0, N = 0, K = 0;
  int mma_m = 0, mma_n = 0;       // MMA-M<-N tier, MMA-N<-M tile
  int km = KMERGE;               // super-stage fold factor actually launched
  SplitKPlan plan;
  FuseNormLaunchFn launch_fn;       // pre-built (TMA descriptors encoded ONCE)
  // Reporting fields (names mirror GemmCtx for the benchmark driver).
  int block_m = 0, block_n = 0, ks = 1, total_tiles = 0, num_blocks = 0;
  bool valid = false;
};

// Apply an ALREADY-DECIDED TileConfig: allocate the split-K workspace ONCE and
// PRE-BUILD the launch closure (TMA descriptor encode happens here, not per run
// -- like GemmCtx's d_desc_A/B). Shared by fusenorm_setup / fusenorm_setup_forced.
static void fusenorm_apply_config(FuseNormCtx& ctx, const bf16_t* A, const bf16_t* B,
                                  int M, int N, int K, TileConfig cfg) {
  ctx.dA = reinterpret_cast<const ElementA*>(A);
  ctx.dB = reinterpret_cast<const ElementB*>(B);
  ctx.M = M; ctx.N = N; ctx.K = K;

  ctx.valid = cfg.valid;
  if (!ctx.valid) { ctx.plan = SplitKPlan{}; return; }

  ctx.mma_m       = cfg.mma_m;
  ctx.mma_n       = cfg.mma_n;
  ctx.km          = cfg.km;
  ctx.plan        = alloc_splitk(/*Md=*/N, /*Nd=*/M, cfg);
  ctx.block_m     = cfg.mma_m;
  ctx.block_n     = cfg.mma_n;
  ctx.ks          = cfg.ks;
  ctx.total_tiles = cfg.base_tiles * cfg.ks;
  ctx.num_blocks  = cfg.base_tiles * cfg.ks;       // one CTA per tile (1SM, no pairing)

  // Encode TMA descriptors ONCE and capture them in the launch closure.
  ctx.launch_fn = dispatch_build(ctx.mma_m, ctx.mma_n, ctx.km, ctx.dA, ctx.dB,
                                 ctx.M, ctx.N, ctx.K, ctx.plan.ks);
  if (!ctx.launch_fn) ctx.valid = false;
}

// Build a TileConfig with a FORCED (BM,BN); ks is auto (same wave-fill rule as
// choose_tile_config) unless force_ks>0, in which case ks is pinned (clamped to
// [1,8]=reduce hard cap, and renormalized like fusenorm_split_depth). force_km>0
// pins the super-stage fold (only 1/2 are instantiated). For sweeps.
static TileConfig make_forced_tile_config(int M, int N, int K, int bm, int bn,
                                          int force_ks = 0, int force_km = 0) {
  TileConfig c;
  if (bm <= 0 || bn <= 0 || N % bm != 0 || K % MMA_K != 0) return c;  // invalid
  const int mtiles = (M + bn - 1) / bn;
  const int base   = (N / bm) * mtiles;
  int ks, kps;
  if (force_ks > 0) {
    const int ktiles_total = K / MMA_K;
    ks = force_ks;
    if (ks > 8)            ks = 8;              // reduce kernel hard cap
    if (ks > ktiles_total) ks = ktiles_total;
    if (ks < 1)            ks = 1;
    kps = (ktiles_total + ks - 1) / ks;         // ceil
    ks  = (ktiles_total + kps - 1) / kps;       // renormalize (mirror fusenorm_split_depth)
  } else {
    fusenorm_split_depth(base, K, ks, kps);       // ks auto (same rule as choose_tile_config)
  }
  c.mma_m = bm; c.mma_n = bn; c.ks = ks;
  c.ktiles_per_split = kps; c.base_tiles = base;
  c.km = (force_km > 0) ? force_km : KMERGE;    // pin fold factor (else production default)
  c.valid = true;
  return c;
}

// Standard setup: adaptive tile (BN=16 + auto ks) via choose_tile_config.
static void fusenorm_setup(FuseNormCtx& ctx, const bf16_t* A, const bf16_t* B,
                           int M, int N, int K) {
  fusenorm_apply_config(ctx, A, B, M, N, K, choose_tile_config(M, N, K));
}

// Setup with a FORCED (BM,BN[,ks,km]) tile (BM defaults to 128, ks/km default to
// production); only for benchmarking / sweeps -- production uses fusenorm_setup.
static void fusenorm_setup_forced(FuseNormCtx& ctx, const bf16_t* A, const bf16_t* B,
                                  int M, int N, int K, int force_bn, int force_bm = 128,
                                  int force_ks = 0, int force_km = 0) {
  fusenorm_apply_config(ctx, A, B, M, N, K,
                        make_forced_tile_config(M, N, K, force_bm, force_bn, force_ks, force_km));
}

// Pure launch: reuses the pre-encoded TMA descriptors (no per-call encode).
// rms_w: device pointer, length N (weight features); eps: RMSNorm epsilon.
static void fusenorm_run(FuseNormCtx& ctx, bf16_t* out, const float* rms_w,
                         float eps = 1e-6f, cudaStream_t stream = 0) {
  if (!ctx.valid || !ctx.launch_fn) return;
  ctx.launch_fn(reinterpret_cast<ElementC*>(out), rms_w, eps, ctx.plan, 1.0f, stream);
}

static void fusenorm_free(FuseNormCtx& ctx) {
  if (ctx.plan.d_ws) cudaFree(ctx.plan.d_ws);
  ctx.plan = SplitKPlan{};
  ctx.launch_fn = nullptr;
  ctx.valid = false;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
