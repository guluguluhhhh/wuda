#pragma once

// Shared low-level support for the CSA swap-AB front projection.
// The retired no-swap kernel intentionally does not live in this header.

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cluster_mma_fp8.cuh"

namespace front_mixed_csa {

constexpr int K = 7168;
constexpr int N = 3072;
constexpr int N_FP8 = 2048;  // wq_a 1536 | window KV 512
constexpr int TMA_K_BF16 = 64;
constexpr int TMA_K_FP8 = 128;

constexpr uint64_t TMA_CACHE_HINT = 0x1000000000000000ull;
constexpr uint32_t TMA_2SM_PEER_MASK = 0xfeffffffu;

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

__device__ __forceinline__ void st_shared_u32(void* ptr, uint32_t value) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("st.shared.u32 [%0], %1;" :: "r"(addr), "r"(value) : "memory");
}

struct SmemDescriptor {
  uint32_t lo;
  uint32_t hi;
};

__device__ __forceinline__ SmemDescriptor make_smem_desc(void* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  return {addr >> 4, 0x40004040u};
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

namespace hc_tail {
constexpr int HC = 4;
constexpr int N_OUT = (2 + HC) * HC;
constexpr int SINKHORN_ITERS = 20;
}  // namespace hc_tail

struct HcTailArgs {
  const float* mix = nullptr;    // [m,24], reduced and RMS-folded
  const float* base = nullptr;   // [24]
  const float* scale = nullptr;  // [3]
  float hc_eps = 1e-6f;
  int m = 0;
  float* post_out = nullptr;     // [m,4]
  float* comb_out = nullptr;     // [m,4,4]
};

// Direct front-projection handoff used by the decode chain. When main_state
// is non-null, only q [0,1536) is written to the bf16 output; the remaining
// projection segments are published directly from the fp32 accumulators.
constexpr int APE_RATIO = 4;

struct FrontEmitArgs {
  float* main_state = nullptr;         // [blocks, ring, 2048]
  const float* main_ape = nullptr;     // [4,1024]
  const int64_t* main_state_row = nullptr; // [M], -1 skips a row
  const int* ape_phase = nullptr;      // [M]
  float* idx_state = nullptr;          // [blocks, ring, 512]
  const int64_t* idx_state_row = nullptr;  // [M], -1 skips a row
  const float* idx_ape = nullptr;      // [4,256]
  float* win_y2 = nullptr;             // [M,512]
  float* w64 = nullptr;                // [M,64]
};

__device__ __forceinline__ float hc_tail_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ static void hc_tail_run(HcTailArgs const& a) {
  using namespace hc_tail;
  const int lane = threadIdx.x & 31;
  const float s_post = a.scale[1];
  const float s_comb = a.scale[2];
  for (int pos = blockIdx.x; pos < a.m; pos += gridDim.x) {
    const float mix = (lane < N_OUT - HC)
        ? __ldg(a.mix + static_cast<size_t>(pos) * N_OUT + HC + lane) : 0.f;

    if (lane < HC) {
      a.post_out[static_cast<size_t>(pos) * HC + lane] =
          2.0f * hc_tail_sigmoid(mix * s_post + a.base[HC + lane]);
    }

    float v = __shfl_sync(0xffffffffu, mix, (lane + HC) & 31);
    v = (lane < HC * HC) ? v * s_comb + a.base[2 * HC + lane] : 0.f;

    float max_v = v;
    #pragma unroll
    for (int off = 1; off < HC; off <<= 1) {
      max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffffu, max_v, off));
    }
    const float e = __expf(v - max_v);
    float row_sum = e;
    #pragma unroll
    for (int off = 1; off < HC; off <<= 1) {
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off);
    }
    v = e / row_sum + a.hc_eps;
    float col_sum = v;
    #pragma unroll
    for (int off = HC; off < HC * HC; off <<= 1) {
      col_sum += __shfl_xor_sync(0xffffffffu, col_sum, off);
    }
    v /= col_sum + a.hc_eps;
    #pragma unroll 1
    for (int iter = 0; iter < SINKHORN_ITERS - 1; ++iter) {
      row_sum = v;
      #pragma unroll
      for (int off = 1; off < HC; off <<= 1) {
        row_sum += __shfl_xor_sync(0xffffffffu, row_sum, off);
      }
      v /= row_sum + a.hc_eps;
      col_sum = v;
      #pragma unroll
      for (int off = HC; off < HC * HC; off <<= 1) {
        col_sum += __shfl_xor_sync(0xffffffffu, col_sum, off);
      }
      v /= col_sum + a.hc_eps;
    }
    if (lane < HC * HC) {
      a.comb_out[static_cast<size_t>(pos) * HC * HC + lane] = v;
    }
  }
}

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

}  // namespace front_mixed_csa
