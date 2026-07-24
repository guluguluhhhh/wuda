// ============================================================
// topk_v2.cu — __global__ kernels + host dispatcher + PyTorch binding for the
// DeepSeek-V4 (DSA indexer) top-k, migrated from sglang:
//   csrc/deepseek_v4/topk_v2.cuh  (host/kernels; tvm-ffi -> torch here)
//   include/sgl_kernel/deepseek_v4/topk_impl.cuh -> include/topk_v2.cuh (device)
//
// Runs AFTER the score-attention kernel (mqa_logits_fp4 decode) as a SEPARATE
// launch (no fusion): scores = decode logits [B, stride] fp32 (stride % 4 == 0,
// use out_dtype=float32), seq_lens = per-batch valid lengths (rows' -inf tails
// beyond seq_len are never read). DSV4 decode: topk = index_topk = 512.
//
// Dispatch (verbatim from upstream):
//   max_seq_len <= 8192            -> main kernel, Register<2> (1 read, regs)
//   max_seq_len <= 16384           -> main kernel, Register<4>
//   max_seq_len <= cluster_floor   -> main kernel, + Streaming (2 global passes)
//   above cluster_floor:
//     batch <= 120 (persistent clusters) -> fused small-batch cluster kernel
//     else -> persistent 8-CTA-cluster pool (long items, planned by topk_plan)
//             + main kernel level 3 (short items + epilogue transform)
//
// [MEGAKERNEL EDIT]s vs upstream:
//   1. tvm-ffi TensorMatcher/LaunchKernel -> torch checks + <<<>>> launches
//      (__cluster_dims__ keeps the compile-time cluster shape).
//   2. kPDL instantiated FALSE: standalone stream-ordered launches (PDL only
//      helped when fused behind the scorer via programmatic launch; the
//      device-side griddepcontrol code stays, gated by the template).
//   3. C++20 std::has_single_bit / countr_zero -> builtin equivalents.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "topk_v2.cuh"

namespace {

namespace impl = device::topk;
using impl::TopKProblem;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterMaxBatch = 512;
// Persistent pool size: upstream's B200(occ2) pick; B300 has the same 148-SM
// GPC geometry, keep as-is (tunable).
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;      // optional raw (pre-transform) indices output; nullptr if unused
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    return TopKProblem{
        scores + batch_id * score_stride,                            // in
        page_indices + batch_id * k,                                 // out
        raw_indices != nullptr ? raw_indices + batch_id * k : nullptr,
        page_table + batch_id * page_table_stride,                   // page_table
        topk,
        seq_len,
        page_bits,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};

/**
 * \brief Persistent cluster kernel for the long items (short items are handled
 * by the separate topk_main_kernel).
 */
template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Cluster::Smem> smem;
  const uint32_t num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
#pragma unroll 1
  for (uint32_t w = blockIdx.x; w < num_cluster_items; w += kNumPersistentClusters) {
    const auto it = params.item(w);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    Cluster::forward<false>(problem, &smem);
    __syncthreads();
  }
}

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      f(tx, i);
    }
  }
}

template <bool kPDL>
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    problem.transform_output(tx, tx < problem.seq_len ? static_cast<int32_t>(tx) : -1);
  });
}

SGL_DEVICE void problem_transform(TopKProblem& problem, int32_t* output_ptr) {
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t source_index[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
  problem.out = output_ptr;
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { problem.transform_output(tx, source_index[i]); });
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * kLevel: 0: <=8192 reg2 | 1: <=16384 reg4 | 2: + streaming | 3: + cluster epilogue
 */
template <bool kPDL, int kLevel>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  constexpr bool kHandleCluster = (kLevel == 3);
  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    Register4::forward<kPDL>(problem, &smem);
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    // if using cluster, we can delay the PDL wait
    constexpr bool kPDLEarly = kPDL && !kHandleCluster;
    constexpr bool kPDLFinal = kPDL && kHandleCluster;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {  // cluster path do nothing here
      problem.out = params.get_output_ptr(blockIdx.x);
    }
    device::PDLWaitPrimary<kPDLFinal>();
  }

  // page-table transform pass (gathers kept out of the hot scatter loop),
  // then trigger the dependent kernel only after the full output is written.
  device::PDLTriggerSecondary<kPDL>();
  __syncthreads();
  problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  __shared__ impl::MaxSmem<Streaming::Smem, Cluster::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  // randomly elect one worker rank to avoid workload imbalance
  const auto worker_rank = blockIdx.x % kClusterSize;

  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (blockIdx.y == worker_rank) Register4::forward<kPDL>(problem, &smem);
  } else if (problem.seq_len <= params.cluster_floor) {
    if (blockIdx.y == worker_rank) Streaming::forward<kPDL>(problem, &smem);
  } else {
    auto cluster = cooperative_groups::this_cluster();
    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
    cluster.sync();
  }

  device::PDLWaitPrimary<kPDL>();
  __syncthreads();
  if (blockIdx.y == worker_rank) problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan_kernel(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate (threshold T_j, cap_j) pairs, T strictly increasing. The plan lowers
  // cluster_threshold to T_j while #(items with seq_len > T_j) <= cap_j; cap_j is
  // the measured cluster-vs-streaming crossover (B200, occ2) and grows with T.
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {65536, 30},    // (65536,98304]:    ~1 pool wave, streams beyond 30
      {98304, 48},    // (98304,131072]
      {131072, 60},   // (131072,196608]
      {196608, 80},   // (196608,262144]
      {262144, 112},  // (262144,393216]
      {393216, 128},  // (393216,inf)
  };
  constexpr uint32_t kNumCandidates = sizeof(kCandidates) / sizeof(kCandidates[0]);
  static_assert(kCandidates[0].threshold == kClusterFloor);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) atomicAdd(&s_counts[count - 1], 1);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kCandidates[kNumCandidates - 1].threshold;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];  // # items with seq_len > kCandidates[j].threshold
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    GlobalMetadata gm;
    gm.cluster_threshold = cluster_threshold;
    gm.num_cluster_items = s_count;
    *g = gm;
  }
}

}  // namespace

// ======================== Host: plan + transform ========================

// Device-side plan (cheap 1-block kernel): picks cluster_threshold from the
// seq_len distribution and compacts the long items into metadata[1..N].
// metadata: int32 [B+1, 2], caller-allocated (contents fully overwritten).
static void topk_v2_plan(torch::Tensor seq_lens, torch::Tensor metadata,
                         int64_t static_cluster_threshold) {
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.is_contiguous() &&
                seq_lens.scalar_type() == torch::kInt32, "seq_lens must be CUDA i32 [B]");
    const auto B = seq_lens.numel();
    TORCH_CHECK(metadata.is_cuda() && metadata.is_contiguous() &&
                metadata.scalar_type() == torch::kInt32 &&
                metadata.numel() == 2 * (B + 1), "metadata must be CUDA i32 [B+1, 2]");
    auto stream = at::cuda::getCurrentCUDAStream();
    topk_plan_kernel<<<1, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(seq_lens.data_ptr<int>()),
        reinterpret_cast<PlanItem*>(metadata.data_ptr<int>()),
        static_cast<uint32_t>(B), static_cast<uint32_t>(static_cluster_threshold));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "topk_plan launch failed: ", cudaGetErrorString(err));
}

// Main transform: per-batch top-k over scores[b, :seq_lens[b]] -> page-table-
// transformed indices in page_indices [B, K] (+ optional raw indices).
static void topk_v2_transform(torch::Tensor scores, torch::Tensor seq_lens,
                              torch::Tensor page_table, torch::Tensor page_indices,
                              int64_t page_size, torch::Tensor metadata,
                              c10::optional<torch::Tensor> raw_indices) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat &&
                scores.dim() == 2 && scores.stride(1) == 1,
                "scores must be CUDA fp32 [B, L] row-major");
    const int64_t B = scores.size(0), L = scores.size(1), S = scores.stride(0);
    TORCH_CHECK(S % 4 == 0, "score row stride must be a multiple of 4 (16B vector loads), got ", S);
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.is_contiguous() &&
                seq_lens.scalar_type() == torch::kInt32 && seq_lens.numel() == B, "seq_lens [B] i32");
    TORCH_CHECK(page_table.is_cuda() && page_table.scalar_type() == torch::kInt32 &&
                page_table.dim() == 2 && page_table.size(0) == B && page_table.stride(1) == 1,
                "page_table must be CUDA i32 [B, P] row-major");
    TORCH_CHECK(page_indices.is_cuda() && page_indices.is_contiguous() &&
                page_indices.scalar_type() == torch::kInt32 &&
                page_indices.dim() == 2 && page_indices.size(0) == B, "page_indices [B, K] i32");
    TORCH_CHECK(metadata.is_cuda() && metadata.is_contiguous() &&
                metadata.scalar_type() == torch::kInt32 && metadata.numel() == 2 * (B + 1),
                "metadata must be CUDA i32 [B+1, 2] (from topk_v2_plan)");
    const int64_t K = page_indices.size(1);
    TORCH_CHECK(K > 0 && K <= (int64_t)kMaxTopK, "topk must be in (0, ", kMaxTopK, "]");
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a power of 2");

    int32_t* raw_ptr = nullptr;
    if (raw_indices.has_value()) {
        auto& r = *raw_indices;
        TORCH_CHECK(r.is_cuda() && r.is_contiguous() && r.scalar_type() == torch::kInt32 &&
                    r.numel() == B * K, "raw_indices must be CUDA i32 [B, K]");
        raw_ptr = r.data_ptr<int>();
    }

    const auto page_bits = static_cast<uint32_t>(__builtin_ctzll((uint64_t)page_size));
    const auto batch_size = static_cast<uint32_t>(B);
    const auto max_seq_len = static_cast<uint32_t>(L);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Small-batch fused-cluster floor (upstream: one wave of 15 8-CTA clusters at
    // occ2 stays latency-bound, so the 8-way split beats streaming from ~32K).
    constexpr uint32_t kClusterFloorSmall = 32768;
    constexpr uint32_t kSmallBatchLowFloor = 15;
    TopKLaunchParams params;
    params.scores            = scores.data_ptr<float>();
    params.seq_lens          = seq_lens.data_ptr<int>();
    params.page_table        = page_table.data_ptr<int>();
    params.page_indices      = page_indices.data_ptr<int>();
    params.raw_indices       = raw_ptr;
    params.metadata          = reinterpret_cast<const PlanItem*>(metadata.data_ptr<int>());
    params.score_stride      = S;
    params.page_table_stride = page_table.stride(0);
    params.topk              = static_cast<uint32_t>(K);
    params.page_bits         = page_bits;
    params.cluster_floor     = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor;

    // [MEGAKERNEL EDIT] standalone launches: kPDL=false (stream order supplies
    // the plan->transform and scorer->topk dependencies).
    constexpr bool kUsePDL = false;
    const bool use_cluster = (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
    if (use_cluster) {
        if (batch_size <= kNumPersistentClusters) {
            dim3 grid(batch_size, kClusterSize);
            topk_small_batch_kernel<kUsePDL><<<grid, kBlockSize, 0, stream>>>(params);
        } else {
            const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
            dim3 grid(num_clusters, kClusterSize);
            topk_persistent_cluster_kernel<kUsePDL><<<grid, kBlockSize, 0, stream>>>(params);
            topk_main_kernel<kUsePDL, /*kLevel=*/3><<<batch_size, kBlockSize, 0, stream>>>(params);
        }
    } else if (max_seq_len <= kReg2MaxSeqLen) {
        topk_main_kernel<kUsePDL, /*kLevel=*/0><<<batch_size, kBlockSize, 0, stream>>>(params);
    } else if (max_seq_len <= kReg4MaxSeqLen) {
        topk_main_kernel<kUsePDL, /*kLevel=*/1><<<batch_size, kBlockSize, 0, stream>>>(params);
    } else {
        topk_main_kernel<kUsePDL, /*kLevel=*/2><<<batch_size, kBlockSize, 0, stream>>>(params);
    }
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "topk launch failed: ", cudaGetErrorString(err));
}

// Convenience one-shot: allocates page_indices (+metadata), runs plan (only when
// the cluster path can trigger) + transform. For benchmarking use the *_plan /
// *_transform pair with hoisted buffers instead.
static torch::Tensor topk_v2(torch::Tensor scores, torch::Tensor seq_lens,
                             torch::Tensor page_table, int64_t topk, int64_t page_size,
                             c10::optional<torch::Tensor> raw_indices) {
    const int64_t B = scores.size(0);
    auto i32 = scores.options().dtype(torch::kInt32);
    auto page_indices = torch::empty({B, topk}, i32);
    auto metadata = torch::zeros({B + 1, 2}, i32);
    if ((uint64_t)scores.size(1) > 32768)   // plan needed only for possible cluster routing
        topk_v2_plan(seq_lens, metadata, /*static_cluster_threshold=*/0);
    topk_v2_transform(scores, seq_lens, page_table, page_indices, page_size, metadata, raw_indices);
    return page_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_v2_plan", &topk_v2_plan,
          "device-side plan: pick cluster_threshold + compact long items into metadata [B+1,2] i32",
          py::arg("seq_lens"), py::arg("metadata"), py::arg("static_cluster_threshold") = 0);
    m.def("topk_v2_transform", &topk_v2_transform,
          "DSV4 top-k over scores[b, :seq_lens[b]] -> page-transformed indices [B,K] "
          "(+ optional raw indices). scores fp32 [B,L], row stride % 4 == 0",
          py::arg("scores"), py::arg("seq_lens"), py::arg("page_table"),
          py::arg("page_indices"), py::arg("page_size"), py::arg("metadata"),
          py::arg("raw_indices") = c10::nullopt);
    m.def("topk_v2", &topk_v2,
          "one-shot convenience (alloc + plan + transform) -> page_indices [B, topk] i32",
          py::arg("scores"), py::arg("seq_lens"), py::arg("page_table"),
          py::arg("topk"), py::arg("page_size") = 64,
          py::arg("raw_indices") = c10::nullopt);
}
