#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "o_proj_b_tp2_symm.cuh"

namespace {

constexpr int kMMax = 128;
constexpr int kN = 7168;
constexpr int kK = 8192;
constexpr int kStoreBlockM = 16;
constexpr int kBlockN = 128;
constexpr int kBlockK = 128;
constexpr int kNumSMs = 148;
constexpr int kClusterSize = 2;
constexpr int kThreads = 256;
constexpr int kDynamicSmemBytes = 230188;
constexpr int kGridDoneElements =
    deep_gemm::o_proj_b_tp2_symm::kGridDoneElements;
constexpr int kValuesPerVector = 8;
constexpr int kVectorsPerRow = kN / kValuesPerVector;
constexpr int kBenchmarkReadyOffset = 16;
constexpr int kSelectReadyOffset = 4;
constexpr int kMhcReadyOffset = 64;
constexpr uint64_t kBenchmarkEnqueueGraceNs = 1'000'000;

using KernelDType = cutlass::bfloat16_t;

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void check_driver(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(
      std::string(operation) + ": " + (message == nullptr ? "unknown" : message));
}

CUtensorMap make_2d_map(
    void* ptr, CUtensorMapDataType dtype, int element_bytes,
    uint64_t inner_dim, uint64_t outer_dim, uint64_t outer_stride_elements,
    uint32_t box_inner, uint32_t box_outer,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap result{};
  const cuuint64_t global_dims[2] = {inner_dim, outer_dim};
  const cuuint64_t global_strides[1] = {
      outer_stride_elements * static_cast<uint64_t>(element_bytes)};
  const cuuint32_t box_dims[2] = {box_inner, box_outer};
  const cuuint32_t element_strides[2] = {1, 1};
  check_driver(
      cuTensorMapEncodeTiled(
          &result, dtype, 2, ptr, global_dims, global_strides, box_dims,
          element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return result;
}

template <int kBlockM>
CUtensorMap make_a_map(const torch::Tensor& tensor, int m) {
  return make_2d_map(
      tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kK, m, tensor.stride(0), kBlockK, kBlockM / kClusterSize,
      CU_TENSOR_MAP_SWIZZLE_128B);
}

CUtensorMap make_b_map(const torch::Tensor& tensor) {
  return make_2d_map(
      tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kK, kN, tensor.stride(0), kBlockK, kBlockN,
      CU_TENSOR_MAP_SWIZZLE_128B);
}

CUtensorMap make_output_map(void* pointer, int m) {
  // The 128B swizzle makes each TMA store cover 64 BF16 columns.
  return make_2d_map(
      pointer, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      kN, m, kN, 64, kStoreBlockM,
      CU_TENSOR_MAP_SWIZZLE_128B);
}

CUtensorMap make_scale_map(
    const torch::Tensor& tensor, int mn, int block_mn) {
  const int aligned_mn = (mn + 3) / 4 * 4;
  constexpr int kPackedScaleK = kK / (128 * 4);
  return make_2d_map(
      tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
      aligned_mn, kPackedScaleK, aligned_mn,
      block_mn, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <uint32_t kLoadStages, uint32_t kStoreStages, uint32_t kBlockM,
          bool kRuntimeSelect, bool kTilePublish,
          bool kStoreSecondDestination, bool kVectorPeerStore>
auto kernel_ptr() {
  return &deep_gemm::o_proj_b_tp2_symm_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      128, 128, 128,
      0, kN, kK,
      kBlockM, kBlockN, kBlockK,
      1,
      128, 128, 128,
      kLoadStages, kStoreStages,
      128, 128,
      2, true,
      kNumSMs,
      kRuntimeSelect,
      kTilePublish,
      kStoreSecondDestination,
      kVectorPeerStore,
      true, true,
      deep_gemm::GemmType::Normal, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, KernelDType,
      deep_gemm::epilogue::transform::EpilogueIdentity>;
}

void validate_tensor(
    const torch::Tensor& tensor, at::ScalarType dtype,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

__device__ __forceinline__ uint32_t add_bf16x2(
    uint32_t lhs, uint32_t rhs) {
  union Packed {
    uint32_t bits;
    __nv_bfloat162 values;
  } a, b, result;
  a.bits = lhs;
  b.bits = rhs;
  result.values = __hadd2(a.values, b.values);
  return result.bits;
}

__device__ __forceinline__ uint4 add_bf16x8(
    const uint4& lhs, const uint4& rhs) {
  return uint4{
      add_bf16x2(lhs.x, rhs.x), add_bf16x2(lhs.y, rhs.y),
      add_bf16x2(lhs.z, rhs.z), add_bf16x2(lhs.w, rhs.w)};
}

template <bool kReducePartials, bool kSelectMode = false,
          bool kPeerOneShot = false, bool kPeerRankReady = false>
__global__ void mhc_post_kernel(
    const __nv_bfloat16* partial0, const __nv_bfloat16* partial1,
    const __nv_bfloat16* local_partial0,
    const __nv_bfloat16* local_partial1,
    const uint32_t* runtime_mode,
    uint32_t* block_generations,
    const uint32_t* local_ready,
    uint32_t* peer_ready,
    const __nv_bfloat16* residual, const float* post, const float* comb,
    __nv_bfloat16* output, int m) {
  __shared__ float mix[20];
  const int row = static_cast<int>(blockIdx.y);
  if (row >= m) return;

  if (threadIdx.x < 4) mix[threadIdx.x] = post[row * 4 + threadIdx.x];
  if (threadIdx.x < 16) mix[4 + threadIdx.x] = comb[row * 16 + threadIdx.x];
  if constexpr (kPeerRankReady) {
    if (threadIdx.x == 0) {
      using namespace deep_gemm::o_proj_b_tp2_symm;
      const uint32_t expected = load_relaxed_gpu(block_generations);
      const uint64_t wait_start = globaltimer();
      uint32_t spins = 0;
      uint32_t ready = load_relaxed_sys(local_ready);
      while (ready < expected) {
        spin_pause();
        ready = load_relaxed_sys(local_ready);
        check_spin_timeout(
            wait_start, ++spins, "consumer", expected, ready, 0);
      }
      while (load_acquire_sys(local_ready) < expected) spin_pause();
    }
  } else if constexpr (kPeerOneShot) {
    if (threadIdx.x == 0) {
      const uint32_t cta = row * gridDim.x + blockIdx.x;
      const uint32_t next = block_generations[cta] + 1;
      // This is a start barrier: the preceding GEMM kernel boundary already
      // completed the local partial, so no release fence is required here.
      asm volatile("st.volatile.global.u32 [%0], %1;" ::
                   "l"(peer_ready + kMhcReadyOffset + cta), "r"(next)
                   : "memory");
      uint32_t ready;
      do {
        asm volatile("ld.volatile.global.u32 %0, [%1];"
                     : "=r"(ready)
                     : "l"(local_ready + kMhcReadyOffset + cta)
                     : "memory");
      } while (ready != next);
      block_generations[cta] = next;
    }
  }
  __syncthreads();

  for (int vector = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       vector < kVectorsPerRow;
       vector += static_cast<int>(gridDim.x) * blockDim.x) {
    const int dim = vector * kValuesPerVector;
    const auto use_peer = !kSelectMode ||
        deep_gemm::o_proj_b_tp2_symm::load_relaxed_gpu(runtime_mode) >= 2;
    const auto* x0_base = use_peer ? partial0 : local_partial0;
    const auto* x1_base = use_peer ? partial1 : local_partial1;
    const auto* x0_ptr = x0_base + row * kN + dim;
    uint4 x_bits = *reinterpret_cast<const uint4*>(x0_ptr);
    if constexpr (kReducePartials) {
      const auto* x1_ptr = x1_base + row * kN + dim;
      x_bits = add_bf16x8(
          x_bits, *reinterpret_cast<const uint4*>(x1_ptr));
    }

    alignas(16) __nv_bfloat16 x[8];
    alignas(16) __nv_bfloat16 r[4][8];
    *reinterpret_cast<uint4*>(x) = x_bits;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<uint4*>(r[i]) = *reinterpret_cast<const uint4*>(
          residual + (row * 4 + i) * kN + dim);
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      alignas(16) __nv_bfloat16 out[8];
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        float value = mix[j] * __bfloat162float(x[lane]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          value = fmaf(
              mix[4 + i * 4 + j], __bfloat162float(r[i][lane]), value);
        }
        out[lane] = __float2bfloat16_rn(value);
      }
      *reinterpret_cast<uint4*>(output + (row * 4 + j) * kN + dim) =
          *reinterpret_cast<const uint4*>(out);
    }
  }
}

template <int kProducerBlockM>
__global__ void tile_mhc_post_kernel(
    const __nv_bfloat16* local_partial,
    const __nv_bfloat16* peer_partial,
    uint32_t* tile_generations,
    const uint32_t* local_ready,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    __nv_bfloat16* output,
    int m,
    uint32_t rank) {
  __shared__ float mix[kProducerBlockM][20];
  constexpr uint32_t kNumNBlocks = kN / kBlockN;
  constexpr uint32_t kVectorsPerNBlock = kBlockN / kValuesPerVector;
  const uint32_t n_block = blockIdx.x;
  const uint32_t m_block = blockIdx.y;
  const uint32_t row_begin = m_block * kProducerBlockM;
  const uint32_t rows = min(
      static_cast<uint32_t>(kProducerBlockM),
      static_cast<uint32_t>(m) - row_begin);
  const uint32_t post_tile_id = m_block * kNumNBlocks + n_block;
  const uint32_t consumer_generation_idx =
      deep_gemm::o_proj_b_tp2_symm::kMaxOutputTiles + post_tile_id;
  const uint32_t expected =
      deep_gemm::o_proj_b_tp2_symm::load_relaxed_gpu(
          tile_generations + consumer_generation_idx) + 1;

  for (uint32_t idx = threadIdx.x; idx < rows * 20; idx += blockDim.x) {
    const uint32_t row_in_block = idx / 20;
    const uint32_t coefficient = idx % 20;
    const uint32_t row = row_begin + row_in_block;
    mix[row_in_block][coefficient] = coefficient < 4
        ? post[row * 4 + coefficient]
        : comb[row * 16 + coefficient - 4];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    using namespace deep_gemm::o_proj_b_tp2_symm;
    const uint32_t peer_rank = rank ^ 1u;
    const uint64_t wait_start = globaltimer();
    uint32_t spins = 0;
    const uint32_t local_flag_idx =
        kTileReadyOffset + rank * kMaxOutputTiles + post_tile_id;
    const uint32_t peer_flag_idx =
        kTileReadyOffset + peer_rank * kMaxOutputTiles + post_tile_id;
    uint32_t local_flag = load_relaxed_sys(local_ready + local_flag_idx);
    uint32_t peer_flag = load_relaxed_sys(local_ready + peer_flag_idx);
    while (local_flag < expected || peer_flag < expected) {
      spin_pause();
      local_flag = load_relaxed_sys(local_ready + local_flag_idx);
      peer_flag = load_relaxed_sys(local_ready + peer_flag_idx);
      check_spin_timeout(
          wait_start, ++spins, "tile-consumer", expected,
          local_flag < expected ? local_flag : peer_flag, rank);
    }
    while (load_acquire_sys(local_ready + local_flag_idx) < expected ||
           load_acquire_sys(local_ready + peer_flag_idx) < expected) {
      spin_pause();
    }
  }
  __syncthreads();

  const uint32_t tile_vectors = rows * kVectorsPerNBlock;
  for (uint32_t vector = threadIdx.x; vector < tile_vectors;
       vector += blockDim.x) {
    const uint32_t row_in_block = vector / kVectorsPerNBlock;
    const uint32_t row = row_begin + row_in_block;
    const uint32_t vector_in_row = vector % kVectorsPerNBlock;
    const uint32_t dim =
        n_block * kBlockN + vector_in_row * kValuesPerVector;
    const uint32_t offset = row * kN + dim;
    uint4 x_bits = add_bf16x8(
        *reinterpret_cast<const uint4*>(local_partial + offset),
        *reinterpret_cast<const uint4*>(peer_partial + offset));

    alignas(16) __nv_bfloat16 x[8];
    alignas(16) __nv_bfloat16 r[4][8];
    *reinterpret_cast<uint4*>(x) = x_bits;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<uint4*>(r[i]) = *reinterpret_cast<const uint4*>(
          residual + (row * 4 + i) * kN + dim);
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      alignas(16) __nv_bfloat16 out[8];
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        float value = mix[row_in_block][j] * __bfloat162float(x[lane]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          value = fmaf(
              mix[row_in_block][4 + i * 4 + j],
              __bfloat162float(r[i][lane]), value);
        }
        out[lane] = __float2bfloat16_rn(value);
      }
      *reinterpret_cast<uint4*>(
          output + (row * 4 + j) * kN + dim) =
          *reinterpret_cast<const uint4*>(out);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    deep_gemm::o_proj_b_tp2_symm::store_relaxed_gpu(
        tile_generations + consumer_generation_idx, expected);
  }
}

void validate_mhc_common(
    const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& output, int m) {
  validate_tensor(residual, torch::kBFloat16, "residual");
  validate_tensor(post, torch::kFloat32, "post");
  validate_tensor(comb, torch::kFloat32, "comb");
  validate_tensor(output, torch::kBFloat16, "output");
  TORCH_CHECK(m >= 1 && m <= kMMax, "M must be in [1,128]");
  TORCH_CHECK(residual.sizes() == torch::IntArrayRef({m, 4, kN}) &&
                  residual.is_contiguous(),
              "residual must be contiguous BF16 [M,4,7168]");
  TORCH_CHECK(post.sizes() == torch::IntArrayRef({m, 4}) &&
                  post.is_contiguous(),
              "post must be contiguous FP32 [M,4]");
  TORCH_CHECK(comb.sizes() == torch::IntArrayRef({m, 4, 4}) &&
                  comb.is_contiguous(),
              "comb must be contiguous FP32 [M,4,4]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({m, 4, kN}) &&
                  output.is_contiguous(),
              "output must be contiguous BF16 [M,4,7168]");
}

void validate_mhc_inputs(
    const torch::Tensor& partials, const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& output, int m) {
  validate_tensor(partials, torch::kBFloat16, "partials");
  validate_mhc_common(residual, post, comb, output, m);
  TORCH_CHECK(partials.sizes() ==
                  torch::IntArrayRef({2, kMMax, kN}) &&
                  partials.is_contiguous(),
              "partials must be contiguous BF16 [2,128,7168]");
}

template <int kProducerBlockM>
void launch_tile_mhc_post(
    const __nv_bfloat16* local_partial,
    const __nv_bfloat16* peer_partial,
    uint32_t* tile_generations,
    const uint32_t* local_ready,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int m,
    uint32_t rank) {
  auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(
      residual.data_ptr<at::BFloat16>());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(
      output.data_ptr<at::BFloat16>());
  const float* post_ptr = post.data_ptr<float>();
  const float* comb_ptr = comb.data_ptr<float>();
  void* args[] = {
      &local_partial, &peer_partial, &tile_generations, &local_ready,
      &residual_ptr, &post_ptr, &comb_ptr, &output_ptr, &m, &rank};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(
      kN / kBlockN, (m + kProducerBlockM - 1) / kProducerBlockM, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  check_cuda(
      cudaLaunchKernelExC(
          &config,
          reinterpret_cast<void*>(tile_mhc_post_kernel<kProducerBlockM>),
          args),
      "tile_mhc_post_kernel launch");
}

template <bool kReducePartials, bool kSelectMode = false,
          bool kPeerOneShot = false, bool kPeerRankReady = false>
void launch_mhc_post_impl(
    const __nv_bfloat16* partial0,
    const __nv_bfloat16* partial1,
    const __nv_bfloat16* local_partial0,
    const __nv_bfloat16* local_partial1,
    const uint32_t* runtime_mode,
    uint32_t* block_generations,
    const uint32_t* local_ready,
    uint32_t* peer_ready,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int m) {
  constexpr int kMhcThreads = 256;
  const dim3 block(kMhcThreads);
  constexpr int kMaxGridX =
      (kVectorsPerRow + kMhcThreads - 1) / kMhcThreads;
  const int grid_x = kMaxGridX;
  const dim3 grid(grid_x, m);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(
      residual.data_ptr<at::BFloat16>());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(
      output.data_ptr<at::BFloat16>());
  mhc_post_kernel<kReducePartials, kSelectMode, kPeerOneShot, kPeerRankReady>
      <<<grid, block, 0, stream>>>(
      partial0, partial1, local_partial0, local_partial1, runtime_mode,
      block_generations, local_ready, peer_ready,
      residual_ptr, post.data_ptr<float>(),
      comb.data_ptr<float>(), output_ptr, m);
  check_cuda(cudaGetLastError(), "mhc_post_kernel launch");
}

void launch_mhc_post(
    const torch::Tensor& partials, const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& output) {
  const int m = static_cast<int>(residual.size(0));
  validate_mhc_inputs(partials, residual, post, comb, output, m);
  const auto* base = reinterpret_cast<const __nv_bfloat16*>(
      partials.data_ptr<at::BFloat16>());
  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  const auto* partial1 = base + kSlotStride;
  launch_mhc_post_impl<true>(
      base, partial1, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      residual, post, comb, output, m);
}

void launch_mhc_post_rank_ready(
    const torch::Tensor& partials,
    const torch::Tensor& generation,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int64_t rank) {
  const int m = static_cast<int>(residual.size(0));
  validate_mhc_inputs(partials, residual, post, comb, output, m);
  validate_tensor(generation, torch::kInt32, "generation");
  TORCH_CHECK(generation.numel() == 1,
              "generation must contain one INT32");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(symmetric_ptrs.size() == 2,
              "exactly two symmetric pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");

  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  const auto* local_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank])) + rank * kSlotStride;
  const auto* peer_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank ^ 1])) +
      (rank ^ 1) * kSlotStride;
  launch_mhc_post_impl<true, false, false, true>(
      local_partial, peer_partial, nullptr, nullptr, nullptr,
      reinterpret_cast<uint32_t*>(generation.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(signal_pad_ptrs[rank]),
      nullptr,
      residual, post, comb, output, m);
}

void launch_mhc_post_one_shot(
    const torch::Tensor& partials,
    const torch::Tensor& block_generations,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int64_t rank) {
  const int m = static_cast<int>(residual.size(0));
  validate_mhc_inputs(partials, residual, post, comb, output, m);
  validate_tensor(block_generations, torch::kInt32, "block_generations");
  TORCH_CHECK(block_generations.is_contiguous()
                  && block_generations.numel() >= kNumSMs,
              "block_generations must contain at least one int32 per SM");
  TORCH_CHECK(symmetric_ptrs.size() == 2,
              "exactly two symmetric pointers required");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");

  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  const auto* local_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank])) + rank * kSlotStride;
  const auto* peer_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank ^ 1]))
      + (rank ^ 1) * kSlotStride;
  launch_mhc_post_impl<true, false, true>(
      local_partial, peer_partial, nullptr, nullptr, nullptr,
      reinterpret_cast<uint32_t*>(block_generations.data_ptr<int>()),
      reinterpret_cast<const uint32_t*>(signal_pad_ptrs[rank]),
      reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank ^ 1]),
      residual, post, comb, output, m);
}

void launch_mhc_post_peer(
    const torch::Tensor& partials,
    const std::vector<int64_t>& symmetric_ptrs,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int64_t rank) {
  const int m = static_cast<int>(residual.size(0));
  validate_mhc_inputs(partials, residual, post, comb, output, m);
  TORCH_CHECK(symmetric_ptrs.size() == 2,
              "exactly two symmetric pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");

  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  const auto* local_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank])) + rank * kSlotStride;
  const auto* peer_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank ^ 1]))
      + (rank ^ 1) * kSlotStride;
  launch_mhc_post_impl<true>(
      local_partial, peer_partial, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      residual, post, comb, output, m);
}

void launch_peer_copy(
    const torch::Tensor& partials,
    const std::vector<int64_t>& symmetric_ptrs,
    int64_t rank,
    int64_t m) {
  validate_tensor(partials, torch::kBFloat16, "partials");
  TORCH_CHECK(partials.sizes() == torch::IntArrayRef({2, kMMax, kN})
                  && partials.is_contiguous(),
              "partials must be contiguous BF16 [2,128,7168]");
  TORCH_CHECK(symmetric_ptrs.size() == 2,
              "exactly two symmetric pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(m >= 1 && m <= kMMax, "M must be in [1,128]");

  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  auto* local_slot = reinterpret_cast<__nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank])) + rank * kSlotStride;
  auto* peer_slot = reinterpret_cast<__nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank ^ 1])) + rank * kSlotStride;
  const size_t bytes = static_cast<size_t>(m) * kN * sizeof(__nv_bfloat16);
  check_cuda(
      cudaMemcpyAsync(
          peer_slot, local_slot, bytes, cudaMemcpyDeviceToDevice,
          at::cuda::getCurrentCUDAStream()),
      "cudaMemcpyAsync partial");
}

void launch_mhc_post_plain(
    const torch::Tensor& reduced, const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& output) {
  const int m = static_cast<int>(residual.size(0));
  validate_tensor(reduced, torch::kBFloat16, "reduced");
  TORCH_CHECK(reduced.sizes() == torch::IntArrayRef({m, kN}) &&
                  reduced.is_contiguous(),
              "reduced must be contiguous BF16 [M,7168]");
  validate_mhc_common(residual, post, comb, output, m);
  const auto* reduced_ptr = reinterpret_cast<const __nv_bfloat16*>(
      reduced.data_ptr<at::BFloat16>());
  launch_mhc_post_impl<false>(
      reduced_ptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      residual, post, comb, output, m);
}

void launch_mhc_post_local_benchmark(
    const torch::Tensor& partials,
    const torch::Tensor& local_second_output,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int64_t rank) {
  const int m = static_cast<int>(residual.size(0));
  validate_mhc_inputs(partials, residual, post, comb, output, m);
  validate_tensor(
      local_second_output, torch::kBFloat16, "local_second_output");
  TORCH_CHECK(
      local_second_output.sizes() == torch::IntArrayRef({kMMax, kN}) &&
          local_second_output.is_contiguous(),
      "local_second_output must be contiguous BF16 [128,7168]");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  const auto* partials_base = reinterpret_cast<const __nv_bfloat16*>(
      partials.data_ptr<at::BFloat16>());
  const auto* local_partial = partials_base + rank * kSlotStride;
  const auto* second_partial = reinterpret_cast<const __nv_bfloat16*>(
      local_second_output.data_ptr<at::BFloat16>());
  launch_mhc_post_impl<true>(
      local_partial, second_partial, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      residual, post, comb, output, m);
}

void launch_mhc_post_select_benchmark(
    const torch::Tensor& partials,
    const torch::Tensor& local_second_output,
    const torch::Tensor& runtime_mode,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int64_t rank) {
  const int m = static_cast<int>(residual.size(0));
  validate_mhc_inputs(partials, residual, post, comb, output, m);
  validate_tensor(
      local_second_output, torch::kBFloat16, "local_second_output");
  validate_tensor(runtime_mode, torch::kInt32, "runtime_mode");
  TORCH_CHECK(
      local_second_output.sizes() == torch::IntArrayRef({kMMax, kN}) &&
          local_second_output.is_contiguous(),
      "local_second_output must be contiguous BF16 [128,7168]");
  TORCH_CHECK(runtime_mode.numel() == 1,
              "runtime_mode must contain one INT32");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  constexpr int64_t kSlotStride = static_cast<int64_t>(kMMax) * kN;
  const auto* base = reinterpret_cast<const __nv_bfloat16*>(
      partials.data_ptr<at::BFloat16>());
  const auto* local_partial = base + rank * kSlotStride;
  const auto* local_second = reinterpret_cast<const __nv_bfloat16*>(
      local_second_output.data_ptr<at::BFloat16>());
  launch_mhc_post_impl<true, true>(
      base, base + kSlotStride, local_partial, local_second,
      reinterpret_cast<const uint32_t*>(runtime_mode.data_ptr<int32_t>()),
      nullptr, nullptr, nullptr,
      residual, post, comb, output, m);
}

template <bool kSelectMode>
__global__ void peer_handshake_kernel(
    uint32_t* generation, const uint32_t* local_ready, uint32_t* peer_ready,
    const uint32_t* runtime_mode, uint32_t ready_offset, uint32_t rank) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if constexpr (kSelectMode) {
    if (deep_gemm::o_proj_b_tp2_symm::load_relaxed_gpu(runtime_mode) == 0)
      return;
  }

  using namespace deep_gemm::o_proj_b_tp2_symm;
  const uint32_t next = load_relaxed_gpu(generation) + 1;
  // Start-barrier semantics match vLLM's one-stage custom all-reduce: the
  // preceding producer kernel boundary already completed the partial.
  asm volatile("st.volatile.global.u32 [%0], %1;" ::
               "l"(peer_ready + ready_offset + rank), "r"(next)
               : "memory");

  const uint64_t wait_start = globaltimer();
  uint32_t spins = 0;
  const uint32_t peer_rank = rank ^ 1u;
  uint32_t ready;
  do {
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(ready)
                 : "l"(local_ready + ready_offset + peer_rank)
                 : "memory");
    check_spin_timeout(wait_start, ++spins, "peer", next, ready, rank);
  } while (ready != next);
  store_relaxed_gpu(generation, next);
}

template <bool kSelectMode>
void launch_peer_handshake_impl(
    const torch::Tensor& generation,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank,
    const torch::Tensor* runtime_mode,
    uint32_t ready_offset) {
  validate_tensor(generation, torch::kInt32, "generation");
  TORCH_CHECK(generation.numel() == 1, "generation must contain one INT32");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  const uint32_t* mode_ptr = nullptr;
  if constexpr (kSelectMode) {
    TORCH_CHECK(runtime_mode != nullptr, "select handshake requires mode");
    validate_tensor(*runtime_mode, torch::kInt32, "runtime_mode");
    TORCH_CHECK(runtime_mode->numel() == 1,
                "runtime_mode must contain one INT32");
    mode_ptr = reinterpret_cast<const uint32_t*>(
        runtime_mode->data_ptr<int32_t>());
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  peer_handshake_kernel<kSelectMode><<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(generation.data_ptr<int32_t>()),
      reinterpret_cast<const uint32_t*>(signal_pad_ptrs[rank]),
      reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank ^ 1]),
      mode_ptr, ready_offset, static_cast<uint32_t>(rank));
  check_cuda(cudaGetLastError(), "peer_handshake_kernel launch");
}

void launch_peer_handshake(
    const torch::Tensor& generation,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank) {
  launch_peer_handshake_impl<false>(
      generation, signal_pad_ptrs, rank, nullptr, 0);
}

void launch_peer_handshake_select(
    const torch::Tensor& generation,
    const torch::Tensor& runtime_mode,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank) {
  launch_peer_handshake_impl<true>(
      generation, signal_pad_ptrs, rank, &runtime_mode,
      kSelectReadyOffset);
}

__global__ void benchmark_barrier_kernel(
    uint32_t* generation, uint32_t* local_ready, uint32_t* peer_ready,
    uint32_t rank) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  using namespace deep_gemm::o_proj_b_tp2_symm;
  const uint32_t next = load_relaxed_gpu(generation) + 1;
  fence_acq_rel_sys();
  store_relaxed_sys(peer_ready + kBenchmarkReadyOffset + rank, next);
  store_relaxed_sys(local_ready + kBenchmarkReadyOffset + rank, next);

  const uint64_t wait_start = globaltimer();
  uint32_t spins = 0;
  auto ready0 = load_relaxed_sys(local_ready + kBenchmarkReadyOffset);
  auto ready1 = load_relaxed_sys(local_ready + kBenchmarkReadyOffset + 1);
  while (ready0 < next || ready1 < next) {
    spin_pause();
    ready0 = load_relaxed_sys(local_ready + kBenchmarkReadyOffset);
    ready1 = load_relaxed_sys(local_ready + kBenchmarkReadyOffset + 1);
    check_spin_timeout(
        wait_start, ++spins, "benchmark", next,
        ready0 < next ? ready0 : ready1, rank);
  }
  fence_acquire_sys();
  store_relaxed_gpu(generation, next);
  // Keep this excluded rendezvous resident long enough for both Python hosts
  // to enqueue start-event -> graph -> end-event behind it. Otherwise the
  // faster host can release the barrier before the peer has submitted its
  // graph, and that host-launch skew appears as communication wait time.
  const uint64_t release = globaltimer();
  while (globaltimer() - release < kBenchmarkEnqueueGraceNs) spin_pause();
}

void launch_benchmark_barrier(
    const torch::Tensor& generation,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank) {
  validate_tensor(generation, torch::kInt32, "generation");
  TORCH_CHECK(generation.numel() == 1, "generation must contain one INT32");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  auto* local_ready = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank]);
  auto* peer_ready = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank ^ 1]);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  benchmark_barrier_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(generation.data_ptr<int32_t>()),
      local_ready, peer_ready, static_cast<uint32_t>(rank));
  check_cuda(cudaGetLastError(), "benchmark_barrier_kernel launch");
}

template <uint32_t kLoadStages, uint32_t kStoreStages, uint32_t kBlockM,
          bool kRuntimeSelect, bool kTilePublish,
          bool kStoreSecondDestination, bool kVectorPeerStore>
void launch_impl(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& symmetric_partials,
    const torch::Tensor& grid_done,
    const torch::Tensor& generation,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank,
    bool static_use_peer,
    void* local_second_destination,
    const uint32_t* runtime_mode,
    uint32_t ready_offset,
    uint32_t* tile_generations,
    bool common_local_slot) {
  validate_tensor(a, torch::kFloat8_e4m3fn, "a");
  validate_tensor(sfa, torch::kInt32, "sfa");
  validate_tensor(b, torch::kFloat8_e4m3fn, "b");
  validate_tensor(sfb, torch::kInt32, "sfb");
  validate_tensor(symmetric_partials, torch::kBFloat16, "symmetric_partials");
  validate_tensor(grid_done, torch::kInt64, "grid_done");
  validate_tensor(generation, torch::kInt32, "generation");

  TORCH_CHECK(a.dim() == 2 && a.size(1) == kK, "a must be [M,8192]");
  const int m = static_cast<int>(a.size(0));
  TORCH_CHECK(m >= 1 && m <= kMMax, "M must be in [1,128]");
  const int aligned_m = (m + 3) / 4 * 4;
  TORCH_CHECK(sfa.dim() == 2 && sfa.size(0) == m && sfa.size(1) == 16 &&
                  sfa.stride(0) == 1 && sfa.stride(1) == aligned_m,
              "sfa must be DeepGEMM MN-major INT32 [M,16]");
  TORCH_CHECK(b.sizes() == torch::IntArrayRef({kN, kK}),
              "b must be [7168,8192]");
  TORCH_CHECK(sfb.dim() == 2 && sfb.size(0) == kN && sfb.size(1) == 16 &&
                  sfb.stride(0) == 1 && sfb.stride(1) == kN,
              "sfb must be DeepGEMM MN-major INT32 [7168,16]");
  TORCH_CHECK(symmetric_partials.dim() == 3 &&
                  symmetric_partials.size(0) == 2 &&
                  symmetric_partials.size(1) == kMMax &&
                  symmetric_partials.size(2) == kN &&
                  symmetric_partials.is_contiguous(),
              "symmetric_partials must be contiguous BF16 [2,128,7168]");
  TORCH_CHECK(
      grid_done.numel() == kGridDoneElements && generation.numel() == 1,
      "grid_done must contain ", kGridDoneElements,
      " INT64 values and generation one INT32");
  TORCH_CHECK(symmetric_ptrs.size() == 2, "exactly two symmetric pointers required");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");

  const CUtensorMap tensor_map_a = make_a_map<kBlockM>(a, m);
  const CUtensorMap tensor_map_b = make_b_map(b);
  const CUtensorMap tensor_map_sfa = make_scale_map(sfa, m, kBlockM);
  const CUtensorMap tensor_map_sfb = make_scale_map(sfb, kN, kBlockN);
  constexpr int64_t kSlotElements =
      static_cast<int64_t>(kMMax) * kN;
  auto* local_slot = reinterpret_cast<KernelDType*>(symmetric_ptrs[rank]) +
                     (common_local_slot ? 0 : rank) * kSlotElements;
  if constexpr (kVectorPeerStore) {
    TORCH_CHECK(local_second_destination != nullptr,
                "vector-store epilogue requires a local TMA destination");
    local_slot = reinterpret_cast<KernelDType*>(local_second_destination);
  }
  auto* local_second_slot = local_second_destination == nullptr
      ? local_slot
      : reinterpret_cast<KernelDType*>(local_second_destination);
  auto* peer_slot = reinterpret_cast<KernelDType*>(symmetric_ptrs[rank ^ 1]) +
                    rank * kSlotElements;
  auto* vector_store_slot = kVectorPeerStore ? peer_slot : nullptr;
  const CUtensorMap tensor_map_local_cd = make_output_map(local_slot, m);
  const CUtensorMap tensor_map_local_second_cd =
      make_output_map(local_second_slot, m);
  const CUtensorMap tensor_map_peer_cd = make_output_map(peer_slot, m);

  auto kernel = kernel_ptr<
      kLoadStages, kStoreStages, kBlockM, kRuntimeSelect, kTilePublish,
      kStoreSecondDestination, kVectorPeerStore>();
  check_cuda(
      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kDynamicSmemBytes),
      "cudaFuncSetAttribute");

  auto* grid_done_ptr = reinterpret_cast<uint64_t*>(grid_done.data_ptr<int64_t>());
  auto* generation_ptr = reinterpret_cast<uint32_t*>(generation.data_ptr<int32_t>());
  auto* local_ready = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank]);
  auto* peer_ready = reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank ^ 1]);
  uint32_t shape_m = static_cast<uint32_t>(m);
  uint32_t shape_n = kN;
  uint32_t shape_k = kK;
  uint32_t rank_u32 = static_cast<uint32_t>(rank);
  int* grouped_layout = nullptr;

  void* args[] = {
      &grouped_layout, &shape_m, &shape_n, &shape_k,
      const_cast<CUtensorMap*>(&tensor_map_a),
      const_cast<CUtensorMap*>(&tensor_map_b),
      const_cast<CUtensorMap*>(&tensor_map_sfa),
      const_cast<CUtensorMap*>(&tensor_map_sfb),
      const_cast<CUtensorMap*>(&tensor_map_local_cd),
      const_cast<CUtensorMap*>(&tensor_map_local_second_cd),
      const_cast<CUtensorMap*>(&tensor_map_peer_cd),
      &grid_done_ptr, &generation_ptr, &local_ready, &peer_ready,
      &rank_u32, &runtime_mode, &static_use_peer, &ready_offset,
      &tile_generations, &vector_store_slot};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSMs, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = kDynamicSmemBytes;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kClusterSize;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;

  check_cuda(
      cudaLaunchKernelExC(
          &config, reinterpret_cast<void*>(kernel), args),
      "cudaLaunchKernelExC");
}

template <bool kRuntimeSelect, bool kTilePublish = false,
          bool kStoreSecondDestination = true,
          bool kVectorPeerStore = false>
void launch_mode(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& symmetric_partials,
    const torch::Tensor& grid_done,
    const torch::Tensor& generation,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank,
    bool static_use_peer,
    void* local_second_destination,
    const uint32_t* runtime_mode,
    uint32_t ready_offset,
    uint32_t* tile_generations = nullptr,
    bool common_local_slot = false) {
  if (a.size(0) <= 32) {
    launch_impl<12, 2, 16, kRuntimeSelect, kTilePublish,
                kStoreSecondDestination, kVectorPeerStore>(
        a, sfa, b, sfb, symmetric_partials, grid_done, generation,
        symmetric_ptrs, signal_pad_ptrs, rank,
        static_use_peer, local_second_destination, runtime_mode, ready_offset,
        tile_generations, common_local_slot);
  } else {
    launch_impl<10, 2, 64, kRuntimeSelect, kTilePublish,
                kStoreSecondDestination, kVectorPeerStore>(
        a, sfa, b, sfb, symmetric_partials, grid_done, generation,
        symmetric_ptrs, signal_pad_ptrs, rank,
        static_use_peer, local_second_destination, runtime_mode, ready_offset,
        tile_generations, common_local_slot);
  }
}

void launch(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& symmetric_partials,
    const torch::Tensor& grid_done,
    const torch::Tensor& generation,
    const torch::Tensor& tile_generations,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    const torch::Tensor& local_projected,
    const torch::Tensor& residual,
    const torch::Tensor& post,
    const torch::Tensor& comb,
    const torch::Tensor& output,
    int64_t rank) {
  const int m = static_cast<int>(a.size(0));
  validate_mhc_inputs(
      symmetric_partials, residual, post, comb, output, m);
  validate_tensor(tile_generations, torch::kInt32, "tile_generations");
  TORCH_CHECK(
      tile_generations.is_contiguous() &&
          tile_generations.numel() >= kMMax * 4,
      "tile_generations must contain one generation per mHC CTA");
  TORCH_CHECK(symmetric_ptrs.size() == 2,
              "exactly two symmetric pointers required");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  validate_tensor(local_projected, torch::kBFloat16, "local_projected");
  TORCH_CHECK(local_projected.sizes() == torch::IntArrayRef({m, kN}) &&
                  local_projected.is_contiguous(),
              "local_projected must be contiguous BF16 [M,7168]");
  auto* tile_generation_ptr = reinterpret_cast<uint32_t*>(
      tile_generations.data_ptr<int32_t>());
  launch_mode<false, false, false, true>(
      a, sfa, b, sfb, symmetric_partials, grid_done, generation,
      symmetric_ptrs, signal_pad_ptrs, rank, false,
      local_projected.data_ptr<at::BFloat16>(), nullptr, 0,
      tile_generation_ptr, false);
  const auto* local_partial = reinterpret_cast<const __nv_bfloat16*>(
      local_projected.data_ptr<at::BFloat16>());
  constexpr int64_t kSlotElements = static_cast<int64_t>(kMMax) * kN;
  const auto* peer_partial = reinterpret_cast<const __nv_bfloat16*>(
      static_cast<uintptr_t>(symmetric_ptrs[rank])) +
      (rank ^ 1) * kSlotElements;
  launch_mhc_post_impl<true, false, true>(
      local_partial, peer_partial,
      nullptr, nullptr, nullptr,
      tile_generation_ptr,
      reinterpret_cast<const uint32_t*>(signal_pad_ptrs[rank]),
      reinterpret_cast<uint32_t*>(signal_pad_ptrs[rank ^ 1]),
      residual, post, comb, output, m);
}

void launch_local_benchmark(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& symmetric_partials,
    const torch::Tensor& local_second_output,
    const torch::Tensor& grid_done,
    const torch::Tensor& generation,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank) {
  validate_tensor(
      local_second_output, torch::kBFloat16, "local_second_output");
  TORCH_CHECK(
      local_second_output.sizes() == torch::IntArrayRef({kMMax, kN}) &&
          local_second_output.is_contiguous(),
      "local_second_output must be contiguous BF16 [128,7168]");
  constexpr uint32_t kLocalBenchmarkReadyOffset = 2;
  launch_mode<false>(
      a, sfa, b, sfb, symmetric_partials, grid_done, generation,
      symmetric_ptrs, signal_pad_ptrs, rank, false,
      local_second_output.data_ptr<at::BFloat16>(),
      nullptr,
      kLocalBenchmarkReadyOffset);
}

void launch_select_benchmark(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& symmetric_partials,
    const torch::Tensor& local_second_output,
    const torch::Tensor& grid_done,
    const torch::Tensor& generation,
    const torch::Tensor& runtime_mode,
    const std::vector<int64_t>& symmetric_ptrs,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank) {
  validate_tensor(
      local_second_output, torch::kBFloat16, "local_second_output");
  validate_tensor(runtime_mode, torch::kInt32, "runtime_mode");
  TORCH_CHECK(
      local_second_output.sizes() == torch::IntArrayRef({kMMax, kN}) &&
          local_second_output.is_contiguous(),
      "local_second_output must be contiguous BF16 [128,7168]");
  TORCH_CHECK(runtime_mode.numel() == 1,
              "runtime_mode must contain one INT32");
  constexpr uint32_t kSelectBenchmarkReadyOffset = 4;
  launch_mode<true>(
      a, sfa, b, sfb, symmetric_partials, grid_done, generation,
      symmetric_ptrs, signal_pad_ptrs, rank, true,
      local_second_output.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint32_t*>(runtime_mode.data_ptr<int32_t>()),
      kSelectBenchmarkReadyOffset);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "o_proj_b", &launch,
      "TP2 local O-proj B with vector peer publication and fused mHC post");
  m.def(
      "o_proj_b_local", &launch_local_benchmark,
      "Benchmark-only O-proj B with two local destinations");
  m.def(
      "o_proj_b_select", &launch_select_benchmark,
      "Benchmark-only O-proj B with device-selected second destination");
  m.def(
      "mhc_post", &launch_mhc_post,
      "mHC post with fused BF16 pairwise reduction");
  m.def(
      "mhc_post_rank_ready", &launch_mhc_post_rank_ready,
      "TP2 mHC post with producer-published rank readiness");
  m.def(
      "mhc_post_one_shot", &launch_mhc_post_one_shot,
      "TP2 one-shot peer reduction fused into mHC post");
  m.def(
      "mhc_post_peer", &launch_mhc_post_peer,
      "TP2 peer reduction in the full-grid mHC post");
  m.def(
      "copy_partial_peer", &launch_peer_copy,
      "TP2 copy-engine partial push");
  m.def(
      "mhc_post_plain", &launch_mhc_post_plain,
      "Benchmark-only mHC post over a pre-reduced input");
  m.def(
      "mhc_post_local", &launch_mhc_post_local_benchmark,
      "Benchmark-only fused mHC post over two local partials");
  m.def(
      "mhc_post_select", &launch_mhc_post_select_benchmark,
      "Benchmark-only off/signal/peer fused mHC post");
  m.def(
      "peer_handshake", &launch_peer_handshake,
      "TP2 producer-completion peer barrier");
  m.def(
      "peer_handshake_select", &launch_peer_handshake_select,
      "Benchmark-only peer barrier selector");
  m.def(
      "benchmark_barrier", &launch_benchmark_barrier,
      "Excluded TP2 device rendezvous for paired benchmarks");
}
