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
constexpr int kValuesPerVector = 8;
constexpr int kVectorsPerRow = kN / kValuesPerVector;
constexpr int kBenchmarkReadyOffset = 16;
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

template <uint32_t kLoadStages, uint32_t kStoreStages, uint32_t kBlockM>
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

__global__ void mhc_post_kernel(
    const __nv_bfloat16* partial0, const __nv_bfloat16* partial1,
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
  __syncthreads();

  for (int vector = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       vector < kVectorsPerRow;
       vector += static_cast<int>(gridDim.x) * blockDim.x) {
    const int dim = vector * kValuesPerVector;
    const auto* x0_ptr = partial0 + row * kN + dim;
    const auto* x1_ptr = partial1 + row * kN + dim;
    uint4 x_bits = add_bf16x8(
        *reinterpret_cast<const uint4*>(x0_ptr),
        *reinterpret_cast<const uint4*>(x1_ptr));

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

void validate_mhc_inputs(
    const torch::Tensor& partials, const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& output, int m) {
  validate_tensor(partials, torch::kBFloat16, "partials");
  validate_tensor(residual, torch::kBFloat16, "residual");
  validate_tensor(post, torch::kFloat32, "post");
  validate_tensor(comb, torch::kFloat32, "comb");
  validate_tensor(output, torch::kBFloat16, "output");
  TORCH_CHECK(m >= 1 && m <= kMMax, "M must be in [1,128]");
  TORCH_CHECK(partials.sizes() ==
                  torch::IntArrayRef({2, kMMax, kN}) &&
                  partials.is_contiguous(),
              "partials must be contiguous BF16 [2,128,7168]");
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

void launch_mhc_post_impl(
    const __nv_bfloat16* partial0,
    const __nv_bfloat16* partial1,
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
  mhc_post_kernel<<<grid, block, 0, stream>>>(
      partial0, partial1, block_generations, local_ready, peer_ready,
      residual_ptr, post.data_ptr<float>(),
      comb.data_ptr<float>(), output_ptr, m);
  check_cuda(cudaGetLastError(), "mhc_post_kernel launch");
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
  const auto signals = wuda::tp2::make_symmetric_view(
      signal_pad_ptrs, static_cast<uint32_t>(rank));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  benchmark_barrier_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<uint32_t*>(generation.data_ptr<int32_t>()),
      signals.local<uint32_t>(), signals.peer_base<uint32_t>(), signals.rank);
  check_cuda(cudaGetLastError(), "benchmark_barrier_kernel launch");
}

template <uint32_t kLoadStages, uint32_t kStoreStages, uint32_t kBlockM>
void launch_impl(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const std::vector<int64_t>& symmetric_ptrs,
    int64_t rank,
    void* local_destination) {
  validate_tensor(a, torch::kFloat8_e4m3fn, "a");
  validate_tensor(sfa, torch::kInt32, "sfa");
  validate_tensor(b, torch::kFloat8_e4m3fn, "b");
  validate_tensor(sfb, torch::kInt32, "sfb");

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

  const CUtensorMap tensor_map_a = make_a_map<kBlockM>(a, m);
  const CUtensorMap tensor_map_b = make_b_map(b);
  const CUtensorMap tensor_map_sfa = make_scale_map(sfa, m, kBlockM);
  const CUtensorMap tensor_map_sfb = make_scale_map(sfb, kN, kBlockN);
  const auto partials_view = wuda::tp2::make_symmetric_view(
      symmetric_ptrs, static_cast<uint32_t>(rank));
  constexpr int64_t kSlotElements =
      static_cast<int64_t>(kMMax) * kN;
  // The local partial stays outside the symmetric buffer: only the peer's copy
  // travels, and it lands in this rank's slot of the peer's allocation.
  auto* local_slot = reinterpret_cast<KernelDType*>(local_destination);
  auto* peer_slot = partials_view.peer_base<KernelDType>() +
                    rank * kSlotElements;
  const CUtensorMap tensor_map_local_cd = make_output_map(local_slot, m);

  auto kernel = kernel_ptr<kLoadStages, kStoreStages, kBlockM>();
  check_cuda(
      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kDynamicSmemBytes),
      "cudaFuncSetAttribute");

  uint32_t shape_m = static_cast<uint32_t>(m);
  uint32_t shape_n = kN;
  uint32_t shape_k = kK;
  int* grouped_layout = nullptr;

  void* args[] = {
      &grouped_layout, &shape_m, &shape_n, &shape_k,
      const_cast<CUtensorMap*>(&tensor_map_a),
      const_cast<CUtensorMap*>(&tensor_map_b),
      const_cast<CUtensorMap*>(&tensor_map_sfa),
      const_cast<CUtensorMap*>(&tensor_map_sfb),
      const_cast<CUtensorMap*>(&tensor_map_local_cd),
      &peer_slot};

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

void launch(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& symmetric_partials,
    const torch::Tensor& block_generations,
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
  validate_tensor(block_generations, torch::kInt32, "block_generations");
  TORCH_CHECK(
      block_generations.is_contiguous() &&
          block_generations.numel() >= kMMax * 4,
      "block_generations must contain one generation per mHC CTA");
  TORCH_CHECK(symmetric_ptrs.size() == 2,
              "exactly two symmetric pointers required");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers required");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  validate_tensor(local_projected, torch::kBFloat16, "local_projected");
  TORCH_CHECK(local_projected.sizes() == torch::IntArrayRef({m, kN}) &&
                  local_projected.is_contiguous(),
              "local_projected must be contiguous BF16 [M,7168]");
  auto* block_generation_ptr = reinterpret_cast<uint32_t*>(
      block_generations.data_ptr<int32_t>());
  const auto partials_view = wuda::tp2::make_symmetric_view(
      symmetric_ptrs, static_cast<uint32_t>(rank));
  const auto signals = wuda::tp2::make_symmetric_view(
      signal_pad_ptrs, static_cast<uint32_t>(rank));
  if (m <= 32) {
    launch_impl<12, 2, 16>(
        a, sfa, b, sfb, symmetric_ptrs, rank,
        local_projected.data_ptr<at::BFloat16>());
  } else {
    launch_impl<10, 2, 64>(
        a, sfa, b, sfb, symmetric_ptrs, rank,
        local_projected.data_ptr<at::BFloat16>());
  }
  const auto* local_partial = reinterpret_cast<const __nv_bfloat16*>(
      local_projected.data_ptr<at::BFloat16>());
  constexpr int64_t kSlotElements = static_cast<int64_t>(kMMax) * kN;
  const auto* peer_partial =
      partials_view.local<const __nv_bfloat16>() +
      partials_view.peer_rank() * kSlotElements;
  launch_mhc_post_impl(
      local_partial, peer_partial,
      block_generation_ptr,
      signals.local<const uint32_t>(), signals.peer_base<uint32_t>(),
      residual, post, comb, output, m);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "o_proj_b", &launch,
      "TP2 local O-proj B with vector peer publication and fused mHC post");
  m.def(
      "benchmark_barrier", &launch_benchmark_barrier,
      "Excluded TP2 device rendezvous for paired benchmarks");
}
