#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

#include "o_proj_b_tp2_symm.cuh"
#include "o_proj_a_tp2_overlap.cuh"

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
constexpr int kMhcGridReadyOffset = 64 + kMMax * 4;
// Measured crossover of the two start-barrier flag schemes; see mhc_post_kernel.
constexpr int kMhcSingleFlagMinCtas = 256;
constexpr uint64_t kBenchmarkEnqueueGraceNs = 1'000'000;

// WoA is the grouped batched GEMM between the MLA output and the per-group
// [1024, 4096] weights.  These values match the DeepGEMM heuristic for
// GemmType::Batched on SM100 (G=8, N=1024, K=4096).  The heuristic selects
// swap-AB with a 2-CTA cluster and N tiles of 128; the M tile is 16/32/64 as
// M crosses 32/64/128.
constexpr int kWoAGroups = 8;
constexpr int kWoAMax = 128;
constexpr int kWoAN = 1024;
constexpr int kWoAK = 4096;
constexpr int kWoABlockN = 128;
constexpr int kWoABlockK = 128;
constexpr int kWoAClusterSize = 2;
constexpr int kWoANumSMs = 148;
constexpr int kWoAThreads = 256;
constexpr int kWoADynamicSmemBytes = 230188;

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

CUtensorMap make_3d_map(
    void* ptr, CUtensorMapDataType dtype, int element_bytes,
    uint64_t inner_dim, uint64_t outer_dim, uint64_t batch_dim,
    uint64_t outer_stride_elements, uint64_t batch_stride_elements,
    uint32_t box_inner, uint32_t box_outer, uint32_t box_batch,
    CUtensorMapSwizzle swizzle) {
  CUtensorMap result{};
  const cuuint64_t global_dims[3] = {inner_dim, outer_dim, batch_dim};
  const cuuint64_t global_strides[2] = {
      outer_stride_elements * static_cast<uint64_t>(element_bytes),
      batch_stride_elements * static_cast<uint64_t>(element_bytes)};
  const cuuint32_t box_dims[3] = {box_inner, box_outer, box_batch};
  const cuuint32_t element_strides[3] = {1, 1, 1};
  check_driver(
      cuTensorMapEncodeTiled(
          &result, dtype, 3, ptr, global_dims, global_strides, box_dims,
          element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(3D)");
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

template <uint32_t kStages, uint32_t kBlockM>
auto wo_a_kernel_ptr() {
  return &deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      128, 128, 128,
      0, kWoAN, kWoAK,
      kBlockM, kWoABlockN, kWoABlockK,
      kWoAGroups,
      128, 128, 128,
      kStages,
      128, 128,
      kWoAClusterSize, true,
      kWoANumSMs,
      true, true,
      deep_gemm::GemmType::Batched, false,
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, KernelDType,
      deep_gemm::epilogue::transform::EpilogueIdentity>;
}

template <uint32_t kStages, uint32_t kBlockM>
auto wo_a_overlap_kernel_ptr() {
  return &deep_gemm::sm100_fp8_fp4_gemm_1d1d_tp2_overlap_impl<
      cute::UMMA::Major::K, cute::UMMA::Major::K,
      128, 128, 128,
      0, kWoAN, kWoAK,
      kBlockM, kWoABlockN, kWoABlockK,
      kWoAGroups,
      128, 128, 128,
      kStages,
      128, 128,
      kWoAClusterSize, true,
      kWoANumSMs,
      true, true,
      deep_gemm::GemmType::Batched, false,
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
  // Start barrier. Every CTA of a rank waits on the same fact -- that the
  // peer's GEMM retired, and with it the peer's pushes into this rank's
  // partial slot -- so either one flag per grid or one per CTA carries it.
  // Measured: the single flag wins by 0.1..0.5 us from 256 CTAs up, and loses
  // by 0.15..0.30 us below that, where few enough CTAs poll it that its remote
  // stores no longer pay for themselves.
  const uint32_t cta = row * gridDim.x + blockIdx.x;
  const bool cta_flags =
      static_cast<uint32_t>(m) * gridDim.x < kMhcSingleFlagMinCtas;
  if (threadIdx.x == 0) {
    const uint32_t next = block_generations[cta] + 1;
    // The preceding GEMM kernel boundary already completed the local partial
    // and made the peer's stores visible, so no release fence is required.
    if (cta_flags) {
      asm volatile("st.volatile.global.u32 [%0], %1;" ::
                   "l"(peer_ready + kMhcReadyOffset + cta), "r"(next)
                   : "memory");
    } else if (cta == 0) {
      asm volatile("st.volatile.global.u32 [%0], %1;" ::
                   "l"(peer_ready + kMhcGridReadyOffset), "r"(next)
                   : "memory");
    }
    const uint32_t* flag = cta_flags
        ? local_ready + kMhcReadyOffset + cta
        : local_ready + kMhcGridReadyOffset;
    uint32_t ready;
    do {
      asm volatile("ld.volatile.global.u32 %0, [%1];"
                   : "=r"(ready) : "l"(flag) : "memory");
    } while (cta_flags ? ready != next : ready < next);
    block_generations[cta] = next;
  }

  __syncthreads();

  // One vector per thread: the launch sizes the grid to cover a whole row, so
  // there is no stride loop to carry.
  const int vector = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int dim = vector * kValuesPerVector;
  if (vector < kVectorsPerRow) {
    alignas(16) __nv_bfloat16 x[8];
    alignas(16) __nv_bfloat16 r[4][8];
    *reinterpret_cast<uint4*>(x) =
        *reinterpret_cast<const uint4*>(partial0 + row * kN + dim);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<uint4*>(r[i]) = *reinterpret_cast<const uint4*>(
          residual + (row * 4 + i) * kN + dim);
    }
    *reinterpret_cast<uint4*>(x) = add_bf16x8(
        *reinterpret_cast<const uint4*>(x),
        *reinterpret_cast<const uint4*>(partial1 + row * kN + dim));

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
  static_assert(kMaxGridX * kMhcThreads >= kVectorsPerRow,
                "the mHC post grid must cover a whole row in one pass");
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

  const uint64_t wait_start = wuda::tp2::globaltimer();
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
  const uint64_t release = wuda::tp2::globaltimer();
  while (wuda::tp2::globaltimer() - release <
         kBenchmarkEnqueueGraceNs) {
    spin_pause();
  }
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

template <uint32_t kStages, uint32_t kBlockM>
void launch_wo_a_impl(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d) {
  const int m = static_cast<int>(a.size(1));
  const int aligned_m = (m + 3) / 4 * 4;

  // The batched DeepGEMM kernel sees [G,M,K] @ [G,N,K]^T.  The Python side
  // passes permuted views, so all three TMA descriptors can use the native
  // grouped strides without a staging copy.
  const CUtensorMap tensor_map_a = make_3d_map(
      const_cast<void*>(a.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kWoAK, m, kWoAGroups,
      static_cast<uint64_t>(a.stride(1)),
      static_cast<uint64_t>(a.stride(0)),
      kWoABlockK, kBlockM / kWoAClusterSize, 1,
      CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tensor_map_b = make_3d_map(
      const_cast<void*>(b.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kWoAK, kWoAN, kWoAGroups,
      static_cast<uint64_t>(b.stride(1)),
      static_cast<uint64_t>(b.stride(0)),
      kWoABlockK, kWoABlockN, 1, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tensor_map_d = make_3d_map(
      const_cast<void*>(d.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      kWoAN, m, kWoAGroups,
      static_cast<uint64_t>(d.stride(1)),
      static_cast<uint64_t>(d.stride(0)),
      // 128B swizzle is measured in bytes; BF16 therefore uses 64
      // elements in the TMA box (the same convention as make_output_map).
      kWoABlockN / 2, 16, 1, CU_TENSOR_MAP_SWIZZLE_128B);

  // Packed UE8M0 scales are stored as [G,M,ceil(K/512)] and [G,N,8] with
  // MN-major strides.  DeepGEMM's SF descriptor flattens the group and K
  // dimensions into the second TMA dimension; the outer stride is the
  // TMA-aligned MN extent, exactly as in its fp8_bmm implementation.
  constexpr int kScaleK = kWoAK / (128 * 4);
  const CUtensorMap tensor_map_sfa = make_2d_map(
      const_cast<void*>(sfa.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
      aligned_m, static_cast<uint64_t>(kScaleK * kWoAGroups), aligned_m,
      kBlockM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tensor_map_sfb = make_2d_map(
      const_cast<void*>(sfb.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
      kWoAN, static_cast<uint64_t>(kScaleK * kWoAGroups), kWoAN,
      kWoABlockN, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

  auto kernel = wo_a_kernel_ptr<kStages, kBlockM>();
  check_cuda(
      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kWoADynamicSmemBytes),
      "cudaFuncSetAttribute(WoA)");

  uint32_t shape_m = static_cast<uint32_t>(m);
  uint32_t shape_n = kWoAN;
  uint32_t shape_k = kWoAK;
  int* grouped_layout = nullptr;
  void* args[] = {
      &grouped_layout, &shape_m, &shape_n, &shape_k,
      const_cast<CUtensorMap*>(&tensor_map_a),
      const_cast<CUtensorMap*>(&tensor_map_b),
      const_cast<CUtensorMap*>(&tensor_map_sfa),
      const_cast<CUtensorMap*>(&tensor_map_sfb),
      const_cast<CUtensorMap*>(&tensor_map_d)};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kWoANumSMs, 1, 1);
  config.blockDim = dim3(kWoAThreads, 1, 1);
  config.dynamicSmemBytes = kWoADynamicSmemBytes;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kWoAClusterSize;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  check_cuda(
      cudaLaunchKernelExC(
          &config, reinterpret_cast<void*>(kernel), args),
      "cudaLaunchKernelExC(WoA)");
}

template <uint32_t kStages, uint32_t kBlockM>
void launch_wo_a_tp2_overlap_impl(
    const torch::Tensor& local_a, const torch::Tensor& remote_a,
    const torch::Tensor& local_sfa, const torch::Tensor& remote_sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const std::vector<int64_t>& signal_pad_ptrs,
    const torch::Tensor& grid_done, int64_t rank) {
  const int m = static_cast<int>(d.size(1));
  const int half_m = m / 2;
  const int aligned_half_m = (half_m + 3) / 4 * 4;
  const int scale_m_stride = static_cast<int>(local_sfa.stride(2));
  const int split = m / 2;
  TORCH_CHECK(m % 2 == 0 && half_m >= 1,
              "fused TP2 WoA requires an even global M");

  // Each descriptor addresses one compact producer slot.  The output remains
  // global-M, so the epilogue writes directly into the normal z workspace.
  const CUtensorMap tensor_map_a_local = make_3d_map(
      const_cast<void*>(local_a.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kWoAK, half_m, kWoAGroups,
      static_cast<uint64_t>(local_a.stride(1)),
      static_cast<uint64_t>(local_a.stride(0)),
      kWoABlockK, kBlockM / kWoAClusterSize, 1,
      CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tensor_map_a_remote = make_3d_map(
      const_cast<void*>(remote_a.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kWoAK, half_m, kWoAGroups,
      static_cast<uint64_t>(remote_a.stride(1)),
      static_cast<uint64_t>(remote_a.stride(0)),
      kWoABlockK, kBlockM / kWoAClusterSize, 1,
      CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tensor_map_b = make_3d_map(
      const_cast<void*>(b.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
      kWoAK, kWoAN, kWoAGroups,
      static_cast<uint64_t>(b.stride(1)),
      static_cast<uint64_t>(b.stride(0)),
      kWoABlockK, kWoABlockN, 1, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tensor_map_sfa_local = make_2d_map(
      const_cast<void*>(local_sfa.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_INT32,
      4, aligned_half_m, static_cast<uint64_t>(8 * kWoAGroups),
      scale_m_stride, kBlockM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tensor_map_sfa_remote = make_2d_map(
      const_cast<void*>(remote_sfa.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_INT32,
      4, aligned_half_m, static_cast<uint64_t>(8 * kWoAGroups),
      scale_m_stride, kBlockM, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tensor_map_sfb = make_2d_map(
      const_cast<void*>(sfb.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
      kWoAN, static_cast<uint64_t>(8 * kWoAGroups), kWoAN,
      kWoABlockN, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  const CUtensorMap tensor_map_d = make_3d_map(
      const_cast<void*>(d.data_ptr()), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      kWoAN, m, kWoAGroups,
      static_cast<uint64_t>(d.stride(1)),
      static_cast<uint64_t>(d.stride(0)),
      kWoABlockN / 2, 16, 1, CU_TENSOR_MAP_SWIZZLE_128B);

  auto kernel = wo_a_overlap_kernel_ptr<kStages, kBlockM>();
  check_cuda(
      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          kWoADynamicSmemBytes),
      "cudaFuncSetAttribute(fused WoA)");

  const auto signals = wuda::tp2::make_symmetric_view(signal_pad_ptrs,
                                                       static_cast<uint32_t>(rank));
  const uint32_t* peer_ready = signals.local<const uint32_t>();
  const uint64_t* grid_done_ptr =
      reinterpret_cast<const uint64_t*>(grid_done.data_ptr<int64_t>());
  uint32_t rank_u = static_cast<uint32_t>(rank);
  uint32_t local_start = rank == 0 ? 0u : static_cast<uint32_t>(split);
  uint32_t remote_start = rank == 0 ? static_cast<uint32_t>(split) : 0u;
  uint32_t shape_m = static_cast<uint32_t>(m);
  uint32_t shape_n = kWoAN;
  uint32_t shape_k = kWoAK;
  int* grouped_layout = nullptr;
  void* args[] = {
      &grouped_layout, &shape_m, &shape_n, &shape_k,
      const_cast<CUtensorMap*>(&tensor_map_a_local),
      const_cast<CUtensorMap*>(&tensor_map_a_remote),
      const_cast<CUtensorMap*>(&tensor_map_b),
      const_cast<CUtensorMap*>(&tensor_map_sfa_local),
      const_cast<CUtensorMap*>(&tensor_map_sfa_remote),
      const_cast<CUtensorMap*>(&tensor_map_sfb),
      const_cast<CUtensorMap*>(&tensor_map_d),
      &peer_ready, &grid_done_ptr, &rank_u, &local_start, &remote_start};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kWoANumSMs, 1, 1);
  config.blockDim = dim3(kWoAThreads, 1, 1);
  config.dynamicSmemBytes = kWoADynamicSmemBytes;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = kWoAClusterSize;
  attributes[0].val.clusterDim.y = 1;
  attributes[0].val.clusterDim.z = 1;
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attributes;
  config.numAttrs = 2;
  check_cuda(
      cudaLaunchKernelExC(
          &config, reinterpret_cast<void*>(kernel), args),
      "cudaLaunchKernelExC(fused WoA)");
}

void launch_wo_a_tp2_overlap(
    const torch::Tensor& local_a, const torch::Tensor& remote_a,
    const torch::Tensor& local_sfa, const torch::Tensor& remote_sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const std::vector<int64_t>& signal_pad_ptrs,
    const torch::Tensor& grid_done, int64_t rank) {
  validate_tensor(local_a, torch::kFloat8_e4m3fn, "fused_wo_a_local_input");
  validate_tensor(remote_a, torch::kFloat8_e4m3fn, "fused_wo_a_remote_input");
  validate_tensor(local_sfa, torch::kInt32, "fused_wo_a_local_scale");
  validate_tensor(remote_sfa, torch::kInt32, "fused_wo_a_remote_scale");
  validate_tensor(b, torch::kFloat8_e4m3fn, "fused_wo_a_weight");
  validate_tensor(sfb, torch::kInt32, "fused_wo_a_weight_scale");
  validate_tensor(d, torch::kBFloat16, "fused_wo_a_output");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(local_a.dim() == 3 && remote_a.sizes() == local_a.sizes() &&
                  local_a.size(0) == kWoAGroups && local_a.size(2) == kWoAK &&
                  local_a.stride(2) == 1,
              "fused WoA inputs must be matching [8,M/2,4096] tensors");
  const int half_m = static_cast<int>(local_a.size(1));
  const int m = half_m * 2;
  TORCH_CHECK(m >= 2 && m <= kWoAMax,
              "fused TP2 WoA global M must be in [2,128]");
  TORCH_CHECK(d.sizes() == torch::IntArrayRef({kWoAGroups, m, kWoAN}) &&
                  d.stride(2) == 1,
              "fused WoA output must be [8,M,1024]");
  constexpr int kScaleK = kWoAK / (128 * 4);
  const int aligned_half_m = (half_m + 3) / 4 * 4;
  TORCH_CHECK(
      local_sfa.sizes() == torch::IntArrayRef({kWoAGroups, half_m, kScaleK}) &&
          remote_sfa.sizes() == local_sfa.sizes() &&
          local_sfa.stride(1) == 1 && local_sfa.stride(2) >= aligned_half_m &&
          local_sfa.stride(0) == kScaleK * local_sfa.stride(2) &&
          remote_sfa.stride(0) == local_sfa.stride(0) &&
          remote_sfa.stride(1) == local_sfa.stride(1) &&
          remote_sfa.stride(2) == local_sfa.stride(2),
      "fused WoA scales must have grouped MN-major layout");
  TORCH_CHECK(b.sizes() == torch::IntArrayRef({kWoAGroups, kWoAN, kWoAK}) &&
                  b.is_contiguous(),
              "fused WoA weight must be contiguous [8,1024,4096]");
  TORCH_CHECK(
      sfb.sizes() == torch::IntArrayRef({kWoAGroups, kWoAN, kScaleK}) &&
          sfb.stride(1) == 1 && sfb.stride(2) == kWoAN &&
          sfb.stride(0) == kScaleK * kWoAN,
      "fused WoA weight scales have the wrong layout");
  TORCH_CHECK(grid_done.is_cuda() && grid_done.scalar_type() == torch::kInt64 &&
                  grid_done.is_contiguous() && grid_done.numel() >= 2 &&
                  signal_pad_ptrs.size() == 2,
              "fused WoA requires TP2 generation and signal storage");
  const int fused_block_m =
      half_m <= 16 ? 16 : (half_m <= 32 ? 32 : 64);
  TORCH_CHECK(half_m % fused_block_m == 0,
              "fused WoA requires each token half to align to its M tile");

  if (half_m <= 16) {
    launch_wo_a_tp2_overlap_impl<12, 16>(
        local_a, remote_a, local_sfa, remote_sfa, b, sfb, d,
        signal_pad_ptrs, grid_done, rank);
  } else if (half_m <= 32) {
    launch_wo_a_tp2_overlap_impl<11, 32>(
        local_a, remote_a, local_sfa, remote_sfa, b, sfb, d,
        signal_pad_ptrs, grid_done, rank);
  } else {
    launch_wo_a_tp2_overlap_impl<10, 64>(
        local_a, remote_a, local_sfa, remote_sfa, b, sfb, d,
        signal_pad_ptrs, grid_done, rank);
  }
}

void launch_wo_a(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d) {
  validate_tensor(a, torch::kFloat8_e4m3fn, "wo_a_input");
  validate_tensor(sfa, torch::kInt32, "wo_a_input_scale");
  validate_tensor(b, torch::kFloat8_e4m3fn, "wo_a_weight");
  validate_tensor(sfb, torch::kInt32, "wo_a_weight_scale");
  validate_tensor(d, torch::kBFloat16, "wo_a_output");

  TORCH_CHECK(a.dim() == 3 && a.size(0) == kWoAGroups &&
                  a.size(2) == kWoAK && a.stride(2) == 1,
              "wo_a_input must be [8,M,4096] with K contiguous");
  const int m = static_cast<int>(a.size(1));
  TORCH_CHECK(m >= 1 && m <= kWoAMax, "WoA M must be in [1,128]");
  TORCH_CHECK(b.sizes() == torch::IntArrayRef({kWoAGroups, kWoAN, kWoAK}) &&
                  b.is_contiguous(),
              "wo_a_weight must be contiguous [8,1024,4096]");
  TORCH_CHECK(d.sizes() == torch::IntArrayRef({kWoAGroups, m, kWoAN}) &&
                  d.stride(2) == 1,
              "wo_a_output must be [8,M,1024] with N contiguous");
  const int aligned_m = (m + 3) / 4 * 4;
  constexpr int kScaleK = kWoAK / (128 * 4);
  TORCH_CHECK(
      sfa.sizes() == torch::IntArrayRef({kWoAGroups, m, kScaleK}) &&
          sfa.stride(1) == 1 && sfa.stride(2) == aligned_m &&
          sfa.stride(0) == kScaleK * aligned_m,
      "wo_a_input_scale has the wrong grouped MN-major layout");
  TORCH_CHECK(
      sfb.sizes() == torch::IntArrayRef({kWoAGroups, kWoAN, kScaleK}) &&
          sfb.stride(1) == 1 && sfb.stride(2) == kWoAN &&
          sfb.stride(0) == kScaleK * kWoAN,
      "wo_a_weight_scale has the wrong grouped MN-major layout");

  if (m <= 32) {
    launch_wo_a_impl<12, 16>(a, sfa, b, sfb, d);
  } else if (m <= 64) {
    launch_wo_a_impl<11, 32>(a, sfa, b, sfb, d);
  } else {
    launch_wo_a_impl<10, 64>(a, sfa, b, sfb, d);
  }
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
      "o_proj_a", &launch_wo_a,
      "Wuda grouped batched FP8 WoA GEMM",
      py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"),
      py::arg("d"));
  m.def(
      "o_proj_a_tp2_overlap", &launch_wo_a_tp2_overlap,
      "Wuda WoA consuming compact TP2 MLA slots with an in-kernel peer wait",
      py::arg("local_a"), py::arg("remote_a"),
      py::arg("local_sfa"), py::arg("remote_sfa"),
      py::arg("b"), py::arg("sfb"), py::arg("d"),
      py::arg("signal_pad_ptrs"), py::arg("grid_done"),
      py::arg("rank"));
  m.def(
      "o_proj_b", &launch,
      "TP2 local O-proj B with vector peer publication and fused mHC post",
      py::arg("a"), py::arg("sfa"), py::arg("b"), py::arg("sfb"),
      py::arg("symmetric_partials"), py::arg("block_generations"),
      py::arg("symmetric_ptrs"),
      py::arg("signal_pad_ptrs"), py::arg("local_projected"),
      py::arg("residual"), py::arg("post"), py::arg("comb"),
      py::arg("output"), py::arg("rank"));
  m.def(
      "benchmark_barrier", &launch_benchmark_barrier,
      "Excluded TP2 device rendezvous for paired benchmarks");
}
