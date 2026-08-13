// mHC post kernel + torch binding, ported from mega_csa
// (Flash_DeepSeek_V4_Pro): src/runtime/mhc_post.cu, its detail/buffer_range.h
// overlap check, and tests/integration/bindings/mhc_post_binding.cu, merged
// into one translation unit -- this tree has a single consumer, so the upstream
// cross-TU forward declaration (detail/validated_launches.h) is unnecessary and
// bindings live next to the kernel like every other kernel here.
// The math is unchanged.

#include <mhc_post.cuh>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <cstdint>

namespace mega::csa {
namespace {

constexpr int kValuesPerVector = 8;
constexpr int kVectorsPerRow = kMhcHiddenDim / kValuesPerVector;
constexpr int kThreads = 256;

bool aligned(const void* pointer, std::uintptr_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

struct BufferRange {
  const void* data;
  std::size_t bytes;
};

bool ranges_overlap(const BufferRange& lhs, const BufferRange& rhs) {
  if (lhs.bytes == 0 || rhs.bytes == 0) return false;
  const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
  const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
  if (lhs_begin <= rhs_begin) return rhs_begin - lhs_begin < lhs.bytes;
  return lhs_begin - rhs_begin < rhs.bytes;
}

__global__ void mhc_post_kernel(const __nv_bfloat16* attention_out,
                                const __nv_bfloat16* residual,
                                const float* post, const float* comb,
                                __nv_bfloat16* output, int m) {
  __shared__ float mix[20];
  const int row = static_cast<int>(blockIdx.y);
  if (row >= m) return;

  if (threadIdx.x < 4) mix[threadIdx.x] = post[row * 4 + threadIdx.x];
  if (threadIdx.x < 16) mix[4 + threadIdx.x] = comb[row * 16 + threadIdx.x];
  __syncthreads();

  const int vector = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool valid = vector < kVectorsPerRow;
  const int dim = vector * kValuesPerVector;
  alignas(16) __nv_bfloat16 r[4][8];
  if (valid) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *reinterpret_cast<uint4*>(r[i]) = *reinterpret_cast<const uint4*>(
          residual + (row * 4 + i) * kMhcHiddenDim + dim);
    }
  }

  if (valid) {
    alignas(16) __nv_bfloat16 x[8];
    *reinterpret_cast<uint4*>(x) = *reinterpret_cast<const uint4*>(
        attention_out + row * kMhcHiddenDim + dim);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      alignas(16) __nv_bfloat16 out[8];
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        float value = mix[j] * __bfloat162float(x[lane]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          value = fmaf(mix[4 + i * 4 + j], __bfloat162float(r[i][lane]), value);
        }
        out[lane] = __float2bfloat16_rn(value);
      }
      *reinterpret_cast<uint4*>(output + (row * 4 + j) * kMhcHiddenDim + dim) =
          *reinterpret_cast<const uint4*>(out);
    }
  }

}

// Launch-only entry (upstream detail::mhc_post_run_validated): the caller must
// have completed mhc_post_validate for the whole composed chain first.
cudaError_t launch_validated(const MhcPostArgs& args, cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid((kVectorsPerRow + kThreads - 1) / kThreads, args.m);
  mhc_post_kernel<<<grid, block, 0, stream>>>(
      args.attention_out, args.residual, args.post, args.comb, args.output,
      args.m);
  return cudaGetLastError();
}

}  // namespace

cudaError_t mhc_post_validate(const MhcPostArgs& args) {
  if (args.attention_out == nullptr || args.residual == nullptr ||
      args.post == nullptr || args.comb == nullptr || args.output == nullptr ||
      args.m < 1 || args.m > 128 || !aligned(args.attention_out, 16) ||
      !aligned(args.residual, 16) || !aligned(args.output, 16) ||
      !aligned(args.post, alignof(float)) ||
      !aligned(args.comb, alignof(float))) {
    return cudaErrorInvalidValue;
  }

  const auto m = static_cast<std::size_t>(args.m);
  const BufferRange output{
      args.output,
      m * kMhcResidualStreams * kMhcHiddenDim * sizeof(__nv_bfloat16)};
  const BufferRange inputs[] = {
      {args.attention_out, m * kMhcHiddenDim * sizeof(__nv_bfloat16)},
      {args.residual,
       m * kMhcResidualStreams * kMhcHiddenDim * sizeof(__nv_bfloat16)},
      {args.post, m * kMhcResidualStreams * sizeof(float)},
      {args.comb,
       m * kMhcResidualStreams * kMhcResidualStreams * sizeof(float)},
  };
  for (const auto& input : inputs) {
    if (ranges_overlap(output, input)) {
      return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

cudaError_t mhc_post_run(const MhcPostArgs& args, cudaStream_t stream) {
  auto status = mhc_post_validate(args);
  if (status != cudaSuccess) return status;
  return launch_validated(args, stream);
}

}  // namespace mega::csa

namespace {

void mhc_post_out(torch::Tensor attention_out, torch::Tensor residual,
                  torch::Tensor post, torch::Tensor comb,
                  torch::Tensor output) {
  const int64_t m = attention_out.size(0);
  TORCH_CHECK(attention_out.is_cuda() && attention_out.scalar_type() == torch::kBFloat16 &&
                  attention_out.is_contiguous() &&
                  attention_out.sizes() == torch::IntArrayRef({m, 7168}),
              "attention_out must be contiguous CUDA BF16 [M,7168]");
  TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == torch::kBFloat16 &&
                  residual.is_contiguous() &&
                  residual.sizes() == torch::IntArrayRef({m, 4, 7168}),
              "residual must be contiguous CUDA BF16 [M,4,7168]");
  TORCH_CHECK(post.is_cuda() && post.scalar_type() == torch::kFloat32 &&
                  post.is_contiguous() &&
                  post.sizes() == torch::IntArrayRef({m, 4}),
              "post must be contiguous CUDA FP32 [M,4]");
  TORCH_CHECK(comb.is_cuda() && comb.scalar_type() == torch::kFloat32 &&
                  comb.is_contiguous() &&
                  comb.sizes() == torch::IntArrayRef({m, 4, 4}),
              "comb must be contiguous CUDA FP32 [M,4,4]");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
                  output.is_contiguous() &&
                  output.sizes() == torch::IntArrayRef({m, 4, 7168}),
              "output must be contiguous CUDA BF16 [M,4,7168]");

  mega::csa::MhcPostArgs args{};
  args.attention_out = reinterpret_cast<const __nv_bfloat16*>(attention_out.data_ptr());
  args.residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  args.post = post.data_ptr<float>();
  args.comb = comb.data_ptr<float>();
  args.output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  args.m = static_cast<int>(m);
  const cudaError_t status = mega::csa::mhc_post_run(
      args, at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(status == cudaSuccess, "mhc_post_run: ", cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mhc_post_out", &mhc_post_out);
}
