#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kHeadsPerGroup = 8;
constexpr int kTpHeads = 64;
constexpr int kFullHeads = 128;
constexpr int kQuantGroup = 128;

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

__device__ __forceinline__ uint32_t ceil_ue8m0(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t biased_exponent = (bits >> 23) & 0xffu;
  if (biased_exponent == 0xffu) {
    return 254u;
  }
  // 448 = 1.75 * 2^8. All candidates are BF16 values (or the 1e-4
  // floor), so this matches ceil(log2(value / 448)) exactly.
  const uint32_t mantissa = bits & 0x7fffffu;
  const int exponent = static_cast<int>(biased_exponent) - 8 +
                       (mantissa > 0x600000u);
  return static_cast<uint32_t>(min(max(exponent, 1), 254));
}

template <int kHeadsPerCta, int kInputHeads>
__global__ void inv_rope_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin,
    uint8_t* __restrict__ output_fp8,
    uint32_t* __restrict__ output_scale,
    int input_m,
    int aligned_m,
    int fp8_group_stride,
    int fp8_token_stride,
    int scale_group_stride,
    int scale_head_stride,
    int cache_stride) {
  static_assert(kHeadsPerCta == 1 || kHeadsPerCta == 2 ||
                kHeadsPerCta == 4 || kHeadsPerCta == 8);
  static_assert(kInputHeads == kTpHeads || kInputHeads == kFullHeads);
  constexpr uint32_t kFullMask = 0xffffffffu;
  constexpr int kHeadTiles = kInputHeads / kHeadsPerCta;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int total_tiles = aligned_m * kHeadTiles;

  for (int task = blockIdx.x; task < total_tiles; task += gridDim.x) {
    const int token = task / kHeadTiles;
    const int head_tile = task % kHeadTiles;
    const int head = head_tile * kHeadsPerCta + warp;
    const int group = head / kHeadsPerGroup;
    const int head_in_group = head % kHeadsPerGroup;

    if (token >= input_m) {
      if (lane == 0) {
        output_scale[group * scale_group_stride + token +
                     head_in_group * scale_head_stride] = 0u;
      }
      continue;
    }

    const int quant_group = lane / 8;
    const int lane_in_group = lane % 8;
    const int element = quant_group * kQuantGroup + lane_in_group * 16;
    const auto* input_head = input +
        (static_cast<int64_t>(token) * kInputHeads + head) * kHeadDim;
    uint4 input_vectors[2];
    input_vectors[0] = *reinterpret_cast<const uint4*>(input_head + element);
    input_vectors[1] =
        *reinterpret_cast<const uint4*>(input_head + element + 8);
    const auto* input_values =
        reinterpret_cast<const __nv_bfloat16*>(input_vectors);

    float values[16];
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      values[index] = __bfloat162float(input_values[index]);
    }

    if (element >= kNopeDim) {
      const auto* cache = cos_sin + positions[token] * cache_stride;
#pragma unroll
      for (int index = 0; index < 16; index += 2) {
        const int pair = (element + index - kNopeDim) / 2;
        const float real = values[index];
        const float imag = values[index + 1];
        const float cosine = cache[pair];
        const float sine = cache[kRopeDim / 2 + pair];
        const __nv_bfloat162 rounded = __floats2bfloat162_rn(
            real * cosine + imag * sine,
            imag * cosine - real * sine);
        values[index] = __bfloat162float(__low2bfloat16(rounded));
        values[index + 1] = __bfloat162float(__high2bfloat16(rounded));
      }
    }

    float amax = 1.0e-4f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      amax = fmaxf(amax, fabsf(values[index]));
    }
#pragma unroll
    for (int delta = 4; delta > 0; delta >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(kFullMask, amax, delta, 8));
    }
    const uint32_t exponent = ceil_ue8m0(amax);
    const float inverse_scale = exponent == 254u
        ? __uint_as_float(0x00400000u)
        : __uint_as_float((254u - exponent) << 23);

    uint4 packed_fp8{};
    auto* packed_pairs =
        reinterpret_cast<__nv_fp8x2_storage_t*>(&packed_fp8);
#pragma unroll
    for (int index = 0; index < 16; index += 2) {
      const float2 quantized{
          values[index] * inverse_scale,
          values[index + 1] * inverse_scale};
      packed_pairs[index / 2] = __nv_cvt_float2_to_fp8x2(
          quantized, __NV_SATFINITE, __NV_E4M3);
    }

    auto* fp8_address = output_fp8 + group * fp8_group_stride +
                        token * fp8_token_stride +
                        head_in_group * kHeadDim + element;
    *reinterpret_cast<uint4*>(fp8_address) = packed_fp8;

    uint32_t packed_scale = lane_in_group == 0
        ? exponent << (quant_group * 8)
        : 0;
    packed_scale |= __shfl_xor_sync(kFullMask, packed_scale, 16);
    packed_scale |= __shfl_xor_sync(kFullMask, packed_scale, 8);
    if (lane == 0) {
      output_scale[group * scale_group_stride + token +
                   head_in_group * scale_head_stride] = packed_scale;
    }
  }
}

void validate(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& output_fp8,
    const torch::Tensor& output_scale) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 &&
                  input.is_contiguous() && input.dim() == 3,
              "input must be contiguous CUDA BF16 [M,H,512]");
  const int m = static_cast<int>(input.size(0));
  const int heads = static_cast<int>(input.size(1));
  TORCH_CHECK(1 <= m && m <= 128 &&
                  (heads == kTpHeads || heads == kFullHeads) &&
                  input.size(2) == kHeadDim,
              "input must have M in [1,128], H in {64,128}, and D=512");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.is_contiguous() && positions.sizes() ==
                      torch::IntArrayRef({m}),
              "positions must be contiguous CUDA I64 [M]");
  TORCH_CHECK(cos_sin.is_cuda() && cos_sin.scalar_type() == torch::kFloat32 &&
                  cos_sin.is_contiguous() && cos_sin.dim() == 2 &&
                  cos_sin.size(1) == kRopeDim,
              "cos_sin must be contiguous CUDA FP32 [max_pos,64]");
  const int groups = heads / kHeadsPerGroup;
  TORCH_CHECK(output_fp8.is_cuda() &&
                  output_fp8.scalar_type() == torch::kFloat8_e4m3fn &&
                  output_fp8.sizes() == torch::IntArrayRef(
                      {m, groups, kHeadsPerGroup * kHeadDim}) &&
                  output_fp8.stride(0) == kHeadsPerGroup * kHeadDim &&
                  output_fp8.stride(2) == 1,
              "output_fp8 has the wrong grouped layout");
  const int aligned_m = (m + 3) / 4 * 4;
  TORCH_CHECK(output_scale.is_cuda() &&
                  output_scale.scalar_type() == torch::kInt32 &&
                  output_scale.sizes() ==
                      torch::IntArrayRef({m, groups, kHeadsPerGroup}) &&
                  output_scale.stride(0) == 1 &&
                  output_scale.stride(1) == kHeadsPerGroup * aligned_m &&
                  output_scale.stride(2) == aligned_m,
              "output_scale has the wrong MN-major layout");
  TORCH_CHECK(input.device() == positions.device() &&
                  input.device() == cos_sin.device() &&
                  input.device() == output_fp8.device() &&
                  input.device() == output_scale.device(),
              "all tensors must be on the same CUDA device");
}

template <int kHeadsPerCta, int kInputHeads>
void launch_kernel(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& output_fp8,
    const torch::Tensor& output_scale) {
  const int m = static_cast<int>(input.size(0));
  const int aligned_m = (m + 3) / 4 * 4;
  const int total_tiles = aligned_m * (kInputHeads / kHeadsPerCta);
  const int threads = kHeadsPerCta * 32;
  int blocks_per_sm = 1;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          inv_rope_quant_kernel<kHeadsPerCta, kInputHeads>, threads, 0),
      "MLA inverse-RoPE quant occupancy");
  const int sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int blocks = std::min(
      total_tiles, sm_count * std::min(blocks_per_sm, 5));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  inv_rope_quant_kernel<kHeadsPerCta, kInputHeads>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(
              input.data_ptr<at::BFloat16>()),
          positions.data_ptr<int64_t>(),
          cos_sin.data_ptr<float>(),
          reinterpret_cast<uint8_t*>(output_fp8.data_ptr()),
          reinterpret_cast<uint32_t*>(output_scale.data_ptr<int32_t>()),
          m,
          aligned_m,
          static_cast<int>(output_fp8.stride(1)),
          static_cast<int>(output_fp8.stride(0)),
          static_cast<int>(output_scale.stride(1)),
          static_cast<int>(output_scale.stride(2)),
          static_cast<int>(cos_sin.stride(0)));
  check_cuda(cudaGetLastError(), "MLA inverse-RoPE quant launch");
}

template <int kInputHeads>
void dispatch(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& output_fp8,
    const torch::Tensor& output_scale) {
  const int m = static_cast<int>(input.size(0));
  if (m <= 2) {
    launch_kernel<1, kInputHeads>(
        input, positions, cos_sin, output_fp8, output_scale);
  } else if (m <= 8) {
    launch_kernel<2, kInputHeads>(
        input, positions, cos_sin, output_fp8, output_scale);
  } else if (m <= 32) {
    launch_kernel<4, kInputHeads>(
        input, positions, cos_sin, output_fp8, output_scale);
  } else {
    launch_kernel<8, kInputHeads>(
        input, positions, cos_sin, output_fp8, output_scale);
  }
}

void launch(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& output_fp8,
    const torch::Tensor& output_scale) {
  validate(input, positions, cos_sin, output_fp8, output_scale);
  if (input.size(1) == kTpHeads) {
    dispatch<kTpHeads>(
        input, positions, cos_sin, output_fp8, output_scale);
  } else {
    dispatch<kFullHeads>(
        input, positions, cos_sin, output_fp8, output_scale);
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv_rope_quant", &launch,
        "Single-kernel inverse-RoPE and FP8 quantization");
}
