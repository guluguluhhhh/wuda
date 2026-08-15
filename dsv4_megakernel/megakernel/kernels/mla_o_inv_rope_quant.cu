#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tp2_symmetric.cuh"

namespace {

constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kHeadsPerGroup = 8;
constexpr int kTpHeads = 64;
constexpr int kFullHeads = 128;
constexpr int kQuantGroup = 128;
constexpr uint32_t kMlaReadyOffset = 48;

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

__device__ __forceinline__ uint64_t arrive_grid(uint64_t* ptr) {
  uint64_t previous;
  asm volatile("atom.release.gpu.global.add.u64 %0, [%1], 1;"
               : "=l"(previous) : "l"(ptr) : "memory");
  return previous + 1;
}

__device__ __forceinline__ uint64_t load_grid_acquire(
    const uint64_t* pointer) {
  uint64_t value;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(value) : "l"(pointer) : "memory");
  return value;
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint32_t ceil_ue8m0(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t biased_exponent = (bits >> 23) & 0xffu;
  if (biased_exponent == 0xffu) {
    return 254u;
  }
  // 448 = 1.75 * 2^8.  All candidates are BF16 values (or the 1e-4
  // floor), so comparing the FP32 mantissa against exact 1.75 gives the
  // same ceil(log2(value / 448)) as a correctly rounded FP32 division.
  const uint32_t mantissa = bits & 0x7fffffu;
  const int exponent = static_cast<int>(biased_exponent) - 8 +
                       (mantissa > 0x600000u);
  return static_cast<uint32_t>(min(max(exponent, 1), 254));
}

__device__ __forceinline__ void store_peer_u32(
    uint32_t* pointer, uint32_t value) {
  asm volatile("st.global.u32 [%0], %1;" ::
               "l"(pointer), "r"(value) : "memory");
}

__device__ __forceinline__ void start_peer_bulk_store(
    void* destination, const void* source, uint32_t bytes) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(source));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
      :: "l"(destination), "r"(shared_address), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_peer_bulk_store() {
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

template <int kHeadsPerCta, int kInputHeads, bool kTp2,
          bool kBatch128 = false>
__global__ void inv_rope_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin,
    uint8_t* __restrict__ local_fp8,
    uint8_t* __restrict__ peer_fp8,
    uint32_t* __restrict__ local_scale,
    uint32_t* __restrict__ peer_scale,
    uint64_t* __restrict__ grid_done,
    const uint32_t* __restrict__ local_ready,
    uint32_t* __restrict__ peer_ready,
    int input_m,
    int global_m,
    int aligned_m,
    int fp8_group_stride,
    int fp8_token_stride,
    int scale_group_stride,
    int scale_head_stride,
    int cache_stride,
    uint32_t rank) {
  static_assert(kHeadsPerCta == 1 || kHeadsPerCta == 2 ||
                kHeadsPerCta == 4 || kHeadsPerCta == 8);
  static_assert(kInputHeads == kTpHeads || kInputHeads == kFullHeads);
  static_assert(!kTp2 || kInputHeads == kFullHeads);
  static_assert(!kBatch128 || kTp2);
  constexpr uint32_t kFullMask = 0xffffffffu;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  __shared__ __align__(16) uint8_t peer_staging[
      kTp2 ? kHeadsPerCta * kHeadDim : 16];
  constexpr int kHeadTiles = kInputHeads / kHeadsPerCta;
  const int effective_global_m = kBatch128 ? 128 : global_m;
  const int effective_input_m = kBatch128 ? 64 : input_m;
  const int effective_aligned_m = kBatch128 ? 128 : aligned_m;
  const int padding = effective_aligned_m - effective_global_m;
  const int work_m = kTp2
      ? effective_input_m + (rank == 1 ? padding : 0)
      : effective_aligned_m;
  const int total_tiles = work_m * kHeadTiles;

  for (int task = blockIdx.x; task < total_tiles; task += gridDim.x) {
    const int local_token = task / kHeadTiles;
    const int logical_head_tile = task % kHeadTiles;
    int source_head_base;
    if constexpr (kTp2) {
      const int destination = logical_head_tile & 1;
      const int tile_in_destination = logical_head_tile >> 1;
      source_head_base = destination * kTpHeads +
                         tile_in_destination * kHeadsPerCta;
    } else {
      source_head_base = logical_head_tile * kHeadsPerCta;
    }

    const int source_head = source_head_base + warp;
    const int global_token = kTp2
        ? local_token + (rank == 0 ? 0 : (effective_global_m + 1) / 2)
        : local_token;
    const bool valid_token = kBatch128 ||
        (local_token < effective_input_m && global_token < effective_global_m);

    const int destination_rank = kTp2 ? source_head / kTpHeads : 0;
    const int destination_head = kTp2 ? source_head % kTpHeads : source_head;
    const int group = destination_head / kHeadsPerGroup;
    const int head_in_group = destination_head % kHeadsPerGroup;
    const bool remote = kTp2 && destination_rank != static_cast<int>(rank);
    auto* fp8_output = remote ? peer_fp8 : local_fp8;
    auto* scale_output = remote ? peer_scale : local_scale;

    if (!valid_token) {
      if (lane == 0) {
        auto* address = scale_output + group * scale_group_stride +
                        global_token + head_in_group * scale_head_stride;
        if (remote) {
          store_peer_u32(address, 0u);
        } else {
          *address = 0u;
        }
      }
      continue;
    }

    const int quant_group = lane / 8;
    const int lane_in_group = lane % 8;
    const int element = quant_group * kQuantGroup + lane_in_group * 16;
    const auto* input_head = input +
        (static_cast<int64_t>(local_token) * kInputHeads + source_head) *
            kHeadDim;
    uint4 input_vectors[2];
    input_vectors[0] = *reinterpret_cast<const uint4*>(input_head + element);
    input_vectors[1] = *reinterpret_cast<const uint4*>(input_head + element + 8);
    const auto* input_values =
        reinterpret_cast<const __nv_bfloat16*>(input_vectors);

    float values[16];
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      values[index] = __bfloat162float(input_values[index]);
    }

    if (element >= kNopeDim) {
      const int64_t position = positions[global_token];
      const auto* cache = cos_sin + position * cache_stride;
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
    // UE8M0 scales are exact powers of two.  Build their reciprocal directly
    // so quantization needs only multiplies and retains bit-identical rounding.
    const float inverse_scale = exponent == 254u
        ? __uint_as_float(0x00400000u)
        : __uint_as_float((254u - exponent) << 23);

    uint4 packed_fp8{};
    auto* packed_pairs =
        reinterpret_cast<__nv_fp8x2_storage_t*>(&packed_fp8);
#pragma unroll
    for (int index = 0; index < 16; index += 2) {
      float2 quantized{
          values[index] * inverse_scale,
          values[index + 1] * inverse_scale};
      packed_pairs[index / 2] = __nv_cvt_float2_to_fp8x2(
          quantized, __NV_SATFINITE, __NV_E4M3);
    }

    auto* fp8_address = fp8_output + group * fp8_group_stride +
                        global_token * fp8_token_stride +
                        head_in_group * kHeadDim + element;
    if (remote) {
      auto* warp_staging = peer_staging + warp * kHeadDim;
      *reinterpret_cast<uint4*>(warp_staging + element) = packed_fp8;
      __syncthreads();
      if (threadIdx.x == 0) {
        start_peer_bulk_store(
            fp8_address, peer_staging, kHeadsPerCta * kHeadDim);
      }
    } else {
      *reinterpret_cast<uint4*>(fp8_address) = packed_fp8;
    }

    uint32_t packed_scale = lane_in_group == 0
        ? exponent << (quant_group * 8)
        : 0;
    packed_scale |= __shfl_xor_sync(kFullMask, packed_scale, 16);
    packed_scale |= __shfl_xor_sync(kFullMask, packed_scale, 8);
    if (lane == 0) {
      auto* scale_address = scale_output + group * scale_group_stride +
                            global_token +
                            head_in_group * scale_head_stride;
      if (remote) {
        store_peer_u32(scale_address, packed_scale);
      } else {
        *scale_address = packed_scale;
      }
    }
    if (remote) {
      if (threadIdx.x == 0) {
        wait_peer_bulk_store();
      }
      __syncthreads();
    }
  }

  if constexpr (kTp2) {
    const bool remote_block = (blockIdx.x & 1u) != rank;
    __syncthreads();
    if (threadIdx.x == 0 && remote_block) {
      // Only the CTAs that publish to the peer join this release sequence.
      // Local work is ordered by normal kernel completion, while the peer can
      // publish readiness as soon as the payload needed by the other WoA is done.
      const uint64_t arrived = arrive_grid(grid_done);
      const uint32_t remote_blocks = gridDim.x / 2;
      if (arrived % remote_blocks == 0) {
        while (load_grid_acquire(grid_done) < arrived) {
          wuda::tp2::spin_pause();
        }
        const uint32_t generation =
            static_cast<uint32_t>(arrived / remote_blocks);
        wuda::tp2::store_release_sys(
            peer_ready + kMlaReadyOffset + rank, generation);
        const uint32_t peer_rank = rank ^ 1u;
        const uint64_t start = globaltimer();
        uint32_t spins = 0;
        while (wuda::tp2::load_acquire_sys(
                   local_ready + kMlaReadyOffset + peer_rank) < generation) {
          wuda::tp2::spin_pause();
          if ((++spins & 1023u) == 0u &&
              globaltimer() - start > 10'000'000'000ull) {
            printf("TP2 MLA quant barrier timeout rank=%u generation=%u\n",
                   rank, generation);
            asm volatile("trap;");
          }
        }
      }
    }
  }
}

void validate_common(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& output_fp8,
    const torch::Tensor& output_scale,
    int global_m,
    int output_heads) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 &&
                  input.is_contiguous() && input.dim() == 3,
              "input must be contiguous CUDA BF16 [M,H,512]");
  TORCH_CHECK(input.size(2) == kHeadDim,
              "input head dimension must be 512");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 &&
                  positions.is_contiguous() && positions.dim() == 1 &&
                  positions.size(0) == global_m,
              "positions must be contiguous CUDA I64 [global_M]");
  TORCH_CHECK(cos_sin.is_cuda() && cos_sin.scalar_type() == torch::kFloat32 &&
                  cos_sin.is_contiguous() && cos_sin.dim() == 2 &&
                  cos_sin.size(1) == kRopeDim,
              "cos_sin must be contiguous CUDA FP32 [max_pos,64]");
  const int groups = output_heads / kHeadsPerGroup;
  TORCH_CHECK(output_fp8.is_cuda() &&
                  output_fp8.scalar_type() == torch::kFloat8_e4m3fn &&
                  output_fp8.dim() == 3 && output_fp8.size(0) == global_m &&
                  output_fp8.size(1) == groups &&
                  output_fp8.size(2) == kHeadsPerGroup * kHeadDim &&
                  output_fp8.stride(0) == kHeadsPerGroup * kHeadDim &&
                  output_fp8.stride(2) == 1,
              "output_fp8 has the wrong grouped layout");
  const int aligned_m = (global_m + 3) / 4 * 4;
  TORCH_CHECK(output_scale.is_cuda() &&
                  output_scale.scalar_type() == torch::kInt32 &&
                  output_scale.dim() == 3 &&
                  output_scale.sizes() ==
                      torch::IntArrayRef({global_m, groups, kHeadsPerGroup}) &&
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

template <int kHeadsPerCta, int kInputHeads, bool kTp2,
          bool kBatch128 = false>
void launch_kernel(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& local_fp8,
    const torch::Tensor& peer_fp8,
    const torch::Tensor& local_scale,
    const torch::Tensor& peer_scale,
    const torch::Tensor& grid_done,
    const std::vector<int64_t>& signal_pad_ptrs,
    int global_m,
    uint32_t rank) {
  const int input_m = static_cast<int>(input.size(0));
  const int effective_global_m = kBatch128 ? 128 : global_m;
  const int effective_input_m = kBatch128 ? 64 : input_m;
  const int aligned_m = (effective_global_m + 3) / 4 * 4;
  const int padding = aligned_m - effective_global_m;
  const int work_m = kTp2
      ? effective_input_m + (rank == 1 ? padding : 0)
      : aligned_m;
  const int total_tiles = work_m * (kInputHeads / kHeadsPerCta);
  const int threads = kHeadsPerCta * 32;
  int blocks_per_sm = 1;
  check_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          inv_rope_quant_kernel<
              kHeadsPerCta, kInputHeads, kTp2, kBatch128>, threads, 0),
      "MLA inverse-RoPE quant occupancy");
  const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int blocks = std::min(
      total_tiles, sm_count * std::min(blocks_per_sm, 5));
  if constexpr (kTp2) {
    blocks &= ~1;
  }
  TORCH_CHECK(!kTp2 || (blocks >= 2 && blocks % 2 == 0),
              "TP2 quant launch requires an even CTA count");

  const uint32_t* local_ready = nullptr;
  uint32_t* peer_ready = nullptr;
  uint64_t* grid_done_ptr = nullptr;
  if constexpr (kTp2) {
    const auto signals = wuda::tp2::make_symmetric_view(signal_pad_ptrs, rank);
    local_ready = signals.local<const uint32_t>();
    peer_ready = signals.peer_base<uint32_t>();
    grid_done_ptr = reinterpret_cast<uint64_t*>(grid_done.data_ptr<int64_t>());
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  inv_rope_quant_kernel<kHeadsPerCta, kInputHeads, kTp2, kBatch128>
      <<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      positions.data_ptr<int64_t>(),
      cos_sin.data_ptr<float>(),
      reinterpret_cast<uint8_t*>(local_fp8.data_ptr()),
      kTp2 ? reinterpret_cast<uint8_t*>(peer_fp8.data_ptr()) : nullptr,
      reinterpret_cast<uint32_t*>(local_scale.data_ptr<int32_t>()),
      kTp2 ? reinterpret_cast<uint32_t*>(peer_scale.data_ptr<int32_t>()) : nullptr,
      grid_done_ptr,
      local_ready,
      peer_ready,
      input_m,
      global_m,
      aligned_m,
      static_cast<int>(local_fp8.stride(1)),
      static_cast<int>(local_fp8.stride(0)),
      static_cast<int>(local_scale.stride(1)),
      static_cast<int>(local_scale.stride(2)),
      static_cast<int>(cos_sin.stride(0)),
      rank);
  check_cuda(cudaGetLastError(), "MLA inverse-RoPE quant launch");
}

template <int kInputHeads, bool kTp2>
void dispatch(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& local_fp8,
    const torch::Tensor& peer_fp8,
    const torch::Tensor& local_scale,
    const torch::Tensor& peer_scale,
    const torch::Tensor& grid_done,
    const std::vector<int64_t>& signal_pad_ptrs,
    int global_m,
    uint32_t rank) {
  const int local_m = static_cast<int>(input.size(0));
  if (local_m <= 2) {
    launch_kernel<1, kInputHeads, kTp2>(
        input, positions, cos_sin, local_fp8, peer_fp8,
        local_scale, peer_scale, grid_done, signal_pad_ptrs, global_m, rank);
  } else if (local_m <= 8) {
    launch_kernel<2, kInputHeads, kTp2>(
        input, positions, cos_sin, local_fp8, peer_fp8,
        local_scale, peer_scale, grid_done, signal_pad_ptrs, global_m, rank);
  } else if (local_m <= 32) {
    launch_kernel<4, kInputHeads, kTp2>(
        input, positions, cos_sin, local_fp8, peer_fp8,
        local_scale, peer_scale, grid_done, signal_pad_ptrs, global_m, rank);
  } else {
    if constexpr (kTp2) {
      if (global_m == 128) {
        launch_kernel<8, kInputHeads, kTp2, true>(
            input, positions, cos_sin, local_fp8, peer_fp8,
            local_scale, peer_scale, grid_done, signal_pad_ptrs, global_m, rank);
      } else {
        launch_kernel<8, kInputHeads, kTp2>(
            input, positions, cos_sin, local_fp8, peer_fp8,
            local_scale, peer_scale, grid_done, signal_pad_ptrs, global_m, rank);
      }
    } else {
      launch_kernel<8, kInputHeads, kTp2>(
          input, positions, cos_sin, local_fp8, peer_fp8,
          local_scale, peer_scale, grid_done, signal_pad_ptrs, global_m, rank);
    }
  }
}

void launch_local(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& output_fp8,
    const torch::Tensor& output_scale) {
  const int m = static_cast<int>(input.size(0));
  const int heads = static_cast<int>(input.size(1));
  TORCH_CHECK(heads == kTpHeads || heads == kFullHeads,
              "local input must contain 64 or 128 heads");
  validate_common(input, positions, cos_sin, output_fp8, output_scale, m, heads);
  if (heads == kTpHeads) {
    dispatch<kTpHeads, false>(
        input, positions, cos_sin, output_fp8, torch::Tensor(),
        output_scale, torch::Tensor(), torch::Tensor(), {}, m, 0);
  } else {
    dispatch<kFullHeads, false>(
        input, positions, cos_sin, output_fp8, torch::Tensor(),
        output_scale, torch::Tensor(), torch::Tensor(), {}, m, 0);
  }
}

void launch_tp2(
    const torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin,
    const torch::Tensor& local_fp8,
    const torch::Tensor& peer_fp8,
    const torch::Tensor& local_scale,
    const torch::Tensor& peer_scale,
    const torch::Tensor& grid_done,
    const std::vector<int64_t>& signal_pad_ptrs,
    int64_t rank) {
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  const int global_m = static_cast<int>(positions.size(0));
  const int expected_local_m = rank == 0 ? (global_m + 1) / 2 : global_m / 2;
  TORCH_CHECK(input.size(0) == expected_local_m && input.size(1) == kFullHeads,
              "TP2 input must be the rank-local [M,128,512] FlashMLA output");
  validate_common(
      input, positions, cos_sin, local_fp8, local_scale, global_m, kTpHeads);
  TORCH_CHECK(peer_fp8.is_cuda() && peer_fp8.scalar_type() == torch::kUInt8 &&
                  peer_fp8.numel() == local_fp8.numel(),
              "peer_fp8 must be a symmetric byte view of local_fp8");
  TORCH_CHECK(peer_scale.is_cuda() &&
                  peer_scale.scalar_type() == torch::kInt32 &&
                  peer_scale.numel() ==
                      kTpHeads * ((global_m + 3) / 4 * 4),
              "peer_scale must cover the symmetric scale allocation");
  TORCH_CHECK(grid_done.is_cuda() && grid_done.scalar_type() == torch::kInt64 &&
                  grid_done.numel() == 1,
              "grid_done must be one CUDA int64");
  TORCH_CHECK(signal_pad_ptrs.size() == 2,
              "exactly two signal-pad pointers are required");
  dispatch<kFullHeads, true>(
      input, positions, cos_sin, local_fp8, peer_fp8,
      local_scale, peer_scale, grid_done, signal_pad_ptrs,
      global_m, static_cast<uint32_t>(rank));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inv_rope_quant", &launch_local,
        "Single-kernel inverse-RoPE and FP8 quantization");
  m.def("inv_rope_quant_tp2", &launch_tp2,
        "Single-kernel TP2 inverse-RoPE, FP8 quantization, and exchange");
}
