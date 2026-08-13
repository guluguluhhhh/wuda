// Query RMSNorm + RoPE for the official MODEL1 FlashMLA query layout.
//
// Input rows are [batch, input_heads, 512] BF16.  Output rows are
// [output_batch, output_heads, 512] BF16; output_head % input_heads selects
// the source head, which performs the TP->DP head duplication without glue.
// RMSNorm has no gamma (matching origin/model.py), and RoPE is applied to the
// final 64 dimensions.  A capped persistent grid keeps this side branch small
// enough to overlap the MQA kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#include "query_rms_rope.cuh"

namespace {

using bf16 = __nv_bfloat16;
using namespace wuda_query_rms_rope;

constexpr int kWarpsPerBlock = 4;

template <bool kUsePdl>
__global__ __launch_bounds__(kWarpsPerBlock * kWarp)
void rmsnorm_rope_kernel(
        const bf16* __restrict__ x, const int64_t* __restrict__ positions,
        const float* __restrict__ cos_table,
        const float* __restrict__ sin_table, bf16* __restrict__ y,
        int input_heads, int output_heads, int64_t output_rows, float eps) {
    if constexpr (kUsePdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // Q_B may still be running while this grid performs launch setup.
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // MQA reads a different Q_B output.  Release it after Q_B is known
        // complete, before this side branch consumes any SM execution time.
        asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
    }

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

    for (int64_t out_row = first; out_row < output_rows; out_row += stride) {
        const int batch = static_cast<int>(out_row / output_heads);
        const int out_head = static_cast<int>(out_row -
                                               static_cast<int64_t>(batch) * output_heads);
        const int in_head = out_head % input_heads;
        const int64_t in_row = static_cast<int64_t>(batch) * input_heads + in_head;

        const uint4* xp = reinterpret_cast<const uint4*>(x + in_row * kHidden) + lane;
        uint4* yp = reinterpret_cast<uint4*>(y + out_row * kHidden) + lane;
        uint4 xv[kVecsPerThread];
#pragma unroll
        for (int v = 0; v < kVecsPerThread; ++v)
            xv[v] = xp[v * kWarp];

        float acc = 0.f;
#pragma unroll
        for (int v = 0; v < kVecsPerThread; ++v)
            acc += vec_sumsq(xv[v]);
        const float scale = rsqrtf(warp_sum(acc) / kHidden + eps);

        const int64_t pos = positions[batch];
        const float* cos = cos_table + pos * (kRope / 2);
        const float* sin = sin_table + pos * (kRope / 2);
        yp[0] = scale_vec(xv[0], scale);
        if (lane >= (kHidden - kRope) / kElemsPerVec - kWarp) {
            const int pair_base = (lane - 24) * (kElemsPerVec / 2);
            yp[kWarp] = scale_rope_vec(xv[1], scale, cos, sin, pair_base);
        } else {
            yp[kWarp] = scale_vec(xv[1], scale);
        }
    }
}

void check_bf16_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.scalar_type() == at::kBFloat16,
                name, " must be contiguous CUDA bfloat16");
}

}  // namespace

// Allocation-free graph-friendly entry point.
void rmsnorm_rope_out(const torch::Tensor& x,
                      const torch::Tensor& positions,
                      const torch::Tensor& cos_table,
                      const torch::Tensor& sin_table,
                      torch::Tensor out, int64_t input_heads,
                      double eps, bool pdl, int64_t max_blocks) {
    check_bf16_cuda_contiguous(x, "x");
    check_bf16_cuda_contiguous(out, "out");
    TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
                    positions.scalar_type() == at::kLong && positions.dim() == 1,
                "positions must be contiguous CUDA int64 [output_batch]");
    TORCH_CHECK(cos_table.is_cuda() && sin_table.is_cuda() &&
                    cos_table.is_contiguous() && sin_table.is_contiguous() &&
                    cos_table.scalar_type() == at::kFloat &&
                    sin_table.scalar_type() == at::kFloat &&
                    cos_table.sizes() == sin_table.sizes() &&
                    cos_table.dim() == 2 && cos_table.size(1) == kRope / 2,
                "cos_table and sin_table must be contiguous CUDA float32 [max_pos,32]");
    TORCH_CHECK(input_heads > 0 && x.numel() % (input_heads * kHidden) == 0,
                "x must contain complete [batch,input_heads,512] rows");
    const int64_t output_batch = positions.numel();
    TORCH_CHECK(output_batch > 0 && out.numel() % (output_batch * kHidden) == 0,
                "out must contain complete [output_batch,output_heads,512] rows");
    const int64_t input_batch = x.numel() / (input_heads * kHidden);
    const int64_t output_heads = out.numel() / (output_batch * kHidden);
    TORCH_CHECK(output_batch <= input_batch && output_heads > 0 &&
                    output_heads % input_heads == 0,
                "invalid input/output batch or head geometry");
    TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
    int64_t output_rows = output_batch * output_heads;
    const int grid = static_cast<int>(std::min<int64_t>(
        max_blocks, (output_rows + kWarpsPerBlock - 1) / kWarpsPerBlock));

    const auto* xp = reinterpret_cast<const bf16*>(x.data_ptr());
    const auto* posp = positions.data_ptr<int64_t>();
    const auto* cosp = cos_table.data_ptr<float>();
    const auto* sinp = sin_table.data_ptr<float>();
    auto* yp = reinterpret_cast<bf16*>(out.data_ptr());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(xp) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(yp) % 16 == 0,
                "x and out must be 16-byte aligned");

    const auto stream = at::cuda::getCurrentCUDAStream();
    if (pdl) {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(grid, 1, 1);
        config.blockDim = dim3(kWarpsPerBlock * kWarp, 1, 1);
        config.stream = stream;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        const cudaError_t err = cudaLaunchKernelEx(
            &config, rmsnorm_rope_kernel<true>, xp, posp, cosp, sinp, yp,
            static_cast<int>(input_heads), static_cast<int>(output_heads),
            output_rows, static_cast<float>(eps));
        TORCH_CHECK(err == cudaSuccess, "rmsnorm_rope PDL launch failed: ",
                    cudaGetErrorString(err));
    } else {
        rmsnorm_rope_kernel<false>
            <<<grid, kWarpsPerBlock * kWarp, 0, stream>>>(
                xp, posp, cosp, sinp, yp, static_cast<int>(input_heads),
                static_cast<int>(output_heads), output_rows,
                static_cast<float>(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_rope_out", &rmsnorm_rope_out,
          py::arg("x"), py::arg("positions"), py::arg("cos_table"),
          py::arg("sin_table"), py::arg("out"), py::arg("input_heads"),
          py::arg("eps") = 1e-6, py::arg("pdl") = true,
          py::arg("max_blocks") = 1024,
          "Allocation-free query RMSNorm+RoPE with optional PDL");
}
