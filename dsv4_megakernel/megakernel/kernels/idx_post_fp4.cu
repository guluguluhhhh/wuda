// ============================================================
// idx_post_fp4.cu
// Standalone DSV4 indexer-q post-processing — Host + PyTorch Binding
//
// iq_f32[M,64,128] (fp32 indexer projection, e.g. the wq_b merged GEMM's
// drained scratch) ->
//   round bf16 -> RoPE(tail 64, per-token pos) -> Hadamard-128 (* 128^-1/2)
//   -> per-32 MXFP4 quant (scale = 2^ceil(log2(amax/6)), ue8m0)
// -> iq_fp4[M,64,64] i8 (packed, low nibble = even) + iq_sf[M,64] i32
//    (packed-ue8m0) -- EXACTLY the q / sf_q layout the score-attention
//    kernel's TMAs consume.
//
// The device chain lives in include/idx_post_fp4.cuh and is SHARED with the
// fused path inside wq_b_fp8_gemm.cu (async transform warpgroup): identical
// rounding, bit-identical outputs on identical inputs.
//
// ======================== USAGE ========================
//   iq_fp4, iq_sf = module.idx_postprocess(
//       iq_f32,    # [M,64,128] fp32, contiguous CUDA
//       q_pos,     # [M] int32 rotary positions
//       rope_cos,  # [max_pos,32] fp32
//       rope_sin,  # [max_pos,32] fp32
//   )
// Any M >= 1. Grid-parallel across the whole GPU (32 rows / 256-thread CTA);
// memory-bound: reads 32KB + writes 4.25KB per 64 rows.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "idx_post_fp4.cuh"

static std::vector<torch::Tensor> idx_postprocess(
    torch::Tensor iq_f32, torch::Tensor q_pos,
    torch::Tensor rope_cos, torch::Tensor rope_sin)
{
    TORCH_CHECK(iq_f32.is_cuda() && iq_f32.is_contiguous()
                && iq_f32.scalar_type() == torch::kFloat
                && iq_f32.dim() == 3 && iq_f32.size(1) == idx_post::NUM_HEADS
                && iq_f32.size(2) == idx_post::HEAD_DIM,
                "iq_f32 must be contiguous CUDA fp32 [M,64,128]");
    const int M = iq_f32.size(0);
    TORCH_CHECK(M >= 1, "empty batch");
    TORCH_CHECK(q_pos.is_cuda() && q_pos.scalar_type() == torch::kInt32
                && q_pos.numel() == M && q_pos.is_contiguous(), "q_pos [M] i32");
    TORCH_CHECK(rope_cos.is_cuda() && rope_cos.scalar_type() == torch::kFloat
                && rope_cos.dim() == 2 && rope_cos.size(1) == 32
                && rope_cos.is_contiguous(), "rope_cos [max_pos,32] f32");
    TORCH_CHECK(rope_sin.is_cuda() && rope_sin.scalar_type() == torch::kFloat
                && rope_sin.sizes() == rope_cos.sizes()
                && rope_sin.is_contiguous(), "rope_sin [max_pos,32] f32");

    auto iq_fp4 = torch::empty({M, idx_post::NUM_HEADS, idx_post::HEAD_DIM / 2},
                               iq_f32.options().dtype(torch::kInt8));
    auto iq_sf = torch::empty({M, idx_post::NUM_HEADS},
                              iq_f32.options().dtype(torch::kInt32));
    const int rows = M * idx_post::NUM_HEADS;
    auto stream = at::cuda::getCurrentCUDAStream();
    idx_post_kernel<<<(rows + 31) / 32, 256, 0, stream>>>(
        iq_f32.data_ptr<float>(), q_pos.data_ptr<int>(),
        rope_cos.data_ptr<float>(), rope_sin.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(iq_fp4.data_ptr()),
        iq_sf.data_ptr<int>(), rows);
    return {iq_fp4, iq_sf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("idx_postprocess", &idx_postprocess,
          "Indexer-q post-processing: rope+hadamard+MXFP4 over fp32 iq "
          "[M,64,128] -> (iq_fp4 i8 [M,64,64], iq_sf i32 [M,64]) in the "
          "score-attention q/sf_q layout",
          py::arg("iq_f32"), py::arg("q_pos"), py::arg("rope_cos"), py::arg("rope_sin"));
}
