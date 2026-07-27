// ============================================================
// front_mixed_gemm.cu — host launcher + PyTorch binding for the
// MIXED-PRECISION front projection GEMM (see include/front_mixed_gemm.cuh).
//
//   y[M,4672] = x[M,7168] @ [ w_fp8[2048,7168] (E4M3 + UE8M0 scales)
//                           | w_bf16[2624,7168] ]^T          -> BF16
//
// Reordered output layout (all segments 64-col tile aligned):
//   [0   ,1536)  wq_a       FP8   [1536,2048)  wkv        FP8
//   [2048,4096)  main_comp  BF16  [4096,4608)  idx_comp   BF16
//   [4608,4672)  w_proj     BF16
// Scales: w_sf [16,56] u8 (128x128 UE8M0), x_sf [M,56] u8 (1x128 UE8M0).
// First stage supports M in [1,128] (decode swap window; M>128 and the M<16
// split-K variant follow once the mixed path is validated).
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "front_mixed_gemm.cuh"

namespace front_mixed {

torch::Tensor front_mixed_gemm_op(
    torch::Tensor x, torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w_bf16, torch::Tensor w_fp8, torch::Tensor w_sf,
    c10::optional<torch::Tensor> out_opt,
    c10::optional<torch::Tensor> q_ssq_opt,
    c10::optional<torch::Tensor> kv_ssq_opt,
    c10::optional<torch::Tensor> hc_mix_opt,
    c10::optional<torch::Tensor> hc_base_opt,
    c10::optional<torch::Tensor> hc_scale_opt,
    c10::optional<torch::Tensor> hc_post_opt,
    c10::optional<torch::Tensor> hc_comb_opt,
    double hc_eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
              x.scalar_type() == torch::kBFloat16 &&
              x.dim() == 2 && x.size(1) == K, "x must be bf16 [M,", K, "]");
  const int m = static_cast<int>(x.size(0));
  TORCH_CHECK(m >= 1 && m <= 128, "front_mixed stage-1 supports M in [1,128]");
  TORCH_CHECK(x_fp8.is_cuda() && x_fp8.is_contiguous() &&
              x_fp8.scalar_type() == torch::kFloat8_e4m3fn &&
              x_fp8.sizes() == torch::IntArrayRef({m, K}),
              "x_fp8 must be e4m3 [M,", K, "]");
  TORCH_CHECK(x_sf.is_cuda() && x_sf.is_contiguous() &&
              (x_sf.scalar_type() == torch::kUInt8 ||
               x_sf.scalar_type() == torch::kFloat8_e8m0fnu) &&
              x_sf.numel() == (int64_t)m * NUM_K_TILES,
              "x_sf must be u8/ue8m0 [M,", NUM_K_TILES, "]");
  TORCH_CHECK(w_bf16.is_cuda() && w_bf16.is_contiguous() &&
              w_bf16.scalar_type() == torch::kBFloat16 &&
              w_bf16.sizes() == torch::IntArrayRef({N - N_FP8, K}),
              "w_bf16 must be bf16 [", N - N_FP8, ",", K, "]");
  TORCH_CHECK(w_fp8.is_cuda() && w_fp8.is_contiguous() &&
              w_fp8.scalar_type() == torch::kFloat8_e4m3fn &&
              w_fp8.sizes() == torch::IntArrayRef({N_FP8, K}),
              "w_fp8 must be e4m3 [", N_FP8, ",", K, "]");
  TORCH_CHECK(w_sf.is_cuda() && w_sf.is_contiguous() &&
              (w_sf.scalar_type() == torch::kUInt8 ||
               w_sf.scalar_type() == torch::kFloat8_e8m0fnu) &&
              w_sf.numel() == (int64_t)(N_FP8 / 128) * NUM_K_TILES,
              "w_sf must be u8/ue8m0 [", N_FP8 / 128, ",", NUM_K_TILES, "]");

  torch::Tensor output;
  if (out_opt.has_value() && out_opt->numel() > 0) {
    output = *out_opt;
    TORCH_CHECK(output.is_cuda() && output.is_contiguous() &&
                output.scalar_type() == torch::kBFloat16 &&
                output.size(0) == m && output.size(1) == N);
  } else {
    output = torch::empty({m, N}, x.options().dtype(torch::kBFloat16));
  }

  // Tensor maps cached per input pointer (matches the fused-caller contract:
  // encoding stays off the timed path).
  static thread_local CUtensorMap desc_a{}, desc_b{}, desc_a8{}, desc_b8{},
      desc_d{};
  static thread_local const void* cached_a = nullptr;
  static thread_local const void* cached_b = nullptr;
  static thread_local const void* cached_a8 = nullptr;
  static thread_local const void* cached_b8 = nullptr;
  static thread_local const void* cached_d = nullptr;
  static thread_local int cached_m = 0, cached_m8 = 0, cached_d_m = 0;
  auto encode = [](CUresult r, const char* what) {
    TORCH_CHECK(r == CUDA_SUCCESS, what, " tensor map failed: ",
                static_cast<int>(r));
  };
  if (cached_a != x.data_ptr() || cached_m != m) {
    encode(make_bf16_map(&desc_a, x.data_ptr(), m, CTA_M), "x");
    cached_a = x.data_ptr();
    cached_m = m;
  }
  if (cached_b != w_bf16.data_ptr()) {
    encode(make_bf16_map(&desc_b, w_bf16.data_ptr(), N - N_FP8, CTA_N_BF16),
           "w_bf16");
    cached_b = w_bf16.data_ptr();
  }
  if (cached_a8 != x_fp8.data_ptr() || cached_m8 != m) {
    encode(make_fp8_map(&desc_a8, x_fp8.data_ptr(), m, CTA_M), "x_fp8");
    cached_a8 = x_fp8.data_ptr();
    cached_m8 = m;
  }
  if (cached_b8 != w_fp8.data_ptr()) {
    encode(make_fp8_map(&desc_b8, w_fp8.data_ptr(), N_FP8, CTA_N_FP8), "w_fp8");
    cached_b8 = w_fp8.data_ptr();
  }
  if (cached_d != output.data_ptr() || cached_d_m != m) {
    encode(make_bf16_output_map(&desc_d, output.data_ptr(), m), "y");
    cached_d = output.data_ptr();
    cached_d_m = m;
  }

  static bool configured = false;
  if (!configured) {
    cudaError_t status = configure_front_mixed_kernel();
    TORCH_CHECK(status == cudaSuccess, "cudaFuncSetAttribute failed: ",
                cudaGetErrorString(status));
    configured = true;
  }

  // Optional RMSNorm ssq folds (q_norm 1536-col / kv_norm 512-col segments):
  // fp32 [M], RED-accumulated -- caller zero-initializes (head_ssq contract).
  auto ssq_ptr = [&](c10::optional<torch::Tensor>& t,
                     const char* what) -> float* {
    if (!t.has_value() || t->numel() == 0) return nullptr;
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() &&
                t->scalar_type() == torch::kFloat32 && t->numel() >= m,
                what, " must be contiguous fp32 [M]");
    return t->data_ptr<float>();
  };
  float* q_ssq = ssq_ptr(q_ssq_opt, "q_ssq");
  float* kv_ssq = ssq_ptr(kv_ssq_opt, "kv_ssq");

  // Optional [TC/CC] MHC post+comb tail bundle: all five tensors together.
  HcTailArgs hc{};
  if (hc_mix_opt.has_value() && hc_mix_opt->numel() > 0) {
    TORCH_CHECK(hc_base_opt.has_value() && hc_scale_opt.has_value() &&
                hc_post_opt.has_value() && hc_comb_opt.has_value(),
                "hc_mix requires hc_base/hc_scale/hc_post/hc_comb");
    auto f32 = [](const torch::Tensor& t, int64_t n, const char* what) {
      TORCH_CHECK(t.is_cuda() && t.is_contiguous() &&
                  t.scalar_type() == torch::kFloat32 && t.numel() >= n,
                  what, " must be contiguous fp32 with >= ", n, " elems");
    };
    const int64_t hm = hc_mix_opt->size(0);
    f32(*hc_mix_opt, hm * 24, "hc_mix");
    f32(*hc_base_opt, 24, "hc_base");
    f32(*hc_scale_opt, 3, "hc_scale");
    f32(*hc_post_opt, hm * 4, "hc_post");
    f32(*hc_comb_opt, hm * 16, "hc_comb");
    hc.mix = hc_mix_opt->data_ptr<float>();
    hc.base = hc_base_opt->data_ptr<float>();
    hc.scale = hc_scale_opt->data_ptr<float>();
    hc.post_out = hc_post_opt->data_ptr<float>();
    hc.comb_out = hc_comb_opt->data_ptr<float>();
    hc.hc_eps = static_cast<float>(hc_eps);
    hc.m = static_cast<int>(hm);
  }

  cudaError_t status = launch_front_mixed(
      desc_a, desc_b, desc_a8, desc_b8, desc_d,
      reinterpret_cast<const uint8_t*>(x_sf.data_ptr()),
      reinterpret_cast<const uint8_t*>(w_sf.data_ptr()),
      q_ssq, kv_ssq, hc,
      m, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cudaSuccess, "front_mixed launch failed: ",
              cudaGetErrorString(status));
  return output;
}

}  // namespace front_mixed

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("front_mixed_gemm", &front_mixed::front_mixed_gemm_op,
             "Mixed-precision front projection GEMM (fp8 cols [0,2048) + bf16 "
             "cols [2048,4672), reordered layout) -> bf16 [M,4672]",
             py::arg("x"), py::arg("x_fp8"), py::arg("x_sf"),
             py::arg("w_bf16"), py::arg("w_fp8"), py::arg("w_sf"),
             py::arg("out") = c10::nullopt,
             py::arg("q_ssq") = c10::nullopt,
             py::arg("kv_ssq") = c10::nullopt,
             py::arg("hc_mix") = c10::nullopt,
             py::arg("hc_base") = c10::nullopt,
             py::arg("hc_scale") = c10::nullopt,
             py::arg("hc_post") = c10::nullopt,
             py::arg("hc_comb") = c10::nullopt,
             py::arg("hc_eps") = 1e-6);
}
