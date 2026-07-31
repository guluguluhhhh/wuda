// PyTorch binding for the independent HCA swap-AB front projection.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "front_mixed_gemm_hca_swapab.cuh"

namespace front_mixed_hca_swapab {

torch::Tensor run_op(
    torch::Tensor x, torch::Tensor x_fp8, torch::Tensor x_sf,
    torch::Tensor w_bf16, torch::Tensor w_fp8, torch::Tensor w_sf,
    c10::optional<torch::Tensor> out_opt, int batch_n_override,
    c10::optional<torch::Tensor> task_times_opt,
    c10::optional<torch::Tensor> hc_mix_opt,
    c10::optional<torch::Tensor> hc_base_opt,
    c10::optional<torch::Tensor> hc_scale_opt,
    c10::optional<torch::Tensor> hc_post_opt,
    c10::optional<torch::Tensor> hc_comb_opt,
    bool enable_tail, double hc_eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
              x.scalar_type() == torch::kBFloat16 && x.dim() == 2 &&
              x.size(1) == K, "x must be contiguous bf16 [M,", K, "]");
  const int m = static_cast<int>(x.size(0));
  TORCH_CHECK(m >= 1 && m <= 256, "swapAB supports M in [1,256]");
  TORCH_CHECK(x_fp8.is_cuda() && x_fp8.is_contiguous() &&
              x_fp8.scalar_type() == torch::kFloat8_e4m3fn &&
              x_fp8.sizes() == torch::IntArrayRef({m, K}),
              "x_fp8 must be contiguous e4m3 [M,", K, "]");
  TORCH_CHECK(x_sf.is_cuda() && x_sf.is_contiguous() &&
              (x_sf.scalar_type() == torch::kUInt8 ||
               x_sf.scalar_type() == torch::kFloat8_e8m0fnu) &&
              x_sf.numel() == static_cast<int64_t>(m) * FP8_SCALE_TILES,
              "x_sf must be u8/ue8m0 [M,", FP8_SCALE_TILES, "]");
  TORCH_CHECK(w_bf16.is_cuda() && w_bf16.is_contiguous() &&
              w_bf16.scalar_type() == torch::kBFloat16 &&
              w_bf16.sizes() == torch::IntArrayRef({N - N_FP8, K}),
              "w_bf16 must be bf16 [", N - N_FP8, ",", K, "]");
  TORCH_CHECK(w_fp8.is_cuda() && w_fp8.is_contiguous() &&
              w_fp8.scalar_type() == torch::kFloat8_e4m3fn &&
              w_fp8.sizes() == torch::IntArrayRef({N_FP8, K}),
              "w_fp8 must be e4m3 [2048,7168]");
  TORCH_CHECK(w_sf.is_cuda() && w_sf.is_contiguous() &&
              (w_sf.scalar_type() == torch::kUInt8 ||
               w_sf.scalar_type() == torch::kFloat8_e8m0fnu) &&
              w_sf.numel() == static_cast<int64_t>(N_FP8 / 128) * FP8_SCALE_TILES,
              "w_sf must be u8/ue8m0 [16,56]");

  torch::Tensor out;
  if (out_opt.has_value() && out_opt->numel() != 0) {
    out = *out_opt;
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == torch::kBFloat16 &&
                out.sizes() == torch::IntArrayRef({m, N}),
                "out must be contiguous bf16 [M,", N, "]");
  } else {
    out = torch::empty({m, N}, x.options().dtype(torch::kBFloat16));
  }

  // Batch splits into AT MOST TWO tiles: M<=16 -> one BN16 tile; otherwise
  // two tiles of the smallest ladder step covering ceil(M/2) -- N-side
  // parallelism (29 feature tiles) carries the machine.
  int batch_n = batch_n_override;
  if (batch_n == 0) {
    batch_n = m <= 16 ? 16
            : m <= 32 ? 16
            : m <= 64 ? 32
            : m <= 128 ? 64 : 128;
  }
  TORCH_CHECK(batch_n == 16 || batch_n == 32 || batch_n == 64 ||
              batch_n == 128,
              "batch_n_override must be 0, 16, 32, 64, or 128");

  uint64_t* task_times = nullptr;
  if (task_times_opt.has_value() && task_times_opt->numel() != 0) {
    auto& times = *task_times_opt;
    const int batch_tiles = (m + batch_n - 1) / batch_n;
    const int num_tasks = batch_tiles * TASKS_PER_BATCH;
    TORCH_CHECK(times.is_cuda() && times.is_contiguous() &&
                times.scalar_type() == torch::kInt64 && times.dim() == 2 &&
                times.size(0) == num_tasks && times.size(1) == TIMING_FIELDS,
                "task_times must be contiguous CUDA int64 [", num_tasks,
                ",", TIMING_FIELDS, "]");
    task_times = reinterpret_cast<uint64_t*>(times.data_ptr<int64_t>());
  }

  base::HcTailArgs hc{};
  if (enable_tail) {
    TORCH_CHECK(hc_mix_opt.has_value() && hc_mix_opt->numel() > 0,
                "enable_tail=True requires hc_mix");
    TORCH_CHECK(hc_base_opt.has_value() && hc_scale_opt.has_value() &&
                hc_post_opt.has_value() && hc_comb_opt.has_value(),
                "hc_mix requires hc_base/hc_scale/hc_post/hc_comb");
    auto f32 = [](const torch::Tensor& tensor, int64_t numel,
                  const char* name) {
      TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kFloat32 &&
                  tensor.numel() == numel,
                  name, " must be contiguous CUDA fp32 with ", numel,
                  " elements");
    };
    const auto& hc_mix = *hc_mix_opt;
    TORCH_CHECK(hc_mix.dim() == 2 && hc_mix.size(1) == base::hc_tail::N_OUT,
                "hc_mix must be fp32 [tail_m,24]");
    const int64_t tail_m = hc_mix.size(0);
    TORCH_CHECK(tail_m >= 1 && tail_m <= m,
                "tail_m must be in [1, physical M]");
    f32(hc_mix, tail_m * base::hc_tail::N_OUT, "hc_mix");
    f32(*hc_base_opt, base::hc_tail::N_OUT, "hc_base");
    f32(*hc_scale_opt, 3, "hc_scale");
    f32(*hc_post_opt, tail_m * base::hc_tail::HC, "hc_post");
    f32(*hc_comb_opt, tail_m * base::hc_tail::HC * base::hc_tail::HC,
        "hc_comb");
    hc.mix = hc_mix.data_ptr<float>();
    hc.base = hc_base_opt->data_ptr<float>();
    hc.scale = hc_scale_opt->data_ptr<float>();
    hc.post_out = hc_post_opt->data_ptr<float>();
    hc.comb_out = hc_comb_opt->data_ptr<float>();
    hc.hc_eps = static_cast<float>(hc_eps);
    hc.m = static_cast<int>(tail_m);
  }

  static thread_local CUtensorMap desc_x16{}, desc_w16{}, desc_x8{}, desc_w8{};
  static thread_local const void* cached_x16 = nullptr;
  static thread_local const void* cached_w16 = nullptr;
  static thread_local const void* cached_x8 = nullptr;
  static thread_local const void* cached_w8 = nullptr;
  static thread_local int cached_m = 0, cached_batch_n = 0;
  auto encode = [](CUresult result, const char* name) {
    TORCH_CHECK(result == CUDA_SUCCESS, name, " tensor map failed: ",
                static_cast<int>(result));
  };
  const int fp8_cta_batch = batch_n / CLUSTER_SIZE;
  const int bf16_cta_batch = batch_n / CLUSTER_SIZE;
  if (cached_x16 != x.data_ptr() || cached_x8 != x_fp8.data_ptr() ||
      cached_m != m ||
      cached_batch_n != batch_n) {
    encode(base::make_bf16_map(&desc_x16, x.data_ptr(), m, bf16_cta_batch),
           "x");
    encode(base::make_fp8_map(&desc_x8, x_fp8.data_ptr(), m, fp8_cta_batch),
           "x_fp8");
    cached_x16 = x.data_ptr();
    cached_x8 = x_fp8.data_ptr();
    cached_m = m;
    cached_batch_n = batch_n;
  }
  if (cached_w16 != w_bf16.data_ptr()) {
    encode(base::make_bf16_map(&desc_w16, w_bf16.data_ptr(), N - N_FP8,
                               BF16_CTA_N), "w_bf16");
    cached_w16 = w_bf16.data_ptr();
  }
  if (cached_w8 != w_fp8.data_ptr()) {
    encode(base::make_fp8_map(&desc_w8, w_fp8.data_ptr(), N_FP8, FP8_CTA_N),
           "w_fp8");
    cached_w8 = w_fp8.data_ptr();
  }
  static bool configured = false;
  if (!configured) {
    const cudaError_t s16 = configure_kernel<16, false>();
    const cudaError_t s32 = configure_kernel<32, false>();
    const cudaError_t s64 = configure_kernel<64, false>();
    const cudaError_t s128 = configure_kernel<128, false>();
    const cudaError_t status = s16 != cudaSuccess ? s16 :
                               (s32 != cudaSuccess ? s32 :
                               (s64 != cudaSuccess ? s64 : s128));
    TORCH_CHECK(status == cudaSuccess, "swapAB cudaFuncSetAttribute failed: ",
                cudaGetErrorString(status));
    configured = true;
  }
  static bool timing_configured = false;
  if (task_times != nullptr && !timing_configured) {
    const cudaError_t s16 = configure_kernel<16, true>();
    const cudaError_t s32 = configure_kernel<32, true>();
    const cudaError_t s64 = configure_kernel<64, true>();
    const cudaError_t s128 = configure_kernel<128, true>();
    const cudaError_t status = s16 != cudaSuccess ? s16 :
                               (s32 != cudaSuccess ? s32 :
                               (s64 != cudaSuccess ? s64 : s128));
    TORCH_CHECK(status == cudaSuccess,
                "swapAB timing cudaFuncSetAttribute failed: ",
                cudaGetErrorString(status));
    timing_configured = true;
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xsf = reinterpret_cast<const uint8_t*>(x_sf.data_ptr());
  const auto* wsf = reinterpret_cast<const uint8_t*>(w_sf.data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaError_t status = cudaSuccess;
  if (batch_n == 16) {
    status = task_times != nullptr
        ? launch<16, true>(desc_x16, desc_w16, desc_x8, desc_w8,
                           xsf, wsf, out_ptr, hc, m, task_times, stream)
        : launch<16, false>(desc_x16, desc_w16, desc_x8, desc_w8,
                            xsf, wsf, out_ptr, hc, m, nullptr, stream);
  } else if (batch_n == 32) {
    status = task_times != nullptr
        ? launch<32, true>(desc_x16, desc_w16, desc_x8, desc_w8,
                           xsf, wsf, out_ptr, hc, m, task_times, stream)
        : launch<32, false>(desc_x16, desc_w16, desc_x8, desc_w8,
                            xsf, wsf, out_ptr, hc, m, nullptr, stream);
  } else if (batch_n == 64) {
    status = task_times != nullptr
        ? launch<64, true>(desc_x16, desc_w16, desc_x8, desc_w8,
                           xsf, wsf, out_ptr, hc, m, task_times, stream)
        : launch<64, false>(desc_x16, desc_w16, desc_x8, desc_w8,
                            xsf, wsf, out_ptr, hc, m, nullptr, stream);
  } else {
    status = task_times != nullptr
        ? launch<128, true>(desc_x16, desc_w16, desc_x8, desc_w8,
                            xsf, wsf, out_ptr, hc, m, task_times, stream)
        : launch<128, false>(desc_x16, desc_w16, desc_x8, desc_w8,
                             xsf, wsf, out_ptr, hc, m, nullptr, stream);
  }
  TORCH_CHECK(status == cudaSuccess, "swapAB launch failed: ",
              cudaGetErrorString(status));
  return out;
}

}  // namespace front_mixed_hca_swapab

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "front_mixed_gemm_hca_swapab", &front_mixed_hca_swapab::run_op,
      "HCA mixed front projection, swap-AB, BF16 output",
      py::arg("x"), py::arg("x_fp8"), py::arg("x_sf"),
      py::arg("w_bf16"), py::arg("w_fp8"), py::arg("w_sf"),
      py::arg("out") = py::none(), py::arg("batch_n_override") = 0,
      py::arg("task_times") = py::none(),
      py::arg("hc_mix") = py::none(), py::arg("hc_base") = py::none(),
      py::arg("hc_scale") = py::none(), py::arg("hc_post") = py::none(),
      py::arg("hc_comb") = py::none(), py::arg("enable_tail") = false,
      py::arg("hc_eps") = 1e-6);
}
