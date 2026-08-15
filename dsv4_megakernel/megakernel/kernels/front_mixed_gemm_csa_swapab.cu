// PyTorch binding for the independent CSA swap-AB front projection.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "front_mixed_gemm_csa_swapab.cuh"

namespace front_mixed_csa_swapab {

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
    bool enable_tail, double hc_eps,
    c10::optional<torch::Tensor> main_state_opt,
    c10::optional<torch::Tensor> main_ape_opt,
    c10::optional<torch::Tensor> main_state_row_opt,
    c10::optional<torch::Tensor> ape_phase_opt,
    c10::optional<torch::Tensor> idx_state_opt,
    c10::optional<torch::Tensor> idx_state_row_opt,
    c10::optional<torch::Tensor> idx_ape_opt,
    c10::optional<torch::Tensor> win_y2_opt,
    c10::optional<torch::Tensor> w64_opt,
    bool pdl) {
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

  // One batch tile launches only 37 two-CTA clusters, occupying half of B300's
  // 148 SMs. Above M=48, use exactly two tiles so both SM waves can overlap.
  // Use BN128 for M>192 to retain its deeper FP8 pipeline.
  int batch_n = batch_n_override;
  if (batch_n == 0) {
    batch_n = m <= 48 ? ((m + 15) / 16) * 16
            : m <= 192 ? ((m + 31) / 32) * 16
            : 128;
  }
  TORCH_CHECK(batch_n == 16 || batch_n == 32 || batch_n == 48 ||
              batch_n == 64 || batch_n == 80 || batch_n == 96 ||
              batch_n == 128,
              "batch_n_override must be 0, 16, 32, 48, 64, 80, 96, or 128");

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

  base::FrontEmitArgs emit{};
  if (main_state_opt.has_value() && main_state_opt->numel() > 0) {
    TORCH_CHECK(main_ape_opt && main_state_row_opt && ape_phase_opt &&
                idx_state_opt && idx_state_row_opt && idx_ape_opt &&
                win_y2_opt && w64_opt,
                "front-emit tensors must be given together");
    auto f32 = [](const torch::Tensor& tensor, int64_t min_numel,
                  const char* name) {
      TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kFloat32 &&
                  tensor.numel() >= min_numel,
                  name, " must be contiguous CUDA fp32 with >= ", min_numel,
                  " elements");
    };
    auto i32 = [m](const torch::Tensor& tensor, const char* name) {
      TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                  tensor.scalar_type() == torch::kInt32 && tensor.numel() >= m,
                  name, " must be contiguous CUDA int32 with >= M elements");
    };
    f32(*main_state_opt, 2048, "main_state");
    TORCH_CHECK(main_state_opt->size(-1) == 2048,
                "main_state last dimension must be 2048");
    f32(*main_ape_opt, base::APE_RATIO * 1024, "main_ape");
    i32(*main_state_row_opt, "main_state_row");
    i32(*ape_phase_opt, "ape_phase");
    f32(*idx_state_opt, 512, "idx_state");
    TORCH_CHECK(idx_state_opt->size(-1) == 512,
                "idx_state last dimension must be 512");
    i32(*idx_state_row_opt, "idx_state_row");
    f32(*idx_ape_opt, base::APE_RATIO * 256, "idx_ape");
    f32(*win_y2_opt, static_cast<int64_t>(m) * 512, "win_y2");
    f32(*w64_opt, static_cast<int64_t>(m) * 64, "w64");
    emit.main_state = main_state_opt->data_ptr<float>();
    emit.main_ape = main_ape_opt->data_ptr<float>();
    emit.main_state_row = main_state_row_opt->data_ptr<int>();
    emit.ape_phase = ape_phase_opt->data_ptr<int>();
    emit.idx_state = idx_state_opt->data_ptr<float>();
    emit.idx_state_row = idx_state_row_opt->data_ptr<int>();
    emit.idx_ape = idx_ape_opt->data_ptr<float>();
    emit.win_y2 = win_y2_opt->data_ptr<float>();
    emit.w64 = w64_opt->data_ptr<float>();
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
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xsf = reinterpret_cast<const uint8_t*>(x_sf.data_ptr());
  const auto* wsf = reinterpret_cast<const uint8_t*>(w_sf.data_ptr());
  auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaError_t status = cudaSuccess;
#define DISPATCH_BATCH_N(BatchN)                                             \
  case BatchN: {                                                            \
    if (task_times != nullptr) {                                            \
      static bool timing_configured = false;                                \
      if (!timing_configured) {                                             \
        status = configure_kernel<BatchN, true>();                           \
        if (status != cudaSuccess) break;                                   \
        timing_configured = true;                                           \
      }                                                                     \
      status = launch<BatchN, true>(                                        \
          desc_x16, desc_w16, desc_x8, desc_w8, xsf, wsf, out_ptr, hc, emit,\
          m, task_times, stream, pdl);                                       \
    } else {                                                                \
      static bool configured = false;                                       \
      if (!configured) {                                                    \
        status = configure_kernel<BatchN, false>();                          \
        if (status != cudaSuccess) break;                                   \
        configured = true;                                                  \
      }                                                                     \
      status = launch<BatchN, false>(                                       \
          desc_x16, desc_w16, desc_x8, desc_w8, xsf, wsf, out_ptr, hc, emit,\
          m, nullptr, stream, pdl);                                          \
    }                                                                       \
    break;                                                                  \
  }
  switch (batch_n) {
    DISPATCH_BATCH_N(16)
    DISPATCH_BATCH_N(32)
    DISPATCH_BATCH_N(48)
    DISPATCH_BATCH_N(64)
    DISPATCH_BATCH_N(80)
    DISPATCH_BATCH_N(96)
    DISPATCH_BATCH_N(128)
    default:
      TORCH_CHECK(false, "unreachable BatchN dispatch: ", batch_n);
  }
#undef DISPATCH_BATCH_N
  TORCH_CHECK(status == cudaSuccess, "swapAB launch failed: ",
              cudaGetErrorString(status));
  return out;
}

}  // namespace front_mixed_csa_swapab

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "front_mixed_gemm_csa_swapab", &front_mixed_csa_swapab::run_op,
      "CSA mixed front projection, swap-AB, BF16 output",
      py::arg("x"), py::arg("x_fp8"), py::arg("x_sf"),
      py::arg("w_bf16"), py::arg("w_fp8"), py::arg("w_sf"),
      py::arg("out") = py::none(), py::arg("batch_n_override") = 0,
      py::arg("task_times") = py::none(),
      py::arg("hc_mix") = py::none(), py::arg("hc_base") = py::none(),
      py::arg("hc_scale") = py::none(), py::arg("hc_post") = py::none(),
      py::arg("hc_comb") = py::none(), py::arg("enable_tail") = false,
      py::arg("hc_eps") = 1e-6,
      py::arg("main_state") = py::none(), py::arg("main_ape") = py::none(),
      py::arg("main_state_row") = py::none(), py::arg("ape_phase") = py::none(),
      py::arg("idx_state") = py::none(), py::arg("idx_state_row") = py::none(),
      py::arg("idx_ape") = py::none(), py::arg("win_y2") = py::none(),
      py::arg("w64") = py::none(), py::arg("pdl") = false);
}
