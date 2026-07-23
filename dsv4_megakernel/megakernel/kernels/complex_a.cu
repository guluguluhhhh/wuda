// ============================================================
// complex_a.cu — PyTorch binding for include/complex_a.cuh (fusenorm_* API).
//
// complex_gemm STEP 1+2 (decode regime): y = x @ w.T (bf16 swap-AB 1SM UMMA,
// split-K ws-always, persistent) followed by the fused split-K reduce +
// per-token RMSNorm over the LEADING FUSENORM_NORM_DIM (=1536, clamped to N)
// columns (y1):
//     out[m, :norm_len] = (Σ_ks ws) * rsqrt(mean(row[:norm_len]^2) + eps) * rms_w
// Columns [norm_len, N) are NOT written — their split-K partials stay in ws
// for step-3 (op B) to reduce. The test asserts that tail stays untouched.
//
// Handle-based API (repo *_out convention: the expensive part — split-K ws
// alloc + TMA descriptor encode — happens ONCE in setup; run is launch-only):
//     h = complex_a_setup(x, w[, force_bn, force_bm, force_ks, force_km])
//     complex_a_run(h, out, rms_w, eps[, hc_*...])
//     complex_a_info(h) -> (mma_m, mma_n, ks, total_tiles, km)
//     complex_a_free(h)
// The handle OWNS references to x/w (their data pointers are captured inside
// the launch closure's TMA descriptors), so lifetime is safe by construction.
// force_bn == 0 -> adaptive tile (choose_tile_config); > 0 -> forced sweep tile
// (BN mult of 16 <= 128, BM in {64,128}, KM in {1,2} — the instantiated set).
//
// Optional [TC/CC DUAL-PATH] HC tail (all fp32): passing hc_mix/hc_base/
// hc_scale/hc_post/hc_comb activates the extra CUDA-core warp that computes
// hc_fused_kernel_tc's POST gate + Sinkhorn COMB concurrently with the GEMM
// warps (hidden under this op's weight-read window; post/comb are consumed
// only by hc_post AFTER the attention block, so they are off the critical path).
// hc_mix [m,24] = the ALREADY-REDUCED + rms-folded mix (split-K reduce + Σx²
// done upstream by the hc epilogue); outputs hc_post [m,4] / hc_comb [m,4,4].
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <tuple>

#include "complex_a.cuh"

namespace {

struct ComplexAHandle {
    FuseNormCtx ctx;
    torch::Tensor x, w;   // keep operands alive: TMA descriptors point at them
};

ComplexAHandle* to_handle(int64_t h) {
    TORCH_CHECK(h != 0, "invalid complex_a handle (0)");
    return reinterpret_cast<ComplexAHandle*>(static_cast<uintptr_t>(h));
}

}  // namespace

static int64_t complex_a_setup(torch::Tensor x, torch::Tensor w,
                               int64_t force_bn, int64_t force_bm,
                               int64_t force_ks, int64_t force_km) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16,
                "x must be CUDA bf16 contiguous [M,K]");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16,
                "w must be CUDA bf16 contiguous [N,K]");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1),
                "x [M,K] / w [N,K] with matching K, got x=", x.sizes(), " w=", w.sizes());
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128, got ", N);
    TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64, got ", K);
    TORCH_CHECK(M >= 1, "M must be >= 1");

    auto* h = new ComplexAHandle();
    h->x = x;
    h->w = w;
    const auto* a_ptr = reinterpret_cast<const bf16_t*>(x.data_ptr());
    const auto* b_ptr = reinterpret_cast<const bf16_t*>(w.data_ptr());
    if (force_bn > 0) {
        fusenorm_setup_forced(h->ctx, a_ptr, b_ptr, M, N, K, (int)force_bn,
                              force_bm > 0 ? (int)force_bm : 128,
                              (int)force_ks, (int)force_km);
    } else {
        fusenorm_setup(h->ctx, a_ptr, b_ptr, M, N, K);
    }
    if (!h->ctx.valid) {
        fusenorm_free(h->ctx);
        delete h;
        TORCH_CHECK(false, "complex_a setup failed (unsupported shape/tile): M=", M,
                    " N=", N, " K=", K, " force_bn=", force_bn, " force_bm=", force_bm,
                    " force_km=", force_km);
    }
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(h));
}

// Launch-only (TMA descriptors pre-encoded in setup). Two kernels on the current
// stream: gemm_device (split-K partials -> ws; + optional HC post/comb tail warp)
// then gemm_rmsnorm_kernel (PDL-overlapped reduce + RMSNorm -> out[:, :norm_len]).
static void complex_a_run(int64_t handle, torch::Tensor out, torch::Tensor rms_w,
                          double eps,
                          c10::optional<torch::Tensor> hc_mix,
                          c10::optional<torch::Tensor> hc_base,
                          c10::optional<torch::Tensor> hc_scale,
                          c10::optional<torch::Tensor> hc_post,
                          c10::optional<torch::Tensor> hc_comb,
                          double hc_eps) {
    auto* h = to_handle(handle);
    auto& ctx = h->ctx;
    const int norm_len = ctx.N < FUSENORM_NORM_DIM ? ctx.N : FUSENORM_NORM_DIM;
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16,
                "out must be CUDA bf16 contiguous [M,N]");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == ctx.M && out.size(1) == ctx.N,
                "out must be [", ctx.M, ",", ctx.N, "], got ", out.sizes());
    TORCH_CHECK(rms_w.is_cuda() && rms_w.is_contiguous() && rms_w.scalar_type() == torch::kFloat,
                "rms_w must be CUDA fp32 contiguous");
    TORCH_CHECK(rms_w.numel() >= norm_len,
                "rms_w needs >= norm_len=", norm_len, " elements, got ", rms_w.numel());

    // Optional [TC/CC] HC post+comb tail bundle (all fp32; disabled when absent).
    HcTailArgs hc{};
    if (hc_mix.has_value()) {
        TORCH_CHECK(hc_base && hc_scale && hc_post && hc_comb,
                    "hc tail needs ALL of hc_mix/hc_base/hc_scale/hc_post/hc_comb");
        const auto f32 = [](const torch::Tensor& t, const char* n) {
            TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kFloat,
                        n, " must be CUDA fp32 contiguous");
        };
        f32(*hc_mix, "hc_mix"); f32(*hc_base, "hc_base"); f32(*hc_scale, "hc_scale");
        f32(*hc_post, "hc_post"); f32(*hc_comb, "hc_comb");
        TORCH_CHECK(hc_mix->dim() == 2 && hc_mix->size(1) == hc_tail::N_OUT,
                    "hc_mix must be [m,", hc_tail::N_OUT, "] (already-reduced, rms-folded), got ",
                    hc_mix->sizes());
        const int hm = (int)hc_mix->size(0);
        TORCH_CHECK(hc_base->numel() == hc_tail::N_OUT,
                    "hc_base must have ", hc_tail::N_OUT, " elements");
        TORCH_CHECK(hc_scale->numel() == 3, "hc_scale must have 3 elements");
        TORCH_CHECK(hc_post->numel() == (int64_t)hm * hc_tail::HC, "hc_post must be [m,4]");
        TORCH_CHECK(hc_comb->numel() == (int64_t)hm * hc_tail::HC * hc_tail::HC,
                    "hc_comb must be [m,4,4]");
        hc.mix      = hc_mix->data_ptr<float>();
        hc.base     = hc_base->data_ptr<float>();
        hc.scale    = hc_scale->data_ptr<float>();
        hc.hc_eps   = (float)hc_eps;
        hc.m        = hm;
        hc.post_out = hc_post->data_ptr<float>();
        hc.comb_out = hc_comb->data_ptr<float>();
    }

    fusenorm_run(ctx, reinterpret_cast<bf16_t*>(out.data_ptr()),
                 rms_w.data_ptr<float>(), (float)eps,
                 at::cuda::getCurrentCUDAStream(), hc);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "complex_a launch failed: ", cudaGetErrorString(err));
}

// (mma_m, mma_n, ks, total_tiles, km) of the tile config actually launched.
static std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>
complex_a_info(int64_t handle) {
    auto& ctx = to_handle(handle)->ctx;
    return {ctx.mma_m, ctx.mma_n, ctx.ks, ctx.total_tiles, ctx.km};
}

static void complex_a_free(int64_t handle) {
    auto* h = to_handle(handle);
    fusenorm_free(h->ctx);
    delete h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("complex_a_setup", &complex_a_setup,
          "complex_gemm step-1 setup: split-K ws alloc + TMA descriptor encode (ONCE). "
          "Returns a handle owning x/w refs. force_bn=0 -> adaptive tile; "
          "else forced (BN mult-of-16 <=128, BM 64/128, KM 1/2) for sweeps",
          py::arg("x"), py::arg("w"), py::arg("force_bn") = 0, py::arg("force_bm") = 0,
          py::arg("force_ks") = 0, py::arg("force_km") = 0);
    m.def("complex_a_run", &complex_a_run,
          "fused bf16 GEMM (split-K, ws-always) + PDL reduce + per-token RMSNorm over "
          "the leading min(N,1536) columns; out[:, norm_len:] is NOT written. "
          "Optional hc_* bundle activates the [TC/CC] HC post+comb tail warp "
          "(fp32; hc_mix = already-reduced, rms-folded mix [m,24])",
          py::arg("handle"), py::arg("out"), py::arg("rms_w"), py::arg("eps") = 1e-6,
          py::arg("hc_mix") = c10::nullopt, py::arg("hc_base") = c10::nullopt,
          py::arg("hc_scale") = c10::nullopt, py::arg("hc_post") = c10::nullopt,
          py::arg("hc_comb") = c10::nullopt, py::arg("hc_eps") = 1e-6);
    m.def("complex_a_info", &complex_a_info,
          "-> (mma_m, mma_n, ks, total_tiles, km)", py::arg("handle"));
    m.def("complex_a_free", &complex_a_free,
          "release split-K ws + handle", py::arg("handle"));
}
