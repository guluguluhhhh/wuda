"""Correctness and B300 benchmark for the HYBRID mhc op:

    deep_gemm.tf32_hc_prenorm_gemm   (split-K tf32 GEMM + sum(x^2) partials)
  + hc_reduce_fuse_out               (OUR epilogue: reduce + RMSNorm + gates
                                      + Sinkhorn + collapse [+ attn_norm],
                                      PDL secondary)

The old in-house tcgen05 split-K GEMM path was DELETED (deep_gemm measured
2-3us faster on B300, while our epilogue beats vLLM's TileLang fuse by
3-4us). deep_gemm is a HARD dep here now."""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim


HC = 4
DIM = 7168
K_DIM = HC * DIM
N_OUT = (2 + HC) * HC
HC_EPS = 1e-6
RMS_NORM_EPS = 1e-6
SINKHORN_ITERS = 20
# decode regime only (M<=256): all 32-aligned M plus 1/4/16 edge cases.
PROFILE_M = [1, 4, 16, 32, 64, 96, 128, 160, 192, 224, 256]

DG = None


def get_dg():
    """deep_gemm (HARD dep): shared resolver in bench_utils (env
    DEEP_GEMM_DIR > installed wheel > sibling checkout), PDL enabled once."""
    global DG
    if DG is None:
        from bench_utils import get_deep_gemm
        DG = get_deep_gemm()
        assert hasattr(DG, "tf32_hc_prenorm_gemm"), \
            "deep_gemm build lacks tf32_hc_prenorm_gemm"
    return DG


def hc_n_splits(m):
    """vLLM compute_num_split(block_k=64, k=K_DIM, grid=cdiv(m,64))."""
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(min(n_sms // max((m + 63) // 64, 1), (K_DIM // 64) // 4), 1)


def load_cuda_module():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < 100:
        raise RuntimeError(
            f"deep_gemm tf32_hc_prenorm_gemm requires sm_100+, got sm_{sm}")

    # The epilogue is plain CUDA (no CUTLASS/tcgen05 left -- the in-house
    # GEMM engine was deleted with the deep_gemm hybrid switch).
    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
    ]
    return load(
        name="hc_fused_tc",
        sources=[os.path.join(proj_dir, "kernels", "hc_fused_kernel_tc.cu")],
        extra_include_paths=[os.path.join(proj_dir, "include")],
        extra_cuda_cflags=cuda_flags,
        verbose=True,
    )


def run_hybrid(module, hidden, weight, base, scale, with_post_comb=False,
               norm_w=None, bufs=None, xq=None, xsf=None):
    """The production call pair. bufs (optional) = dict of preallocated
    ws/sq/collapsed/pre/post/comb for allocation-free hot loops. xq/xsf
    (norm variant only) enable the FUSED fp8 activation emission."""
    dg = get_dg()
    hs2 = hidden.contiguous().view(-1, K_DIM)
    m = hs2.size(0)
    if bufs is None:
        S = hc_n_splits(m)
        bufs = dict(
            ws=torch.empty(S, m, N_OUT, device="cuda"),
            sq=torch.empty(S, m, device="cuda"),
            collapsed=torch.empty(m, DIM, device="cuda",
                                  dtype=torch.bfloat16),
            pre=torch.empty(m, HC, device="cuda"),
            post=torch.empty(m, HC, device="cuda"),
            comb=torch.empty(m, HC, HC, device="cuda"))
    dg.tf32_hc_prenorm_gemm(hs2, weight, bufs["ws"], bufs["sq"],
                            bufs["ws"].size(0))
    module.hc_reduce_fuse_out(hidden, bufs["ws"], bufs["sq"], base, scale,
                              HC_EPS, RMS_NORM_EPS, bufs["collapsed"],
                              bufs["pre"], bufs["post"], bufs["comb"],
                              with_post_comb=with_post_comb,
                              attn_norm_w=norm_w,
                              attn_norm_eps=RMS_NORM_EPS,
                              xq_out=xq, xsf_out=xsf)
    return bufs


def hc_reference(hidden_states, weight, base, scale):
    squeeze = hidden_states.dim() == 2
    hs = hidden_states.reshape(-1, HC, DIM)
    x = hs.reshape(-1, K_DIM).float()
    rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    mix = F.linear(x, weight.float()) * rms

    pre_w, post_w, comb_w = torch.split(mix, [HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = torch.split(base, [HC, HC, HC * HC])
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + HC_EPS
    post = 2.0 * torch.sigmoid(post_w * scale[1] + post_b)
    comb_logits = comb_w.view(-1, HC, HC) * scale[2] + comb_b.view(HC, HC)
    comb = torch.softmax(comb_logits, dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)

    collapsed = (pre.unsqueeze(-1) * hs.float()).sum(dim=1).to(torch.bfloat16)
    if squeeze:
        return collapsed[0], pre[0], post[0], comb[0]
    return collapsed, pre, post, comb


def make_inputs(m, weight=None, base=None, scale=None):
    device = "cuda"
    hidden = torch.randn(m, HC, DIM, device=device, dtype=torch.bfloat16)
    if m == 1:
        hidden = hidden[0]
    if weight is None:
        # fp32 weight (deep_gemm's tf32 GEMM reads it as tf32), matching the
        # official DeepSeek-V4 hc_attn_fn / hc_ffn_fn fp32 parameters.
        weight = (
            torch.randn(N_OUT, K_DIM, device=device, dtype=torch.float32) * 0.01
        )
    if base is None:
        base = torch.randn(N_OUT, device=device, dtype=torch.float32) * 0.1
    if scale is None:
        scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    return hidden, weight, base, scale


def error_stats(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-5)
    return diff.max().item(), diff.mean().item(), (diff / denom).mean().item()


def test_correctness(module, positions):
    print("\nCorrectness: deep_gemm prenorm GEMM + our fused epilogue")
    print(
        f"{'M':>6} {'output':>10} {'max abs':>12} {'mean abs':>12} "
        f"{'mean rel':>12} {'result':>8}"
    )
    print("-" * 68)
    all_ok = True
    tolerances = {
        "collapsed": (3e-2, 1e-2),
        "pre": (2e-3, 2e-3),
        "post": (2e-3, 2e-3),
        "comb": (5e-4, 2e-3),
    }

    for m in positions:
        torch.manual_seed(1000 + m)
        hidden, weight, base, scale = make_inputs(m)
        expected = hc_reference(hidden, weight, base, scale)

        # full variant (post/comb + Sinkhorn in-kernel)
        b = run_hybrid(module, hidden, weight, base, scale,
                       with_post_comb=True)
        torch.cuda.synchronize()
        mm = b["pre"].size(0)
        actual = (b["collapsed"], b["pre"], b["post"], b["comb"])
        for name, got, ref in zip(
            ("collapsed", "pre", "post", "comb"), actual, expected
        ):
            atol, rtol = tolerances[name]
            ref2 = ref.reshape(got.shape)
            max_abs, mean_abs, mean_rel = error_stats(got, ref2)
            ok = torch.allclose(got.float(), ref2.float(), atol=atol, rtol=rtol)
            all_ok &= ok
            print(
                f"{m:6d} {name:>10} {max_abs:12.4e} {mean_abs:12.4e} "
                f"{mean_rel:12.4e} {('PASS' if ok else 'FAIL'):>8}"
            )

        # fused attn_norm (lite, the PRODUCTION form): reference = the
        # separate-norm chain applied to the hybrid's OWN raw lite collapsed
        # (same in-block ssq order -> near-bitwise).
        b_lite = run_hybrid(module, hidden, weight, base, scale)
        # gamma BF16 = the checkpoint dtype (model.py: "stored in bf16")
        norm_w = (torch.rand(DIM, device="cuda") + 0.5).bfloat16()
        b_nrm = run_hybrid(module, hidden, weight, base, scale, norm_w=norm_w)
        torch.cuda.synchronize()
        xb = b_lite["collapsed"].float()
        r = torch.rsqrt(xb.pow(2).sum(-1, keepdim=True) / DIM + RMS_NORM_EPS)
        nref = ((xb * r) * norm_w).to(torch.bfloat16)
        bitmatch = (b_nrm["collapsed"].view(torch.int16) ==
                    nref.view(torch.int16)).float().mean().item()
        nrm_ok = torch.allclose(b_nrm["collapsed"].float(), nref.float(),
                                rtol=1e-2, atol=1e-3)
        all_ok &= nrm_ok
        print(
            f"{m:6d} {'fusednorm':>10} "
            f"{(b_nrm['collapsed'].float() - nref.float()).abs().max().item():12.4e} "
            f"{'bit=' + format(bitmatch, '.4f'):>12} {'':>12} "
            f"{('PASS' if nrm_ok else 'FAIL'):>8}"
        )
        # FUSED fp8 emission ablation-correctness: bytes must match the
        # torch golden built with the SAME bit-inspected pow2-ceil exponent
        # (quant source = the emitted bf16 collapsed itself).
        xq = torch.empty(mm, DIM, device="cuda", dtype=torch.uint8)
        xsf = torch.empty(mm, DIM // 128, device="cuda", dtype=torch.uint8)
        b_q8 = run_hybrid(module, hidden, weight, base, scale, norm_w=norm_w,
                          xq=xq, xsf=xsf)
        torch.cuda.synchronize()
        v = b_q8["collapsed"].float().view(mm, DIM // 128, 128)
        t = (v.abs().amax(-1) * (1.0 / 448.0)).clamp_min(1e-4)
        bits = t.view(torch.int32)
        e_ref = ((bits >> 23) & 0xff) - 127 + ((bits & 0x7fffff) != 0).int()
        sf_ok = (xsf == (e_ref + 127).to(torch.uint8)).float().mean().item()
        qr = (v * torch.exp2(-e_ref.float()).unsqueeze(-1)) \
            .to(torch.float8_e4m3fn).view(torch.uint8).view(mm, DIM)
        q_ok = (xq == qr).float().mean().item()
        q8_ok = sf_ok == 1.0 and q_ok > 0.9999
        all_ok &= q8_ok
        print(
            f"{m:6d} {'fusedq8':>10} {'sf=' + format(sf_ok, '.4f'):>12} "
            f"{'q=' + format(q_ok, '.6f'):>12} {'':>12} "
            f"{('PASS' if q8_ok else 'FAIL'):>8}"
        )
        del hidden, expected, b, b_lite, b_nrm, b_q8

    print("-" * 68)
    print("ALL PASSED" if all_ok else "CORRECTNESS FAILED")
    return all_ok


def time_cuda_us(fn, kernel_names):
    """DeepGEMM bench_kineto: 8GB L2 flush before EVERY call, kineto kernel
    device time only, warmup cycle discarded, MEAN over instances; tuple
    names are summed. NOTE with PDL the epilogue's device time includes its
    griddepcontrol wait -> the kernel-sum slightly overstates the wall."""
    t = bench_kineto(fn, kernel_names, suppress_kineto_output=True,
                     with_multiple_kernels=True)
    return 1e6 * (sum(t) if isinstance(t, tuple) else t)


def probe_kernel_names(fn):
    from torch.profiler import profile, ProfilerActivity
    fn()
    torch.cuda.synchronize()
    try:
        prof_ctx = profile(activities=[ProfilerActivity.CUDA], acc_events=True)
    except TypeError:
        prof_ctx = profile(activities=[ProfilerActivity.CUDA])
    with prof_ctx as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        n = e.name
        if any(x in n for x in ('elementwise', 'Memset', 'memset', 'fill',
                                'Memcpy', 'vectorized_')):
            continue
        d = getattr(e, 'device_time', None)
        if d is None:
            d = getattr(e, 'cuda_time', 0.0)
        if d and d > 0:
            n = n[:80]
            if n not in names:
                names.append(n)
    assert names, 'probe found no kernels'
    return tuple(names)


def time_cuda_us_probed(fn):
    return time_cuda_us(fn, probe_kernel_names(fn))


def benchmark(module, positions):
    torch.manual_seed(42)
    _, weight, base, scale = make_inputs(1)
    dev = "cuda"
    wb = weight.to(torch.bfloat16)
    norm_w = (torch.rand(DIM, device=dev) + 0.5).bfloat16()
    print(f"\nBenchmark: {torch.cuda.get_device_name()}")
    print("HYBRID mhc: deep_gemm tf32 prenorm GEMM + our fused epilogue (PDL)")
    print("  full = with post/comb+Sinkhorn; no-pc = lite; +nrm = lite with")
    print("  FULLY fused attn_norm (production). +q8 = +nrm PLUS the fused")
    print("  fp8 activation emission (ablation: +q8 minus +nrm = its cost).")
    print("  gemm/fuse = per-kernel split of +q8 (kineto; PDL overlap")
    print("  double-counted). cuBLAS = middle GEMM only (bf16 floor).")
    print(
        f"{'M':>6} {'splits':>7} {'full':>9} {'no-pc':>8} {'+nrm':>8} "
        f"{'+q8':>8} {'gemm':>8} {'fuse':>8} {'cuBLAS':>9}"
    )
    print("-" * 78)

    for m in positions:
        hidden, _, _, _ = make_inputs(m, weight, base, scale)
        x = hidden.reshape(m, K_DIM)
        S = hc_n_splits(m)
        bufs = dict(
            ws=torch.empty(S, m, N_OUT, device=dev),
            sq=torch.empty(S, m, device=dev),
            collapsed=torch.empty(m, DIM, device=dev, dtype=torch.bfloat16),
            pre=torch.empty(m, HC, device=dev),
            post=torch.empty(m, HC, device=dev),
            comb=torch.empty(m, HC, HC, device=dev))

        full_fn = lambda: run_hybrid(module, hidden, weight, base, scale,
                                     with_post_comb=True, bufs=bufs)
        lite_fn = lambda: run_hybrid(module, hidden, weight, base, scale,
                                     bufs=bufs)
        nrm_fn = lambda: run_hybrid(module, hidden, weight, base, scale,
                                    norm_w=norm_w, bufs=bufs)
        xq8 = torch.empty(m, DIM, device=dev, dtype=torch.uint8)
        xsf8 = torch.empty(m, DIM // 128, device=dev, dtype=torch.uint8)
        q8_fn = lambda: run_hybrid(module, hidden, weight, base, scale,
                                   norm_w=norm_w, bufs=bufs, xq=xq8,
                                   xsf=xsf8)
        names = probe_kernel_names(q8_fn)
        full_us = time_cuda_us(full_fn, probe_kernel_names(full_fn))
        lite_us = time_cuda_us(lite_fn, probe_kernel_names(lite_fn))
        nrm_us = time_cuda_us(nrm_fn, probe_kernel_names(nrm_fn))
        ts = bench_kineto(q8_fn, names, suppress_kineto_output=True,
                          with_multiple_kernels=True)
        ts = [1e6 * v for v in (ts if isinstance(ts, tuple) else (ts,))]
        gemm = sum(t for n, t in zip(names, ts) if "prenorm" in n.lower()
                   or "gemm" in n.lower())
        fuse = sum(ts) - gemm
        cublas_us = time_cuda_us_probed(lambda: F.linear(x, wb))
        print(
            f"{m:6d} {S:7d} {full_us:9.3f} {lite_us:8.3f} {nrm_us:8.3f} "
            f"{sum(ts):8.3f} {gemm:8.3f} {fuse:8.3f} {cublas_us:9.3f}"
        )
        del hidden, x, bufs, xq8, xsf8
    print("-" * 78)


def parse_positions(value):
    return [int(v) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--correctness-positions", type=parse_positions,
        default=[1, 4, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this test must run on B300.")
        return 0

    # Correctness reference stays TRUE fp32 (matmul precision 'highest').
    torch.set_float32_matmul_precision("highest")

    major, minor = torch.cuda.get_device_capability()
    print(
        f"device={torch.cuda.get_device_name()} sm_{major}{minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print("JIT compiling hc_fused_kernel_tc.cu ...")
    module = load_cuda_module()
    get_dg()

    if not args.skip_correctness:
        if not test_correctness(module, args.correctness_positions):
            return 1
    benchmark(module, PROFILE_M)
    return 0


if __name__ == "__main__":
    sys.exit(main())
