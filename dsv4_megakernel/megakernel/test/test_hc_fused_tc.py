"""Correctness and B300 benchmark for hc_fused_kernel_tc.cu."""

import argparse
import os
import sys

import torch
import torch.nn.functional as F


HC = 4
DIM = 7168
K_DIM = HC * DIM
N_OUT = (2 + HC) * HC
HC_EPS = 1e-6
RMS_NORM_EPS = 1e-6
SINKHORN_ITERS = 20
PROFILE_M = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]


def load_cuda_module():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, "..", "cutlass", "include")
    cutlass_tools_dir = os.path.join(
        proj_dir, "..", "cutlass", "tools", "util", "include"
    )

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < 100:
        raise RuntimeError(f"tcgen05 requires Blackwell sm_100+, got sm_{sm}")

    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
        "-DCUTE_ARCH_TCGEN05_TMEM_ENABLED=1",
        "-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1",
        "-DCUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
    ]
    return load(
        name="hc_fused_tc",
        sources=[os.path.join(proj_dir, "kernels", "hc_fused_kernel_tc.cu")],
        extra_include_paths=[
            os.path.join(proj_dir, "include"),
            cutlass_dir,
            cutlass_tools_dir,
        ],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


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
        weight = (
            torch.randn(N_OUT, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
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
    print("\nCorrectness: full RMSNorm + GEMM + HC epilogue")
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
        actual = module.hc_fused_forward_full(
            hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS
        )
        torch.cuda.synchronize()

        for name, got, ref in zip(
            ("collapsed", "pre", "post", "comb"), actual, expected
        ):
            atol, rtol = tolerances[name]
            max_abs, mean_abs, mean_rel = error_stats(got, ref)
            ok = torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
            all_ok &= ok
            print(
                f"{m:6d} {name:>10} {max_abs:12.4e} {mean_abs:12.4e} "
                f"{mean_rel:12.4e} {('PASS' if ok else 'FAIL'):>8}"
            )
        del hidden, expected, actual

    print("-" * 68)
    print("ALL PASSED" if all_ok else "CORRECTNESS FAILED")
    return all_ok


def time_cuda_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def benchmark(module, positions, warmup, iters):
    torch.manual_seed(42)
    _, weight, base, scale = make_inputs(1)
    device_name = torch.cuda.get_device_name()
    print(f"\nBenchmark: {device_name}")
    print(
        f"{'M':>6} {'MT':>4} {'NT':>4} {'splitK':>7} {'Ktile/s':>8} {'grid':>6} "
        f"{'fused us':>11} {'cuBLAS us':>11} {'fused/GEMM':>12}"
    )
    print("-" * 86)

    for m in positions:
        hidden, _, _, _ = make_inputs(m, weight, base, scale)
        x = hidden.reshape(m, K_DIM)
        cfg = module.hc_fused_tc_config(m)
        local_iters = max(20, iters // 2) if m >= 2048 else iters
        fused_us = time_cuda_us(
            lambda: module.hc_fused_forward_full(
                hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS
            ),
            warmup,
            local_iters,
        )
        gemm_us = time_cuda_us(lambda: F.linear(x, weight), warmup, local_iters)
        print(
            f"{m:6d} {cfg[7]:4d} {cfg[0]:4d} {cfg[1]:7d} {cfg[2]:8d} {cfg[3]:6d} "
            f"{fused_us:11.3f} {gemm_us:11.3f} {fused_us / gemm_us:12.3f}"
        )
        del hidden, x
    print("-" * 86)
    print("config columns: MT, NT, splitK, K tiles per split, physical CTA grid")


def profile_breakdown(module, m, repeats=10):
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(9000 + m)
    hidden, weight, base, scale = make_inputs(m)
    for _ in range(10):
        module.hc_fused_forward_full(
            hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS
        )
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):
            module.hc_fused_forward_full(
                hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS
            )
        torch.cuda.synchronize()

    print(f"\nCUDA kernel breakdown: M={m}, calls={repeats}")
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=20
        )
    )


def parse_positions(value):
    return [int(v) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--correctness-positions", type=parse_positions, default=[1, 4, 16, 64, 128, 256]
    )
    parser.add_argument(
        "--benchmark-positions", type=parse_positions, default=PROFILE_M
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--profile-breakdown",
        type=int,
        default=0,
        metavar="M",
        help="print per-kernel CUDA time for one M",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this test must run on B300.")
        return 0

    major, minor = torch.cuda.get_device_capability()
    print(
        f"device={torch.cuda.get_device_name()} sm_{major}{minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda}"
    )
    print("JIT compiling hc_fused_kernel_tc.cu ...")
    module = load_cuda_module()

    if not args.skip_correctness:
        if not test_correctness(module, args.correctness_positions):
            return 1
    if args.benchmark:
        benchmark(
            module, args.benchmark_positions, args.warmup, args.iters
        )
    if args.profile_breakdown:
        profile_breakdown(module, args.profile_breakdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
