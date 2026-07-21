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
# decode regime only (M<=256): all 32-aligned M plus 1/4/16 edge cases. M>256 is out
# of scope (the kernel has two configs: NT=8/splitK=18 for M<=128, NT=32/splitK=35 above).
PROFILE_M = [1, 4, 16, 32, 64, 96, 128, 160, 192, 224, 256]


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
        # tf32 MMA enable is auto-defined by cute/arch/config.hpp for sm_10xa; do not
        # pass -DCUTE_ARCH_TCGEN05_TF32_MMA_ENABLED (it would redefine -> warning).
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
        # tf32 GEMM path: weight is fp32 (read as tf32 by the MMA), matching the
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


BENCH_WARMUP = 50
BENCH_ITERS = 200


def benchmark(module, positions):
    torch.manual_seed(42)
    _, weight, base, scale = make_inputs(1)
    dev = "cuda"
    # cuBLAS baseline = bf16 F.linear: a stable "fastest vendor GEMM floor" (its fp32/tf32
    # path is pathological for N=24). Our kernel is tf32 -- higher precision than this.
    wb = weight.to(torch.bfloat16)
    # Config is constant across all M (single tuned config) -> print once in the header.
    cfg0 = module.hc_fused_tc_config(positions[0])
    print(f"\nBenchmark: {torch.cuda.get_device_name()}")
    print("Full MHC op: RMSNorm + GEMM(split-K) + activation + Sinkhorn + collapse")
    print(f"config (all M): M_TILE={cfg0[7]}  N_TILE={cfg0[0]}  splitK={cfg0[1]}  "
          f"K_tiles/split={cfg0[2]}")
    print("  mhc/cuBLAS = wall time (us). gemm/reduce/sinkhorn/collapse = clock64 stage")
    print("  medians (exclude launch/gap); sinkhorn ‖ collapse run concurrently.")
    print(
        f"{'M':>6} {'grid':>6} {'mhc us':>10} {'cuBLAS':>9} "
        f"{'gemm':>8} {'reduce':>8} {'sinkhrn':>8} {'collapse':>9}"
    )
    print("-" * 76)

    for m in positions:
        hidden, _, _, _ = make_inputs(m, weight, base, scale)
        x = hidden.reshape(m, K_DIM)
        cfg = module.hc_fused_tc_config(m)

        # preallocate outputs so we time the op, not per-call allocation
        collapsed = torch.empty(m, DIM, device=dev, dtype=torch.bfloat16)
        pre = torch.empty(m, HC, device=dev, dtype=torch.float32)
        post = torch.empty(m, HC, device=dev, dtype=torch.float32)
        comb = torch.empty(m, HC, HC, device=dev, dtype=torch.float32)
        mhc_us = time_cuda_us(
            lambda: module.hc_fused_forward_out(
                hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS,
                collapsed, pre, post, comb),
            BENCH_WARMUP, BENCH_ITERS,
        )
        cublas_us = time_cuda_us(lambda: F.linear(x, wb), BENCH_WARMUP, BENCH_ITERS)
        gemm, reduce, sinkhorn, collapse = profile_stages(
            module, hidden, weight, base, scale)
        print(
            f"{m:6d} {cfg[3]:6d} {mhc_us:10.3f} {cublas_us:9.3f} "
            f"{gemm:8.3f} {reduce:8.3f} {sinkhorn:8.3f} {collapse:9.3f}"
        )
        del hidden, x, collapsed, pre, post, comb
    print("-" * 76)


def gpu_clock_mhz():
    """SM clock in MHz for converting clock64 cycles -> us. Falls back to 2100."""
    try:
        khz = torch.cuda.get_device_properties(0).clock_rate  # kHz
        if khz and khz > 0:
            return khz / 1000.0
    except Exception:
        pass
    return 2100.0


def profile_stages(module, hidden, weight, base, scale, repeats=30):
    """Median per-stage us from in-kernel clock64 stamps (prof buffer, block 0).
    Layout: [0]gemm start [1]gemm end | [2]epi start [3]reduce+rms [4]act |
            [5]sinkhorn end (warp0) | [6]collapse start [7]collapse end (warp1).
    Sinkhorn (warp0) ‖ collapse (warp1) run concurrently, so collapse is timed on its
    own warp -> [7]-[6] is the real collapse duration, not just the overhang.
    Returns (gemm, reduce, sinkhorn, collapse) in microseconds.
    """
    for _ in range(5):
        module.hc_fused_forward_profiled(hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        res = module.hc_fused_forward_profiled(
            hidden, weight, base, scale, HC_EPS, RMS_NORM_EPS)
        torch.cuda.synchronize()
        samples.append([int(v) for v in res[-1].cpu().tolist()])
    clk = gpu_clock_mhz()

    def med(a, b):
        vals = sorted((s[b] - s[a]) / clk for s in samples)
        return vals[len(vals) // 2]

    return med(0, 1), med(2, 3), med(4, 5), med(6, 7)  # gemm, reduce, sinkhorn, collapse


def parse_positions(value):
    return [int(v) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--correctness-positions", type=parse_positions,
        default=[1, 4, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this test must run on B300.")
        return 0

    # Correctness reference stays TRUE fp32 (matmul precision 'highest', the default) so
    # the allclose error reflects our tf32 kernel vs a perfect fp32 ground truth. The
    # cuBLAS *perf* baseline flips to tf32 locally inside benchmark() for a fair compare.
    torch.set_float32_matmul_precision("highest")

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
        benchmark(module, PROFILE_M)
    return 0


if __name__ == "__main__":
    sys.exit(main())
