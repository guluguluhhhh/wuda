"""
Test & Benchmark: wq_b_proj_gemm (tcgen05 FP8 block-scale).
  M<=128: swap-AB UMMA_N=M x BN128; M>=160: non-swap BM128 x BN224.
  x_fp8[M,1536] @ w_fp8[65536,1536].T -> y[M,65536] (FP32)
Requires: NVIDIA Blackwell (sm_100+), CUDA 12.8+, CUTLASS 3.x.

Native DSV4 scale-factor layout expected by the kernel:
  - dtype float8_e8m0fnu (raw uint8 is also accepted).
  - x_sf: [M, K/128] = [M, 12], one scale per token/K128.
  - w_sf: [N/128, K/128] = [512, 12], one scale per N128xK128 block.
  - The kernel replicates each K128 scale into four K32 UMMA scale IDs.
  - UE8M0 byte e encodes scale 2^(e-127); e=127 (0x7F) => scale 1.0.
"""
import os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

K_DIM   = 1536
N_TOTAL = 65536          # 128 heads x 512 dim
BLOCK_K = 128
QUANT_BLOCK_K = 128
WEIGHT_QUANT_BLOCK_N = 128
SF_K    = K_DIM // QUANT_BLOCK_K          # 12
UE8M0_ONE = 0x7F                    # exponent 127 -> 2^0 = 1.0


def load_module():
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, '..', 'cutlass', 'include')
    cutlass_tools_dir = os.path.join(proj_dir, '..', 'cutlass', 'tools', 'util', 'include')

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    assert sm >= 100, f'tcgen05 block-scale requires sm_100+, got sm_{sm}'

    cuda_flags = [
        '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
        '-DCUTE_ARCH_TCGEN05_TMEM_ENABLED=1',
        '-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1',
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
        f'-gencode=arch=compute_{sm}a,code=sm_{sm}a',
    ]
    return load(
        name='wq_b_fp8_gemm',
        sources=[os.path.join(proj_dir, 'kernels', 'wq_b_fp8_gemm.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'), cutlass_dir, cutlass_tools_dir],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],  # driver API for TMA
        verbose=True,
    )


def as_ue8m0(exps):
    """Reinterpret raw UE8M0 exponent bytes without numerical conversion."""
    assert hasattr(torch, 'float8_e8m0fnu')
    return exps.contiguous().view(torch.float8_e8m0fnu)


def make_act_sf_ones(m, device):
    """Native activation SF [M,K/128], one UE8M0 byte per 1x128 block."""
    raw = torch.full((m, SF_K), UE8M0_ONE, dtype=torch.uint8, device=device)
    return as_ue8m0(raw)


def make_weight_sf_ones(device):
    """Native weight SF [N/128,K/128], one UE8M0 byte per 128x128 block."""
    raw = torch.full(
        (N_TOTAL // WEIGHT_QUANT_BLOCK_N, SF_K), UE8M0_ONE,
        dtype=torch.uint8, device=device)
    return as_ue8m0(raw)


def dequant_act(fp8, exps):
    """Apply native per-token/K128 activation scales."""
    scale = torch.pow(2.0, exps.float() - 127.0)
    return fp8.float() * scale.repeat_interleave(QUANT_BLOCK_K, dim=1)


def dequant_weight(fp8, exps):
    """Apply native N128xK128 weight scales."""
    scale = torch.pow(2.0, exps.float() - 127.0)
    scale = scale.repeat_interleave(WEIGHT_QUANT_BLOCK_N, dim=0)[:N_TOTAL]
    return fp8.float() * scale.repeat_interleave(QUANT_BLOCK_K, dim=1)


def test_correctness(module, M):
    """Scale = 1 reference: plain fp8 -> fp32 matmul (near-exact e4m3)."""
    print("=" * 60)
    print(f"Correctness (scale=1): wq_b_proj_gemm FP8 (M={M})")
    print("=" * 60)
    dev = 'cuda'
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    x_sf = make_act_sf_ones(M, dev)
    w_sf = make_weight_sf_ones(dev)

    out = module.wq_b_proj_gemm(x, x_sf, w, w_sf)          # [M, N] FP32
    ref = x.float() @ w.float().t()

    print(f"  Output shape: {tuple(out.shape)} (expect [{M}, {N_TOTAL}]), dtype {out.dtype}")
    diff = (out.float() - ref).abs()
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    print(f"  cos_sim:  {cos:.6f}")
    print(f"  max_diff: {diff.max().item():.4e}")
    ok = cos > 0.99
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_correctness_scaled(module, M):
    """Non-trivial native scales: exposes K128 replication and N128 weight broadcast."""
    print("\n" + "=" * 60)
    print(f"Correctness (native K128 scale): wq_b_proj_gemm FP8 (M={M})")
    print("=" * 60)
    dev = 'cuda'
    torch.manual_seed(M)
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    # Activation: distinct per token/K128. Weight: distinct per N128xK128 block.
    ea = torch.randint(125, 130, (M, SF_K), device=dev, dtype=torch.uint8)
    eb = torch.randint(
        125, 130, (N_TOTAL // WEIGHT_QUANT_BLOCK_N, SF_K),
        device=dev, dtype=torch.uint8)
    x_sf = as_ue8m0(ea)
    w_sf = as_ue8m0(eb)

    out = module.wq_b_proj_gemm(x, x_sf, w, w_sf)
    ref = dequant_act(x, ea) @ dequant_weight(w, eb).t()

    diff = (out.float() - ref).abs()
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    rel = (diff / (ref.abs() + 1e-4)).mean().item()
    print(f"  cos_sim:  {cos:.6f}")
    print(f"  max_diff: {diff.max().item():.4e}")
    print(f"  rel_diff: {rel:.4e}")
    ok = cos > 0.99
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def test_head_ssq(module, M):
    """RMSNorm scale folding: the epilogue RED-accumulates per-(row, head)
    sum-of-squares into head_ssq[M,128]. Reference = the kernel's OWN fp32 output
    (isolates the accumulation from fp8 quant error); tolerance covers the
    non-deterministic f32 atomic order. Also sanity-checks s_h = rsqrt(ssq/512+eps)."""
    print("\n" + "=" * 60)
    print(f"Correctness (head_ssq / RMSNorm scale folding, M={M})")
    print("=" * 60)
    dev = 'cuda'
    torch.manual_seed(M + 7)
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    x_sf = make_act_sf_ones(M, dev)
    w_sf = make_weight_sf_ones(dev)

    ssq = torch.zeros(M, N_TOTAL // 512, device=dev, dtype=torch.float32)
    out = module.wq_b_proj_gemm(x, x_sf, w, w_sf, head_ssq=ssq)
    ssq_ref = out.view(M, N_TOTAL // 512, 512).double().square().sum(-1).float()
    rel = ((ssq - ssq_ref).abs() / (ssq_ref + 1e-6)).max().item()
    s_h = torch.rsqrt(ssq / 512.0 + 1e-6)
    s_ref = torch.rsqrt(ssq_ref / 512.0 + 1e-6)
    s_rel = ((s_h - s_ref).abs() / s_ref).max().item()
    ok = rel < 1e-3
    print(f"  ssq max rel diff vs own-output reference: {rel:.3e}")
    print(f"  s_h max rel diff: {s_rel:.3e}   (s_h range: {s_h.min().item():.4f} .. {s_h.max().item():.4f})")
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(module):
    """Latency sweep, DeepGEMM bench methodology end-to-end:
      * timing = vendored bench_kineto (8GB L2 flush per call, kineto kernel
        device-time, warmup cycle discarded, mean of instances);
      * cuBLAS baseline = BARE fp8 GEMM with per-tensor scale (torch._scaled_mm,
        scale=1.0) -- the same operation deep_gemm.cublaslt_gemm_nt performs --
        timed via its ('nvjet', 'reduce') kernel pair exactly like DeepGEMM's
        own tests (the reduce kernel appears only when cuBLASLt picks split-K;
        a missing name contributes 0)."""
    print("\n" + "=" * 60)
    print("Benchmark: wq_b_proj_gemm FP8 latency sweep")
    print("=" * 60)
    dev = 'cuda'
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w_sf = make_weight_sf_ones(dev)
    w_t = w.t()                                 # [K,N] column-major for _scaled_mm
    one = torch.ones((), device=dev, dtype=torch.float32)
    weight_bytes = N_TOTAL * K_DIM * 1          # 100.7 MB (fp8)
    weight_sf_bytes = (N_TOTAL // WEIGHT_QUANT_BLOCK_N) * SF_K

    print(f"  K={K_DIM}, N={N_TOTAL}; weight {weight_bytes/1e6:.1f} MB (e4m3)")
    print("  Dispatch: M<=128 swap-AB UMMA_N=M x BN128; M>=160 non-swap BM128xBN224")
    print("  Timing: DeepGEMM bench_kineto (vendored). cuBLAS = bare-fp8 cuBLASLt")
    print("  via torch._scaled_mm(scale=1), nvjet+reduce kernel pair.")
    print(f"  {'M':<5} {'ours(us)':<10} {'+ssq(us)':<10} {'cuBLAS(us)':<11} {'ours_BW':<10} {'cuBLAS_BW':<11} {'TFLOPS':<9} {'%cuBLAS':<8}")
    print("  " + "-" * 78)

    for M in [32, 64, 96, 128, 160, 192, 224, 256]:
        x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
        x_sf = make_act_sf_ones(M, dev)
        ssqbuf = torch.zeros(M, 128, device=dev, dtype=torch.float32)

        us = 1e6 * bench_kineto(
            lambda: module.wq_b_proj_gemm(x, x_sf, w, w_sf),
            'wq_b_proj_kernel', suppress_kineto_output=True)
        # + head_ssq (RMSNorm scale folding); buffer accumulates garbage across
        # timing iters -- irrelevant for latency; must be ~= ours.
        sq = 1e6 * bench_kineto(
            lambda: module.wq_b_proj_gemm(x, x_sf, w, w_sf, head_ssq=ssqbuf),
            'wq_b_proj_kernel', suppress_kineto_output=True)
        try:
            cb_pair = bench_kineto(
                lambda: torch._scaled_mm(x, w_t, scale_a=one, scale_b=one,
                                         out_dtype=torch.float32),
                ('nvjet', 'reduce'), suppress_kineto_output=True)
            cb = 1e6 * sum(cb_pair)
        except Exception as err:
            cb = None
            if M == 32:
                print(f"  (cuBLAS baseline unavailable: {err})")

        obytes = (weight_bytes + weight_sf_bytes + M * K_DIM +
                  M * SF_K + M * N_TOTAL * 4)
        bw = obytes / (us * 1e-6) / 1e9
        tflops = 2 * M * N_TOTAL * K_DIM / (us * 1e-6) / 1e12
        cb_s  = f"{cb:<11.1f}" if cb else f"{'n/a':<11}"
        cb_bw = f"{obytes/(cb*1e-6)/1e9:<11.1f}" if cb else f"{'-':<11}"
        pct   = f"{cb/us*100:<8.1f}" if cb else f"{'-':<8}"
        print(f"  {M:<5} {us:<10.1f} {sq:<10.1f} {cb_s} {bw:<10.1f} {cb_bw} {tflops:<9.1f} {pct}")


def profile_pipeline(module, M=128, clock_ghz=1.8):
    """Visualize load / MMA / epilogue overlap from wq_b_proj_gemm_profiled.
    timing[max_iters,7] = [load_s, load_e, mma_s, mma_e, epi_s, epi_e, mma_wait]
    (clock64 cycles, cluster0/CTA0 -> same SM). col6 mma_wait = cycles the MMA warp
    spent WAITING (tmem_empty + with_sf_full); MMA_active = (mma_e-mma_s) - mma_wait
    is the warp's real (non-stalled) work -> tells compute vs stall directly."""
    import numpy as np
    print("\n" + "=" * 76)
    print(f"Pipeline overlap (clock64): LOAD vs MMA vs EPILOGUE (M={M})")
    print("=" * 76)
    dev = 'cuda'
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    x_sf = make_act_sf_ones(M, dev); w_sf = make_weight_sf_ones(dev)

    for _ in range(5):
        module.wq_b_proj_gemm_profiled(x, x_sf, w, w_sf)
    torch.cuda.synchronize()
    _, timing = module.wq_b_proj_gemm_profiled(x, x_sf, w, w_sf)
    torch.cuda.synchronize()

    t = timing.cpu().numpy().astype(np.int64)
    t = t[~(t == 0).all(axis=1)]                 # drop iterations that never ran
    if len(t) == 0:
        print("  no timing rows captured"); return
    ls, le, ms, me, es, ee = (t[:, i] for i in range(6))
    mw = t[:, 6]                                  # MMA wait cycles (duration, not origin-relative)
    origin = int(min(ls.min(), ms.min(), es.min()))
    ls, le, ms, me, es, ee = (a - origin for a in (ls, le, ms, me, es, ee))
    span = int(max(le.max(), me.max(), ee.max()))
    if span <= 0:
        print("  degenerate span"); return
    c2us = lambda c: c / clock_ghz / 1e3
    n = len(t)
    mma_active = (me - ms) - mw                   # MMA warp real work (excl. wait)
    print(f"  {n} persistent iterations; span={span} cyc (~{c2us(span):.1f} us @ {clock_ghz}GHz assumed)")

    # ---- per-iteration windows (relative cycles), first 12. MMA split into wait|active ----
    print(f"  {'it':>3}  {'LOAD dur':>9}  {'MMA dur':>9} {'(wait':>7}{'/active)':>9}  {'EPI dur':>9}")
    for i in range(min(n, 12)):
        print(f"  {i:>3}  {le[i]-ls[i]:>9}  {me[i]-ms[i]:>9} ({mw[i]:>6}/{mma_active[i]:>7})  {ee[i]-es[i]:>9}")

    # ---- 3-track ASCII timeline over the full span ----
    W = 100
    def track(s, e, ch):
        line = [' '] * W
        for a, b in zip(s, e):
            i0 = int(a / span * W); i1 = max(i0 + 1, int(b / span * W))
            for j in range(i0, min(i1, W)):
                line[j] = ch
        return ''.join(line)
    print("  timeline (each track = that stage's windows across all iters):")
    print("   LOAD |" + track(ls, le, 'L') + "|")
    print("   MMA  |" + track(ms, me, 'M') + "|")
    print("   EPI  |" + track(es, ee, 'E') + "|")

    # ---- summary: durations + track fill (clock-agnostic) ----
    ld, md, ed = le - ls, me - ms, ee - es
    print(f"  mean window: LOAD {ld.mean():.0f}  MMA {md.mean():.0f} (wait {mw.mean():.0f} / active {mma_active.mean():.0f})  EPI {ed.mean():.0f} cyc")
    print(f"  track fill (sum/span): LOAD {ld.sum()/span:.2f}  MMA {md.sum()/span:.2f}  MMA_active {mma_active.sum()/span:.2f}  EPI {ed.sum()/span:.2f}")
    print("    MMA_active << MMA => MMA warp is mostly WAITING (load/consume-bound), not compute-slow.")
    print("    If MMA_active ~ MMA fill ~1 => genuinely MMA-warp-bound.")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")
    if sm < 100:
        print(f"ERROR: tcgen05 block-scale requires sm_100+ (Blackwell), got sm_{sm}")
        sys.exit(1)

    module = load_module()

    results = []
    for M in [32, 64, 96, 128, 160, 192, 224, 256]:
        results.append(test_correctness(module, M))
    for M in [32, 96, 128, 160, 256]:
        results.append(test_correctness_scaled(module, M))
    for M in [32, 128, 160, 256]:
        results.append(test_head_ssq(module, M))

    benchmark(module)

    profile_pipeline(module, M=128)   # swap-AB path
    profile_pipeline(module, M=256)   # non-swap path

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
