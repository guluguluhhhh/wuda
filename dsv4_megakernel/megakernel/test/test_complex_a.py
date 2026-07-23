"""Correctness (vs fp32 torch golden) + B300 benchmark for the fused
complex_gemm STEP 1+2 operator (include/complex_a.cuh via kernels/complex_a.cu).

Op under test (decode regime, real shape M=1..64, N=4608, K=7168):
    y = x @ w.T                       (bf16 swap-AB 1SM UMMA, split-K ws-always)
    out[:, :1536] = rmsnorm(y[:, :1536], rms_w, eps)   (fused reduce + norm, y1 only)
    out[:, 1536:] : NOT written (y2/y3/y4 split-K partials stay in ws for step-3)
FUSENORM_NORM_DIM=1536 is clamped to N, so shapes with N<1536 norm the full row.

Structure mirrors test_wq_b_fp8_gemm.py: JIT compile via cpp_extension.load,
timing via the vendored DeepGEMM bench_kineto (8GB L2 flush before EVERY call,
kineto kernel device-time, warmup cycle discarded, MEAN over instances). The op
is TWO kernels per call (gemm_device -> PDL-overlapped gemm_rmsnorm_kernel), so
both are timed as a name tuple; cuBLAS baseline = bare bf16 torch.matmul via
its ('nvjet', 'reduce') cuBLASLt kernel pair, exactly like DeepGEMM's tests
(GEMM-only, no norm -> a conservative denominator for our fused sum).

Correctness golden follows test_complex.cu Test 1: fp32 GEMM -> fp32 RMSNorm
over the FIRST norm_len cols -> bf16 (both sides normalize full-precision
accumulators; residual = final bf16 round + split-K order). The tail
[norm_len, N) is pre-filled with a sentinel and must stay bit-untouched.

[TC/CC] HC post+comb tail: complex_a's extra CUDA-core warp computes
hc_fused_kernel_tc's POST gate + 20-iter Sinkhorn COMB (all fp32) hidden under
the GEMM. The tail takes the ALREADY-REDUCED + rms-folded mix [m,24] (split-K
reduce + Σx² done upstream by the hc epilogue) -> pure CUDA-core compute,
~80B in/out per position. test_hc_tail checks (a) post/comb vs an fp64 torch
golden implementing hc_reduce_and_fuse_kernel's exact math, (b) the y1 output
is BITWISE identical with the tail on vs off (full decoupling), and the
benchmark reports the fused-vs-base gemm_device delta (the tail must ride ~free).

    python test/test_complex_a.py                  # correctness + latency/BW/cuBLAS bench
    python test/test_complex_a.py --skip-correctness
    python test/test_complex_a.py --sweep-tiles    # extra forced BN x KM tile sweep
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

# complex_gemm step-1 real shape (test_complex.cu): N = 1536+512+2048+512
N_DEFAULT = 4608
K_DEFAULT = 7168
NORM_DIM = 1536          # FUSENORM_NORM_DIM (clamped to N inside the op)

# HC tail constants (must match complex_a.cuh::hc_tail == hc_fused_kernel_tc.cuh)
HC = 4                   # hc_mult
HC_NOUT = 24             # mix cols: pre[0,4) | post[4,8) | comb[8,24)
HC_SINKHORN_ITERS = 20


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def load_cuda_module():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, "..", "cutlass", "include")
    cutlass_tools_dir = os.path.join(proj_dir, "..", "cutlass", "tools", "util", "include")

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < 100:
        raise RuntimeError(f"tcgen05 requires Blackwell sm_100+, got sm_{sm}")

    cuda_flags = [
        "-O3", "--use_fast_math", "-std=c++17", "-lineinfo",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
        "-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
    ]
    return load(
        name="complex_a",
        sources=[os.path.join(proj_dir, "kernels", "complex_a.cu")],
        extra_include_paths=[
            os.path.join(proj_dir, "include"),
            cutlass_dir,
            cutlass_tools_dir,
        ],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


def make_inputs(M, N, K, seed):
    """test_complex.cu conventions: x/w uniform [-1,1] bf16, rms_w uniform [0.5,1.5) fp32."""
    torch.manual_seed(seed)
    x = torch.empty(M, K, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    w = torch.empty(N, K, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    rms_w = torch.empty(N, device="cuda", dtype=torch.float32).uniform_(0.5, 1.5)
    return x, w, rms_w


def ref_forward(x, w, rms_w, eps):
    """fp32 golden (test_complex.cu Test 1): fp32 GEMM -> fp32 RMSNorm over the
    FIRST norm_len cols -> bf16. Returns only the normed slice [M, norm_len]."""
    y = x.float() @ w.float().t()
    norm_len = min(NORM_DIM, y.shape[1])
    y1 = y[:, :norm_len]
    rms = torch.rsqrt(y1.square().mean(-1, keepdim=True) + eps)
    return (y1 * rms * rms_w[:norm_len]).to(torch.bfloat16)


def make_hc_inputs(M, seed):
    """Already-reduced + rms-folded mix [M,24] (the tail's contract; the upstream
    hc epilogue produces it). N(0,4) values give mid-range sigmoid / Sinkhorn
    logits after base/scale."""
    torch.manual_seed(seed)
    mix = torch.randn(M, HC_NOUT, device="cuda") * 2.0
    base = torch.randn(HC_NOUT, device="cuda")
    scale = torch.randn(3, device="cuda")
    return mix, base, scale


def hc_ref(mix, base, scale, hc_eps=1e-6):
    """fp64 golden of hc_reduce_and_fuse_kernel's post + Sinkhorn-comb branches
    on the pre-reduced mix (exact eps placement: round 1 row-norm ADDS eps AFTER
    the division; every other normalization has eps in the denominator)."""
    mix = mix.double()
    b, s = base.double(), scale.double()
    post = 2.0 * torch.sigmoid(mix[:, HC:2 * HC] * s[1] + b[HC:2 * HC])
    x = (mix[:, 2 * HC:] * s[2] + b[2 * HC:]).view(-1, HC, HC)      # row-major 4x4
    e = torch.exp(x - x.amax(-1, keepdim=True))
    v = e / e.sum(-1, keepdim=True) + hc_eps
    v = v / (v.sum(-2, keepdim=True) + hc_eps)
    for _ in range(HC_SINKHORN_ITERS - 1):
        v = v / (v.sum(-1, keepdim=True) + hc_eps)
        v = v / (v.sum(-2, keepdim=True) + hc_eps)
    return post.float(), v.float()


# --------------------------------------------------------------------- correctness
def run_case(module, M, N, K, eps=1e-6):
    x, w, rms_w = make_inputs(M, N, K, seed=42 + M)
    norm_len = min(NORM_DIM, N)

    handle = module.complex_a_setup(x, w)
    info = module.complex_a_info(handle)
    # sentinel fill: the op must write ONLY [:, :norm_len]; +inf survives bitwise
    out = torch.full((M, N), float("inf"), device="cuda", dtype=torch.bfloat16)
    module.complex_a_run(handle, out, rms_w, eps)
    torch.cuda.synchronize()
    module.complex_a_free(handle)

    ref = ref_forward(x, w, rms_w, eps)
    diff = calc_diff(out[:, :norm_len].float(), ref.float())
    tail_ok = norm_len >= N or bool(torch.all(out[:, norm_len:] == float("inf")))
    return diff, tail_ok, info


def test_correctness(module):
    print("\n[complex_a] fused GEMM + y1-RMSNorm vs fp32 torch golden")
    print(f"{'M':>4} {'N':>6} {'K':>6} {'BM':>4} {'BN':>4} {'ks':>3} {'km':>3} "
          f"{'diff':>12} {'tail':>6} {'result':>8}")
    print("-" * 66)
    ok_all = True
    # test_complex.cu M sweep (incl. non-aligned 7/17/31) + the norm_len-clamp
    # case (N=1024 < 1536 -> full-row norm, no unwritten tail)
    cases = [(m, N_DEFAULT, K_DEFAULT) for m in (1, 2, 4, 7, 8, 16, 17, 31, 32, 48, 64)]
    cases += [(8, 1024, 512), (16, 2048, 1024)]
    for M, N, K in cases:
        diff, tail_ok, (bm, bn, ks, tiles, km) = run_case(module, M, N, K)
        ok = diff < 1e-4 and tail_ok
        ok_all &= ok
        print(f"{M:4d} {N:6d} {K:6d} {bm:4d} {bn:4d} {ks:3d} {km:3d} "
              f"{diff:12.4e} {('ok' if tail_ok else 'DIRTY'):>6} "
              f"{('PASS' if ok else 'FAIL'):>8}")
    print("-" * 66)
    return ok_all


def test_hc_tail(module, eps=1e-6):
    """[TC/CC] HC post+comb tail correctness:
      * post/comb vs the fp64 torch golden (kernel is fp32 + --use_fast_math
        __expf, and the Sinkhorn normalizations are contraction maps, so ~1e-5
        abs is expected; threshold 1e-3);
      * y1 output must be BITWISE identical with the tail on vs off (the tail
        shares no barrier/SMEM/TMEM with the GEMM warps -- full decoupling);
      * M=1 (single position) and M=64 (grid-stride) cover both schedule paths."""
    print("\n[hc-tail] fused HC post + Sinkhorn comb (fp32, pre-reduced mix) vs fp64 golden")
    print(f"{'M':>4} {'d_post':>10} {'d_comb':>10} {'y1':>6} {'result':>8}")
    print("-" * 44)
    ok_all = True
    N, K = N_DEFAULT, K_DEFAULT
    for M in (1, 8, 33, 64):
        x, w, rms_w = make_inputs(M, N, K, seed=100 + M)
        handle = module.complex_a_setup(x, w)

        # base run (tail off) -> y1 reference for the bitwise check
        out_base = torch.full((M, N), float("inf"), device="cuda", dtype=torch.bfloat16)
        module.complex_a_run(handle, out_base, rms_w, eps)
        torch.cuda.synchronize()

        mix, base, scale = make_hc_inputs(M, seed=M * 31 + 7)
        post = torch.full((M, HC), float("nan"), device="cuda", dtype=torch.float32)
        comb = torch.full((M, HC, HC), float("nan"), device="cuda", dtype=torch.float32)
        out = torch.full((M, N), float("inf"), device="cuda", dtype=torch.bfloat16)
        module.complex_a_run(handle, out, rms_w, eps,
                             hc_mix=mix, hc_base=base, hc_scale=scale,
                             hc_post=post, hc_comb=comb)
        torch.cuda.synchronize()
        module.complex_a_free(handle)

        ref_post, ref_comb = hc_ref(mix, base, scale)
        d_post = (post - ref_post).abs().max().item()
        d_comb = (comb - ref_comb).abs().max().item()
        y1_same = bool(torch.equal(out, out_base))
        ok = d_post < 1e-3 and d_comb < 1e-3 and y1_same
        ok_all &= ok
        print(f"{M:4d} {d_post:10.3e} {d_comb:10.3e} "
              f"{('same' if y1_same else 'DIFF'):>6} {('PASS' if ok else 'FAIL'):>8}")
    print("-" * 44)
    return ok_all


# --------------------------------------------------------------------- benchmark
def bench_one(module, M, N, K, force=(0, 0, 0, 0), eps=1e-6, with_hc=False):
    """Setup once (hoisted out of the timed loop), then bench_kineto over the two
    fused-op kernels as a name tuple. with_hc=True additionally enables the
    [TC/CC] HC post+comb tail on the pre-reduced mix (gemm_us then measures
    gemm_device WITH the tail warp running). Returns (gemm_us, norm_us, info)."""
    x, w, rms_w = make_inputs(M, N, K, seed=42 + M)
    handle = module.complex_a_setup(x, w, *force)
    info = module.complex_a_info(handle)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    if with_hc:
        mix, base, scale = make_hc_inputs(M, seed=7 * M + 1)
        post = torch.empty(M, HC, device="cuda", dtype=torch.float32)
        comb = torch.empty(M, HC, HC, device="cuda", dtype=torch.float32)
        call = lambda: module.complex_a_run(handle, out, rms_w, eps,
                                            hc_mix=mix, hc_base=base, hc_scale=scale,
                                            hc_post=post, hc_comb=comb)
    else:
        call = lambda: module.complex_a_run(handle, out, rms_w, eps)
    gemm_s, norm_s = bench_kineto(
        call, ('gemm_device', 'gemm_rmsnorm'), suppress_kineto_output=True)
    module.complex_a_free(handle)
    del x, w, rms_w, out
    torch.cuda.empty_cache()
    return 1e6 * gemm_s, 1e6 * norm_s, info


def benchmark(module):
    """Latency sweep, DeepGEMM bench methodology end-to-end (test_wq_b_fp8_gemm.py
    style):
      * timing = vendored bench_kineto (8GB L2 flush per call, kineto kernel
        device-time, warmup cycle discarded, mean of instances); ours is the
        ('gemm_device', 'gemm_rmsnorm') kernel pair -- sum is an upper bound
        since the norm kernel PDL-overlaps the GEMM tail on-stream;
      * cuBLAS baseline = BARE bf16 GEMM (torch.matmul, cuBLASLt) timed via its
        ('nvjet', 'reduce') kernel pair exactly like DeepGEMM's own tests (the
        reduce kernel appears only when cuBLASLt picks split-K; a missing name
        contributes 0). GEMM-only -- no RMSNorm -- so %cuBLAS is conservative."""
    print("\n" + "=" * 60)
    print("Benchmark: complex_a fused GEMM + y1-RMSNorm latency sweep")
    print("=" * 60)
    dev = 'cuda'
    N, K = N_DEFAULT, K_DEFAULT
    norm_len = min(NORM_DIM, N)
    weight_bytes = N * K * 2                      # 63.1 MB (bf16) -- the access floor

    print(f"  N={N}, K={K}; weight {weight_bytes/1e6:.1f} MB (bf16); norm width {norm_len}")
    print("  Timing: DeepGEMM bench_kineto (vendored). ours = gemm_device +")
    print("  gemm_rmsnorm pair; cuBLAS = bare-bf16 cuBLASLt via torch.matmul,")
    print("  nvjet+reduce kernel pair (GEMM only, no norm).")
    print("  +hc = gemm_device WITH the [TC/CC] HC post+comb tail (pre-reduced mix:")
    print("  pure compute, ~80B in/out per position); d = +hc - gemm (should be ~0).")
    print("  BW = ESSENTIAL bytes (x + w + rms_w read + y1 write; split-K ws excluded).")
    print(f"  {'M':<4} {'tile':<14} {'gemm(us)':<9} {'norm(us)':<9} {'sum(us)':<9} "
          f"{'+hc(us)':<8} {'d(us)':<7} {'cuBLAS(us)':<11} {'ours_BW':<9} {'TFLOPS':<8} {'%cuBLAS':<8}")
    print("  " + "-" * 98)

    for M in [1, 2, 4, 8, 16, 32, 48, 64]:
        # ABBA bracket: each bench_one is preceded by 60x 8GB flush memsets, so
        # clocks drift DOWN monotonically through the sweep -- a fixed base-then-
        # fused order biases d positive on a still-cool card (observed ~+1us on
        # the first rows, decaying to 0 by M>=32). Measure A(base) B(+hc) B A and
        # take MIN per config (the mqa kernel_us lesson: min excludes throttled
        # instances by construction, unlike DeepGEMM's table mean).
        g1, n1, (bm, bn, ks, tiles, km) = bench_one(module, M, N, K)
        h1, _, _ = bench_one(module, M, N, K, with_hc=True)
        h2, _, _ = bench_one(module, M, N, K, with_hc=True)
        g2, n2, _ = bench_one(module, M, N, K)
        gemm_us, norm_us = min(g1, g2), min(n1, n2)
        hc_gemm_us = min(h1, h2)
        total = gemm_us + norm_us

        # cuBLAS bf16 GEMM baseline on the same operands (fresh tensors: bench_one
        # freed its own to keep peak memory flat across the sweep)
        x, w, _ = make_inputs(M, N, K, seed=42 + M)
        try:
            cb_pair = bench_kineto(
                lambda: torch.matmul(x, w.t()),
                ('nvjet', 'reduce'), suppress_kineto_output=True)
            cb = 1e6 * sum(cb_pair)
        except Exception as err:
            cb = None
            if M == 1:
                print(f"  (cuBLAS baseline unavailable: {err})")
        del x, w
        torch.cuda.empty_cache()

        obytes = M * K * 2 + weight_bytes + norm_len * 4 + M * norm_len * 2
        bw = obytes / (total * 1e-6) / 1e9
        tflops = 2 * M * N * K / (gemm_us * 1e-6) / 1e12
        tile = f"{bm}x{bn} ks{ks} km{km}"
        cb_s = f"{cb:<11.1f}" if cb else f"{'n/a':<11}"
        pct = f"{cb/total*100:<8.1f}" if cb else f"{'-':<8}"
        print(f"  {M:<4} {tile:<14} {gemm_us:<9.2f} {norm_us:<9.2f} {total:<9.2f} "
              f"{hc_gemm_us:<8.2f} {hc_gemm_us-gemm_us:<7.2f} {cb_s} {bw:<9.1f} "
              f"{tflops:<8.1f} {pct}")
    print("  " + "-" * 98)
    print("  (%cuBLAS > 100 => fused GEMM+norm beats cuBLAS's BARE GEMM)")


def sweep_tiles(module):
    """Forced-tile sweep (BN x KM per M) to re-verify choose_tile_config's fitted
    rule (BM=128, BN=cover16(M), ks auto) -- mirrors the sweep noted in the header."""
    print("\n" + "=" * 60)
    print(f"Tile sweep (forced BN x KM, BM=128, ks auto), N={N_DEFAULT} K={K_DEFAULT}")
    print("=" * 60)
    print(f"  {'M':>4} {'BN':>4} {'km':>3} {'ks':>3} {'gemm(us)':>9} {'norm(us)':>9} {'sum(us)':>8}")
    print("  " + "-" * 46)
    N, K = N_DEFAULT, K_DEFAULT
    for M in (1, 8, 16, 32, 64):
        best = None
        for bn in (16, 32, 48, 64, 96, 128):
            for km in (1, 2):
                gemm_us, norm_us, (_, _, ks, _, _) = bench_one(
                    module, M, N, K, force=(bn, 128, 0, km))
                total = gemm_us + norm_us
                if best is None or total < best[0]:
                    best = (total, bn, km)
                print(f"  {M:4d} {bn:4d} {km:3d} {ks:3d} {gemm_us:9.2f} {norm_us:9.2f} "
                      f"{total:8.2f}")
        print(f"  -> best for M={M}: BN={best[1]} km={best[2]} ({best[0]:.2f} us)")
        print("  " + "-" * 46)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-tiles", action="store_true",
                        help="forced BN x KM sweep to re-fit choose_tile_config's rule")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this test must run on B300.")
        return 0

    torch.set_float32_matmul_precision("highest")
    major, minor = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} sm_{major}{minor} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    print("JIT compiling complex_a.cu ...")
    module = load_cuda_module()

    ok = True
    if not args.skip_correctness:
        ok = test_correctness(module)
        ok &= test_hc_tail(module)
        print("\nALL PASSED" if ok else "\nCORRECTNESS FAILED")
    # benchmark always runs (test_wq_b_fp8_gemm.py convention): latency + BW + cuBLAS
    benchmark(module)
    if args.sweep_tiles:
        sweep_tiles(module)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
