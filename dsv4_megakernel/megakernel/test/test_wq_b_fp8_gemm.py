"""
Test & Benchmark: wq_b_proj_gemm (tcgen05 FP8 block-scale).
  M<=128: swap-AB BM=M x BN128; M>=160: non-swap BM128 x BN224.
  x_fp8[M,1536] @ w_fp8[65536,1536].T -> y[M,65536] (FP32)
Requires: NVIDIA Blackwell (sm_100+), CUDA 12.8+, CUTLASS 3.x.

Scale-factor (SF) physical layout expected by the kernel (DeepGEMM 1D1D, 4-scale/uint32):
  - dtype int32, one uint32 packs 4 UE8M0 exponents (one per 32-K sub-block).
  - physical shape [sf_k, mn] with mn contiguous, sf_k = K/128 = 12.
    x_sf: [12, M] (per token), w_sf: [12, N] (per weight row).
  - UE8M0 byte e encodes scale 2^(e-127); e=127 (0x7F) => scale 1.0.
"""
import os, sys, torch
import torch.nn.functional as F

K_DIM   = 1536
N_TOTAL = 65536          # 128 heads x 512 dim
BLOCK_K = 128
GRAN_K  = 32
SF_K    = K_DIM // BLOCK_K          # 12
NUM_32K = K_DIM // GRAN_K           # 48 sub-blocks
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


def make_sf_ones(mn, device):
    """SF tensor [SF_K, mn] int32, all UE8M0 = 127 (scale 1.0)."""
    packed = (UE8M0_ONE | (UE8M0_ONE << 8) | (UE8M0_ONE << 16) | (UE8M0_ONE << 24))
    return torch.full((SF_K, mn), packed, dtype=torch.int32, device=device)


def make_cublaslt_sf_ones(mn, device):
    """cuBLASLt VEC32_UE8M0 scale storage for an outer dimension of size mn.

    cuBLASLt pads the outer dimension to 128 and the number of 32-K blocks to
    a multiple of four. Its scale tiles have an additional 128x4 permutation,
    which is immaterial here because every byte encodes scale 1.0.
    """
    if not hasattr(torch, 'float8_e8m0fnu'):
        return None
    aligned_mn = ((mn + 127) // 128) * 128
    padded_k_blocks = ((NUM_32K + 3) // 4) * 4
    raw = torch.full(
        (aligned_mn * padded_k_blocks,), UE8M0_ONE,
        dtype=torch.uint8, device=device)
    return raw.view(torch.float8_e8m0fnu)


def pack_sf(exps):
    """Pack per-32-K UE8M0 exponents into DeepGEMM MN-major layout [SF_K, mn] int32.
    exps: [mn, NUM_32K] uint8; sub-block s = kb*4 + j -> byte j of uint32[kb, mn]."""
    mn = exps.shape[0]
    e = exps.view(mn, SF_K, 4).to(torch.int64)
    packed = e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)  # [mn, SF_K]
    return packed.t().contiguous().to(torch.int32)                                  # [SF_K, mn]


def dequant(fp8, exps):
    """fp8 [mn,K], exps [mn,NUM_32K] -> float [mn,K] with per-32-K UE8M0 scale."""
    scale = torch.pow(2.0, exps.float() - 127.0)              # [mn, NUM_32K]
    return fp8.float() * scale.repeat_interleave(GRAN_K, dim=1)


def test_correctness(module, M):
    """Scale = 1 reference: plain fp8 -> fp32 matmul (near-exact e4m3)."""
    print("=" * 60)
    print(f"Correctness (scale=1): wq_b_proj_gemm FP8 (M={M})")
    print("=" * 60)
    dev = 'cuda'
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    x_sf = make_sf_ones(M, dev)
    w_sf = make_sf_ones(N_TOTAL, dev)

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
    """Non-trivial per-32-K scales: exposes SFA/SFB swap, sf_id order, packed layout."""
    print("\n" + "=" * 60)
    print(f"Correctness (per-32K scale): wq_b_proj_gemm FP8 (M={M})")
    print("=" * 60)
    dev = 'cuda'
    torch.manual_seed(M)
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    # exponents around 127 (scale in [2^-2, 2^2]); distinct per row & sub-block.
    ea = torch.randint(125, 130, (M, NUM_32K), device=dev, dtype=torch.uint8)
    eb = torch.randint(125, 130, (N_TOTAL, NUM_32K), device=dev, dtype=torch.uint8)
    x_sf = pack_sf(ea); w_sf = pack_sf(eb)

    out = module.wq_b_proj_gemm(x, x_sf, w, w_sf)
    ref = dequant(x, ea) @ dequant(w, eb).t()

    diff = (out.float() - ref).abs()
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    rel = (diff / (ref.abs() + 1e-4)).mean().item()
    print(f"  cos_sim:  {cos:.6f}")
    print(f"  max_diff: {diff.max().item():.4e}")
    print(f"  rel_diff: {rel:.4e}")
    ok = cos > 0.99
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(module):
    print("\n" + "=" * 60)
    print("Benchmark: wq_b_proj_gemm FP8 latency sweep")
    print("=" * 60)
    dev = 'cuda'
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w_sf = make_sf_ones(N_TOTAL, dev)
    w_t = w.t()                                 # [K,N] column-major for _scaled_mm
    weight_bytes = N_TOTAL * K_DIM * 1          # 100.7 MB (fp8)
    weight_sf_bytes = N_TOTAL * NUM_32K          # one UE8M0 byte per 32 K

    def timeit(fn, iters=50, batches=6, warmup=30):
        # min-of-batches: robust to DVFS/thermal throttling (the fastest batch ~=
        # unthrottled latency). Otherwise a hot GPU makes every number look ~2x slow.
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        best = float('inf')
        for _ in range(batches):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters): fn()
            e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) / iters * 1000)  # us
        return best

    # Same-math cuBLASLt baseline: e8m0 scale tensors select 32-element 1D block
    # scaling in torch._scaled_mm. Do not fall back to scalar scaling: that is a
    # different operation and produced a misleadingly fast reference.
    def cublas_us(x, M):
        if not hasattr(torch, "_scaled_mm"):
            return None
        if cublas_w_sf is None or cublas_sfs[M] is None:
            if M == 32:
                print("  (cuBLASLt block-scale baseline unavailable: "
                      "torch.float8_e8m0fnu is missing)")
            return None
        try:
            fn = lambda: torch._scaled_mm(
                x, w_t,
                scale_a=cublas_sfs[M], scale_b=cublas_w_sf,
                out_dtype=torch.float32, use_fast_accum=False)
            fn()  # probe
            return timeit(fn)
        except Exception as err:
            if M == 32:
                print(f"  (cuBLASLt per-32K block-scale baseline unavailable: {err})")
            return None

    print(f"  K={K_DIM}, N={N_TOTAL}; weight {weight_bytes/1e6:.1f} MB (e4m3)")
    print("  Dispatch: M<=128 swap-AB BM=M x BN128; M>=160 non-swap BM128xBN224")
    print("  Small-M stages: M32=11, M64=10, M96=9, M128=8")
    print(f"  NOTE: min-of-batches latency (robust to throttling). For stable numbers")
    print(f"        lock clocks: nvidia-smi -lgc <freq>. %cuBLAS = cuBLAS_us/ours_us.")
    print(f"        Baseline = torch._scaled_mm (cuBLASLt FP8, per-32K UE8M0); out FP32.")
    print(f"  {'M':<5} {'ours(us)':<10} {'cuBLAS(us)':<11} {'ours_BW':<10} {'cuBLAS_BW':<11} {'TFLOPS':<9} {'%cuBLAS':<8}")
    print("  " + "-" * 68)

    Ms = [32, 64, 96, 128, 160, 192, 224, 256]
    xs = {M: (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn) for M in Ms}
    sfs = {M: make_sf_ones(M, dev) for M in Ms}
    cublas_w_sf = make_cublaslt_sf_ones(N_TOTAL, dev)
    cublas_sfs = {M: make_cublaslt_sf_ones(M, dev) for M in Ms}

    # Two separate passes (ours, then baseline) so the compute-heavy baseline does
    # not heat the GPU in-between our measurements.
    ours_us  = {M: timeit(lambda M=M: module.wq_b_proj_gemm(xs[M], sfs[M], w, w_sf)) for M in Ms}
    cbls_us  = {M: cublas_us(xs[M], M) for M in Ms}

    for M in Ms:
        us = ours_us[M]; cb = cbls_us[M]
        # Logical bytes for the same block-scaled operation. Physical cuBLASLt
        # scale padding is deliberately excluded from this algorithm-level metric.
        obytes = (weight_bytes + weight_sf_bytes + M * K_DIM +
                  M * NUM_32K + M * N_TOTAL * 4)
        bw = obytes / (us * 1e-6) / 1e9
        tflops = 2 * M * N_TOTAL * K_DIM / (us * 1e-6) / 1e12
        cb_s  = f"{cb:<11.1f}" if cb else f"{'n/a':<11}"
        cb_bw = f"{obytes/(cb*1e-6)/1e9:<11.1f}" if cb else f"{'-':<11}"
        pct   = f"{cb/us*100:<8.1f}" if cb else f"{'-':<8}"
        print(f"  {M:<5} {us:<10.1f} {cb_s} {bw:<10.1f} {cb_bw} {tflops:<9.1f} {pct}")


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
    x_sf = make_sf_ones(M, dev); w_sf = make_sf_ones(N_TOTAL, dev)

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

    benchmark(module)

    profile_pipeline(module, M=128)   # swap-AB path
    profile_pipeline(module, M=256)   # non-swap path

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
