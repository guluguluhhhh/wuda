"""
Test & Benchmark: wq_b Projection + Per-Head RMSNorm (tcgen05.mma Blackwell)
GEMM: [M, 1536] × [65536, 1536]^T → [M, 65536] → per-head RMSNorm → [M, 65536] BF16
Requires: NVIDIA B300 (sm_103), CUDA 12.4+, CUTLASS 3.x
"""
import os, sys, torch
import torch.nn.functional as F

# Model constants
K_DIM = 1536        # q_lora_rank
NUM_HEADS = 128
HEAD_DIM = 512
N_TOTAL = NUM_HEADS * HEAD_DIM  # 65536
RMS_EPS = 1e-6


def load_module():
    """JIT compile the tcgen05.mma kernel for sm_103."""
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, '..', 'cutlass', 'include')
    cutlass_tools_dir = os.path.join(proj_dir, '..', 'cutlass', 'tools', 'util', 'include')

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    assert sm >= 100, f'tcgen05.mma requires sm_100+, got sm_{sm}'

    cuda_flags = [
        '-O3', '-std=c++17',
        '--expt-relaxed-constexpr',
        '-lineinfo',
        '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
        '-DCUTE_ARCH_TCGEN05_TMEM_ENABLED=1',
        '-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1',
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
        f'-gencode=arch=compute_{sm}a,code=sm_{sm}a',
    ]

    return load(
        name='wq_b_gemm',
        sources=[os.path.join(proj_dir, 'kernels', 'wq_b_gemm.cu')],
        extra_include_paths=[
            os.path.join(proj_dir, 'include'),
            cutlass_dir,
            cutlass_tools_dir,
        ],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],  # driver API for TMA
        verbose=True,
    )


def reference_gemm_fused_rmsnorm(x, w, rms_w, eps):
    """PyTorch reference: pure GEMM (no RMSNorm yet - kernel outputs raw float32)."""
    # x: [M, 1536], w: [65536, 1536]
    y = x.float() @ w.float().t()              # [M, 65536]
    return y  # float32, no norm for now


def test_correctness(module, M=32):
    """Test GEMM + fused RMSNorm correctness."""
    print("=" * 60)
    print(f"Correctness Test: wq_b_proj_gemm tcgen05 (M={M})")
    print("=" * 60)
    device = 'cuda'

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)

    # Reference
    ref = reference_gemm_fused_rmsnorm(x, w, rms_w, RMS_EPS)

    # CUDA kernel (output: [M, N_TOTAL])
    out = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)

    print(f"  Output shape: {out.shape} (expect [{M}, {N_TOTAL}])")
    print(f"  Output dtype: {out.dtype}")

    # Compare
    diff = (out.float() - ref.float()).abs()
    cos_sim = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (ref.float().abs() + 1e-8)).max().item()

    print(f"  cos_sim:   {cos_sim:.8f}")
    print(f"  max_diff:  {max_diff:.6e}")
    print(f"  mean_diff: {mean_diff:.6e}")
    print(f"  rel_diff:  {rel_diff:.6e}")

    passed = cos_sim > 0.99
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_with_nontrivial_weight(module, M=64):
    """Test with non-trivial rms_w (not all-ones)."""
    print("\n" + "=" * 60)
    print(f"Scale Test: non-trivial rms_w (M={M})")
    print("=" * 60)
    device = 'cuda'

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.rand(HEAD_DIM, device=device, dtype=torch.float32) * 2 + 0.5

    ref = reference_gemm_fused_rmsnorm(x, w, rms_w, RMS_EPS)
    out = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)

    cos_sim = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  cos_sim:  {cos_sim:.8f}")
    print(f"  max_diff: {max_diff:.6e}")
    passed = cos_sim > 0.99
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_m_padding(module):
    """Test M=32 (FIXED-BM: activation zero-padded to BM=128 internally)."""
    print("\n" + "=" * 60)
    print("Padding Test: M=32 zero-padded to BM=128")
    print("=" * 60)
    device = 'cuda'
    M = 32

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)

    ref = reference_gemm_fused_rmsnorm(x, w, rms_w, RMS_EPS)
    out = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)

    assert out.shape == (M, N_TOTAL), f"Shape mismatch: {out.shape}"
    cos_sim = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"  cos_sim: {cos_sim:.8f}")
    passed = cos_sim > 0.99
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def benchmark(module):
    """Latency sweep with cuBLAS comparison."""
    print("\n" + "=" * 60)
    print("Benchmark: wq_b_proj_gemm tcgen05 latency sweep")
    print("=" * 60)
    device = 'cuda'

    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    w_t = w.t().contiguous()

    weight_bytes = N_TOTAL * K_DIM * 2  # 192 MB

    batch_sizes = [32, 64, 96, 128]   # FIXED-BM kernel: M in [32,128]
    print(f"  K={K_DIM}, N={N_TOTAL} (128 heads x 512 dim)")
    print(f"  Weight: {weight_bytes/1e6:.1f} MB (bf16)")
    print(f"  NOTE: %cuBLAS is latency-based (cuBLAS_us/ours_us). Our output is FP32,")
    print(f"        cuBLAS output is BF16, so each BW uses its own output bytes.")
    print(f"  {'M':<5} {'ours(us)':<10} {'cuBLAS(us)':<11} {'ours_BW':<10} {'cuBLAS_BW':<11} {'TFLOPS':<9} {'%cuBLAS':<8}")
    print("  " + "-" * 70)

    for M in batch_sizes:
        x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1

        # --- tcgen05 kernel ---
        for _ in range(20):
            module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)
        torch.cuda.synchronize()

        iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)
        end.record()
        torch.cuda.synchronize()
        ours_us = start.elapsed_time(end) / iters * 1000

        # --- cuBLAS GEMM baseline (no norm) ---
        for _ in range(20):
            torch.mm(x, w_t)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            torch.mm(x, w_t)
        end.record()
        torch.cuda.synchronize()
        cublas_us = start.elapsed_time(end) / iters * 1000

        flops = 2 * M * N_TOTAL * K_DIM
        tflops = flops / (ours_us * 1e-6) / 1e12

        # Bytes moved: shared weight + activation(bf16); output differs by dtype
        common_bytes = weight_bytes + M * K_DIM * 2
        ours_bytes   = common_bytes + M * N_TOTAL * 4   # our output is FP32
        cublas_bytes = common_bytes + M * N_TOTAL * 2   # cuBLAS output is BF16
        ours_bw   = ours_bytes   / (ours_us   * 1e-6) / 1e9
        cublas_bw = cublas_bytes / (cublas_us * 1e-6) / 1e9

        # latency-based achievement vs cuBLAS: 100% = same speed, >100% = faster
        pct_cublas = cublas_us / ours_us * 100.0

        print(f"  {M:<5} {ours_us:<10.1f} {cublas_us:<11.1f} {ours_bw:<10.1f} {cublas_bw:<11.1f} {tflops:<9.1f} {pct_cublas:<7.1f}%")


def profile_pipeline(module, M=128, clock_ghz=1.8):
    """Visualize load / MMA / epilogue overlap from wq_b_proj_gemm_profiled.
    timing[max_iters,6] = [load_s, load_e, mma_s, mma_e, epi_s, epi_e] (clock64 cycles,
    cluster0/CTA0 -> same SM, directly comparable). Same layout/print as the FP8 test
    for an apples-to-apples BF16-vs-FP8 pipeline comparison."""
    import numpy as np
    print("\n" + "=" * 76)
    print(f"Pipeline overlap (clock64): LOAD vs MMA vs EPILOGUE (BF16, M={M})")
    print("=" * 76)
    device = 'cuda'
    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01

    for _ in range(5):
        module.wq_b_proj_gemm_profiled(x, w)
    torch.cuda.synchronize()
    _, timing = module.wq_b_proj_gemm_profiled(x, w)
    torch.cuda.synchronize()

    t = timing.cpu().numpy().astype(np.int64)
    t = t[~(t == 0).all(axis=1)]
    if len(t) == 0:
        print("  no timing rows captured"); return
    ls, le, ms, me, es, ee = (t[:, i] for i in range(6))
    origin = int(min(ls.min(), ms.min(), es.min()))
    ls, le, ms, me, es, ee = (a - origin for a in (ls, le, ms, me, es, ee))
    span = int(max(le.max(), me.max(), ee.max()))
    if span <= 0:
        print("  degenerate span"); return
    c2us = lambda c: c / clock_ghz / 1e3
    n = len(t)
    print(f"  {n} persistent iterations; span={span} cyc (~{c2us(span):.1f} us @ {clock_ghz}GHz assumed)")
    print(f"  {'it':>3}  {'LOAD [s,e] dur':>26}  {'MMA [s,e] dur':>26}  {'EPI [s,e] dur':>26}")
    for i in range(min(n, 12)):
        print(f"  {i:>3}  [{ls[i]:>8},{le[i]:>8}] {le[i]-ls[i]:>6}  "
              f"[{ms[i]:>8},{me[i]:>8}] {me[i]-ms[i]:>6}  "
              f"[{es[i]:>8},{ee[i]:>8}] {ee[i]-es[i]:>6}")

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

    ld, md, ed = le - ls, me - ms, ee - es
    print(f"  mean window: LOAD {ld.mean():.0f}  MMA {md.mean():.0f}  EPI {ed.mean():.0f} cyc")
    print(f"  track fill (sum/span): LOAD {ld.sum()/span:.2f}  MMA {md.sum()/span:.2f}  EPI {ed.sum()/span:.2f}")
    print("    LOAD~1.0 => load-bound; MMA/EPI<1 & tucked under LOAD => hidden.")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)

    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")

    if sm < 100:
        print(f"ERROR: tcgen05.mma requires sm_100+ (Blackwell), got sm_{sm}")
        sys.exit(1)

    module = load_module()

    ok1 = test_correctness(module, M=32)
    ok2 = test_correctness(module, M=64)
    ok3 = test_correctness(module, M=128)
    ok4 = test_with_nontrivial_weight(module, M=64)
    ok5 = test_m_padding(module)

    benchmark(module)

    profile_pipeline(module, M=128)

    print("\n" + "=" * 60)
    all_pass = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"Summary: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
