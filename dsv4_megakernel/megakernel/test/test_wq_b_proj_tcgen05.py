"""
Test & Benchmark: wq_b Projection (pure GEMM, tcgen05.mma Blackwell)
GEMM: [M, 1536] x [65536, 1536]^T -> [M, 65536], BF16 in -> BF16 out.
Requires: NVIDIA B300 (sm_103), CUDA 12.4+, CUTLASS 3.x

This branch is the *unfused* pure-GEMM kernel (no RMSNorm). The benchmark also
reports the unfused total = cuBLAS GEMM + a standalone per-head RMSNorm pass, so
it can be compared against the fused GEMM+RMSNorm kernel on the other branch.
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
        name='wq_b_proj_tcgen05',
        sources=[os.path.join(proj_dir, 'kernels', 'wq_b_proj_gemm_tcgen05.cu')],
        extra_include_paths=[
            os.path.join(proj_dir, 'include'),
            cutlass_dir,
            cutlass_tools_dir,
        ],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],  # driver API for TMA
        verbose=True,
    )


def reference_gemm(x, w):
    """PyTorch reference: pure GEMM (no norm). Kernel outputs BF16."""
    return x.float() @ w.float().t()            # [M, 65536] fp32


def rmsnorm_per_head(y):
    """Standalone per-head RMSNorm on [M, N] (weightless), matches model.py wq_b."""
    yh = y.unflatten(-1, (NUM_HEADS, HEAD_DIM))                       # [M, 128, 512]
    inv = torch.rsqrt(yh.float().square().mean(-1, keepdim=True) + RMS_EPS)
    return (yh.float() * inv).to(y.dtype).flatten(-2)                 # [M, N]


def test_correctness(module, M):
    print("=" * 60)
    print(f"Correctness Test: wq_b_proj_gemm tcgen05 (M={M})")
    print("=" * 60)
    device = 'cuda'

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)

    ref = reference_gemm(x, w)                          # fp32 reference
    out = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)   # bf16 kernel output

    print(f"  Output shape: {tuple(out.shape)} dtype: {out.dtype} (expect bf16)")
    diff = (out.float() - ref.float()).abs()
    cos_sim = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"  cos_sim:  {cos_sim:.8f}")
    print(f"  max_diff: {diff.max().item():.6e}  mean_diff: {diff.mean().item():.6e}")
    passed = cos_sim > 0.99
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_rmsnorm(module, M):
    print("=" * 60)
    print(f"Correctness Test: standalone rmsnorm kernel (M={M})")
    print("=" * 60)
    device = 'cuda'
    y = torch.randn(M, N_TOTAL, device=device, dtype=torch.bfloat16)
    ref = rmsnorm_per_head(y)                  # torch reference
    out = module.rmsnorm(y, RMS_EPS)           # our kernel
    cos_sim = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    oh = out.float().unflatten(-1, (NUM_HEADS, HEAD_DIM))
    head_rms = oh.square().mean(-1).sqrt()
    print(f"  dtype: {out.dtype}  cos_sim: {cos_sim:.8f}")
    print(f"  per-head RMS: mean={head_rms.mean().item():.6f} (expect ~1.0)")
    passed = cos_sim > 0.99 and abs(head_rms.mean().item() - 1.0) < 0.05
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def _time_us(fn, iters=100, warmup=20):
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
    return start.elapsed_time(end) / iters * 1000  # us


def benchmark(module):
    print("\n" + "=" * 60)
    print("Benchmark: pure GEMM vs cuBLAS, and unfused total (GEMM + RMSNorm)")
    print("=" * 60)
    device = 'cuda'

    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    w_t = w.t().contiguous()
    weight_bytes = N_TOTAL * K_DIM * 2  # 201 MB (bf16)

    batch_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
    print(f"  K={K_DIM}, N={N_TOTAL} (128 heads x 512 dim), Weight {weight_bytes/1e6:.1f} MB")
    print(f"  All BW in GB/s (BF16 in/out). %cuBLAS = cuBLAS_us/ours_us (GEMM only).")
    print(f"  unfused = OUR GEMM kernel -> our RMSNorm kernel, timed END-TO-END (chained).")
    print(f"  ours_BW/unfBW use the SAME numerator = logical min bytes (weight+act+q), so they and")
    print(f"  the fused kernel are directly comparable (only the time differs).")
    print(f"  {'M':<5} {'ours(us)':<9} {'cuBLAS(us)':<11} {'unfused(us)':<12} "
          f"{'ours_BW':<9} {'unfBW':<9} {'%cuBLAS':<8}")
    print("  " + "-" * 80)

    for M in batch_sizes:
        x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1

        def unfused_pipeline():
            q = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)  # GEMM -> q (bf16)
            module.rmsnorm(q, RMS_EPS)                        # RMSNorm(q) reads GEMM's output

        ours_us    = _time_us(lambda: module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS))
        cublas_us  = _time_us(lambda: torch.mm(x, w_t))
        unfused_us = _time_us(unfused_pipeline)               # end-to-end GEMM + RMSNorm

        # Logical minimum data movement = weight + activation + one q output (bf16).
        # Same numerator for GEMM, end-to-end, and the fused kernel -> directly comparable.
        logical_bytes = weight_bytes + M * K_DIM * 2 + M * N_TOTAL * 2
        ours_bw    = logical_bytes / (ours_us    * 1e-6) / 1e9
        unfused_bw = logical_bytes / (unfused_us * 1e-6) / 1e9
        pct_cublas = cublas_us / ours_us * 100.0

        print(f"  {M:<5} {ours_us:<9.1f} {cublas_us:<11.1f} {unfused_us:<12.1f} "
              f"{ours_bw:<9.1f} {unfused_bw:<9.1f} {pct_cublas:<7.1f}%")


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

    all_pass = True
    for M in range(32, 257, 32):
        all_pass &= test_correctness(module, M)
    all_pass &= test_rmsnorm(module, 64)
    all_pass &= test_rmsnorm(module, 256)

    benchmark(module)

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
