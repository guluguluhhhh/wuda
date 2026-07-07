"""
Test & Benchmark: wq_b Projection GEMM via CUTLASS SM100 Collective
GEMM: [M, 1536] × [65536, 1536]^T → [M, 65536] FP32
Requires: NVIDIA L20D/B300 (sm_103), CUDA 12.4+, CUTLASS 3.x with SM100 support
"""
import os, sys, torch
import torch.nn.functional as F

K_DIM = 1536
NUM_HEADS = 128
HEAD_DIM = 512
N_TOTAL = NUM_HEADS * HEAD_DIM


def load_module():
    """JIT compile the CUTLASS SM100 GEMM kernel."""
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, '..', 'cutlass', 'include')
    cutlass_tools_dir = os.path.join(proj_dir, '..', 'cutlass', 'tools', 'util', 'include')

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    assert sm >= 100, f'SM100 required, got sm_{sm}'

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
        name='wq_b_proj_cutlass',
        sources=[os.path.join(proj_dir, 'kernels', 'wq_b_proj_gemm_cutlass.cu')],
        extra_include_paths=[
            os.path.join(proj_dir, 'include'),
            cutlass_dir,
            cutlass_tools_dir,
        ],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],
        verbose=True,
    )


def test_correctness(module, M=32):
    """Test GEMM correctness."""
    print("=" * 60)
    print(f"Correctness Test: CUTLASS SM100 GEMM (M={M})")
    print("=" * 60)
    device = 'cuda'

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01

    # Reference: PyTorch matmul
    ref = x.float() @ w.float().t()

    # CUTLASS kernel
    out = module.wq_b_proj_gemm_cutlass(x, w)

    print(f"  Output shape: {out.shape} (expect [{M}, {N_TOTAL}])")
    print(f"  Output dtype: {out.dtype}")

    diff = (out.float() - ref.float()).abs()
    cos_sim = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  cos_sim:   {cos_sim:.8f}")
    print(f"  max_diff:  {max_diff:.6e}")
    print(f"  mean_diff: {mean_diff:.6e}")

    passed = cos_sim > 0.99
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def benchmark(module):
    """Latency sweep with cuBLAS comparison."""
    print("\n" + "=" * 60)
    print("Benchmark: CUTLASS SM100 GEMM vs cuBLAS")
    print("=" * 60)
    device = 'cuda'

    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    w_t = w.t().contiguous()

    weight_bytes = N_TOTAL * K_DIM * 2

    batch_sizes = [32, 64, 128, 256]
    print(f"  K={K_DIM}, N={N_TOTAL} (128 heads x 512 dim)")
    print(f"  Weight: {weight_bytes/1e6:.1f} MB (bf16)")
    print(f"  {'M':<6} {'CUTLASS(us)':<12} {'cuBLAS(us)':<12} {'Ratio':<8} {'TFLOPS':<10} {'BW(GB/s)':<10}")
    print("  " + "-" * 60)

    for M in batch_sizes:
        x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1

        # --- CUTLASS kernel ---
        for _ in range(20):
            module.wq_b_proj_gemm_cutlass(x, w)
        torch.cuda.synchronize()

        iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.wq_b_proj_gemm_cutlass(x, w)
        end.record()
        torch.cuda.synchronize()
        cutlass_us = start.elapsed_time(end) / iters * 1000

        # --- cuBLAS baseline ---
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
        tflops = flops / (cutlass_us * 1e-6) / 1e12
        bytes_total = weight_bytes + M * K_DIM * 2 + M * N_TOTAL * 4
        bw_gbs = bytes_total / (cutlass_us * 1e-6) / 1e9
        ratio = cutlass_us / cublas_us

        print(f"  {M:<6} {cutlass_us:<12.1f} {cublas_us:<12.1f} {ratio:<8.2f} {tflops:<10.2f} {bw_gbs:<10.1f}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)

    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")

    if sm < 100:
        print(f"ERROR: SM100+ required, got sm_{sm}")
        sys.exit(1)

    module = load_module()

    ok1 = test_correctness(module, M=32)
    ok2 = test_correctness(module, M=64)
    ok3 = test_correctness(module, M=128)

    if ok1 and ok2 and ok3:
        benchmark(module)
    else:
        print("\nSkipping benchmark due to correctness failures")

    print("\n" + "=" * 60)
    all_pass = ok1 and ok2 and ok3
    print(f"Summary: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
