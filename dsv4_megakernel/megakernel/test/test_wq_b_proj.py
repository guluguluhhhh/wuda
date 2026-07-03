"""
Test & Benchmark: wq_b Projection + Per-Head RMSNorm (Fused)
GEMM: [M, 1536] × [65536, 1536]^T → [M, 65536] → reshape [M, 128, 512] → RMSNorm
"""
import os, sys, torch
import torch.nn.functional as F

# Model constants
K_DIM = 1536        # q_lora_rank
NUM_HEADS = 128
HEAD_DIM = 512
N_TOTAL = NUM_HEADS * HEAD_DIM  # 65536
RMS_EPS = 1e-6


def load_module(skip_norm=False):
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, '..', 'cutlass', 'include')

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    assert sm >= 90, f'TMA requires sm_90+, got sm_{sm}'

    cuda_flags = [
        '-O3', '-std=c++17',
        '--expt-relaxed-constexpr',
        '-lineinfo',
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
        f'-gencode=arch=compute_{sm},code=sm_{sm}',
    ]
    if skip_norm:
        cuda_flags.append('-DSKIP_NORM')

    name = 'wq_b_proj_nonorm' if skip_norm else 'wq_b_proj'
    return load(
        name=name,
        sources=[os.path.join(proj_dir, 'kernels', 'wq_b_proj_gemm.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'), cutlass_dir],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],  # driver API for cuTensorMapEncodeTiled
        verbose=True,
    )


def reference_gemm_fuse_norm_2(x, w, rms_w, eps):
    """PyTorch reference: GEMM + reshape + per-head RMSNorm."""
    # x: [M, 1536], w: [65536, 1536]
    y = x.float() @ w.float().t()           # [M, 65536]
    M = x.size(0)
    y = y.reshape(M * NUM_HEADS, HEAD_DIM)   # [M*128, 512]
    # RMSNorm
    mean_sq = y.square().mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(mean_sq + eps)
    y = y * scale * rms_w.unsqueeze(0)
    return y.reshape(M, NUM_HEADS, HEAD_DIM).to(torch.bfloat16)


def test_correctness(module, M=16):
    print("=" * 60)
    print(f"Correctness Test: wq_b_proj_gemm (M={M})")
    print("=" * 60)
    device = 'cuda'

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)

    # Reference
    ref = reference_gemm_fuse_norm_2(x, w, rms_w, RMS_EPS)

    # CUDA kernel
    out = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)

    print(f"  Output shape: {out.shape} (expect [{M}, {NUM_HEADS}, {HEAD_DIM}])")
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


def test_with_nontrivial_weight(module, M=32):
    """Test with non-trivial rms_w (not all-ones)."""
    print("\n" + "=" * 60)
    print(f"Scale Test: non-trivial rms_w (M={M})")
    print("=" * 60)
    device = 'cuda'

    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.rand(HEAD_DIM, device=device, dtype=torch.float32) * 2 + 0.5

    ref = reference_gemm_fuse_norm_2(x, w, rms_w, RMS_EPS)
    out = module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)

    cos_sim = F.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"  cos_sim: {cos_sim:.8f}")
    passed = cos_sim > 0.99
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def benchmark(module):
    print("\n" + "=" * 60)
    print("Benchmark: wq_b_proj_gemm latency sweep")
    print("=" * 60)
    device = 'cuda'

    # Weight is loaded once (static), so we pre-allocate
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)

    weight_bytes = N_TOTAL * K_DIM * 2  # 65536 * 1536 * 2 = 192 MB

    batch_sizes = [32, 64, 96, 128, 192, 256]
    print(f"  K={K_DIM}, N={N_TOTAL} (128 heads × 512 dim)")
    print(f"  Weight: {weight_bytes/1e6:.1f} MB (bf16)")
    print(f"  {'M':<8} {'Latency(us)':<12} {'TFLOPS':<10} {'BW(GB/s)':<10}")
    print("  " + "-" * 48)

    for M in batch_sizes:
        x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1

        # Warmup
        for _ in range(20):
            module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)
        torch.cuda.synchronize()

        # Benchmark
        iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)
        end.record()
        torch.cuda.synchronize()
        elapsed_us = start.elapsed_time(end) / iters * 1000

        flops = 2 * M * N_TOTAL * K_DIM
        tflops = flops / (elapsed_us * 1e-6) / 1e12
        # BW: weight(192MB) + input(M*1536*2) + output(M*128*512*2)
        bytes_total = weight_bytes + M * K_DIM * 2 + M * NUM_HEADS * HEAD_DIM * 2
        bw_gbs = bytes_total / (elapsed_us * 1e-6) / 1e9
        print(f"  {M:<8} {elapsed_us:<12.1f} {tflops:<10.2f} {bw_gbs:<10.1f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-norm', action='store_true',
                        help='Skip RMSNorm (measure pure GEMM latency)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)

    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute: sm_{cap[0]*10+cap[1]}")

    if args.skip_norm:
        print("[MODE] SKIP_NORM: pure GEMM correctness + latency")
        module = load_module(skip_norm=True)
        # Quick correctness check (no norm)
        device = 'cuda'
        M = 32
        x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
        rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
        ref = (x.float() @ w.float().t()).reshape(M, NUM_HEADS, HEAD_DIM).to(torch.bfloat16)
        out = module.wq_b_proj_gemm(x, w, rms_w, 1e-6)
        cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        print(f"  SKIP_NORM cos_sim: {cos:.8f} ({'PASS' if cos > 0.99 else 'FAIL'})")
        benchmark(module)
    else:
        module = load_module(skip_norm=False)
        ok1 = test_correctness(module, M=32)
        ok2 = test_correctness(module, M=64)
        ok3 = test_correctness(module, M=128)
        ok4 = test_with_nontrivial_weight(module, M=96)
        benchmark(module)
        print("\n" + "=" * 60)
        print(f"Summary: {'ALL PASS' if (ok1 and ok2 and ok3 and ok4) else 'SOME FAILED'}")
        print("=" * 60)
