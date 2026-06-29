"""
Test & Benchmark for Phase 2 Projection Kernel (wq_a + wkv GEMV with norm)
Matches origin/model.py: bf16 input, bf16 weight, bf16 output
"""
import os, sys, argparse
import torch
import torch.nn.functional as F

# ============================================================
# Config (DeepSeek-V4 Pro)
# ============================================================
D = 7168          # model hidden_size
N_QA = 1536       # q_lora_rank
N_KV = 512        # head_dim
RMS_NORM_EPS = 1e-6

# BW calculation: weight dominates
# wq_a: D*N_QA*2B = 21MB, wkv: D*N_KV*2B = 7MB
# input: D*2B = 14KB, outputs: (N_QA+N_KV)*2B = 4KB
BYTES_PER_POS = D * N_QA * 2 + D * N_KV * 2 + D * 2 + (N_QA + N_KV) * 2 + \
               N_QA * 4 + N_KV * 4  # norm weights
HBM_PEAK_GBS = 8000


# ============================================================
# Reference implementation (matching origin/model.py)
# ============================================================
def phase2_reference(x_normed, wq_a, q_a_norm_w, wkv, kv_norm_w, eps=RMS_NORM_EPS):
    """
    x_normed: [B, D] bf16 (already RMSNorm'd from Phase 1)
    wq_a: [N_QA, D] bf16
    q_a_norm_w: [N_QA] fp32
    wkv: [N_KV, D] bf16
    kv_norm_w: [N_KV] fp32
    """
    dtype = x_normed.dtype

    # wq_a GEMV
    qr = F.linear(x_normed, wq_a)  # [B, D] × [N_QA, D]^T → [B, N_QA]

    # q_norm: RMSNorm with weight
    qr_f = qr.float()
    qr_f = qr_f * torch.rsqrt(qr_f.square().mean(-1, keepdim=True) + eps)
    qr = (q_a_norm_w * qr_f).to(dtype)

    # wkv GEMV
    kv = F.linear(x_normed, wkv)  # [B, D] × [N_KV, D]^T → [B, N_KV]

    # kv_norm: RMSNorm with weight
    kv_f = kv.float()
    kv_f = kv_f * torch.rsqrt(kv_f.square().mean(-1, keepdim=True) + eps)
    kv = (kv_norm_w * kv_f).to(dtype)

    return qr, kv


# ============================================================
# JIT compile
# ============================================================
def load_cuda_module():
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)  # megakernel/
    cuda_flags = ['-O3', '--use_fast_math', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo']
    try:
        cap = torch.cuda.get_device_capability()
        cuda_flags.append(f'-gencode=arch=compute_{cap[0]*10+cap[1]},code=sm_{cap[0]*10+cap[1]}')
    except:
        cuda_flags.append('-gencode=arch=compute_100,code=sm_100')

    # CuTe/CUTLASS headers
    cutlass_include = '/home/admin/workspace/aop_lab/app_source/jinhua/dsv4_megakernel/cutlass/include'

    return load(
        name='phase2_proj',
        sources=[os.path.join(proj_dir, 'kernels', 'phase2_proj_kernel.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'), cutlass_include],
        extra_cuda_cflags=cuda_flags, verbose=True,
    )


# ============================================================
# Correctness test
# ============================================================
def test_correctness(module):
    print("=" * 60)
    print("Correctness Test: Phase 2 (wq_a + wkv GEMV with norm)")
    print("=" * 60)
    torch.manual_seed(42)
    device = 'cuda'

    # Single position test
    x_normed = torch.randn(1, D, device=device, dtype=torch.bfloat16) * 0.1
    wq_a = torch.randn(N_QA, D, device=device, dtype=torch.bfloat16) * 0.01
    q_a_norm_w = torch.ones(N_QA, device=device, dtype=torch.float32)
    wkv = torch.randn(N_KV, D, device=device, dtype=torch.bfloat16) * 0.01
    kv_norm_w = torch.ones(N_KV, device=device, dtype=torch.float32)

    # Reference
    ref_qr, ref_kv = phase2_reference(x_normed, wq_a, q_a_norm_w, wkv, kv_norm_w)

    # Kernel
    fused_qr, fused_kv = module.phase2_proj_forward(
        x_normed, wq_a, q_a_norm_w, wkv, kv_norm_w, RMS_NORM_EPS)

    print(f"\n{'Output':<12} {'Max Abs Err':<15} {'Mean Abs Err':<15} {'Cos Sim':<15} {'Pass'}")
    print("-" * 70)

    def check(name, ref, fused, atol):
        diff = (ref.float() - fused.float()).abs()
        abs_max = diff.max().item()
        abs_mean = diff.mean().item()
        cos = F.cosine_similarity(ref.float().flatten(), fused.float().flatten(), dim=0).item()
        passed = abs_max < atol and cos > 0.999
        print(f"{name:<12} {abs_max:<15.6e} {abs_mean:<15.6e} {cos:<15.8f} {'PASS' if passed else 'FAIL'}")
        return passed

    ok = True
    ok &= check("qr", ref_qr, fused_qr, 1e-2)
    ok &= check("kv", ref_kv, fused_kv, 1e-2)

    # Multi-position test
    print("\n--- Multi-position (batch=64) ---")
    x_normed_batch = torch.randn(64, D, device=device, dtype=torch.bfloat16) * 0.1
    ref_qr_b, ref_kv_b = phase2_reference(x_normed_batch, wq_a, q_a_norm_w, wkv, kv_norm_w)
    fused_qr_b, fused_kv_b = module.phase2_proj_forward(
        x_normed_batch, wq_a, q_a_norm_w, wkv, kv_norm_w, RMS_NORM_EPS)
    ok &= check("qr(B=64)", ref_qr_b, fused_qr_b, 1e-2)
    ok &= check("kv(B=64)", ref_kv_b, fused_kv_b, 1e-2)

    print("-" * 60)
    print("ALL TESTS PASSED!" if ok else "SOME TESTS FAILED!")
    return ok


# ============================================================
# Bandwidth sweep benchmark
# ============================================================
def benchmark(module, warmup=100, iters=500):
    print("\n" + "=" * 70)
    print(f"Bandwidth Sweep | {torch.cuda.get_device_name()} | HBM peak: {HBM_PEAK_GBS} GB/s")
    print("=" * 70)
    device = 'cuda'
    torch.manual_seed(42)

    wq_a = torch.randn(N_QA, D, device=device, dtype=torch.bfloat16) * 0.01
    q_a_norm_w = torch.ones(N_QA, device=device, dtype=torch.float32)
    wkv = torch.randn(N_KV, D, device=device, dtype=torch.bfloat16) * 0.01
    kv_norm_w = torch.ones(N_KV, device=device, dtype=torch.float32)

    batch_sizes = [1, 4, 16, 64, 128, 256, 512, 1024]
    print(f"\nPer-position data: {BYTES_PER_POS / 1024 / 1024:.1f} MB")
    print(f"\n{'B×S':<8} {'Latency(us)':<13} {'Throughput':<15} {'BW (GB/s)':<12} {'HBM Util':<10}")
    print("-" * 65)

    for num_pos in batch_sizes:
        x = torch.randn(num_pos, D, device=device, dtype=torch.bfloat16) * 0.1
        for _ in range(warmup):
            module.phase2_proj_forward(x, wq_a, q_a_norm_w, wkv, kv_norm_w, RMS_NORM_EPS)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.phase2_proj_forward(x, wq_a, q_a_norm_w, wkv, kv_norm_w, RMS_NORM_EPS)
        end.record()
        torch.cuda.synchronize()
        total_us = start.elapsed_time(end) / iters * 1000

        throughput = num_pos / (total_us * 1e-6)
        bw = (num_pos * BYTES_PER_POS) / (total_us * 1e-6) / 1e9
        util = bw / HBM_PEAK_GBS * 100
        print(f"{num_pos:<8} {total_us:<13.1f} {throughput/1e6:.2f} M pos/s   {bw:<12.1f} {util:<10.1f}%")

    print("-" * 65)


# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--skip-correctness', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available."); sys.exit(0)

    print(f"Config: D={D}, N_QA={N_QA}, N_KV={N_KV}")
    print(f"        block=1024, grid=2*SM")
    print(f"        Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"        Compute: sm_{cap[0]*10+cap[1]}\n")

    print("JIT compiling...")
    module = load_cuda_module()
    print("Done!\n")

    if not args.skip_correctness:
        if not test_correctness(module): sys.exit(1)
    if args.benchmark:
        benchmark(module)
