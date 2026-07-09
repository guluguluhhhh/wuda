"""
Test & Benchmark: wq_b Projection + Per-Head RMSNorm (tcgen05.mma Blackwell)
GEMM: [M, 1536] × [65536, 1536]^T → [M, 65536] → per-head RMSNorm → [M, 65536] BF16
Requires: NVIDIA B300 (sm_103), CUDA 12.4+, CUTLASS 3.x

Profiling (clean, fast — skips correctness/benchmark via WQB_PROFILE_M):
sudo -E env WQB_PROFILE_M=32 /usr/local/cuda/bin/ncu \
  --set full --kernel-name-base demangled \
  -k "regex:wq_b_proj_kernel<\(int\)32>" \
  --launch-skip 20 --launch-count 1 \
  --csv --page raw --log-file m32_profile.csv \
  /home/admin/miniconda3/bin/python test/test_wq_b_proj_tcgen05.py
  (WQB_PROFILE_M runs a pure warm loop of that M; launch #21 = steady state.
   Omit the env var to run the full correctness+benchmark suite instead — the
   same ncu command still lands on a warm M=32 kernel, just after more launches.)
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


def reference_gemm_fused_rmsnorm(x, w, rms_w, eps):
    """PyTorch reference: GEMM + fused *weightless* per-head RMSNorm.

    Mirrors model.py wq_b exactly:
        q = wq_b(x).unflatten(-1, (n_heads, head_dim))
        q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)
    The kernel is weightless, so `rms_w` is unused (kept for API parity).
    """
    # x: [M, 1536], w: [65536, 1536]
    y = x.float() @ w.float().t()                       # [M, 65536]
    y = y.unflatten(-1, (NUM_HEADS, HEAD_DIM))          # [M, 128, 512]
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y.flatten(-2)                                # [M, 65536], float32


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

    # Sanity: fused RMSNorm makes each head's RMS == 1 (up to +eps).
    out_heads = out.float().unflatten(-1, (NUM_HEADS, HEAD_DIM))
    head_rms = out_heads.square().mean(-1).sqrt()          # [M, 128]
    print(f"  per-head RMS: mean={head_rms.mean().item():.6f} "
          f"min={head_rms.min().item():.6f} max={head_rms.max().item():.6f} (expect ~1.0)")

    passed = cos_sim > 0.99 and abs(head_rms.mean().item() - 1.0) < 0.05
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

    batch_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
    print(f"  K={K_DIM}, N={N_TOTAL} (128 heads x 512 dim)")
    print(f"  Weight: {weight_bytes/1e6:.1f} MB (bf16)")
    print(f"  NOTE: %cuBLAS is latency-based (cuBLAS_us/ours_us). Our output is FP32,")
    print(f"        cuBLAS output is BF16, so each BW uses its own output bytes.")
    print(f"        Ours also fuses per-head RMSNorm; cuBLAS baseline is GEMM-only")
    print(f"        (so unfused would additionally read+write D ~= 2*M*N*4 bytes).")
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


def profile_loop(module, M, iters=40):
    """Pure warm launch loop for one M — for a clean ncu capture (no correctness/benchmark).

    With `--launch-skip 20 --launch-count 1` the profiled kernel is launch #21,
    which is fully warm here (JIT done, caches warm, steady state).
    """
    print(f"[profile] warm loop: M={M}, {iters} launches of wq_b_proj_kernel<{M}>")
    device = 'cuda'
    x = torch.randn(M, K_DIM, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(N_TOTAL, K_DIM, device=device, dtype=torch.bfloat16) * 0.01
    rms_w = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    for _ in range(iters):
        module.wq_b_proj_gemm(x, w, rms_w, RMS_EPS)
    torch.cuda.synchronize()


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

    # Clean profiling fast-path: `WQB_PROFILE_M=32 ... ncu ... python test.py`
    # Runs only a warm launch loop of that M so the ncu capture is isolated & fast.
    prof_m = os.environ.get('WQB_PROFILE_M')
    if prof_m:
        M = int(prof_m)
        assert 32 <= M <= 256 and M % 32 == 0, f"WQB_PROFILE_M must be a multiple of 32 in [32,256], got {M}"
        profile_loop(module, M, iters=int(os.environ.get('WQB_PROFILE_ITERS', '40')))
        sys.exit(0)

    # Correctness sweep: every 32-aligned M in [32, 256].
    all_pass = True
    for M in range(32, 257, 32):
        all_pass &= test_correctness(module, M=M)

    benchmark(module)

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
