"""
Test: CUTLASS SM100 Block-Scaled FP8 GEMM
Compile with sm_103a to enable TCGEN05 instructions
"""
import os, sys, torch

D = 7168
N_QA = 1536

def load_module():
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, '..', 'cutlass', 'include')
    cutlass_util_dir = os.path.join(proj_dir, '..', 'cutlass', 'tools', 'util', 'include')

    cuda_flags = [
        '-O3', '-std=c++17',
        '--expt-relaxed-constexpr',
        '-lineinfo',
        # SM103a: enables TCGEN05 (TMEM + block-scaled FP8 MMA)
        '-arch=sm_103a',
        # CUTLASS needs these
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
    ]

    return load(
        name='qa_kv_proj',
        sources=[os.path.join(proj_dir, 'kernels', 'qa_kv_proj_gemm.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'), cutlass_dir, cutlass_util_dir],
        extra_cuda_cflags=cuda_flags, verbose=True,
    )


def test_compile():
    """Just test that the module compiles and loads."""
    print("=" * 60)
    print("Compile Test: CUTLASS SM100 FP8 GEMM")
    print("=" * 60)
    module = load_module()
    print("Compile SUCCESS!")
    return module


def test_basic(module):
    """Basic correctness test with random FP8 data."""
    print("\n" + "=" * 60)
    print("Basic Test: FP8 GEMM correctness")
    print("=" * 60)

    M, K, N = 128, 7168, 1536
    device = 'cuda'

    # Generate random FP8 data
    A_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1
    B_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.01

    A_fp8 = A_bf16.to(torch.float8_e4m3fn)
    B_fp8 = B_bf16.to(torch.float8_e4m3fn)

    # Merge weights: cat([wq_a, wkv], dim=0) → [2048, K]
    B_merged = torch.cat([B_fp8, torch.randn(512, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)], dim=0)
    N = B_merged.size(0)  # 1536 + 512 = 2048
    N_QA = 1536

    # Scale factors
    sf_size_A = M * ((K + 31) // 32)
    sf_size_B = N * ((K + 31) // 32)
    A_scale = torch.ones(sf_size_A, device=device, dtype=torch.float8_e8m0fnu)
    B_scale = torch.ones(sf_size_B, device=device, dtype=torch.float8_e8m0fnu)

    # Run CUTLASS merged GEMM
    qr, kv = module.qa_kv_proj_gemm(A_fp8, A_scale, B_merged, B_scale, N_QA)
    print(f"  qr shape: {qr.shape}, kv shape: {kv.shape}")
    print(f"  qr norm: {qr.float().norm().item():.4f}")
    print(f"  kv norm: {kv.float().norm().item():.4f}")
    print(f"  qr[:3,:3]: {qr[:3,:3]}")

    # Reference
    A_deq = A_fp8.to(torch.bfloat16)
    B_deq = B_merged.to(torch.bfloat16)
    C_ref = torch.mm(A_deq, B_deq.t())
    qr_ref = C_ref[:, :N_QA]
    kv_ref = C_ref[:, N_QA:]

    # Compare
    diff_qr = (qr.float() - qr_ref.float()).abs()
    diff_kv = (kv.float() - kv_ref.float()).abs()
    cos_qr = torch.nn.functional.cosine_similarity(
        qr.float().flatten(), qr_ref.float().flatten(), dim=0).item()
    cos_kv = torch.nn.functional.cosine_similarity(
        kv.float().flatten(), kv_ref.float().flatten(), dim=0).item()
    print(f"  qr max_err: {diff_qr.max().item():.6e}, cos_sim: {cos_qr:.8f}")
    print(f"  kv max_err: {diff_kv.max().item():.6e}, cos_sim: {cos_kv:.8f}")
    passed = cos_qr > 0.999 and cos_kv > 0.999
    print(f"  Result: {'PASS' if passed else 'FAIL'}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute: sm_{cap[0]*10+cap[1]}")

    module = test_compile()
    test_basic(module)

    # Profile: measure kernel-only latency (excluding Python overhead)
    print("\n" + "=" * 60)
    print("Profile: FP8 GEMM kernel timing")
    print("=" * 60)
    device = 'cuda'
    M, K, N = 128, D, N_QA + 512
    A_fp8 = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B_fp8 = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    sf_A = M * ((K + 31) // 32)
    sf_B = N * ((K + 31) // 32)
    A_scale = torch.ones(sf_A, device=device, dtype=torch.float8_e8m0fnu)
    B_scale = torch.ones(sf_B, device=device, dtype=torch.float8_e8m0fnu)

    # Warmup
    for _ in range(50):
        module.qa_kv_proj_gemm(A_fp8, A_scale, B_fp8, B_scale, N_QA)
    torch.cuda.synchronize()

    # Single call profiling with events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    qr, kv = module.qa_kv_proj_gemm(A_fp8, A_scale, B_fp8, B_scale, N_QA)
    end_evt.record()
    torch.cuda.synchronize()
    single_us = start_evt.elapsed_time(end_evt) * 1000

    # Amortized (100 calls)
    start_evt.record()
    for _ in range(100):
        module.qa_kv_proj_gemm(A_fp8, A_scale, B_fp8, B_scale, N_QA)
    end_evt.record()
    torch.cuda.synchronize()
    avg_us = start_evt.elapsed_time(end_evt) / 100 * 1000

    flops = 2 * M * N * K
    tflops = flops / (avg_us * 1e-6) / 1e12
    weight_bytes = N * K * 1  # FP8 weight
    print(f"  Problem: M={M}, N={N}, K={K}")
    print(f"  Single call:   {single_us:.1f} us")
    print(f"  Amortized:     {avg_us:.1f} us")
    print(f"  Host overhead: ~{single_us - avg_us:.1f} us (launch + python)")
    print(f"  TFLOPS:        {tflops:.2f}")
    print(f"  Weight:        {weight_bytes/1e6:.1f} MB (FP8)")
    print(f"  Arithmetic intensity: {flops / weight_bytes:.1f} FLOP/Byte")
    print(f"  Compute/BW bound:  {'Compute' if flops / weight_bytes > 500 else 'Bandwidth'}")

    # Quick benchmark
    print("\n" + "=" * 60)
    print("Benchmark: FP8 GEMM latency sweep")
    print("=" * 60)
    device = 'cuda'
    K, N = D, N_QA + 512  # merged N = 2048
    batch_sizes = [128, 256, 512, 1024, 2048, 4096]

    print(f"  K={K}, N={N} (merged wq_a[1536] + wkv[512])")
    print(f"  {'M':<8} {'Latency(us)':<12} {'TFLOPS':<10} {'BW(GB/s)':<10} {'HBM%':<8}")
    print("  " + "-" * 55)

    HBM_PEAK = 8000  # GB/s

    for M in batch_sizes:
        A_fp8 = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        B_fp8 = torch.randn(N, K, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sf_A = M * ((K + 31) // 32)
        sf_B = N * ((K + 31) // 32)
        A_scale = torch.ones(sf_A, device=device, dtype=torch.float8_e8m0fnu)
        B_scale = torch.ones(sf_B, device=device, dtype=torch.float8_e8m0fnu)

        # Warmup
        for _ in range(20):
            module.qa_kv_proj_gemm(A_fp8, A_scale, B_fp8, B_scale, N_QA)
        torch.cuda.synchronize()

        # Benchmark
        iters = 100
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.qa_kv_proj_gemm(A_fp8, A_scale, B_fp8, B_scale, N_QA)
        end.record()
        torch.cuda.synchronize()
        elapsed_us = start.elapsed_time(end) / iters * 1000

        flops = 2 * M * N * K
        tflops = flops / (elapsed_us * 1e-6) / 1e12
        bytes_total = (M * K + N * K) * 1 + M * N * 2  # FP8 A/B + BF16 D
        bw_gbs = bytes_total / (elapsed_us * 1e-6) / 1e9
        util = bw_gbs / HBM_PEAK * 100
        print(f"  {M:<8} {elapsed_us:<12.1f} {tflops:<10.2f} {bw_gbs:<10.1f} {util:<8.1f}%")
