"""
Test & Benchmark: w1_merged_fp8_gemm (tcgen05 FP8 block-scale, 2SM swap-AB).
  x_fp8[M,7168] @ w1_fp8[4352,7168].T -> y_all_bf16[M,4352]
Requires: NVIDIA Blackwell (sm_100+), CUDA 12.8+, CUTLASS 3.x.

Scale-factor (SF) physical layout expected by the kernel (DeepGEMM 1D1D):
  - dtype int32, one uint32 packs 4 UE8M0 exponents (one per 32-K sub-block).
  - physical shape [sf_k, mn] with mn contiguous, sf_k = K/128 = 56.
    x_sf: [56, M] (per token), w1_sf: [56, N] (per weight row).
  - UE8M0 byte e encodes scale 2^(e-127); e=127 (0x7F) => scale 1.0.
"""
import os, sys, torch
import torch.nn.functional as F

K_DIM   = 7168
N_TOTAL = 4352
BLOCK_K = 128
GRAN_K  = 32
SF_K    = K_DIM // BLOCK_K          # 56
NUM_32K = K_DIM // GRAN_K           # 224 sub-blocks
UE8M0_ONE = 0x7F                    # exponent 127 -> 2^0 = 1.0


def forced_bm(M):
    return 128 if M == 256 else M


def splitk_config(M):
    bm = forced_bm(M)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    physical_grid = (sms // 2) * 2
    mn_blocks = ((M + bm - 1) // bm) * (N_TOTAL // 128)
    target = max(physical_grid // mn_blocks, 1)
    k_tiles_per_split = (SF_K + target - 1) // target
    split_k = (SF_K + k_tiles_per_split - 1) // k_tiles_per_split
    return bm, split_k, k_tiles_per_split


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
        name='w1_merged_fp8_gemm',
        sources=[os.path.join(proj_dir, 'kernels', 'w1_merged_fp8_gemm.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'), cutlass_dir, cutlass_tools_dir],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],
        verbose=True,
    )


def make_sf_ones(mn, device):
    """SF tensor [SF_K, mn] int32, all UE8M0 = 127 (scale 1.0)."""
    packed = (UE8M0_ONE
              | (UE8M0_ONE << 8)
              | (UE8M0_ONE << 16)
              | (UE8M0_ONE << 24))
    return torch.full((SF_K, mn), packed, dtype=torch.int32, device=device)


def dequant_reference(x_fp8, w_fp8):
    """Scale = 1 reference: plain fp8 -> fp32 matmul."""
    return x_fp8.float() @ w_fp8.float().t()   # [M, N]


def pack_sf(exps):
    """Pack per-32-K UE8M0 exponents into DeepGEMM MN-major layout [SF_K, mn] int32.
    exps: [mn, NUM_32K] uint8; sub-block s=kb*4+j -> byte j of uint32[kb, mn]."""
    mn = exps.shape[0]
    e = exps.view(mn, SF_K, 4).to(torch.int64)
    packed = e[..., 0] | (e[..., 1] << 8) | (e[..., 2] << 16) | (e[..., 3] << 24)  # [mn, SF_K]
    return packed.t().contiguous().to(torch.int32)                                  # [SF_K, mn]


def dequant(fp8, exps):
    """fp8 [mn,K], exps [mn,NUM_32K] -> float [mn,K] with per-32-K UE8M0 scale."""
    scale = torch.pow(2.0, exps.float() - 127.0)              # [mn, NUM_32K]
    return fp8.float() * scale.repeat_interleave(GRAN_K, dim=1)


def test_correctness_scaled(module, M):
    """Non-trivial per-32-K scales: exposes SFA/SFB swap, sf_id order, packed layout."""
    print("=" * 60)
    print(f"Correctness (per-32K scale): w1_merged_fp8_gemm (M={M})")
    print("=" * 60)
    dev = 'cuda'
    torch.manual_seed(M)
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    # exponents around 127 (scale in [2^-2, 2^2]); distinct per row & sub-block.
    ea = torch.randint(125, 130, (M, NUM_32K), device=dev, dtype=torch.uint8)
    eb = torch.randint(125, 130, (N_TOTAL, NUM_32K), device=dev, dtype=torch.uint8)
    x_sf = pack_sf(ea); w_sf = pack_sf(eb)

    out = module.w1_merged_fp8_gemm(x, x_sf, w, w_sf)
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


def test_correctness(module, M):
    print("=" * 60)
    print(f"Correctness (scale=1): w1_merged_fp8_gemm (M={M})")
    print("=" * 60)
    dev = 'cuda'
    # Small values so e4m3 quantization is near-exact.
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    x_sf = make_sf_ones(M, dev)
    w_sf = make_sf_ones(N_TOTAL, dev)

    out = module.w1_merged_fp8_gemm(x, x_sf, w, w_sf)   # [M, N] bf16
    ref = dequant_reference(x, w)

    assert out.shape == (M, N_TOTAL), f"shape {out.shape}"
    diff = (out.float() - ref).abs()
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    print(f"  cos_sim:  {cos:.6f}")
    print(f"  max_diff: {diff.max().item():.4e}")
    print(f"  mean_diff:{diff.mean().item():.4e}")
    ok = cos > 0.99
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    return ok


def time_cuda_us(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000


def _recoverable_cublas_error(err):
    msg = str(err).lower()
    if "unspecified launch failure" in msg or "illegal memory access" in msg:
        return False
    return any(x in msg for x in (
        "not implemented", "not support", "not supported", "unsupported",
        "expected", "invalid", "argument", "missing", "unexpected",
        "cublas_status_not_supported",
    ))


def make_cublas_fp8_runner(x, w_t):
    """Return a cuBLASLt FP8 GEMM runner for x[M,K] @ w_t[K,N]."""
    one = torch.ones((), device=x.device, dtype=torch.float32)
    errors = []

    candidates = []
    if hasattr(torch, "_scaled_mm"):
        candidates.append((
            "_scaled_mm_kw",
            lambda: torch._scaled_mm(x, w_t, scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
        ))
        candidates.append((
            "_scaled_mm_pos",
            lambda: torch._scaled_mm(x, w_t, one, one, out_dtype=torch.bfloat16),
        ))
    candidates.append(("torch.mm", lambda: torch.mm(x, w_t)))

    for name, fn in candidates:
        try:
            out = fn()
            if isinstance(out, tuple):
                out = out[0]
            torch.cuda.synchronize()

            def runner(fn=fn):
                out = fn()
                return out[0] if isinstance(out, tuple) else out

            return runner, out.dtype, name
        except TypeError as err:
            errors.append(f"{name}: {err}")
        except RuntimeError as err:
            if not _recoverable_cublas_error(err):
                raise
            errors.append(f"{name}: {err}")

    raise RuntimeError("no usable FP8 cuBLASLt baseline; last errors: " + " | ".join(errors[-3:]))


def benchmark(module):
    print("\n" + "=" * 60)
    print("Benchmark: w1_merged_fp8_gemm latency sweep")
    print("=" * 60)
    dev = 'cuda'
    w = (torch.randn(N_TOTAL, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w_sf = make_sf_ones(N_TOTAL, dev)
    w_t = w.t()

    weight_bytes = N_TOTAL * K_DIM              # fp8
    weight_sf_bytes = SF_K * N_TOTAL * 4        # int32 packed UE8M0

    print(f"  K={K_DIM}, N={N_TOTAL}, SF_K={SF_K}")
    print(f"  Weight: {weight_bytes/1e6:.1f} MB fp8, weight SF: {weight_sf_bytes/1e6:.1f} MB int32")
    print("  NOTE: %cuBLAS is latency-based (cuBLAS_us/ours_us).")
    print("        ours uses a preallocated output buffer.")
    print("        split-K uses TMA reduce-add into BF16 output after zeroing it.")
    print("        ours_BW counts fp8 A/B + int32 SF + bf16 output; cuBLAS_BW counts fp8 A/B + its output.")
    print(f"  {'M':<5} {'BM':<5} {'splitK':<7} {'Kt/s':<5} {'ours(us)':<10} {'cuBLAS(us)':<11} {'ours_BW':<10} {'cuBLAS_BW':<11} {'TFLOPS':<9} {'%cuBLAS':<8} {'cuBLAS':<13}")
    print("  " + "-" * 108)

    for M in [32, 64, 96, 128, 160, 192, 224, 256]:
        bm, split_k, k_tiles_per_split = splitk_config(M)
        x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
        x_sf = make_sf_ones(M, dev)

        out = torch.empty((M, N_TOTAL), device=dev, dtype=torch.bfloat16)
        ours_fn = lambda: module.w1_merged_fp8_gemm_out(x, x_sf, w, w_sf, out)
        ours_us = time_cuda_us(ours_fn)

        try:
            cublas_fn, cublas_dtype, cublas_name = make_cublas_fp8_runner(x, w_t)
            cublas_us = time_cuda_us(cublas_fn)
            cublas_out_bytes = M * N_TOTAL * torch.empty((), dtype=cublas_dtype).element_size()
            cublas_bytes = weight_bytes + M * K_DIM + cublas_out_bytes
            cublas_bw = cublas_bytes / (cublas_us * 1e-6) / 1e9
            pct_cublas = cublas_us / ours_us * 100.0
            cublas_us_s = f"{cublas_us:<11.1f}"
            cublas_bw_s = f"{cublas_bw:<11.1f}"
            pct_s = f"{pct_cublas:<7.1f}%"
        except RuntimeError as err:
            cublas_name = "unavailable"
            cublas_us_s = f"{'N/A':<11}"
            cublas_bw_s = f"{'N/A':<11}"
            pct_s = f"{'N/A':<8}"
            print(f"  cuBLAS baseline unavailable for M={M}: {err}")

        tflops = 2 * M * N_TOTAL * K_DIM / (ours_us * 1e-6) / 1e12
        ours_bytes = weight_bytes + M * K_DIM + weight_sf_bytes + SF_K * M * 4 + M * N_TOTAL * 2
        ours_bw = ours_bytes / (ours_us * 1e-6) / 1e9

        print(f"  {M:<5} {bm:<5} {split_k:<7} {k_tiles_per_split:<5} {ours_us:<10.1f} {cublas_us_s} {ours_bw:<10.1f} {cublas_bw_s} {tflops:<9.1f} {pct_s} {cublas_name:<13}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")
    if sm < 100:
        print("ERROR: requires sm_100+ (Blackwell)"); sys.exit(1)

    module = load_module()
    results = []
    for M in [32, 64, 96, 128, 160, 192, 224, 256]:
        results.append(test_correctness(module, M))
    for M in [32, 96, 128, 160, 192, 224, 256]:
        results.append(test_correctness_scaled(module, M))
    benchmark(module)
    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    sys.exit(0 if all(results) else 1)
