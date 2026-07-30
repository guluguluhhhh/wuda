"""
Test & Benchmark: standalone indexer-q post-processing (idx_post_fp4.cu).

  iq_f32[M,64,128] -> round bf16 -> rope(tail 64) -> hadamard-128 (* 128^-1/2)
  -> per-32 MXFP4 -> iq_fp4[M,64,64] i8 + iq_sf[M,64] i32

Correctness reference: the torch chain from test_wq_b_fp8_gemm (single ref
implementation shared with the fused-path tests). Requires sm_100+ (uses the
hardware cvt.rn.satfinite.e2m1x2 fp4 cast).
"""
import os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim
from test_wq_b_fp8_gemm import (       # shared torch reference + helpers
    ref_idx_chain, dequant_kernel_iq, make_rope_tables)


def load_module():
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    assert sm >= 100, f'e2m1x2 fp4 cast requires sm_100+, got sm_{sm}'

    cuda_flags = [
        '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        f'-gencode=arch=compute_{sm}a,code=sm_{sm}a',
    ]
    return load(
        name='idx_post_fp4',
        sources=[os.path.join(proj_dir, 'kernels', 'idx_post_fp4.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include')],
        extra_cuda_cflags=cuda_flags,
        verbose=True,
    )


def test_correctness(module, M):
    """Quiet on PASS: dequant(kernel outputs) vs the shared torch ref chain
    (same fp32 input -> same rounding order; fp4 tie-breaking is the only
    sub-ulp divergence source, absorbed by the cos/sf thresholds)."""
    dev = 'cuda'
    torch.manual_seed(M + 29)
    iq_f32 = (torch.randn(M, 64, 128, device=dev, dtype=torch.float32) * 0.5)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)

    iq_fp4, iq_sf = module.idx_postprocess(iq_f32, q_pos, cos_tab, sin_tab)

    ref_deq, ref_sf = ref_idx_chain(iq_f32.view(M, -1), q_pos.long(), cos_tab, sin_tab)
    ker_deq, ker_scale = dequant_kernel_iq(iq_fp4, iq_sf)

    cos = F.cosine_similarity(ker_deq.flatten(), ref_deq.flatten(), dim=0).item()
    sf_match = (ker_scale == ref_sf).float().mean().item()
    ok = cos > 0.999 and sf_match > 0.98
    line = f"cos={cos:.6f} sf={sf_match*100:.2f}%"
    if ok:
        print(f"  [PASS] idx_post M={M:<4} {line}")
    else:
        print(f"  [FAIL] idx_post M={M}")
        print(f"    {line}")
        print(f"    max_diff: {(ker_deq - ref_deq).abs().max().item():.4e}")
    return ok


def benchmark(module):
    """Latency + effective bandwidth. Memory-bound op: bytes = fp32 in
    (M*8192*4) + fp4/sf out (M*64*68) + rope tables (amortized, ignored)."""
    print("\n" + "=" * 60)
    print("Benchmark: idx_post_fp4 (rope + hadamard-128 + MXFP4)")
    print("=" * 60)
    dev = 'cuda'
    cos_tab, sin_tab = make_rope_tables()
    print(f"  {'M':<6} {'us':<8} {'GB/s':<8}")
    print("  " + "-" * 24)
    for M in [1, 4, 16, 32, 64, 128, 256, 1024, 4096]:
        torch.cuda.empty_cache()   # canonical allocator state per cell
        iq_f32 = torch.randn(M, 64, 128, device=dev, dtype=torch.float32)
        q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
        us = 1e6 * bench_kineto(
            lambda: module.idx_postprocess(iq_f32, q_pos, cos_tab, sin_tab),
            'idx_post_kernel', suppress_kineto_output=True)
        bytes_ = M * 64 * 128 * 4 + M * 64 * 68
        print(f"  {M:<6} {us:<8.2f} {bytes_ / (us * 1e-6) / 1e9:<8.1f}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")
    if sm < 100:
        print(f"ERROR: requires sm_100+ (Blackwell), got sm_{sm}")
        sys.exit(1)

    module = load_module()

    print("\nCorrectness (arbitrary batch):")
    results = [test_correctness(module, M)
               for M in [1, 2, 4, 7, 16, 31, 32, 61, 64, 97, 128, 333, 1024]]

    benchmark(module)

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
