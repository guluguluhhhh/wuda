"""
Test & Benchmark: front_mixed_gemm (mixed-precision front projection).

  y[M,4672] = x[M,7168] @ [ w_fp8[2048,7168] | w_bf16[2624,7168] ]^T -> bf16
  fp8 segment: E4M3 weights + 128x128 UE8M0 scales, activation 1x128 UE8M0
  (checkpoint-accurate: wq_a + wkv are FP8, compressors/weights_proj are BF16).

Reordered output layout (all 64-aligned):
  [0,1536) wq_a | [1536,2048) wkv | [2048,4096) main_comp
  | [4096,4608) idx_comp | [4608,4672) w_proj

Correctness: fp8 segment vs fp32-dequant reference; bf16 segment vs torch.mm
(both accumulation-order tolerance via DeepGEMM calc_diff). Benchmark:
bench_kineto (8GB L2 flush, kineto kernel time) vs cuBLAS all-BF16 merge and
the cuBLAS fp8 upper bound.
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto

K = 7168
N = 4672
N_FP8 = 2048
SF_K = K // 128          # 56


def load_module():
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, '..', 'cutlass', 'include')
    cutlass_tools_dir = os.path.join(proj_dir, '..', 'cutlass', 'tools', 'util', 'include')

    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    assert sm >= 100, f'tcgen05 requires sm_100+, got sm_{sm}'

    cuda_flags = [
        '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
        '-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1',
        # TMEM / F16F32 TCGEN05 enables are auto-defined by cute/arch/config.hpp
        # for sm_10xa; passing them again -> "redefined" warnings.
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
        f'-gencode=arch=compute_{sm}a,code=sm_{sm}a',
    ]
    return load(
        name='front_mixed_gemm',
        sources=[os.path.join(proj_dir, 'kernels', 'front_mixed_gemm.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'),
                             cutlass_dir, cutlass_tools_dir],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],
        verbose=True,
    )


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _ue8m0_ceil_exp(amax_over_max):
    """fp32 -> UE8M0 exponent byte (ceil to the next power of two), clamp 1..254."""
    bits = amax_over_max.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    return exp.clamp(1, 254).to(torch.uint8)


def quant_act_fp8(x):
    """x [M,K] bf16 -> (x_fp8 e4m3, x_sf u8 [M,56]); 1x128 UE8M0 (pow2 ceil)."""
    m = x.size(0)
    v = x.float().view(m, SF_K, 128)
    amax = v.abs().amax(-1).clamp_min(1e-4)
    exp = _ue8m0_ceil_exp(amax / 448.0)                       # [M,56]
    scale = torch.pow(2.0, exp.float() - 127.0)
    q = (v / scale.unsqueeze(-1)).clamp(-448, 448).view(m, K)
    return q.to(torch.float8_e4m3fn).contiguous(), exp.contiguous()


def quant_weight_fp8(w):
    """w [Nf,K] bf16 -> (w_fp8 e4m3, w_sf u8 [Nf/128,56]); 128x128 UE8M0."""
    nf = w.size(0)
    v = w.float().view(nf // 128, 128, SF_K, 128)
    amax = v.abs().amax(dim=(1, 3)).clamp_min(1e-4)           # [Nf/128, 56]
    exp = _ue8m0_ceil_exp(amax / 448.0)
    scale = torch.pow(2.0, exp.float() - 127.0)
    q = (v / scale[:, None, :, None]).clamp(-448, 448).view(nf, K)
    return q.to(torch.float8_e4m3fn).contiguous(), exp.contiguous()


def dequant_fp8(q, exp, act):
    """back to fp32 with the same scales (reference input)."""
    scale = torch.pow(2.0, exp.float() - 127.0)
    if act:
        m = q.size(0)
        return (q.float().view(m, SF_K, 128) * scale.view(m, SF_K, 1)).view(m, K)
    nf = q.size(0)
    return (q.float().view(nf // 128, 128, SF_K, 128) *
            scale.view(nf // 128, 1, SF_K, 1)).view(nf, K)


def make_inputs(m, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(m, K, device='cuda', dtype=torch.bfloat16) * 0.1)
    w_bf16 = (torch.randn(N - N_FP8, K, device='cuda', dtype=torch.bfloat16) * 0.01)
    w_full = (torch.randn(N_FP8, K, device='cuda', dtype=torch.bfloat16) * 0.01)
    x_fp8, x_sf = quant_act_fp8(x)
    w_fp8, w_sf = quant_weight_fp8(w_full)
    return x, x_fp8, x_sf, w_bf16, w_fp8, w_sf


def hc_ref(mix, base, scale, eps=1e-6):
    """Reference for the TC/CC MHC tail: post gate + 20-iter Sinkhorn comb."""
    post = 2.0 * torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8])
    v = (mix[:, 8:24] * scale[2] + base[8:24]).view(-1, 4, 4)
    v = torch.softmax(v, dim=-1) + eps                     # row softmax (+eps)
    v = v / (v.sum(dim=-2, keepdim=True) + eps)            # col norm
    for _ in range(19):
        v = v / (v.sum(dim=-1, keepdim=True) + eps)        # row norm
        v = v / (v.sum(dim=-2, keepdim=True) + eps)        # col norm
    return post, v.reshape(-1, 16)


def make_hc(m, seed=1):
    g = torch.Generator(device='cuda').manual_seed(seed)
    t = {
        'hc_mix': torch.randn(m, 24, device='cuda', dtype=torch.float32, generator=g),
        'hc_base': torch.randn(24, device='cuda', dtype=torch.float32, generator=g),
        'hc_scale': torch.rand(3, device='cuda', dtype=torch.float32, generator=g) + 0.5,
        'hc_post': torch.empty(m, 4, device='cuda', dtype=torch.float32),
        'hc_comb': torch.empty(m, 16, device='cuda', dtype=torch.float32),
    }
    return t


def test_correctness(module, m):
    x, x_fp8, x_sf, w_bf16, w_fp8, w_sf = make_inputs(m, seed=m)
    hc = make_hc(m, seed=m + 7)
    # x arrives ALREADY attn_norm'ed (the MHC collapse fuses the full norm);
    # q/kv ssq now live in the wq_b quant prologue (deterministic warp order).
    out = module.front_mixed_gemm(x, x_fp8, x_sf, w_bf16, w_fp8, w_sf, **hc)
    torch.cuda.synchronize()
    assert out.shape == (m, N)

    # fp8 segment: reference = fp32 matmul on the SAME dequantized values
    # (difference is accumulation order only)
    ref8 = dequant_fp8(x_fp8, x_sf, act=True) @ dequant_fp8(w_fp8, w_sf, act=False).t()
    d8 = calc_diff(out[:, :N_FP8].float(), ref8)
    # bf16 segment: reference = cuBLAS bf16 (fp32 accumulate)
    ref16 = torch.mm(x, w_bf16.t()).float()
    d16 = calc_diff(out[:, N_FP8:].float(), ref16)

    # TC/CC MHC tail: post gate + Sinkhorn comb vs torch reference (__expf is
    # an approximate exp -> rtol, not bitwise)
    post_ref, comb_ref = hc_ref(hc['hc_mix'], hc['hc_base'], hc['hc_scale'])
    hc_ok = (torch.allclose(hc['hc_post'], post_ref, rtol=1e-4, atol=1e-6) and
             torch.allclose(hc['hc_comb'], comb_ref, rtol=1e-4, atol=1e-6))

    ok = d8 < 1e-5 and d16 < 1e-5 and hc_ok
    if ok:
        print(f"  M={m:<4} PASS")
    else:
        print(f"  M={m:<4} FAIL  fp8_diff={d8:.3e}  bf16_diff={d16:.3e}  "
              f"hc={'OK' if hc_ok else 'MISMATCH'}")
    return ok


def benchmark(module):
    print("\n" + "=" * 64)
    print("Benchmark: front_mixed_gemm vs cuBLAS (bench_kineto, 8GB L2 flush)")
    print("=" * 64)
    mixed_bytes = N_FP8 * K + (N_FP8 // 128) * SF_K + (N - N_FP8) * K * 2
    bf16_bytes = N * K * 2
    print(f"  weight bytes: mixed {mixed_bytes / 1e6:.1f} MB vs all-bf16 "
          f"{bf16_bytes / 1e6:.1f} MB (x{bf16_bytes / mixed_bytes:.2f})")
    print("  columns: mixed = bare GEMM | +hc = +MHC post/comb tail "
          "(x arrives pre-normed; q/kv ssq live in the wq_b quant prologue)")
    print(f"  {'M':<5} {'mixed(us)':<10} {'+hc(us)':<9} "
          f"{'cb_bf16(us)':<12} {'cb_fp8(us)':<11} {'mixed_BW':<9} {'x_bf16':<7}")
    print("  " + "-" * 70)

    one = torch.ones((), device='cuda', dtype=torch.float32)
    for m in (2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128):
        torch.cuda.empty_cache()   # canonical allocator state per cell
        x, x_fp8, x_sf, w_bf16, w_fp8, w_sf = make_inputs(m, seed=m)
        # production contract: M<16 runs 16-row padded; the mixed column IS
        # the padded number. Root cause (exp_tma_tiny_rows.py + exp_tma_micro.cu):
        # NOT a TMA-unit descriptor slow path (grid=1 is flat at any desc rows)
        # but an L2 hotspot -- all 146 CTAs replicate reads of a tiny in-bounds
        # A footprint (<10 rows x 14KB), serializing on few L2 slices and
        # doubling load completion latency (~1.3x kernel time). Measured
        # recovery threshold is desc rows >= 10; 16 keeps margin.
        mp = max(m, 16)
        if mp != m:
            xm = torch.zeros(mp, K, device='cuda', dtype=torch.bfloat16)
            xm[:m] = x
            x_fp8, x_sf = quant_act_fp8(xm)
        else:
            xm = x
        # merged all-bf16 twin for the cuBLAS baseline
        w_all = torch.cat([dequant_fp8(w_fp8, w_sf, act=False).bfloat16(), w_bf16])
        w_all_fp8 = w_all.to(torch.float8_e4m3fn)
        out = torch.empty(mp, N, device='cuda', dtype=torch.bfloat16)
        cb_out = torch.empty(m, N, device='cuda', dtype=torch.bfloat16)

        # correctness gate BEFORE timing (never report perf on wrong results)
        got = module.front_mixed_gemm(xm, x_fp8, x_sf, w_bf16, w_fp8, w_sf,
                                      hc_mix=None, hc_base=None, hc_scale=None,
                                      hc_post=None, hc_comb=None, out=out)
        d8 = calc_diff(got[:m, :N_FP8].float(),
                       dequant_fp8(x_fp8, x_sf, True)[:m] @ dequant_fp8(w_fp8, w_sf, False).t())
        d16 = calc_diff(got[:m, N_FP8:].float(), torch.mm(x, w_bf16.t()).float())
        if d8 >= 1e-5 or d16 >= 1e-5:
            raise AssertionError(f"M={m}: diff8={d8:.3e} diff16={d16:.3e}")

        mixed = 1e6 * bench_kineto(
            lambda: module.front_mixed_gemm(xm, x_fp8, x_sf, w_bf16, w_fp8, w_sf,
                                            hc_mix=None, hc_base=None, hc_scale=None,
                                            hc_post=None, hc_comb=None, out=out),
            'front_mixed_kernel', suppress_kineto_output=True)
        # +hc: MHC post/comb tail on (production form)
        hc = make_hc(mp, seed=m + 7)
        wssq = 1e6 * bench_kineto(
            lambda: module.front_mixed_gemm(xm, x_fp8, x_sf, w_bf16, w_fp8, w_sf,
                                            out=out, **hc),
            'front_mixed_kernel', suppress_kineto_output=True)
        cb = 1e6 * sum(bench_kineto(
            lambda: torch.mm(x, w_all.t(), out=cb_out),
            ('nvjet', 'reduce'), suppress_kineto_output=True))
        if mixed <= 0 or cb <= 0:
            raise RuntimeError(
                f"M={m}: profiler missed expected kernels "
                f"(mixed={mixed}, cublas={cb}) -- kernel name drift?")
        try:
            cb8 = 1e6 * sum(bench_kineto(
                lambda: torch._scaled_mm(x_fp8, w_all_fp8.t(), scale_a=one,
                                         scale_b=one, out_dtype=torch.bfloat16),
                ('nvjet', 'reduce'), suppress_kineto_output=True))
            cb8_s = f"{cb8:<11.1f}"
        except Exception:
            cb8_s = f"{'n/a':<11}"

        io_bytes = mixed_bytes + mp * (K * 2 + K + SF_K) + mp * N * 2
        bw = io_bytes / (mixed * 1e-6) / 1e9
        print(f"  {m:<5} {mixed:<10.1f} {wssq:<9.1f} {cb:<12.1f} "
              f"{cb8_s} {bw:<9.1f} {cb / mixed:<7.2f}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}  "
          f"torch={torch.__version__} CUDA={torch.version.cuda}")
    module = load_module()

    print("Correctness (fp8 seg vs dequant ref, bf16 seg vs torch.mm; "
          "calc_diff < 1e-5):")
    results = []
    for m in (1, 2, 4, 8, 16, 32, 64, 96, 100, 128):   # incl. non-16-aligned M
        results.append(test_correctness(module, m))

    benchmark(module)

    print("\n" + "=" * 64)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print("=" * 64)
    sys.exit(0 if all(results) else 1)
