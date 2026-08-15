"""
Test & Benchmark: MERGED wq_b projection (tcgen05 FP8 block-scale, swap-AB,
M<=128) with the sparse-indexer projection fused in. Direct execution defaults
to the full-rank shape:

  x_fp8[M,1536] @ w_fp8[73728,1536].T
    -> y[M,65536] BF16                         (128 main-Q heads)
    -> iq_fp4[M,64,64] i8 + iq_sf[M,64] i32   (64 index-Q heads:
       rope + hadamard-128 + per-32 MXFP4, run by the async xform warpgroup)

Use --tpdp for main-Q TP2 with all 64 index-Q heads replicated; token DP is
outside this single-rank benchmark.

Requires: NVIDIA Blackwell (sm_100+), CUDA 12.8+, CUTLASS 3.x.

Native DSV4 scale-factor layout expected by the kernel:
  - dtype float8_e8m0fnu (raw uint8 is also accepted).
  - x_sf: [M, K/128] = [M, 12], one scale per token/K128.
  - w_sf: [N/128, K/128], one scale per N128xK128 block.
  - UE8M0 byte e encodes scale 2^(e-127); e=127 (0x7F) => scale 1.0.
"""
import argparse
import os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

K_DIM   = 1536
FULL_N_TOTAL = 65536
FULL_N_IDX = 64 * 128
TPDP_N_TOTAL = 32768

# Importers retain full-rank defaults. Entry points call configure_geometry()
# before constructing tensors when they need the TPDP idx-replicated geometry.
N_TOTAL = FULL_N_TOTAL
N_IDX = FULL_N_IDX
N_MERGED = N_TOTAL + N_IDX
QUANT_BLOCK_K = 128
WEIGHT_QUANT_BLOCK_N = 128
SF_K    = K_DIM // QUANT_BLOCK_K          # 12
UE8M0_ONE = 0x7F                    # exponent 127 -> 2^0 = 1.0


def configure_geometry(tpdp=False):
    """Select full rank or TP2 main-Q with replicated index-Q."""
    global N_TOTAL, N_IDX, N_MERGED
    N_TOTAL = TPDP_N_TOTAL if tpdp else FULL_N_TOTAL
    N_IDX = FULL_N_IDX
    N_MERGED = N_TOTAL + N_IDX


def load_module(tpdp=False):
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
        # CUTE_ARCH_* gates are NOT hand-forced: sm103_cutlass_shim.h routes
        # sm_103a through CUTLASS's own arch table (no-op on CUTLASS >= 4.2).
        # The old -DCUTE_ARCH_TCGEN05_TMEM_ENABLED here is what produced the
        # "redefined" warning at cute/arch/config.hpp:137.
        '-include', os.path.join(proj_dir, 'include', 'sm103_cutlass_shim.h'),
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
        f'-gencode=arch=compute_{sm}a,code=sm_{sm}a',
    ]
    if tpdp:
        cuda_flags.append('-DWQ_B_TPDP=1')
    name = 'wq_b_fp8_gemm_tpdp' if tpdp else 'wq_b_fp8_gemm'
    return load(
        name=name,
        sources=[os.path.join(proj_dir, 'kernels', 'wq_b_fp8_gemm.cu')],
        extra_include_paths=[os.path.join(proj_dir, 'include'), cutlass_dir, cutlass_tools_dir],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=['-lcuda'],  # driver API for TMA
        verbose=True,
    )


# ==================== helpers ====================
def as_ue8m0(exps):
    """Reinterpret raw UE8M0 exponent bytes without numerical conversion."""
    assert hasattr(torch, 'float8_e8m0fnu')
    return exps.contiguous().view(torch.float8_e8m0fnu)


def make_act_sf_ones(m, device):
    raw = torch.full((m, SF_K), UE8M0_ONE, dtype=torch.uint8, device=device)
    return as_ue8m0(raw)


def make_weight_sf_ones(device):
    raw = torch.full((N_MERGED // WEIGHT_QUANT_BLOCK_N, SF_K), UE8M0_ONE,
                     dtype=torch.uint8, device=device)
    return as_ue8m0(raw)


def dequant_act(fp8, exps):
    scale = torch.pow(2.0, exps.float() - 127.0)
    return fp8.float() * scale.repeat_interleave(QUANT_BLOCK_K, dim=1)


def dequant_weight(fp8, exps):
    """Apply native N128xK128 weight scales to any row range."""
    scale = torch.pow(2.0, exps.float() - 127.0)
    scale = scale.repeat_interleave(WEIGHT_QUANT_BLOCK_N, dim=0)[:fp8.shape[0]]
    return fp8.float() * scale.repeat_interleave(QUANT_BLOCK_K, dim=1)


def _ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x):
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def _dequantize_from_fp4_e2m1(x):
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              device=x.device, dtype=torch.float)
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


_H128 = None


def hadamard_128(dev):
    """Sylvester H_128 == fast_hadamard_transform == the kernel's FWHT butterflies."""
    global _H128
    if _H128 is None:
        H = torch.ones(1, 1, device=dev, dtype=torch.float32)
        while H.shape[0] < 128:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H128 = H
    return _H128


def make_rope_tables(max_pos=4096):
    ang = torch.outer(torch.arange(max_pos, device="cuda", dtype=torch.float32),
                      1.0 / (10000.0 ** (torch.arange(32, device="cuda") / 32.0)))
    return torch.cos(ang).contiguous(), torch.sin(ang).contiguous()


def ref_idx_chain(iq_f32, pos, cos_tab, sin_tab):
    """Torch reference of the fused chain, kernel rounding order:
    round bf16 -> rope(tail 64, bf16 round) -> @H128 * 128^-0.5 (bf16 round)
    -> per-32 fp4 quant-dequant sim (scale = 2^ceil(log2(amax/6)), ue8m0)."""
    M = iq_f32.shape[0]
    num_heads = iq_f32.numel() // (M * 128)
    x = iq_f32.float().view(M, num_heads, 128).bfloat16().float()
    c = cos_tab[pos].unsqueeze(1)
    s = sin_tab[pos].unsqueeze(1)
    e = x[..., 64::2].clone()
    o = x[..., 65::2].clone()
    x[..., 64::2] = (e * c - o * s).bfloat16().float()
    x[..., 65::2] = (e * s + o * c).bfloat16().float()
    x = (x @ hadamard_128(x.device) * (128.0 ** -0.5)).bfloat16().float()
    blk = x.view(M, num_heads, 4, 32)
    sf = _ceil_to_ue8m0(blk.abs().amax(-1).clamp_min(6.0 * 2.0 ** -126) / 6.0)
    q = _quantize_to_fp4_e2m1(blk / sf.unsqueeze(-1))
    deq = _dequantize_from_fp4_e2m1(q) * sf.unsqueeze(-1)
    return deq.view(M, num_heads, 128), sf


def ref_idx_fp8_chain(iq_bf16, weights, pos, cos_tab, sin_tab):
    """RTP Indexer-Q: BF16 boundary -> RoPE -> per-head E4M3 + weight fold."""
    M, num_heads, _ = iq_bf16.shape
    x = iq_bf16.float()
    c = cos_tab[pos].unsqueeze(1)
    s = sin_tab[pos].unsqueeze(1)
    e, o = x[..., 64::2].clone(), x[..., 65::2].clone()
    x[..., 64::2] = (e * c - o * s).bfloat16().float()
    x[..., 65::2] = (e * s + o * c).bfloat16().float()
    scale = (x.abs().amax(-1) / 448.0).clamp_min(1e-12)
    q = (x / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q, weights * scale


def dequant_kernel_iq(iq_fp4, iq_sf):
    """Kernel outputs -> float: iq_fp4 [M,64,64] i8 packed (low nibble = even elem),
    iq_sf [M,64] i32 packed-ue8m0 (byte b = 32-col block b)."""
    M = iq_fp4.shape[0]
    num_heads = iq_fp4.shape[1]
    p = iq_fp4.view(M, num_heads, 64)
    un = torch.zeros(M, num_heads, 128, dtype=torch.int8, device=p.device)
    un[..., 0::2] = p & 0x0F
    un[..., 1::2] = (p >> 4) & 0x0F
    vals = _dequantize_from_fp4_e2m1(un)
    sf_b = iq_sf.contiguous().view(torch.uint8).view(M, num_heads, 4).to(torch.int)
    scale = torch.pow(2.0, sf_b.float() - 127.0)                                # [M,64,4]
    return vals * scale.repeat_interleave(32, dim=-1), scale


# ==================== correctness ====================
def test_merged(module, M):
    """One shot per M, quiet on PASS:
      A) BF16 main q vs the dequantized torch GEMM (non-trivial scales)
      B) dequant(iq_fp4, iq_sf) vs the torch reference chain
      C) SF words vs the reference power-of-2 scales
      D) standalone idx_post_kernel vs the same torch reference (same fp32 input)
      E) FP32-accumulator head_ssq is close to a BF16-output reference
      F) BITWISE async-handoff closed loop: the mock run exports the kernel's
         own drained bf16 iq; the standalone kernel over it (same
         idx_postprocess_row code, same input) must reproduce the fused
         iq_fp4/iq_sf bit-exactly -- catches drain/barrier/xform races that a
         cosine threshold would absorb"""
    dev = 'cuda'
    torch.manual_seed(M + 13)
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    ea = torch.randint(125, 130, (M, SF_K), device=dev, dtype=torch.uint8)
    eb = torch.randint(125, 130, (N_MERGED // WEIGHT_QUANT_BLOCK_N, SF_K),
                       device=dev, dtype=torch.uint8)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)

    ssq = torch.zeros(M, N_TOTAL // 512, device=dev, dtype=torch.float32)
    # mock_post=False: explicitly enable the fused post-processing (OFF by
    # default -- the default path is GEMM+ssq only with 256 threads)
    y, iq_fp4, iq_sf, _ = module.wq_b_proj_gemm_merged(
        x, as_ue8m0(ea), w, as_ue8m0(eb), q_pos, cos_tab, sin_tab,
        head_ssq=ssq, mock_post=False)
    dtype_ok = y.dtype == torch.bfloat16

    # A) main q vs torch ref -- the indexer weight LEADS w, main q follows
    xd = dequant_act(x, ea)
    ref_main = xd @ dequant_weight(w[N_IDX:], eb[N_IDX // 128:]).t()
    main_cos = F.cosine_similarity(y.float().flatten(), ref_main.flatten(), dim=0).item()

    # B/C) iq chain vs torch ref from the fp32 GEMM of the FIRST N_IDX weight rows
    iq_ref_f32 = xd @ dequant_weight(w[:N_IDX], eb[:N_IDX // 128]).t()          # [M, N_IDX]
    ref_deq, ref_sf = ref_idx_chain(iq_ref_f32, q_pos.long(), cos_tab, sin_tab)
    ker_deq, ker_scale = dequant_kernel_iq(iq_fp4, iq_sf)
    iq_cos = F.cosine_similarity(ker_deq.flatten(), ref_deq.flatten(), dim=0).item()
    sf_match = (ker_scale == ref_sf).float().mean().item()

    # D) standalone kernel, same chain, same fp32 input
    sa_fp4, sa_sf = module.idx_postprocess_standalone(
        iq_ref_f32.view(M, N_IDX // 128, 128).contiguous(), q_pos, cos_tab, sin_tab)
    sa_deq, _ = dequant_kernel_iq(sa_fp4, sa_sf)
    sa_cos = F.cosine_similarity(sa_deq.flatten(), ref_deq.flatten(), dim=0).item()

    # Fused SSQ uses pre-round FP32 accumulators; materialized y also includes
    # BF16 store rounding.
    ssq_ref = y.float().view(M, N_TOTAL // 512, 512).double().square().sum(-1).float()
    ssq_rel = ((ssq - ssq_ref).abs() / (ssq_ref + 1e-6)).max().item()

    # G) standalone head_ssq kernel over the same y
    ssq_sa = module.head_ssq_standalone(y)
    ssq_sa_rel = ((ssq_sa - ssq_ref).abs() / (ssq_ref + 1e-6)).max().item()

    # F) bitwise: fused outputs == standalone over the kernel's own fp32 iq
    #    (the mock GEMM is schedule-identical -> its drained scratch is bit-equal
    #    to what the fused run's xform warpgroup consumed)
    iq_ws = module.wq_b_proj_gemm_merged(
        x, as_ue8m0(ea), w, as_ue8m0(eb), q_pos, cos_tab, sin_tab,
        mock_post=True)[-1]
    bw_fp4, bw_sf = module.idx_postprocess_standalone(iq_ws, q_pos, cos_tab, sin_tab)
    bit_ok = torch.equal(bw_fp4, iq_fp4) and torch.equal(bw_sf, iq_sf)

    ok = (dtype_ok and main_cos > 0.99 and iq_cos > 0.99 and sf_match > 0.98
          and sa_cos > 0.999 and ssq_rel < 3e-3 and ssq_sa_rel < 1e-3 and bit_ok)
    line = (f"y_dtype={y.dtype} main_cos={main_cos:.6f} iq_cos={iq_cos:.6f} "
            f"sf={sf_match*100:.2f}% "
            f"sa_cos={sa_cos:.6f} ssq_rel={ssq_rel:.1e} ssq_sa_rel={ssq_sa_rel:.1e} "
            f"bitwise={'OK' if bit_ok else 'MISMATCH'}")
    if ok:
        print(f"  [PASS] merged M={M:<4} {line}")
    else:
        print(f"  [FAIL] merged M={M}")
        print(f"    {line}")
        print(f"    main max_diff: {(y.float() - ref_main).abs().max().item():.4e}")
        print(f"    iq   max_diff: {(ker_deq - ref_deq).abs().max().item():.4e}")
        if not bit_ok:
            nf = (bw_fp4 != iq_fp4).sum().item()
            ns = (bw_sf != iq_sf).sum().item()
            print(f"    bitwise mismatch: fp4 bytes {nf}, sf words {ns}")
    return ok


def test_indexer_q_fp8(module, M):
    """Fused WQ_B FP8 Q transform must match the standalone and Torch ABI."""
    dev = 'cuda'
    torch.manual_seed(M + 2026)
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(
        torch.float8_e4m3fn)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev,
                          dtype=torch.int32)
    weights = torch.randn(M, N_IDX // 128, device=dev)
    args = (x, make_act_sf_ones(M, dev), w, make_weight_sf_ones(dev),
            q_pos, cos_tab, sin_tab)

    fused = module.wq_b_proj_gemm_merged(
        *args, mock_post=False, enable_ssq=False,
        indexer_fp8=True, iq_weights=weights)
    iq_ws = module.wq_b_proj_gemm_merged(
        *args, mock_post=True, enable_ssq=False)[-1]
    sa_q, sa_w = module.idx_postprocess_fp8_standalone(
        iq_ws, weights, q_pos, cos_tab, sin_tab)
    ref_q, ref_w = ref_idx_fp8_chain(iq_ws, weights, q_pos.long(),
                                     cos_tab, sin_tab)

    fused_ok = (torch.equal(fused[1].view(torch.uint8), sa_q.view(torch.uint8))
                and torch.equal(fused[2], sa_w))
    q_match = (sa_q.view(torch.uint8) == ref_q.view(torch.uint8)).float().mean().item()
    w_rel = ((sa_w - ref_w).abs() / ref_w.abs().clamp_min(1e-8)).max().item()
    ok = fused_ok and q_match > 0.999 and w_rel < 1e-5
    print(f"  [{'PASS' if ok else 'FAIL'}] idx-q fp8 M={M:<4} "
          f"fused={'OK' if fused_ok else 'MISMATCH'} "
          f"q={q_match*100:.3f}% w_rel={w_rel:.2e}")
    return ok


# ==================== fused activation-quant prologue (isolation) ====================
def ref_quant_k128(q_y):
    """Torch twin of quant_k128_ue8m0 (plain 1x128, NO rmsnorm)."""
    t = q_y.float().view(-1, SF_K, 128)
    amax = t.abs().amax(-1).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (t * torch.pow(2.0, -e).unsqueeze(-1)).view(-1, K_DIM)
    return q.to(torch.float8_e4m3fn), (e + 127).to(torch.uint8)


def ref_qnorm_quant(q_y, gamma, eps=1e-6):
    """Torch twin of qnorm_quant_row_cta: rmsnorm on the bf16 row ->
    bf16-materialized round -> plain 1x128 quant. Block-partial ssq order
    differs in ulps from torch's row sum -> gate on byte match rate."""
    t = q_y.float()
    r = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
    qr = ((t * r) * gamma).bfloat16()
    return ref_quant_k128(qr)


def _quant_case(module, M, tag, gate_kw, ref_fn):
    """Shared harness: run the fused prologue twice, gate on bytes vs the
    torch twin + GEMM-consumed-fresh-quant cosine + run-to-run bitwise."""
    dev = 'cuda'
    torch.manual_seed(M + 31)
    q_y = (torch.randn(M, 4672, device=dev) * 0.1).bfloat16()[:, :K_DIM]
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    eb = torch.randint(125, 130, (N_MERGED // WEIGHT_QUANT_BLOCK_N, SF_K),
                       device=dev, dtype=torch.uint8)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
    x_fp8 = torch.empty(M, K_DIM, device=dev, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty(M, SF_K, device=dev, dtype=torch.uint8)

    def run():
        return module.wq_b_proj_gemm_merged(
            x_fp8, as_ue8m0(x_sf), w, as_ue8m0(eb), q_pos, cos_tab, sin_tab,
            mock_post=True, enable_ssq=False, q_y=q_y, **gate_kw)[0]

    y1 = run()
    q1, s1 = x_fp8.clone(), x_sf.clone()
    y2 = run()
    det_ok = (torch.equal(q1.view(torch.uint8), x_fp8.view(torch.uint8))
              and torch.equal(s1, x_sf) and torch.equal(y1, y2))

    qr, sr = ref_fn(q_y)
    fp8_match = (q1.view(torch.uint8) == qr.view(torch.uint8)).float().mean().item()
    sf_match = (s1 == sr).float().mean().item()
    # main q consumes the FRESH fused quant; the indexer weight LEADS w, so the
    # main-q rows start at N_IDX
    ref_main = dequant_act(q1, s1) @ dequant_weight(w[N_IDX:], eb[N_IDX // 128:]).t()
    main_cos = F.cosine_similarity(y1.float().flatten(), ref_main.flatten(), dim=0).item()

    ok = fp8_match > 0.999 and sf_match > 0.999 and main_cos > 0.99 and det_ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} M={M:<4} fp8={fp8_match*100:.2f}% "
          f"sf={sf_match*100:.2f}% main_cos={main_cos:.6f} "
          f"det={'OK' if det_ok else 'MISMATCH'}")
    return ok


def test_quant_iso(module, M):
    """J) fused quant prologue ISOLATION (delivery opB port, no rmsnorm)."""
    return _quant_case(module, M, 'quant_iso   ', {}, ref_quant_k128)


def test_qnorm(module, M):
    """K) fused rmsnorm(gamma)+quant prologue (CTA-wide, deterministic)."""
    torch.manual_seed(M + 37)
    gamma = torch.rand(K_DIM, device='cuda') + 0.5
    return _quant_case(module, M, 'qnorm+quant ', {'q_norm_w': gamma},
                       lambda q_y: ref_qnorm_quant(q_y, gamma))


# ==================== benchmark ====================
def benchmark_merged(module):
    """Single-table accounting at the configured merged shape:
      gemm     = plain merged GEMM (mock_post, no ssq/comp/win/quant)
      d_*      = (that ONE feature on) - gemm : each feature's net latency
      fused    = EVERYTHING ON in one real run (idxpost + ssq + compressor +
                 winkv + rmsnorm-quant prologue) -- NOT a sum of the d_ columns
      d_all    = fused - gemm (measured total net cost of all fusions)
    comp/win rows use all-compress positions (upper bound; production is 1-in-4).
    cuBLAS is bare fp8 _scaled_mm at the same local shape; fused_BW uses fused."""
    print("\n" + "=" * 60)
    mode = "TPDP (main TP2 + index replicated)" if N_TOTAL == TPDP_N_TOTAL else "full rank"
    print(f"Benchmark: MERGED wq_b ({mode}, main={N_TOTAL}, index={N_IDX}, "
          f"N={N_MERGED}), all fusions")
    print("=" * 60)
    dev = 'cuda'
    wm = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w_sfm = make_weight_sf_ones(dev)
    wm_t = wm.t()
    one = torch.ones((), device=dev, dtype=torch.float32)
    cos_tab, sin_tab = make_rope_tables()
    weight_bytes = N_MERGED * K_DIM
    num_tiles = N_MERGED // 256
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    num_clusters = min(num_sms, num_tiles * 2) // 2
    # Mirror the kernel's minimum-wave scheduler.
    waves = -(-num_tiles // num_clusters)
    num_clusters = -(-num_tiles // waves)
    requested_clusters = int(os.environ.get('WQ_B_CLUSTERS', 0))
    if requested_clusters > 0:
        num_clusters = min(requested_clusters, min(num_sms, num_tiles * 2) // 2)
    print(f"  merged weight {weight_bytes/1e6:.1f} MB (e4m3); M dispatches to the "
          f"next 32-row template (TMA handles OOB rows; no caller-side padding)")
    print(f"  scheduler {num_tiles} cluster tiles / {num_clusters} resident clusters "
          f"-> max {((num_tiles + num_clusters - 1) // num_clusters)} iterations/cluster")
    print(f"  {'M':<5} {'gemm(us)':<9} {'fused(us)':<10} {'d_post':<7} {'d_ssq':<7} "
          f"{'d_comp':<7} {'d_win':<7} {'d_quant':<8} {'d_q+norm':<9} {'d_all':<7} "
          f"{'cuBLAS(us)':<11} {'fused_BW':<9} {'%cuBLAS':<8}")
    print("  " + "-" * 116)
    # ADDRESS-PLACEMENT DISCRIMINATOR for the wandering d_ anomalies (they
    # reproduce EXACTLY under a fixed sweep order and move when the order
    # changes -> allocator-history-dependent buffer addresses colliding
    # with the weight TMA stream, NOT kernel behavior):
    #   - empty_cache() per cell -> canonical allocator state
    #   - ssqbuf from a FIXED max-size pool -> same address every cell
    ssqbuf_pool = torch.zeros(128, N_TOTAL // 512, device=dev, dtype=torch.float32)
    # M < 32 shares one specialization, so sample its boundaries and midpoint.
    for M in [1, 16, 31, 32, 61, 64, 96, 97, 127, 128]:
        torch.cuda.empty_cache()
        x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
        x_sf = make_act_sf_ones(M, dev)
        q_y = (torch.randn(M, 4672, device=dev) * 0.1).bfloat16()[:, :K_DIM]
        q_gamma = torch.rand(K_DIM, device=dev) + 0.5
        q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
        comp = make_comp_inputs(M, dev, all_compress=True)
        win = make_win_inputs(M, dev)
        # accumulates garbage across timing iters -- irrelevant for latency
        ssqbuf = ssqbuf_pool[:M]

        def run(mock=True, ssq=None, en_ssq=False, wc=False, ww=False,
                qy=False, qn=False):
            kw = {}
            if wc or ww:
                kw.update(cmp_pos=comp['cmp_pos'], cos_tab=cos_tab, sin_tab=sin_tab)
            if wc:
                kw.update({k: v for k, v in comp.items()
                           if k != 'cmp_pos' and not k.startswith('_')})
            if ww:
                kw.update(win)
            if qy:
                kw['q_y'] = q_y
            if qn:
                kw['q_norm_w'] = q_gamma
            return module.wq_b_proj_gemm_merged(
                x, x_sf, wm, w_sfm, q_pos, cos_tab, sin_tab,
                head_ssq=ssq, mock_post=mock, enable_ssq=en_ssq, **kw)

        def bench(**kw):
            return 1e6 * bench_kineto(lambda: run(**kw), 'wq_b_proj_kernel',
                                      suppress_kineto_output=True)

        ug     = bench()
        upost  = bench(mock=False)
        ussq   = bench(ssq=ssqbuf, en_ssq=True)
        ucomp  = bench(wc=True)
        uwin   = bench(ww=True)
        uq     = bench(qy=True)
        uqn    = bench(qy=True, qn=True)
        uf     = bench(mock=False, ssq=ssqbuf, en_ssq=True, wc=True, ww=True,
                       qy=True, qn=True)
        try:
            cb_pair = bench_kineto(
                lambda: torch._scaled_mm(x, wm_t, scale_a=one, scale_b=one,
                                         out_dtype=torch.bfloat16),
                ('nvjet', 'reduce'), suppress_kineto_output=True)
            cb = 1e6 * sum(cb_pair)
        except Exception as err:
            cb = None
            if M == 32:
                print(f"  (cuBLAS baseline unavailable: {err})")
        obytes = (weight_bytes + N_MERGED // WEIGHT_QUANT_BLOCK_N * SF_K +
                  M * K_DIM + M * SF_K + M * N_TOTAL * 2 +
                  M * (N_IDX // 128) * 68 + M * (N_TOTAL // 512) * 4)
        bw = obytes / (uf * 1e-6) / 1e9
        cb_s = f"{cb:<11.1f}" if cb else f"{'n/a':<11}"
        pct  = f"{cb/uf*100:<8.1f}" if cb else f"{'-':<8}"
        print(f"  {M:<5} {ug:<9.1f} {uf:<10.1f} {upost-ug:<7.2f} {ussq-ug:<7.2f} "
              f"{ucomp-ug:<7.2f} {uwin-ug:<7.2f} {uq-ug:<8.2f} {uqn-ug:<9.2f} "
              f"{uf-ug:<7.2f} {cb_s} {bw:<9.1f} {pct}")


# ==================== fused indexer compressor (delivery port) ====================
def make_comp_inputs(M, dev, all_compress=False):
    """Caller-owned compressor bundle. pos >= 3 keeps rope_pos = p+1-4 >= 0.
    Keys starting with '_' are TEST FIXTURES (front-emit simulation), not
    kernel kwargs -- the fresh state row (+ape) is published by the FRONT
    epilogue in production, so the kernel no longer takes y4/ape."""
    torch.manual_seed(M * 7 + 3)
    pos = (torch.full((M,), 3, dtype=torch.int64, device=dev) if all_compress
           else torch.randint(3, 4000, (M,), dtype=torch.int64, device=dev))
    state = torch.randn(M, 8, 512, device=dev)
    state_rows = (torch.arange(M, device=dev) * 8 + pos % 8).long()
    return dict(
        cmp_pos=pos,
        _y4=torch.randn(M, 512, device=dev),
        _ape=torch.randn(4, 256, device=dev),
        idx_norm=torch.rand(128, device=dev) + 0.5,
        idx_state=state,
        idx_state_row=state_rows,
        state_ring_entries=8)


def publish_fresh(state, state_rows, y4, pos, ape):
    """FRONT-EMIT simulation over RTP [kv|score] state entries."""
    M = pos.shape[0]
    dev = y4.device
    pmod = (pos % 4).long()
    fresh = state.view(-1, 512)[state_rows.long()]
    fresh[:, :256] = y4[:, :256]
    fresh[:, 256:] = y4[:, 256:] + ape[pmod]


def ref_comp_step(state, state_rows, pos, norm_w, cos_tab, sin_tab, eps=1e-6,
                  fp8=False):
    """Torch reference of one fused-compressor step (POOL-READER form: the
    fresh row is ALREADY in kv/sc via publish_fresh): overlap-cat 8-slot
    softmax -> bf16 RMSNorm -> rope(last 64, bf16) -> @H128*128^-0.5 (bf16)
    -> fp4 sim. kv/sc are READ-ONLY here (the kernel writes nothing either).
    The caller supplies the current folded RTP row; older rows are recovered
    by walking backward inside the same physical block."""
    M = pos.shape[0]
    dev = state.device
    compress = ((pos + 1) % 4) == 0
    current = state_rows.long() % 8
    lrow = (current.view(M, 1) - 7 +
            torch.arange(8, device=dev).view(1, 8)) % 8
    ridx = torch.arange(M, device=dev).view(M, 1)
    entries = state[ridx, lrow]
    kv_l, sc_l = entries[..., :256], entries[..., 256:]
    kv_cat = torch.cat([kv_l[:, :4, :128], kv_l[:, 4:, 128:]], dim=1)
    sc_cat = torch.cat([sc_l[:, :4, :128], sc_l[:, 4:, 128:]], dim=1)
    cmp = (torch.softmax(sc_cat, dim=1) * kv_cat).sum(1)            # [M,128]
    v = cmp.bfloat16().float()
    rms = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    x = (v * rms * norm_w).bfloat16().float()
    pr = (pos + 1 - 4).clamp_min(0)
    c, s = cos_tab[pr], sin_tab[pr]
    e, o = x[:, 64::2].clone(), x[:, 65::2].clone()
    x[:, 64::2] = (e * c - o * s).bfloat16().float()
    x[:, 65::2] = (e * s + o * c).bfloat16().float()
    if fp8:
        scale = (x.abs().amax(-1) / 448.0).clamp_min(1e-12)
        return (x / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale, compress
    x = (x @ hadamard_128(dev) * (128.0 ** -0.5)).bfloat16().float()
    blk = x.view(M, 4, 32)
    sf = _ceil_to_ue8m0(blk.abs().amax(-1).clamp_min(6.0 * 2.0 ** -126) / 6.0)
    q = _quantize_to_fp4_e2m1(blk / sf.unsqueeze(-1))
    deq = _dequantize_from_fp4_e2m1(q) * sf.unsqueeze(-1)
    s4 = (sf.log2() + 127).to(torch.uint8)                          # [M,4]
    return deq.view(M, 128), s4, compress


def test_comp(module, M, mock_post=True):
    """H) fused indexer compressor vs the torch reference (delivery semantics):
    state ring updates bit-exact; q4/s4 by match rate (expf/sum-order ulps can
    flip rare fp4 ties). Random pos mixes compress and plain rows.
    mock_post=False additionally exercises comp AND idxpost on the same
    warpgroup (idxpost served first; checks the coexistence path)."""
    dev = 'cuda'
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
    comp = make_comp_inputs(M, dev)
    # FRONT-EMIT simulation: publish the fresh row into the KERNEL's pools
    # (in production front's epilogue does this before wq_b launches).
    publish_fresh(comp['idx_state'], comp['idx_state_row'], comp['_y4'],
                  comp['cmp_pos'], comp['_ape'])
    state_ref = comp['idx_state'].clone()
    kw = {k: v for k, v in comp.items() if not k.startswith('_')}

    out = module.wq_b_proj_gemm_merged(
        x, make_act_sf_ones(M, dev), w, make_weight_sf_ones(dev),
        q_pos, cos_tab, sin_tab, mock_post=mock_post,
        cos_tab=cos_tab, sin_tab=sin_tab, **kw)
    q4, s4 = out[-2], out[-1]                    # [M,64] u8, [M,4] u8

    ref_deq, ref_s4, cmask = ref_comp_step(
        state_ref, comp['idx_state_row'], comp['cmp_pos'],
        comp['idx_norm'], cos_tab, sin_tab)

    state_ok = torch.equal(state_ref, comp['idx_state'])
    if cmask.any():
        un = torch.zeros(M, 128, dtype=torch.int8, device=dev)
        un[:, 0::2] = (q4 & 0x0F).to(torch.int8)
        un[:, 1::2] = ((q4 >> 4) & 0x0F).to(torch.int8)
        scale = torch.pow(2.0, s4.float() - 127.0)
        ker_deq = _dequantize_from_fp4_e2m1(un) * scale.repeat_interleave(32, -1)
        q_match = (ker_deq[cmask] == ref_deq[cmask]).float().mean().item()
        s_match = (s4[cmask] == ref_s4[cmask]).float().mean().item()
    else:
        q_match = s_match = 1.0
    ok = state_ok and q_match > 0.98 and s_match > 0.98
    tag = 'PASS' if ok else 'FAIL'
    mode = 'comp+idxpost' if not mock_post else 'comp        '
    print(f"  [{tag}] {mode} M={M:<4} state={'OK' if state_ok else 'MISMATCH'} "
          f"q4={q_match*100:.2f}% s4={s_match*100:.2f}% "
          f"({int(cmask.sum())}/{M} compress rows)")
    return ok


# ==================== local kv window (winkv, CSA stage 4) ====================
def make_win_inputs(M, dev):
    torch.manual_seed(M * 11 + 5)
    return dict(win_y2=torch.randn(M, 512, device=dev),
                win_norm=torch.rand(512, device=dev) + 0.5)


def ref_win_step(y2, pos, norm_w, cos_tab, sin_tab, eps=1e-6):
    """Torch reference, kernel order: raw-fp32 RMSNorm -> weighted -> [0,448)
    bf16 round + per-64 fp8 (MODEL1 scale = pow2-ceil(clamp(amax/448, 1e-4)),
    e8m0-exact) ; [448,512) fp32 rope
    (token's own pos) -> bf16."""
    M = y2.shape[0]
    v = y2.float()
    rms = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    vw = v * rms * norm_w
    b = vw[:, :448].bfloat16().float().view(M, 7, 64)
    amax = b.abs().amax(-1)
    s8 = torch.pow(2.0, (amax / 448.0).clamp_min(1e-4).log2().ceil())
    q8 = (b / s8.unsqueeze(-1)).to(torch.float8_e4m3fn).view(M, 448)
    e, o = vw[:, 448::2], vw[:, 449::2]
    c, s = cos_tab[pos], sin_tab[pos]
    r = torch.empty(M, 64, device=y2.device)
    r[:, 0::2] = e * c - o * s
    r[:, 1::2] = e * s + o * c
    return q8, s8, r.bfloat16()


def test_win(module, M, mock_post=True):
    """I) fused local-kv-window chain vs the torch reference. Match rates
    tolerate rms ulp differences (sum-order); every row is processed."""
    dev = 'cuda'
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
    pos = torch.randint(3, 4000, (M,), dtype=torch.int64, device=dev)
    win = make_win_inputs(M, dev)

    out = module.wq_b_proj_gemm_merged(
        x, make_act_sf_ones(M, dev), w, make_weight_sf_ones(dev),
        q_pos, cos_tab, sin_tab, mock_post=mock_post,
        cmp_pos=pos, cos_tab=cos_tab, sin_tab=sin_tab, **win)
    wq8, ws8, wrope = out[-3], out[-2], out[-1]

    rq8, rs8, rrope = ref_win_step(win['win_y2'], pos, win['win_norm'],
                                   cos_tab, sin_tab)
    q_match = (wq8.view(torch.uint8) == rq8.view(torch.uint8)).float().mean().item()
    s_match = (ws8 == rs8).float().mean().item()
    r_match = (wrope.view(torch.uint8) == rrope.view(torch.uint8)).float().mean().item()
    ok = q_match > 0.98 and s_match > 0.98 and r_match > 0.98
    tag = 'PASS' if ok else 'FAIL'
    print(f"  [{tag}] winkv        M={M:<4} q8={q_match*100:.2f}% "
          f"s8={s_match*100:.2f}% rope={r_match*100:.2f}%")
    return ok


def test_rtp_paged_pool_writes(module, indexer_fp8=False):
    """Direct Indexer/SWA writes must not touch neighboring RTP records."""
    dev, M = 'cuda', 8
    torch.manual_seed(20260809)
    pos = torch.tensor([3, 4, 5, 6, 7, 8, 9, 11], device=dev,
                       dtype=torch.int64)
    cos_tab, sin_tab = make_rope_tables()
    idx_norm = torch.rand(128, device=dev) + 0.5
    win = make_win_inputs(M, dev)

    block_ids = torch.arange(M + 1, 2 * M + 1, device=dev)
    state_rows = (block_ids * 8 + pos % 8).long()
    idx_state = torch.randn((2 * M + 1) * 8, 512, device=dev)
    y4 = torch.randn(M, 512, device=dev)
    ape = torch.randn(4, 256, device=dev)
    publish_fresh(idx_state, state_rows, y4, pos, ape)
    state_before = idx_state.clone()

    # Keep a valid destination on row 7 but suppress it with state_row=-1.
    # This catches negative ring addressing at a compression boundary.
    kernel_rows = state_rows.clone()
    kernel_rows[7] = -1
    idx_epb, idx_body, idx_scale = 64, (128 if indexer_fp8 else 64), 4
    idx_stride = idx_epb * (idx_body + idx_scale) + 128
    idx_cache = torch.full((5, idx_stride), 0xA5, device=dev,
                           dtype=torch.uint8)
    idx_dst = torch.full((M,), -1, device=dev, dtype=torch.int64)
    idx_dst[0] = idx_epb + 63
    idx_dst[4] = 2 * idx_epb + 7
    idx_dst[7] = 3 * idx_epb + 19

    swa_epb, swa_body, swa_scale = 256, 576, 8
    swa_payload = swa_epb * (swa_body + swa_scale)
    swa_stride = (swa_payload + swa_body - 1) // swa_body * swa_body
    swa_cache = torch.full((9, swa_stride), 0xA5, device=dev,
                           dtype=torch.uint8)
    swa_dst = torch.tensor([
        swa_epb + 3, 2 * swa_epb + 255, 3 * swa_epb + 64, -1,
        4 * swa_epb + 128, 5 * swa_epb + 7, 6 * swa_epb + 200,
        7 * swa_epb + 1], device=dev, dtype=torch.int64)

    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(
        torch.float8_e4m3fn)
    module.wq_b_proj_gemm_merged(
        x, make_act_sf_ones(M, dev), w, make_weight_sf_ones(dev),
        pos.int(), cos_tab, sin_tab, mock_post=True, enable_ssq=False,
        cmp_pos=pos, idx_norm=idx_norm, cos_tab=cos_tab, sin_tab=sin_tab,
        idx_state=idx_state, idx_state_row=kernel_rows,
        state_ring_entries=8, win_y2=win['win_y2'], win_norm=win['win_norm'],
        idx_cache=idx_cache, idx_dst=idx_dst,
        idx_entries_per_block=idx_epb, idx_block_stride_bytes=idx_stride,
        indexer_fp8=indexer_fp8,
        swa_cache=swa_cache, swa_dst=swa_dst,
        swa_entries_per_block=swa_epb, swa_block_stride_bytes=swa_stride)
    torch.cuda.synchronize()

    idx_rows = (0, 4)
    swa_rows = tuple(i for i in range(M) if int(swa_dst[i]) >= 0)
    idx_allowed = torch.zeros_like(idx_cache, dtype=torch.bool)
    swa_allowed = torch.zeros_like(swa_cache, dtype=torch.bool)
    got_q4, got_s4 = [], []
    for row in idx_rows:
        page, off = divmod(int(idx_dst[row]), idx_epb)
        got_q4.append(idx_cache[page, off * idx_body:(off + 1) * idx_body])
        scale_base = idx_epb * idx_body + off * idx_scale
        got_s4.append(idx_cache[page, scale_base:scale_base + idx_scale])
        idx_allowed[page, off * idx_body:(off + 1) * idx_body] = True
        idx_allowed[page, scale_base:scale_base + idx_scale] = True

    got_q8, got_s8, got_rope = [], [], []
    for row in swa_rows:
        page, off = divmod(int(swa_dst[row]), swa_epb)
        body_base = off * swa_body
        body = swa_cache[page, body_base:body_base + swa_body]
        got_q8.append(body[:448])
        got_rope.append(body[448:].contiguous().view(torch.bfloat16))
        scale_base = swa_epb * swa_body + off * swa_scale
        got_s8.append(swa_cache[page, scale_base:scale_base + 7])
        swa_allowed[page, body_base:body_base + swa_body] = True
        # The eighth scale-record byte is padding and must remain canary.
        swa_allowed[page, scale_base:scale_base + 7] = True

    untouched = (
        torch.equal(idx_cache[~idx_allowed],
                    torch.full_like(idx_cache[~idx_allowed], 0xA5))
        and torch.equal(swa_cache[~swa_allowed],
                        torch.full_like(swa_cache[~swa_allowed], 0xA5))
        and torch.equal(idx_state, state_before))

    selected_state = state_before.view(-1, 8, 512)[block_ids.long()]
    ref_deq, ref_s4, _ = ref_comp_step(
        selected_state, (pos % 8).int(), pos, idx_norm, cos_tab, sin_tab,
        fp8=indexer_fp8)
    got_q4, got_s4 = torch.stack(got_q4), torch.stack(got_s4)
    rows = torch.tensor(idx_rows, device=dev)
    if indexer_fp8:
        got_scale = got_s4.contiguous().view(torch.float32).view(-1)
        idx_ok = (
            (got_q4 == ref_deq[rows].view(torch.uint8)).float().mean().item()
            > 0.999
            and torch.allclose(got_scale, ref_s4[rows], rtol=1e-6, atol=0))
    else:
        unpacked = torch.empty(len(idx_rows), 128, dtype=torch.int8, device=dev)
        unpacked[:, 0::2] = (got_q4 & 0x0F).to(torch.int8)
        unpacked[:, 1::2] = ((got_q4 >> 4) & 0x0F).to(torch.int8)
        got_scale = torch.pow(2.0, got_s4.float() - 127.0)
        got_deq = (_dequantize_from_fp4_e2m1(unpacked)
                   * got_scale.repeat_interleave(32, -1))
        idx_ok = ((got_deq == ref_deq[rows]).float().mean().item() > 0.98
                  and (got_s4 == ref_s4[rows]).float().mean().item() > 0.98)

    ref_q8, ref_scale8, ref_rope = ref_win_step(
        win['win_y2'], pos, win['win_norm'], cos_tab, sin_tab)
    got_q8, got_s8, got_rope = (torch.stack(got_q8), torch.stack(got_s8),
                                 torch.stack(got_rope))
    rows = torch.tensor(swa_rows, device=dev)
    ref_exp8 = (ref_scale8.log2() + 127).to(torch.uint8)
    swa_ok = (
        (got_q8 == ref_q8[rows].view(torch.uint8)).float().mean().item() > 0.98
        and (got_s8 == ref_exp8[rows]).float().mean().item() > 0.98
        and (got_rope.view(torch.uint8)
             == ref_rope[rows].contiguous().view(torch.uint8)).float().mean().item()
        > 0.98)
    ok = untouched and idx_ok and swa_ok
    fmt = 'FP8' if indexer_fp8 else 'FP4'
    print(f"  RTP paged-pool writers ({fmt} Indexer): {'PASS' if ok else 'FAIL'}")
    return ok


def profile_pipeline(module, M=128, clock_ghz=1.8):
    """Visualize load / MMA / epilogue overlap (merged path).
    timing[max_iters,7] = [load_s, load_e, mma_s, mma_e, epi_s, epi_e, mma_wait]
    (clock64 cycles, cluster0/CTA0 -> same SM). col6 mma_wait = cycles the MMA warp
    spent WAITING (tmem_empty + with_sf_full); MMA_active = (mma_e-mma_s) - mma_wait."""
    import numpy as np
    print("\n" + "=" * 76)
    print(f"Pipeline overlap (clock64): LOAD vs MMA vs EPILOGUE (merged, M={M})")
    print("=" * 76)
    dev = 'cuda'
    x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    x_sf = make_act_sf_ones(M, dev)
    w_sf = make_weight_sf_ones(dev)
    cos_tab, sin_tab = make_rope_tables()
    q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)

    args = (x, x_sf, w, w_sf, q_pos, cos_tab, sin_tab)
    for _ in range(5):
        module.wq_b_proj_gemm_merged_profiled(*args)
    torch.cuda.synchronize()
    timing = module.wq_b_proj_gemm_merged_profiled(*args)[-1]
    torch.cuda.synchronize()

    t = timing.cpu().numpy().astype(np.int64)
    t = t[~(t == 0).all(axis=1)]                 # drop iterations that never ran
    if len(t) == 0:
        print("  no timing rows captured"); return
    ls, le, ms, me, es, ee = (t[:, i] for i in range(6))
    mw = t[:, 6]
    origin = int(min(ls.min(), ms.min(), es.min()))
    ls, le, ms, me, es, ee = (a - origin for a in (ls, le, ms, me, es, ee))
    span = int(max(le.max(), me.max(), ee.max()))
    if span <= 0:
        print("  degenerate span"); return
    c2us = lambda c: c / clock_ghz / 1e3
    n = len(t)
    mma_active = (me - ms) - mw
    print(f"  {n} persistent iterations; span={span} cyc (~{c2us(span):.1f} us @ {clock_ghz}GHz assumed)")

    print(f"  {'it':>3}  {'LOAD dur':>9}  {'MMA dur':>9} {'(wait':>7}{'/active)':>9}  {'EPI dur':>9}")
    for i in range(min(n, 12)):
        print(f"  {i:>3}  {le[i]-ls[i]:>9}  {me[i]-ms[i]:>9} ({mw[i]:>6}/{mma_active[i]:>7})  {ee[i]-es[i]:>9}")

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
    print(f"  mean window: LOAD {ld.mean():.0f}  MMA {md.mean():.0f} (wait {mw.mean():.0f} / active {mma_active.mean():.0f})  EPI {ed.mean():.0f} cyc")
    print(f"  track fill (sum/span): LOAD {ld.sum()/span:.2f}  MMA {md.sum()/span:.2f}  MMA_active {mma_active.sum()/span:.2f}  EPI {ed.sum()/span:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tpdp', action='store_true',
        help='main-Q TP2 plus all 64 index-Q heads replicated '
             '(main=32768/index=8192); default is full rank')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")
    if sm < 100:
        print(f"ERROR: tcgen05 block-scale requires sm_100+ (Blackwell), got sm_{sm}")
        sys.exit(1)

    configure_geometry(args.tpdp)
    geo = 'TPDP main TP2 + replicated indexer' if args.tpdp else 'full rank'
    print(f"Geometry: {geo} "
          f"main={N_TOTAL}, index={N_IDX}, merged={N_MERGED}, "
          f"cluster_tiles={N_MERGED // 256}")
    module = load_module(tpdp=args.tpdp)
    assert (module.n_main, module.n_index, module.n_merged) == \
        (N_TOTAL, N_IDX, N_MERGED), "Python/kernel WQB geometry mismatch"
    assert module.slot_dtype_bits == 64, "FP8 CSA slot ABI must be int64"

    print("\nCorrectness (merged, non-trivial scales, arbitrary batch):")
    results = [test_merged(module, M)
               for M in [1, 31, 32, 61, 64, 96, 97, 127, 128]]

    print("\nCorrectness (RTP FP8 Indexer-Q transform):")
    results += [test_indexer_q_fp8(module, M) for M in [1, 61, 128]]

    print("\nCorrectness (fused indexer compressor, delivery port):")
    results += [test_comp(module, M) for M in [1, 32, 61, 128]]
    results += [test_comp(module, M, mock_post=False) for M in [31, 61, 128]]

    print("\nCorrectness (fused local kv window, CSA stage 4):")
    results += [test_win(module, M) for M in [1, 32, 61, 128]]
    results += [test_win(module, M, mock_post=False) for M in [61, 128]]
    results.append(test_rtp_paged_pool_writes(module))
    results.append(test_rtp_paged_pool_writes(module, indexer_fp8=True))

    print("\nCorrectness (fused quant prologue, isolation):")
    results += [test_quant_iso(module, M) for M in [1, 32, 61, 128]]

    print("\nCorrectness (fused rmsnorm+quant prologue, production form):")
    results += [test_qnorm(module, M) for M in [1, 32, 61, 128]]

    benchmark_merged(module)

    profile_pipeline(module, M=128)

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
