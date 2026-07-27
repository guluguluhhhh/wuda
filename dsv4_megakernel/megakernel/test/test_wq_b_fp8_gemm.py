"""
Test & Benchmark: MERGED wq_b projection (tcgen05 FP8 block-scale, swap-AB,
M<=128) with the sparse-indexer projection fused in:

  x_fp8[M,1536] @ w_fp8[73728,1536].T
    -> y[M,65536] FP32                        (main q, CSA stage 3)
    -> iq_fp4[M,64,64] i8 + iq_sf[M,64] i32   (indexer q, CSA stage 7:
       rope + hadamard-128 + per-32 MXFP4, run by the async xform warpgroup)

Requires: NVIDIA Blackwell (sm_100+), CUDA 12.8+, CUTLASS 3.x.

Native DSV4 scale-factor layout expected by the kernel:
  - dtype float8_e8m0fnu (raw uint8 is also accepted).
  - x_sf: [M, K/128] = [M, 12], one scale per token/K128.
  - w_sf: [N/128, K/128] = [576, 12], one scale per N128xK128 block.
  - UE8M0 byte e encodes scale 2^(e-127); e=127 (0x7F) => scale 1.0.
"""
import os, sys, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

K_DIM   = 1536
N_TOTAL = 65536          # main q: 128 heads x 512 dim
N_IDX   = 64 * 128       # indexer q: 64 heads x 128 dim
N_MERGED = N_TOTAL + N_IDX
QUANT_BLOCK_K = 128
WEIGHT_QUANT_BLOCK_N = 128
SF_K    = K_DIM // QUANT_BLOCK_K          # 12
UE8M0_ONE = 0x7F                    # exponent 127 -> 2^0 = 1.0


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
        name='wq_b_fp8_gemm',
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
    x = iq_f32.float().view(M, 64, 128).bfloat16().float()
    c = cos_tab[pos].unsqueeze(1)
    s = sin_tab[pos].unsqueeze(1)
    e = x[..., 64::2].clone()
    o = x[..., 65::2].clone()
    x[..., 64::2] = (e * c - o * s).bfloat16().float()
    x[..., 65::2] = (e * s + o * c).bfloat16().float()
    x = (x @ hadamard_128(x.device) * (128.0 ** -0.5)).bfloat16().float()
    blk = x.view(M, 64, 4, 32)
    sf = _ceil_to_ue8m0(blk.abs().amax(-1).clamp_min(6.0 * 2.0 ** -126) / 6.0)  # [M,64,4]
    q = _quantize_to_fp4_e2m1(blk / sf.unsqueeze(-1))
    deq = _dequantize_from_fp4_e2m1(q) * sf.unsqueeze(-1)
    return deq.view(M, 64, 128), sf


def dequant_kernel_iq(iq_fp4, iq_sf):
    """Kernel outputs -> float: iq_fp4 [M,64,64] i8 packed (low nibble = even elem),
    iq_sf [M,64] i32 packed-ue8m0 (byte b = 32-col block b)."""
    M = iq_fp4.shape[0]
    p = iq_fp4.view(M, 64, 64)
    un = torch.zeros(M, 64, 128, dtype=torch.int8, device=p.device)
    un[..., 0::2] = p & 0x0F
    un[..., 1::2] = (p >> 4) & 0x0F
    vals = _dequantize_from_fp4_e2m1(un)
    sf_b = iq_sf.contiguous().view(torch.uint8).view(M, 64, 4).to(torch.int)
    scale = torch.pow(2.0, sf_b.float() - 127.0)                                # [M,64,4]
    return vals * scale.repeat_interleave(32, dim=-1), scale


# ==================== correctness ====================
def test_merged(module, M):
    """One shot per M, quiet on PASS:
      A) main q [M,65536] vs the dequantized torch GEMM (non-trivial scales)
      B) dequant(iq_fp4, iq_sf) vs the torch reference chain
      C) SF words vs the reference power-of-2 scales
      D) standalone idx_post_kernel vs the same torch reference (same fp32 input)
      E) head_ssq RED accumulation vs the kernel's own main-q output
      F) BITWISE async-handoff closed loop: the mock run exports the kernel's
         own drained fp32 iq; the standalone kernel over it (same
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

    # A) main q vs torch ref
    xd = dequant_act(x, ea)
    ref_main = xd @ dequant_weight(w[:N_TOTAL], eb[:N_TOTAL // 128]).t()
    main_cos = F.cosine_similarity(y.flatten(), ref_main.flatten(), dim=0).item()

    # B/C) iq chain vs torch ref from the fp32 GEMM of the LAST N_IDX weight rows
    iq_ref_f32 = xd @ dequant_weight(w[N_TOTAL:], eb[N_TOTAL // 128:]).t()      # [M, 8192]
    ref_deq, ref_sf = ref_idx_chain(iq_ref_f32, q_pos.long(), cos_tab, sin_tab)
    ker_deq, ker_scale = dequant_kernel_iq(iq_fp4, iq_sf)
    iq_cos = F.cosine_similarity(ker_deq.flatten(), ref_deq.flatten(), dim=0).item()
    sf_match = (ker_scale == ref_sf).float().mean().item()

    # D) standalone kernel, same chain, same fp32 input
    sa_fp4, sa_sf = module.idx_postprocess_standalone(
        iq_ref_f32.view(M, 64, 128).contiguous(), q_pos, cos_tab, sin_tab)
    sa_deq, _ = dequant_kernel_iq(sa_fp4, sa_sf)
    sa_cos = F.cosine_similarity(sa_deq.flatten(), ref_deq.flatten(), dim=0).item()

    # E) head_ssq vs the kernel's OWN main-q output (isolates the RED accumulation)
    ssq_ref = y.view(M, 128, 512).double().square().sum(-1).float()
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

    ok = (main_cos > 0.99 and iq_cos > 0.99 and sf_match > 0.98
          and sa_cos > 0.999 and ssq_rel < 1e-3 and ssq_sa_rel < 1e-3 and bit_ok)
    line = (f"main_cos={main_cos:.6f} iq_cos={iq_cos:.6f} sf={sf_match*100:.2f}% "
            f"sa_cos={sa_cos:.6f} ssq_rel={ssq_rel:.1e} ssq_sa_rel={ssq_sa_rel:.1e} "
            f"bitwise={'OK' if bit_ok else 'MISMATCH'}")
    if ok:
        print(f"  [PASS] merged M={M:<4} {line}")
    else:
        print(f"  [FAIL] merged M={M}")
        print(f"    {line}")
        print(f"    main max_diff: {(y - ref_main).abs().max().item():.4e}")
        print(f"    iq   max_diff: {(ker_deq - ref_deq).abs().max().item():.4e}")
        if not bit_ok:
            nf = (bw_fp4 != iq_fp4).sum().item()
            ns = (bw_sf != iq_sf).sum().item()
            print(f"    bitwise mismatch: fp4 bytes {nf}, sf words {ns}")
    return ok


# ==================== benchmark ====================
def benchmark_merged(module):
    """Fused vs separate-kernel accounting at the merged shape (N=73728):
      fused    = merged GEMM with BOTH fused extras on (head_ssq + post)
      gemm     = same kernel, mock_post=True, no head_ssq -> the pure merged GEMM
      d_post   = (post only) - gemm : net latency of the fused post-processing
      sep_post = standalone idx_post_kernel over the fp32 iq (whole-GPU parallel)
      d_ssq    = (ssq only) - gemm  : net latency of the fused head_ssq
      sep_ssq  = standalone head_ssq_kernel over the fp32 y (memory-bound floor)
      cuBLAS   = bare fp8 _scaled_mm at [M,73728] (nvjet), fp32 out
    Fusion wins iff d_post < sep_post and d_ssq < sep_ssq."""
    print("\n" + "=" * 60)
    print("Benchmark: MERGED indexer projection (N=73728)")
    print("=" * 60)
    dev = 'cuda'
    wm = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w_sfm = make_weight_sf_ones(dev)
    wm_t = wm.t()
    one = torch.ones((), device=dev, dtype=torch.float32)
    cos_tab, sin_tab = make_rope_tables()
    weight_bytes = N_MERGED * K_DIM
    print(f"  merged weight {weight_bytes/1e6:.1f} MB (e4m3); fused outputs: y fp32 "
          f"[M,65536] + iq fp4+sf [M,64,68B] + ssq [M,128]")
    print(f"  (M < 32 pads to the 32-row template: GEMM time ~= M=32)")
    print(f"  {'M':<5} {'fused(us)':<10} {'gemm(us)':<9} {'d_post':<7} {'sep_post':<9} "
          f"{'d_ssq':<7} {'sep_ssq':<8} {'cuBLAS(us)':<11} {'fused_BW':<9} {'%cuBLAS':<8}")
    print("  " + "-" * 88)
    for M in [1, 2, 7, 16, 31, 32, 61, 64, 96, 97, 127, 128]:
        x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
        x_sf = make_act_sf_ones(M, dev)
        q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
        iq_f32 = torch.randn(M, 64, 128, device=dev, dtype=torch.float32)
        y_f32 = torch.randn(M, N_TOTAL, device=dev, dtype=torch.float32)
        # accumulates garbage across timing iters -- irrelevant for latency
        ssqbuf = torch.zeros(M, 128, device=dev, dtype=torch.float32)

        def run(ssq, mock, en_ssq):
            return module.wq_b_proj_gemm_merged(
                x, x_sf, wm, w_sfm, q_pos, cos_tab, sin_tab,
                head_ssq=ssq, mock_post=mock, enable_ssq=en_ssq)

        uf = 1e6 * bench_kineto(lambda: run(ssqbuf, False, True),
                                'wq_b_proj_kernel', suppress_kineto_output=True)
        ug = 1e6 * bench_kineto(lambda: run(None, True, False),
                                'wq_b_proj_kernel', suppress_kineto_output=True)
        upost = 1e6 * bench_kineto(lambda: run(None, False, False),
                                   'wq_b_proj_kernel', suppress_kineto_output=True)
        ussq = 1e6 * bench_kineto(lambda: run(ssqbuf, True, True),
                                  'wq_b_proj_kernel', suppress_kineto_output=True)
        up = 1e6 * bench_kineto(
            lambda: module.idx_postprocess_standalone(iq_f32, q_pos, cos_tab, sin_tab),
            'idx_post_kernel', suppress_kineto_output=True)
        us = 1e6 * bench_kineto(
            lambda: module.head_ssq_standalone(y_f32),
            'head_ssq_kernel', suppress_kineto_output=True)
        try:
            cb_pair = bench_kineto(
                lambda: torch._scaled_mm(x, wm_t, scale_a=one, scale_b=one,
                                         out_dtype=torch.float32),
                ('nvjet', 'reduce'), suppress_kineto_output=True)
            cb = 1e6 * sum(cb_pair)
        except Exception as err:
            cb = None
            if M == 32:
                print(f"  (cuBLAS baseline unavailable: {err})")
        obytes = (weight_bytes + N_MERGED // WEIGHT_QUANT_BLOCK_N * SF_K +
                  M * K_DIM + M * SF_K + M * N_TOTAL * 4 + M * 64 * 68 + M * 128 * 4)
        bw = obytes / (uf * 1e-6) / 1e9
        cb_s = f"{cb:<11.1f}" if cb else f"{'n/a':<11}"
        pct  = f"{cb/uf*100:<8.1f}" if cb else f"{'-':<8}"
        print(f"  {M:<5} {uf:<10.1f} {ug:<9.1f} {upost-ug:<7.2f} {up:<9.2f} "
              f"{ussq-ug:<7.2f} {us:<8.2f} {cb_s} {bw:<9.1f} {pct}")


# ==================== fused indexer compressor (delivery port) ====================
def make_comp_inputs(M, dev, all_compress=False):
    """Caller-owned compressor bundle. pos >= 3 keeps rope_pos = p+1-4 >= 0."""
    torch.manual_seed(M * 7 + 3)
    pos = (torch.full((M,), 3, dtype=torch.int64, device=dev) if all_compress
           else torch.randint(3, 4000, (M,), dtype=torch.int64, device=dev))
    return dict(
        cmp_pos=pos,
        idx_y4=torch.randn(M, 512, device=dev),
        idx_ape=torch.randn(4, 256, device=dev),
        idx_norm=torch.rand(128, device=dev) + 0.5,
        idx_kv=torch.randn(M, 8, 256, device=dev),
        idx_sc=torch.randn(M, 8, 256, device=dev))


def ref_comp_step(kv, sc, y4, pos, ape, norm_w, cos_tab, sin_tab, eps=1e-6):
    """Torch reference of one fused-compressor step, kernel order & rounding:
    fresh write (+ape) -> [compress] overlap-cat 8-slot softmax -> shift ->
    bf16 RMSNorm -> rope(last 64, bf16) -> @H128*128^-0.5 (bf16) -> fp4 sim.
    kv/sc are updated IN PLACE (pass clones)."""
    M = pos.shape[0]
    dev = y4.device
    pmod = (pos % 4).long()
    kv[torch.arange(M, device=dev), 4 + pmod] = y4[:, :256]
    sc[torch.arange(M, device=dev), 4 + pmod] = y4[:, 256:] + ape[pmod]
    compress = ((pos + 1) % 4) == 0
    # aggregate BEFORE the shift (kernel order), overlap-cat columns
    kv_cat = torch.cat([kv[:, :4, :128], kv[:, 4:, 128:]], dim=1)   # [M,8,128]
    sc_cat = torch.cat([sc[:, :4, :128], sc[:, 4:, 128:]], dim=1)
    cmp = (torch.softmax(sc_cat, dim=1) * kv_cat).sum(1)            # [M,128]
    kv[compress, :4] = kv[compress, 4:].clone()
    sc[compress, :4] = sc[compress, 4:].clone()
    v = cmp.bfloat16().float()
    rms = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    x = (v * rms * norm_w).bfloat16().float()
    pr = (pos + 1 - 4).clamp_min(0)
    c, s = cos_tab[pr], sin_tab[pr]
    e, o = x[:, 64::2].clone(), x[:, 65::2].clone()
    x[:, 64::2] = (e * c - o * s).bfloat16().float()
    x[:, 65::2] = (e * s + o * c).bfloat16().float()
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
    kv_ref, sc_ref = comp['idx_kv'].clone(), comp['idx_sc'].clone()

    out = module.wq_b_proj_gemm_merged(
        x, make_act_sf_ones(M, dev), w, make_weight_sf_ones(dev),
        q_pos, cos_tab, sin_tab, mock_post=mock_post,
        cos_tab=cos_tab, sin_tab=sin_tab, **comp)
    q4, s4 = out[-2], out[-1]                    # [M,64] u8, [M,4] u8

    ref_deq, ref_s4, cmask = ref_comp_step(
        kv_ref, sc_ref, comp['idx_y4'], comp['cmp_pos'], comp['idx_ape'],
        comp['idx_norm'], cos_tab, sin_tab)

    state_ok = (torch.equal(kv_ref, comp['idx_kv'])
                and torch.equal(sc_ref, comp['idx_sc']))
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
    bf16 round + per-64 fp8 (scale=max(amax,1e-4)/448) ; [448,512) fp32 rope
    (token's own pos) -> bf16."""
    M = y2.shape[0]
    v = y2.float()
    rms = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    vw = v * rms * norm_w
    b = vw[:, :448].bfloat16().float().view(M, 7, 64)
    amax = b.abs().amax(-1).clamp_min(1e-4)
    s8 = amax / 448.0
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


def benchmark_comp(module):
    """Additivity check for the warp-8..11 residents (all-compress rows):
      base  = default GEMM+ssq (256 threads)
      d_post/d_comp/d_win = each feature alone - base
      d_all = idxpost + compressor + winkv together - base
    Coexistence is free iff d_all <~ sum (one row per warp, spread mapping)."""
    print("\nFused post-processing coexistence (all-compress rows):")
    print(f"  {'M':<5} {'base(us)':<9} {'d_post':<7} {'d_comp':<7} {'d_win':<7} "
          f"{'d_all':<7} {'sum':<7}")
    print("  " + "-" * 52)
    dev = 'cuda'
    wm = (torch.randn(N_MERGED, K_DIM, device=dev) * 0.05).to(torch.float8_e4m3fn)
    w_sfm = make_weight_sf_ones(dev)
    cos_tab, sin_tab = make_rope_tables()
    for M in [1, 16, 32, 64, 96, 128]:
        x = (torch.randn(M, K_DIM, device=dev) * 0.1).to(torch.float8_e4m3fn)
        x_sf = make_act_sf_ones(M, dev)
        q_pos = torch.randint(0, cos_tab.shape[0], (M,), device=dev, dtype=torch.int32)
        comp = make_comp_inputs(M, dev, all_compress=True)
        win = make_win_inputs(M, dev)

        def run(mock, wc, ww):
            kw = {}
            if wc or ww:
                kw.update(cmp_pos=comp['cmp_pos'], cos_tab=cos_tab, sin_tab=sin_tab)
            if wc:
                kw.update({k: v for k, v in comp.items() if k != 'cmp_pos'})
            if ww:
                kw.update(win)
            return module.wq_b_proj_gemm_merged(
                x, x_sf, wm, w_sfm, q_pos, cos_tab, sin_tab,
                mock_post=mock, **kw)

        t = {}
        for name, mock, wc, ww in [('base', True, False, False),
                                   ('post', False, False, False),
                                   ('comp', True, True, False),
                                   ('win', True, False, True),
                                   ('all', False, True, True)]:
            t[name] = 1e6 * bench_kineto(lambda: run(mock, wc, ww),
                                         'wq_b_proj_kernel',
                                         suppress_kineto_output=True)
        dp, dc, dw = t['post'] - t['base'], t['comp'] - t['base'], t['win'] - t['base']
        da = t['all'] - t['base']
        print(f"  {M:<5} {t['base']:<9.1f} {dp:<7.2f} {dc:<7.2f} {dw:<7.2f} "
              f"{da:<7.2f} {dp+dc+dw:<7.2f}")


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
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    sm = cap[0] * 10 + cap[1]
    print(f"Compute: sm_{sm}")
    if sm < 100:
        print(f"ERROR: tcgen05 block-scale requires sm_100+ (Blackwell), got sm_{sm}")
        sys.exit(1)

    module = load_module()

    print("\nCorrectness (merged, non-trivial scales, arbitrary batch):")
    # 32-aligned template points + small/pow2 + primes (host pads to 32)
    results = [test_merged(module, M)
               for M in [1, 2, 7, 16, 31, 32, 61, 64, 96, 97, 127, 128]]

    print("\nCorrectness (fused indexer compressor, delivery port):")
    results += [test_comp(module, M) for M in [1, 2, 7, 32, 61, 128]]
    results += [test_comp(module, M, mock_post=False) for M in [7, 61, 128]]

    print("\nCorrectness (fused local kv window, CSA stage 4):")
    results += [test_win(module, M) for M in [1, 7, 32, 61, 128]]
    results += [test_win(module, M, mock_post=False) for M in [61, 128]]

    benchmark_merged(module)

    benchmark_comp(module)

    profile_pipeline(module, M=128)

    print("\n" + "=" * 60)
    print(f"Summary: {'ALL PASS' if all(results) else 'SOME FAILED'}")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)
