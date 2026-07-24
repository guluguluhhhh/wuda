"""Correctness (vs DeepGEMM ref_fp8_mqa_logits) + B300 benchmark for the migrated
FP4 MQA-logits kernel (kernels/mqa_logits_fp4.cu).

Perf metric: kernel_us only — DeepGEMM bench_kineto methodology (pure GPU kernel
time, L2 flushed before every call); DeepGEMM's test suite reports no wall time.

Covers both entry points:
  * mqa_logits_fp4         — single-sequence, NON-compressed RAW logits (masked on host
                             to match DeepGEMM's clean_logits, then compared to ref).
  * mqa_logits_fp4_decode  — MULTI-BATCH decode, compressed, ONE launch (tile-pool
                             schedule: grid.x = #SMs, global KV tiles balanced across
                             CTAs); compared to a per-batch ref over a context-length
                             GRADIENT (uniform + mixed per-seq valid_len).

Requires: B300 (sm_100+), CUDA >= 12.8. Fully self-contained — no `deep_gemm`
package needed (FP4 quant/dequant + calc_diff are inlined below).

    python test/test_mqa_logits_fp4.py            # correctness, then the fuse-comp table
    python test/test_mqa_logits_fp4.py --base     # attention-only table instead
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

NUM_HEADS = 64
HEAD_DIM = 128


# ==================================================================================
# FP4 quantization + metric helpers, inlined from DeepGEMM (deep_gemm/utils/math.py
# and deep_gemm/testing/numeric.py) so this test is self-contained — no `deep_gemm`
# package needed on the B300 box. These MUST stay bit-identical to DeepGEMM so the
# kernel's inputs (packed int8 fp4 + int32 packed-ue8m0 sf) match what it expects.
# ==================================================================================
def _align(x, y):
    return (x + y - 1) // y * y


def _ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _pack_ue8m0_to_int(x):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def _unpack_ue8m0_from_int(packed_sf):
    return (packed_sf.view(torch.uint8).to(torch.int) << 23).view(torch.float)


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


def per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True):
    m, n = x.shape
    assert n % 2 == 0 and (not use_packed_ue8m0 or use_ue8m0)
    padded_n = _align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)  # int8
    return packed[:, :n // 2].contiguous(), (_pack_ue8m0_to_int(sf) if use_packed_ue8m0 else sf)


def cast_back_from_fp4(packed, sf, gran_k=32, use_packed_ue8m0=True):
    m, n2 = packed.shape
    n = n2 * 2
    if use_packed_ue8m0:
        sf = _unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x_dequantized = _dequantize_from_fp4_e2m1(unpacked)
    group_idx = torch.arange(n, device=packed.device) // gran_k
    return x_dequantized * sf[:, group_idx]


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def load_cuda_module():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    cutlass_dir = os.path.join(proj_dir, "..", "cutlass", "include")
    cutlass_tools_dir = os.path.join(proj_dir, "..", "cutlass", "tools", "util", "include")

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < 100:
        raise RuntimeError(f"tcgen05 requires Blackwell sm_100+, got sm_{sm}")

    cuda_flags = [
        "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo",
        # Register-diet checkpoint: ptxas prints per-kernel register usage in the JIT
        # verbose log. The math path MUST compile to <= 128 regs/thread (see
        # kNumMathRegisters in mqa_logits_fp4.cuh); if "Used NNN registers" exceeds
        # 128, raise kNumMathRegisters to that value rounded up to a multiple of 8.
        "--ptxas-options=-v",
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
        "-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        # NOTE: cute/arch/config.hpp auto-defines CUTE_ARCH_TCGEN05_{TMEM,F16F32_MMA}_ENABLED
        # for sm_10xa; passing them again triggers "redefined" warnings — so we don't
        # (same lesson the repo's test_hc_fused_tc.py records for the TF32 macro).
        # -diag-suppress=3288: DeepGEMM's tmem_load uses a C++20 explicit-lambda-template under -std=c++17.
        "-diag-suppress=3288",
        f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
    ]
    return load(
        name="mqa_logits_fp4",
        sources=[os.path.join(proj_dir, "kernels", "mqa_logits_fp4.cu")],
        extra_include_paths=[
            os.path.join(proj_dir, "include"),
            cutlass_dir,
            cutlass_tools_dir,
        ],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-lcuda"],
        verbose=True,
    )


def ref_fp8_mqa_logits(q, kv, weights, ks, ke):
    """Verbatim from DeepGEMM tests/test_attention.py::ref_fp8_mqa_logits."""
    seq_len_kv = kv.shape[0]
    k = kv.float()
    q = q.float()
    mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= ks[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < ke[:, None]
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


def quantize_fp4(x, last_dim):
    """x[..., last_dim] bf16 -> (packed int8, sf int32, simulated bf16). gran_k=32, ue8m0."""
    flat = x.reshape(-1, last_dim)
    packed, sf = per_token_cast_to_fp4(flat, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    sim = cast_back_from_fp4(packed, sf, gran_k=32, use_packed_ue8m0=True).to(torch.bfloat16)
    return packed, sf.to(torch.int32), sim.view_as(x)


# --------------------------------------------------------------------- single-seq
def run_single(module, seq_len, seq_len_kv, out_dtype):
    torch.manual_seed(seq_len * 131 + seq_len_kv)
    q = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, NUM_HEADS, device="cuda", dtype=torch.float32)
    # simple causal-ish ranges (disable_cp path)
    ks = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
    ke = torch.arange(seq_len, dtype=torch.int32, device="cuda") + (seq_len_kv - seq_len)

    q_p, q_sf, q_sim = quantize_fp4(q, HEAD_DIM)
    kv_p, kv_sf, kv_sim = quantize_fp4(kv, HEAD_DIM)
    q_p = q_p.view(seq_len, NUM_HEADS, HEAD_DIM // 2).contiguous()
    q_sf = q_sf.view(seq_len, NUM_HEADS).contiguous()
    kv_p = kv_p.view(seq_len_kv, HEAD_DIM // 2).contiguous()
    kv_sf = kv_sf.view(seq_len_kv).contiguous()

    ref = ref_fp8_mqa_logits(q, kv, weights, ks, ke)
    sim = ref_fp8_mqa_logits(q_sim, kv_sim, weights, ks, ke)
    got = module.mqa_logits_fp4(q_p, q_sf, kv_p, kv_sf, weights, ks, ke, out_dtype)
    torch.cuda.synchronize()

    # forward returns RAW logits -> clean on host to the [ks,ke) mask (== DeepGEMM clean_logits)
    valid = ref != float("-inf")
    got_f = torch.where(valid, got.float(), torch.zeros_like(got.float()))
    ref_f = ref.masked_fill(~valid, 0).float()
    sim_f = sim.masked_fill(~valid, 0).float()
    return calc_diff(got_f, ref_f), calc_diff(got_f, sim_f)


def test_single(module):
    print("\n[single-seq] FP4 MQA-logits (RAW, host-cleaned) vs DeepGEMM ref")
    print(f"{'S':>6} {'Skv':>7} {'dtype':>6} {'diff_ref':>12} {'diff_sim':>12} {'result':>8}")
    print("-" * 54)
    ok_all = True
    for out_dtype in (torch.float32, torch.bfloat16):
        for s, skv in [(256, 1024), (512, 1024), (2048, 4096), (4096, 8192)]:
            diff, sim = run_single(module, s, skv, out_dtype)
            ok = diff < 0.02 and sim < 2e-3
            ok_all &= ok
            print(f"{s:6d} {skv:7d} {str(out_dtype).split('.')[-1]:>6} "
                  f"{diff:12.4e} {sim:12.4e} {('PASS' if ok else 'FAIL'):>8}")
    print("-" * 54)
    return ok_all


# ------------------------------------------------------------------ multi-batch decode
def make_valid(B, T, valid):
    """valid: None (full T) | int (uniform) | "mixed" (per-batch length gradient, to
    exercise the tile-pool scheduler's cross-token balancing). -> (list, tensor|None)"""
    if valid is None:
        return [T] * B, None
    if valid == "mixed":
        fracs = torch.linspace(0.1, 1.0, B).tolist()
        valid_list = [max(1, int(T * f)) for f in fracs]
        return valid_list, torch.tensor(valid_list, dtype=torch.int32, device="cuda")
    return [valid] * B, torch.full((B,), valid, dtype=torch.int32, device="cuda")


def run_decode(module, B, T, out_dtype, valid=None):
    torch.manual_seed(B * 7 + T)
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)     # iq per token
    kv = torch.randn(B, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)            # idx_kv_cache
    weights = torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.float32)

    q_p, q_sf, q_sim = quantize_fp4(q, HEAD_DIM)
    kv_p, kv_sf, kv_sim = quantize_fp4(kv, HEAD_DIM)
    q_p = q_p.view(B, NUM_HEADS, HEAD_DIM // 2).contiguous()
    q_sf = q_sf.view(B, NUM_HEADS).contiguous()
    kv_p = kv_p.view(B, T, HEAD_DIM // 2).contiguous()
    kv_sf = kv_sf.view(B, T).contiguous()
    q_sim = q_sim.view(B, NUM_HEADS, HEAD_DIM)
    kv_sim = kv_sim.view(B, T, HEAD_DIM)

    valid_list, valid_t = make_valid(B, T, valid)

    got = module.mqa_logits_fp4_decode(q_p, q_sf, kv_p, kv_sf, weights, valid_t, out_dtype)
    torch.cuda.synchronize()
    assert got.shape == (B, T), got.shape

    # Per-batch reference over that batch's own [0, valid_b) window, then compare
    # over the WHOLE [B,T] tensor at once (DeepGEMM's methodology; a per-row calc_diff
    # over a single 1024-elem row is far noisier and not the fp4-tolerance target).
    got_rows, ref_rows, sim_rows = [], [], []
    for b in range(B):
        vb = valid_list[b]
        ks = torch.zeros(1, dtype=torch.int32, device="cuda")
        ke = torch.full((1,), vb, dtype=torch.int32, device="cuda")
        ref_b = ref_fp8_mqa_logits(q[b:b+1], kv[b], weights[b:b+1], ks, ke)[0]      # [T]
        sim_b = ref_fp8_mqa_logits(q_sim[b:b+1], kv_sim[b], weights[b:b+1], ks, ke)[0]
        valid_mask = ref_b != float("-inf")
        got_rows.append(torch.where(valid_mask, got[b].float(), torch.zeros_like(got[b].float())))
        ref_rows.append(ref_b.masked_fill(~valid_mask, 0).float())
        sim_rows.append(sim_b.masked_fill(~valid_mask, 0).float())
        if vb < T:  # kernel fills the tail (>= valid_b) with -inf
            assert torch.all(got[b, vb:] == float("-inf")), f"batch {b} tail not -inf"
    G = torch.stack(got_rows); R = torch.stack(ref_rows); S = torch.stack(sim_rows)
    return calc_diff(G, R), calc_diff(G, S)


def test_decode(module):
    print("\n[decode] multi-batch FP4 MQA-logits (compressed, tile-pool, one launch) vs per-batch ref")
    print(f"{'B':>4} {'T':>7} {'valid':>6} {'dtype':>6} {'diff_ref':>12} {'diff_sim':>12} {'result':>8}")
    print("-" * 62)
    ok_all = True
    for out_dtype in (torch.float32, torch.bfloat16):
        # T = per-seq kv slots (DSV4 indexer: slots = ctx/4). Gradient covers
        # 4K -> 128K ctx plus tail-clean, tile-pool balance and mixed-length cases.
        for B, T, valid in [
            (1, 1024, None), (4, 1024, None), (4, 1024, 500), (8, 512, None),
            # context-length gradient: 8K / 32K / 128K ctx (slots = 2K / 8K / 32K)
            (4, 2048, None), (4, 8192, None), (2, 32768, None), (4, 32768, 20000),
            # decode-realistic batch + mixed per-seq lengths (cross-token chunks)
            (32, 4096, None), (32, 4096, "mixed"), (64, 1024, "mixed"),
        ]:
            diff, sim = run_decode(module, B, T, out_dtype, valid)
            ok = diff < 0.02 and sim < 2e-3
            ok_all &= ok
            print(f"{B:4d} {T:7d} {str(valid):>6} {str(out_dtype).split('.')[-1]:>6} "
                  f"{diff:12.4e} {sim:12.4e} {('PASS' if ok else 'FAIL'):>8}")
    print("-" * 62)
    return ok_all


def kernel_us(fn, name_substr="mqa_logits", num_tests=30):
    """Thin adapter over bench_utils.bench_kineto (DeepGEMM's bench, vendored
    verbatim): 8GB L2 flush before EVERY call (cold-HBM KV reads + GPU chill
    time), kineto kernel device-time, warmup cycle discarded, MEAN over
    instances. NOTE: estimator switched from min to DeepGEMM's mean so every
    operator in this repo reports the same number -- expect slightly higher
    values than historical (min-based) tables."""
    return 1e6 * bench_kineto(fn, name_substr, num_tests=num_tests,
                              suppress_kineto_output=True)


BLOCK_Q = 1   # decode: 1 query token per q-block (UMMA_N=64); mirrors the kernel config
BLOCK_KV = 256


def test_main_compressor(module):
    """MAIN compressor fused into the score-attention tail (gemm_fuse_norm_b
    compressor_process_row, d=512 part): per COMPRESS row ((pos+1)%4==0)
      overlap-cat softmax aggregate -> weighted bf16 RMSNorm ->
      RoPE(last 64) -> fp8 e4m3 block-64 quant.
    [B1] state rows are a pos-derived PING-PONG window (physical row =
    (4*(pos//4 % 2) + rr) & 7 for logical row rr); the kernel never writes the
    state, so ALL rows must come back untouched (the old shift is gone).
    Checks vs a torch reference with the same per-step bf16 rounding (softmax /
    RMSNorm reduce ORDER differs -> tolerance-based), logits bitwise unchanged."""
    print("\n[main-compressor] tail port vs torch ref")
    torch.manual_seed(11)
    B, T = 8, 512
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.float32)
    q_p, q_sf, _ = quantize_fp4(q, HEAD_DIM)
    kv_p, kv_sf, _ = quantize_fp4(kv, HEAD_DIM)
    q_p = q_p.view(B, NUM_HEADS, HEAD_DIM // 2).contiguous()
    q_sf = q_sf.view(B, NUM_HEADS).contiguous()
    kv_p = kv_p.view(B, T, HEAD_DIM // 2).contiguous()
    kv_sf = kv_sf.view(B, T).contiguous()
    stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
    ks = torch.arange(B, dtype=torch.int32, device="cuda") * T
    ke = ks + T
    logits_base = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
    module.mqa_logits_fp4_decode_out(q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits_base, 0, 0)
    torch.cuda.synchronize()

    # compressor inputs: rows 3 and 7 are compress rows ((pos+1)%4 == 0)
    pos = torch.arange(B, dtype=torch.int64, device="cuda")
    comp_norm = (torch.rand(512, device="cuda") + 0.5)
    S = 64
    ang = torch.outer(torch.arange(S, device="cuda", dtype=torch.float32),
                      1.0 / (10000.0 ** (torch.arange(32, device="cuda") / 32.0)))
    cos_tab, sin_tab = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
    comp_kv = torch.randn(B, 8, 1024, device="cuda", dtype=torch.float32)
    comp_sc = torch.randn(B, 8, 1024, device="cuda", dtype=torch.float32)
    kv0, sc0 = comp_kv.clone(), comp_sc.clone()
    comp_q8 = torch.full((B, 448), 0xAB, device="cuda", dtype=torch.uint8)   # sentinel
    comp_s8 = torch.full((B, 7), -1.0, device="cuda", dtype=torch.float32)
    comp_rope = torch.zeros(B, 64, device="cuda", dtype=torch.bfloat16)

    logits = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
    module.mqa_logits_fp4_decode_out(
        q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, 0,
        cmp_pos=pos, comp_norm=comp_norm, cos_tab=cos_tab, sin_tab=sin_tab,
        comp_kv=comp_kv, comp_sc=comp_sc, comp_q8=comp_q8, comp_s8=comp_s8,
        comp_rope=comp_rope)
    torch.cuda.synchronize()

    ok = torch.equal(logits, logits_base)
    print(f"  logits bitwise unchanged: {'PASS' if ok else 'FAIL'}")

    idx = torch.arange(512, device="cuda")
    for m in range(B):
        p = int(pos[m])
        if (p + 1) % 4 != 0:   # untouched row
            row_ok = (torch.equal(comp_kv[m], kv0[m]) and torch.equal(comp_sc[m], sc0[m])
                      and bool((comp_q8[m] == 0xAB).all()) and bool((comp_s8[m] == -1.0).all()))
            ok &= row_ok
            print(f"  row {m} (pos {p}, skip): untouched {'PASS' if row_ok else 'FAIL'}")
            continue
        # torch reference; logical row rr lives at physical row (base+rr)&7
        # (B1 ping-pong: pos 3 -> base 0 = identity, pos 7 -> base 4 = flipped)
        base = 4 * ((p >> 2) & 1)
        perm = [(base + r) & 7 for r in range(8)]
        col = torch.stack([idx if r < 4 else idx + 512 for r in range(8)])       # [8,512]
        sc8 = torch.gather(sc0[m][perm], 1, col)
        kv8 = torch.gather(kv0[m][perm], 1, col)
        e = torch.exp(sc8 - sc8.max(0).values)
        agg = (e * kv8).sum(0) / e.sum(0)
        vb = agg.bfloat16().float()
        rms = torch.rsqrt((vb * vb).sum() / 512.0 + 1e-6)
        ro = (vb * rms * comp_norm).bfloat16().float()
        ri = p + 1 - 4
        ev, ov = ro[448::2].clone(), ro[449::2].clone()
        ro[448::2] = (ev * cos_tab[ri] - ov * sin_tab[ri]).bfloat16().float()
        ro[449::2] = (ev * sin_tab[ri] + ov * cos_tab[ri]).bfloat16().float()
        blk = ro[:448].view(7, 64)
        scale_ref = (blk.abs().max(1).values.clamp_min(1e-4)) / 448.0
        # checks (reduce-order ulps -> tolerances; B1: state must be UNTOUCHED)
        state_ok = torch.equal(comp_kv[m], kv0[m]) and torch.equal(comp_sc[m], sc0[m])
        s8_diff = ((comp_s8[m] - scale_ref).abs() / scale_ref).max().item()
        deq = comp_q8[m].view(torch.float8_e4m3fn).float().view(7, 64) * comp_s8[m][:, None]
        q8_diff = ((deq - blk).abs() / (blk.abs() + comp_s8[m][:, None] * 448 * 0.01)).max().item()
        rope_diff = (comp_rope[m].float() - ro[448:]).abs().max().item()
        row_ok = state_ok and s8_diff < 1e-2 and q8_diff < 0.15 and rope_diff < 0.05
        ok &= row_ok
        print(f"  row {m} (pos {p}, compress): state {'ok' if state_ok else 'WRITTEN!'} "
              f"s8 {s8_diff:.2e} q8 {q8_diff:.3f} rope {rope_diff:.3f} "
              f"{'PASS' if row_ok else 'FAIL'}")
    return bool(ok)


def cast_to_fp4_chunked(x2d, chunk_rows=1 << 21):
    """per_token_cast_to_fp4 over row-chunks. The one-shot path materializes several
    full-size intermediates (~5x the bf16 input); at B=256 x T=262144 that would peak
    near 90GB, so bound the peak to ~3GB per 2M-row chunk instead."""
    packed_parts, sf_parts = [], []
    for i in range(0, x2d.shape[0], chunk_rows):
        p, s = per_token_cast_to_fp4(x2d[i:i + chunk_rows])
        packed_parts.append(p)
        sf_parts.append(s.to(torch.int32))
    return torch.cat(packed_parts), torch.cat(sf_parts)


def benchmark(module, sweep_stages=False, fuse_comp=False):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count
    print(f"\nBenchmark decode: {torch.cuda.get_device_name()} ({num_sms} SMs)")
    print("  Tile-pool schedule: grid.x = #SMs, global KV tiles balanced across CTAs.")
    print("  kernel_us = DeepGEMM bench_kineto methodology (profiler schedule w+a, L2")
    print("  flushed with 8GB memset before EVERY call -> cold-HBM KV reads, as in real")
    print("  decode). stg = KV pipeline depth. kernel_us = MIN of per-instance times.")
    if fuse_comp:
        print("  fuse-comp: tail warpgroup hides the MAIN-indexer compressor rows under the")
        print("  KV stream (REALISTIC trigger: staggered positions -> ~B/4 compress rows per")
        print("  step, matching complex_b's cmp_pos semantics in steady-state decode).")
        print("  d_us = all_us - base_us: the compressor fusion's marginal latency.")
        print("  tail_us = attention MOCKED OUT (384 attn threads exit at entry): the 4 tail")
        print("  warps/CTA run the compressor alone, in situ -> the tail's uncovered wall.")
        print("  cmp1_us = standalone compressor kernel (own launch, same estimator);")
        print("  sep_us = base_us + cmp1_us (each op as its OWN kernel; real separate")
        print("  launches would be WORSE by per-launch CPU overhead). fx = sep_us / all_us.")
        print(f"{'B':>4} {'T':>7} {'ctx':>8} {'stg':>5} {'base_us':>9} {'all_us':>8} "
              f"{'d_us':>7} {'tail_us':>8} {'cmp1_us':>8} {'sep_us':>8} {'fx':>5} {'bw_GB/s':>9}")
        print("-" * 100)
    else:
        print("  bytes = q/sf_q/weights reads + KV+SF reads + logits writes (DeepGEMM accounting).")
        print(f"{'B':>4} {'T':>7} {'ctx':>8} {'tiles':>7} {'stg':>5} {'kernel_us':>11} {'TFLOPS':>7} {'bw_GB/s':>9}")
        print("-" * 66)
    stage_opts = (4, 6, 8, 10) if sweep_stages else (0,)
    # Full B x T grid: every batch size covers the complete kv-slot gradient
    # (T = ctx/4 for the DSV4 indexer): 4K / 32K / 128K / 1M context.
    for B in (32, 64, 128, 256):
        for T in (1024, 8192, 32768, 262144):
            torch.manual_seed(0)
            q = torch.randn(B, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(B, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            weights = torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.float32)
            # skip the dequant-sim path (quantize_fp4) and chunk the cast — bounds peak memory
            q_p, q_sf = per_token_cast_to_fp4(q.reshape(-1, HEAD_DIM))
            kv_p, kv_sf = cast_to_fp4_chunked(kv.reshape(-1, HEAD_DIM))
            del q, kv
            q_p = q_p.view(B, NUM_HEADS, HEAD_DIM // 2).contiguous()
            q_sf = q_sf.to(torch.int32).view(B, NUM_HEADS).contiguous()
            kv_p = kv_p.view(B, T, HEAD_DIM // 2).contiguous()
            kv_sf = kv_sf.view(B, T).contiguous()

            # hoist per-call host work out of the timed region (repo *_out convention)
            stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
            logits = torch.full((B, stride), float("-inf"),
                                device="cuda", dtype=torch.float32)
            ks = (torch.arange(B, dtype=torch.int32, device="cuda") * T)
            ke = ks + T
            total_tiles = B * ((T + BLOCK_KV - 1) // BLOCK_KV)

            for stg in stage_opts:
                call = lambda s=stg: module.mqa_logits_fp4_decode_out(
                    q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, s)  # ctas=0 -> per SM
                kus = kernel_us(call)
                eff_stg = stg if stg else 6   # 0 = auto, which resolves to 6
                if fuse_comp:
                    # MAIN compressor, REALISTIC trigger: staggered decode positions
                    # -> (pos+1)%4==0 on exactly B/4 rows, spread one-per-CTA (aligns
                    # complex_b's cmp_pos semantics; full-compress was a false worst case)
                    cpos = torch.arange(B, dtype=torch.int64, device="cuda")
                    cnorm = torch.rand(512, device="cuda") + 0.5
                    ctab = torch.rand(4, 32, device="cuda")
                    stab = torch.rand(4, 32, device="cuda")
                    ckv = torch.randn(B, 8, 1024, device="cuda")
                    csc = torch.randn(B, 8, 1024, device="cuda")
                    cq8 = torch.empty(B, 448, dtype=torch.uint8, device="cuda")
                    cs8 = torch.empty(B, 7, device="cuda")
                    crope = torch.empty(B, 64, dtype=torch.bfloat16, device="cuda")
                    acall = lambda s=stg: module.mqa_logits_fp4_decode_out(
                        q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, s,
                        cmp_pos=cpos, comp_norm=cnorm, cos_tab=ctab, sin_tab=stab,
                        comp_kv=ckv, comp_sc=csc, comp_q8=cq8, comp_s8=cs8, comp_rope=crope)
                    aus = kernel_us(acall)
                    # "each op as its OWN kernel" reference: standalone compressor
                    # (one warp per row, full grid), SAME L2-flushed estimator.
                    cmp1 = kernel_us(lambda: module.mqa_compressor_standalone(
                        cpos, cnorm, ctab, stab, ckv, csc, cq8, cs8, crope, 1e-6),
                                     name_substr="standalone_compressor")
                    sep = kus + cmp1
                    # tail IN SITU solo: attention mocked out (384 threads exit at
                    # entry), same 512-thread launch shape, only the tail warpgroup
                    # works -> the tail's uncovered wall in its real environment.
                    tcall = lambda s=stg: module.mqa_logits_fp4_decode_out(
                        q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, s,
                        cmp_pos=cpos, comp_norm=cnorm, cos_tab=ctab, sin_tab=stab,
                        comp_kv=ckv, comp_sc=csc, comp_q8=cq8, comp_s8=cs8, comp_rope=crope,
                        mock_attn=True)
                    tus = kernel_us(tcall)
                    attn_bytes = (B * NUM_HEADS * (HEAD_DIM // 2 + 4 + 4)
                                  + B * T * (HEAD_DIM // 2 + 4) + B * T * 4)
                    bw = attn_bytes / 1e3 / kus
                    print(f"{B:4d} {T:7d} {4*T:8d} {eff_stg:5d} "
                          f"{kus:9.3f} {aus:8.3f} {aus-kus:7.3f} {tus:8.3f} "
                          f"{cmp1:8.3f} {sep:8.3f} {sep/aus:5.2f} {bw:9.0f}")
                    del cpos, cnorm, ctab, stab, ckv, csc, cq8, cs8, crope
                else:
                    # DeepGEMM test_attention.py accounting (paged decode path):
                    #   reads:  q fp4-packed + sf_q i32 + weights f32, KV fp4-packed 64B + sf 4B per slot
                    #   writes: logits (fp32 here), valid region = B*T
                    q_w_bytes = B * NUM_HEADS * (HEAD_DIM // 2 + 4 + 4)
                    kv_bytes = B * T * (HEAD_DIM // 2 + 4)
                    out_bytes = B * T * 4
                    bw = (q_w_bytes + kv_bytes + out_bytes) / 1e3 / kus
                    tflops = 2 * B * T * NUM_HEADS * HEAD_DIM / 1e6 / kus
                    print(f"{B:4d} {T:7d} {4*T:8d} {total_tiles:7d} {eff_stg:5d} "
                          f"{kus:11.3f} {tflops:7.1f} {bw:9.0f}")
            del weights, q_p, q_sf, kv_p, kv_sf, logits, ks, ke
            torch.cuda.empty_cache()
        print("-" * 66)


def timeline(module, B, T, stg=0):
    """ASCII per-CTA timeline from DEVICE globaltimer stamps (one L2-flushed stamped
    call, gemm_fuse_norm_b prof pattern): directly SHOWS the score-attention path
    (t0->t1) and the compressor tail (t2->t3) running in parallel on each CTA."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count
    torch.manual_seed(0)
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.float32)
    q_p, q_sf = per_token_cast_to_fp4(q.reshape(-1, HEAD_DIM))
    kv_p, kv_sf = cast_to_fp4_chunked(kv.reshape(-1, HEAD_DIM))
    del q, kv
    q_p = q_p.view(B, NUM_HEADS, HEAD_DIM // 2).contiguous()
    q_sf = q_sf.to(torch.int32).view(B, NUM_HEADS).contiguous()
    kv_p = kv_p.view(B, T, HEAD_DIM // 2).contiguous()
    kv_sf = kv_sf.view(B, T).contiguous()
    stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
    logits = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
    ks = torch.arange(B, dtype=torch.int32, device="cuda") * T
    ke = ks + T
    # MAIN compressor (realistic 1/4 trigger) so the phase stamps have work to show
    cpos = torch.arange(B, dtype=torch.int64, device="cuda")
    cnorm = torch.rand(512, device="cuda") + 0.5
    ctab = torch.rand(4, 32, device="cuda")
    stab = torch.rand(4, 32, device="cuda")
    ckv = torch.randn(B, 8, 1024, device="cuda")
    csc = torch.randn(B, 8, 1024, device="cuda")
    cq8 = torch.empty(B, 448, dtype=torch.uint8, device="cuda")
    cs8 = torch.empty(B, 7, device="cuda")
    crope = torch.empty(B, 64, dtype=torch.bfloat16, device="cuda")
    prof_t = torch.zeros(num_sms * 8, dtype=torch.int64, device="cuda")

    fcall = lambda p=None: module.mqa_logits_fp4_decode_out(
        q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, stg,
        prof=p,
        cmp_pos=cpos, comp_norm=cnorm, cos_tab=ctab, sin_tab=stab,
        comp_kv=ckv, comp_sc=csc, comp_q8=cq8, comp_s8=cs8, comp_rope=crope)
    fcall()  # warmup
    torch.cuda.synchronize()
    torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda").zero_()  # cold L2/HBM
    fcall(prof_t)
    torch.cuda.synchronize()

    p = prof_t.view(-1, 8).cpu()
    # Origin = earliest stamp of any kind. The tail stamps at CTA entry while the
    # attention t0 is post-prologue, so min must include t2 or tail bars go negative.
    t0 = min(p[:, 0].min().item(), p[:, 2].min().item())
    span = max(p[:, 1].max().item(), p[:, 3].max().item()) - t0  # ns
    span = max(span, 1)
    WIDTH = 64

    def bar(s_ns, e_ns):
        s = int((s_ns - t0) * WIDTH // span)
        e = max(s + 1, int((e_ns - t0) * WIDTH // span))
        return " " * s + "\u2588" * (e - s) + " " * (WIDTH - e)

    print(f"\nTimeline B={B} T={T} stg={stg if stg else 6}: one stamped call, L2 flushed.")
    print(f"t=0 = earliest stamp (CTA entry); full width = {span/1e3:.1f} us.")
    print("attn = t0->t1 (t0 is POST-prologue: barrier init + TMEM alloc + ks/ke scan);")
    print("tail = t2->t3 (t2 at CTA entry -- the tail does not wait for the prologue).")
    idx = sorted(set([0, 1, num_sms // 4, num_sms // 2, 3 * num_sms // 4, num_sms - 2, num_sms - 1]))
    for i in idx:
        a0, a1 = (p[i, 0].item() - t0) / 1e3, (p[i, 1].item() - t0) / 1e3
        b0, b1 = (p[i, 2].item() - t0) / 1e3, (p[i, 3].item() - t0) / 1e3
        print(f"CTA {i:3d} attn |{bar(p[i, 0].item(), p[i, 1].item())}| {a0:9.2f} -{a1:9.2f} us")
        print(f"        tail |{bar(p[i, 2].item(), p[i, 3].item())}| {b0:9.2f} -{b1:9.2f} us")
    attn_end, tail_end = p[:, 1], p[:, 3]
    tail_dur = (p[:, 3] - p[:, 2]).float() / 1e3
    overlap = (torch.minimum(attn_end, tail_end) - torch.maximum(p[:, 0], p[:, 2])).clamp_min(0).float() / 1e3
    inside = int((tail_end <= attn_end).sum())
    hang = ((tail_end - attn_end).clamp_min(0).float() / 1e3).max().item()
    ratio = (overlap / tail_dur.clamp_min(1e-9)).mean().item() * 100
    print(f"tail fully inside attn window: {inside}/{num_sms} CTAs; "
          f"mean tail = {tail_dur.mean().item():.2f} us, {ratio:.1f}% of it overlapped; "
          f"max hang = {hang:.2f} us")

    # ---- compressor phase breakdown, critical (latest-finishing) compress-row CTA
    # (test_complex.cu pattern); phases relative to tail start. Slots 4/6 retired
    # (rms section gone; shift gone with the B1 ping-pong state).
    if int(p[:, 7].max()) > 0:
        i = int(p[:, 7].argmax())
        t2, t5, t7 = (p[i, k].item() for k in (2, 5, 7))
        print(f"compressor phases (critical CTA {i}, us): "
              f"agg {(t5-t2)/1e3:.2f} | norm+rope+quant {(t7-t5)/1e3:.2f} | "
              f"comp total {(t7-t2)/1e3:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true",
                        help="attention-only benchmark table (default is the fuse-comp table)")
    parser.add_argument("--sweep-stages", action="store_true",
                        help="attention-only table with KV stages 4/6/8/10 per (B,T)")
    parser.add_argument("--timeline", nargs="+", type=int, metavar="N",
                        help="B T [STG]: per-CTA ASCII timeline of attn vs tail from device stamps")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; this test must run on B300.")
        return 0

    torch.set_float32_matmul_precision("highest")
    major, minor = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} sm_{major}{minor} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    print("JIT compiling mqa_logits_fp4.cu ...")
    module = load_cuda_module()

    ok = True
    if not args.skip_correctness:
        ok &= test_single(module)
        ok &= test_decode(module)
        ok &= test_main_compressor(module)
        print("\nALL PASSED" if ok else "\nCORRECTNESS FAILED")
    if not args.skip_bench:
        # default: fuse-comp table; --base / --sweep-stages fall back to attention-only
        benchmark(module, sweep_stages=args.sweep_stages,
                  fuse_comp=not (args.base or args.sweep_stages))
    if args.timeline:
        assert len(args.timeline) >= 2, "--timeline B T [STG]"
        timeline(module, args.timeline[0], args.timeline[1],
                 args.timeline[2] if len(args.timeline) > 2 else 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
