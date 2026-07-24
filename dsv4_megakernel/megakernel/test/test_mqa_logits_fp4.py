"""Correctness (vs per-batch torch ref) + B300 benchmark for the VERBATIM DeepGEMM
sm100 paged MQA-logits kernel vendored in this repo (kernels/mqa_logits_fp4.cu +
include/dg_paged_mqa_logits.cuh; mxfp4, H=64, D=128, next_n=1, page 64).

Perf metric: kernel_us only — DeepGEMM bench_kineto methodology (pure GPU kernel
time, L2 flushed before every call). schedule_meta is precomputed OUTSIDE the
timed fn (== DeepGEMM's own table口径); its cost is reported as a separate
meta_us column (every real decode step pays it).

Entries:
  * get_paged_mqa_logits_metadata — DG schedule kernel ([num_sms+1, 2] i32).
  * mqa_logits_fp4_decode(_out)   — DG fp8_fp4_paged_mqa_logits specialization,
                                    clean_logits=False semantics (RAW row tails);
                                    validated with SHUFFLED page tables over a
                                    context-length gradient (uniform + mixed).
                                    decode_out optionally fuses the DSV4
                                    MAIN-indexer compressor rows into the tail
                                    warpgroup (cmp_* bundle, TPB=512).

Requires: B300 (sm_100+), CUDA >= 12.8. Fully self-contained — no `deep_gemm`
package needed (FP4 quant/dequant + calc_diff are inlined below).

    python test/test_mqa_logits_fp4.py            # correctness, then the benchmark
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


# ------------------------------------------------------------------ multi-batch decode
PAGE_KV = 64      # fused-page tokens (RTP-LLM tokens_per_block); mirrors the kernel
PAGE_BYTES = PAGE_KV * (HEAD_DIM // 2 + 4)   # 4352: fp4 bytes then per-token i32 sf


def build_paged_cache(kv_p, kv_sf, B, T, shuffle=False):
    """kv_p [B*T, D/2] i8 + kv_sf [B*T] i32 (logical order) -> fused page cache
    uint8 [num_blocks, PAGE_BYTES] + block_table [B, T//PAGE_KV] i32.
    Fused page layout (DeepGEMM kv_cache_cast_to_mxfp4-compatible):
    [PAGE_KV*(D/2) fp4 bytes | PAGE_KV*4 sf bytes]. shuffle=True scatters logical
    pages across the physical pool (REAL paged semantics: exercises block_table
    indirection); False keeps identity mapping (contiguous HBM ranges, benchmark)."""
    assert T % PAGE_KV == 0
    num_blocks = B * T // PAGE_KV
    perm = (torch.randperm(num_blocks, device="cuda")
            if shuffle else torch.arange(num_blocks, device="cuda"))
    fused = torch.empty(num_blocks, PAGE_BYTES, device="cuda", dtype=torch.uint8)
    fused[perm, :PAGE_KV * (HEAD_DIM // 2)] = \
        kv_p.contiguous().view(torch.uint8).view(num_blocks, PAGE_KV * (HEAD_DIM // 2))
    fused[perm, PAGE_KV * (HEAD_DIM // 2):] = \
        kv_sf.contiguous().view(num_blocks, PAGE_KV).view(torch.uint8).view(num_blocks, PAGE_KV * 4)
    block_table = perm.to(torch.int32).view(B, T // PAGE_KV).contiguous()
    return fused, block_table


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
    ctx = (valid_t if valid_t is not None
           else torch.full((B,), T, dtype=torch.int32, device="cuda"))
    # SHUFFLED page table: the kernel must reassemble logical order via block_table
    fused, block_table = build_paged_cache(kv_p.view(-1, HEAD_DIM // 2),
                                           kv_sf.view(-1), B, T, shuffle=True)

    got = module.mqa_logits_fp4_decode(q_p, q_sf, fused, weights, ctx, block_table,
                                       T, out_dtype)
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
        # NOTE: entries >= valid_b are RAW garbage (DG clean_logits=False semantics);
        # the masked compare above is the correctness contract.
    G = torch.stack(got_rows); R = torch.stack(ref_rows); S = torch.stack(sim_rows)
    return calc_diff(G, R), calc_diff(G, S)


def test_decode(module):
    print("\n[decode] multi-batch PAGED FP4 MQA-logits (fused pages + shuffled block_table,"
          " one launch) vs per-batch ref")
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
    """MAIN compressor: STANDALONE kernel vs torch ref, then FUSED into the tail
    warpgroup of the attention kernel (decode_out + cmp_* bundle, TPB=512).
    Math unchanged: per COMPRESS row ((pos+1)%4==0) overlap-cat softmax aggregate
    -> weighted bf16 RMSNorm -> RoPE(last 64) -> fp8 e4m3 block-64 quant; [B1]
    pos-derived PING-PONG state window (physical row = (4*(pos//4 % 2) + rr) & 7),
    the kernel never writes the state so ALL rows must come back untouched.
    Checks vs a torch reference with the same per-step bf16 rounding; the fused
    pass must be BITWISE identical on both sides (same code path, same order)."""
    print("\n[main-compressor] standalone kernel vs torch ref")
    torch.manual_seed(11)
    B = 8

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

    module.mqa_compressor_standalone(pos, comp_norm, cos_tab, sin_tab,
                                     comp_kv, comp_sc, comp_q8, comp_s8,
                                     comp_rope, 1e-6)
    torch.cuda.synchronize()

    ok = True

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

    # ---- FUSED pass: the same rows through the attention kernel's tail warpgroup
    # (decode_out + cmp_* bundle). Same math in the same order on both sides:
    #   * compressor outputs must be BITWISE identical to the standalone pass
    #   * logits must be BITWISE identical to the attention-only launch
    q8_ref, s8_ref, rope_ref = comp_q8.clone(), comp_s8.clone(), comp_rope.clone()
    T = 1024
    torch.manual_seed(12)
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.float32)
    q_p, q_sf, _ = quantize_fp4(q, HEAD_DIM)
    kv_p, kv_sf, _ = quantize_fp4(kv, HEAD_DIM)
    q_p = q_p.view(B, NUM_HEADS, HEAD_DIM // 2).contiguous()
    q_sf = q_sf.view(B, NUM_HEADS).contiguous()
    fused_c, block_table = build_paged_cache(kv_p.view(-1, HEAD_DIM // 2),
                                             kv_sf.view(-1), B, T, shuffle=True)
    ctx = torch.full((B,), T, dtype=torch.int32, device="cuda")
    meta = module.get_paged_mqa_logits_metadata(ctx)
    l_base = torch.full((B, T), -3.0, device="cuda", dtype=torch.float32)
    l_fuse = torch.full((B, T), -3.0, device="cuda", dtype=torch.float32)
    module.mqa_logits_fp4_decode_out(q_p, q_sf, fused_c, weights, ctx, block_table,
                                     meta, l_base)
    comp_q8.fill_(0xAB); comp_s8.fill_(-1.0); comp_rope.zero_()
    module.mqa_logits_fp4_decode_out(q_p, q_sf, fused_c, weights, ctx, block_table,
                                     meta, l_fuse,
                                     cmp_pos=pos, comp_norm=comp_norm,
                                     cos_tab=cos_tab, sin_tab=sin_tab,
                                     comp_kv=comp_kv, comp_sc=comp_sc,
                                     comp_q8=comp_q8, comp_s8=comp_s8,
                                     comp_rope=comp_rope, comp_eps=1e-6)
    torch.cuda.synchronize()
    fused_ok = (torch.equal(l_base, l_fuse)
                and torch.equal(comp_q8, q8_ref) and torch.equal(comp_s8, s8_ref)
                and torch.equal(comp_rope, rope_ref)
                and torch.equal(comp_kv, kv0) and torch.equal(comp_sc, sc0))
    ok &= fused_ok
    print(f"  fused tail (decode_out + cmp_*): logits bitwise == attention-only, "
          f"compressor bitwise == standalone: {'PASS' if fused_ok else 'FAIL'}")
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


def benchmark(module):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count
    print(f"\nBenchmark decode: {torch.cuda.get_device_name()} ({num_sms} SMs)")
    print("  Verbatim DeepGEMM sm100 paged kernel (mxfp4, next_n=1, page 64, stages 3/10,")
    print("  splits_per_chunk=16) + fused MAIN-compressor tail warpgroup (TPB=512).")
    print("  kernel_us = bench_kineto (8GB L2 flush before EVERY call, kineto device")
    print("  time, mean); schedule_meta precomputed OUTSIDE the timed fn (DG口径).")
    print("  base = attention only | all = + fused compressor rows | d = all - base")
    print("  tail = compressor alone IN the fused launch shape (attention mocked out)")
    print("  cmp1 = standalone compressor kernel | meta = the per-step metadata kernel")
    print(f"{'B':>4} {'T':>7} {'ctx':>8} {'base_us':>9} {'all_us':>8} {'d_us':>6} "
          f"{'tail_us':>8} {'cmp1':>6} {'meta':>6} {'TFLOPS':>7} {'bw_GB/s':>9}")
    print("-" * 88)
    # Full B x T grid: every batch size covers the complete kv-slot gradient
    # (T = ctx/4 for the DSV4 indexer): 4K / 32K / 128K / 1M context.
    for B in (32, 64, 128, 256):
        # MAIN-compressor inputs (per B): 1/4 of the rows are compress rows
        pos = torch.arange(B, dtype=torch.int64, device="cuda")
        comp_norm = (torch.rand(512, device="cuda") + 0.5)
        ang = torch.outer(torch.arange(64, device="cuda", dtype=torch.float32),
                          1.0 / (10000.0 ** (torch.arange(32, device="cuda") / 32.0)))
        cos_tab, sin_tab = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
        comp_kv = torch.randn(B, 8, 1024, device="cuda", dtype=torch.float32)
        comp_sc = torch.randn(B, 8, 1024, device="cuda", dtype=torch.float32)
        comp_q8 = torch.empty(B, 448, device="cuda", dtype=torch.uint8)
        comp_s8 = torch.empty(B, 7, device="cuda", dtype=torch.float32)
        comp_rope = torch.empty(B, 64, device="cuda", dtype=torch.bfloat16)
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
            # fused page cache with an IDENTITY block table (contiguous HBM ranges;
            # the indirection's correctness is covered by test_decode's shuffled tables)
            fused, block_table = build_paged_cache(kv_p, kv_sf, B, T)
            del kv_p, kv_sf

            # hoist per-call host work out of the timed region (DG methodology):
            # RAW logits buffer (align to SPLIT_KV and 1024B), precomputed meta
            stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
            stride = ((stride + 255) // 256) * 256          # 1024B / fp32
            logits = torch.empty(B, stride, device="cuda", dtype=torch.float32)
            ctx = torch.full((B,), T, dtype=torch.int32, device="cuda")
            meta = module.get_paged_mqa_logits_metadata(ctx)

            comp_kwargs = dict(cmp_pos=pos, comp_norm=comp_norm,
                               cos_tab=cos_tab, sin_tab=sin_tab,
                               comp_kv=comp_kv, comp_sc=comp_sc,
                               comp_q8=comp_q8, comp_s8=comp_s8,
                               comp_rope=comp_rope, comp_eps=1e-6)
            call_base = lambda: module.mqa_logits_fp4_decode_out(
                q_p, q_sf, fused, weights, ctx, block_table, meta, logits)
            call_all = lambda: module.mqa_logits_fp4_decode_out(
                q_p, q_sf, fused, weights, ctx, block_table, meta, logits, **comp_kwargs)
            call_tail = lambda: module.mqa_logits_fp4_decode_out(
                q_p, q_sf, fused, weights, ctx, block_table, meta, logits,
                mock_attn=True, **comp_kwargs)
            base = kernel_us(call_base)
            allu = kernel_us(call_all)
            tail = kernel_us(call_tail)
            cmp1 = kernel_us(lambda: module.mqa_compressor_standalone(
                                 pos, comp_norm, cos_tab, sin_tab, comp_kv, comp_sc,
                                 comp_q8, comp_s8, comp_rope, 1e-6),
                             name_substr="compressor")
            mus = kernel_us(lambda: module.get_paged_mqa_logits_metadata(ctx),
                            name_substr="metadata")

            # DeepGEMM test_attention.py accounting (paged decode path)
            q_w_bytes = B * NUM_HEADS * (HEAD_DIM // 2 + 4 + 4)
            kv_bytes = B * T * (HEAD_DIM // 2 + 4)
            out_bytes = B * T * 4
            bw = (q_w_bytes + kv_bytes + out_bytes) / 1e3 / base
            tflops = 2 * B * T * NUM_HEADS * HEAD_DIM / 1e6 / base
            print(f"{B:4d} {T:7d} {4*T:8d} {base:9.3f} {allu:8.3f} {allu-base:6.2f} "
                  f"{tail:8.3f} {cmp1:6.2f} {mus:6.2f} {tflops:7.1f} {bw:9.0f}")
            del weights, q_p, q_sf, fused, block_table, logits, ctx, meta
            torch.cuda.empty_cache()
        print("-" * 88)


def main():
    parser = argparse.ArgumentParser()
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
        ok &= test_decode(module)
        ok &= test_main_compressor(module)
        print("\nALL PASSED" if ok else "\nCORRECTNESS FAILED")
    if not args.skip_bench:
        benchmark(module)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
