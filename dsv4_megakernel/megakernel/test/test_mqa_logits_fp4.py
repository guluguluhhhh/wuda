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

    python test/test_mqa_logits_fp4.py
    python test/test_mqa_logits_fp4.py --benchmark
"""

import argparse
import os
import sys

import torch

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


def kernel_us(fn, name_substr="mqa_logits", num_tests=30, flush_l2=True, cooldown=0.5):
    """Pure GPU kernel time, DeepGEMM bench_kineto framing (profiler schedule w+a,
    L2 flushed with an 8GB memset before EVERY call -> cold-HBM KV reads).
    Estimator: MIN over the per-instance kernel durations of the active cycle
    (unthrottled capability; DVFS-throttled instances are excluded by construction,
    unlike DeepGEMM's table average). `cooldown` lets clocks recover between cells."""
    import time
    from torch.profiler import profile, ProfilerActivity, schedule
    flush_l2_size = int(8e9 // 4)
    fn()  # warm the JIT/first-call path before profiling
    torch.cuda.synchronize()
    if cooldown > 0:
        time.sleep(cooldown)
    try:
        prof = profile(activities=[ProfilerActivity.CUDA],
                       schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
                       acc_events=True)
    except TypeError:  # older torch without acc_events
        prof = profile(activities=[ProfilerActivity.CUDA],
                       schedule=schedule(wait=0, warmup=1, active=1, repeat=1))
    with prof:
        for _ in range(2):  # cycle 1 = warmup (discarded), cycle 2 = active (measured)
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                fn()
            torch.cuda.synchronize()
            prof.step()
    # Per-instance durations (us) of the matching kernel in the active cycle
    insts = []
    for e in prof.events():
        if name_substr not in e.name.lower():
            continue
        d = getattr(e, "device_time", None)
        if d is None:
            d = getattr(e, "cuda_time", 0.0)
        if d and d > 0:
            insts.append(float(d))
    assert insts, f"no kernel matching '{name_substr}' in profile"
    return min(insts)


BLOCK_Q = 1   # decode: 1 query token per q-block (UMMA_N=64); mirrors the kernel config
BLOCK_KV = 256


def rmsnorm_ref(y, eps=1e-6):
    """model.py L498: q *= rsqrt(q.square().mean(-1) + eps), weightless, head_dim=512."""
    yv = y.view(-1, 512)
    return (yv * torch.rsqrt(yv.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)


def test_fused_rmsnorm(module):
    """Correctness of the hidden wq_b per-head RMSNorm (tail warpgroup):
      * qn vs torch ref (bf16 tolerance: reduce-order differs -> <= ~1 bf16 ulp);
      * logits with rms active must be BITWISE identical to the base run (the tail
        must not perturb the score-attention paths in any way).
    Row counts include non-multiples of the warp grid to exercise the grid-stride tail."""
    print("\n[fused-rmsnorm] tail warpgroup vs torch ref (+ logits bitwise unchanged)")
    print(f"{'rows':>6} {'groups':>7} {'max_diff':>10} {'logits':>8} {'result':>8}")
    print("-" * 46)
    torch.manual_seed(7)
    B, T = 8, 512   # minimal attention workload; rms correctness is workload-independent
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

    ok_all = True
    for rows in (1, 3, 32, 256):   # 128 / 384 / 4096 / 32768 groups
        y = torch.randn(rows, 128 * 512, device="cuda", dtype=torch.float32)
        qn = torch.empty(rows, 128 * 512, device="cuda", dtype=torch.bfloat16)
        logits = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
        module.mqa_logits_fp4_decode_out(q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, 0,
                                         rms_y=y, rms_out=qn, rms_eps=1e-6)
        torch.cuda.synchronize()
        diff = (qn.view(-1, 512).float() - rmsnorm_ref(y).float()).abs().max().item()
        logits_same = torch.equal(logits, logits_base)
        ok = diff < 0.05 and logits_same   # <= ~1 bf16 ulp at |x|<=4 + reduce-order noise
        ok_all &= ok
        print(f"{rows:6d} {rows*128:7d} {diff:10.4e} {('same' if logits_same else 'DIFF'):>8} "
              f"{('PASS' if ok else 'FAIL'):>8}")
    print("-" * 46)
    return ok_all


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


def benchmark(module, sweep_stages=False, fuse_norm=False, rms_delay_us=0.0):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count
    print(f"\nBenchmark decode: {torch.cuda.get_device_name()} ({num_sms} SMs)")
    print("  Tile-pool schedule: grid.x = #SMs, global KV tiles balanced across CTAs.")
    print("  kernel_us = DeepGEMM bench_kineto methodology (profiler schedule w+a, L2")
    print("  flushed with 8GB memset before EVERY call -> cold-HBM KV reads, as in real")
    print("  decode). stg = KV pipeline depth. kernel_us = MIN of per-instance times.")
    if fuse_norm:
        print("  fuse-norm: tail warpgroup hides the wq_b per-head RMSNorm ([B,128,512]")
        print("  fp32 -> bf16) under the KV stream.")
        print("  bw0 = attention bytes / base_us; bw1 = (attention + rmsnorm bytes) / fused_us")
        print("  -> TOTAL achieved HBM bandwidth before/after fusion (bw1 > bw0 means the")
        print("  fused kernel moves more bytes per second, i.e. the tail rides free-ish).")
        print("  d_us = fused_us - base_us: the latency the fusion adds.")
        if rms_delay_us > 0:
            print(f"  tail deferred start: {rms_delay_us:.1f} us (head-start hypothesis knob)")
        print(f"{'B':>4} {'T':>7} {'ctx':>8} {'stg':>5} {'base_us':>9} {'fused_us':>9} "
              f"{'d_us':>7} {'bw0_GB/s':>9} {'bw1_GB/s':>9}")
        print("-" * 74)
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
                if fuse_norm:
                    # wq_b main-attention q: [B, 128 heads x 512], fp32 GEMM out -> bf16 normed
                    y = torch.randn(B, 128 * 512, device="cuda", dtype=torch.float32)
                    qn = torch.empty(B, 128 * 512, device="cuda", dtype=torch.bfloat16)
                    fcall = lambda s=stg: module.mqa_logits_fp4_decode_out(
                        q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, s,
                        rms_y=y, rms_out=qn, rms_eps=1e-6, rms_delay_us=rms_delay_us)
                    fus = kernel_us(fcall)
                    # correctness sentinel (not printed; loud failure only)
                    diff = (qn.view(-1, 512).float() - rmsnorm_ref(y).float()).abs().max().item()
                    assert diff < 0.05, f"fused rmsnorm diverged: max diff {diff}"
                    # TOTAL achieved HBM bandwidth, DeepGEMM byte accounting:
                    #   attention: q/sf_q/weights reads + KV+SF reads + logits writes
                    #   rmsnorm:   y fp32 read + q_normed bf16 write
                    attn_bytes = (B * NUM_HEADS * (HEAD_DIM // 2 + 4 + 4)
                                  + B * T * (HEAD_DIM // 2 + 4) + B * T * 4)
                    tail_bytes = B * 128 * 512 * (4 + 2)
                    bw0 = attn_bytes / 1e3 / kus
                    bw1 = (attn_bytes + tail_bytes) / 1e3 / fus
                    print(f"{B:4d} {T:7d} {4*T:8d} {eff_stg:5d} "
                          f"{kus:9.3f} {fus:9.3f} {fus-kus:7.3f} {bw0:9.0f} {bw1:9.0f}")
                    del y, qn
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
    (t0->t1) and the RMSNorm tail (t2->t3) running in parallel on each CTA."""
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
    y = torch.randn(B, 128 * 512, device="cuda", dtype=torch.float32)
    qn = torch.empty(B, 128 * 512, device="cuda", dtype=torch.bfloat16)
    prof_t = torch.zeros(num_sms * 4, dtype=torch.int64, device="cuda")

    fcall = lambda p=None: module.mqa_logits_fp4_decode_out(
        q_p, q_sf, kv_p, kv_sf, weights, ks, ke, logits, 0, stg,
        rms_y=y, rms_out=qn, rms_eps=1e-6, prof=p)
    fcall()  # warmup
    torch.cuda.synchronize()
    torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda").zero_()  # cold L2/HBM
    fcall(prof_t)
    torch.cuda.synchronize()

    p = prof_t.view(-1, 4).cpu()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sweep-stages", action="store_true",
                        help="benchmark every (B,T) with KV stages 4/6/8/10 to fit the dispatch threshold")
    parser.add_argument("--fuse-norm", action="store_true",
                        help="benchmark base vs fused (tail warpgroup hides wq_b per-head RMSNorm)")
    parser.add_argument("--rms-delay-us", type=float, default=0.0,
                        help="defer the tail start by N us (tests the head-start-slows-attention hypothesis)")
    parser.add_argument("--timeline", nargs="+", type=int, metavar="N",
                        help="B T [STG]: per-CTA ASCII timeline of attn vs tail from device stamps")
    parser.add_argument("--skip-correctness", action="store_true")
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
        ok &= test_fused_rmsnorm(module)
        print("\nALL PASSED" if ok else "\nCORRECTNESS FAILED")
    if args.benchmark or args.sweep_stages or args.fuse_norm:
        benchmark(module, sweep_stages=args.sweep_stages, fuse_norm=args.fuse_norm,
                  rms_delay_us=args.rms_delay_us)
    if args.timeline:
        assert len(args.timeline) >= 2, "--timeline B T [STG]"
        timeline(module, args.timeline[0], args.timeline[1],
                 args.timeline[2] if len(args.timeline) > 2 else 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
