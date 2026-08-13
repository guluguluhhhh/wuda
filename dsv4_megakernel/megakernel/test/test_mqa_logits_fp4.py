"""Correctness (vs DeepGEMM ref_fp8_mqa_logits) + B300 benchmark for the migrated
FP4 MQA-logits kernel (kernels/mqa_logits_fp4.cu), PAGED decode only.

Perf metric: kernel_us only — DeepGEMM bench_kineto methodology (pure GPU kernel
time, L2 flushed before every call); DeepGEMM's test suite reports no wall time.

Entry point:
  * mqa_logits_fp4_decode(_out) — MULTI-BATCH PAGED decode, compressed +
    self-clean, ONE launch (IN-KERNEL tile-pool schedule: grid.x = #SMs, global
    KV tiles balanced across CTAs — NO metadata kernel, no schedule_meta).
    kv_cache = fused pages [num_blocks, PAGE_KV*(D/2+4)] bytes + block_table;
    validated with SHUFFLED page tables over a context-length GRADIENT
    (uniform + mixed per-seq context_lens).

Requires: B300 (sm_100+), CUDA >= 12.8. Fully self-contained — no `deep_gemm`
package needed (FP4 quant/dequant + calc_diff are inlined below).

    python test/test_mqa_logits_fp4.py            # correctness, then the fuse-comp table
    python test/test_mqa_logits_fp4.py --base     # attention-only table instead
    python test/test_mqa_logits_fp4.py --fp8      # FP8 vs DeepGEMM + fused MAIN
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

NUM_HEADS = 64
HEAD_DIM = 128


def _deep_gemm_dir():
    override = os.environ.get("DEEP_GEMM_DIR")
    if override:
        candidates = [os.path.abspath(override)]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.abspath(os.path.join(
                here, "..", "..", "..", "DeepGEMM")),
            os.path.abspath(os.path.join(
                here, "..", "..", "..", "..", "DeepGEMM")),
        ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "deep_gemm", "include")):
            return candidate
    source = "DEEP_GEMM_DIR" if override else "local checkout"
    raise FileNotFoundError(
        f"{source} has no deep_gemm/include directory: {candidates}")


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
        # CUTE_ARCH_* gates are NOT hand-forced: sm103_cutlass_shim.h routes
        # sm_103a through CUTLASS's own arch table (no-op on CUTLASS >= 4.2).
        "-include", os.path.join(proj_dir, "include", "sm103_cutlass_shim.h"),
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


def load_cuda_module_fp8():
    """Build the standalone FP8+MAIN implementation from its own translation unit."""
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)
    deep_gemm_dir = _deep_gemm_dir()
    cutlass_dir = os.path.join(proj_dir, "..", "cutlass", "include")
    cutlass_tools_dir = os.path.join(
        proj_dir, "..", "cutlass", "tools", "util", "include")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    cuda_flags = [
        "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
        "-lineinfo", "--ptxas-options=-v",
        "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
        "-include", os.path.join(proj_dir, "include", "sm103_cutlass_shim.h"),
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-diag-suppress=3288",
        f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
    ]
    return load(
        name="mqa_logits_fp8_fused",
        sources=[os.path.join(proj_dir, "kernels", "mqa_logits_fp8.cu")],
        extra_include_paths=[
            os.path.join(proj_dir, "include"),
            os.path.join(deep_gemm_dir, "deep_gemm", "include"),
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


def build_paged_cache(kv_p, kv_sf, B, T, shuffle=False,
                      entries_per_block=PAGE_KV, block_stride_bytes=None):
    """kv_p [B*T, D/2] i8 + kv_sf [B*T] i32 (logical order) -> fused page cache
    uint8 [num_blocks, PAGE_BYTES] + block_table [B, T//PAGE_KV] i32.
    Fused page layout (DeepGEMM kv_cache_cast_to_mxfp4-compatible):
    [PAGE_KV*(D/2) fp4 bytes | PAGE_KV*4 sf bytes]. shuffle=True scatters logical
    pages across the physical pool (REAL paged semantics: exercises block_table
    indirection); False keeps identity mapping (contiguous HBM ranges, benchmark)."""
    epb = entries_per_block
    assert T % epb == 0
    payload_bytes = epb * (HEAD_DIM // 2 + 4)
    stride_bytes = block_stride_bytes or payload_bytes
    assert stride_bytes >= payload_bytes and stride_bytes % 16 == 0
    num_blocks = B * T // epb
    perm = (torch.randperm(num_blocks, device="cuda")
            if shuffle else torch.arange(num_blocks, device="cuda"))
    fused = torch.full((num_blocks, stride_bytes), 0xA5, device="cuda",
                       dtype=torch.uint8)
    fused[perm, :epb * (HEAD_DIM // 2)] = \
        kv_p.contiguous().view(torch.uint8).view(num_blocks, epb * (HEAD_DIM // 2))
    fused[perm, epb * (HEAD_DIM // 2):payload_bytes] = \
        kv_sf.contiguous().view(num_blocks, epb).view(torch.uint8).view(num_blocks, epb * 4)
    block_table = perm.to(torch.int32).view(B, T // epb).contiguous()
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


def run_decode(module, B, T, out_dtype, valid=None, entries_per_block=PAGE_KV,
               block_stride_padding=0):
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
    payload = entries_per_block * (HEAD_DIM // 2 + 4)
    stride_bytes = payload + block_stride_padding
    fused, block_table = build_paged_cache(
        kv_p.view(-1, HEAD_DIM // 2), kv_sf.view(-1), B, T, shuffle=True,
        entries_per_block=entries_per_block,
        block_stride_bytes=stride_bytes)

    got = module.mqa_logits_fp4_decode(q_p, q_sf, fused, weights, ctx, block_table,
                                       T, out_dtype,
                                       kv_entries_per_block=entries_per_block,
                                       kv_block_stride_bytes=stride_bytes)
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
    print("\n[decode] multi-batch PAGED FP4 MQA-logits (fused pages + shuffled block_table,"
          " in-kernel tile-pool, one launch) vs per-batch ref")
    print(f"{'B':>4} {'T':>7} {'valid':>6} {'dtype':>6} {'diff_ref':>12} {'diff_sim':>12} {'result':>8}")
    print("-" * 62)
    ok_all = True
    for out_dtype in (torch.float32, torch.bfloat16):
        # T = per-seq kv slots (DSV4 indexer: slots = ctx/4). Gradient covers
        # 4K -> 128K ctx plus tail-clean, tile-pool balance and mixed-length cases.
        for B, T, valid in [
            (1, 1024, None), (4, 1024, None), (4, 1024, 500), (8, 512, None),
            # context gradient: 8K / 32K / 64K / 128K ctx
            # (slots = 2K / 8K / 16K / 32K)
            (4, 2048, None), (4, 8192, None), (2, 16384, None),
            (2, 32768, None), (4, 32768, 20000),
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


def test_runtime_page_geometry(module):
    """RTP logical entries and physical block stride are independent."""
    print("\n[decode] runtime page geometry with physical padding")
    ok_all = True
    for epb in (32, 64, 128):
        diff, sim = run_decode(module, 2, 256, torch.float32,
                               entries_per_block=epb,
                               block_stride_padding=128)
        ok = diff < 0.02 and sim < 2e-3
        ok_all &= ok
        print(f"  epb={epb:<3} padded_stride: diff_ref={diff:.3e} "
              f"diff_sim={sim:.3e} {'PASS' if ok else 'FAIL'}")
    return ok_all


def test_query_rms_rope(module):
    """Idle MQA warps produce the normalized, RoPE-applied FlashMLA query."""
    print("\n[query-rms-rope] fused MQA worker path vs torch ref")
    torch.manual_seed(20260814)
    dev, batch = "cuda", 3

    # Minimal valid attention tensors; mock_attn leaves all non-tail warps
    # available to exercise the shared query work queue.
    q = torch.zeros(batch, NUM_HEADS, HEAD_DIM // 2, device=dev,
                    dtype=torch.int8)
    q_sf = torch.zeros(batch, NUM_HEADS, device=dev, dtype=torch.int32)
    cache = torch.zeros(batch, PAGE_BYTES, device=dev, dtype=torch.uint8)
    weights = torch.zeros(batch, NUM_HEADS, device=dev)
    context_lens = torch.ones(batch, device=dev, dtype=torch.int32)
    block_table = torch.arange(batch, device=dev, dtype=torch.int32).view(batch, 1)
    logits = torch.empty(batch, BLOCK_KV, device=dev)

    positions = torch.tensor([0, 7, 31], device=dev, dtype=torch.int64)
    angles = torch.outer(
        torch.arange(32, device=dev, dtype=torch.float32),
        1.0 / (10000.0 ** (torch.arange(32, device=dev) / 32.0)))
    cos_tab, sin_tab = torch.cos(angles).contiguous(), torch.sin(angles).contiguous()
    eps = 1e-6
    ok_all = True
    for input_heads, output_heads in ((64, 64), (64, 128), (128, 128)):
        query_x = torch.randn(batch, input_heads, 512, device=dev,
                              dtype=torch.bfloat16)
        query_out = torch.empty(batch, output_heads, 512, device=dev,
                                dtype=torch.bfloat16)
        module.mqa_logits_fp4_decode_out(
            q, q_sf, cache, weights, context_lens, block_table, logits, 0, 0,
            mock_attn=True, query_x=query_x, query_positions=positions,
            query_cos=cos_tab, query_sin=sin_tab, query_out=query_out,
            query_input_heads=input_heads, query_eps=eps)
        torch.cuda.synchronize()

        source = query_x.repeat(1, output_heads // input_heads, 1)
        source_f = source.float()
        ref = source_f * torch.rsqrt(source_f.square().mean(-1, keepdim=True) + eps)
        even, odd = ref[..., 448::2].clone(), ref[..., 449::2].clone()
        cos = cos_tab[positions, None, :]
        sin = sin_tab[positions, None, :]
        ref[..., 448::2] = even * cos - odd * sin
        ref[..., 449::2] = even * sin + odd * cos
        diff = (query_out.float() - ref.bfloat16().float()).abs().max().item()
        ok = diff == 0.0
        ok_all &= ok
        print(f"  heads={input_heads}->{output_heads}: max_abs={diff:.3e} "
              f"{'PASS' if ok else 'FAIL'}")
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


# ------------------------------------------------------------------ FP8 MQA + MAIN

def get_deep_gemm():
    """Resolve the checkout explicitly so an older installed wheel cannot win."""
    deep_gemm_dir = _deep_gemm_dir()
    if deep_gemm_dir not in sys.path:
        sys.path.insert(0, deep_gemm_dir)
    import deep_gemm
    return deep_gemm


def build_fp8_paged_cache(kv, entries_per_block=64, shuffle=True,
                          block_stride_padding=0):
    """Quantize [B,T,128] BF16 into DeepGEMM's planar FP8 paged ABI.

    The object passed to DeepGEMM has logical shape [pages, page, 1, 132], but
    each physical page is planar: all E4M3 K bytes, then all FP32 token scales.
    """
    B, T, D = kv.shape
    assert D == HEAD_DIM and T % entries_per_block == 0
    epb = entries_per_block
    num_blocks = B * T // epb
    amax = kv.abs().float().amax(dim=-1).clamp_min(1e-4)
    scale = amax / 448.0
    kv_fp8 = (kv * scale.reciprocal().unsqueeze(-1)).to(torch.float8_e4m3fn)
    kv_sim = (kv_fp8.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

    payload = epb * (D + 4)
    stride = payload + block_stride_padding
    assert stride >= payload and stride % 16 == 0
    permutation = (torch.randperm(num_blocks, device="cuda") if shuffle else
                   torch.arange(num_blocks, device="cuda"))
    pages = torch.full((num_blocks, stride), 0xA5, dtype=torch.uint8,
                       device="cuda")
    pages[permutation, :epb * D] = kv_fp8.contiguous().view(
        num_blocks, epb * D).view(torch.uint8)
    pages[permutation, epb * D:payload] = scale.contiguous().view(
        num_blocks, epb).view(torch.uint8).view(num_blocks, epb * 4)
    block_table = permutation.to(torch.int32).view(B, T // epb).contiguous()
    dg_view = pages.as_strided(
        (num_blocks, epb, 1, D + 4),
        (pages.stride(0), D + 4, D + 4, 1))
    return pages, dg_view, block_table, kv_sim


def make_fp8_attention_case(B, T, entries_per_block=64,
                            block_stride_padding=0, mixed_context=False):
    dg = get_deep_gemm()
    torch.manual_seed(B * 1009 + T * 7 + entries_per_block)
    q_bf16 = torch.randn(B, NUM_HEADS, HEAD_DIM, device="cuda",
                         dtype=torch.bfloat16)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn).contiguous()
    kv = torch.randn(B, T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    pages, dg_pages, block_table, kv_sim = build_fp8_paged_cache(
        kv, entries_per_block, shuffle=True,
        block_stride_padding=block_stride_padding)

    # The RTP producer folds its per-head Q scale into this tensor. Random
    # positive scales exercise exactly that ABI while keeping Q itself plain E4M3.
    original_weights = torch.randn(B, NUM_HEADS, device="cuda")
    q_scale = torch.rand(B, NUM_HEADS, device="cuda") * 1.5 + 0.25
    folded_weights = (original_weights * q_scale).contiguous()
    if mixed_context:
        lo = max(1, entries_per_block // 2)
        context = torch.linspace(lo, T, B, device="cuda").to(torch.int32)
    else:
        context = torch.full((B,), T, device="cuda", dtype=torch.int32)
    context_2d = context.view(B, 1).contiguous()
    schedule = dg.get_paged_mqa_logits_metadata(
        context_2d, entries_per_block, dg.get_num_sms())
    return dict(
        q=q_fp8, q_sim=q_fp8.to(torch.bfloat16), kv=kv, kv_sim=kv_sim,
        pages=pages, dg_pages=dg_pages, weights=folded_weights,
        context=context, context_2d=context_2d, block_table=block_table,
        schedule=schedule, entries_per_block=entries_per_block,
        block_stride_bytes=pages.size(1), T=T)


def call_wuda_fp8(module, case, logits, num_kv_stages=0, **compressor):
    module.mqa_logits_fp8_decode_out(
        case["q"], case["pages"], case["weights"], case["context_2d"],
        case["block_table"], case["schedule"], logits,
        case["entries_per_block"], case["block_stride_bytes"],
        num_kv_stages, 1e-6, **compressor)


def call_deep_gemm_fp8(case):
    dg = get_deep_gemm()
    return dg.fp8_paged_mqa_logits(
        case["q"].view(case["q"].size(0), 1, NUM_HEADS, HEAD_DIM),
        case["dg_pages"], case["weights"], case["context_2d"],
        case["block_table"], case["schedule"], case["T"])


def test_fp8_attention(module):
    print("\n[fp8-attention] Wuda vs deep_gemm.fp8_paged_mqa_logits")
    print(f"{'page':>6} {'padding':>8} {'valid':>8} {'bitwise':>9} "
          f"{'max_abs':>10} {'values':>9} {'result':>8}")
    print("-" * 68)
    ok_all = True
    for epb, padding, mixed in ((32, 128, True), (64, 0, False),
                                (128, 256, True)):
        case = make_fp8_attention_case(
            8, 512, epb, block_stride_padding=padding,
            mixed_context=mixed)
        sentinel = -12345.0
        logits = torch.full((8, 512), sentinel, device="cuda")
        call_wuda_fp8(module, case, logits)
        reference = call_deep_gemm_fp8(case)
        torch.cuda.synchronize()
        positions = torch.arange(512, device="cuda")[None, :]
        valid = positions < case["context"][:, None]
        bitwise = torch.equal(logits[valid], reference[valid])
        max_abs = (logits[valid] - reference[valid]).abs().max().item()
        # DeepGEMM's clean_logits=False contract writes whole 256-token splits;
        # values at positions >= context_len are unspecified and not consumed.
        ok = bitwise
        ok_all &= ok
        valid_desc = "mixed" if mixed else "full"
        print(f"{epb:6d} {padding:8d} {valid_desc:>8} {str(bitwise):>9} "
              f"{max_abs:10.3e} {int(valid.sum()):9d} "
              f"{('PASS' if ok else 'FAIL'):>8}")
        del case, logits, reference
    return ok_all


def make_main_compressor_inputs(B):
    positions = torch.arange(B, dtype=torch.int64, device="cuda")
    norm = torch.rand(512, device="cuda") + 0.5
    angle = torch.outer(
        torch.arange(max(B, 8), device="cuda", dtype=torch.float32),
        1.0 / (10000.0 ** (torch.arange(32, device="cuda") / 32.0)))
    cos_tab = torch.cos(angle).contiguous()
    sin_tab = torch.sin(angle).contiguous()
    state = torch.randn(B, 8, 2048, device="cuda")
    state_row = (torch.arange(B, device="cuda") * 8 + positions % 8).long()
    return positions, norm, cos_tab, sin_tab, state, state_row


def test_fp8_main_compressor(module):
    """Fused and standalone paths share the device function and must be exact."""
    print("\n[fp8-main] fused attention + MAIN compressor")
    B, T = 8, 512
    case = make_fp8_attention_case(B, T, 64)
    pos, norm, cos_tab, sin_tab, state, state_row = \
        make_main_compressor_inputs(B)
    state_before = state.clone()
    base_logits = torch.full((B, T), -12345.0, device="cuda")
    fused_logits = torch.full_like(base_logits, -12345.0)
    call_wuda_fp8(module, case, base_logits)

    fused_q8 = torch.full((B, 448), 0xAB, dtype=torch.uint8, device="cuda")
    fused_s8 = torch.full((B, 7), -1.0, device="cuda")
    fused_rope = torch.full((B, 64), -1.0, dtype=torch.bfloat16, device="cuda")
    ref_q8 = fused_q8.clone()
    ref_s8 = fused_s8.clone()
    ref_rope = fused_rope.clone()

    cmp_epb = 4
    cmp_payload = cmp_epb * (576 + 8)
    cmp_stride = cmp_payload + 128
    cmp_cache = torch.full((2, cmp_stride), 0xA5, dtype=torch.uint8,
                           device="cuda")
    cmp_dst = torch.arange(B, dtype=torch.int64, device="cuda")
    compressor = dict(
        cmp_pos=pos, comp_norm=norm, cos_tab=cos_tab, sin_tab=sin_tab,
        comp_state=state, comp_state_row=state_row,
        comp_state_ring_entries=8, comp_q8=fused_q8, comp_s8=fused_s8,
        comp_rope=fused_rope, cmp_cache=cmp_cache, cmp_dst=cmp_dst,
        cmp_entries_per_block=cmp_epb,
        cmp_block_stride_bytes=cmp_stride)
    call_wuda_fp8(module, case, fused_logits, **compressor)
    module.mqa_compressor_fp8_standalone(
        pos, norm, cos_tab, sin_tab, state, state_row, 8,
        ref_q8, ref_s8, ref_rope, 1e-6)
    torch.cuda.synchronize()

    logits_ok = torch.equal(base_logits, fused_logits)
    compact_ok = (torch.equal(fused_q8, ref_q8) and
                  torch.equal(fused_s8, ref_s8) and
                  torch.equal(fused_rope, ref_rope))
    state_ok = torch.equal(state, state_before)
    trigger = ((pos + 1) % 4 == 0).cpu().tolist()
    cache_ok = True
    cache_cpu = cmp_cache.cpu()
    q8_cpu, s8_cpu = fused_q8.cpu(), fused_s8.cpu()
    rope_bytes = fused_rope.contiguous().view(torch.uint8).cpu()
    for row, writes in enumerate(trigger):
        page, offset = row // cmp_epb, row % cmp_epb
        body = cache_cpu[page, offset * 576:(offset + 1) * 576]
        scale_offset = cmp_epb * 576 + offset * 8
        scale_record = cache_cpu[page, scale_offset:scale_offset + 8]
        if writes:
            expected_scale = (s8_cpu[row].log2() + 127).to(torch.uint8)
            cache_ok &= torch.equal(body[:448], q8_cpu[row])
            cache_ok &= torch.equal(body[448:], rope_bytes[row])
            cache_ok &= torch.equal(scale_record[:7], expected_scale)
            cache_ok &= bool(scale_record[7] == 0xA5)
        else:
            cache_ok &= bool((body == 0xA5).all())
            cache_ok &= bool((scale_record == 0xA5).all())
    cache_ok &= bool((cache_cpu[:, cmp_payload:] == 0xA5).all())

    checks = (("attention bitwise unchanged", logits_ok),
              ("standalone compressor bitwise match", compact_ok),
              ("compressor state read-only", state_ok),
              ("MODEL1 paged write + canaries", cache_ok))
    for label, passed in checks:
        print(f"  {label:<38} {'PASS' if passed else 'FAIL'}")
    return all(passed for _, passed in checks)


def benchmark_fp8(module, num_tests=30):
    """Cold-HBM comparison; Wuda auto-selects 3/4/5 KV stages."""
    print("\n[fp8-perf] DeepGEMM vs Wuda FP8 attention and fused MAIN")
    print("  All values are mean CUDA kernel time with 8GB L2 flush per call.")
    print(f"{'B':>4} {'T':>7} {'stg':>4} {'DG_us':>9} {'Wuda_us':>9} {'W/DG':>7} "
          f"{'cmp_us':>8} {'fused_us':>10} {'fuse-base':>10} {'result':>8}")
    print("-" * 96)
    ok_all = True
    for B, T in ((16, 1024), (16, 8192), (64, 8192), (128, 16384)):
        case = make_fp8_attention_case(B, T, 64)
        logits = torch.empty((B, T), device="cuda")
        splits_per_cta = (B * ((T + 255) // 256) +
                          get_deep_gemm().get_num_sms() - 1) // \
            get_deep_gemm().get_num_sms()
        stage = 3 if splits_per_cta <= 1 else (4 if splits_per_cta <= 8 else 5)
        base_call = lambda: call_wuda_fp8(module, case, logits, 0)
        dg_call = lambda: call_deep_gemm_fp8(case)

        pos, norm, cos_tab, sin_tab, state, state_row = \
            make_main_compressor_inputs(B)
        q8 = torch.empty(B, 448, dtype=torch.uint8, device="cuda")
        s8 = torch.empty(B, 7, device="cuda")
        rope = torch.empty(B, 64, dtype=torch.bfloat16, device="cuda")
        compressor = dict(
            cmp_pos=pos, comp_norm=norm, cos_tab=cos_tab, sin_tab=sin_tab,
            comp_state=state, comp_state_row=state_row,
            comp_state_ring_entries=8, comp_q8=q8, comp_s8=s8,
            comp_rope=rope)
        fused_call = lambda: call_wuda_fp8(
            module, case, logits, 0, **compressor)
        standalone_call = lambda: module.mqa_compressor_fp8_standalone(
            pos, norm, cos_tab, sin_tab, state, state_row, 8,
            q8, s8, rope, 1e-6)

        # Warm both JIT paths before profiling; DeepGEMM compiles lazily.
        dg_call(); base_call(); fused_call(); standalone_call()
        torch.cuda.synchronize()
        dg_us = kernel_us(dg_call, "sm100_paged_mqa_logits", num_tests)
        base_us = kernel_us(
            base_call, "sm100_fp8_paged_mqa_logits_fused", num_tests)
        cmp_us = kernel_us(
            standalone_call, "standalone_compressor_kernel", num_tests)
        fused_us = kernel_us(
            fused_call, "sm100_fp8_paged_mqa_logits_fused", num_tests)
        align_ratio = base_us / dg_us
        fuse_delta = fused_us - base_us
        # Mean-profiler noise on short kernels is about 0.2us. Treat <=3% or
        # <=0.5us as aligned; fusion must stay within the same envelope.
        aligned = base_us <= dg_us * 1.03 + 0.5
        no_regression = fused_us <= base_us * 1.03 + 0.5
        passed = aligned and no_regression
        ok_all &= passed
        print(f"{B:4d} {T:7d} {stage:4d} {dg_us:9.3f} {base_us:9.3f} "
              f"{align_ratio:7.3f} {cmp_us:8.3f} {fused_us:10.3f} "
              f"{fuse_delta:10.3f} {('PASS' if passed else 'FAIL'):>8}")
        del case, logits, pos, norm, cos_tab, sin_tab, state, state_row
        del q8, s8, rope
        torch.cuda.empty_cache()
    return ok_all


BLOCK_Q = 1   # decode: 1 query token per q-block (UMMA_N=64); mirrors the kernel config
BLOCK_KV = 256


def test_main_compressor(module):
    """MAIN compressor fused into the score-attention tail (gemm_fuse_norm_b
    compressor_process_row, d=512 part): per COMPRESS row ((pos+1)%4==0)
      overlap-cat softmax aggregate -> weighted bf16 RMSNorm ->
      RoPE(last 64) -> fp8 e4m3 block-64 quant.
    State is the RTP [kv|score] ring; the kernel never writes it, so every
    entry must come back untouched.
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
    fused_c, block_table = build_paged_cache(kv_p.view(-1, HEAD_DIM // 2),
                                             kv_sf.view(-1), B, T, shuffle=True)
    ctx = torch.full((B,), T, dtype=torch.int32, device="cuda")
    stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
    logits_base = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
    module.mqa_logits_fp4_decode_out(q_p, q_sf, fused_c, weights, ctx, block_table,
                                     logits_base, 0, 0)
    torch.cuda.synchronize()

    # compressor inputs: rows 3 and 7 are compress rows ((pos+1)%4 == 0)
    pos = torch.arange(B, dtype=torch.int64, device="cuda")
    comp_norm = (torch.rand(512, device="cuda") + 0.5)
    S = 64
    ang = torch.outer(torch.arange(S, device="cuda", dtype=torch.float32),
                      1.0 / (10000.0 ** (torch.arange(32, device="cuda") / 32.0)))
    cos_tab, sin_tab = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
    comp_state = torch.randn(B, 8, 2048, device="cuda", dtype=torch.float32)
    state_rows = (torch.arange(B, device="cuda") * 8 + pos % 8).int()
    state0 = comp_state.clone()
    comp_q8 = torch.full((B, 448), 0xAB, device="cuda", dtype=torch.uint8)   # sentinel
    comp_s8 = torch.full((B, 7), -1.0, device="cuda", dtype=torch.float32)
    comp_rope = torch.zeros(B, 64, device="cuda", dtype=torch.bfloat16)

    logits = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
    module.mqa_logits_fp4_decode_out(
        q_p, q_sf, fused_c, weights, ctx, block_table, logits, 0, 0,
        cmp_pos=pos, comp_norm=comp_norm, cos_tab=cos_tab, sin_tab=sin_tab,
        comp_state=comp_state, comp_state_row=state_rows,
        comp_state_ring_entries=8, comp_q8=comp_q8, comp_s8=comp_s8,
        comp_rope=comp_rope)
    torch.cuda.synchronize()

    ok = torch.equal(logits, logits_base)
    print(f"  logits bitwise unchanged: {'PASS' if ok else 'FAIL'}")

    idx = torch.arange(512, device="cuda")
    for m in range(B):
        p = int(pos[m])
        if (p + 1) % 4 != 0:   # untouched row
            row_ok = (torch.equal(comp_state[m], state0[m])
                      and bool((comp_q8[m] == 0xAB).all()) and bool((comp_s8[m] == -1.0).all()))
            ok &= row_ok
            print(f"  row {m} (pos {p}, skip): untouched {'PASS' if row_ok else 'FAIL'}")
            continue
        # Torch reference over the eight ring rows ending at the current row.
        current = int(state_rows[m]) % 8
        perm = [(current - 7 + r) % 8 for r in range(8)]
        kv0, sc0 = state0[m, :, :1024], state0[m, :, 1024:]
        col = torch.stack([idx if r < 4 else idx + 512 for r in range(8)])       # [8,512]
        sc8 = torch.gather(sc0[perm], 1, col)
        kv8 = torch.gather(kv0[perm], 1, col)
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
        # MODEL1 scale: pow2-ceil(clamp(amax/448, 1e-4)) -- e8m0-exact
        scale_ref = torch.pow(2.0, (blk.abs().max(1).values / 448.0)
                              .clamp_min(1e-4).log2().ceil())
        # checks (reduce-order ulps -> tolerances; state must be read-only)
        state_ok = torch.equal(comp_state[m], state0[m])
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


def test_main_compressor_pool_write(module):
    """MAIN MODEL1 writes obey RTP block ids, record layout, and padding."""
    torch.manual_seed(20260809)
    dev, B = 'cuda', 3
    positions = torch.tensor([3, 4, 7], device=dev, dtype=torch.int64)
    state_blocks = torch.tensor([2, 3, 5], device=dev)
    state_rows = (state_blocks * 8 + positions % 8).int()
    state = torch.randn(6 * 8, 2048, device=dev) * 0.1
    norm = torch.rand(512, device=dev) + 0.5
    ang = torch.outer(torch.arange(64, device=dev, dtype=torch.float32),
                      1.0 / (10000.0 ** (torch.arange(32, device=dev) / 32.0)))
    cos_tab, sin_tab = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()

    cmp_epb, body_bytes, scale_bytes = 32, 576, 8
    cmp_payload = cmp_epb * (body_bytes + scale_bytes)
    cmp_stride = (cmp_payload + body_bytes - 1) // body_bytes * body_bytes
    cmp_cache = torch.full((6, cmp_stride), 0xA5, device=dev, dtype=torch.uint8)
    cmp_dst = torch.tensor([cmp_epb + 31, 2 * cmp_epb + 7,
                            4 * cmp_epb + 3], device=dev, dtype=torch.int32)

    idx_epb = 32
    idx_stride = idx_epb * (HEAD_DIM // 2 + 4) + 128
    idx_cache = torch.full((4, idx_stride), 0xA5, device=dev, dtype=torch.uint8)
    block_table = torch.tensor([[2], [0], [3]], device=dev, dtype=torch.int32)
    q = torch.zeros(B, NUM_HEADS, HEAD_DIM // 2, device=dev, dtype=torch.int8)
    q_sf = torch.full((B, NUM_HEADS), 0x7F7F7F7F, device=dev,
                      dtype=torch.int32)
    weights = torch.zeros(B, NUM_HEADS, device=dev)
    context_lens = torch.ones(B, device=dev, dtype=torch.int32)
    logits = torch.full((B, BLOCK_KV), float('-inf'), device=dev)

    module.mqa_logits_fp4_decode_out(
        q, q_sf, idx_cache, weights, context_lens, block_table, logits, 4, 0,
        cmp_pos=positions, comp_norm=norm, cos_tab=cos_tab, sin_tab=sin_tab,
        comp_state=state, comp_state_row=state_rows,
        comp_state_ring_entries=8, cmp_cache=cmp_cache, cmp_dst=cmp_dst,
        kv_entries_per_block=idx_epb, kv_block_stride_bytes=idx_stride,
        cmp_entries_per_block=cmp_epb, cmp_block_stride_bytes=cmp_stride,
        mock_attn=True)
    torch.cuda.synchronize()

    compress_rows = (0, 2)
    allowed = torch.zeros_like(cmp_cache, dtype=torch.bool)
    got_q8, got_s8, got_rope = [], [], []
    for row in compress_rows:
        page, off = divmod(int(cmp_dst[row]), cmp_epb)
        body_base = off * body_bytes
        body = cmp_cache[page, body_base:body_base + body_bytes]
        got_q8.append(body[:448])
        got_rope.append(body[448:].contiguous().view(torch.bfloat16))
        scale_base = cmp_epb * body_bytes + off * scale_bytes
        got_s8.append(cmp_cache[page, scale_base:scale_base + 7])
        allowed[page, body_base:body_base + body_bytes] = True
        # MODEL1 reserves eight scale bytes but writes seven exponent bytes.
        allowed[page, scale_base:scale_base + 7] = True
    untouched = torch.equal(cmp_cache[~allowed],
                            torch.full_like(cmp_cache[~allowed], 0xA5))

    current = state_rows.long() % 8
    block_base = state_rows.long() - current
    logical = ((current[:, None] - 7
                + torch.arange(8, device=dev)[None, :]) % 8)
    entries = state[(block_base[:, None] + logical).long()]
    kv, score = entries[..., :1024], entries[..., 1024:]
    kv_overlap = torch.cat((kv[:, :4, :512], kv[:, 4:, 512:]), dim=1)
    score_overlap = torch.cat((score[:, :4, :512], score[:, 4:, 512:]), dim=1)
    value = (torch.softmax(score_overlap, dim=1) * kv_overlap).sum(dim=1)
    value = value.bfloat16().float()
    value = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
             * norm).bfloat16().float()
    rope_pos = positions + 1 - 4
    even, odd = value[:, 448::2].clone(), value[:, 449::2].clone()
    value[:, 448::2] = (even * cos_tab[rope_pos]
                        - odd * sin_tab[rope_pos]).bfloat16().float()
    value[:, 449::2] = (even * sin_tab[rope_pos]
                        + odd * cos_tab[rope_pos]).bfloat16().float()
    blocks = value[:, :448].view(B, 7, 64)
    ref_scale = torch.pow(2.0, (blocks.abs().amax(-1) / 448.0)
                              .clamp_min(1e-4).log2().ceil())
    ref_q8 = (blocks / ref_scale.unsqueeze(-1)).to(torch.float8_e4m3fn) \
        .view(B, 448).view(torch.uint8)
    ref_s8 = (ref_scale.log2() + 127).to(torch.uint8)
    ref_rope = value[:, 448:].bfloat16()

    rows = torch.tensor(compress_rows, device=dev)
    got_q8, got_s8, got_rope = (torch.stack(got_q8), torch.stack(got_s8),
                                 torch.stack(got_rope))
    values_ok = (
        (got_q8 == ref_q8[rows]).float().mean().item() > 0.98
        and (got_s8 == ref_s8[rows]).float().mean().item() > 0.98
        and (got_rope.view(torch.uint8)
             == ref_rope[rows].contiguous().view(torch.uint8)).float().mean().item()
        > 0.98)
    ok = untouched and values_ok
    print(f"  RTP MAIN paged-pool writer: {'PASS' if ok else 'FAIL'}")
    return ok


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
    print("  decode). stg = KV pipeline depth. kernel_us = MEAN over profiler instances.")
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
    # (T = ctx/4 for the DSV4 indexer): 4K / 32K / 64K / 128K / 1M context.
    for B in (32, 64, 128, 256):
        for T in (1024, 8192, 16384, 32768, 262144):
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
            fused, block_table = build_paged_cache(kv_p, kv_sf.to(torch.int32), B, T)
            del kv_p, kv_sf

            # hoist per-call host work out of the timed region (repo *_out convention)
            stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
            logits = torch.full((B, stride), float("-inf"),
                                device="cuda", dtype=torch.float32)
            ctx = torch.full((B,), T, dtype=torch.int32, device="cuda")
            total_tiles = B * ((T + BLOCK_KV - 1) // BLOCK_KV)
            # mirror the host's auto rule for the printed stg column
            per_cta = (total_tiles + num_sms - 1) // num_sms
            auto_stg = 4 if per_cta <= 8 else (6 if per_cta <= 16 else 8)

            for stg in stage_opts:
                call = lambda s=stg: module.mqa_logits_fp4_decode_out(
                    q_p, q_sf, fused, weights, ctx, block_table, logits, 0, s)  # ctas=0 -> per SM
                kus = kernel_us(call)
                eff_stg = stg if stg else auto_stg
                if fuse_comp:
                    # MAIN compressor, REALISTIC trigger: staggered decode positions
                    # -> (pos+1)%4==0 on exactly B/4 rows, spread one-per-CTA (aligns
                    # complex_b's cmp_pos semantics; full-compress was a false worst case)
                    cpos = torch.arange(B, dtype=torch.int64, device="cuda")
                    cnorm = torch.rand(512, device="cuda") + 0.5
                    ctab = torch.rand(4, 32, device="cuda")
                    stab = torch.rand(4, 32, device="cuda")
                    cstate = torch.randn(B, 8, 2048, device="cuda")
                    crows = (torch.arange(B, device="cuda") * 8 + cpos % 8).int()
                    cq8 = torch.empty(B, 448, dtype=torch.uint8, device="cuda")
                    cs8 = torch.empty(B, 7, device="cuda")
                    crope = torch.empty(B, 64, dtype=torch.bfloat16, device="cuda")
                    acall = lambda s=stg: module.mqa_logits_fp4_decode_out(
                        q_p, q_sf, fused, weights, ctx, block_table, logits, 0, s,
                        cmp_pos=cpos, comp_norm=cnorm, cos_tab=ctab, sin_tab=stab,
                        comp_state=cstate, comp_state_row=crows,
                        comp_state_ring_entries=8,
                        comp_q8=cq8, comp_s8=cs8, comp_rope=crope)
                    aus = kernel_us(acall)
                    # "each op as its OWN kernel" reference: standalone compressor
                    # (one warp per row, full grid), SAME L2-flushed estimator.
                    cmp1 = kernel_us(lambda: module.mqa_compressor_standalone(
                        cpos, cnorm, ctab, stab, cstate, crows, 8,
                        cq8, cs8, crope, 1e-6),
                                     name_substr="standalone_compressor")
                    sep = kus + cmp1
                    # tail IN SITU solo: attention mocked out (384 threads exit at
                    # entry), same 512-thread launch shape, only the tail warpgroup
                    # works -> the tail's uncovered wall in its real environment.
                    tcall = lambda s=stg: module.mqa_logits_fp4_decode_out(
                        q_p, q_sf, fused, weights, ctx, block_table, logits, 0, s,
                        cmp_pos=cpos, comp_norm=cnorm, cos_tab=ctab, sin_tab=stab,
                        comp_state=cstate, comp_state_row=crows,
                        comp_state_ring_entries=8,
                        comp_q8=cq8, comp_s8=cs8, comp_rope=crope,
                        mock_attn=True)
                    tus = kernel_us(tcall)
                    attn_bytes = (B * NUM_HEADS * (HEAD_DIM // 2 + 4 + 4)
                                  + B * T * (HEAD_DIM // 2 + 4) + B * T * 4)
                    bw = attn_bytes / 1e3 / kus
                    print(f"{B:4d} {T:7d} {4*T:8d} {eff_stg:5d} "
                          f"{kus:9.3f} {aus:8.3f} {aus-kus:7.3f} {tus:8.3f} "
                          f"{cmp1:8.3f} {sep:8.3f} {sep/aus:5.2f} {bw:9.0f}")
                    del cpos, cnorm, ctab, stab, cstate, crows, cq8, cs8, crope
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
            del weights, q_p, q_sf, fused, block_table, logits, ctx
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
    fused, block_table = build_paged_cache(kv_p, kv_sf.to(torch.int32), B, T)
    del kv_p, kv_sf
    stride = ((T + BLOCK_KV - 1) // BLOCK_KV) * BLOCK_KV
    logits = torch.full((B, stride), float("-inf"), device="cuda", dtype=torch.float32)
    ctx = torch.full((B,), T, dtype=torch.int32, device="cuda")
    # MAIN compressor (realistic 1/4 trigger) so the phase stamps have work to show
    cpos = torch.arange(B, dtype=torch.int64, device="cuda")
    cnorm = torch.rand(512, device="cuda") + 0.5
    ctab = torch.rand(4, 32, device="cuda")
    stab = torch.rand(4, 32, device="cuda")
    cstate = torch.randn(B, 8, 2048, device="cuda")
    crows = (torch.arange(B, device="cuda") * 8 + cpos % 8).int()
    cq8 = torch.empty(B, 448, dtype=torch.uint8, device="cuda")
    cs8 = torch.empty(B, 7, device="cuda")
    crope = torch.empty(B, 64, dtype=torch.bfloat16, device="cuda")
    prof_t = torch.zeros(num_sms * 8, dtype=torch.int64, device="cuda")

    fcall = lambda p=None: module.mqa_logits_fp4_decode_out(
        q_p, q_sf, fused, weights, ctx, block_table, logits, 0, stg,
        prof=p,
        cmp_pos=cpos, comp_norm=cnorm, cos_tab=ctab, sin_tab=stab,
        comp_state=cstate, comp_state_row=crows, comp_state_ring_entries=8,
        comp_q8=cq8, comp_s8=cs8, comp_rope=crope)
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

    print(f"\nTimeline B={B} T={T} stg={stg if stg else 'auto'}: one stamped call, L2 flushed.")
    print(f"t=0 = earliest stamp (CTA entry); full width = {span/1e3:.1f} us.")
    print("attn = t0->t1 (t0 is POST-prologue: barrier init + TMEM alloc + ctx scan);")
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
    # (rms section gone; the RTP ring needs no state-shift phase).
    if int(p[:, 7].max()) > 0:
        i = int(p[:, 7].argmax())
        t2, t5, t7 = (p[i, k].item() for k in (2, 5, 7))
        print(f"compressor phases (critical CTA {i}, us): "
              f"agg {(t5-t2)/1e3:.2f} | norm+rope+quant {(t7-t5)/1e3:.2f} | "
              f"comp total {(t7-t2)/1e3:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", action="store_true",
                        help="test the separate FP8 MQA + MAIN implementation")
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
    if args.fp8:
        print("JIT compiling the separate mqa_logits_fp8.cu ...")
        module = load_cuda_module_fp8()
        ok = True
        if not args.skip_correctness:
            ok &= test_fp8_attention(module)
            ok &= test_fp8_main_compressor(module)
            print("\nALL FP8 CORRECTNESS PASSED" if ok else
                  "\nFP8 CORRECTNESS FAILED")
        if not args.skip_bench:
            ok &= benchmark_fp8(module)
            print("\nALL FP8 PERF GATES PASSED" if ok else
                  "\nFP8 PERF GATE FAILED")
        return 0 if ok else 1

    print("JIT compiling mqa_logits_fp4.cu ...")
    module = load_cuda_module()

    ok = True
    if not args.skip_correctness:
        ok &= test_runtime_page_geometry(module)
        ok &= test_decode(module)
        ok &= test_query_rms_rope(module)
        ok &= test_main_compressor(module)
        ok &= test_main_compressor_pool_write(module)
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
