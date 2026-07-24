"""Correctness + benchmark for the migrated DeepSeek-V4 top-k (kernels/topk_v2.cu,
device impl include/topk_v2.cuh; migrated from sglang csrc/deepseek_v4/topk_v2.cuh).

Runs AFTER the score-attention kernel (mqa_logits_fp4 decode) as a separate
launch: scores fp32 [B, L] (row stride % 4 == 0; rows' tails beyond seq_len are
never read), per-batch seq_lens, page-table transform on the selected indices.
DSV4 decode: topk = index_topk = 512.

Correctness golden, matching the kernel's ACTUAL contract:
  * random-float cases (fp32 duplicates ~impossible): the selected raw-index SET
    must equal torch's argsort(-scores, stable)[:k] set exactly;
  * tie-heavy cases (exact fp32 duplicates): the radix tie path keys on VALUE
    only -- elements with identical fp32 scores are slotted in atomic arrival
    order (index-asc tie-break exists only in the <=128-tie warp paths), and
    candidates cap at kMaxNumTie=2048. So the check is the tight guarantee the
    kernel does make: selected indices unique & in-range, and the selected
    score MULTISET equals the reference top-k score multiset (any such set is
    equivalent for downstream sparse attention).
Also checked: page transform math, raw_indices output, seq_len <= topk trivial
path.

Paths covered (dispatch mirrors upstream):
  L <= 8192  Register<2> | L <= 16384 Register<4> | L <= floor Streaming
  L > floor: B <= 30 fused small-batch 8-CTA cluster / else persistent pool + main

    python test/test_topk_v2.py            # correctness + bench
    python test/test_topk_v2.py --skip-correctness
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto   # DeepGEMM's bench_kineto, vendored verbatim

TOPK = 512          # DSV4 index_topk
PAGE_SIZE = 64


def load_cuda_module():
    from torch.utils.cpp_extension import load

    this_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(this_dir)

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm < 90:
        raise RuntimeError(f"cluster topk requires sm_90+, got sm_{sm}")

    cuda_flags = [
        "-O3", "-std=c++17", "-lineinfo", "--use_fast_math",
        f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
    ]
    return load(
        name="topk_v2",
        sources=[os.path.join(proj_dir, "kernels", "topk_v2.cu")],
        extra_include_paths=[os.path.join(proj_dir, "include")],
        extra_cuda_cflags=cuda_flags,
        verbose=True,
    )


def make_case(B, L, k, lens, tie_levels=0, seed=0):
    """scores [B, L] fp32 with -inf tails beyond each row's seq_len (mimics the
    mqa decode logits buffer); random page table; tie_levels>0 quantizes scores
    to that many distinct values to stress the exact tie-break machinery."""
    torch.manual_seed(seed)
    scores = torch.randn(B, L, device="cuda", dtype=torch.float32)
    if tie_levels > 0:
        scores = torch.randint(0, tie_levels, (B, L), device="cuda").float()
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    for b in range(B):
        scores[b, lens[b]:] = float("-inf")
    num_pages = (L + PAGE_SIZE - 1) // PAGE_SIZE
    # arbitrary page ids (< 2^20 so `page << 6` stays well inside int32); the
    # reference transform uses the SAME table, so no uniqueness needed
    page_table = torch.randint(0, 1 << 20, (B, max(num_pages, 1)),
                               device="cuda", dtype=torch.int32).contiguous()
    return scores, seq_lens, page_table


def ref_page_transform(page_table_row, raw):
    """page_to_indices: (page_table[i >> bits] << bits) | (i & mask); -1 stays -1."""
    bits = PAGE_SIZE.bit_length() - 1
    out = torch.where(
        raw < 0,
        torch.full_like(raw, -1),
        (page_table_row[(raw.clamp(min=0) >> bits).long()] << bits) | (raw.clamp(min=0) & (PAGE_SIZE - 1)),
    )
    return out


def check_case(module, name, B, L, k, lens, tie_levels=0, seed=0):
    scores, seq_lens, page_table = make_case(B, L, k, lens, tie_levels, seed)
    raw = torch.full((B, k), -7, device="cuda", dtype=torch.int32)
    got = module.topk_v2(scores, seq_lens, page_table, k, PAGE_SIZE, raw_indices=raw)
    torch.cuda.synchronize()

    ok = True
    detail = ""
    for b in range(B):
        n = lens[b]
        raw_b = raw[b]
        if n <= k:
            # trivial path: raw = [0..n) then -1 padding
            exp_raw = torch.arange(k, device="cuda", dtype=torch.int32)
            exp_raw[n:] = -1
            if not torch.equal(raw_b, exp_raw):
                ok, detail = False, f"b{b}: trivial raw mismatch"
                break
        else:
            ref_idx = torch.argsort(-scores[b, :n], stable=True)[:k].int()
            if tie_levels > 0:
                # exact-fp32-duplicate ties: kernel guarantees the score MULTISET,
                # not which equal-valued indices (see module docstring)
                valid = (raw_b.min().item() >= 0 and raw_b.max().item() < n
                         and raw_b.unique().numel() == k)
                sel_v = torch.sort(scores[b, raw_b.long()], descending=True)[0]
                ref_v = torch.sort(scores[b, ref_idx.long()], descending=True)[0]
                if not (valid and torch.equal(sel_v, ref_v)):
                    ok, detail = False, f"b{b}: tie multiset/validity mismatch"
                    break
            elif not torch.equal(torch.sort(raw_b)[0], torch.sort(ref_idx)[0]):
                inter = len(set(raw_b.tolist()) & set(ref_idx.tolist()))
                ok, detail = False, f"b{b}: set overlap {inter}/{k}"
                break
        # page transform consistency (over ALL slots, incl. -1 padding)
        exp_page = ref_page_transform(page_table[b], raw_b)
        if not torch.equal(got[b], exp_page.int()):
            ok, detail = False, f"b{b}: page transform mismatch"
            break
    print(f"  {name:<34} B={B:<4} L={L:<7} k={k:<5} {'PASS' if ok else 'FAIL ' + detail}")
    return ok


def test_correctness(module):
    print("\n[topk_v2] exact-set correctness vs torch stable argsort")
    ok = True
    # register<2> path (L <= 8192), mixed lens incl. len < k (trivial rows)
    ok &= check_case(module, "reg2 mixed lens", 4, 1024, TOPK, [100, 512, 777, 1024], seed=1)
    ok &= check_case(module, "reg2 8192", 8, 8192, TOPK, [8192] * 8, seed=2)
    # register<4> path (L <= 16384)
    ok &= check_case(module, "reg4 12288", 8, 12288, TOPK, [12288, 9000, 16, 12000, 512, 513, 11111, 30], seed=3)
    # streaming path (16384 < L <= floor; B > 15 keeps floor at 65536)
    ok &= check_case(module, "streaming 40000 (B=32)", 32, 40000, TOPK, [40000 - i * 700 for i in range(32)], seed=4)
    # fused small-batch cluster (B <= 15, L > 32768 -> 8-CTA DSMEM path)
    ok &= check_case(module, "small-batch cluster 40000", 4, 40000, TOPK, [40000, 39001, 35000, 33000], seed=5)
    # persistent pool + main level-3 (B > 30, L > 65536; plan routes long items)
    lens = [100000 - i * 900 for i in range(64)]
    ok &= check_case(module, "persistent pool 100000 (B=64)", 64, 100000, TOPK, lens, seed=6)
    # tie-heavy: few distinct values -> exact radix tie-break does all the work
    ok &= check_case(module, "tie-heavy (4 levels)", 4, 8192, TOPK, [8192] * 4, tie_levels=4, seed=7)
    ok &= check_case(module, "tie-heavy streaming (2 levels)", 4, 40000, TOPK, [40000] * 4, tie_levels=2, seed=8)
    # odd k values
    ok &= check_case(module, "k=13", 4, 4096, 13, [4096] * 4, seed=9)
    ok &= check_case(module, "k=2048 (max)", 4, 12288, 2048, [12288, 2048, 2049, 8000], seed=10)
    print("  ->", "ALL PASS" if ok else "SOME FAILED")
    return ok


def benchmark(module):
    """bench_kineto (8GB L2 flush per call, kernel device-time mean). plan +
    output buffers hoisted out of the timed call (repo *_out convention);
    bytes = one pass over the valid scores (streaming does 2 -> BW halves)."""
    print("\n" + "=" * 60)
    print("Benchmark: topk_v2 (after-scorer standalone launch), k=512")
    print("=" * 60)
    print(f"  {'B':<5} {'T':<8} {'path':<12} {'us':<9} {'GB/s(1pass)':<12}")
    print("  " + "-" * 50)
    for B, T in [(32, 1024), (128, 1024), (32, 8192), (128, 8192),
                 (32, 32768), (128, 32768), (8, 131072), (64, 131072)]:
        lens = [T] * B
        scores, seq_lens, page_table = make_case(B, T, TOPK, lens, seed=B + T)
        page_indices = torch.empty(B, TOPK, device="cuda", dtype=torch.int32)
        metadata = torch.zeros(B + 1, 2, device="cuda", dtype=torch.int32)
        if T > 32768:
            module.topk_v2_plan(seq_lens, metadata)   # hoisted (seq_lens static)
        call = lambda: module.topk_v2_transform(
            scores, seq_lens, page_table, page_indices, PAGE_SIZE, metadata)

        if T > 65536 or (B <= 15 and T > 32768):
            if B <= 30:
                names, path = ("topk_small_batch_kernel",), "sb-cluster"
            else:
                names, path = ("topk_persistent_cluster_kernel", "topk_main_kernel"), "pool+main"
        else:
            names, path = ("topk_main_kernel",), "main"
        ts = bench_kineto(call, names if len(names) > 1 else names[0],
                          suppress_kineto_output=True)
        us = 1e6 * (sum(ts) if isinstance(ts, tuple) else ts)
        bw = B * T * 4 / (us * 1e-6) / 1e9
        print(f"  {B:<5} {T:<8} {path:<12} {us:<9.1f} {bw:<12.0f}")
        del scores, seq_lens, page_table, page_indices, metadata
        torch.cuda.empty_cache()
    print("  (streaming/cluster paths read scores twice -> effective BW ~2x the shown 1-pass number)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 0
    major, minor = torch.cuda.get_device_capability()
    print(f"device={torch.cuda.get_device_name()} sm_{major}{minor} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")
    print("JIT compiling topk_v2.cu ...")
    module = load_cuda_module()

    ok = True
    if not args.skip_correctness:
        ok = test_correctness(module)
    benchmark(module)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
