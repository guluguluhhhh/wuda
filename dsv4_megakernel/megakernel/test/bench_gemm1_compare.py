"""
FAIR gemm1 comparison: delivery opA (bf16 split-K + rms kernel)  vs
front_mixed (mixed-precision, no split-K), SAME process / GPU state /
bench_kineto discipline / SOURCE WEIGHTS.

Deliverable-state ledger (the two sides owe OPPOSITE debts -- both listed,
neither silently ignored):
  opA+rms delivers: y1 (1536) RMSNormed bf16; y2..y5 (3136) still fp32
                    split-K partials in ws  -> OWES the tail reduce (paid
                    later inside opB's CUDA-core tail).
  mixed   delivers: ALL 4672 cols as final bf16 (fp8 segment from quantized
                    weights = checkpoint semantics) -> OWES y1's RMSNorm
                    (plan: folded into opB's fused-quant prologue for free).
Weight bytes: opA streams 67.0MB bf16; mixed streams 52.3MB (fp8 2048 cols +
bf16 2624 cols). That gap is an ARCHITECTURE advantage, not a measurement
artifact -- the real checkpoint stores wq_a/wkv in fp8.

cuBLAS baseline: identical for both sides -- torch.mm with out=, kineto
('nvjet','reduce') pair summed (bench.py's single-'nvjet' cbA under-counts
cuBLAS split-K; fixed here).

Run (needs delivery/complex_op.so built + CUTLASS for the JIT build):
    python3 bench_gemm1_compare.py
"""
import json
import os, sys, tempfile, torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# delivery moved to <repo>/dsv4_megakernel/delivery (complex_op.so lives there)
DELIVERY_DIR = os.path.join(os.path.dirname(os.path.dirname(THIS_DIR)), "delivery")
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, DELIVERY_DIR)

from bench_utils import bench_kineto
from test_front_mixed_gemm import (
    load_module as load_front_mixed, quant_act_fp8, quant_weight_fp8, K, N, N_FP8)

N1, K1 = 4672, 7168
EPS = 1e-6
M_LIST = (1, 2, 4, 8, 16, 32, 48, 64, 96, 112, 128)


def load_complex_op():
    try:
        import complex_op
        return complex_op
    except ImportError as e:
        print(f"complex_op.so not found in {DELIVERY_DIR} ({e}); "
              f"build per delivery/README.md first.")
        sys.exit(1)


def bench_delivery_a(fn):
    """opA-link timing with PDL-aware WALL span.

    gemm_rmsnorm_kernel is a PDL consumer: it spins in
    cudaGridDependencySynchronize() until opA completes, and that wait is
    INSIDE its kineto device time. Summing opA+rms double-counts the overlap
    (rms ~19-20us at small M is mostly concurrent wait). The honest cost is
    the wall span fusenorm.start -> rms.end, parsed from the chrome trace.
    Returns (opA_us, rms_device_us, wall_us).
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        a_s, r_s = bench_kineto(fn, ('fusenorm', 'gemm_rmsnorm_kernel'),
                                trace_path=path, suppress_kineto_output=True)
        with open(path) as f:
            evs = [e for e in json.load(f).get("traceEvents", [])
                   if e.get("ph") == "X"]
    finally:
        os.unlink(path)
    fus = sorted((e for e in evs if "fusenorm" in e.get("name", "")),
                 key=lambda e: e["ts"])
    rms = sorted((e for e in evs if "gemm_rmsnorm_kernel" in e.get("name", "")),
                 key=lambda e: e["ts"])
    n = min(len(fus), len(rms))
    if n == 0:
        raise RuntimeError("trace missing fusenorm/gemm_rmsnorm events")
    walls = [(rms[i]["ts"] + rms[i]["dur"]) - fus[i]["ts"] for i in range(n)]
    return 1e6 * a_s, 1e6 * r_s, sum(walls) / len(walls)   # trace ts/dur are us


def make_delivery_buffers(M, x, w1):
    """comp-OFF buffer set per complex_op.cuh API (mirrors delivery/bench.py)."""
    dev = "cuda"
    t = {"x": x, "w1": w1}
    t["w_cat"] = torch.randn(65536 + 8192, 1536, device=dev, dtype=torch.bfloat16) * 0.05
    t["rms_w1"] = torch.ones(N1, device=dev, dtype=torch.float32)
    t["dA_out"] = torch.empty(M, N1, device=dev, dtype=torch.bfloat16)
    t["dB_out"] = torch.empty(M, 65536, device=dev, dtype=torch.float32)
    t["dB_out2"] = torch.empty(M, 8192, device=dev, dtype=torch.float32)
    t["ssq"] = torch.zeros(M, 128, device=dev, dtype=torch.float32)
    t["rms_w2"] = torch.ones(512, device=dev, dtype=torch.float32)
    t["rope_cs"] = torch.randn(M, 64, device=dev, dtype=torch.float32)
    return t


def run_delivery(op, t, M):
    p = lambda k: t[k].data_ptr() if k in t else 0
    return op.complex_run(
        p("x"), p("w1"), p("w_cat"), M, N1, K1, EPS,
        p("rms_w1"), p("dA_out"), p("dB_out"), p("dB_out2"),
        p("ssq"), p("rms_w2"), p("rope_cs"),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # comp ptrs OFF
        0)                                                  # mode 0


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}  torch={torch.__version__}")
    complex_op = load_complex_op()
    mixed_mod = load_front_mixed()

    torch.manual_seed(20260726)
    # ONE weight source for both sides. Column order does not affect timing
    # (random data); mixed quantizes the first 2048 rows to checkpoint-style
    # fp8 (128x128 UE8M0) and keeps the rest bf16.
    w1 = torch.randn(N1, K1, device="cuda", dtype=torch.bfloat16) * 0.05
    w_fp8, w_sf = quant_weight_fp8(w1[:N_FP8])
    w_bf16 = w1[N_FP8:].contiguous()

    print("\nFAIR gemm1: delivery opA(bf16 splitK)+rms  vs  front_mixed (no splitK)")
    print("  A_wall = WALL span fusenorm.start->rms.end from chrome trace (the rms")
    print("  kernel's device time contains its PDL wait for opA -- overlap would be")
    print("  double-counted by opA+rms; rms* column shown for reference only).")
    print("  Debts: A still owes y2..y5 ws-reduce (paid in opB tail); mixed still")
    print("  owes y1 RMSNorm (folds into opB quant). mixed uses the 16-row pad")
    print("  contract for M<16 (production form).")
    print("  weight stream: A 67.0MB bf16  |  mixed 52.3MB (checkpoint fp8+bf16)")
    print(f"  {'M':<5} {'opA':<7} {'rms*':<7} {'A_wall':<8} {'mixed':<7} "
          f"{'A/mix':<7} {'cbA':<7} {'A%cbA':<7} {'mix%cbA':<8}")
    print("  " + "-" * 70)

    for M in M_LIST:
        x = torch.randn(M, K1, device="cuda", dtype=torch.bfloat16) * 0.1
        td = make_delivery_buffers(M, x, w1)

        a_us, r_us, a_wall = bench_delivery_a(lambda: run_delivery(complex_op, td, M))

        # mixed: 16-row pad contract below M=16 (TMA tiny-desc-rows pathology)
        Mp = max(M, 16)
        if Mp != M:
            xm = torch.zeros(Mp, K1, device="cuda", dtype=torch.bfloat16)
            xm[:M] = x
        else:
            xm = x
        x_fp8, x_sf = quant_act_fp8(xm)
        out_mixed = torch.empty(Mp, N, device="cuda", dtype=torch.bfloat16)
        mixed_us = 1e6 * bench_kineto(
            lambda: mixed_mod.front_mixed_gemm(xm, x_fp8, x_sf, w_bf16, w_fp8, w_sf,
                                               out=out_mixed),
            'front_mixed_kernel', suppress_kineto_output=True)

        cb_out = torch.empty(M, N1, device="cuda", dtype=torch.bfloat16)
        cba = 1e6 * sum(bench_kineto(
            lambda: torch.mm(x, w1.t(), out=cb_out),
            ('nvjet', 'reduce'), suppress_kineto_output=True))

        if a_us <= 0 or mixed_us <= 0 or cba <= 0:
            raise RuntimeError(f"M={M}: profiler missed kernels "
                               f"(opA={a_us}, mixed={mixed_us}, cbA={cba})")
        print(f"  {M:<5} {a_us:<7.1f} {r_us:<7.1f} {a_wall:<8.1f} {mixed_us:<7.1f} "
              f"{a_wall / mixed_us:<7.2f} {cba:<7.1f} "
              f"{100 * a_wall / cba:<7.1f} {100 * mixed_us / cba:<8.1f}")


if __name__ == "__main__":
    main()
