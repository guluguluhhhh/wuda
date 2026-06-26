"""
Test & Benchmark for HC Fused Kernel (DeepSeek-V4 attn_hc)
Matches origin/model.py: bf16 input, bf16 weight, bf16 output
"""
import os, sys, argparse
import torch
import torch.nn.functional as F

# ============================================================
# Config
# ============================================================
HC = 4
DIM = 7168           # model hidden_size
HC_DIM = HC * DIM    # 28672 (flat input size / GEMV K dim)
N_OUT = (2 + HC) * HC  # 24
HC_EPS = 1e-6
RMS_NORM_EPS = 1e-6
SINKHORN_ITERS = 20

# BW = B*S * bytes_per_pos / latency
# weight(bf16): 24*28672*2B=1344KB, input(bf16): 28672*2B=56KB, output(bf16): 7168*2B=14KB
BYTES_PER_POS = N_OUT * HC_DIM * 2 + HC_DIM * 2 + DIM * 2 + \
               (HC + HC + HC * HC) * 4 + N_OUT * 4 + 3 * 4  # ~1.38 MB
HBM_PEAK_GBS = 8000


# ============================================================
# Reference (matching origin/model.py hc_pre)
# ============================================================
def hc_reference(hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale,
                 hc_eps=HC_EPS, rms_norm_eps=RMS_NORM_EPS, sinkhorn_iters=SINKHORN_ITERS):
    dtype = hidden_states.dtype
    hs = hidden_states.unsqueeze(0).unsqueeze(0)  # [1,1,HC,D]
    x = hs.reshape(1, 1, HC * DIM).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mix = F.linear(x, attn_hc_fn.float()) * rsqrt

    pre_w, post_w, comb_w = torch.split(mix, [HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = torch.split(attn_hc_base, [HC, HC, HC * HC])

    pre = torch.sigmoid(pre_w * attn_hc_scale[0] + pre_b) + hc_eps
    post = 2.0 * torch.sigmoid(post_w * attn_hc_scale[1] + post_b)
    comb_logits = comb_w.view(1, 1, HC, HC) * attn_hc_scale[2] + comb_b.view(HC, HC)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    collapsed = (pre.unsqueeze(-1) * hs).sum(dim=2).to(dtype)
    return (collapsed.squeeze(0).squeeze(0), pre.squeeze(0).squeeze(0),
            post.squeeze(0).squeeze(0), comb.squeeze(0).squeeze(0))


# ============================================================
# JIT compile
# ============================================================
def load_cuda_module():
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_flags = ['-O3', '--use_fast_math', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo']
    try:
        cap = torch.cuda.get_device_capability()
        cuda_flags.append(f'-gencode=arch=compute_{cap[0]*10+cap[1]},code=sm_{cap[0]*10+cap[1]}')
    except:
        cuda_flags.append('-gencode=arch=compute_100,code=sm_100')
    return load(
        name='hc_fused',
        sources=[os.path.join(this_dir, 'kernels', 'hc_fused_kernel.cu')],
        extra_include_paths=[os.path.join(this_dir, 'include')],
        extra_cuda_cflags=cuda_flags, verbose=True,
    )


# ============================================================
# Correctness test
# ============================================================
def test_correctness(module):
    print("=" * 60)
    print("Correctness Test (bf16 weight, matching origin)")
    print("=" * 60)
    torch.manual_seed(42)
    device = 'cuda'

    hidden_states = torch.randn(HC, DIM, device=device, dtype=torch.bfloat16)
    attn_hc_fn = torch.randn(N_OUT, HC * DIM, device=device, dtype=torch.bfloat16) * 0.01
    attn_hc_base = torch.randn(N_OUT, device=device, dtype=torch.float32) * 0.1
    attn_hc_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    ref_collapsed, ref_pre, ref_post, ref_comb = hc_reference(
        hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale)
    fused = module.hc_fused_forward_full(
        hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale, HC_EPS, RMS_NORM_EPS)
    fused_collapsed, fused_pre, fused_post, fused_comb = fused

    print(f"\n{'Output':<12} {'Max Abs Err':<15} {'Mean Abs Err':<15} {'Max Rel Err':<15} {'Pass'}")
    print("-" * 75)

    def check(name, ref, fused, atol, rtol):
        diff = (ref.float() - fused.float()).abs()
        abs_max = diff.max().item()
        abs_mean = diff.mean().item()
        rel_max = (diff / ref.float().abs().clamp(min=1e-8)).max().item()
        passed = abs_max < atol or rel_max < rtol
        print(f"{name:<12} {abs_max:<15.6e} {abs_mean:<15.6e} {rel_max:<15.6e} {'PASS' if passed else 'FAIL'}")
        return passed

    ok = True
    ok &= check("collapsed", ref_collapsed, fused_collapsed, 1e-2, 1e-2)
    ok &= check("pre", ref_pre, fused_pre, 1e-5, 1e-5)
    ok &= check("post", ref_post, fused_post, 1e-5, 1e-5)
    ok &= check("comb", ref_comb, fused_comb, 1e-5, 1e-5)
    print("-" * 60)
    print("ALL TESTS PASSED!" if ok else "SOME TESTS FAILED!")
    return ok


# ============================================================
# Bandwidth sweep benchmark
# ============================================================
def benchmark(module, warmup=100, iters=500):
    print("\n" + "=" * 70)
    print(f"Bandwidth Sweep | {torch.cuda.get_device_name()} | HBM peak: {HBM_PEAK_GBS} GB/s")
    print("=" * 70)
    device = 'cuda'
    torch.manual_seed(42)

    attn_hc_fn = torch.randn(N_OUT, HC * DIM, device=device, dtype=torch.bfloat16) * 0.01
    attn_hc_base = torch.randn(N_OUT, device=device, dtype=torch.float32) * 0.1
    attn_hc_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    batch_sizes = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"\nPer-position data: {BYTES_PER_POS / 1024:.1f} KB")
    print(f"\n{'B×S':<8} {'Latency(us)':<13} {'Throughput':<15} {'BW (GB/s)':<12} {'HBM Util':<10}")
    print("-" * 65)

    for num_pos in batch_sizes:
        hs = torch.randn(num_pos, HC, DIM, device=device, dtype=torch.bfloat16)
        for _ in range(warmup):
            module.hc_fused_forward_full(hs, attn_hc_fn, attn_hc_base, attn_hc_scale, HC_EPS, RMS_NORM_EPS)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            module.hc_fused_forward_full(hs, attn_hc_fn, attn_hc_base, attn_hc_scale, HC_EPS, RMS_NORM_EPS)
        end.record()
        torch.cuda.synchronize()
        total_us = start.elapsed_time(end) / iters * 1000

        throughput = num_pos / (total_us * 1e-6)
        bw = (num_pos * BYTES_PER_POS) / (total_us * 1e-6) / 1e9
        util = bw / HBM_PEAK_GBS * 100
        print(f"{num_pos:<8} {total_us:<13.1f} {throughput/1e6:.2f} M pos/s   {bw:<12.1f} {util:<10.1f}%")

    print("-" * 65)


# ============================================================
# Intra-Kernel Profiling + Chrome Trace export
# ============================================================
def profile_kernel(module, num_pos=64, gpu_clock_mhz=2100):
    """Run profiled kernel and export Chrome Trace JSON."""
    import json
    device = 'cuda'
    torch.manual_seed(42)

    hs = torch.randn(num_pos, HC, DIM, device=device, dtype=torch.bfloat16)
    w = torch.randn(N_OUT, HC * DIM, device=device, dtype=torch.bfloat16) * 0.01
    base = torch.randn(N_OUT, device=device, dtype=torch.float32) * 0.1
    scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        module.hc_fused_forward_profiled(hs, w, base, scale, HC_EPS, RMS_NORM_EPS)
    torch.cuda.synchronize()

    # Profile run
    timing = module.hc_fused_forward_profiled(hs, w, base, scale, HC_EPS, RMS_NORM_EPS)
    torch.cuda.synchronize()

    # timing: [grid_size, 10] int64 = 5 phases x (start, end)
    timing_cpu = timing.cpu().numpy()
    phase_names = ["RMSNorm", "GEMV", "Activation", "Sinkhorn", "Collapse"]

    print("\n" + "=" * 60)
    print(f"Intra-Kernel Profile (num_pos={num_pos}, GPU clock={gpu_clock_mhz} MHz)")
    print("=" * 60)

    # Print per-block timing for first few blocks
    print(f"\n{'Block':<8}", end="")
    for name in phase_names:
        print(f"{name:<12}", end="")
    print("Total")
    print("-" * 75)

    # Each block's clock is independent (per-SM), so use per-block relative time
    # Place blocks sequentially with a small gap for visual clarity
    BLOCK_GAP_US = 2.0  # gap between blocks in trace

    chrome_events = []
    for bid in range(min(timing_cpu.shape[0], 8)):  # first 8 blocks
        row = timing_cpu[bid]
        if row[0] == 0 and row[1] == 0:
            continue  # unused block
        block_base = row[0]
        block_offset = 0  # all blocks aligned to same start

        print(f"{bid:<8}", end="")
        total_cycles = 0
        for p in range(5):
            start_clk = row[p*2]
            end_clk = row[p*2 + 1]
            cycles = end_clk - start_clk
            us = cycles / gpu_clock_mhz
            total_cycles += cycles
            print(f"{us:>8.1f} us  ", end="")

            # Chrome trace event (per-block relative + visual offset)
            chrome_events.append({
                "name": phase_names[p],
                "cat": "kernel",
                "ph": "X",
                "ts": float(start_clk - block_base) / gpu_clock_mhz + block_offset,
                "dur": float(cycles) / gpu_clock_mhz,
                "pid": 0,
                "tid": bid,
                "args": {"block": bid, "phase": p, "cycles": int(cycles)}
            })
        print(f"{total_cycles / gpu_clock_mhz:>8.1f} us")

    # Export Chrome Trace
    trace_file = "hc_kernel_trace.json"
    with open(trace_file, 'w') as f:
        json.dump({"traceEvents": chrome_events}, f)
    print(f"\nChrome Trace exported: {trace_file}")
    print(f"Open with: https://ui.perfetto.dev or chrome://tracing")


# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--skip-correctness', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Run intra-kernel profiling')
    parser.add_argument('--profile-positions', type=int, default=64)
    parser.add_argument('--gpu-clock-mhz', type=int, default=2100, help='GPU clock freq in MHz')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available."); sys.exit(0)

    print(f"Config: HC={HC}, DIM={DIM}, HC_DIM={HC_DIM}, N_OUT={N_OUT}")
    print(f"        Sinkhorn={SINKHORN_ITERS}, block=1024, cluster=2 blocks, grid=2*SM clusters")
    print(f"        Launch: cudaLaunchKernelEx + cluster_dim(2,1,1)")
    print(f"        Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"        Compute: sm_{cap[0]*10+cap[1]}\n")

    print("JIT compiling...")
    module = load_cuda_module()
    print("Done!\n")

    if not args.skip_correctness and not args.profile:
        if not test_correctness(module): sys.exit(1)
    if args.benchmark:
        benchmark(module)
    if args.profile:
        profile_kernel(module, args.profile_positions, args.gpu_clock_mhz)
