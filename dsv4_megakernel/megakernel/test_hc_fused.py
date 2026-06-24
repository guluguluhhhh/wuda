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
D = 1792
HIDDEN_SIZE = HC * D  # 7168
N_OUT = (2 + HC) * HC  # 24
HC_EPS = 1e-6
RMS_NORM_EPS = 1e-6
SINKHORN_ITERS = 20

# BW = B×S × bytes_per_pos / latency
# weight(bf16): 24*7168*2B=336KB, input(bf16): 7168*2B=14KB, output(bf16): 1792*2B=3.5KB
# + pre/post/comb: ~100B, base/scale: ~100B
BYTES_PER_POS = N_OUT * HIDDEN_SIZE * 2 + HIDDEN_SIZE * 2 + D * 2 + \
               (HC + HC + HC * HC) * 4 + N_OUT * 4 + 3 * 4  # ~354 KB
HBM_PEAK_GBS = 8000


# ============================================================
# Reference (matching origin/model.py hc_pre)
# ============================================================
def hc_reference(hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale,
                 hc_eps=HC_EPS, rms_norm_eps=RMS_NORM_EPS, sinkhorn_iters=SINKHORN_ITERS):
    dtype = hidden_states.dtype
    hs = hidden_states.unsqueeze(0).unsqueeze(0)  # [1,1,HC,D]
    x = hs.reshape(1, 1, HC * D).float()
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

    hidden_states = torch.randn(HC, D, device=device, dtype=torch.bfloat16)
    attn_hc_fn = torch.randn(N_OUT, HC * D, device=device, dtype=torch.bfloat16) * 0.01
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

    attn_hc_fn = torch.randn(N_OUT, HC * D, device=device, dtype=torch.bfloat16) * 0.01
    attn_hc_base = torch.randn(N_OUT, device=device, dtype=torch.float32) * 0.1
    attn_hc_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)

    batch_sizes = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]
    print(f"\nPer-position data: {BYTES_PER_POS / 1024:.1f} KB")
    print(f"\n{'B×S':<8} {'Latency(us)':<13} {'Throughput':<15} {'BW (GB/s)':<12} {'HBM Util':<10}")
    print("-" * 65)

    for num_pos in batch_sizes:
        hs = torch.randn(num_pos, HC, D, device=device, dtype=torch.bfloat16)
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--skip-correctness', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available."); sys.exit(0)

    print(f"Config: HC={HC}, D={D}, hidden_size={HIDDEN_SIZE}, N_OUT={N_OUT}")
    print(f"        Sinkhorn={SINKHORN_ITERS}, block=1024, grid=2*SM")
    print(f"        Device: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"        Compute: sm_{cap[0]*10+cap[1]}\n")

    print("JIT compiling...")
    module = load_cuda_module()
    print("Done!\n")

    if not args.skip_correctness:
        if not test_correctness(module): sys.exit(1)
    if args.benchmark:
        benchmark(module)
