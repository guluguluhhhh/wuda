"""Correctness and performance tests for the independent HCA swap-AB front GEMM."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import bench_kineto
# Quant helpers / references are shared with the main front test (same K=7168,
# same 1x128 / 128x128 UE8M0 scheme); only the HCA output shape differs.
from test_front_mixed_gemm import (
    K, N_FP8, calc_diff, dequant_fp8, hc_ref, make_hc,
    quant_act_fp8, quant_weight_fp8,
)

N = 4672  # CSA shape: fp8 [0,2048) wq_a|wkv + bf16 [2048,4672)
TASKS_PER_BATCH = 29   # 8 fp8 (N256) + 21 bf16 (N128, tail tile 64 rows)


def batch_n_for(phys_m):
    """Host ladder: one BN16 tile at M<=16, otherwise exactly TWO tiles."""
    return (16 if phys_m <= 32 else
            32 if phys_m <= 64 else
            64 if phys_m <= 128 else 128)


def make_activations(m, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, K, device='cuda', dtype=torch.bfloat16) * 0.1
    x_fp8, x_sf = quant_act_fp8(x)
    return x, x_fp8, x_sf


def make_weights(seed=0):
    torch.manual_seed(seed)
    w_bf16 = torch.randn(N - N_FP8, K, device='cuda',
                         dtype=torch.bfloat16) * 0.01
    w_full = torch.randn(N_FP8, K, device='cuda',
                         dtype=torch.bfloat16) * 0.01
    w_fp8, w_sf = quant_weight_fp8(w_full)
    return w_bf16, w_fp8, w_sf


BENCH_B = (
    1, 2, 4, 8, 12, 15, 16, 17, 20, 24, 31, 32, 33, 40, 48, 56,
    63, 64, 65, 80, 96, 112, 127, 128, 129, 144, 160, 176, 192,
    208, 224, 240, 256,
)
TIMELINE_M = (16, 17, 64, 65, 128, 129, 256)
TAIL_B = (1, 16, 17, 64, 65, 128, 129, 256)
TAIL_BENCH = (1, 4, 8, 16, 17, 24, 32, 48, 64, 65, 96, 128, 129, 160,
              192, 224, 256)
TIMING_FIELDS = 13


def load_module(name, source):
    from torch.utils.cpp_extension import load
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_dir)
    cutlass = os.path.join(root, '..', 'cutlass', 'include')
    cutlass_tools = os.path.join(root, '..', 'cutlass', 'tools', 'util', 'include')
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    flags = [
        '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
        '-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1',
        '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
        f'-gencode=arch=compute_{sm}a,code=sm_{sm}a',
    ]
    return load(
        name=name,
        sources=[os.path.join(root, 'kernels', source)],
        extra_include_paths=[os.path.join(root, 'include'), cutlass, cutlass_tools],
        extra_cuda_cflags=flags,
        extra_ldflags=['-lcuda'],
        verbose=True,
    )


def run_swap(module, x, x8, xsf, weights, out=None, batch_n=0,
             task_times=None, enable_tail=False, hc=None):
    w16, w8, wsf = weights
    hc = hc or {}
    return module.front_mixed_gemm_hca_swapab(
        x, x8, xsf, w16, w8, wsf, out=out,
        batch_n_override=batch_n, task_times=task_times,
        hc_mix=hc.get('hc_mix'), hc_base=hc.get('hc_base'),
        hc_scale=hc.get('hc_scale'), hc_post=hc.get('hc_post'),
        hc_comb=hc.get('hc_comb'), enable_tail=enable_tail)


def correctness(module, weights, smoke=False):
    max_b = 64 if smoke else 256
    x, x8, xsf = make_activations(max_b, seed=4100)
    w16, w8, wsf = weights
    ref8 = dequant_fp8(x8, xsf, True) @ dequant_fp8(w8, wsf, False).t()
    ref16 = torch.mm(x, w16.t()).float()
    batches = (1, 15, 16, 17, 31, 32, 33, 48, 63, 64) if smoke else range(1, 257)
    worst8 = (0.0, 0)
    worst16 = (0.0, 0)
    for b in batches:
        out = torch.full(
            (b, N), float('nan'), device='cuda', dtype=torch.bfloat16)
        run_swap(module, x[:b], x8[:b], xsf[:b], weights, out=out)
        if not torch.isfinite(out).all():
            bad = torch.nonzero(~torch.isfinite(out), as_tuple=False)[0].tolist()
            raise AssertionError(f'B={b}: output not fully written at {bad}')
        d8 = calc_diff(out[:, :N_FP8].float(), ref8[:b])
        d16 = calc_diff(out[:, N_FP8:].float(), ref16[:b])
        worst8 = max(worst8, (d8, b))
        worst16 = max(worst16, (d16, b))
        if d8 >= 1e-5 or d16 >= 1e-5:
            raise AssertionError(
                f'B={b}: fp8_diff={d8:.3e}, bf16_diff={d16:.3e}')
    print(f'PASS correctness B=1..{max_b if not smoke else "smoke"}; '
          f'worst fp8={worst8[0]:.3e}@B{worst8[1]}, '
          f'bf16={worst16[0]:.3e}@B{worst16[1]}')

    tested_tail_b = tuple(b for b in TAIL_B if b <= max_b)
    for b in tested_tail_b:
        hc = make_hc(b, seed=4200 + b)
        hc['hc_post'].fill_(123.0)
        hc['hc_comb'].fill_(123.0)
        run_swap(module, x[:b], x8[:b], xsf[:b], weights,
                 enable_tail=False, hc=hc)
        torch.cuda.synchronize()
        if not (torch.all(hc['hc_post'] == 123.0) and
                torch.all(hc['hc_comb'] == 123.0)):
            raise AssertionError(f'B={b}: disabled tail modified outputs')
        run_swap(module, x[:b], x8[:b], xsf[:b], weights,
                 enable_tail=True, hc=hc)
        post_ref, comb_ref = hc_ref(
            hc['hc_mix'], hc['hc_base'], hc['hc_scale'])
        if not (torch.allclose(hc['hc_post'], post_ref, rtol=1e-4, atol=1e-6)
                and torch.allclose(
                    hc['hc_comb'], comb_ref, rtol=1e-4, atol=1e-6)):
            raise AssertionError(f'B={b}: enabled sinkhorn tail mismatch')
    print('PASS sinkhorn tail switch at B=' + ','.join(map(str, tested_tail_b)))


def benchmark_tail(swap, weights):
    print('\nSinkhorn tail cost (cold L2 kernel us; M<16 padded to physical M=16)')
    print(f"  {'M':>4} {'physical':>8} {'GEMM':>9} {'GEMM+tail':>11} "
          f"{'delta':>8} {'tail/base':>10}")
    print('  ' + '-' * 56)
    for b in TAIL_BENCH:
        physical_b = max(16, b)
        x, x8, xsf = make_activations(physical_b, seed=8000 + b)
        out = torch.empty(physical_b, N, device='cuda', dtype=torch.bfloat16)
        hc = make_hc(b, seed=8100 + b)
        gemm_us = 1e6 * bench_kineto(
            lambda: run_swap(swap, x, x8, xsf, weights, out=out),
            'swapab_kernel<', num_tests=20, suppress_kineto_output=True)
        tail_us = 1e6 * bench_kineto(
            lambda: run_swap(swap, x, x8, xsf, weights, out=out,
                             enable_tail=True, hc=hc),
            'swapab_kernel<', num_tests=20, suppress_kineto_output=True)
        print(f'{b:6d} {physical_b:8d} {gemm_us:9.2f} {tail_us:11.2f} '
              f'{tail_us - gemm_us:8.2f} {tail_us / gemm_us:10.3f}')


def benchmark(swap, weights):
    print('\nPerformance (cold L2 kernel us; M<16 caller-padded to physical M=16)')
    print(f'cuBLAS BF16/FP8 are full same-shape [M,{K}] x [{K},{N}] GEMMs.')
    print(f"  {'M':>4} {'phys':>5} {'tile':>4} {'swapAB':>8} "
          f"{'cuBLAS16':>9} {'cuBLAS8':>9} "
          f"{'cb16/swap':>9} {'cb8/swap':>8}")
    print('  ' + '-' * 72)

    w16, w8, wsf = weights
    w_all_bf16 = torch.cat(
        [dequant_fp8(w8, wsf, False).bfloat16(), w16]).contiguous()
    w_all_fp8 = w_all_bf16.to(torch.float8_e4m3fn).contiguous()
    one = torch.ones((), device='cuda', dtype=torch.float32)

    for b in BENCH_B:
        physical_b = max(16, b)
        x, x8, xsf = make_activations(physical_b, seed=5000 + b)
        out_swap = torch.empty(physical_b, N, device='cuda', dtype=torch.bfloat16)
        out_cb16 = torch.empty_like(out_swap)
        tile_n8 = batch_n_for(physical_b)
        swap_us = 1e6 * bench_kineto(
            lambda: run_swap(swap, x, x8, xsf, weights, out_swap),
            'swapab_kernel<',
            num_tests=20, suppress_kineto_output=True)
        cb16_us = 1e6 * sum(bench_kineto(
            lambda: torch.mm(x, w_all_bf16.t(), out=out_cb16),
            ('nvjet', 'reduce'), num_tests=20,
            suppress_kineto_output=True))
        try:
            cb8_us = 1e6 * sum(bench_kineto(
                lambda: torch._scaled_mm(
                    x8, w_all_fp8.t(), scale_a=one, scale_b=one,
                    out_dtype=torch.bfloat16),
                ('nvjet', 'reduce'), num_tests=20,
                suppress_kineto_output=True))
            cb8_text = f'{cb8_us:9.2f}'
            cb8_ratio = f'{cb8_us / swap_us:8.3f}'
        except Exception as exc:
            cb8_text = f"{'n/a':>9}"
            cb8_ratio = f"{'n/a':>8}"
            if b == BENCH_B[0]:
                print(f'  cuBLAS FP8 unavailable: {exc}')
        print(f'{b:6d} {physical_b:5d} {tile_n8:4d} {swap_us:8.2f} '
              f'{cb16_us:9.2f} {cb8_text} '
              f'{cb16_us / swap_us:9.3f} '
              f'{cb8_ratio}')


def timeline(module, weights):
    print('\nFine-grained internal timeline from %globaltimer (microseconds)')
    print('TMA producer and MMA consumer windows overlap; phase columns are not additive.')
    end_rows = []
    setup_rows = []
    pipeline_rows = []

    for m in TIMELINE_M:
        physical_m = max(16, m)
        batch_n = batch_n_for(physical_m)
        num_tasks = ((physical_m + batch_n - 1) // batch_n) * TASKS_PER_BATCH
        x, x8, xsf = make_activations(physical_m, seed=7000 + m)
        out = torch.empty(physical_m, N, device='cuda', dtype=torch.bfloat16)
        times = torch.empty(
            num_tasks, TIMING_FIELDS, device='cuda', dtype=torch.int64)

        for _ in range(3):
            run_swap(module, x, x8, xsf, weights, out=out,
                     task_times=times)
        torch.cuda.synchronize()

        samples = []
        for _ in range(15):
            times.zero_()
            run_swap(module, x, x8, xsf, weights, out=out,
                     task_times=times)
            torch.cuda.synchronize()
            host = times.cpu()
            required = host[:, (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
            fp8_rows = torch.arange(num_tasks).remainder(TASKS_PER_BATCH) < 8
            if (required <= 0).any() or (host[fp8_rows, 3] <= 0).any():
                raise AssertionError(f'M={m}: invalid task timestamps')
            task_ids = torch.arange(num_tasks)
            fp8 = task_ids.remainder(TASKS_PER_BATCH) < 8
            bf16 = ~fp8
            origin = host[:, 0].min()
            task_end = torch.maximum(host[:, 11], host[:, 12])
            fp8_end = (task_end[fp8].max() - origin).item() / 1000.0
            bf16_end = (task_end[bf16].max() - origin).item() / 1000.0
            span = (task_end.max() - origin).item() / 1000.0

            run_parts = []
            for mask in (fp8, bf16):
                task = host[mask].double()
                setup = torch.stack((
                    task[:, 1] - task[:, 0],
                    task[:, 2] - task[:, 0],
                    task[:, 3] - task[:, 0],
                    task[:, 4] - task[:, 0],
                ), dim=1).median(dim=0).values / 1000.0
                pipeline = torch.stack((
                    task[:, 5] - task[:, 4],
                    task[:, 6] - task[:, 5],
                    task[:, 7] - task[:, 4],
                    task[:, 9] - task[:, 5],
                    task[:, 8] - task[:, 7],
                    task[:, 10] - task[:, 9],
                    task[:, 11] - task[:, 10],
                    (task[:, 12] - task[:, 11]).clamp_min(0),
                    torch.maximum(task[:, 11], task[:, 12]) - task[:, 0],
                ), dim=1).median(dim=0).values / 1000.0
                run_parts.append((setup, pipeline))
            samples.append((fp8_end, bf16_end, span, run_parts))

        fp8_end = torch.tensor([s[0] for s in samples]).median().item()
        bf16_end = torch.tensor([s[1] for s in samples]).median().item()
        span = torch.tensor([s[2] for s in samples]).median().item()
        end_rows.append((m, physical_m, batch_n, num_tasks,
                         fp8_end, bf16_end, bf16_end - fp8_end, span))
        for kind, part_idx in (('FP8', 0), ('BF16', 1)):
            setup = torch.stack(
                [s[3][part_idx][0] for s in samples]).median(dim=0).values
            pipeline = torch.stack(
                [s[3][part_idx][1] for s in samples]).median(dim=0).values
            setup_rows.append((m, kind, *setup.tolist()))
            pipeline_rows.append((m, kind, *pipeline.tolist()))

    print('\nLast task completion, relative to earliest cluster entry')
    print(f"  {'M':>4} {'phys':>5} {'BN':>3} {'tasks':>5} "
          f"{'FP8 end':>8} {'BF16 end':>9} {'BF-FP8':>8} {'span':>7}")
    print('  ' + '-' * 58)
    for row in end_rows:
        print(f'{row[0]:6d} {row[1]:5d} {row[2]:3d} {row[3]:5d} '
              f'{row[4]:8.2f} {row[5]:9.2f} {row[6]:8.2f} {row[7]:7.2f}')

    print('\nSetup events, elapsed from task entry (events overlap)')
    print(f"  {'M':>4} {'kind':>5} {'barrier':>8} {'alloc':>8} "
          f"{'scale':>8} {'join':>8}")
    print('  ' + '-' * 47)
    for m, kind, barrier, alloc, scale, join in setup_rows:
        scale_text = f'{scale:8.2f}' if kind == 'FP8' else f"{'n/a':>8}"
        print(f'{m:6d} {kind:>5} {barrier:8.2f} {alloc:8.2f} '
              f'{scale_text} {join:8.2f}')

    print('\nPipeline phases per median task')
    print(f"  {'M':>4} {'kind':>5} {'1stTMA':>7} {'1stMMA':>7} "
          f"{'TMAprod':>7} {'MMAwin':>7} {'tailTMA':>7} {'drain':>7} "
          f"{'epi':>7} {'dealloc':>7} {'total':>7}")
    print('  ' + '-' * 88)
    for row in pipeline_rows:
        print(f'{row[0]:6d} {row[1]:>5} ' +
              ' '.join(f'{value:7.2f}' for value in row[2:]))
    print('TIMELINE PASS')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    print(f'Device: {torch.cuda.get_device_name()} capability='
          f'{torch.cuda.get_device_capability()} torch={torch.__version__}')
    swap_module = load_module(
        'front_mixed_gemm_hca_swapab', 'front_mixed_gemm_hca_swapab.cu')
    shared_weights = make_weights(seed=4000)
    if '--timeline' in sys.argv:
        timeline(swap_module, shared_weights)
        raise SystemExit(0)
    smoke_mode = '--smoke' in sys.argv
    correctness(swap_module, shared_weights, smoke=smoke_mode)
    if '--tail' in sys.argv:
        benchmark_tail(swap_module, shared_weights)
        print('\nALL PASS')
        raise SystemExit(0)
    if not smoke_mode:
        benchmark(swap_module, shared_weights)
        print('\nALL PASS')
