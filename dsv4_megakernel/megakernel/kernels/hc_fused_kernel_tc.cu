// DeepSeek-V4 MHC fused forward for NVIDIA Blackwell/B300.
//
// PRODUCTION FORM (hybrid): the [M,28672]x[24,28672]^T tf32 GEMM comes from
// deep_gemm.tf32_hc_prenorm_gemm (split-K mix partials + sum(x^2) partials);
// this TU owns the fused epilogue (split-K reduce + RMSNorm + gates +
// Sinkhorn + collapse [+ full attn_norm]) and the PyTorch binding.
// The old in-house tcgen05 split-K GEMM path was DELETED (deep_gemm measured
// 2-3us faster on B300; our epilogue beats vLLM's TileLang fuse by 3-4us).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <vector>

#include "hc_fused_kernel_tc.cuh"

namespace hc_tc {

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// ============================================================
// Epilogue: split-K reduce + RMSNorm + activation + Sinkhorn + collapse.
//   grid = M (one block per position), block = EPILOGUE_THREADS.
// ============================================================
// kWithPostComb=false (A/B lite): the split-K reduce is UNTOUCHED (full 24
// columns -- DeepGEMM-style gather, already optimal); only the post/comb
// activation + Sinkhorn are dropped, and ALL 8 warps run the collapse.
// kFusedNorm (lite only): the collapse additionally applies the FULL attn_norm
// in-block -- this block owns the whole 7168-dim row, so ssq/r/gamma all live
// here (bit-exact vs a separate norm kernel reading the bf16 collapsed:
// x_b = bf16(acc) is kept in registers, ssq sums x_b^2, and the output is
// bf16((x_b * r) * w) with the reference multiply order).
template <bool kWithPostComb, bool kFusedNorm = false>
__global__ void __launch_bounds__(EPILOGUE_THREADS, 2)
hc_reduce_and_fuse_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ workspace,
    const float* __restrict__ sqr_sum,   // [num_splits, M] input-RMSNorm Σx² partials
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float hc_eps,
    float rms_eps,
    int num_positions,
    int num_splits,
    __nv_bfloat16* __restrict__ collapsed_out,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    const __nv_bfloat16* __restrict__ attn_norm_w,   // [7168] gamma, BF16 --
    // the CHECKPOINT dtype (model.py L188: "stored in bf16"); the official
    // chain computes in fp32 with the bf16 gamma upcast LOSSLESSLY, so the
    // in-kernel widen below is bit-identical to the reference while halving
    // the gamma traffic (14KB/CTA vs fp32's 28KB).
    float attn_norm_eps,
    int64_t* __restrict__ prof) {
    static_assert(!(kWithPostComb && kFusedNorm),
                  "fused attn_norm needs ALL warps in the collapse (lite only)");
    const int pos = static_cast<int>(blockIdx.x);
    if (pos >= num_positions) return;
    // clock64 phase stamps on block 0 only (see PROF_SLOTS layout in the header).
    const bool prof0 = (prof != nullptr && pos == 0);
    if (prof0 && threadIdx.x == 0) prof[2] = ptx::rdclock();

    // Hidden is not staged in smem: RMSNorm Σx² comes from the GEMM's sqr_sum partials,
    // and collapse reads hidden straight from global -> tiny smem, high occupancy.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* scratch = reinterpret_cast<float*>(smem_raw);
    float* rms_smem = scratch;               // 1
    float* sqsum_smem = rms_smem + 1;        // 1 (unused now; kept for layout offset)
    float* mix_smem = sqsum_smem + 1;        // N_OUT
    float* pre_smem = mix_smem + N_OUT;      // HC
    float* post_smem = pre_smem + HC;        // HC
    float* comb_smem = post_smem + HC;       // HC*HC

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    // PDL: when launched with PROGRAMMATIC_STREAM_SERIALIZATION (hybrid
    // path: deep_gemm producer emits launch_dependents), this blocks until
    // the producer GEMM fully completed (its workspace/sqr_sum writes are
    // visible). WITHOUT the attribute it is a no-op. Everything above --
    // launch, reg/smem setup -- overlaps the producer's tail.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    // Split-K reduce (NO smem atomics -- the old atomicAdd(432 vals -> 24 slots) +
    // 18-way pile onto one sqsum slot was the bottleneck). Instead:
    //   mix[n] = Σ_split ws[split,pos,n]  : warp 0 lanes<N_OUT own one column each and
    //            sum its splits in registers (reads are COALESCED -- at a fixed split
    //            lanes 0..N_OUT-1 read N_OUT contiguous floats). No contention.
    //   Σx² = Σ_split sqr_sum[split,pos]  : warp 1 reduces the splits via __shfl and
    //            writes rms directly. Runs concurrently with warp 0's mix reduce.
    // Split-K reduce (latency-bound -> maximize loads in flight): each warp reduces a
    // 3-column slice of mix (lane = split, __shfl); warp 0 also reduces Σx² -> rms.
    const float* wbase = workspace + static_cast<int64_t>(pos) * N_OUT;
    const int64_t sstride = static_cast<int64_t>(num_positions) * N_OUT;
    if (warp_id == 0) {
        float acc = 0.f;
        for (int s = lane_id; s < num_splits; s += 32)
            acc += sqr_sum[static_cast<int64_t>(s) * num_positions + pos];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane_id == 0) rms_smem[0] = rsqrtf(acc / static_cast<float>(K_DIM) + rms_eps);
    }
    constexpr int NWARPS = EPILOGUE_THREADS / 32;                  // 8
    constexpr int COLS_PER_WARP = (N_OUT + NWARPS - 1) / NWARPS;   // ceil(24/8) = 3
    const int c0 = warp_id * COLS_PER_WARP;
    const int c1 = min(c0 + COLS_PER_WARP, N_OUT);
    for (int c = c0; c < c1; ++c) {
        float acc = 0.f;
        for (int s = lane_id; s < num_splits; s += 32)
            acc += wbase[static_cast<int64_t>(s) * sstride + c];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
        if (lane_id == 0) mix_smem[c] = acc;
    }
    __syncthreads();                                  // mix_smem + rms_smem visible
    if (tid < N_OUT) mix_smem[tid] *= rms_smem[0];    // fold RMSNorm scale into mix
    __syncthreads();                                  // folded mix visible to activation
    if (prof0 && threadIdx.x == 0) prof[3] = ptx::rdclock();  // reduce + rms done

    if (tid < HC) {
        pre_smem[tid] = fast_sigmoid(mix_smem[tid] * scale[0] + base[tid]) + hc_eps;
        if constexpr (kWithPostComb)
            post_smem[tid] = 2.0f * fast_sigmoid(
                mix_smem[HC + tid] * scale[1] + base[HC + tid]);
    }
    if constexpr (kWithPostComb) {
        if (tid < HC * HC)
            comb_smem[tid] = mix_smem[2 * HC + tid] * scale[2] + base[2 * HC + tid];
    }
    __syncthreads();
    if (prof0 && threadIdx.x == 0) prof[4] = ptx::rdclock();  // activation done

    // pre/post gates are ready (activation); write them now -- independent of the
    // Sinkhorn/collapse split below.
    if (tid < HC) {
        pre_out[pos * HC + tid] = pre_smem[tid];
        if constexpr (kWithPostComb)
            post_out[pos * HC + tid] = post_smem[tid];
    }

    // OVERLAP: warp 0 runs Sinkhorn on comb (needs comb_smem); warps 1..7 run the
    // collapse (needs only pre_smem). Independent, run concurrently; which one is the
    // pole depends on M (small M -> Sinkhorn; large M -> collapse). prof: warp 0 stamps
    // sinkhorn end [5]; warp 1 stamps collapse start [6] / end [7] -> real collapse time.
    if (kWithPostComb && warp_id == 0) {
        float v = lane_id < HC * HC ? comb_smem[lane_id] : 0.0f;
        float max_v = v;
        #pragma unroll
        for (int offset = 1; offset < HC; offset <<= 1) {
            max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffffu, max_v, offset));
        }
        const float e = __expf(v - max_v);
        float row_sum = e;
        #pragma unroll
        for (int offset = 1; offset < HC; offset <<= 1) {
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, offset);
        }
        v = e / row_sum + hc_eps;

        float col_sum = v;
        #pragma unroll
        for (int offset = HC; offset < HC * HC; offset <<= 1) {
            col_sum += __shfl_xor_sync(0xffffffffu, col_sum, offset);
        }
        v /= col_sum + hc_eps;

        #pragma unroll 1
        for (int iter = 0; iter < SINKHORN_ITERS - 1; ++iter) {
            row_sum = v;
            #pragma unroll
            for (int offset = 1; offset < HC; offset <<= 1) {
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, offset);
            }
            v /= row_sum + hc_eps;
            col_sum = v;
            #pragma unroll
            for (int offset = HC; offset < HC * HC; offset <<= 1) {
                col_sum += __shfl_xor_sync(0xffffffffu, col_sum, offset);
            }
            v /= col_sum + hc_eps;
        }
        if (lane_id < HC * HC) comb_out[pos * HC * HC + lane_id] = v;   // final gate
        if (prof0 && lane_id == 0) prof[5] = ptx::rdclock();  // sinkhorn end (warp0)
    } else {
        // Collapse: with post/comb, warps 1..7 (the global read overlaps warp 0's
        // Sinkhorn); lite variant: ALL 8 warps (no Sinkhorn to hide).
        //   out[d] = Σ_h pre[h] * hidden[pos, h, d].
        // Vectorized: each thread handles VEC=8 consecutive d as one int4 (16B) load per
        // h + one int4 store -> wide coalesced transactions instead of 2B/thread. hidden
        // (=X) is re-read here from L2; 16B/thread keeps that read cheap.
        constexpr int COLLAPSE_BASE = kWithPostComb ? 32 : 0;
        if (prof0 && threadIdx.x == COLLAPSE_BASE) prof[6] = ptx::rdclock();  // collapse start
        constexpr int VEC = 8;                        // 8 bf16 = 16B = int4
        static_assert(DIM % VEC == 0, "DIM must be VEC-aligned for the int4 collapse");
        float pre_r[HC];
        #pragma unroll
        for (int h = 0; h < HC; ++h) pre_r[h] = pre_smem[h];
        const __nv_bfloat16* src = hidden_states + static_cast<int64_t>(pos) * K_DIM;
        __nv_bfloat16* collapsed = collapsed_out + static_cast<int64_t>(pos) * DIM;
        const int lane_vec = static_cast<int>(threadIdx.x) - COLLAPSE_BASE;
        constexpr int nthreads = EPILOGUE_THREADS - COLLAPSE_BASE;
        if constexpr (kFusedNorm) {
            // ---- Two-phase collapse + FULL fused attn_norm (lite: all 8 warps,
            // this block owns the whole row).
            // Phase 1: compute + KEEP the bf16-rounded x_b in registers (the
            // exact value the reference chain would materialize) and sum x_b^2.
            constexpr int MAX_SEGS = (DIM / VEC + nthreads - 1) / nthreads;   // 4
            __nv_bfloat162 keep[MAX_SEGS][VEC / 2];
            float local_ssq = 0.f;
            int nseg = 0;
            for (int vi = lane_vec; vi < DIM / VEC; vi += nthreads, ++nseg) {
                const int d0 = vi * VEC;
                float acc[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;
                #pragma unroll
                for (int h = 0; h < HC; ++h) {
                    const int4 raw = *reinterpret_cast<const int4*>(src + h * DIM + d0);
                    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                    #pragma unroll
                    for (int k = 0; k < VEC / 2; ++k) {
                        const float2 f = __bfloat1622float2(v2[k]);
                        acc[2 * k]     += pre_r[h] * f.x;
                        acc[2 * k + 1] += pre_r[h] * f.y;
                    }
                }
                #pragma unroll
                for (int k = 0; k < VEC / 2; ++k) {
                    const __nv_bfloat162 xb =
                        __float22bfloat162_rn(make_float2(acc[2 * k], acc[2 * k + 1]));
                    keep[nseg][k] = xb;
                    const float2 f = __bfloat1622float2(xb);
                    local_ssq += f.x * f.x + f.y * f.y;
                }
            }
            // Block-wide ssq reduce (comb_smem is free in the lite variant).
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                local_ssq += __shfl_down_sync(0xffffffffu, local_ssq, o);
            if ((threadIdx.x & 31) == 0) comb_smem[threadIdx.x >> 5] = local_ssq;
            __syncthreads();
            if (threadIdx.x == 0) {
                float total = 0.f;
                #pragma unroll
                for (int w = 0; w < EPILOGUE_THREADS / 32; ++w) total += comb_smem[w];
                sqsum_smem[0] = rsqrtf(total / static_cast<float>(DIM) + attn_norm_eps);
            }
            __syncthreads();
            const float r = sqsum_smem[0];
            // Phase 2: out = bf16( (x_b * r) * w )  -- reference multiply order.
            nseg = 0;
            for (int vi = lane_vec; vi < DIM / VEC; vi += nthreads, ++nseg) {
                const int d0 = vi * VEC;
                const int4 wraw = *reinterpret_cast<const int4*>(attn_norm_w + d0);
                const __nv_bfloat162* w2 =
                    reinterpret_cast<const __nv_bfloat162*>(&wraw);
                float wv[VEC];
                #pragma unroll
                for (int k = 0; k < VEC / 2; ++k) {
                    const float2 f = __bfloat1622float2(w2[k]);   // exact widen
                    wv[2 * k] = f.x; wv[2 * k + 1] = f.y;
                }
                alignas(16) __nv_bfloat162 out2[VEC / 2];
                #pragma unroll
                for (int k = 0; k < VEC / 2; ++k) {
                    const float2 f = __bfloat1622float2(keep[nseg][k]);
                    out2[k] = __floats2bfloat162_rn((f.x * r) * wv[2 * k],
                                                    (f.y * r) * wv[2 * k + 1]);
                }
                *reinterpret_cast<int4*>(collapsed + d0) = *reinterpret_cast<const int4*>(out2);
            }
        } else {
        // Streaming collapse: out = bf16(sum_h pre_r[h] * x_h), no norm (the
        // fused attn_norm variant above is the production form; the old
        // attn_ssq RED fold was deleted with its last consumer).
        for (int vi = lane_vec; vi < DIM / VEC; vi += nthreads) {
            const int d0 = vi * VEC;
            float acc[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;
            #pragma unroll
            for (int h = 0; h < HC; ++h) {
                const int4 raw = *reinterpret_cast<const int4*>(src + h * DIM + d0);
                const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                #pragma unroll
                for (int k = 0; k < VEC / 2; ++k) {
                    const float2 f = __bfloat1622float2(v2[k]);
                    acc[2 * k]     += pre_r[h] * f.x;
                    acc[2 * k + 1] += pre_r[h] * f.y;
                }
            }
            alignas(16) __nv_bfloat162 out2[VEC / 2];   // 16B-aligned for the int4 store
            #pragma unroll
            for (int k = 0; k < VEC / 2; ++k)
                out2[k] = __float22bfloat162_rn(make_float2(acc[2 * k], acc[2 * k + 1]));
            *reinterpret_cast<int4*>(collapsed + d0) = *reinterpret_cast<const int4*>(out2);
        }
        }   // !kFusedNorm (streaming collapse)
        if (prof0 && threadIdx.x == COLLAPSE_BASE) prof[7] = ptx::rdclock();  // collapse end
    }
    // PDL exit (parity with vLLM's pdl_trigger): let the NEXT op in the
    // chain prelaunch under our tail; its own griddepcontrol.wait provides
    // the ordering. No-op for non-PDL consumers.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Launch one fuse-kernel variant, optionally as a PDL SECONDARY (PSS attr):
// with a PDL-aware producer (deep_gemm set_pdl(True)) the epilogue's launch
// latency + prologue hide under the GEMM tail; the in-kernel
// griddepcontrol.wait provides the ordering.
using FuseKernelT = void (*)(
    const __nv_bfloat16*, const float*, const float*, const float*,
    const float*, float, float, int, int, __nv_bfloat16*, float*, float*,
    float*, const __nv_bfloat16*, float, int64_t*);

static void launch_fuse_variant(
    FuseKernelT k, int m, int smem_bytes, cudaStream_t stream, bool pdl,
    const __nv_bfloat16* x_ptr, const float* ws, const float* sq,
    const float* base, const float* scale, float hc_eps, float rms_eps,
    int num_splits, __nv_bfloat16* out_ptr, float* pre, float* post,
    float* comb, const __nv_bfloat16* norm_w, float norm_eps) {
    if (!pdl) {
        k<<<m, EPILOGUE_THREADS, smem_bytes, stream>>>(
            x_ptr, ws, sq, base, scale, hc_eps, rms_eps, m, num_splits,
            out_ptr, pre, post, comb, norm_w, norm_eps, nullptr);
        return;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(m));
    cfg.blockDim = dim3(EPILOGUE_THREADS);
    cfg.dynamicSmemBytes = static_cast<size_t>(smem_bytes);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    int64_t* prof = nullptr;
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, k, x_ptr, ws, sq, base, scale,
                                   hc_eps, rms_eps, m, num_splits, out_ptr,
                                   pre, post, comb, norm_w, norm_eps,
                                   prof) == cudaSuccess,
                "PDL fuse launch failed");
}

// Epilogue-only entry: consume an EXTERNALLY produced split-K workspace and
// run OUR fused epilogue (reduce + RMSNorm + gates + collapse [+ attn_norm]).
// Contract == deep_gemm.tf32_hc_prenorm_gemm's outputs, which match our own
// GEMM's scratch layout exactly:
//   workspace [n_splits, M, N_OUT] fp32   (mix partials)
//   sqr_sum   [n_splits, M]        fp32   (input-RMSNorm sum(x^2) partials)
// Hybrid rationale (measured on B300): deep_gemm's tf32 GEMM beats our
// splitk by ~2-3us while our epilogue beats vLLM's TileLang fuse by ~3-4us.
static void hc_reduce_fuse_out(
    torch::Tensor hidden_states, torch::Tensor workspace, torch::Tensor sqr_sum,
    torch::Tensor attn_hc_base, torch::Tensor attn_hc_scale,
    double hc_eps, double rms_norm_eps,
    torch::Tensor collapsed, torch::Tensor pre, torch::Tensor post,
    torch::Tensor comb, bool with_post_comb,
    c10::optional<torch::Tensor> attn_norm_w, double attn_norm_eps,
    bool pdl) {
    TORCH_CHECK(hidden_states.is_cuda() &&
                hidden_states.scalar_type() == torch::kBFloat16,
                "hidden_states must be CUDA bf16");
    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    auto hs = hidden_states.contiguous().view({-1, K_DIM});
    const int m = static_cast<int>(hs.size(0));
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() &&
                workspace.scalar_type() == torch::kFloat32 &&
                workspace.dim() == 3 && workspace.size(1) == m &&
                workspace.size(2) == N_OUT,
                "workspace must be contiguous fp32 [n_splits, m, ", N_OUT, "]");
    const int num_splits = static_cast<int>(workspace.size(0));
    TORCH_CHECK(sqr_sum.is_cuda() && sqr_sum.is_contiguous() &&
                sqr_sum.scalar_type() == torch::kFloat32 &&
                sqr_sum.numel() == static_cast<int64_t>(num_splits) * m,
                "sqr_sum must be contiguous fp32 [n_splits, m]");
    TORCH_CHECK(collapsed.is_cuda() && collapsed.is_contiguous() &&
                collapsed.scalar_type() == torch::kBFloat16 &&
                collapsed.numel() == static_cast<int64_t>(m) * DIM,
                "collapsed must be contiguous bf16 with m*DIM elements");
    TORCH_CHECK(pre.is_contiguous() && pre.numel() == static_cast<int64_t>(m) * HC &&
                post.is_contiguous() && post.numel() == static_cast<int64_t>(m) * HC &&
                comb.is_contiguous() && comb.numel() == static_cast<int64_t>(m) * HC * HC,
                "pre/post/comb must be contiguous fp32");

    const __nv_bfloat16* attn_norm_w_ptr = nullptr;
    if (attn_norm_w.has_value() && attn_norm_w->numel() > 0) {
        TORCH_CHECK(!with_post_comb, "fused attn_norm requires the lite variant");
        TORCH_CHECK(attn_norm_w->is_cuda() && attn_norm_w->is_contiguous() &&
                    attn_norm_w->scalar_type() == torch::kBFloat16 &&
                    attn_norm_w->numel() == DIM,
                    "attn_norm_w must be bf16 [DIM] (the CHECKPOINT dtype; "
                    "the kernel widens to fp32 losslessly)");
        attn_norm_w_ptr =
            reinterpret_cast<const __nv_bfloat16*>(attn_norm_w->data_ptr());
    }

    constexpr int scratch_floats = 1 + 1 + N_OUT + HC + HC + HC * HC;
    constexpr int fuse_smem_bytes = scratch_floats * sizeof(float);
    static bool cfgd = false;
    if (!cfgd) {
        for (auto* fptr : {reinterpret_cast<void*>(&hc_reduce_and_fuse_kernel<true, false>),
                           reinterpret_cast<void*>(&hc_reduce_and_fuse_kernel<false, false>),
                           reinterpret_cast<void*>(&hc_reduce_and_fuse_kernel<false, true>)}) {
            TORCH_CHECK(cudaFuncSetAttribute(fptr,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        fuse_smem_bytes) == cudaSuccess, "smem attr failed");
        }
        cfgd = true;
    }
    const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(hs.data_ptr());
    auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    FuseKernelT k = with_post_comb ? hc_reduce_and_fuse_kernel<true, false>
                  : (attn_norm_w_ptr != nullptr
                     ? hc_reduce_and_fuse_kernel<false, true>
                     : hc_reduce_and_fuse_kernel<false, false>);
    launch_fuse_variant(k, m, fuse_smem_bytes, stream, pdl,
                        x_ptr, workspace.data_ptr<float>(),
                        sqr_sum.data_ptr<float>(),
                        attn_hc_base.data_ptr<float>(),
                        attn_hc_scale.data_ptr<float>(),
                        static_cast<float>(hc_eps),
                        static_cast<float>(rms_norm_eps), num_splits,
                        out_ptr, pre.data_ptr<float>(), post.data_ptr<float>(),
                        comb.data_ptr<float>(), attn_norm_w_ptr,
                        static_cast<float>(attn_norm_eps));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hc_reduce_fuse launch failed");
}

}  // namespace hc_tc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_reduce_fuse_out", &hc_tc::hc_reduce_fuse_out,
          "Epilogue-only: reduce an EXTERNAL split-K workspace (layout == "
          "deep_gemm.tf32_hc_prenorm_gemm outputs: mul [S,m,24] + sqrsum [S,m]) "
          "with OUR fused epilogue (hybrid: deep_gemm GEMM + our reduce/fuse)",
          py::arg("hidden_states"), py::arg("workspace"), py::arg("sqr_sum"),
          py::arg("attn_hc_base"), py::arg("attn_hc_scale"),
          py::arg("hc_eps"), py::arg("rms_norm_eps"),
          py::arg("collapsed"), py::arg("pre"), py::arg("post"), py::arg("comb"),
          py::arg("with_post_comb") = false,
          py::arg("attn_norm_w") = c10::nullopt,
          py::arg("attn_norm_eps") = 1e-6,
          py::arg("pdl") = true);
}
