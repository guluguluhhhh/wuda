// DeepSeek-V4 MHC fused forward for NVIDIA Blackwell/B300.
//
// The projection is a very tall-K, narrow-N tf32 GEMM (bf16 activation, fp32 weight):
//     [M, 28672] x [24, 28672]^T -> [M, 24].
// The tcgen05 split-K GEMM engine lives in include/hc_fused_kernel_tc.cuh; this TU owns
// the fused epilogue (split-K reduce + RMSNorm + gates + Sinkhorn + collapse)
// and the PyTorch binding.

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
template <bool kWithPostComb>
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
    int64_t* __restrict__ prof) {
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
        if (prof0 && threadIdx.x == COLLAPSE_BASE) prof[7] = ptx::rdclock();  // collapse end
    }
}

static void hc_validate_inputs(
    const torch::Tensor& hidden_states, const torch::Tensor& attn_hc_fn,
    const torch::Tensor& attn_hc_base, const torch::Tensor& attn_hc_scale) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bf16");
    TORCH_CHECK(hidden_states.dim() == 2 || hidden_states.dim() == 3,
                "hidden_states must be [HC,DIM] or [M,HC,DIM]");
    if (hidden_states.dim() == 2) {
        TORCH_CHECK(hidden_states.size(0) == HC && hidden_states.size(1) == DIM,
                    "2D hidden_states must be [4,7168]");
    } else {
        TORCH_CHECK(hidden_states.size(1) == HC && hidden_states.size(2) == DIM,
                    "3D hidden_states must be [M,4,7168]");
    }
    TORCH_CHECK(attn_hc_fn.is_cuda() && attn_hc_fn.scalar_type() == torch::kFloat32,
                "attn_hc_fn must be CUDA fp32 (tf32 GEMM reads the fp32 weight as tf32)");
    TORCH_CHECK(attn_hc_fn.dim() == 2 && attn_hc_fn.size(0) == N_OUT &&
                attn_hc_fn.size(1) == K_DIM, "attn_hc_fn must be [24,28672]");
    TORCH_CHECK(attn_hc_base.is_cuda() && attn_hc_base.scalar_type() == torch::kFloat32 &&
                attn_hc_base.numel() == N_OUT, "attn_hc_base must be CUDA fp32 [24]");
    TORCH_CHECK(attn_hc_scale.is_cuda() && attn_hc_scale.scalar_type() == torch::kFloat32 &&
                attn_hc_scale.numel() == 3, "attn_hc_scale must be CUDA fp32 [3]");
    TORCH_CHECK(hidden_states.get_device() == attn_hc_fn.get_device() &&
                hidden_states.get_device() == attn_hc_base.get_device() &&
                hidden_states.get_device() == attn_hc_scale.get_device(),
                "all inputs must be on the same CUDA device");
}

// Core launch: prepared contiguous inputs + caller-owned output pointers. No
// allocation of outputs here (mirrors cgemm's fusenorm_run(ctx,out,...)); the
// split-K workspace is a reused thread_local scratch. prof_dev != nullptr enables
// clock64 stamps.
static void hc_launch_core(
    const torch::Tensor& hs, const torch::Tensor& weight,
    const float* base_ptr, const float* scale_ptr,
    float hc_eps, float rms_eps,
    __nv_bfloat16* collapsed_ptr, float* pre_ptr, float* post_ptr, float* comb_ptr,
    int64_t* prof_dev, bool with_post_comb = true) {
    const int m = static_cast<int>(hs.size(0));
    const SplitConfig cfg = make_split_config(m);
    auto fp32_opts = hs.options().dtype(torch::kFloat32);

    // Split-K partials: mix[splits,M,N_OUT] and the input-RMSNorm Σx²[splits,M].
    // Both are internal scratch -> reuse cached buffers (no per-call alloc).
    static thread_local torch::Tensor ws_cache;
    const int64_t ws_need = static_cast<int64_t>(cfg.num_splits) * m * N_OUT;
    if (!ws_cache.defined() || ws_cache.numel() < ws_need ||
        ws_cache.device() != hs.device()) {
        ws_cache = torch::empty({ws_need}, fp32_opts);
    }
    float* workspace_ptr = ws_cache.data_ptr<float>();

    static thread_local torch::Tensor sq_cache;
    const int64_t sq_need = static_cast<int64_t>(cfg.num_splits) * m;
    if (!sq_cache.defined() || sq_cache.numel() < sq_need ||
        sq_cache.device() != hs.device()) {
        sq_cache = torch::empty({sq_need}, fp32_opts);
    }
    float* sqr_sum_ptr = sq_cache.data_ptr<float>();

    // tf32 path: activation x stays bf16 (cast group upcasts to tf32 on-chip); weight is
    // fp32 (read as tf32 by the MMA) -- this is where the precision vs bf16 is recovered.
    const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(hs.data_ptr());
    const auto* w_ptr = reinterpret_cast<const float*>(weight.data_ptr());
    thread_local TmaCache cache;
    if (cache.x_ptr != x_ptr || cache.w_ptr != w_ptr ||
        cache.m != m || cache.m_tile != cfg.m_tile || cache.n_tile != cfg.n_tile) {
        cache.x = make_tma_bf16_2d("hidden", x_ptr, m, K_DIM, cfg.m_tile);
        cache.w = make_tma_fp32_2d("weight", w_ptr, N_OUT, K_DIM, cfg.n_tile);
        cache.x_ptr = x_ptr; cache.w_ptr = w_ptr; cache.m = m;
        cache.m_tile = cfg.m_tile; cache.n_tile = cfg.n_tile;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gemm_dispatch(cache.x, cache.w, cfg, m, workspace_ptr, sqr_sum_ptr, prof_dev, stream);

    // epilogue smem scratch: rms + sqsum + mix + pre + post + comb (no hidden staged).
    constexpr int scratch_floats = 1 + 1 + N_OUT + HC + HC + HC * HC;
    constexpr int fuse_smem_bytes = scratch_floats * sizeof(float);
    static bool fuse_configured = false;
    if (!fuse_configured) {
        for (auto* fptr : {reinterpret_cast<void*>(&hc_reduce_and_fuse_kernel<true>),
                           reinterpret_cast<void*>(&hc_reduce_and_fuse_kernel<false>)}) {
            const auto attr_err = cudaFuncSetAttribute(
                fptr, cudaFuncAttributeMaxDynamicSharedMemorySize, fuse_smem_bytes);
            TORCH_CHECK(attr_err == cudaSuccess,
                        "cudaFuncSetAttribute(fuse) failed: ", cudaGetErrorString(attr_err),
                        " smem=", fuse_smem_bytes);
        }
        fuse_configured = true;
    }

    if (with_post_comb) {
        hc_reduce_and_fuse_kernel<true><<<m, EPILOGUE_THREADS, fuse_smem_bytes, stream>>>(
            x_ptr, workspace_ptr, sqr_sum_ptr, base_ptr, scale_ptr, hc_eps, rms_eps,
            m, cfg.num_splits, collapsed_ptr, pre_ptr, post_ptr, comb_ptr, prof_dev);
    } else {
        hc_reduce_and_fuse_kernel<false><<<m, EPILOGUE_THREADS, fuse_smem_bytes, stream>>>(
            x_ptr, workspace_ptr, sqr_sum_ptr, base_ptr, scale_ptr, hc_eps, rms_eps,
            m, cfg.num_splits, collapsed_ptr, pre_ptr, post_ptr, comb_ptr, prof_dev);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "HC reduce/fuse launch failed: ", cudaGetErrorString(cudaGetLastError()));
}

static std::vector<torch::Tensor> hc_run_impl(
    torch::Tensor hidden_states,
    torch::Tensor attn_hc_fn,
    torch::Tensor attn_hc_base,
    torch::Tensor attn_hc_scale,
    double hc_eps,
    double rms_norm_eps,
    bool profile,
    bool with_post_comb = true) {
    hc_validate_inputs(hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale);

    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    auto hs = hidden_states.contiguous().view({-1, K_DIM});
    auto weight = attn_hc_fn.contiguous();
    auto base = attn_hc_base.contiguous();
    auto scale = attn_hc_scale.contiguous();
    const int m = static_cast<int>(hs.size(0));
    TORCH_CHECK(m > 0, "num_positions must be positive");

    auto fp32_opts = hs.options().dtype(torch::kFloat32);
    auto bf16_opts = hs.options().dtype(torch::kBFloat16);
    auto collapsed = torch::empty({m, DIM}, bf16_opts);
    auto pre = torch::empty({m, HC}, fp32_opts);
    auto post = torch::empty({m, HC}, fp32_opts);
    auto comb = torch::empty({m, HC, HC}, fp32_opts);
    int64_t* prof_dev = nullptr;
    torch::Tensor timing;
    if (profile) {
        timing = torch::zeros({PROF_SLOTS}, hs.options().dtype(torch::kInt64));
        prof_dev = timing.data_ptr<int64_t>();
    }

    hc_launch_core(hs, weight, base.data_ptr<float>(), scale.data_ptr<float>(),
                   static_cast<float>(hc_eps), static_cast<float>(rms_norm_eps),
                   reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr()),
                   pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
                   prof_dev, with_post_comb);

    std::vector<torch::Tensor> out;
    if (hidden_states.dim() == 2) {
        out = {collapsed.squeeze(0), pre.squeeze(0), post.squeeze(0), comb.squeeze(0)};
    } else {
        out = {collapsed, pre, post, comb};
    }
    if (profile) out.push_back(timing);  // int64 [PROF_SLOTS] clock64 stamps
    return out;
}

// Preallocated-output variant (cgemm-style: caller owns collapsed/pre/post/comb).
// Skips the per-call output torch::empty + Python list marshalling -> lower eager
// host dispatch. Outputs must be contiguous, correct dtype, and hold m rows.
static void hc_fused_forward_out(
    torch::Tensor hidden_states, torch::Tensor attn_hc_fn,
    torch::Tensor attn_hc_base, torch::Tensor attn_hc_scale,
    double hc_eps, double rms_norm_eps,
    torch::Tensor collapsed, torch::Tensor pre, torch::Tensor post, torch::Tensor comb,
    bool with_post_comb) {
    hc_validate_inputs(hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale);

    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    auto hs = hidden_states.contiguous().view({-1, K_DIM});
    auto weight = attn_hc_fn.contiguous();
    auto base = attn_hc_base.contiguous();
    auto scale = attn_hc_scale.contiguous();
    const int m = static_cast<int>(hs.size(0));
    TORCH_CHECK(m > 0, "num_positions must be positive");
    TORCH_CHECK(collapsed.is_cuda() && collapsed.is_contiguous() &&
                collapsed.scalar_type() == torch::kBFloat16 &&
                collapsed.numel() == static_cast<int64_t>(m) * DIM,
                "collapsed must be contiguous bf16 with m*DIM elements");
    TORCH_CHECK(pre.is_contiguous()  && pre.numel()  == static_cast<int64_t>(m) * HC &&
                post.is_contiguous() && post.numel() == static_cast<int64_t>(m) * HC &&
                comb.is_contiguous() && comb.numel() == static_cast<int64_t>(m) * HC * HC &&
                pre.scalar_type() == torch::kFloat32 &&
                post.scalar_type() == torch::kFloat32 &&
                comb.scalar_type() == torch::kFloat32,
                "pre/post/comb must be contiguous fp32 [m,HC]/[m,HC]/[m,HC,HC]");

    hc_launch_core(hs, weight, base.data_ptr<float>(), scale.data_ptr<float>(),
                   static_cast<float>(hc_eps), static_cast<float>(rms_norm_eps),
                   reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr()),
                   pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
                   nullptr, with_post_comb);
}

static std::vector<torch::Tensor> hc_fused_forward_full(
    torch::Tensor hidden_states, torch::Tensor attn_hc_fn,
    torch::Tensor attn_hc_base, torch::Tensor attn_hc_scale,
    double hc_eps, double rms_norm_eps) {
    return hc_run_impl(hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale,
                       hc_eps, rms_norm_eps, /*profile=*/false);
}

// Profiled: returns {collapsed, pre, post, comb, timing[PROF_SLOTS]} where timing
// holds clock64 stamps (block 0). Layout (see hc_fused_kernel_tc.cuh):
//   [0]gemm start [1]gemm end | [2]epi start [3]reduce+rms [4]act
//   [5]sinkhorn end | [6]collapse start [7]collapse end (warp 1).
//   stages: gemm=[1]-[0] reduce=[3]-[2] sinkhorn=[5]-[4] collapse=[7]-[6].
static std::vector<torch::Tensor> hc_fused_forward_profiled(
    torch::Tensor hidden_states, torch::Tensor attn_hc_fn,
    torch::Tensor attn_hc_base, torch::Tensor attn_hc_scale,
    double hc_eps, double rms_norm_eps, bool with_post_comb) {
    return hc_run_impl(hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale,
                       hc_eps, rms_norm_eps, /*profile=*/true, with_post_comb);
}

static std::vector<int64_t> hc_fused_tc_config(int64_t num_positions) {
    TORCH_CHECK(num_positions > 0 && num_positions <= INT32_MAX,
                "num_positions is out of range");
    const SplitConfig cfg = make_split_config(static_cast<int>(num_positions));
    return {cfg.n_tile, cfg.num_splits, cfg.k_tiles_per_split,
            cfg.grid, cfg.num_m_tiles, cfg.num_n_tiles, cfg.num_sms, cfg.m_tile};
}

}  // namespace hc_tc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_fused_forward_full", &hc_tc::hc_fused_forward_full,
          "MHC fused forward (tf32 tcgen05 + split-K + fused HC epilogue)");
    m.def("hc_fused_forward_profiled", &hc_tc::hc_fused_forward_profiled,
          "MHC fused forward + clock64 stamps -> {collapsed,pre,post,comb,timing[8]}; "
          "with_post_comb=False profiles the lite variant ([5] sinkhorn not stamped)",
          py::arg("hidden_states"), py::arg("attn_hc_fn"), py::arg("attn_hc_base"),
          py::arg("attn_hc_scale"), py::arg("hc_eps"), py::arg("rms_norm_eps"),
          py::arg("with_post_comb") = true);
    m.def("hc_fused_forward_out", &hc_tc::hc_fused_forward_out,
          "MHC fused forward into preallocated {collapsed,pre,post,comb} (no alloc, no return); "
          "with_post_comb=False = lite: full reduce, but post/comb activation + Sinkhorn "
          "dropped and ALL warps collapse",
          py::arg("hidden_states"), py::arg("attn_hc_fn"), py::arg("attn_hc_base"),
          py::arg("attn_hc_scale"), py::arg("hc_eps"), py::arg("rms_norm_eps"),
          py::arg("collapsed"), py::arg("pre"), py::arg("post"), py::arg("comb"),
          py::arg("with_post_comb") = true);
    m.def("hc_fused_tc_config", &hc_tc::hc_fused_tc_config,
          "Return [N tile, split-K, K tiles/split, grid, M tiles, N tiles, SMs, M tile]");
}
