// ============================================================
// DeepSeek-V4 Phase 3: wq_b Projection + Per-Head RMSNorm
// BN=128 + cluster(4,1,1): high GEMM efficiency + DSMEM RMSNorm
// 4 blocks per cluster = 4×128 = 512 = head_dim → full norm in-cluster
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm80.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/copy.hpp"

using namespace cute;

// ============================================================
// Configuration
// ============================================================
constexpr int BN = 512;         // = head_dim (full head for fused RMSNorm)
constexpr int BK = 64;
constexpr int PAD = 8;
constexpr int STAGES = 2;
constexpr int NUM_THREADS = 1024;

constexpr int SA_STRIDE = BK + PAD;  // 40
constexpr int SB_STRIDE = BK + PAD;  // 40

constexpr int NUM_HEADS = 128;
constexpr int HEAD_DIM = 512;   // = CLUSTER_N × BN
constexpr int K_DIM = 1536;
constexpr int N_TOTAL = NUM_HEADS * HEAD_DIM;

using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
// cp.async TiledCopy for B tile [512, 32]
// 1024 thread slots (256×4), all threads participate
using CopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, __nv_bfloat16>;
using TiledCopyB = decltype(make_tiled_copy(
    CopyAtomB{},
    make_layout(Shape<_256, _4>{}, GenRowMajor{}),  // 1024 threads
    make_layout(Shape<_1, _8>{})                     // 128-bit = 8 bf16
));

// Smem layouts (templated on BM)
template <int BM>
struct SmemConfig {
    using LayoutA = decltype(make_layout(
        make_shape(Int<BM>{}, Int<BK>{}, Int<STAGES>{}),
        make_stride(Int<SA_STRIDE>{}, Int<1>{}, Int<BM * SA_STRIDE>{})));
    using LayoutB = decltype(make_layout(
        make_shape(Int<BN>{}, Int<BK>{}, Int<STAGES>{}),
        make_stride(Int<SB_STRIDE>{}, Int<1>{}, Int<BN * SB_STRIDE>{})));
};

// ============================================================
// Device utilities
// ============================================================
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================
// Kernel: BN=128 GEMM + cluster DSMEM RMSNorm
// Grid: (N_TOTAL/BN=512, M/BM), cluster_dim(4,1,1)
// Each cluster of 4 blocks covers 1 head (512 dims)
// ============================================================
template <int BM>
__global__ void __launch_bounds__(NUM_THREADS)
wq_b_proj_kernel(
    const __nv_bfloat16* __restrict__ gX_ptr,
    const __nv_bfloat16* __restrict__ gW_ptr,
    const float* __restrict__ rms_w,
    __nv_bfloat16* __restrict__ gOut_ptr,
    float eps,
    int M
) {
    const int head_idx = blockIdx.x;   // [0, 128) = N_TOTAL/BN = 128 heads
    const int m_block = blockIdx.y;

    constexpr int MMA_M = BM / 16;
    constexpr int MMA_N = 32 / MMA_M;
    using TiledMma = decltype(make_tiled_mma(
        MmaAtom{}, make_layout(Shape<Int<MMA_M>, Int<MMA_N>, _1>{})));

    // ================================================================
    // Shared memory
    // ================================================================
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B = smem_A + STAGES * BM * SA_STRIDE;

    auto sA = make_tensor(make_smem_ptr(smem_A), typename SmemConfig<BM>::LayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B), typename SmemConfig<BM>::LayoutB{});

    // ================================================================
    // Global memory & copy setup
    // ================================================================
    const __nv_bfloat16* gA_base = gX_ptr + m_block * BM * K_DIM;
    const __nv_bfloat16* gB_base = gW_ptr + head_idx * HEAD_DIM * K_DIM;

    // B global tensor for CuTe TiledCopy
    auto mB_g = make_tensor(make_gmem_ptr(gB_base),
                            make_layout(make_shape(Int<BN>{}, K_DIM),
                                        make_stride(K_DIM, Int<1>{})));
    auto gB = local_tile(mB_g, make_shape(Int<BN>{}, Int<BK>{}), make_coord(0, _));

    // cp.async B: all 1024 threads participate (no guard needed for BN=512)
    TiledCopyB tiled_copy_B;
    auto thr_copy_B = tiled_copy_B.get_slice(threadIdx.x);

    // Manual A copy (tiny: BM*BK elements)
    auto load_A = [&](int kt, int stage) {
        const __nv_bfloat16* src = gA_base + kt * BK;
        __nv_bfloat16* dst = smem_A + stage * BM * SA_STRIDE;
        constexpr int A_ELEMS = BM * BK;
        #pragma unroll
        for (int i = 0; i < (A_ELEMS + NUM_THREADS - 1) / NUM_THREADS; ++i) {
            int flat = threadIdx.x + i * NUM_THREADS;
            if (flat < A_ELEMS) {
                int row = flat / BK;
                int col = flat % BK;
                dst[row * SA_STRIDE + col] = src[row * K_DIM + col];
            }
        }
    };

    // ================================================================
    // TiledMMA + accumulator
    // ================================================================
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto out_ref = make_tensor(
        static_cast<float*>(nullptr),
        make_layout(make_shape(Int<BM>{}, Int<BN>{}), make_stride(Int<BN>{}, Int<1>{})));
    auto tCrC = thr_mma.partition_fragment_C(out_ref);
    clear(tCrC);

    auto cOut = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto tCcOut = thr_mma.partition_C(cOut);

    // ================================================================
    // GEMM K-loop (cp.async double-buffer, threads 0-511 do B copy)
    // ================================================================
    constexpr int NUM_K_TILES = K_DIM / BK;

    // Prologue: load first tile
    load_A(0, 0);
    {
        auto tBgB_0 = thr_copy_B.partition_S(gB(_, _, 0));
        auto tBsB_0 = thr_copy_B.partition_D(sB(_, _, 0));
        copy(tiled_copy_B, tBgB_0, tBsB_0);
    }
    cp_async_fence();

    for (int kt = 0; kt < NUM_K_TILES; ++kt) {
        const int cur_stage = kt % STAGES;

        // Wait for current stage
        cp_async_wait<0>();
        __syncthreads();

        // Issue next load into OTHER stage
        if (kt + 1 < NUM_K_TILES) {
            const int nxt_stage = (kt + 1) % STAGES;
            load_A(kt + 1, nxt_stage);
            auto tBgB_n = thr_copy_B.partition_S(gB(_, _, kt + 1));
            auto tBsB_n = thr_copy_B.partition_D(sB(_, _, nxt_stage));
            copy(tiled_copy_B, tBgB_n, tBsB_n);
        }
        cp_async_fence();

        // MMA on current stage
        auto tCsA_k = thr_mma.partition_A(sA(_, _, cur_stage));
        auto tCsB_k = thr_mma.partition_B(sB(_, _, cur_stage));
        cute::gemm(tiled_mma, tCsA_k, tCsB_k, tCrC);

        __syncthreads();
    }

    // ================================================================
    // Epilogue: In-CTA RMSNorm (BN=512 covers full head_dim)
    // ================================================================
#ifndef SKIP_NORM

    constexpr int NUM_WARPS = NUM_THREADS / 32;
    float* warp_partials = reinterpret_cast<float*>(smem_raw);
    float* row_sq_sums = warp_partials + BM * NUM_WARPS;
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    float my_sq[BM];
    #pragma unroll
    for (int r = 0; r < BM; ++r) my_sq[r] = 0.0f;

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcOut(i);
        my_sq[get<0>(coord)] += tCrC(i) * tCrC(i);
    }

    #pragma unroll
    for (int r = 0; r < BM; ++r) {
        float ws = warp_reduce_sum_f32(my_sq[r]);
        if (lane_id == 0) warp_partials[r * NUM_WARPS + warp_id] = ws;
    }
    __syncthreads();

    if (threadIdx.x < BM) {
        float total = 0.0f;
        for (int w = 0; w < NUM_WARPS; ++w) total += warp_partials[threadIdx.x * NUM_WARPS + w];
        row_sq_sums[threadIdx.x] = rsqrtf(total / (float)HEAD_DIM + eps);
    }
    __syncthreads();

    const int out_base_m = m_block * BM;
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcOut(i);
        int row = get<0>(coord);
        int col = get<1>(coord);
        int gm = out_base_m + row;
        if (gm < M) {
            float normed = tCrC(i) * row_sq_sums[row] * rms_w[col];
            gOut_ptr[gm * NUM_HEADS * HEAD_DIM + head_idx * HEAD_DIM + col] =
                __float2bfloat16(normed);
        }
    }

#else  // SKIP_NORM

    const int out_base_m = m_block * BM;

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcOut(i);
        int row = get<0>(coord);
        int col = get<1>(coord);
        int gm = out_base_m + row;
        if (gm < M) {
            gOut_ptr[gm * NUM_HEADS * HEAD_DIM + head_idx * HEAD_DIM + col] =
                __float2bfloat16(tCrC(i));
        }
    }

#endif  // SKIP_NORM
}

// ============================================================
// PyTorch Binding
// ============================================================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps
) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(rms_w.scalar_type() == torch::kFloat32);

    const int M = x.size(0);
    TORCH_CHECK(x.size(1) == K_DIM);
    TORCH_CHECK(w.size(0) == N_TOTAL && w.size(1) == K_DIM);
    TORCH_CHECK(rms_w.numel() == HEAD_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0);

    auto out = torch::empty({M, NUM_HEADS, HEAD_DIM},
                            torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16));

    auto stream = at::cuda::getCurrentCUDAStream();
    auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto w_ptr = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    auto launch = [&](auto bm_tag) {
        constexpr int BM = decltype(bm_tag)::value;
        dim3 grid(NUM_HEADS, M / BM);  // (128, M/BM)
        dim3 block(NUM_THREADS);
        int smem = STAGES * (BM * SA_STRIDE + BN * SB_STRIDE) * sizeof(__nv_bfloat16);
        cudaFuncSetAttribute(wq_b_proj_kernel<BM>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        wq_b_proj_kernel<BM><<<grid, block, smem, stream>>>(
            x_ptr, w_ptr, rms_w.data_ptr<float>(), out_ptr, (float)eps, M);
    };

    if (M % 64 == 0) launch(Int<64>{});
    else              launch(Int<32>{});

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj + RMSNorm fused (BN=512, cp.async + PAD + CuTe MMA)");
}
