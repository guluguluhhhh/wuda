// ============================================================
// DeepSeek-V4 Phase 2: Small Projections (CuTe TC GEMM)
// Batched GEMM: [B, K] x [K, N] -> [B, N] using Tensor Core
// + RMSNorm fusion on output
// Weight: bf16 [N, K] original, transposed to [K, N] at Python side
// Uses CuTe MMA atoms for Tensor Core, batch≥16 for efficiency
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include "../include/phase2_proj_kernel.cuh"

using namespace cute;

// ============================================================
// Block tile configuration
// ============================================================
constexpr int BM = 64;    // output rows per block
constexpr int BN = 64;    // output cols per block
constexpr int BK = 32;    // K-tile per iteration
constexpr int WARPS_GEMM = 4;  // 4 warps = 128 threads per block
constexpr int THREADS_GEMM = WARPS_GEMM * 32;  // 128

// ============================================================
// CuTe TC GEMM Kernel: C[M,N] = A[M,K] x B[K,N]
// A: [M, K] row-major (activations)
// B: [K, N] row-major (transposed weight)
// C: [M, N] row-major (output, fp32 accumulator)
// ============================================================
__global__ void __launch_bounds__(128)
gemm_bf16_tc_kernel(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [K, N] row-major
    float* __restrict__ C,                // [M, N] row-major (fp32)
    int M, int K, int N
) {
    // Block tile coordinates
    int bm = blockIdx.x;  // M-tile index
    int bn = blockIdx.y;  // N-tile index

    // Global tensor views
    auto mA = make_tensor(make_gmem_ptr(A),
        make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto mB = make_tensor(make_gmem_ptr(B),
        make_layout(make_shape(K, N), make_stride(N, Int<1>{})));
    auto mC = make_tensor(make_gmem_ptr(C),
        make_layout(make_shape(M, N), make_stride(N, Int<1>{})));

    // Extract this block's tile from global
    auto gA = local_tile(mA, make_shape(Int<BM>{}, Int<BK>{}), make_coord(bm, _));  // [BM, BK, k]
    auto gB = local_tile(mB, make_shape(Int<BK>{}, Int<BN>{}), make_coord(_, bn));  // [BK, BN, k]
    auto gC = local_tile(mC, make_shape(Int<BM>{}, Int<BN>{}), make_coord(bm, bn)); // [BM, BN]

    // Shared memory for A and B tiles
    __shared__ __nv_bfloat16 smemA[BM * BK];  // 64*32*2 = 4 KB
    __shared__ __nv_bfloat16 smemB[BK * BN];  // 32*64*2 = 4 KB

    auto sA = make_tensor(make_smem_ptr(smemA),
        make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{})));
    auto sB = make_tensor(make_smem_ptr(smemB),
        make_layout(make_shape(Int<BK>{}, Int<BN>{}), make_stride(Int<BN>{}, Int<1>{})));

    // Define MMA: SM80_16x8x16 for bf16 -> fp32
    // TN layout: A is row-major (M-major), B is col-major (N-major) for the MMA
    // But our B is [K, N] row-major, so we need TN: A=row(M,K), B=col(N,K)
    // Since B stored as [K,N] row-major = [N,K] col-major transposed
    // We treat B as: reading B[K,N] but MMA wants B as (N,K) -> we transpose in smem load
    auto tiled_mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_2, _2, _1>>{}  // 2x2 warp arrangement = 4 warps
    );

    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));  // accumulator
    clear(tCrC);

    // Copy atoms for loading A and B to smem
    auto copyA = make_tiled_copy(
        Copy_Atom<UniversalCopy<__nv_bfloat16>, __nv_bfloat16>{},
        Layout<Shape<_32, _4>>{},   // 128 threads: 32 threads x 4
        Layout<Shape<_1, _1>>{}     // each thread copies 1 element
    );
    auto copyB = make_tiled_copy(
        Copy_Atom<UniversalCopy<__nv_bfloat16>, __nv_bfloat16>{},
        Layout<Shape<_32, _4>>{},
        Layout<Shape<_1, _1>>{}
    );

    auto thr_copyA = copyA.get_slice(threadIdx.x);
    auto thr_copyB = copyB.get_slice(threadIdx.x);

    // Number of K-tiles
    int num_k = K / BK;

    // Main loop over K dimension
    for (int k = 0; k < num_k; k++) {
        // Bounds check for M edge tiles
        int m_start = bm * BM;
        int n_start = bn * BN;

        // Load A tile: gA[:, :, k] -> sA
        // Manual cooperative load (simple, correct)
        {
            int elems_A = BM * BK;  // 2048
            int per_thr = elems_A / THREADS_GEMM;  // 16
            #pragma unroll
            for (int i = 0; i < per_thr; i++) {
                int flat_idx = threadIdx.x * per_thr + i;
                int row = flat_idx / BK;
                int col = flat_idx % BK;
                int global_m = m_start + row;
                int global_k = k * BK + col;
                if (global_m < M && global_k < K)
                    smemA[flat_idx] = A[global_m * K + global_k];
                else
                    smemA[flat_idx] = __float2bfloat16(0.0f);
            }
        }

        // Load B tile: gB[:, :, k] -> sB
        // B is [K, N] row-major. We load B[k*BK:(k+1)*BK, bn*BN:(bn+1)*BN]
        // and store transposed as sB[BN, BK] for MMA (col-major in N)
        {
            int elems_B = BK * BN;  // 2048
            int per_thr = elems_B / THREADS_GEMM;  // 16
            #pragma unroll
            for (int i = 0; i < per_thr; i++) {
                int flat_idx = threadIdx.x * per_thr + i;
                int row = flat_idx / BN;  // K index within tile
                int col = flat_idx % BN;  // N index within tile
                int global_k = k * BK + row;
                int global_n = n_start + col;
                // Store as [BK, BN] row-major = same layout as global
                if (global_k < K && global_n < N)
                    smemB[flat_idx] = B[global_k * N + global_n];
                else
                    smemB[flat_idx] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // Perform MMA on tile
        // Partition smem for MMA
        auto tCsA = thr_mma.partition_A(sA);
        auto tCsB = thr_mma.partition_B(sB);

        cute::gemm(tiled_mma, tCsA, tCsB, tCrC);

        __syncthreads();
    }

    // Store accumulator to global C (fp32)
    // tCrC is partitioned fragment -> write back
    auto tCgC = thr_mma.partition_C(gC(_, _));
    int m_start = bm * BM;
    int n_start = bn * BN;

    // Copy fragment to global with bounds check
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        auto coord = tCgC.get_flat_coord(i);
        int gm = m_start + get<0>(coord);
        int gn = n_start + get<1>(coord);
        if (gm < M && gn < N) {
            C[gm * N + gn] = tCrC(i);
        }
    }
}

// ============================================================
// RMSNorm kernel: per-row norm on fp32 buffer, output bf16
// y[i] = (x[i] * rsqrt(mean(x^2) + eps)) * weight[i]
// ============================================================
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,     // [M, N] fp32
    const float* __restrict__ weight,    // [N] fp32
    __nv_bfloat16* __restrict__ output,  // [M, N] bf16
    int M, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* x = input + row * N;
    __nv_bfloat16* y = output + row * N;

    // Compute sum of squares (block-cooperative)
    extern __shared__ float smem[];
    float sq_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = x[i];
        sq_sum += v * v;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = sq_sum;
    __syncthreads();

    if (warp_id == 0) {
        sq_sum = (lane_id < (blockDim.x / 32)) ? smem[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);
    }
    __syncthreads();

    // Broadcast
    __shared__ float rms_inv;
    if (threadIdx.x == 0) {
        rms_inv = rsqrtf(sq_sum / (float)N + eps);
    }
    __syncthreads();

    // Apply: (weight * x * rms_inv) -> bf16
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        y[i] = __float2bfloat16(weight[i] * x[i] * rms_inv);
    }
}

// ============================================================
// Host launcher: GEMM + RMSNorm for one projection
// ============================================================
static void launch_proj_with_norm(
    const __nv_bfloat16* input,   // [M, K]
    const __nv_bfloat16* weight,  // [K, N] transposed
    const float* norm_weight,     // [N]
    float* gemm_buf,              // [M, N] fp32 temp
    __nv_bfloat16* output,        // [M, N] bf16
    int M, int K, int N,
    float eps,
    cudaStream_t stream
) {
    // Launch GEMM
    dim3 grid_gemm((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block_gemm(THREADS_GEMM);
    gemm_bf16_tc_kernel<<<grid_gemm, block_gemm, 0, stream>>>(
        input, weight, gemm_buf, M, K, N
    );

    // Launch RMSNorm (1 block per row)
    int norm_threads = min(256, N);
    int norm_smem = (norm_threads / 32) * sizeof(float);
    rmsnorm_kernel<<<M, norm_threads, norm_smem, stream>>>(
        gemm_buf, norm_weight, output, M, N, eps
    );
}

// ============================================================
// PyTorch binding
// ============================================================
std::vector<torch::Tensor> phase2_proj_forward(
    torch::Tensor x_normed,         // [num_pos, D] bf16
    torch::Tensor wq_a,             // [N_QA, D] bf16 (original layout)
    torch::Tensor q_a_norm_weight,  // [N_QA] fp32
    torch::Tensor wkv,              // [N_KV, D] bf16 (original layout)
    torch::Tensor kv_norm_weight,   // [N_KV] fp32
    double rms_norm_eps
) {
    TORCH_CHECK(x_normed.is_cuda() && x_normed.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(wq_a.is_cuda() && wq_a.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(wkv.is_cuda() && wkv.scalar_type() == torch::kBFloat16);

    auto x_flat = x_normed.contiguous().view({-1, D_PROJ});
    int M = x_flat.size(0);

    // Transpose weights: [N, K] -> [K, N] for GEMM B operand
    auto wq_a_t = wq_a.contiguous().t().contiguous();
    auto wkv_t = wkv.contiguous().t().contiguous();

    auto opts_bf16 = torch::TensorOptions().device(x_normed.device()).dtype(torch::kBFloat16);
    auto opts_fp32 = torch::TensorOptions().device(x_normed.device()).dtype(torch::kFloat32);

    auto qr = torch::empty({M, N_QA}, opts_bf16);
    auto kv = torch::empty({M, N_KV}, opts_bf16);

    // Temp buffer for GEMM fp32 output (before norm)
    auto gemm_buf_qa = torch::empty({M, N_QA}, opts_fp32);
    auto gemm_buf_kv = torch::empty({M, N_KV}, opts_fp32);

    auto stream = at::cuda::getCurrentCUDAStream();

    // wq_a projection + q_norm
    launch_proj_with_norm(
        reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wq_a_t.data_ptr<at::BFloat16>()),
        q_a_norm_weight.contiguous().data_ptr<float>(),
        gemm_buf_qa.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(qr.data_ptr<at::BFloat16>()),
        M, D_PROJ, N_QA,
        (float)rms_norm_eps, stream
    );

    // wkv projection + kv_norm
    launch_proj_with_norm(
        reinterpret_cast<const __nv_bfloat16*>(x_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(wkv_t.data_ptr<at::BFloat16>()),
        kv_norm_weight.contiguous().data_ptr<float>(),
        gemm_buf_kv.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
        M, D_PROJ, N_KV,
        (float)rms_norm_eps, stream
    );

    return {qr, kv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("phase2_proj_forward", &phase2_proj_forward,
          "Phase 2: wq_a + wkv projections with TC GEMM + RMSNorm (CuTe, bf16)");
}
