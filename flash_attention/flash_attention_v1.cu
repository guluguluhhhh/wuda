#include "../gemm/warp/mma.cuh"
#include "../gemm/block/prelogue.cuh"
#include "../gemm/block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================
// FlashAttention v1
//   按论文 Algorithm 1 写:
//     - grid = (B*H,)                  一个 block 负责一个 (b, h)
//     - 外层循环 KV (Tc 段), 内层循环 Q (Tr 段)
//     - O, m, l 全程在 HBM, 每个内层 iter 从 HBM 读 → smem 算 → 写回 HBM
//     - O 在每步都做 (1/l_new) 归一化, 末尾不再除
//
//   公式 (P 用 m_new 形式, 等价于 paper 形式):
//     m_new   = max(m_old, m_tile)
//     P       = exp(S*scale - m_new)
//     l_tile  = rowsum(P)
//     α       = exp(m_old - m_new)
//     l_new   = α * l_old + l_tile
//     O_new   = (α * l_old / l_new) * O_old + (1/l_new) * (P @ V)
// ============================================================

// ------------------------------------------------------------
// 一个 block 的 FA1 主体: 处理一个 (b, h)
// ------------------------------------------------------------
template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM
>
__device__
void bflash_attention_v1_f16(
    const __half* Q,       // [N, D]
    const __half* K,       // [N, D]
    const __half* V,       // [N, D]
    __half*       O,       // [N, D]  in/out, 初始 0
    float*        m_state, // [N]     in/out, 初始 -inf
    float*        l_state, // [N]     in/out, 初始 0
    const int32_t N
) {
    constexpr int32_t WarpM = Br / WarpCountM;

    // ---------- 静态 shared memory (Br=Bc=32, D=64 时 ~22KB) ----------
    __shared__ __half smem_Q [Br * D];     // 内层 iter 换
    __shared__ __half smem_K [Bc * D];     // 外层 iter 换
    __shared__ __half smem_V [Bc * D];     // 外层 iter 换
    __shared__ __half smem_SP[Br * Bc];    // S/P 共用
    __shared__ float  smem_O [Br * D];     // 当前 Q 段 O, R-M-W HBM
    __shared__ float  smem_m [Br];
    __shared__ float  smem_l [Br];

    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    // 各 warp 在 smem 上的行起点
    __half* warp_Q_smem  = smem_Q  + warp_id * WarpM * D;
    __half* warp_SP_smem = smem_SP + warp_id * WarpM * Bc;
    float*  warp_O_smem  = smem_O  + warp_id * WarpM * D;

    const float scale = 1.0f / sqrtf((float)D);
    const int32_t Tc = N / Bc;
    const int32_t Tr = N / Br;

    // ============ 外层: KV tile ============
    for (int32_t j = 0; j < Tc; j++) {
        // 加载 K_j, V_j (整个内层循环复用)
        block_mma_prelogue_f16<TPB, Bc, D>(K + j * Bc * D, smem_K, D, D);
        block_mma_prelogue_f16<TPB, Bc, D>(V + j * Bc * D, smem_V, D, D);
        __syncthreads();

        // ============ 内层: Q tile ============
        for (int32_t i = 0; i < Tr; i++) {
            // ---- 1) 从 HBM 加载 Q_i, O_i, m_i, l_i ----
            block_mma_prelogue_f16<TPB, Br, D>(Q + i * Br * D, smem_Q, D, D);
            block_mma_prelogue_f16_f32<TPB, Br, D>(O + i * Br * D, smem_O, D);
            // m, l 数量很少 (Br=32), 前 Br 个线程各取一个
            if (tid < Br) {
                smem_m[tid] = m_state[i * Br + tid];
                smem_l[tid] = l_state[i * Br + tid];
            }
            __syncthreads();

            // ---- 2) S = Q_i @ K_j^T ----
            {
                WarpHMMA_f16<WarpM, Bc, D> wmma_qk;
                wmma_qk.zero();
                wmma_qk.forward(warp_Q_smem, smem_K, D, D);
                wmma_qk.stmatrix(warp_SP_smem, Bc);
            }
            __syncthreads();

            // ---- 3) Softmax + FA1 在线归一化 (per row) ----
            if (lane_id < WarpM) {
                const int32_t row = warp_id * WarpM + lane_id;
                __half* sp_row = smem_SP + row * Bc;
                float*  o_row  = smem_O  + row * D;

                const float m_old = smem_m[row];
                const float l_old = smem_l[row];

                // (a) 本 tile 行最大
                float m_tile = -INFINITY;
                #pragma unroll
                for (int32_t jj = 0; jj < Bc; jj++) {
                    const float v = __half2float(sp_row[jj]) * scale;
                    if (v > m_tile) m_tile = v;
                }
                const float m_new = fmaxf(m_old, m_tile);

                // (b) P = exp(S*scale - m_new), 累加 l_tile
                float l_tile = 0.0f;
                #pragma unroll
                for (int32_t jj = 0; jj < Bc; jj++) {
                    const float p = expf(__half2float(sp_row[jj]) * scale - m_new);
                    sp_row[jj] = __float2half(p);
                    l_tile += p;
                }

                // (c) FA1 关键: 每步都把 O 归一化
                const float alpha     = expf(m_old - m_new);
                const float l_new     = alpha * l_old + l_tile;
                const float inv_l_new = 1.0f / l_new;
                const float fac_old   = alpha * l_old * inv_l_new;   // = α*l_old/l_new

                // O_old 已 rescale 成: O_old *= (α*l_old/l_new)
                #pragma unroll
                for (int32_t k = 0; k < D; k++) {
                    o_row[k] *= fac_old;
                }

                // P 预乘 (1/l_new), 这样后面的 MMA 直接给出 (1/l_new)·(P@V)
                #pragma unroll
                for (int32_t jj = 0; jj < Bc; jj++) {
                    const float p_scaled = __half2float(sp_row[jj]) * inv_l_new;
                    sp_row[jj] = __float2half(p_scaled);
                }

                // 更新 m, l
                smem_m[row] = m_new;
                smem_l[row] = l_new;
            }
            __syncthreads();

            // ---- 4) O_i += P @ V_j  (P 已 pre-scaled by 1/l_new) ----
            {
                WarpHMMA_NN_f16<WarpM, D, Bc> wmma_pv;
                wmma_pv.load_C(warp_O_smem, D);                  // 装入已 rescale 的 O
                wmma_pv.forward(warp_SP_smem, smem_V, Bc, D);    // 累加 (1/l_new)·(P@V)
                wmma_pv.store_C_f32(warp_O_smem, D);
            }
            __syncthreads();

            // ---- 5) 写回 HBM: O_i, m_i, l_i ----
            block_mma_eplogue_f32_f16<TPB, Br, D>(smem_O, O + i * Br * D, D);
            if (tid < Br) {
                m_state[i * Br + tid] = smem_m[tid];
                l_state[i * Br + tid] = smem_l[tid];
            }
            __syncthreads();
        } // end inner Q loop
    } // end outer KV loop
}

template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM
>
__global__
void device_flash_attention_v1_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       O,
    float*        m_state,
    float*        l_state,
    int32_t       N
) {
    const int32_t bh_id     = blockIdx.x;
    const int32_t bh_offset = bh_id * N * D;
    const int32_t ml_offset = bh_id * N;

    bflash_attention_v1_f16<TPB, Br, Bc, D, WarpCountM>(
        Q + bh_offset,
        K + bh_offset,
        V + bh_offset,
        O + bh_offset,
        m_state + ml_offset,
        l_state + ml_offset,
        N
    );
}

// ============================================================
// 初始化 kernel: O=0 用 cudaMemset, m, l 需 kernel (cudaMemset 设 -inf)
// ============================================================
__global__
void init_m_l_kernel(float* m, float* l, int32_t total) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        m[tid] = -INFINITY;
        l[tid] = 0.0f;
    }
}

// ============================================================
// Host launcher
//   Q, K, V, O: [B, H, N, D] row-major fp16 (device)
//   内部分配 m, l 两段 HBM (B*H*N fp32 each), 调用完释放
// ============================================================
void flash_attention_v1_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       O,
    int32_t B, int32_t H, int32_t N, int32_t D
) {
    constexpr int32_t Br = 32;
    constexpr int32_t Bc = 32;
    constexpr int32_t WarpCountM = 2;
    constexpr int32_t TPB = WarpCountM * 32;
    constexpr int32_t kD = 128;

    if (D != kD) {
        fprintf(stderr, "flash_attention_v1_f16: only D=%d supported (got %d)\n", kD, D);
        return;
    }
    if (N % Br != 0 || N % Bc != 0) {
        fprintf(stderr, "flash_attention_v1_f16: N=%d must be multiple of Br=%d and Bc=%d\n",
                N, Br, Bc);
        return;
    }

    const size_t bh_total = (size_t)B * H;
    const size_t ml_total = bh_total * N;

    // 1) 分配 HBM 状态: m, l
    float *d_m, *d_l;
    cudaMalloc(&d_m, ml_total * sizeof(float));
    cudaMalloc(&d_l, ml_total * sizeof(float));

    // 2) 初始化: O=0 (fp16 0 即 0x0000, memset OK), m=-inf, l=0
    cudaMemset(O, 0, bh_total * N * D * sizeof(__half));
    {
        const int32_t block = 256;
        const int32_t grid_init = (ml_total + block - 1) / block;
        init_m_l_kernel<<<grid_init, block>>>(d_m, d_l, (int32_t)ml_total);
    }

    // 3) 主 kernel: grid = (B*H,)
    const dim3 grid{(unsigned)bh_total, 1, 1};
    device_flash_attention_v1_f16<TPB, Br, Bc, kD, WarpCountM>
        <<<grid, TPB>>>(Q, K, V, O, d_m, d_l, N);

    // 4) 释放临时状态
    cudaFree(d_m);
    cudaFree(d_l);
}
