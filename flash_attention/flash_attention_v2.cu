#include "../gemm/warp/mma.cuh"
#include "../gemm/block/prelogue.cuh"
#include "../gemm/block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================
// FlashAttention v2
//   一个 block 完成 O 上 [Br, D] 行块的 FlashAttention-2 计算
//   Q: [Br, D] / K, V: [N, D] / O: [Br, D]   (指针已偏移到本 block)
//
//   切分:
//     TPB = WarpCountM * 32, 沿 Br 切 WarpCountM 个 warp
//     每 warp 负责 WarpM = Br/WarpCountM 行 (这里 WarpM=16)
//     KV 在 warp 间共享 (split-Q)
// ============================================================
template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM
>
__device__
void bflash_attention_v2_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    const int32_t N
) {
    constexpr int32_t WarpM = Br / WarpCountM;

    // ---------- 静态 shared memory (Br=Bc=32, D=64 时 ~22KB) ----------
    __shared__ __half smem_Q [Br * D];           // Q tile
    __shared__ __half smem_K [Bc * D];           // K tile
    __shared__ __half smem_V [Bc * D];           // V tile
    __shared__ __half smem_SP[Br * Bc];          // S/P 共用 (softmax 原地覆盖)
    __shared__ float  smem_O [Br * D];           // 在线累加 O (fp32 保精度)
    __shared__ float  smem_m [Br];
    __shared__ float  smem_l [Br];

    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    // 各 warp 在 smem 上的行起点
    __half* warp_Q_smem  = smem_Q  + warp_id * WarpM * D;
    __half* warp_SP_smem = smem_SP + warp_id * WarpM * Bc;
    float*  warp_O_smem  = smem_O  + warp_id * WarpM * D;

    // Load Q gmem → smem (一次)
    block_mma_prelogue_f16<TPB, Br, D>(Q, smem_Q, D, D);

    // 初始化 m=-inf, l=0, O=0
    for (int32_t i = tid; i < Br; i += TPB) {
        smem_m[i] = -INFINITY;
        smem_l[i] = 0.0f;
    }
    for (int32_t i = tid; i < Br * D; i += TPB) {
        smem_O[i] = 0.0f;
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)D);
    const int32_t Tc = N / Bc;

    // 外层循环: 逐段扫 KV
    for (int32_t t = 0; t < Tc; t++) {
        // gmem → smem: K tile
        block_mma_prelogue_f16<TPB, Bc, D>(K + t * Bc * D, smem_K, D, D);
        __syncthreads();

        // S = Q @ K^T   (TN: K row-major 作为 col_major B 加载即 K^T)
        {
            WarpHMMA_f16<WarpM, Bc, D> wmma_qk;
            wmma_qk.zero();
            wmma_qk.forward(warp_Q_smem, smem_K, D, D);
            wmma_qk.stmatrix(warp_SP_smem, Bc);   // 输出 fp16 到 smem_SP
        }
        __syncthreads();

        // Online safe softmax (per row)
        //      每 warp 16 行, lane 0..15 各处理一行 (lane 16..31 空转)
        if (lane_id < WarpM) {
            const int32_t row = warp_id * WarpM + lane_id;
            __half* sp_row = smem_SP + row * Bc;        // 读 S, 写 P (同 buffer)
            float*  o_row  = smem_O  + row * D;

            const float m_old = smem_m[row];
            const float l_old = smem_l[row];

            // (a) 本 tile 行最大
            float m_tile = -INFINITY;
            #pragma unroll
            for (int32_t j = 0; j < Bc; j++) {
                const float v = __half2float(sp_row[j]) * scale;
                if (v > m_tile) m_tile = v;
            }
            const float m_new = fmaxf(m_old, m_tile);

            // (b) P = exp(S*scale - m_new), 同时累加行和
            float l_tile = 0.0f;
            #pragma unroll
            for (int32_t j = 0; j < Bc; j++) {
                const float p = expf(__half2float(sp_row[j]) * scale - m_new);
                sp_row[j] = __float2half(p);
                l_tile += p;
            }

            // (c) α = exp(m_old - m_new), 修旧 O
            const float alpha = expf(m_old - m_new);
            #pragma unroll
            for (int32_t k = 0; k < D; k++) {
                o_row[k] *= alpha;
            }

            // (d) 更新 m, l
            smem_m[row] = m_new;
            smem_l[row] = alpha * l_old + l_tile;
        }
        __syncthreads();

        // gmem → smem: V tile
        block_mma_prelogue_f16<TPB, Bc, D>(V + t * Bc * D, smem_V, D, D);
        __syncthreads();

        // O += P @ V   (NN: P, V 都 row_major)
        {
            WarpHMMA_NN_f16<WarpM, D, Bc> wmma_pv;
            wmma_pv.load_C(warp_O_smem, D);                  // 装入已 rescale 的 O
            wmma_pv.forward(warp_SP_smem, smem_V, Bc, D);    // 累加 P @ V
            wmma_pv.store_C_f32(warp_O_smem, D);             // 存回 fp32 smem
        }
        __syncthreads();
    }

    // 归一化 1/l
    if (lane_id < WarpM) {
        const int32_t row = warp_id * WarpM + lane_id;
        float* o_row = smem_O + row * D;
        const float inv_l = 1.0f / smem_l[row];
        #pragma unroll
        for (int32_t k = 0; k < D; k++) {
            o_row[k] *= inv_l;
        }
    }
    __syncthreads();

    // smem_O (fp32, 连续 [Br, D]) → O gmem (fp16, stride=D)
    block_mma_eplogue_f32_f16<TPB, Br, D>(smem_O, O, D);
}

template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM
>
__global__
void device_flash_attention_v2_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int32_t N
) {
    const int32_t q_tile_id = blockIdx.x;
    const int32_t bh_offset = blockIdx.y * N * D;

    bflash_attention_v2_f16<TPB, Br, Bc, D, WarpCountM>(
        Q + bh_offset + q_tile_id * Br * D,
        K + bh_offset,
        V + bh_offset,
        O + bh_offset + q_tile_id * Br * D,
        N
    );
}

// ============================================================
// Host launcher
//   Q, K, V, O: [B, H, N, D] row-major fp16 (device 指针)
//   仅支持 D=64; N 必须是 Br 与 Bc 的整数倍
// ============================================================
void flash_attention_v2_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int32_t B, int32_t H, int32_t N, int32_t D
) {
    constexpr int32_t Br = 32;
    constexpr int32_t Bc = 64;
    constexpr int32_t WarpCountM = 2;   // WarpM = Br/WarpCountM = 16
    constexpr int32_t TPB = WarpCountM * 32;
    constexpr int32_t kD = 64;

    if (D != kD) {
        fprintf(stderr, "flash_attention_v2_f16: only D=%d supported (got %d)\n", kD, D);
        return;
    }
    if (N % Br != 0 || N % Bc != 0) {
        fprintf(stderr, "flash_attention_v2_f16: N=%d must be multiple of Br=%d and Bc=%d\n",
                N, Br, Bc);
        return;
    }

    const dim3 grid{(unsigned)(N / Br), (unsigned)(B * H), 1};
    device_flash_attention_v2_f16<TPB, Br, Bc, kD, WarpCountM>
        <<<grid, TPB>>>(Q, K, V, O, N);
}
