#include "../gemm/warp/mma_ptx.cuh"
#include "../gemm/block/prelogue.cuh"
#include "../gemm/block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================
// Warp内一行的在线 softmax + O rescale
//   全 32 lane 协作处理一行, 每 lane 各拿 Bc/32 个 P + D/32 个 O
//   m, l 入参为 m_old/l_old, 返回时所有 lane 持有 m_new/l_new
// ============================================================
template<int32_t Bc, int32_t D>
__device__ __forceinline__
void warp_online_softmax_row(
    __half* __restrict__ sp_row,   // [Bc] in: S,  out: P
    __half* __restrict__ o_row,    // [D]  in-place rescale by α
    const float scale,
    const int32_t lane_id,
    float& m,                      // in: m_old, out: m_new
    float& l                       // in: l_old, out: l_new
) {
    constexpr int32_t BcPerLane = Bc / 32;
    static_assert(Bc % 32 == 0, "Bc must be multiple of 32");

    const float m_old = m;
    const float l_old = l;

    // (a) 行最大 — 暂存缩放后的值, 避免 (b) 里再 load 一次
    float scaled[BcPerLane];
    float m_tile = -INFINITY;
    #pragma unroll
    for (int32_t i = 0; i < BcPerLane; i++) {
        scaled[i] = __half2float(sp_row[i * 32 + lane_id]) * scale;
        m_tile = fmaxf(m_tile, scaled[i]);
    }
    #pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));
    }
    const float m_new = fmaxf(m_old, m_tile);

    // (b) P = exp(scaled - m_new), 行和
    float l_tile = 0.0f;
    #pragma unroll
    for (int32_t i = 0; i < BcPerLane; i++) {
        const float p = expf(scaled[i] - m_new);
        sp_row[i * 32 + lane_id] = __float2half(p);
        l_tile += p;
    }
    #pragma unroll
    for (int32_t off = 16; off > 0; off >>= 1) {
        l_tile += __shfl_xor_sync(0xffffffff, l_tile, off);
    }

    // (c) α = exp(m_old - m_new), rescale O
    const float alpha = expf(m_old - m_new);
    warp_rescale_row<D>(o_row, alpha, lane_id);

    m = m_new;
    l = alpha * l_old + l_tile;
}

// 全 warp 协作: 一行 O 标量 rescale
template<int32_t D>
__device__ __forceinline__
void warp_rescale_row(__half* o_row, const float factor, const int32_t lane_id) {
    constexpr int32_t DPerLane = D / 32;
    static_assert(D % 32 == 0, "D must be multiple of 32");
    #pragma unroll
    for (int32_t i = 0; i < DPerLane; i++) {
        const int32_t k = i * 32 + lane_id;
        o_row[k] = __float2half(__half2float(o_row[k]) * factor);
    }
}

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
void bflash_attention_perf_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    const int32_t N
) {
    constexpr int32_t WarpM = Br / WarpCountM;

    // ---------- 静态 shared memory ----------
    // K, V 生命周期不重叠 (K 用完才 load V), 合用一块 buffer
    __shared__ __half smem_Q [Br * D];           // Q tile
    __shared__ __half smem_KV[Bc * D];           // K / V 共用
    __shared__ __half smem_SP[Br * Bc];          // S/P 共用 (softmax 原地覆盖)
    __shared__ __half smem_O [Br * D];           // 在线累加 O (fp16, mma PTX)
    __shared__ float  smem_m [Br];
    __shared__ float  smem_l [Br];

    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    // 各 warp 在 smem 上的行起点
    __half* warp_Q_smem  = smem_Q  + warp_id * WarpM * D;
    __half* warp_SP_smem = smem_SP + warp_id * WarpM * Bc;
    __half* warp_O_smem  = smem_O  + warp_id * WarpM * D;

    // Load Q gmem → smem (一次)
    block_mma_prelogue_f16<TPB, Br, D>(Q, smem_Q, D, D);

    // 初始化 m=-inf, l=0, O=0
    for (int32_t i = tid; i < Br; i += TPB) {
        smem_m[i] = -INFINITY;
        smem_l[i] = 0.0f;
    }
    for (int32_t i = tid; i < Br * D; i += TPB) {
        smem_O[i] = __float2half(0.0f);
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)D);
    const int32_t Tc = N / Bc;

    // 外层循环: 逐段扫 KV
    for (int32_t t = 0; t < Tc; t++) {
        // gmem → smem: K tile
        block_mma_prelogue_f16<TPB, Bc, D>(K + t * Bc * D, smem_KV, D, D);
        __syncthreads();

        // S = Q @ K^T   (TN: K row-major 作为 col_major B 加载即 K^T)
        {
            ptx::WarpHMMA_f16<WarpM, Bc, D> mma_qk;
            mma_qk.zero();
            mma_qk.forward(warp_Q_smem, smem_KV, D, D);
            mma_qk.stmatrix(warp_SP_smem, Bc);   // 输出 fp16 到 smem_SP
        }
        __syncthreads();

        // Online safe softmax (per row) — 全 warp 协作每行
        #pragma unroll
        for (int32_t r = 0; r < WarpM; r++) {
            const int32_t row = warp_id * WarpM + r;
            float m = smem_m[row];
            float l = smem_l[row];
            warp_online_softmax_row<Bc, D>(
                smem_SP + row * Bc,
                smem_O  + row * D,
                scale, lane_id, m, l
            );
            if (lane_id == 0) {
                smem_m[row] = m;
                smem_l[row] = l;
            }
        }
        __syncthreads();

        // gmem → smem: V tile (复用 K 的 buffer)
        block_mma_prelogue_f16<TPB, Bc, D>(V + t * Bc * D, smem_KV, D, D);
        __syncthreads();

        // O += P @ V   (NN: P, V 都 row_major)
        {
            ptx::WarpHMMA_Trans_f16<WarpM, D, Bc> mma_pv;
            mma_pv.load_C(warp_O_smem, D);                  // 装入已 rescale 的 O (fp16)
            mma_pv.forward(warp_SP_smem, smem_KV, Bc, D);   // 累加 P @ V
            mma_pv.stmatrix(warp_O_smem, D);                // 存回 fp16 smem
        }
        __syncthreads();
    }

    // 归一化 1/l — 全 warp 协作每行
    #pragma unroll
    for (int32_t r = 0; r < WarpM; r++) {
        const int32_t row = warp_id * WarpM + r;
        warp_rescale_row<D>(smem_O + row * D, 1.0f / smem_l[row], lane_id);
    }
    __syncthreads();

    // smem_O (fp16, 连续 [Br, D]) → O gmem (fp16, stride=D)
    block_mma_eplogue_f16<TPB, Br, D>(smem_O, O, D, D);
}

template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM
>
__global__
void device_flash_attention_perf_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int32_t N
) {
    const int32_t q_tile_id = blockIdx.x;
    const int32_t bh_offset = blockIdx.y * N * D;

    bflash_attention_perf_f16<TPB, Br, Bc, D, WarpCountM>(
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
void flash_attention_perf_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int32_t B, int32_t H, int32_t N, int32_t D
) {
    constexpr int32_t Br = 32;
    constexpr int32_t Bc = 128;
    constexpr int32_t WarpCountM = 2;   // WarpM = Br/WarpCountM = 16
    constexpr int32_t TPB = WarpCountM * 32;
    constexpr int32_t kD = 64;

    if (D != kD) {
        fprintf(stderr, "flash_attention_perf_f16: only D=%d supported (got %d)\n", kD, D);
        return;
    }
    if (N % Br != 0 || N % Bc != 0) {
        fprintf(stderr, "flash_attention_perf_f16: N=%d must be multiple of Br=%d and Bc=%d\n",
                N, Br, Bc);
        return;
    }

    const dim3 grid{(unsigned)(N / Br), (unsigned)(B * H), 1};
    device_flash_attention_perf_f16<TPB, Br, Bc, kD, WarpCountM>
        <<<grid, TPB>>>(Q, K, V, O, N);
}
