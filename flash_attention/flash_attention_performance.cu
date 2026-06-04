#include "../gemm/warp/mma_ptx.cuh"
#include "../gemm/block/prelogue.cuh"
#include "../gemm/block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================
// 在 frag_C 寄存器上 rescale O (O 不落 smem)
//   frag_C[m][n][i] 布局 (per lane, m16n16 双 m16n8 拼接):
//     [0,2] → 上半行 (rows row0 = lane>>2, range 0..7)
//     [1,3] → 下半行 (rows row0+8,         range 8..15)
//   每 lane 只需 2 个 factor (上/下半行各一个)
// ============================================================

// half2 *= half2 (packed in uint32)
__device__ __forceinline__
uint32_t hmul2_pack(uint32_t v, __half2 a) {
    __half2 h = *reinterpret_cast<__half2*>(&v);
    h = __hmul2(h, a);
    return *reinterpret_cast<uint32_t*>(&h);
}

template<int32_t WarpTileM, int32_t WarpTileN>
__device__ __forceinline__
void rescale_frag_O(
    uint32_t (&frag_C)[WarpTileM][WarpTileN][4],
    const __half2 a_upper,
    const __half2 a_lower
) {
    static_assert(WarpTileM == 1, "rescale_frag_O: 当前只覆盖 WarpTileM == 1");
    #pragma unroll
    for (int32_t n = 0; n < WarpTileN; n++) {
        frag_C[0][n][0] = hmul2_pack(frag_C[0][n][0], a_upper);
        frag_C[0][n][1] = hmul2_pack(frag_C[0][n][1], a_lower);
        frag_C[0][n][2] = hmul2_pack(frag_C[0][n][2], a_upper);
        frag_C[0][n][3] = hmul2_pack(frag_C[0][n][3], a_lower);
    }
}

// ============================================================
// Warp 内一行的在线 softmax (不动 O, O 在累加器寄存器里由外面 rescale)
//   全 32 lane 协作处理一行, 每 lane 各拿 Bc/32 个 P
//   m, l 入参 m_old/l_old, 返回 m_new/l_new (所有 lane 同值)
//   函数返回 α = exp(m_old - m_new), 供外面写入 smem_alpha
// ============================================================
template<int32_t Bc>
__device__ __forceinline__
float warp_online_softmax_row(
    __half* __restrict__ sp_row,   // [Bc] in: S,  out: P
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

    const float alpha = expf(m_old - m_new);
    m = m_new;
    l = alpha * l_old + l_tile;
    return alpha;
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
    // smem_Q 已删: Q 只在初始化时借用 smem_KV staging, 之后一直在 frag_Q 寄存器
    // smem_O 已删: O 一直留在 mma_pv.frag_C 累加器寄存器里
    // smem_KV: Q 初始化 / K tile / V tile 三者依次复用 (生命周期不重叠)
    // smem_SP: tile 内 S/P, 末尾再复用为最终 O 写回 gmem 的 staging
    __shared__ __half smem_KV   [Bc * D];        // Q init / K / V
    __shared__ __half smem_SP   [Br * Bc];       // S/P + 末尾 O staging
    __shared__ float  smem_m    [Br];
    __shared__ float  smem_l    [Br];
    __shared__ float  smem_alpha[Br];            // 每行 α, 给 frag_C rescale 用

    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    // 各 warp 在 smem 上的行起点 (Q 借 smem_KV 的前 Br*D 段)
    __half* warp_Q_smem  = smem_KV + warp_id * WarpM * D;
    __half* warp_SP_smem = smem_SP + warp_id * WarpM * Bc;

    // 持久的 O 累加器 (frag_C 跨 t 循环保留, frag_A/B 是 forward 内的临时)
    ptx::WarpHMMA_Trans_f16<WarpM, D, Bc> mma_pv;
    mma_pv.zero();

    // Load Q gmem → smem_KV 前段 (一次性)
    block_mma_prelogue_f16<TPB, Br, D>(Q, smem_KV, D, D);

    // 初始化 m=-inf, l=0
    for (int32_t i = tid; i < Br; i += TPB) {
        smem_m[i] = -INFINITY;
        smem_l[i] = 0.0f;
    }
    __syncthreads();

    // Q 一次性 ldmatrix 到 per-lane 持久寄存器, 后续 smem_KV 就可以被 K/V 覆写
    using QK_mma_t = ptx::WarpHMMA_f16<WarpM, Bc, D>;
    typename QK_mma_t::FragAFull frag_Q;
    QK_mma_t::load_full_A(warp_Q_smem, D, frag_Q);
    __syncthreads();   // 等所有 warp 把自己的 Q 读完, 才能让 K 覆写 smem_KV

    const float scale = 1.0f / sqrtf((float)D);
    const int32_t Tc = N / Bc;

    // 外层循环: 逐段扫 KV
    for (int32_t t = 0; t < Tc; t++) {
        // gmem → smem: K tile
        block_mma_prelogue_f16<TPB, Bc, D>(K + t * Bc * D, smem_KV, D, D);
        __syncthreads();

        // S = Q @ K^T   (TN: K row-major 作为 col_major B 加载即 K^T)
        //   Q 在 frag_Q 寄存器里, mma 只 load K, 不再读 smem_Q
        {
            QK_mma_t mma_qk;
            mma_qk.zero();
            mma_qk.forward_with_A(frag_Q, smem_KV, D);
            mma_qk.stmatrix(warp_SP_smem, Bc);   // 输出 fp16 到 smem_SP
        }
        __syncthreads();

        // Online safe softmax (per row) — 收 α 不动 O
        #pragma unroll
        for (int32_t r = 0; r < WarpM; r++) {
            const int32_t row = warp_id * WarpM + r;
            float m = smem_m[row];
            float l = smem_l[row];
            const float alpha = warp_online_softmax_row<Bc>(
                smem_SP + row * Bc, scale, lane_id, m, l
            );
            if (lane_id == 0) {
                smem_m[row] = m;
                smem_l[row] = l;
                smem_alpha[row] = alpha;
            }
        }
        __syncwarp();   // 等本 warp 的 lane 0 把 α 写齐

        // 按行 α rescale 累加器寄存器里的 O (lane 各持有 2 行: 上半/下半)
        {
            const int32_t r_upper = warp_id * WarpM + (lane_id >> 2);
            const int32_t r_lower = r_upper + 8;
            const __half2 a_upper = __float2half2_rn(smem_alpha[r_upper]);
            const __half2 a_lower = __float2half2_rn(smem_alpha[r_lower]);
            rescale_frag_O(mma_pv.frag_C, a_upper, a_lower);
        }
        __syncthreads();

        // gmem → smem: V tile (复用 K 的 buffer)
        block_mma_prelogue_f16<TPB, Bc, D>(V + t * Bc * D, smem_KV, D, D);
        __syncthreads();

        // O += P @ V   (NN: P, V 都 row_major) — frag_C 持久累加, 无 smem 往返
        mma_pv.forward(warp_SP_smem, smem_KV, Bc, D);
        __syncthreads();
    }

    // 归一化 1/l — 直接在 frag_C 上做, 同样上/下半行各一个 factor
    {
        const int32_t r_upper = warp_id * WarpM + (lane_id >> 2);
        const int32_t r_lower = r_upper + 8;
        const __half2 inv_upper = __float2half2_rn(1.0f / smem_l[r_upper]);
        const __half2 inv_lower = __float2half2_rn(1.0f / smem_l[r_lower]);
        rescale_frag_O(mma_pv.frag_C, inv_upper, inv_lower);
    }

    // frag_C → smem (复用 smem_SP, stride=D, 每 warp 写 WarpM 行连续段) → gmem
    __half* warp_O_smem_out = smem_SP + warp_id * WarpM * D;
    mma_pv.stmatrix(warp_O_smem_out, D);
    __syncthreads();

    block_mma_eplogue_f16<TPB, Br, D>(smem_SP, O, D, D);
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
