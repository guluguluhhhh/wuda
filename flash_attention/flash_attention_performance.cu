#include "../gemm/warp/mma_ptx.cuh"
#include "../gemm/block/prelogue.cuh"
#include "../gemm/block/eplogue.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

__device__ __forceinline__
uint32_t hmul2_pack(uint32_t v, __half2 a) {
    __half2 h = *reinterpret_cast<__half2*>(&v);
    h = __hmul2(h, a);
    return *reinterpret_cast<uint32_t*>(&h);
}

// frag_C[m][n][i] per-lane 布局 (m16n16 = 双 m16n8 拼接):
//   [0,2] → 上半行 (row0 = lane>>2, 范围 0..7)
//   [1,3] → 下半行 (row0 + 8,       范围 8..15)
// 每 lane 只需 2 个 factor (上/下半行各一个)
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

// Warp 内一行的在线 softmax: 32 lane 协作一行, 不动 O
// 返回 α = exp(m_old - m_new), 由外部用来 rescale frag_C
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

    // 行最大 — 缓存 scaled, 避免下一遍再读 smem
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

// FlashAttention-2: 一个 block 算 O 的 [Br, D] 行块
//   Q: [Br, D] / K,V: [N, D] / O: [Br, D]  (指针已偏移到本 block)
//   沿 Br 切 WarpCountM 个 warp (split-Q), KV 在 warp 间共享
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

    // Q 常驻 frag_Q 寄存器, O 常驻 mma_pv.frag_C — 都不落 smem
    // smem_KV: Q 初始化 / K tile / V tile 依次复用 (生命周期不重叠)
    // smem_SP: tile 内 S/P, 末尾复用为 O 写回的 staging
    __shared__ __half smem_KV   [Bc * D];
    __shared__ __half smem_SP   [Br * Bc];
    __shared__ float  smem_m    [Br];
    __shared__ float  smem_l    [Br];
    __shared__ float  smem_alpha[Br];

    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    // Q 借用 smem_KV 前 Br*D 段做 staging
    __half* warp_Q_smem  = smem_KV + warp_id * WarpM * D;
    __half* warp_SP_smem = smem_SP + warp_id * WarpM * Bc;

    // 持久的 O 累加器, 跨 t 循环保留
    ptx::WarpHMMA_Trans_f16<WarpM, D, Bc> mma_pv;
    mma_pv.zero();

    block_mma_prelogue_f16<TPB, Br, D>(Q, smem_KV, D, D);

    for (int32_t i = tid; i < Br; i += TPB) {
        smem_m[i] = -INFINITY;
        smem_l[i] = 0.0f;
    }
    __syncthreads();

    // Q ldmatrix 进 per-lane 寄存器, 之后 smem_KV 就可以被 K/V 覆写
    using QK_mma_t = ptx::WarpHMMA_f16<WarpM, Bc, D>;
    typename QK_mma_t::FragAFull frag_Q;
    QK_mma_t::load_full_A(warp_Q_smem, D, frag_Q);
    __syncthreads();   // 等所有 warp 读完 Q, 才能让 K 覆写 smem_KV

    const float scale = 1.0f / sqrtf((float)D);
    const int32_t Tc = N / Bc;

    for (int32_t t = 0; t < Tc; t++) {
        block_mma_prelogue_f16<TPB, Bc, D>(K + t * Bc * D, smem_KV, D, D);
        __syncthreads();

        // S = Q @ K^T   (TN: K row-major 当 col_major B 加载即 K^T)
        {
            QK_mma_t mma_qk;
            mma_qk.zero();
            mma_qk.forward_with_A(frag_Q, smem_KV, D);
            mma_qk.stmatrix(warp_SP_smem, Bc);
        }
        __syncthreads();

        // 在线 softmax: 每行算 α, 不动 O
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

        // 按行 α rescale frag_C (每 lane 持上/下半各一行)
        {
            const int32_t r_upper = warp_id * WarpM + (lane_id >> 2);
            const int32_t r_lower = r_upper + 8;
            const __half2 a_upper = __float2half2_rn(smem_alpha[r_upper]);
            const __half2 a_lower = __float2half2_rn(smem_alpha[r_lower]);
            rescale_frag_O(mma_pv.frag_C, a_upper, a_lower);
        }
        __syncthreads();

        // V tile 复用 K 的 smem buffer
        block_mma_prelogue_f16<TPB, Bc, D>(V + t * Bc * D, smem_KV, D, D);
        __syncthreads();

        // O += P @ V, frag_C 持久累加
        mma_pv.forward(warp_SP_smem, smem_KV, Bc, D);
        __syncthreads();
    }

    // 1/l 归一化, 同样在 frag_C 上做
    {
        const int32_t r_upper = warp_id * WarpM + (lane_id >> 2);
        const int32_t r_lower = r_upper + 8;
        const __half2 inv_upper = __float2half2_rn(1.0f / smem_l[r_upper]);
        const __half2 inv_lower = __float2half2_rn(1.0f / smem_l[r_lower]);
        rescale_frag_O(mma_pv.frag_C, inv_upper, inv_lower);
    }

    // frag_C → smem_SP staging → gmem
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

// Q, K, V, O: [B, H, N, D] row-major fp16 (device 指针)
// 仅支持 D=64; N 必须是 Br 与 Bc 的整数倍
void flash_attention_perf_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int32_t B, int32_t H, int32_t N, int32_t D
) {
    constexpr int32_t Br = 64;
    constexpr int32_t Bc = 128;
    constexpr int32_t WarpCountM = 4;   // WarpM = Br/WarpCountM = 16
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
