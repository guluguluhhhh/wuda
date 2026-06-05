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

// 泛化在线 softmax: 一个 warp 同时算 32/kLanesPerRow 行
//   Bc=128 → 16 lane/row, 2 rows/warp
//   Bc=64  →  8 lane/row, 4 rows/warp
// 每 lane 始终用 LDS.128 / STS.128 吃 8 fp16, reduction 范围 = kLanesPerRow
template<int32_t Bc>
__device__ __forceinline__
float warp_online_softmax(
    __half* __restrict__ sp_row_base,   // 起始指向 row_base, 下一行 = +sp_stride
    const int32_t sp_stride,            // smem 行 stride (= Bc + PAD)
    const float scale,
    const int32_t lane_id,
    float& m,                           // in: m_old (本 TG 行), out: m_new
    float& l                            // in: l_old (本 TG 行), out: l_new
) {
    constexpr int32_t kElemsPerLane = 8;
    constexpr int32_t kLanesPerRow  = Bc / kElemsPerLane;
    constexpr int32_t kRowsPerWarp  = 32 / kLanesPerRow;
    static_assert(Bc % kElemsPerLane == 0, "Bc must be multiple of 8");
    static_assert(32 % kLanesPerRow == 0,  "kLanesPerRow must divide 32");

    const int32_t tg_id   = lane_id / kLanesPerRow;   // 0..kRowsPerWarp-1
    const int32_t tg_lane = lane_id % kLanesPerRow;   // 0..kLanesPerRow-1

    constexpr float LOG2E = 1.4426950408889634f;
    const float scale_log2e = scale * LOG2E;

    __half* my_row = sp_row_base + tg_id * sp_stride;

    // LDS.128 — 每 lane 8 fp16
    uint4 packed = *reinterpret_cast<const uint4*>(my_row + tg_lane * kElemsPerLane);
    float2 f01 = __half22float2(*reinterpret_cast<__half2*>(&packed.x));
    float2 f23 = __half22float2(*reinterpret_cast<__half2*>(&packed.y));
    float2 f45 = __half22float2(*reinterpret_cast<__half2*>(&packed.z));
    float2 f67 = __half22float2(*reinterpret_cast<__half2*>(&packed.w));

    float scaled[8] = {
        f01.x * scale_log2e, f01.y * scale_log2e,
        f23.x * scale_log2e, f23.y * scale_log2e,
        f45.x * scale_log2e, f45.y * scale_log2e,
        f67.x * scale_log2e, f67.y * scale_log2e,
    };

    // 树形 local max
    float m_tile = fmaxf(
        fmaxf(fmaxf(scaled[0], scaled[1]), fmaxf(scaled[2], scaled[3])),
        fmaxf(fmaxf(scaled[4], scaled[5]), fmaxf(scaled[6], scaled[7]))
    );

    // kLanesPerRow-lane butterfly (off ≤ kLanesPerRow/2 → TG 自治)
    #pragma unroll
    for (int32_t off = kLanesPerRow / 2; off > 0; off >>= 1) {
        m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, off));
    }

    const float m_old = m;
    const float l_old = l;
    const float m_new = fmaxf(m_old, m_tile);

    float p[8];
    #pragma unroll
    for (int32_t i = 0; i < 8; i++) {
        p[i] = exp2f(scaled[i] - m_new);
    }

    float l_tile = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));

    // STS.128
    {
        __half2 hp0 = __floats2half2_rn(p[0], p[1]);
        __half2 hp1 = __floats2half2_rn(p[2], p[3]);
        __half2 hp2 = __floats2half2_rn(p[4], p[5]);
        __half2 hp3 = __floats2half2_rn(p[6], p[7]);
        uint4 packed_p;
        packed_p.x = *reinterpret_cast<uint32_t*>(&hp0);
        packed_p.y = *reinterpret_cast<uint32_t*>(&hp1);
        packed_p.z = *reinterpret_cast<uint32_t*>(&hp2);
        packed_p.w = *reinterpret_cast<uint32_t*>(&hp3);
        *reinterpret_cast<uint4*>(my_row + tg_lane * kElemsPerLane) = packed_p;
    }

    const float alpha = exp2f(m_old - m_new);

    #pragma unroll
    for (int32_t off = kLanesPerRow / 2; off > 0; off >>= 1) {
        l_tile += __shfl_xor_sync(0xffffffff, l_tile, off);
    }

    m = m_new;
    l = alpha * l_old + l_tile;
    return alpha;
}

// FlashAttention-2: 一个 block 算 O 的 [Br, D] 行块  (split-Q + split-N)
//   warp grid: WarpCountM (沿 Br) × WarpCountN (沿 Bc 在 QK / 沿 D 在 PV)
//   QK: 每 warp 算 S 的 [WarpM, WarpN_qk = Bc/WarpCountN] slice
//       K 行切 (row-major K): ptr += warp_n_id * WarpN_qk * D
//   PV: 每 warp 算 O 的 [WarpM, WarpN_pv = D /WarpCountN] slice
//       V 列切 (row-major V): ptr += warp_n_id * WarpN_pv  (ldb=D 不变)
template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM, int32_t WarpCountN
>
__device__
void bflash_attention_perf_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    const int32_t N
) {
    constexpr int32_t WarpCount          = WarpCountM * WarpCountN;
    constexpr int32_t WarpM              = Br / WarpCountM;
    constexpr int32_t WarpN_qk           = Bc / WarpCountN;
    constexpr int32_t WarpN_pv           = D  / WarpCountN;
    constexpr int32_t RowsPerWarpSoftmax = Br / WarpCount;

    // Padding: 每行加 8 fp16 消除 bank conflict (stride 从 128 的 32-bank 倍数 → 136)
    constexpr int32_t PAD       = 8;
    constexpr int32_t KV_STRIDE = D  + PAD;   // smem_KV 行 stride = 136
    constexpr int32_t SP_STRIDE = Bc + PAD;   // smem_SP 行 stride = 136

    static_assert(TPB == WarpCount * 32,         "TPB must be 32 * WarpCountM * WarpCountN");
    static_assert(WarpM == 16,                   "WarpM must be 16 (rescale_frag_O 约束 WarpTileM=1)");
    static_assert(WarpN_qk % 16 == 0,            "WarpN_qk must be multiple of 16 (mma N)");
    static_assert(WarpN_pv % 16 == 0,            "WarpN_pv must be multiple of 16 (mma N)");
    static_assert(RowsPerWarpSoftmax % (32 / (Bc / 8)) == 0,  "RowsPerWarpSoftmax 须整除 kRowsPerCall");
    static_assert(KV_STRIDE % 8 == 0,            "KV_STRIDE must be 16B aligned (ldmatrix)");

    // smem_SP 要同时容纳 S/P [Br, Bc] 和末尾 O staging [Br, D]
    constexpr int32_t SP_ELEMS = Br * (SP_STRIDE > D ? SP_STRIDE : D);

    __shared__ __half smem_KV[Bc * KV_STRIDE];
    extern __shared__ char smem_dyn_raw[];
    __half* smem_SP   = reinterpret_cast<__half*>(smem_dyn_raw);
    float*  smem_m    = reinterpret_cast<float*>(smem_dyn_raw + SP_ELEMS * sizeof(__half));
    float*  smem_l    = smem_m + Br;
    float*  smem_alpha = smem_l + Br;

    const int32_t tid     = threadIdx.x;
    const int32_t warp_id = tid / 32;
    const int32_t lane_id = tid % 32;

    const int32_t warp_m_id = warp_id / WarpCountN;
    const int32_t warp_n_id = warp_id % WarpCountN;

    // 同 M-group 的 N-warps 共享 Q rows (stride = KV_STRIDE)
    __half* warp_Q_smem = smem_KV + warp_m_id * WarpM * KV_STRIDE;
    // QK 写 S (stride = SP_STRIDE)
    __half* qk_S_out    = smem_SP + warp_m_id * WarpM * SP_STRIDE + warp_n_id * WarpN_qk;
    // PV 读 P 整行 (stride = SP_STRIDE)
    __half* pv_P_in     = smem_SP + warp_m_id * WarpM * SP_STRIDE;

    // 持久 O 累加器: 每 warp [WarpM, WarpN_pv] slice
    ptx::WarpHMMA_Trans_f16<WarpM, WarpN_pv, Bc> mma_pv;
    mma_pv.zero();

    // Q load: gmem stride=D, smem stride=KV_STRIDE
    block_mma_prelogue_f16_async<TPB, Br, D>(Q, smem_KV, D, KV_STRIDE);

    for (int32_t i = tid; i < Br; i += TPB) {
        smem_m[i] = -INFINITY;
        smem_l[i] = 0.0f;
    }
    cp_async_wait_group<0>();
    __syncthreads();

    // Q 持久化 (stride = KV_STRIDE)
    using QK_mma_t = ptx::WarpHMMA_f16<WarpM, WarpN_qk, D>;
    typename QK_mma_t::FragAFull frag_Q;
    QK_mma_t::load_full_A(warp_Q_smem, KV_STRIDE, frag_Q);
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)D);
    const int32_t Tc = N / Bc;

    for (int32_t t = 0; t < Tc; t++) {
        // K load: gmem stride=D, smem stride=KV_STRIDE
        block_mma_prelogue_f16_async<TPB, Bc, D>(K + t * Bc * D, smem_KV, D, KV_STRIDE);
        cp_async_wait_group<0>();
        __syncthreads();

        // S = Q @ K^T, K in smem with stride KV_STRIDE
        {
            QK_mma_t mma_qk;
            mma_qk.zero();
            mma_qk.forward_with_A(frag_Q, smem_KV + warp_n_id * WarpN_qk * KV_STRIDE, KV_STRIDE);
            mma_qk.stmatrix(qk_S_out, SP_STRIDE);
        }
        __syncthreads();

        // V async load: gmem stride=D, smem stride=KV_STRIDE
        block_mma_prelogue_f16_async<TPB, Bc, D>(V + t * Bc * D, smem_KV, D, KV_STRIDE);

        // 在线 softmax (smem_SP stride = SP_STRIDE)
        constexpr int32_t kLanesPerRow = Bc / 8;
        constexpr int32_t kRowsPerCall = 32 / kLanesPerRow;
        const int32_t sm_tg_id = lane_id / kLanesPerRow;
        #pragma unroll
        for (int32_t rr = 0; rr < RowsPerWarpSoftmax / kRowsPerCall; rr++) {
            const int32_t row_base = warp_id * RowsPerWarpSoftmax + rr * kRowsPerCall;
            const int32_t my_row   = row_base + sm_tg_id;
            float m = smem_m[my_row];
            float l = smem_l[my_row];
            const float alpha = warp_online_softmax<Bc>(
                smem_SP + row_base * SP_STRIDE, SP_STRIDE, scale, lane_id, m, l
            );
            if (lane_id % kLanesPerRow == 0) {
                smem_m[my_row] = m;
                smem_l[my_row] = l;
                smem_alpha[my_row] = alpha;
            }
        }
        __syncthreads();

        // rescale frag_C
        {
            const int32_t r_upper = warp_m_id * WarpM + (lane_id >> 2);
            const int32_t r_lower = r_upper + 8;
            const __half2 a_upper = __float2half2_rn(smem_alpha[r_upper]);
            const __half2 a_lower = __float2half2_rn(smem_alpha[r_lower]);
            rescale_frag_O(mma_pv.frag_C, a_upper, a_lower);
        }

        // V wait
        cp_async_wait_group<0>();
        __syncthreads();

        // O += P @ V, P stride=SP_STRIDE, V stride=KV_STRIDE
        mma_pv.forward(pv_P_in, smem_KV + warp_n_id * WarpN_pv, SP_STRIDE, KV_STRIDE);
        __syncthreads();
    }

    // 1/l 归一化
    {
        const int32_t r_upper = warp_m_id * WarpM + (lane_id >> 2);
        const int32_t r_lower = r_upper + 8;
        const __half2 inv_upper = __float2half2_rn(1.0f / smem_l[r_upper]);
        const __half2 inv_lower = __float2half2_rn(1.0f / smem_l[r_lower]);
        rescale_frag_O(mma_pv.frag_C, inv_upper, inv_lower);
    }

    // frag_C → smem_SP staging → gmem (O staging 用 stride=D, 不需要 padding)
    __half* warp_O_smem_out = smem_SP + warp_m_id * WarpM * D + warp_n_id * WarpN_pv;
    mma_pv.stmatrix(warp_O_smem_out, D);
    __syncthreads();

    block_mma_eplogue_f16<TPB, Br, D>(smem_SP, O, D, D);
}

template<
    int32_t TPB,
    int32_t Br, int32_t Bc, int32_t D,
    int32_t WarpCountM, int32_t WarpCountN
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

    bflash_attention_perf_f16<TPB, Br, Bc, D, WarpCountM, WarpCountN>(
        Q + bh_offset + q_tile_id * Br * D,
        K + bh_offset,
        V + bh_offset,
        O + bh_offset + q_tile_id * Br * D,
        N
    );
}

// Q, K, V, O: [B, H, N, D] row-major fp16 (device 指针)
// 仅支持 D=128; N 必须是 Br 与 Bc 的整数倍
void flash_attention_perf_f16(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half* O,
    int32_t B, int32_t H, int32_t N, int32_t D
) {
    constexpr int32_t Br = 64;
    constexpr int32_t Bc = 64;    // 缩 Bc: 27KB/block → 可放 2 blocks/SM
    constexpr int32_t WarpCountM = 4;   // WarpM = 16 (rescale_frag_O 约束)
    constexpr int32_t WarpCountN = 2;   // 切 Bc / D, 8 warp/block 提 TLP
    constexpr int32_t TPB = WarpCountM * WarpCountN * 32;
    constexpr int32_t kD = 128;

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

    // dynamic smem = smem_SP (padded) + m + l + alpha
    constexpr int32_t PAD = 8;
    constexpr int32_t SP_STRIDE = Bc + PAD;
    // smem_SP 须容纳 max(S/P [Br, SP_STRIDE], O staging [Br, D])
    constexpr int32_t SP_ELEMS = Br * (SP_STRIDE > kD ? SP_STRIDE : kD);
    const size_t smem_dyn_bytes = SP_ELEMS * sizeof(__half)          // smem_SP
                                + 3 * Br * sizeof(float);            // m + l + alpha

    auto kernel = device_flash_attention_perf_f16<TPB, Br, Bc, kD, WarpCountM, WarpCountN>;
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dyn_bytes);

    kernel<<<grid, TPB, smem_dyn_bytes>>>(Q, K, V, O, N);
}
