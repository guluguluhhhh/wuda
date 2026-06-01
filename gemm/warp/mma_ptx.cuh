#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace ptx {

// ============================================================
// PTX helpers
// ============================================================

__device__ __forceinline__
void ldmatrix_x4_b16(const __half* smem_ptr, uint32_t* dst) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr)
    );
}

__device__ __forceinline__
void ldmatrix_x4_trans_b16(const __half* smem_ptr, uint32_t* dst) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr)
    );
}

__device__ __forceinline__
void mma_m16n8k16_f16f16(uint32_t* d, const uint32_t* a, const uint32_t* b, const uint32_t* c) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1])
    );
}

// ============================================================
// WarpHMMA_f16
//   一个 warp 算 WarpM x WarpN 的输出 tile (沿 K 累加 WarpK)
//   A: row_major  B: col_major  C: fp32 累加，最终转回 fp16
//   底层使用 mma.sync.m16n8k16 (两条凑成 m16n16k16)
// ============================================================
template<int32_t WarpM, int32_t WarpN, int32_t WarpK>
struct WarpHMMA_f16 {
    const static int32_t InstructionM = 16;
    const static int32_t InstructionN = 16;
    const static int32_t InstructionK = 16;
    const static int32_t WarpTileM = WarpM / InstructionM;
    const static int32_t WarpTileN = WarpN / InstructionN;

    // A: 每个 lane 持 8 个 half (= 4 uint32) 覆盖 16x16
    uint32_t frag_A[WarpTileM][4];
    // B: 每个 lane 持 8 个 half (= 4 uint32) 覆盖 K=16 x N=16 (拆为两个 N=8)
    uint32_t frag_B[WarpTileN][4];
    // C: 每个 lane 持 4 个 uint32 = 8 个 half (= 两条 m16n8 累加器拼成 m16n16)
    uint32_t frag_C[WarpTileM][WarpTileN][4];

    __device__ __forceinline__
    void ldmatrix(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        const int32_t strideA,
        const int32_t strideB
    ) {
        const int32_t lane = threadIdx.x & 31;

        // A: row_major [WarpM, K-tile=16]
        // ldmatrix.x4: lane t -> row=t%16, col=(t/16)*8
        const int32_t a_row = lane & 15;
        const int32_t a_col = (lane >> 4) << 3;
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            ldmatrix_x4_b16(
                A + (m * InstructionM + a_row) * strideA + a_col,
                frag_A[m]
            );
        }

        // B: col_major [K=16, WarpN] => smem 实际 [WarpN, K=16] row_major
        // strideB 是 K 方向 stride (元素数)
        // ldmatrix.x4: lane t -> N-row = t%8 + (t/16)*8, K-col = ((t/8)&1)*8
        const int32_t b_row = (lane & 7) + ((lane >> 4) << 3);
        const int32_t b_col = ((lane >> 3) & 1) << 3;
        #pragma unroll
        for (int32_t n = 0; n < WarpTileN; ++n) {
            ldmatrix_x4_b16(
                B + (n * InstructionN + b_row) * strideB + b_col,
                frag_B[n]
            );
        }
    }

    __device__ __forceinline__
    void stmatrix(
        __half* C,
        const int32_t strideC
    ) {
        const int32_t lane = threadIdx.x & 31;
        const int32_t row0 = lane >> 2;        // 0..7
        const int32_t col0 = (lane & 3) << 1;  // 0,2,4,6

        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; ++n) {
                __half* dst = C + m * InstructionM * strideC + n * InstructionN;
                // C[0]: 第 1 个 m16n8, 上半 (M=0..7),  N=0..7
                // C[1]: 第 1 个 m16n8, 下半 (M=8..15), N=0..7
                // C[2]: 第 2 个 m16n8, 上半 (M=0..7),  N=8..15
                // C[3]: 第 2 个 m16n8, 下半 (M=8..15), N=8..15
                *reinterpret_cast<uint32_t*>(dst +  row0      * strideC + col0    ) = frag_C[m][n][0];
                *reinterpret_cast<uint32_t*>(dst + (row0 + 8) * strideC + col0    ) = frag_C[m][n][1];
                *reinterpret_cast<uint32_t*>(dst +  row0      * strideC + col0 + 8) = frag_C[m][n][2];
                *reinterpret_cast<uint32_t*>(dst + (row0 + 8) * strideC + col0 + 8) = frag_C[m][n][3];
            }
        }
    }

    __device__
    void forward(
        const __half* __restrict__ A, // [*, strideA]  smem
        const __half* __restrict__ B, // [*, strideB]  smem (col_major / smem layout [WarpN, K])
        const int32_t strideA,
        const int32_t strideB
    ) {
        for (int32_t k = 0; k < WarpK; k += InstructionK) {
            ldmatrix(A + k, B + k, strideA, strideB);

            #pragma unroll
            for (int32_t m = 0; m < WarpTileM; ++m) {
                #pragma unroll
                for (int32_t n = 0; n < WarpTileN; ++n) {
                    // 第 1 条 m16n8: 用 B[0..1], 累加到 C[0..1]
                    mma_m16n8k16_f16f16(
                        &frag_C[m][n][0], frag_A[m], &frag_B[n][0], &frag_C[m][n][0]
                    );
                    // 第 2 条 m16n8: 用 B[2..3], 累加到 C[2..3]
                    mma_m16n8k16_f16f16(
                        &frag_C[m][n][2], frag_A[m], &frag_B[n][2], &frag_C[m][n][2]
                    );
                }
            }
        }
    }

    __device__ __forceinline__
    void zero() {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; ++n) {
                #pragma unroll
                for (int32_t i = 0; i < 4; ++i) {
                    frag_C[m][n][i] = 0u;  // half2(+0.0, +0.0)
                }
            }
        }
    }
};

// ============================================================
// WarpHMMA_Trans_f16
//   B 是 row_major 的版本（即 B 实际上未转置）
//   底层 ldmatrix 使用 .trans 变体
// ============================================================
template<int32_t WarpM, int32_t WarpN, int32_t WarpK>
struct WarpHMMA_Trans_f16 {
    const static int32_t InstructionM = 16;
    const static int32_t InstructionN = 16;
    const static int32_t InstructionK = 16;
    const static int32_t WarpTileM = WarpM / InstructionM;
    const static int32_t WarpTileN = WarpN / InstructionN;

    uint32_t frag_A[WarpTileM][4];
    uint32_t frag_B[WarpTileN][4];
    uint32_t frag_C[WarpTileM][WarpTileN][4];

    __device__ __forceinline__
    void ldmatrix(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        const int32_t strideA,
        const int32_t strideB
    ) {
        const int32_t lane = threadIdx.x & 31;

        // A: 与非 trans 版本完全相同
        const int32_t a_row = lane & 15;
        const int32_t a_col = (lane >> 4) << 3;
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            ldmatrix_x4_b16(
                A + (m * InstructionM + a_row) * strideA + a_col,
                frag_A[m]
            );
        }

        // B: row_major [K=16, WarpN], 用 ldmatrix.trans 把 8x8 转置成 col_major 视图
        // lane t -> K-row = t%16, N-col = ((t/16)&1)*8
        const int32_t b_row = lane & 15;
        const int32_t b_col = ((lane >> 4) & 1) << 3;
        #pragma unroll
        for (int32_t n = 0; n < WarpTileN; ++n) {
            ldmatrix_x4_trans_b16(
                B + b_row * strideB + n * InstructionN + b_col,
                frag_B[n]
            );
        }
    }

    __device__ __forceinline__
    void stmatrix(
        __half* C,
        const int32_t strideC
    ) {
        const int32_t lane = threadIdx.x & 31;
        const int32_t row0 = lane >> 2;
        const int32_t col0 = (lane & 3) << 1;

        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; ++n) {
                __half* dst = C + m * InstructionM * strideC + n * InstructionN;
                *reinterpret_cast<uint32_t*>(dst +  row0      * strideC + col0    ) = frag_C[m][n][0];
                *reinterpret_cast<uint32_t*>(dst + (row0 + 8) * strideC + col0    ) = frag_C[m][n][1];
                *reinterpret_cast<uint32_t*>(dst +  row0      * strideC + col0 + 8) = frag_C[m][n][2];
                *reinterpret_cast<uint32_t*>(dst + (row0 + 8) * strideC + col0 + 8) = frag_C[m][n][3];
            }
        }
    }

    __device__
    void forward(
        const __half* __restrict__ A, // smem [*, strideA]
        const __half* __restrict__ B, // smem [*, strideB]  (row_major [K, N])
        const int32_t strideA,
        const int32_t strideB
    ) {
        for (int32_t k = 0; k < WarpK; k += InstructionK) {
            ldmatrix(A + k, B + k * strideB, strideA, strideB);

            #pragma unroll
            for (int32_t m = 0; m < WarpTileM; ++m) {
                #pragma unroll
                for (int32_t n = 0; n < WarpTileN; ++n) {
                    mma_m16n8k16_f16f16(
                        &frag_C[m][n][0], frag_A[m], &frag_B[n][0], &frag_C[m][n][0]
                    );
                    mma_m16n8k16_f16f16(
                        &frag_C[m][n][2], frag_A[m], &frag_B[n][2], &frag_C[m][n][2]
                    );
                }
            }
        }
    }

    __device__ __forceinline__
    void zero() {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; ++n) {
                #pragma unroll
                for (int32_t i = 0; i < 4; ++i) {
                    frag_C[m][n][i] = 0u;
                }
            }
        }
    }
};

} // namespace ptx
