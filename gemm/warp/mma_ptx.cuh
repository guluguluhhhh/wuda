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
    const static int32_t KIters    = WarpK / InstructionK;

    // A: 每个 lane 持 8 个 half (= 4 uint32) 覆盖 16x16
    uint32_t frag_A[WarpTileM][4];
    // B: 每个 lane 持 8 个 half (= 4 uint32) 覆盖 K=16 x N=16 (拆为两个 N=8)
    uint32_t frag_B[WarpTileN][4];
    // C: 每个 lane 持 4 个 uint32 = 8 个 half (= 两条 m16n8 累加器拼成 m16n16)
    uint32_t frag_C[WarpTileM][WarpTileN][4];

    // 跨 t 循环持久的 A (= Q) 在外面提前 load 一次, 类型如下
    //   frag_A_full[ki][m][i]: KIters × WarpTileM × 4 uint32 per lane
    using FragAFull = uint32_t[KIters][WarpTileM][4];

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

    // 只加载 B, 给"A 已在 reg 里"的场景用
    __device__ __forceinline__
    void ldmatrix_B_only(const __half* __restrict__ B, const int32_t strideB) {
        const int32_t lane = threadIdx.x & 31;
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

    // 把整个 A (跨所有 KIters) 从 smem 一次性 load 到外部 frag (类型 FragAFull)
    //   后续每个 t 循环不再重复读 smem_A, 直接喂给 forward_with_A
    __device__ __forceinline__
    static void load_full_A(
        const __half* __restrict__ A,
        const int32_t strideA,
        FragAFull& frag
    ) {
        const int32_t lane = threadIdx.x & 31;
        const int32_t a_row = lane & 15;
        const int32_t a_col = (lane >> 4) << 3;
        #pragma unroll
        for (int32_t ki = 0; ki < KIters; ++ki) {
            #pragma unroll
            for (int32_t m = 0; m < WarpTileM; ++m) {
                ldmatrix_x4_b16(
                    A + (m * InstructionM + a_row) * strideA
                      + a_col + ki * InstructionK,
                    frag[ki][m]
                );
            }
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

    // A 来自外部 (已在 reg 里), 每 k-iter 只 load B
    __device__
    void forward_with_A(
        const FragAFull& extA,
        const __half* __restrict__ B,
        const int32_t strideB
    ) {
        #pragma unroll
        for (int32_t ki = 0; ki < KIters; ++ki) {
            ldmatrix_B_only(B + ki * InstructionK, strideB);

            #pragma unroll
            for (int32_t m = 0; m < WarpTileM; ++m) {
                #pragma unroll
                for (int32_t n = 0; n < WarpTileN; ++n) {
                    mma_m16n8k16_f16f16(
                        &frag_C[m][n][0], extA[ki][m], &frag_B[n][0], &frag_C[m][n][0]
                    );
                    mma_m16n8k16_f16f16(
                        &frag_C[m][n][2], extA[ki][m], &frag_B[n][2], &frag_C[m][n][2]
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

    // 从 fp16 smem 加载到 frag_C (stmatrix 的逆操作)
    __device__ __forceinline__
    void load_C(
        const __half* C,
        const int32_t strideC
    ) {
        const int32_t lane = threadIdx.x & 31;
        const int32_t row0 = lane >> 2;
        const int32_t col0 = (lane & 3) << 1;

        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; ++n) {
                const __half* src = C + m * InstructionM * strideC + n * InstructionN;
                frag_C[m][n][0] = *reinterpret_cast<const uint32_t*>(src +  row0      * strideC + col0    );
                frag_C[m][n][1] = *reinterpret_cast<const uint32_t*>(src + (row0 + 8) * strideC + col0    );
                frag_C[m][n][2] = *reinterpret_cast<const uint32_t*>(src +  row0      * strideC + col0 + 8);
                frag_C[m][n][3] = *reinterpret_cast<const uint32_t*>(src + (row0 + 8) * strideC + col0 + 8);
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

// ============================================================
// WarpHMMA_f16_sw64
//   与 WarpHMMA_f16 等价，但 smem 是 TMA SW64 swizzle 布局
//   核心区别：ldmatrix 地址需用 atom_phys = atom_logical XOR (row & 3)
//   要求：strideA == strideB == WarpK == 32 halves (= 64B = 4 atoms 一行)
//          WarpM/WarpN 必须是 4 的倍数 (这样每 warp 起始 row mod 4 = 0)
//   仍是 fp16 累加（mma.sync.m16n8k16.f16.f16.f16.f16）
// ============================================================
template<int32_t WarpM, int32_t WarpN, int32_t WarpK>
struct WarpHMMA_f16_sw64 {
    const static int32_t InstructionM = 16;
    const static int32_t InstructionN = 16;
    const static int32_t InstructionK = 16;
    const static int32_t WarpTileM = WarpM / InstructionM;
    const static int32_t WarpTileN = WarpN / InstructionN;

    static_assert(WarpK == 32, "sw64: WarpK must == 32 (matches 64B swizzle row)");

    uint32_t frag_A[WarpTileM][4];
    uint32_t frag_B[WarpTileN][4];
    uint32_t frag_C[WarpTileM][WarpTileN][4];

    // SW64 atom swizzle: 周期 8 行, pairs of rows 共享 XOR mask
    //   row 0,1 → XOR 0;  row 2,3 → XOR 1;  row 4,5 → XOR 2;  row 6,7 → XOR 3
    //   对应 cute Swizzle<2, 4, 3>: offset ^= ((offset >> 7) & 3) << 4
    static __device__ __forceinline__
    int32_t sw_atom(int32_t atom_logical, int32_t row) {
        return atom_logical ^ ((row >> 1) & 0x3);
    }

    // k_atom_off: 当前这次 ldmatrix 在 stage 内 K 维度的 atom 偏移
    //   k=0  → k_atom_off=0   (atom_local=0 or 1)
    //   k=16 → k_atom_off=2   (atom_local=2 or 3)
    __device__ __forceinline__
    void ldmatrix_sw(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        const int32_t strideA,
        const int32_t strideB,
        const int32_t k_atom_off
    ) {
        const int32_t lane = threadIdx.x & 31;

        const int32_t a_row_local  = lane & 15;
        const int32_t a_atom_local = (lane >> 4) & 1;
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            int32_t row = m * InstructionM + a_row_local;
            int32_t atom_phys = sw_atom(k_atom_off + a_atom_local, row);
            ldmatrix_x4_b16(A + row * strideA + atom_phys * 8, frag_A[m]);
        }

        const int32_t b_row_local  = (lane & 7) + ((lane >> 4) << 3);
        const int32_t b_atom_local = (lane >> 3) & 1;
        #pragma unroll
        for (int32_t n = 0; n < WarpTileN; ++n) {
            int32_t row = n * InstructionN + b_row_local;
            int32_t atom_phys = sw_atom(k_atom_off + b_atom_local, row);
            ldmatrix_x4_b16(B + row * strideB + atom_phys * 8, frag_B[n]);
        }
    }

    __device__ __forceinline__
    void stmatrix(__half* C, const int32_t strideC) {
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
        const __half* __restrict__ A, // smem stage base (swizzled)
        const __half* __restrict__ B, // smem stage base (swizzled)
        const int32_t strideA,        // = WarpK (32 halves)
        const int32_t strideB         // = WarpK (32 halves)
    ) {
        constexpr int32_t Kiters = WarpK / InstructionK;
        #pragma unroll
        for (int32_t k = 0; k < Kiters; ++k) {
            // k=0 → k_atom_off=0; k=1 → k_atom_off=2 (16 halves = 2 atoms)
            int32_t k_atom_off = k * (InstructionK / 8);
            ldmatrix_sw(A, B, strideA, strideB, k_atom_off);

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
