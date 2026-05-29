#pragma once

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// ============================================================
// WarpHMMA_f16
//   一个 warp 算 WarpM x WarpN 的输出 tile (沿 K 累加 WarpK)
//   A: row_major  B: col_major  C: fp32 累加，最终转回 fp16
// ============================================================
template<int32_t WarpM, int32_t WarpN, int32_t WarpK>
struct WarpHMMA_f16 {
    const static int32_t InstructionM = 16;
    const static int32_t InstructionN = 16;
    const static int32_t InstructionK = 16;
    const static int32_t WarpTileM = WarpM / InstructionM;
    const static int32_t WarpTileN = WarpN / InstructionN;

    wmma::fragment<wmma::matrix_a, InstructionM, InstructionN, InstructionK, __half, wmma::row_major> frag_A[WarpTileM];
    wmma::fragment<wmma::matrix_b, InstructionM, InstructionN, InstructionK, __half, wmma::col_major> frag_B[WarpTileN];
    wmma::fragment<wmma::accumulator, InstructionM, InstructionN, InstructionK, float> frag_C[WarpTileM][WarpTileN];
    wmma::fragment<wmma::accumulator, InstructionM, InstructionN, InstructionK, __half> frag_C16;

    __device__ __forceinline__
    void ldmatrix(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        const int32_t strideA,
        const int32_t strideB
    ) {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; ++m) {
            wmma::load_matrix_sync(frag_A[m], A + m * InstructionM * strideA, strideA);
        }

        #pragma unroll
        for (int32_t n = 0; n < WarpTileN; ++n) {
            wmma::load_matrix_sync(frag_B[n], B + n * InstructionN * strideB, strideB);
        }
    }

    __device__ __forceinline__
    void stmatrix(
        __half* C,
        const int32_t strideC
    ) {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; m++) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; n++) {
                #pragma unroll
                for (int32_t k = 0; k < frag_C[m][n].num_elements; k++) {
                    frag_C16.x[k] = __float2half(frag_C[m][n].x[k]);
                }

                wmma::store_matrix_sync(
                    C + m * InstructionM * strideC + n * InstructionN,
                    frag_C16, strideC, wmma::mem_row_major);
            }
        }
    }

    __device__
    void forward(
        const __half* __restrict__ A, // [*, strideA]
        const __half* __restrict__ B, // [*, strideB]
        const int32_t strideA,
        const int32_t strideB
    ) {
        for (int32_t k = 0; k < WarpK; k += InstructionK) {
            ldmatrix(A + k, B + k, strideA, strideB);

            #pragma unroll
            for (int32_t m = 0; m < WarpTileM; m++) {
                #pragma unroll
                for (int32_t n = 0; n < WarpTileN; n++) {
                    wmma::mma_sync(frag_C[m][n], frag_A[m], frag_B[n], frag_C[m][n]);
                }
            }
        }
    }

    __device__ __forceinline__
    void zero() {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; m++) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; n++) {
                wmma::fill_fragment(frag_C[m][n], 0.0f);
            }
        }
    }
};

// ============================================================
// WarpHMMA_Trans_f16
//   B 是 row_major 的版本（即 B 实际上未转置）
// ============================================================
template<int32_t WarpM, int32_t WarpN, int32_t WarpK>
struct WarpHMMA_Trans_f16 {
    const static int32_t InstructionM = 16;
    const static int32_t InstructionN = 16;
    const static int32_t InstructionK = 16;
    const static int32_t WarpTileM = WarpM / InstructionM;
    const static int32_t WarpTileN = WarpN / InstructionN;

    wmma::fragment<wmma::matrix_a, InstructionM, InstructionN, InstructionK, __half, wmma::row_major> frag_A[WarpTileM];
    wmma::fragment<wmma::matrix_b, InstructionM, InstructionN, InstructionK, __half, wmma::row_major> frag_B[WarpTileN];
    wmma::fragment<wmma::accumulator, InstructionM, InstructionN, InstructionK, float> frag_C[WarpTileM][WarpTileN];
    wmma::fragment<wmma::accumulator, InstructionM, InstructionN, InstructionK, __half> frag_C16;

    __device__ __forceinline__
    void ldmatrix(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        const int32_t strideA,
        const int32_t strideB
    ) {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; m++) {
            wmma::load_matrix_sync(frag_A[m], A + m * InstructionM * strideA, strideA);
        }

        #pragma unroll
        for (int32_t n = 0; n < WarpTileN; n++) {
            wmma::load_matrix_sync(frag_B[n], B + n * InstructionN * strideB, strideB);
        }
    }

    __device__ __forceinline__
    void stmatrix(
        __half* C,
        const int32_t strideC
    ) {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; m++) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; n++) {
                #pragma unroll
                for (int32_t k = 0; k < frag_C[m][n].num_elements; k++) {
                    frag_C16.x[k] = __float2half(frag_C[m][n].x[k]);
                }

                wmma::store_matrix_sync(
                    C + m * InstructionM * strideC + n * InstructionN,
                    frag_C16, strideC, wmma::mem_row_major);
            }
        }
    }

    __device__
    void forward(
        const __half* __restrict__ A, // [*, strideA]
        const __half* __restrict__ B, // [*, strideB]
        const int32_t strideA,
        const int32_t strideB
    ) {
        for (int32_t k = 0; k < WarpK; k += InstructionK) {
            ldmatrix(A + k, B + k * strideB, strideA, strideB);

            #pragma unroll
            for (int32_t m = 0; m < WarpTileM; m++) {
                #pragma unroll
                for (int32_t n = 0; n < WarpTileN; n++) {
                    wmma::mma_sync(frag_C[m][n], frag_A[m], frag_B[n], frag_C[m][n]);
                }
            }
        }
    }

    __device__ __forceinline__
    void zero() {
        #pragma unroll
        for (int32_t m = 0; m < WarpTileM; m++) {
            #pragma unroll
            for (int32_t n = 0; n < WarpTileN; n++) {
                wmma::fill_fragment(frag_C[m][n], 0.0f);
            }
        }
    }
};
