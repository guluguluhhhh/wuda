#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// ============================================================================
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
//  Warp reduce 行求和 kernel
//  每个 warp 处理一行，行长 K 由 stride loop 覆盖
// ============================================================================
__global__ void rowsum_warp_reduce(const __half* __restrict__ A,
                                   __half* __restrict__ out,
                                   int32_t M, int32_t K) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (row >= M) return;

    const __half* row_ptr = A + (size_t)row * K;
    float sum = 0.0f;
    for (int i = lane; i < K; i += 32) {
        sum += __half2float(row_ptr[i]);
    }

    sum = warp_reduce_sum_f32(sum);

    if (lane == 0) {
        out[row] = __float2half(sum);
    }
}

// ============================================================================
//  Host wrappers
// ============================================================================
void rowsum_reduce(const __half* d_A, __half* d_out, int32_t M, int32_t K) {
    int warps_needed = M;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int grid = (warps_needed + warps_per_block - 1) / warps_per_block;
    rowsum_warp_reduce<<<grid, threads_per_block>>>(d_A, d_out, M, K);
}

// ============================================================================
//  Tensor core 行求和：直接用 wmma，A[16,16] * ones[16,16] 累加
//
//  线程组织：256 threads/block = 8 warps/block
//  Block 负责：8 warps × 16 rows/warp = 128 行
//  Warp  负责：16 行，沿 K 以 16 步长循环累加
//  Grid：M / 128 个 block
// ============================================================================
__global__ void rowsum_wmma_kernel(const __half* __restrict__ A,
                                   __half* __restrict__ out,
                                   int32_t M, int32_t K) {
    using namespace nvcuda::wmma;

    // global_warp_id = (blockIdx.x * 4 + warp_in_block)
    // 每个 warp 负责 16 行: [warp_row, warp_row+16)
    int warp_row = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * 16;
    if (warp_row >= M) return;

    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    fill_fragment(b_frag, __float2half(1.0f));

    const __half* row_base = A + (size_t)warp_row * K;

    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, row_base + k, K);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // c_frag[m][n] = row sum (all columns same), store 后取第 0 列
    constexpr int WARPS_PER_BLOCK = 8;
    __shared__ float smem_c[WARPS_PER_BLOCK * 16 * 16];
    int warp_in_block = (threadIdx.x / 32);
    float* my_smem = smem_c + warp_in_block * 16 * 16;

    store_matrix_sync(my_smem, c_frag, 16, mem_row_major);

    int lane = threadIdx.x & 31;
    if (lane < 16) {
        out[warp_row + lane] = __float2half(my_smem[lane * 16]);
    }
}

void rowsum_wmma(const __half* d_A, __half* d_out, int32_t M, int32_t K) {
    int warps_needed = (M + 15) / 16;
    constexpr int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int grid = (warps_needed + warps_per_block - 1) / warps_per_block;
    rowsum_wmma_kernel<<<grid, threads_per_block>>>(d_A, d_out, M, K);
}

// ============================================================================
//  正确性测试
// ============================================================================
bool test_correctness() {
    printf("=== Correctness Test (Row Sum) ===\n");

    const int M = 256, K = 512;
    size_t size_A = (size_t)M * K;

    __half* h_A = (__half*)malloc(size_A * sizeof(__half));
    __half* h_ref = (__half*)malloc(M * sizeof(__half));
    __half* h_out = (__half*)malloc(M * sizeof(__half));

    srand(42);
    for (size_t i = 0; i < size_A; i++)
        h_A[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);

    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += __half2float(h_A[m * K + k]);
        h_ref[m] = __float2half(sum);
    }

    __half *d_A, *d_out;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(__half), cudaMemcpyHostToDevice));

    // 测试 warp reduce
    int errors = 0;
    printf("  Warp reduce: ");
    CUDA_CHECK(cudaMemset(d_out, 0, M * sizeof(__half)));
    rowsum_reduce(d_A, d_out, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * sizeof(__half), cudaMemcpyDeviceToHost));
    errors = 0;
    for (int m = 0; m < M; m++) {
        float err = fabsf(__half2float(h_ref[m]) - __half2float(h_out[m]));
        if (err > 0.5f) errors++;
    }
    printf("%s\n", errors == 0 ? "PASSED" : "FAILED");

    // 测试 wmma
    printf("  WMMA: ");
    CUDA_CHECK(cudaMemset(d_out, 0, M * sizeof(__half)));
    rowsum_wmma(d_A, d_out, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * sizeof(__half), cudaMemcpyDeviceToHost));
    errors = 0;
    for (int m = 0; m < M; m++) {
        float err = fabsf(__half2float(h_ref[m]) - __half2float(h_out[m]));
        if (err > 1.0f) errors++;
    }
    printf("%s\n\n", errors == 0 ? "PASSED" : "FAILED");

    free(h_A); free(h_ref); free(h_out);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_out));
    return true;
}

// ============================================================================
//  性能测试
// ============================================================================
void test_performance() {
    printf("=== Performance: Row Sum (reduce vs GEMM) ===\n");
    printf("%-8s %-8s %12s %12s %12s %12s\n",
           "M", "K", "Warp(ms)", "BW(GB/s)", "WMMA(ms)", "BW(GB/s)");
    printf("----------------------------------------------------------------------\n");

    // K 从小到大，M 保证数据超出 L2
    const int configs[][2] = {
        {131072,  16},
        {131072,  32},
        {131072,  64},
        {131072,  128},
        {131072,  256},
        {131072,  512},
        {131072,  1024},
        {131072,  2048},
        {131072,  4096},
        {131072,  8192},
    };
    const int num_configs = sizeof(configs) / sizeof(configs[0]);
    const int warmup = 10;
    const int repeats = 50;

    for (int c = 0; c < num_configs; c++) {
        int M = configs[c][0], K = configs[c][1];

        size_t size_A = (size_t)M * K;
        double data_bytes = (double)size_A * sizeof(__half);

        __half* h_A = (__half*)malloc(size_A * sizeof(__half));
        for (size_t i = 0; i < size_A; i++)
            h_A[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);

        __half *d_A, *d_out_reduce, *d_out_wmma;
        CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out_reduce, M * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out_wmma, M * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(__half), cudaMemcpyHostToDevice));

        // Warmup
        for (int i = 0; i < warmup; i++) {
            rowsum_reduce(d_A, d_out_reduce, M, K);
            rowsum_wmma(d_A, d_out_wmma, M, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Bench warp reduce
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; i++)
            rowsum_reduce(d_A, d_out_reduce, M, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_reduce = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_reduce, start, stop));
        ms_reduce /= repeats;
        double bw_reduce = data_bytes / (ms_reduce * 1e-3) / 1e9;

        // Bench wmma
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; i++)
            rowsum_wmma(d_A, d_out_wmma, M, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_wmma = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms_wmma, start, stop));
        ms_wmma /= repeats;
        double bw_wmma = data_bytes / (ms_wmma * 1e-3) / 1e9;

        printf("%-8d %-8d %12.3f %12.1f %12.3f %12.1f\n",
               M, K, ms_reduce, bw_reduce, ms_wmma, bw_wmma);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        free(h_A);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_out_reduce));
        CUDA_CHECK(cudaFree(d_out_wmma));
    }
}

// ============================================================================
int main() {
    test_correctness();
    test_performance();
    printf("\nDone.\n");
    return 0;
}


// (base) root@autodl-container-24694dbb57-e357ec84:~/wuda/gemm# ./a 
// === Correctness Test (Row Sum) ===
//   Warp reduce: PASSED
//   WMMA: PASSED

// === Performance: Row Sum (reduce vs GEMM) ===
// M        K            Warp(ms)     BW(GB/s)     WMMA(ms)     BW(GB/s)
// ----------------------------------------------------------------------
// 131072   16              0.010        403.7        0.002       1724.2
// 131072   32              0.010        808.6        0.004       2052.8
// 131072   64              0.012       1357.8        0.008       2036.5
// 131072   128             0.010       3220.0        0.012       2707.0
// 131072   256             0.013       5298.8        0.023       2979.9
// 131072   512             0.082       1638.2        0.187        717.7
// 131072   1024            0.161       1671.4        0.436        615.6
// 131072   2048            0.318       1687.9        0.879        610.8
// 131072   4096            0.636       1688.8        1.772        606.0
// 131072   8192            1.276       1682.5        3.561        603.1

// Done.