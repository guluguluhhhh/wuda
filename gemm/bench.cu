#include "gemm_basic.cu"
#include "gemm_double_buffering.cu"
#include "gemm_tma.cu"
#include "gemm_tma_ws.cu"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <unistd.h>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                  \
    do {                                                    \
        cublasStatus_t st = call;                           \
        if (st != CUBLAS_STATUS_SUCCESS) {                  \
            fprintf(stderr, "cuBLAS Error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (int)st);           \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// ============================================================================
//  cuBLAS wrapper (TN layout: A[M,K] row-major, B[N,K] row-major)
//  cuBLAS 是 column-major，TN 对应 cublas 的 NT:
//    C^T[N,M] = B[N,K] * A^T[K,M]
// ============================================================================
static cublasHandle_t g_cublas_handle = nullptr;

void gemm_cublas_f16(const __half* A, const __half* B, __half* C,
                     int32_t M, int32_t N, int32_t K) {
    if (!g_cublas_handle) CUBLAS_CHECK(cublasCreate(&g_cublas_handle));

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    // cuBLAS col-major: C(N,M) = B(N,K) * A^T(K,M)
    CUBLAS_CHECK(cublasHgemm(
        g_cublas_handle,
        CUBLAS_OP_T,   // B^T → B is [N,K] row-major = [K,N] col-major, transposed = [N,K]
        CUBLAS_OP_N,   // A   → A is [M,K] row-major = [K,M] col-major
        N, M, K,
        &alpha,
        B, K,          // ldb = K
        A, K,          // lda = K
        &beta,
        C, N           // ldc = N
    ));
}

// ============================================================================
//  CPU 参考实现
// ============================================================================
void cpu_gemm_tn_f16(const __half* A, const __half* B, __half* C,
                     int32_t M, int32_t N, int32_t K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
            C[m * N + n] = __float2half(acc);
        }
    }
}

// ============================================================================
//  正确性测试
// ============================================================================
bool test_correctness(const char* name,
                      void (*gemm_fn)(const __half*, const __half*, __half*,
                                      int32_t, int32_t, int32_t)) {
    printf("=== Correctness: %s ===\n", name);

    const int M = 256, N = 256, K = 128;
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)N * K;
    size_t size_C = (size_t)M * N;

    __half* h_A = (__half*)malloc(size_A * sizeof(__half));
    __half* h_B = (__half*)malloc(size_B * sizeof(__half));
    __half* h_C_cpu = (__half*)malloc(size_C * sizeof(__half));
    __half* h_C_gpu = (__half*)malloc(size_C * sizeof(__half));

    srand(42);
    for (size_t i = 0; i < size_A; i++)
        h_A[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);
    for (size_t i = 0; i < size_B; i++)
        h_B[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);

    cpu_gemm_tn_f16(h_A, h_B, h_C_cpu, M, N, K);

    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(__half), cudaMemcpyHostToDevice));

    gemm_fn(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(__half), cudaMemcpyDeviceToHost));

    int errors = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < size_C; i++) {
        float cpu_val = __half2float(h_C_cpu[i]);
        float gpu_val = __half2float(h_C_gpu[i]);
        float err = fabsf(cpu_val - gpu_val);
        max_err = fmaxf(max_err, err);
        if (err > 0.1f) {
            if (errors < 5)
                fprintf(stderr, "  Mismatch at [%d,%d]: CPU=%.4f GPU=%.4f\n",
                        (int)(i / N), (int)(i % N), cpu_val, gpu_val);
            errors++;
        }
    }

    bool pass = (errors == 0);
    printf("  %s (max error: %.6f)\n\n", pass ? "PASSED" : "FAILED", max_err);

    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return pass;
}

// ============================================================================
//  性能测试 — 所有 kernel 同框对比，预分配显存，每轮 sleep 防降频
// ============================================================================
typedef void (*gemm_fn_t)(const __half*, const __half*, __half*,
                          int32_t, int32_t, int32_t);

struct GemmKernel {
    const char* name;
    gemm_fn_t fn;
};

void test_performance_all(GemmKernel* kernels, int num_kernels) {
    const int SEP = 512;
    const int MAX_MNK = 8192;
    const int warmup = 2;
    const int repeats = 10;
    const int sleep_us = 100000;  // 100ms

    // 预分配最大矩阵
    size_t max_size = (size_t)MAX_MNK * MAX_MNK;
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, max_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_B, max_size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, max_size * sizeof(__half)));

    // 初始化数据
    __half* h_buf = (__half*)malloc(max_size * sizeof(__half));
    srand(42);
    for (size_t i = 0; i < max_size; i++)
        h_buf[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);
    CUDA_CHECK(cudaMemcpy(d_A, h_buf, max_size * sizeof(__half), cudaMemcpyHostToDevice));
    for (size_t i = 0; i < max_size; i++)
        h_buf[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);
    CUDA_CHECK(cudaMemcpy(d_B, h_buf, max_size * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_buf);

    // 表头
    printf("\n=== Performance: M=N=K sweep, step=%d, warmup=%d, iters=%d ===\n", SEP, warmup, repeats);
    printf("%-6s", "MNK");
    for (int i = 0; i < num_kernels; i++)
        printf(" %15s", kernels[i].name);
    printf("\n");
    for (int i = 0; i < 6 + num_kernels * 16; i++) printf("-");
    printf("\n");

    // 找 cuBLAS 在 kernels 数组中的索引
    int cublas_idx = -1;
    for (int i = 0; i < num_kernels; i++)
        if (strstr(kernels[i].name, "cuBLAS")) cublas_idx = i;

    // 逐规模测试
    for (int MNK = SEP; MNK <= MAX_MNK; MNK += SEP) {
        int M = MNK, N = MNK, K = MNK;
        printf("%-6d", MNK);

        double tflops_arr[16] = {};
        for (int ki = 0; ki < num_kernels; ki++) {
            // warmup
            for (int i = 0; i < warmup; i++)
                kernels[ki].fn(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 计时
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < repeats; i++)
                kernels[ki].fn(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            float avg_ms = ms / repeats;

            double tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12;
            tflops_arr[ki] = tflops;

            if (cublas_idx >= 0 && ki != cublas_idx && tflops_arr[cublas_idx] > 0) {
                double pct = tflops / tflops_arr[cublas_idx] * 100.0;
                printf(" %8.1f(%4.1f%%)", tflops, pct);
            } else {
                printf(" %14.1f", tflops);
            }

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            // sleep 防降频
            CUDA_CHECK(cudaDeviceSynchronize());
            usleep(sleep_us);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// ============================================================================
//  Main
// ============================================================================
int main() {
    GemmKernel kernels[] = {
        {"cuBLAS", gemm_cublas_f16},
        {"basic",  gemm2_f16},
        {"db+big_tile", gemm3_f16},
        {"tma", gemm_tma_f16},
        {"tma_ws", gemm_tma_ws_f16},
    };
    const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    // 正确性测试
    for (int i = 0; i < num_kernels; i++) {
        if (!test_correctness(kernels[i].name, kernels[i].fn)) {
            printf("%s correctness FAILED!\n", kernels[i].name);
            return 1;
        }
    }

    // 性能扫描
    test_performance_all(kernels, num_kernels);

    if (g_cublas_handle) cublasDestroy(g_cublas_handle);
    printf("\nDone.\n");
    return 0;
}


// (base) root@autodl-container-c519kcy9d5-11b71870:~/wuda/gemm# nvcc -O3 -arch=sm_120 bench.cu -lcublas -lcuda -o bench && ./bench
// === Correctness: cuBLAS ===
//   PASSED (max error: 0.003906)

// === Correctness: basic ===
//   PASSED (max error: 0.001953)

// === Correctness: db+big_tile ===
//   PASSED (max error: 0.003906)

// === Correctness: tma ===
//   PASSED (max error: 0.003906)

// === Correctness: tma_ws ===
//   PASSED (max error: 0.003906)


// === Performance: M=N=K sweep, step=512, warmup=2, iters=10 ===
// MNK             cuBLAS           basic     db+big_tile             tma          tma_ws
// --------------------------------------------------------------------------------------
// 512              41.5     31.0(74.6%)     17.8(42.8%)     20.9(50.3%)     23.8(57.4%)
// 1024            127.0    100.7(79.3%)     77.8(61.3%)    110.4(86.9%)    124.6(98.1%)
// 1536            275.8    145.6(52.8%)    181.2(65.7%)    243.5(88.3%)    276.8(100.3%)
// 2048            283.8    165.9(58.5%)    258.8(91.2%)    301.6(106.3%)    303.1(106.8%)
// 2560            314.9    172.9(54.9%)    212.2(67.4%)    310.4(98.5%)    311.8(99.0%)
// 3072            332.0    179.7(54.1%)    303.2(91.3%)    279.9(84.3%)    282.6(85.1%)
// 3584            379.7    182.5(48.1%)    283.0(74.5%)    380.0(100.1%)    382.5(100.7%)
// 4096            364.4    193.7(53.1%)    279.5(76.7%)    369.0(101.2%)    371.6(102.0%)
// 4608            364.2    187.1(51.4%)    332.3(91.2%)    380.3(104.4%)    375.2(103.0%)
// 5120            331.3    182.2(55.0%)    307.7(92.9%)    352.9(106.5%)    352.5(106.4%)
// 5632            363.3    189.5(52.2%)    326.6(89.9%)    373.9(102.9%)    373.6(102.8%)
// 6144            357.0    188.8(52.9%)    331.4(92.8%)    380.3(106.5%)    377.6(105.8%)
// 6656            369.5    191.1(51.7%)    341.6(92.4%)    373.0(100.9%)    370.1(100.2%)
// 7168            385.7    183.1(47.5%)    328.0(85.0%)    389.2(100.9%)    386.3(100.2%)
// 7680            393.9    166.1(42.2%)    319.0(81.0%)    391.4(99.4%)    387.9(98.5%)
// 8192            400.1    170.2(42.5%)    325.2(81.3%)    401.3(100.3%)    392.5(98.1%)

// Done.