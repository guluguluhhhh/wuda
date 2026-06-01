#include "gemm_basic.cu"
#include "gemm_double_buffering.cu"
#include "gemm_tma.cu"

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
