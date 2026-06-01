#include "gemm_basic.cu"
#include "gemm_double_buffering.cu"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
//  CPU 参考实现
// ============================================================================
void cpu_gemm_tn_f16(const __half* A, const __half* B, __half* C,
                     int32_t M, int32_t N, int32_t K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
            }
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
//  性能测试
// ============================================================================
void test_performance(const char* name,
                      void (*gemm_fn)(const __half*, const __half*, __half*,
                                      int32_t, int32_t, int32_t)) {
    printf("=== Performance: %s ===\n", name);
    printf("%-6s %-6s %-6s %10s %10s %10s\n", "M", "N", "K", "Time(ms)", "TFLOPS", "BW(GB/s)");
    printf("--------------------------------------------------------\n");

    const int sizes[][3] = {
        {128,  128,  128},
        {256,  256,  256},
        {512,  512,  512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192},
    };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int warmup = 10;
    const int repeats = 20;

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s][0], N = sizes[s][1], K = sizes[s][2];

        size_t size_A = (size_t)M * K;
        size_t size_B = (size_t)N * K;
        size_t size_C = (size_t)M * N;

        __half* h_A = (__half*)malloc(size_A * sizeof(__half));
        __half* h_B = (__half*)malloc(size_B * sizeof(__half));
        for (size_t i = 0; i < size_A; i++)
            h_A[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);
        for (size_t i = 0; i < size_B; i++)
            h_B[i] = __float2half((float)(rand() % 10 - 5) * 0.1f);

        __half *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(__half), cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; i++)
            gemm_fn(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; i++)
            gemm_fn(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / repeats;

        double flops = 2.0 * M * N * K;
        double tflops = flops / (avg_ms * 1e-3) / 1e12;
        double bytes = (size_A + size_B + size_C) * sizeof(__half);
        double bw_gbs = bytes / (avg_ms * 1e-3) / 1e9;

        printf("%-6d %-6d %-6d %10.3f %10.2f %10.2f\n", M, N, K, avg_ms, tflops, bw_gbs);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        free(h_A); free(h_B);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    printf("\n");
}

// ============================================================================
//  Main — 后续新版本加在这里即可
// ============================================================================
int main() {
    struct GemmKernel {
        const char* name;
        void (*fn)(const __half*, const __half*, __half*, int32_t, int32_t, int32_t);
    };

    GemmKernel kernels[] = {
        {"gemm_basic (wmma)", gemm2_f16},
        {"gemm_double_buf (cp.async)", gemm3_f16},
    };
    const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    for (int i = 0; i < num_kernels; i++) {
        if (!test_correctness(kernels[i].name, kernels[i].fn)) {
            printf("%s correctness FAILED, skipping perf.\n\n", kernels[i].name);
            continue;
        }
        test_performance(kernels[i].name, kernels[i].fn);
    }

    printf("Done.\n");
    return 0;
}
