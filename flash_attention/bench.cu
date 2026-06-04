#include "flash_attention_v1.cu"
#include "flash_attention_v2.cu"
#include "flash_attention_performance.cu"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <unistd.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// ============================================================
// CPU 参考实现:fp32 真值 (全程 fp32, 只在输入端读 fp16)
//   Q, K, V: [B, H, N, D] fp16     O: [B, H, N, D] fp32
// ============================================================
void cpu_attention_f32_ref(
    const __half* Q, const __half* K, const __half* V, float* O,
    int32_t B, int32_t H, int32_t N, int32_t D
) {
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = (float*)malloc(sizeof(float) * N);

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const __half* q_h = Q + ((size_t)b * H + h) * N * D;
            const __half* k_h = K + ((size_t)b * H + h) * N * D;
            const __half* v_h = V + ((size_t)b * H + h) * N * D;
            float*        o_h = O + ((size_t)b * H + h) * N * D;

            for (int i = 0; i < N; i++) {
                float max_s = -INFINITY;
                for (int j = 0; j < N; j++) {
                    float s = 0.0f;
                    for (int k = 0; k < D; k++) {
                        s += __half2float(q_h[i * D + k]) * __half2float(k_h[j * D + k]);
                    }
                    s *= scale;
                    scores[j] = s;
                    if (s > max_s) max_s = s;
                }
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum += scores[j];
                }
                float inv_sum = 1.0f / sum;
                for (int k = 0; k < D; k++) {
                    float acc = 0.0f;
                    for (int j = 0; j < N; j++) {
                        acc += scores[j] * inv_sum * __half2float(v_h[j * D + k]);
                    }
                    o_h[i * D + k] = acc;
                }
            }
        }
    }

    free(scores);
}

// ============================================================
// 误差指标:max abs, mean abs, relative Frobenius
// ============================================================
struct ErrorMetrics {
    float max_abs;
    float mean_abs;
    float rel_frob;
};

ErrorMetrics compute_err(const __half* test, const float* ref, size_t size) {
    float  max_abs     = 0.0f;
    double sum_abs     = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref  = 0.0;
    for (size_t i = 0; i < size; i++) {
        float t = __half2float(test[i]);
        float r = ref[i];
        float d = fabsf(t - r);
        if (d > max_abs) max_abs = d;
        sum_abs     += d;
        sum_sq_diff += (double)d * d;
        sum_sq_ref  += (double)r * r;
    }
    ErrorMetrics m;
    m.max_abs  = max_abs;
    m.mean_abs = (float)(sum_abs / size);
    m.rel_frob = (float)sqrt(sum_sq_diff / (sum_sq_ref + 1e-30));
    return m;
}

// ============================================================
// 正确性测试 (跑多个 kernel)
// ============================================================
typedef void (*fa_fn_t)(const __half*, const __half*, const __half*, __half*,
                        int32_t, int32_t, int32_t, int32_t);

struct FAKernel {
    const char* name;
    fa_fn_t     fn;
};

void test_correctness(FAKernel* kernels, int n_kernels) {
    const int B = 2, H = 32, N = 128, D = 128;
    const size_t size = (size_t)B * H * N * D;

    __half* h_Q     = (__half*)malloc(size * sizeof(__half));
    __half* h_K     = (__half*)malloc(size * sizeof(__half));
    __half* h_V     = (__half*)malloc(size * sizeof(__half));
    float*  h_O_ref = (float*) malloc(size * sizeof(float));
    __half* h_O_gpu = (__half*)malloc(size * sizeof(__half));

    srand(42);
    for (size_t i = 0; i < size; i++) {
        h_Q[i] = __float2half((float)(rand() % 21 - 10) * 0.05f);
        h_K[i] = __float2half((float)(rand() % 21 - 10) * 0.05f);
        h_V[i] = __float2half((float)(rand() % 21 - 10) * 0.05f);
    }

    cpu_attention_f32_ref(h_Q, h_K, h_V, h_O_ref, B, H, N, D);

    __half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_K, size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_V, size * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_O, size * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, size * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, size * sizeof(__half), cudaMemcpyHostToDevice));

    printf("=== Correctness (vs fp32 ref) ===\n");
    printf("%-12s %12s %12s %12s\n", "kernel", "max_abs", "mean_abs", "rel_frob");
    for (int ki = 0; ki < n_kernels; ki++) {
        kernels[ki].fn(d_Q, d_K, d_V, d_O, B, H, N, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, size * sizeof(__half), cudaMemcpyDeviceToHost));

        ErrorMetrics fa = compute_err(h_O_gpu, h_O_ref, size);
        printf("%-12s %12.6f %12.6f %12.6f\n",
               kernels[ki].name, fa.max_abs, fa.mean_abs, fa.rel_frob);
    }
    printf("\n");

    free(h_Q); free(h_K); free(h_V); free(h_O_ref); free(h_O_gpu);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
}

// ============================================================
// 性能扫描 (B, H 固定, N 扫, 多 kernel 并排对比)
// ============================================================
void test_performance(FAKernel* kernels, int n_kernels) {
    const int B = 1, H = 32, D = 128;
    const int warmup = 2, repeats = 10;
    const int sleep_us = 100000;

    const int N_list[] = {512, 1024, 2048, 4096, 8192};
    const int n_cases = sizeof(N_list) / sizeof(int);

    const int Nmax = N_list[n_cases - 1];
    const size_t size_max = (size_t)B * H * Nmax * D;

    __half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, size_max * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_K, size_max * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_V, size_max * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_O, size_max * sizeof(__half)));

    __half* h_buf = (__half*)malloc(size_max * sizeof(__half));
    srand(42);
    for (size_t i = 0; i < size_max; i++)
        h_buf[i] = __float2half((float)(rand() % 21 - 10) * 0.05f);
    CUDA_CHECK(cudaMemcpy(d_Q, h_buf, size_max * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_buf, size_max * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_buf, size_max * sizeof(__half), cudaMemcpyHostToDevice));
    free(h_buf);

    printf("=== Performance: B=%d H=%d D=%d, warmup=%d iters=%d ===\n",
           B, H, D, warmup, repeats);
    printf("%-8s", "N");
    for (int ki = 0; ki < n_kernels; ki++) printf(" %20s", kernels[ki].name);
    printf("\n");
    for (int i = 0; i < 8 + n_kernels * 21; i++) printf("-");
    printf("\n");

    for (int ci = 0; ci < n_cases; ci++) {
        int N = N_list[ci];
        printf("%-8d", N);

        for (int ki = 0; ki < n_kernels; ki++) {
            for (int i = 0; i < warmup; i++)
                kernels[ki].fn(d_Q, d_K, d_V, d_O, B, H, N, D);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t s, e;
            cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s);
            for (int i = 0; i < repeats; i++)
                kernels[ki].fn(d_Q, d_K, d_V, d_O, B, H, N, D);
            cudaEventRecord(e);
            cudaEventSynchronize(e);

            float ms = 0;
            cudaEventElapsedTime(&ms, s, e);
            float avg_ms = ms / repeats;

            double flops  = 4.0 * B * H * (double)N * N * D;
            double tflops = flops / (avg_ms * 1e-3) / 1e12;

            printf(" %10.4fms %6.1fT", avg_ms, tflops);

            cudaEventDestroy(s); cudaEventDestroy(e);
            CUDA_CHECK(cudaDeviceSynchronize());
            usleep(sleep_us);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
}

int main() {
    FAKernel kernels[] = {
        {"fa_v1",   flash_attention_v1_f16},
        {"fa_v2",   flash_attention_v2_f16},
        {"fa_perf", flash_attention_perf_f16},
    };
    const int n_kernels = sizeof(kernels) / sizeof(kernels[0]);

    test_correctness(kernels, n_kernels);
    test_performance(kernels, n_kernels);
    printf("\nDone.\n");
    return 0;
}
