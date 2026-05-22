#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

constexpr int BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

// 1. 通过 warp shuffle 实现跨 lane 的 compare-swap
__device__ void warp_compare_swap(float &val, int stride) {
    float other_val = __shfl_xor_sync(0xFFFFFFFF, val, stride);

    bool keep_min = ((threadIdx.x & 31) & stride) == 0;
    bool should_swap = keep_min ? (other_val < val) : (other_val > val);
    if (should_swap) {
        val = other_val;
    }
}

// 辅助：编译期计算最大的 2^k < n
constexpr __host__ __device__ int largest_pow2_below(int n) {
    int h = 1;
    while (h < n) h <<= 1;
    return h >> 1;
}

// 辅助：本地 compare-swap（stride >= 32 时，两个值在同一 lane 的不同寄存器中）
__device__ void local_compare_swap(float &a, float &b) {
    if (a > b) {
        float tmp = a; a = b; b = tmp;
    }
}

// 2. merge_odd_continue（Algorithm 1 的递归部分）
//    LEN = 子数组的总元素数（必须是 32 的倍数）
//    IsLeft = true 表示 dummy 在左边
//    vals[] 是每个 lane 持有的 LEN/32 个寄存器
template <int LEN, bool IsLeft>
__device__ void merge_odd_continue(float vals[]) {
    if constexpr (LEN == 32) {
        #pragma unroll
        for (int stride = 16; stride >= 1; stride >>= 1) {
            warp_compare_swap(vals[0], stride);
        }
    } else {
        constexpr int H = largest_pow2_below(LEN);
        constexpr int REG_STRIDE = H / 32;
        constexpr int NUM_PAIRS = (LEN - H) / 32;

        #pragma unroll
        for (int r = 0; r < NUM_PAIRS; r++) {
            local_compare_swap(vals[r], vals[r + REG_STRIDE]);
        }

        if constexpr (IsLeft) {
            constexpr int LEFT_LEN = LEN - H;
            merge_odd_continue<LEFT_LEN, true>(vals);
            merge_odd_continue<H, false>(vals + LEFT_LEN / 32);
        } else {
            constexpr int RIGHT_LEN = LEN - H;
            merge_odd_continue<H, true>(vals);
            merge_odd_continue<RIGHT_LEN, false>(vals + H / 32);
        }
    }
}

// 3. Odd-size merge（Algorithm 1）
//    合并两个已排序的 lane-stride register array
template <int L_LEN, int R_LEN>
__device__ void merge_odd(float L_vals[], float R_vals[]) {
    constexpr int M = L_LEN / 32;
    constexpr int NUM_GROUPS = (L_LEN < R_LEN ? L_LEN : R_LEN) / 32;
    int lane_id = threadIdx.x & 31;

    // 第一步：inverted comparison
    // compare-swap(L[ℓ_L - 1 - i], R[i]) 对 i = 0 到 min_len-1，按每 32 个分组 g：
    // 第 g 组 (i = 32g .. 32g+31):
    //   R[32g + j]           → lane j,    寄存器 R_vals[g]
    //   L[ℓ_L - 1 - 32g - j] → lane 31-j, 寄存器 L_vals[M-1-g]   (M = L_LEN/32)
    #pragma unroll
    for (int g = 0; g < NUM_GROUPS; g++) {
        float l_rev = __shfl_sync(0xFFFFFFFF, L_vals[M - 1 - g], 31 - lane_id);
        float r_val = R_vals[g];

        R_vals[g] = fmaxf(l_rev, r_val);
        float min_val = fminf(l_rev, r_val);
        L_vals[M - 1 - g] = __shfl_sync(0xFFFFFFFF, min_val, 31 - lane_id);
    }

    // 第二步：L 和 R 各自独立排序
    merge_odd_continue<L_LEN, true>(L_vals);
    merge_odd_continue<R_LEN, false>(R_vals);
}

// 4. Odd-size sort（Algorithm 2）
//    对一个 lane-stride register array 做全排序
template <int LEN>
__device__ void sort_odd(float vals[]) {
    if constexpr (LEN > 32) {
        constexpr int HALF = LEN / 2;
        sort_odd<HALF>(vals);
        sort_odd<HALF>(vals + HALF / 32);
        merge_odd<HALF, HALF>(vals, vals + HALF / 32);
    } else if constexpr (LEN == 32) {
        int lane_id = threadIdx.x & 31;
        #pragma unroll
        for (int k = 2; k <= 32; k <<= 1) {
            #pragma unroll
            for (int stride = k >> 1; stride >= 1; stride >>= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFF, vals[0], stride);
                bool ascending = ((lane_id & k) == 0);
                bool is_lower = ((lane_id & stride) == 0);
                if (ascending == is_lower) vals[0] = fminf(vals[0], other);
                else vals[0] = fmaxf(vals[0], other);
            }
        }
    }
}

// 5. WarpSelect（Algorithm 3）
//    K = 实际 top-k 数量（任意正整数）
//    T = 每个 lane 的 thread queue 大小
template <int K, int T>
struct WarpSelect {
    static constexpr int K_PAD = ((K + 31) / 32) * 32;

    float thread_queue[T];            // 升序 lane-stride（[T-1] 最大，用作门槛）
    float warp_queue[K_PAD / 32];     // lane-stride，升序（多余位置保持 +∞）

    __device__ void init() {
        #pragma unroll
        for (int i = 0; i < T; i++) thread_queue[i] = __FLT_MAX__;
        #pragma unroll
        for (int i = 0; i < K_PAD / 32; i++) warp_queue[i] = __FLT_MAX__;
    }

    __device__ void add(float val) {
        bool inserted = (val < thread_queue[T - 1]);
        if (inserted) {
            thread_queue[T - 1] = val;
            #pragma unroll
            for (int i = T - 1; i > 0; i--) {
                if (thread_queue[i] < thread_queue[i - 1]) {
                    float tmp = thread_queue[i];
                    thread_queue[i] = thread_queue[i - 1];
                    thread_queue[i - 1] = tmp;
                }
            }
        }

        if (__any_sync(0xFFFFFFFF, inserted)) {
            float warp_max = __shfl_sync(0xFFFFFFFF, warp_queue[K_PAD / 32 - 1], 31);
            if (__any_sync(0xFFFFFFFF, thread_queue[T - 1] < warp_max)) {
                restore();
            }
        }
    }

    __device__ void restore() {
        sort_odd<32 * T>(thread_queue);
        merge_odd<K_PAD, 32 * T>(warp_queue, thread_queue);
    }

    __device__ void finalize() { restore(); }

    __device__ void write(float* out) {
        int lane_id = threadIdx.x & 31;
        #pragma unroll
        for (int r = 0; r < K_PAD / 32; r++) {
            int idx = r * 32 + lane_id;
            if (idx < K) {
                out[idx] = warp_queue[r];
            }
        }
    }
};

// ===== Kernel =====

template <int K, int T>
__global__ void topk_pass1(const float* __restrict__ input, int n,
                           float* __restrict__ partial) {
    __shared__ float smem[WARPS_PER_BLOCK * K];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    WarpSelect<K, T> sel;
    sel.init();

    for (int i = global_tid; i - lane_id < n; i += stride) {
        float val = (i < n) ? input[i] : __FLT_MAX__;
        sel.add(val);
    }
    sel.finalize();

    // block 级归约：各 warp 结果写入 shared memory
    sel.write(smem + warp_id * K);
    __syncthreads();

    // warp 0 对 block 内所有 warp 的结果做二次选择
    if (warp_id == 0) {
        WarpSelect<K, T> block_sel;
        block_sel.init();

        for (int i = lane_id; i < WARPS_PER_BLOCK * K; i += 32) {
            block_sel.add(smem[i]);
        }
        block_sel.finalize();
        block_sel.write(partial + blockIdx.x * K);
    }
}

template <int K, int T>
__global__ void topk_pass2(const float* __restrict__ partial, int num_elements,
                           float* __restrict__ output) {
    int lane_id = threadIdx.x & 31;
    WarpSelect<K, T> sel;
    sel.init();

    for (int i = lane_id; i < num_elements; i += 32) {
        sel.add(partial[i]);
    }
    sel.finalize();
    sel.write(output);
}

// ===== Host 接口 =====

template <int K>
void topk(const float* d_input, int n, float* d_output) {
    constexpr int T = (K <= 64) ? 2 : (K <= 256) ? 4 : 8;
    int num_blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 128);

    float* d_partial;
    cudaMalloc(&d_partial, num_blocks * K * sizeof(float));

    topk_pass1<K, T><<<num_blocks, BLOCK_SIZE>>>(d_input, n, d_partial);
    topk_pass2<K, T><<<1, 32>>>(d_partial, num_blocks * K, d_output);

    cudaFree(d_partial);
}

// ============================================================
// 结果检查：GPU 输出的 k 个值排序后应与 CPU 一致
// ============================================================
template <int K>
bool check_result(const float* h_data, int n, const float* gpu_vals) {
    float* cpu_sorted = new float[n];
    memcpy(cpu_sorted, h_data, n * sizeof(float));
    std::partial_sort(cpu_sorted, cpu_sorted + K, cpu_sorted + n);

    float* gpu_sorted = new float[K];
    memcpy(gpu_sorted, gpu_vals, K * sizeof(float));
    std::sort(gpu_sorted, gpu_sorted + K);

    int error_count = 0;
    for (int i = 0; i < K; i++) {
        if (fabsf(cpu_sorted[i] - gpu_sorted[i]) > 1e-6f) {
            if (error_count < 5) {
                fprintf(stderr, "  Mismatch at rank %d: CPU = %.6f, GPU = %.6f\n",
                        i, cpu_sorted[i], gpu_sorted[i]);
            }
            error_count++;
        }
    }

    delete[] cpu_sorted;
    delete[] gpu_sorted;

    if (error_count > 0) {
        fprintf(stderr, "  Total errors: %d out of %d\n", error_count, K);
        return false;
    }
    return true;
}

// ============================================================
// 正确性测试
// ============================================================
bool test_correctness() {
    printf("=== Correctness Test (WarpSelect TopK) ===\n\n");

    constexpr int K = 64;
    struct TestCase { int n; };
    TestCase cases[] = {
        {1024},
        {10000},
        {100000},
        {1000000},
        {10000000},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    srand(42);

    for (int t = 0; t < num_cases; t++) {
        int n = cases[t].n;
        int num_blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 128);
        printf("  N = %-10d  K = %-6d  Blocks = %-5d ... ", n, K, num_blocks);

        float* h_data = new float[n];
        for (int i = 0; i < n; i++) h_data[i] = (float)rand() / RAND_MAX;

        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_data, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

        topk<K>(d_input, n, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());

        float* h_gpu_vals = new float[K];
        CUDA_CHECK(cudaMemcpy(h_gpu_vals, d_output, K * sizeof(float), cudaMemcpyDeviceToHost));

        bool pass = check_result<K>(h_data, n, h_gpu_vals);
        printf("%s\n", pass ? "PASSED" : "FAILED");

        delete[] h_data;
        delete[] h_gpu_vals;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        if (!pass) return false;
    }

    printf("\n  All correctness tests PASSED\n\n");
    return true;
}

// ============================================================
// 性能测试
// ============================================================
void test_performance() {
    printf("=== Performance Test (WarpSelect TopK) ===\n");

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    printf("Block Size: %d threads, SMs: %d, Grid cap: 128\n\n", BLOCK_SIZE, num_sms);

    printf("┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ %-12s │ %-10s │ %-8s │ %-12s │ %-12s │ %-12s │\n",
           "Data Size", "Elements", "Blocks", "Time (ms)", "Bandwidth", "Throughput");
    printf("│              │            │          │              │   (GB/s)     │ (Melem/s)    │\n");
    printf("├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤\n");

    constexpr int K = 64;
    constexpr int T = 2;

    struct TestCase { int n; };
    TestCase cases[] = {
        {100000000},
        {500000000},
        {1000000000},
        {2000000000},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    for (int t = 0; t < num_cases; t++) {
        int n = cases[t].n;
        int num_blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 128);

        float* h_data = new float[n];
        for (int i = 0; i < n; i++) h_data[i] = (float)rand() / RAND_MAX;

        float *d_input, *d_output, *d_partial;
        CUDA_CHECK(cudaMalloc(&d_input, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * K * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_data, (size_t)n * sizeof(float), cudaMemcpyHostToDevice));

        // 预热
        for (int i = 0; i < 5; i++) {
            topk_pass1<K, T><<<num_blocks, BLOCK_SIZE>>>(d_input, n, d_partial);
            topk_pass2<K, T><<<1, 32>>>(d_partial, num_blocks * K, d_output);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        const int num_iter = (n <= 10000000) ? 50 : 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iter; i++) {
            topk_pass1<K, T><<<num_blocks, BLOCK_SIZE>>>(d_input, n, d_partial);
            topk_pass2<K, T><<<1, 32>>>(d_partial, num_blocks * K, d_output);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iter;

        // WarpSelect 只读一遍输入数据
        float total_bytes = (float)n * sizeof(float);
        float bw_gb_s = (total_bytes / (avg_ms / 1000.0f)) / 1.0e9f;
        float throughput = n / (avg_ms * 1000.0f);

        char size_str[20];
        if (n >= 1000000000) {
            snprintf(size_str, sizeof(size_str), "%.1f GB", n * 4.0 / 1e9);
        } else if (n >= 1000000) {
            snprintf(size_str, sizeof(size_str), "%.0f MB", n * 4.0 / 1e6);
        } else {
            snprintf(size_str, sizeof(size_str), "%.0f KB", n * 4.0 / 1e3);
        }

        char bw_str[20];
        if (bw_gb_s >= 1000.0f) {
            snprintf(bw_str, sizeof(bw_str), "%.2f TB/s", bw_gb_s / 1000.0f);
        } else {
            snprintf(bw_str, sizeof(bw_str), "%.2f GB/s", bw_gb_s);
        }

        printf("│ %-12s │ %-10d │ %-8d │ %-12.3f │ %-12s │ %-12.2f │\n",
               size_str, n, num_blocks, avg_ms, bw_str, throughput);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_partial));
    }

    printf("└──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n");
    printf("\nNotes:\n");
    printf("  • Bandwidth = N * 4B (single pass over input) / kernel time\n");
    printf("  • Throughput = N / kernel_time\n");
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   TopK via WarpSelect (Sorting Network)             ║\n");
    printf("║   Two-Pass: Block Reduce + Final Warp               ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    if (!test_correctness()) {
        printf("Correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }

    test_performance();

    printf("\n All tests completed.\n");
    return 0;
}
