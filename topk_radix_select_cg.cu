#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace cg = cooperative_groups;

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
constexpr int BITS_PER_PASS = 8;
constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;
constexpr int NUM_PASSES = 32 / BITS_PER_PASS;

// ============================================================
// Warp 层：warp 内 shuffle 归约求和
// ============================================================
__device__ int warp_reduce_sum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================
// Block 层：归约每个线程的局部计数，得到 block 总和
// ============================================================
__device__ int block_reduce_sum(int val, int* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    int total = 0;
    if (warp == 0) {
        int v = (lane < WARPS_PER_BLOCK) ? smem[lane] : 0;
        total = warp_reduce_sum(v);
    }
    __syncthreads();
    return total;
}

// ============================================================
// Grid 级共享状态
// ============================================================
struct RadixState {
    int hist[NUM_BUCKETS];
    unsigned int desired;
    unsigned int desired_mask;
    int remaining_k;
    int write_pos;
};

// ============================================================
// 单 Kernel：cooperative groups grid.sync() 实现跨 block 同步
// ============================================================
__global__ void radix_select_kernel(const unsigned int* data,
                                    int n,
                                    int k,
                                    RadixState* state,
                                    unsigned int* d_output,
                                    int* d_output_idx) {
    __shared__ int smem_hist[NUM_BUCKETS];

    cg::grid_group grid = cg::this_grid();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int desired = 0;
    unsigned int desired_mask = 0;

    for (int pass = 0; pass < NUM_PASSES; ++pass) {
        int shift = 32 - (pass + 1) * BITS_PER_PASS;

        for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x)
            smem_hist[i] = 0;
        __syncthreads();

        for (int i = global_tid; i < n; i += stride) {
            unsigned int val = data[i];
            if ((val & desired_mask) == desired) {
                int digit = (val >> shift) & (NUM_BUCKETS - 1);
                atomicAdd(&smem_hist[digit], 1);
            }
        }
        __syncthreads();

        if (threadIdx.x < NUM_BUCKETS) {
            atomicAdd(&state->hist[threadIdx.x], smem_hist[threadIdx.x]);
        }

        grid.sync();

        if (blockIdx.x == 0 && threadIdx.x == 0) {
            int remaining_k = state->remaining_k;
            int cumsum = 0;
            int chosen_bucket = 0;

            for (int b = NUM_BUCKETS - 1; b >= 0; --b) {
                cumsum += state->hist[b];
                if (cumsum >= remaining_k) {
                    chosen_bucket = b;
                    break;
                }
            }

            int count_above = cumsum - state->hist[chosen_bucket];
            state->remaining_k = remaining_k - count_above;
            state->desired = desired | ((unsigned int)chosen_bucket << shift);
            state->desired_mask = desired_mask | (((unsigned int)(NUM_BUCKETS - 1)) << shift);

            for (int b = 0; b < NUM_BUCKETS; b++)
                state->hist[b] = 0;
        }

        grid.sync();

        desired = state->desired;
        desired_mask = state->desired_mask;
    }

    for (int i = global_tid; i < n; i += stride) {
        unsigned int val = data[i];
        if (val >= desired) {
            int pos = atomicAdd(&state->write_pos, 1);
            if (pos < k) {
                d_output[pos] = val;
                d_output_idx[pos] = i;
            }
        }
    }
}

// ============================================================
// 查询硬件最大可驻留 block 数
// ============================================================
int get_max_grid_size() {
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, radix_select_kernel, BLOCK_SIZE, NUM_BUCKETS * sizeof(int));
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    return max_blocks_per_sm * num_sms;
}

// ============================================================
// Host 接口：封装 kernel launch
// ============================================================
void topk_radix_select(const unsigned int* d_data, int n, int k,
                       unsigned int* d_output, int* d_output_idx) {
    RadixState* d_state;
    RadixState init = {};
    init.remaining_k = k;
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(RadixState)));
    CUDA_CHECK(cudaMemcpy(d_state, &init, sizeof(RadixState), cudaMemcpyHostToDevice));

    int max_grid = get_max_grid_size();
    int grid_size = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid);

    void* args[] = {(void*)&d_data, (void*)&n, (void*)&k,
                    (void*)&d_state, (void*)&d_output, (void*)&d_output_idx};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)radix_select_kernel, dim3(grid_size), dim3(BLOCK_SIZE), args));

    CUDA_CHECK(cudaFree(d_state));
}

// ============================================================
// CPU 参考：排序取 Top-K
// ============================================================
void cpu_topk(const unsigned int* data, int n, int k,
              unsigned int* output, int* output_idx) {
    int* indices = new int[n];
    for (int i = 0; i < n; i++) indices[i] = i;

    std::partial_sort(indices, indices + k, indices + n,
        [&](int a, int b) { return data[a] > data[b]; });

    for (int i = 0; i < k; i++) {
        output[i] = data[indices[i]];
        output_idx[i] = indices[i];
    }
    delete[] indices;
}

// ============================================================
// 结果检查：GPU 输出的 k 个值排序后应与 CPU 一致
// ============================================================
bool check_result(const unsigned int* cpu_vals, const unsigned int* gpu_vals, int k) {
    unsigned int* cpu_sorted = new unsigned int[k];
    unsigned int* gpu_sorted = new unsigned int[k];
    std::copy(cpu_vals, cpu_vals + k, cpu_sorted);
    std::copy(gpu_vals, gpu_vals + k, gpu_sorted);
    std::sort(cpu_sorted, cpu_sorted + k, std::greater<unsigned int>());
    std::sort(gpu_sorted, gpu_sorted + k, std::greater<unsigned int>());

    int error_count = 0;
    for (int i = 0; i < k; i++) {
        if (cpu_sorted[i] != gpu_sorted[i]) {
            if (error_count < 5) {
                fprintf(stderr, "Mismatch at rank %d: CPU = %u, GPU = %u\n",
                        i, cpu_sorted[i], gpu_sorted[i]);
            }
            error_count++;
        }
    }

    delete[] cpu_sorted;
    delete[] gpu_sorted;

    if (error_count > 0) {
        fprintf(stderr, "Total errors: %d out of %d\n", error_count, k);
        return false;
    }
    return true;
}

// ============================================================
// 正确性测试
// ============================================================
bool test_correctness() {
    printf("=== Correctness Test (Radix Select TopK) ===\n");

    int max_grid = get_max_grid_size();
    printf("  Hardware max concurrent blocks (grid cap): %d\n\n", max_grid);

    struct TestCase { int n; int k; };
    TestCase cases[] = {
        {1024, 5},
        {10000, 10},
        {100000, 100},
        {1000000, 256},
        {10000000, 1000},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    srand(42);

    for (int t = 0; t < num_cases; t++) {
        int n = cases[t].n;
        int k = cases[t].k;
        int grid = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid);
        printf("  N = %-10d  K = %-6d  Grid = %-5d ... ", n, k, grid);

        unsigned int* h_data = new unsigned int[n];
        for (int i = 0; i < n; i++) h_data[i] = (unsigned int)rand();

        unsigned int* h_cpu_vals = new unsigned int[k];
        int* h_cpu_idx = new int[k];
        cpu_topk(h_data, n, k, h_cpu_vals, h_cpu_idx);

        unsigned int *d_data, *d_output;
        int* d_output_idx;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_output, k * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_output_idx, k * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

        topk_radix_select(d_data, n, k, d_output, d_output_idx);
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int* h_gpu_vals = new unsigned int[k];
        CUDA_CHECK(cudaMemcpy(h_gpu_vals, d_output, k * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        bool pass = check_result(h_cpu_vals, h_gpu_vals, k);
        printf("%s\n", pass ? "PASSED" : "FAILED");

        delete[] h_data;
        delete[] h_cpu_vals;
        delete[] h_cpu_idx;
        delete[] h_gpu_vals;
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_output_idx));

        if (!pass) return false;
    }

    printf("  All correctness tests PASSED\n\n");
    return true;
}

// ============================================================
// 性能测试
// ============================================================
void test_performance() {
    printf("=== Performance Test (Radix Select TopK) ===\n");

    int max_grid = get_max_grid_size();
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    printf("Block Size: %d threads, SMs: %d, Grid cap: %d\n\n",
           BLOCK_SIZE, num_sms, max_grid);

    printf("┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ %-12s │ %-10s │ %-8s │ %-12s │ %-12s │ %-12s │\n",
           "Data Size", "Elements", "Blocks", "Time (ms)", "Bandwidth", "Throughput");
    printf("│              │            │          │              │   (GB/s)     │ (Melem/s)    │\n");
    printf("├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤\n");

    struct TestCase { int n; int k; };
    TestCase cases[] = {
        {100000000,      1000},
        {500000000,      1000},
        {1000000000,     1000},
        {2000000000,     1000},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    for (int t = 0; t < num_cases; t++) {
        int n = cases[t].n;
        int k = cases[t].k;
        int num_blocks = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, max_grid);

        unsigned int* h_data = new unsigned int[n];
        for (int i = 0; i < n; i++) h_data[i] = (unsigned int)rand();

        unsigned int *d_data, *d_output;
        int* d_output_idx;
        RadixState* d_state;
        CUDA_CHECK(cudaMalloc(&d_data, (size_t)n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_output, k * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_output_idx, k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_state, sizeof(RadixState)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, (size_t)n * sizeof(unsigned int), cudaMemcpyHostToDevice));

        RadixState init = {};
        init.remaining_k = k;

        void* args[] = {(void*)&d_data, (void*)&n, (void*)&k,
                        (void*)&d_state, (void*)&d_output, (void*)&d_output_idx};

        // 预热
        for (int i = 0; i < 5; i++) {
            CUDA_CHECK(cudaMemcpy(d_state, &init, sizeof(RadixState), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)radix_select_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), args));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        const int num_iter = (n <= 10000000) ? 50 : 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iter; i++) {
            CUDA_CHECK(cudaMemcpy(d_state, &init, sizeof(RadixState), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)radix_select_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), args));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iter;

        float total_bytes = (float)n * sizeof(unsigned int);
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
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_output_idx));
        CUDA_CHECK(cudaFree(d_state));
    }

    printf("└──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n");
    printf("\nNotes:\n");
    printf("  • Bandwidth = N * 4B (effective: input data size) / kernel time\n");
    printf("  • Throughput = N / kernel_time\n");
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   TopK via Radix Select (Single Kernel)             ║\n");
    printf("║   Last-Block Decision Pattern                       ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    if (!test_correctness()) {
        printf("Correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }

    test_performance();

    printf("\n All tests completed.\n");
    return 0;
}
