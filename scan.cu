#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
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

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define ITEMS_PER_THREAD 32

#define TILE_SIZE (BLOCK_SIZE * ITEMS_PER_THREAD)
#define SMEM_STRIDE (ITEMS_PER_THREAD + 1)

// ============================================================================
// Warp内部前缀和Kogge-Stone inclusive scan kernel
// ============================================================================
__device__ int warp_kogge_stone_inclusive_scan(int val, int& warp_sum) {
    int lane_id = threadIdx.x & 31;
    int mask = 0xffffffff;

    // 步长：1，2，4，8，16
#pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        int other = __shfl_up_sync(mask, val, offset);
        if (lane_id >= offset) {
            val += other;
        }
    }

    warp_sum = __shfl_sync(mask, val, 31);
    return val;
}

__device__ int block_inclusive_scan(int val, int& block_total) {
    __shared__ int warps[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;

    int threads_sum;
    int thread_inclusive = warp_kogge_stone_inclusive_scan(val, threads_sum);

    if (lane == 0) {
        warps[warp] = threads_sum;
    }
    __syncthreads();

    if (warp == 0) {
        int ws = (lane < BLOCK_SIZE / WARP_SIZE) ? warps[lane] : 0;
        int dummy;
        ws = warp_kogge_stone_inclusive_scan(ws, dummy);
        if (lane < BLOCK_SIZE / WARP_SIZE) {
            warps[lane] = ws;
        }
    }
    __syncthreads();

    block_total = warps[BLOCK_SIZE / WARP_SIZE - 1];

    int inclusive = thread_inclusive;
    if (warp > 0) {
        inclusive += warps[warp - 1];
    }
    return inclusive;
}

__global__ void single_pass_scan(const int* d_in, int* d_out, int N,
                                 int* g_counter,
                                 volatile int* g_status,
                                 volatile int* g_partial,
                                 volatile int* g_prefix) {
    __shared__ int s_my_id;
    __shared__ int s_prefix_sum;
    __shared__ int s_items[BLOCK_SIZE * SMEM_STRIDE];

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;

    if (tid == 0) {
        s_my_id = atomicAdd(g_counter, 1);
    }
    __syncthreads();

    int my_id = s_my_id;
    int block_base = my_id * TILE_SIZE;

    // int4 向量化 load: global → padded smem
    if (block_base + TILE_SIZE <= N) {
        const int4* vec_in = (const int4*)(d_in + block_base);
        #pragma unroll
        for (int v = 0; v < ITEMS_PER_THREAD / 4; ++v) {
            int4 data = vec_in[tid + v * BLOCK_SIZE];
            int p = v * 4 * BLOCK_SIZE + 4 * tid;
            int pad = p / ITEMS_PER_THREAD;
            s_items[p + pad + 0] = data.x;
            s_items[p + pad + 1] = data.y;
            s_items[p + pad + 2] = data.z;
            s_items[p + pad + 3] = data.w;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int idx = block_base + tid + i * BLOCK_SIZE;
            int p = tid + i * BLOCK_SIZE;
            s_items[p + p / ITEMS_PER_THREAD] = (idx < N) ? d_in[idx] : 0;
        }
    }
    __syncthreads();

    // padded smem → Blocked read to registers（无 bank conflict）
    int items[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] = s_items[tid * SMEM_STRIDE + i];
    }

    // 线程内串行 inclusive scan
    #pragma unroll
    for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
        items[i] += items[i - 1];
    }

    int items_sum = items[ITEMS_PER_THREAD - 1];
    int block_total;
    int block_inclusive = block_inclusive_scan(items_sum, block_total);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] += block_inclusive - items_sum;
    }

    // 发布 block_total + 32线程并行 lookback
    int num_warps = blockDim.x / WARP_SIZE;

    if (warp == num_warps - 1) {
        if (lane == 0) {
            g_partial[my_id] = block_total;
            __threadfence();
            g_status[my_id] = 1;
        }

        int prefix_sum = 0;
        if (my_id > 0) {
            int look_base = my_id - 1;
            bool done = false;

            while (!done) {
                int look_idx = look_base - lane;
                int status, look_val;

                if (look_idx >= 0) {
                    do {
                        status = g_status[look_idx];
                    } while (status == 0);
                    __threadfence();
                    look_val = (status == 2) ? g_prefix[look_idx] : g_partial[look_idx];
                } else {
                    status = 2;
                    look_val = 0;
                }

                unsigned prefix_mask = __ballot_sync(0xffffffff, status == 2);

                if (prefix_mask != 0) {
                    int first_prefix_lane = __ffs(prefix_mask) - 1;
                    int contribute = (lane <= first_prefix_lane) ? look_val : 0;
                    for (int offset = 16; offset >= 1; offset >>= 1) {
                        contribute += __shfl_down_sync(0xffffffff, contribute, offset);
                    }
                    if (lane == 0) prefix_sum += contribute;
                    done = true;
                } else {
                    int contribute = look_val;
                    for (int offset = 16; offset >= 1; offset >>= 1) {
                        contribute += __shfl_down_sync(0xffffffff, contribute, offset);
                    }
                    if (lane == 0) prefix_sum += contribute;
                    look_base -= WARP_SIZE;
                }
            }
        }

        if (lane == 0) {
            s_prefix_sum = prefix_sum;
            g_prefix[my_id] = prefix_sum + block_total;
            __threadfence();
            g_status[my_id] = 2;
        }
    }
    __syncthreads();

    // 加上 lookback 得到的全局前缀
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] += s_prefix_sum;
    }

    // Blocked → padded smem → int4 向量化 store to global
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        s_items[tid * SMEM_STRIDE + i] = items[i];
    }
    __syncthreads();
    if (block_base + TILE_SIZE <= N) {
        int4* vec_out = (int4*)(d_out + block_base);
        #pragma unroll
        for (int v = 0; v < ITEMS_PER_THREAD / 4; ++v) {
            int p = v * 4 * BLOCK_SIZE + 4 * tid;
            int pad = p / ITEMS_PER_THREAD;
            int4 data;
            data.x = s_items[p + pad + 0];
            data.y = s_items[p + pad + 1];
            data.z = s_items[p + pad + 2];
            data.w = s_items[p + pad + 3];
            vec_out[tid + v * BLOCK_SIZE] = data;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int idx = block_base + tid + i * BLOCK_SIZE;
            if (idx < N) {
                int p = tid + i * BLOCK_SIZE;
                d_out[idx] = s_items[p + p / ITEMS_PER_THREAD];
            }
        }
    }
}

// ============================================================================
//  Single-Pass Scan 主机封装
// ============================================================================
void single_pass_inclusive_scan(const int* d_in, int* d_out, int N) {
    if (N <= 0) return;
    int tile_size = BLOCK_SIZE * ITEMS_PER_THREAD;
    int num_blocks = (N + tile_size - 1) / tile_size;

    int* d_counter;
    int* d_status;
    int* d_partial;
    int* d_prefix;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_status, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix, num_blocks * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_status, 0, num_blocks * sizeof(int)));

    single_pass_scan<<<num_blocks, BLOCK_SIZE>>>(
        d_in, d_out, N, d_counter, d_status, d_partial, d_prefix);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaFree(d_status));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_prefix));
}


// ----------------------------------------------------------------------
//  CPU 参考 inclusive scan
// ----------------------------------------------------------------------
void cpu_inclusive_scan(const int* in, int* out, int N) {
    if (N <= 0) return;
    out[0] = in[0];
    for (int i = 1; i < N; ++i) {
        out[i] = out[i - 1] + in[i];
    }
}

// ============================================================================
//  结果检查
// ============================================================================
bool check_result(const int* cpu, const int* gpu, int N) {
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        if (cpu[i] != gpu[i]) {
            if (error_count < 5) {
                fprintf(stderr, "Mismatch at index %d: CPU = %d, GPU = %d\n",
                        i, cpu[i], gpu[i]);
            }
            error_count++;
        }
    }
    if (error_count > 0) {
        fprintf(stderr, "Total errors: %d out of %d (%.4f%%)\n", 
                error_count, N, 100.0 * error_count / N);
        return false;
    }
    return true;
}

// ============================================================================
//  Single-Pass 正确性测试
// ============================================================================
bool test_correctness_single_pass() {
    printf("=== Correctness Test (Single-Pass) ===\n");

    const int N = 123456789;
    printf("Testing with N = %d\n", N);

    int *h_in = new int[N];
    int *h_out_cpu = new int[N];
    int *h_out_gpu = new int[N];

    srand(58);
    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 100;
    }

    cpu_inclusive_scan(h_in, h_out_cpu, N);

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    single_pass_inclusive_scan(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool pass = check_result(h_out_cpu, h_out_gpu, N);

    if (pass) {
        printf("✓ Correctness test PASSED\n\n");
    } else {
        printf("✗ Correctness test FAILED\n\n");
    }

    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return pass;
}

// ============================================================================
//  Single-Pass 性能测试
// ============================================================================
void test_performance_single_pass() {
    printf("=== Performance Test (Single-Pass) ===\n");
    printf("GPU: RTX 5090, Block Size: %d threads\n", BLOCK_SIZE);
    printf("Strategy: Single-Pass Decoupled Lookback\n\n");

    printf("┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ %-12s │ %-10s │ %-8s │ %-12s │ %-12s │ %-12s │\n",
           "Data Size", "Elements", "Blocks", "Time (ms)", "Bandwidth", "Throughput");
    printf("│              │            │          │              │   (GB/s)     │ (Melem/s)    │\n");
    printf("├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤\n");

    int test_sizes[] = {
        100000,
        1000000,
        10000000,
        100000000,
        500000000,
        1000000000,
        2000000000,
    };
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        int N = test_sizes[t];
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        int *h_data = new int[N];
        int *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

        for (int i = 0; i < N; i++) {
            h_data[i] = rand() % 100;
        }
        CUDA_CHECK(cudaMemcpy(d_in, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

        // 预热
        for (int i = 0; i < 10; i++) {
            single_pass_inclusive_scan(d_in, d_out, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        const int num_iter = (N <= 10000000) ? 100 : 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iter; i++) {
            single_pass_inclusive_scan(d_in, d_out, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iter;

        float total_bytes = 2.0f * N * sizeof(int);
        float bw_gb_s = (total_bytes / (avg_ms / 1000.0f)) / 1.0e9f;
        float throughput = N / (avg_ms * 1000.0f);

        char size_str[20];
        if (N >= 1000000000) {
            snprintf(size_str, sizeof(size_str), "%.1f GB", N * sizeof(int) / 1e9);
        } else if (N >= 1000000) {
            snprintf(size_str, sizeof(size_str), "%.0f MB", N * sizeof(int) / 1e6);
        } else {
            snprintf(size_str, sizeof(size_str), "%.0f KB", N * sizeof(int) / 1e3);
        }

        char bw_str[20];
        if (bw_gb_s >= 1000.0f) {
            snprintf(bw_str, sizeof(bw_str), "%.2f TB/s", bw_gb_s / 1000.0f);
        } else {
            snprintf(bw_str, sizeof(bw_str), "%.2f GB/s", bw_gb_s);
        }

        printf("│ %-12s │ %-10d │ %-8d │ %-12.3f │ %-12s │ %-12.2f │\n",
               size_str, N, num_blocks, avg_ms, bw_str, throughput);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("└──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n");

    printf("\nNotes:\n");
    printf("  • Bandwidth = (input + output) / kernel time\n");
}

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Global Inclusive Scan - Reduce-then-Scan          ║\n");
    printf("║   Recursive Multi-level Implementation              ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    
    if (!test_correctness_single_pass()) {
        printf("Single-pass correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }

    // 性能测试
    test_performance_single_pass();

    printf("\n✓ All tests completed.\n");
    return 0;
}

