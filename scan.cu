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

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// block内部前缀和scan_then_fan kernel
// ============================================================================
__global__ void block_scan_then_fan(const int * d_in, 
                            int * d_out, 
                            int * block_sum, 
                            int N) {
    __shared__ int s_warp_sums[BLOCK_SIZE / WARP_SIZE];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    // Phase 1：warp内scan并将局部结果存入共享内存
    int val = (idx < N) ? d_in[idx]: 0;
    int warp_sum;
    val = warp_kogge_stone_inclusive_scan(val, warp_sum);

    if (lane_id == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    // Phase 2：warp间scan
    if (warp_id == 0) {
        int warp_sum = (lane_id < blockDim.x / WARP_SIZE) ? s_warp_sums[lane_id] : 0;
        int dummy;
        warp_sum = warp_kogge_stone_inclusive_scan(warp_sum, dummy);
        if (lane_id < blockDim.x / WARP_SIZE) {
            s_warp_sums[lane_id] = warp_sum;
        }
    }

    __syncthreads();

    // Phase 3：warp内加上前缀和, warp0除外
    if (warp_id > 0) {
        val += s_warp_sums[warp_id - 1];
    }

    // 写回全局内存
    if (idx < N) {
        d_out[idx] = val;
    }

    // 存储整个 block 的总和（用于跨 block 前缀和计算）
    if (idx == min(N - 1, (blockIdx.x + 1) * blockDim.x - 1)) {
        block_sum[blockIdx.x] = val;
    }
}

// ============================================================================
// block之间前缀和scan kernel
// ============================================================================
__global__ void block_sums_scan(int* d_block_sums,
                                    int num_blocks) {
    __shared__ int s_warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Phase 1：warp内scan并将局部结果存入共享内存
    int val = (idx < num_blocks) ? d_block_sums[idx] : 0;
    int warp_sum;
    int inclusive = warp_kogge_stone_inclusive_scan(val, warp_sum);
    
    if (lane == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Phase 2：warp间scan
    if (warp_id == 0) {
        int ws_val = (lane < blockDim.x / WARP_SIZE) ? s_warp_sums[lane] : 0;
        int dummy;
        int ws_inclusive = warp_kogge_stone_inclusive_scan(ws_val, dummy);
        if (lane < blockDim.x / WARP_SIZE) {
            s_warp_sums[lane] = ws_inclusive;
        }
    }
    __syncthreads();
    
    // Phase 3：warp内加上前缀和, warp0除外
    if (warp_id > 0) {
        inclusive += s_warp_sums[warp_id - 1];
    }
    
    // 写回结果
    if (idx < num_blocks) {
        d_block_sums[idx] = inclusive;
    }
}


// ============================================================================
//  全局 Inclusive Scan - Propagate Phase  
//  将 block 前缀加到各自 block 的元素上
// ============================================================================
__global__ void add_block_prefix(int* d_out,
                                    const int* d_block_prefix,
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // 当前 block 的前缀是前一个 block 的 inclusive scan 结果
        int prefix = (blockIdx.x > 0) ? d_block_prefix[blockIdx.x - 1] : 0;
        d_out[idx] += prefix;
    }
}

// ============================================================================
//  递归全局 Inclusive Scan
// ============================================================================
void global_inclusive_scan_recursive(int* d_in, int* d_out, int N) {
    if (N <= 0) return;
    
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 基础情况：只有一个 block，直接做 block scan
    if (num_blocks == 1) {
        int* d_dummy;
        CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(int)));
        block_scan_then_fan<<<1, BLOCK_SIZE>>>(d_in, d_out, d_dummy, N);
        CUDA_CHECK(cudaFree(d_dummy));
        return;
    }
    
    // 分配临时存储：block 局部和
    int* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    
    // Phase 1: 每个 block 做内部 scan，输出 block 局部前缀和
    block_scan_then_fan<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 2: 对 block 局部前缀和做 inclusive scan
    // 计算需要多少 block 来处理 block_sums
    int num_scan_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (num_scan_blocks == 1) {
        // block sums 数量少，一个 block 就能处理
        block_sums_scan<<<1, BLOCK_SIZE>>>(d_block_sums, num_blocks);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // block sums 数量多，需要递归处理
        // 复制 d_block_sums 到 d_block_sums_scan，作为输入
        int* d_block_sums_scan;
        CUDA_CHECK(cudaMalloc(&d_block_sums_scan, num_blocks * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_block_sums_scan, d_block_sums, 
                             num_blocks * sizeof(int), cudaMemcpyDeviceToDevice));
        
        // 递归地对 block sums 做 inclusive scan
        global_inclusive_scan_recursive(d_block_sums_scan, d_block_sums, num_blocks);
        
        CUDA_CHECK(cudaFree(d_block_sums_scan));
    }
    
    // Phase 3: 将 block 全局前缀加到各个 block 的结果上
    add_block_prefix<<<num_blocks, BLOCK_SIZE>>>(d_out, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFree(d_block_sums));
}

__global__ void single_pass_scan(const int* d_in, int* d_out, int N,
                                 int* g_counter,
                                 volatile int* g_status,
                                 volatile int* g_partial,
                                 volatile int* g_prefix) {
    __shared__ int s_my_id;
    __shared__ int s_prefix_sum;
    __shared__ int warp_prefix_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;
    
    // 抢id，表示负责哪一块数据
    if (tid == 0) {
        s_my_id = atomicAdd(g_counter, 1);
    }
    __syncthreads();

    int my_id = s_my_id;
    int global_tid = my_id * blockDim.x + tid;

    int val = (global_tid < N) ? d_in[global_tid] : 0;
    int warp_sum;
    val = warp_kogge_stone_inclusive_scan(val, warp_sum);
    if (lane == 0) {
        warp_prefix_sums[warp] = warp_sum;
    }
    __syncthreads();
    if (warp == 0) {
        int ws = (lane < blockDim.x / WARP_SIZE) ? warp_prefix_sums[lane] : 0;
        int dummy;
        ws = warp_kogge_stone_inclusive_scan(ws, dummy);
        if (lane < blockDim.x / WARP_SIZE) {
            warp_prefix_sums[lane] = ws;
        }
    }
    __syncthreads();
    if (warp > 0) {
        val += warp_prefix_sums[warp - 1];
    }

    int num_warps = blockDim.x / WARP_SIZE;
    int block_total = warp_prefix_sums[num_warps - 1];

    // 最后一个warp：发布block局部和 + 32线程并行lookback
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

    if (global_tid < N) {
        d_out[global_tid] = val + s_prefix_sum;
    }
}

// ============================================================================
//  Single-Pass Scan 主机封装
// ============================================================================
void single_pass_inclusive_scan(const int* d_in, int* d_out, int N) {
    if (N <= 0) return;
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

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
//  正确性测试
// ============================================================================
bool test_correctness() {
    printf("=== Correctness Test ===\n");
    
    const int N = 123456789;
    printf("Testing with N = %d\n", N);
    
    int *h_in = new int[N];
    int *h_out_cpu = new int[N];
    int *h_out_gpu = new int[N];
    
    // 生成测试数据
    srand(58);
    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 100;
    }
    
    // CPU 参考计算
    cpu_inclusive_scan(h_in, h_out_cpu, N);
    
    // GPU 计算
    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
    
    global_inclusive_scan_recursive(d_in, d_out, N);
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
//  性能测试
// ============================================================================
void test_performance() {
    printf("=== Performance Test ===\n");
    printf("GPU: RTX 5090, Block Size: %d threads\n", BLOCK_SIZE);
    printf("Strategy: Reduce-then-Scan (Recursive)\n\n");
    
    // 表头
    printf("┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ %-12s │ %-10s │ %-8s │ %-12s │ %-12s │ %-12s │\n", 
           "Data Size", "Elements", "Blocks", "Time (ms)", "Bandwidth", "Throughput");
    printf("│              │            │          │              │   (GB/s)     │ (Melem/s)    │\n");
    printf("├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤\n");
    
    // 测试规模：从 100K 到 2G
    int test_sizes[] = {
        100000,          // 100K
        1000000,         // 1M
        10000000,        // 10M
        100000000,       // 100M
        500000000,       // 500M
        1000000000,      // 1G
        2000000000,      // 2G
    };
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    for (int t = 0; t < num_sizes; t++) {
        int N = test_sizes[t];
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // 分配内存
        int *h_data = new int[N];
        int *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
        
        // 初始化数据
        for (int i = 0; i < N; i++) {
            h_data[i] = rand() % 100;
        }
        CUDA_CHECK(cudaMemcpy(d_in, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
        
        // 预热
        for (int i = 0; i < 10; i++) {
            global_inclusive_scan_recursive(d_in, d_out, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 计时
        const int num_iter = (N <= 10000000) ? 100 : 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iter; i++) {
            global_inclusive_scan_recursive(d_in, d_out, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iter;
        
        // 计算有效带宽（GB/s）：输入 + 输出
        float total_bytes = 2.0f * N * sizeof(int);
        float bw_gb_s = (total_bytes / (avg_ms / 1000.0f)) / 1.0e9f;
        
        // 计算吞吐量（M elements/s）
        float throughput = N / (avg_ms * 1000.0f);
        
        // 格式化数据大小
        char size_str[20];
        if (N >= 1000000000) {
            snprintf(size_str, sizeof(size_str), "%.1f GB", N * sizeof(int) / 1e9);
        } else if (N >= 1000000) {
            snprintf(size_str, sizeof(size_str), "%.0f MB", N * sizeof(int) / 1e6);
        } else {
            snprintf(size_str, sizeof(size_str), "%.0f KB", N * sizeof(int) / 1e3);
        }
        
        // 格式化带宽显示
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
    
    // 表尾
    printf("└──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n");
    
    // 额外说明
    printf("\nNotes:\n");
    printf("  • Bandwidth = (input + output) / kernel time\n");
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
    
    // 正确性测试
    if (!test_correctness()) {
        printf("Recursive correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }
    if (!test_correctness_single_pass()) {
        printf("Single-pass correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }

    // 性能测试
    test_performance();
    printf("\n");
    test_performance_single_pass();

    printf("\n✓ All tests completed.\n");
    return 0;
}

