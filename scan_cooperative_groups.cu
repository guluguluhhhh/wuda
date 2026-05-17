#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
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

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define ITEMS_PER_THREAD 32
#define TILE_SIZE (BLOCK_SIZE * ITEMS_PER_THREAD)
#define SMEM_STRIDE (ITEMS_PER_THREAD + 1)

// ============================================================================
// Warp inclusive scan (Kogge-Stone)
// ============================================================================
__device__ int warp_kogge_stone_inclusive_scan(cg::thread_block_tile<32> warp,
                                               int val, int& warp_sum) {
#pragma unroll
    for (int offset = 1; offset <= 16; offset <<= 1) {
        int other = warp.shfl_up(val, offset);
        if (warp.thread_rank() >= offset) {
            val += other;
        }
    }
    warp_sum = warp.shfl(val, 31);
    return val;
}

// ============================================================================
// Block inclusive scan
// ============================================================================
__device__ int block_inclusive_scan(cg::thread_block& block,
                                    cg::thread_block_tile<32>& warp,
                                    int val, int& block_total) {
    __shared__ int warps[BLOCK_SIZE / WARP_SIZE];

    int threads_sum;
    int thread_inclusive = warp_kogge_stone_inclusive_scan(warp, val, threads_sum);

    if (warp.thread_rank() == 0) {
        warps[warp.meta_group_rank()] = threads_sum;
    }
    block.sync();

    if (warp.meta_group_rank() == 0) {
        int ws = (warp.thread_rank() < warp.meta_group_size()) ?
                  warps[warp.thread_rank()] : 0;
        int dummy;
        ws = warp_kogge_stone_inclusive_scan(warp, ws, dummy);
        if (warp.thread_rank() < warp.meta_group_size()) {
            warps[warp.thread_rank()] = ws;
        }
    }
    block.sync();

    block_total = warps[warp.meta_group_size() - 1];

    int inclusive = thread_inclusive;
    if (warp.meta_group_rank() > 0) {
        inclusive += warps[warp.meta_group_rank() - 1];
    }
    return inclusive;
}

// ============================================================================
// Global scan kernel（三层结构：warp → block → global）
// ============================================================================
__global__ void grid_sync_scan(const int* d_in, int* d_out, int N,
                                int* block_sums, int num_blocks_total) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = block.thread_rank();
    int num_blocks = gridDim.x;

    __shared__ int smem[BLOCK_SIZE * SMEM_STRIDE];

    // ===== Pass 1: 每个 block 处理 TILE_SIZE 个元素 =====
    for (int bid = blockIdx.x; bid < num_blocks_total; bid += num_blocks) {
        int block_base = bid * TILE_SIZE;

        // Striped load: global → smem（合并访问）
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int gidx = block_base + tid + i * BLOCK_SIZE;
            int p = tid + i * BLOCK_SIZE;
            smem[p + p / ITEMS_PER_THREAD] = (gidx < N) ? d_in[gidx] : 0;
        }
        block.sync();

        // Blocked read: smem → registers
        int items[ITEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            items[i] = smem[tid * SMEM_STRIDE + i];
        }

        // 线程内串行 inclusive scan
        #pragma unroll
        for (int i = 1; i < ITEMS_PER_THREAD; ++i) {
            items[i] += items[i - 1];
        }

        // Block 级 scan
        int items_sum = items[ITEMS_PER_THREAD - 1];
        int block_total;
        int block_inclusive = block_inclusive_scan(block, warp, items_sum, block_total);

        // 加上 exclusive prefix
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            items[i] += block_inclusive - items_sum;
        }

        // Blocked write: registers → smem
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            smem[tid * SMEM_STRIDE + i] = items[i];
        }
        block.sync();

        // Striped store: smem → global（合并访问）
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int gidx = block_base + tid + i * BLOCK_SIZE;
            if (gidx < N) {
                int p = tid + i * BLOCK_SIZE;
                d_out[gidx] = smem[p + p / ITEMS_PER_THREAD];
            }
        }

        if (tid == 0) {
            block_sums[bid] = block_total;
        }
        block.sync();
    }

    grid.sync();

    // ===== Pass 2: block 0 对 block_sums 做 inclusive scan =====
    if (blockIdx.x == 0) {
        int carry = 0;
        for (int base = 0; base < num_blocks_total; base += BLOCK_SIZE) {
            int idx = base + tid;
            int val = (idx < num_blocks_total) ? block_sums[idx] : 0;

            int bt;
            int inc = block_inclusive_scan(block, warp, val, bt);

            if (idx < num_blocks_total) {
                block_sums[idx] = carry + inc;
            }
            carry += bt;
            block.sync();
        }
    }

    grid.sync();

    // ===== Pass 3: 每个 block 加上全局前缀 =====
    for (int bid = blockIdx.x; bid < num_blocks_total; bid += num_blocks) {
        if (bid > 0) {
            int prefix = block_sums[bid - 1];
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int gidx = bid * TILE_SIZE + tid + i * BLOCK_SIZE;
                if (gidx < N) {
                    d_out[gidx] += prefix;
                }
            }
        }
        block.sync();
    }
}

// ============================================================================
//  主机封装
// ============================================================================
void grid_sync_inclusive_scan(const int* d_in, int* d_out, int N) {
    if (N <= 0) return;

    int max_blocks_per_sm = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, grid_sync_scan, BLOCK_SIZE, 0));
    int max_blocks = max_blocks_per_sm * prop.multiProcessorCount;

    int num_blocks_total = (N + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks = std::min(num_blocks_total, max_blocks);

    int* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks_total * sizeof(int)));

    void* args[] = {(void*)&d_in, (void*)&d_out, (void*)&N,
                    (void*)&d_block_sums, (void*)&num_blocks_total};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)grid_sync_scan, num_blocks, BLOCK_SIZE, args));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_block_sums));
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
    printf("=== Correctness Test (Grid-Sync Persistent CG) ===\n");

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

    grid_sync_inclusive_scan(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool pass = check_result(h_out_cpu, h_out_gpu, N);

    if (pass) {
        printf("Correctness test PASSED\n\n");
    } else {
        printf("Correctness test FAILED\n\n");
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
    printf("=== Performance Test (Grid-Sync Persistent CG) ===\n");

    int max_blocks_per_sm = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, grid_sync_scan, BLOCK_SIZE, 0));
    int max_blocks = max_blocks_per_sm * prop.multiProcessorCount;

    printf("GPU: %s, Block Size: %d threads\n", prop.name, BLOCK_SIZE);
    printf("Strategy: Grid-Sync Persistent Threads (CG), %d items/thread\n", ITEMS_PER_THREAD);
    printf("Max co-resident blocks: %d (max_per_sm=%d)\n\n", max_blocks, max_blocks_per_sm);

    printf("┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("│ %-12s │ %-10s │ %-8s │ %-12s │ %-12s │ %-12s │\n",
           "Data Size", "Elements", "Tiles", "Time (ms)", "Bandwidth", "Throughput");
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
        int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

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
            grid_sync_inclusive_scan(d_in, d_out, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        const int num_iter = (N <= 10000000) ? 100 : 10;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iter; i++) {
            grid_sync_inclusive_scan(d_in, d_out, N);
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
               size_str, N, num_tiles, avg_ms, bw_str, throughput);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    printf("└──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n");

    printf("\nNotes:\n");
    printf("  • Bandwidth = (input + output) / kernel time\n");
    printf("  • Pass 3 adds extra read+write, actual memory traffic is ~3x\n");
}

// ============================================================================
//  Main
// ============================================================================
int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Global Inclusive Scan - Grid-Sync Persistent CG   ║\n");
    printf("║   Cooperative Groups Implementation                 ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    if (!test_correctness()) {
        printf("Correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }

    test_performance();

    printf("\nAll tests completed.\n");
    return 0;
}
