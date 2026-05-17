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
// Warp内部前缀和Kogge-Stone inclusive scan
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
// Block exclusive scan: 输入每线程一个 val，返回 exclusive prefix + block_total
// ============================================================================
__device__ int block_exclusive_scan(cg::thread_block& block,
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

    int exclusive = thread_inclusive - val;
    if (warp.meta_group_rank() > 0) {
        exclusive += warps[warp.meta_group_rank() - 1];
    }
    return exclusive;
}

// ============================================================================
// Grid-sync scan: 三 pass 在一个 kernel 内完成
// ============================================================================
__global__ void grid_sync_scan(const int* d_in, int* d_out, int N,
                                int* block_sums) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    __shared__ int s_items[BLOCK_SIZE * SMEM_STRIDE];

    int tid = block.thread_rank();
    int bid = blockIdx.x;
    int block_base = bid * TILE_SIZE;

    // ===== Pass 1: 每个 block 做局部 inclusive scan =====

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
    block.sync();

    // padded smem → Blocked read to registers
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
    int exclusive_prefix = block_exclusive_scan(block, warp, items_sum, block_total);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] += exclusive_prefix;
    }

    // 写出 block_total
    if (tid == 0) {
        block_sums[bid] = block_total;
    }

    // ===== 全局同步：等所有 block 完成 pass 1 =====
    grid.sync();

    // ===== Pass 2: block 0 对 block_sums 做 inclusive scan =====
    int num_blocks = gridDim.x;
    if (bid == 0) {
        for (int base = 0; base < num_blocks; base += BLOCK_SIZE) {
            int idx = base + tid;
            int val = (idx < num_blocks) ? block_sums[idx] : 0;

            int bt;
            int exc = block_exclusive_scan(block, warp, val, bt);

            if (idx < num_blocks) {
                // inclusive = exclusive + val，再加上前面 chunk 的累计
                block_sums[idx] = exc + val;
            }
            block.sync();
        }
        // 简化版：block 数不超过 BLOCK_SIZE 时一次搞定
        // 超过时需要多轮，这里用串行 fallback
        if (num_blocks > BLOCK_SIZE && tid == 0) {
            for (int i = BLOCK_SIZE; i < num_blocks; i++) {
                block_sums[i] += block_sums[i - 1];
            }
        }
    }

    // ===== 全局同步：等 pass 2 完成 =====
    grid.sync();

    // ===== Pass 3: 每个 block 加上全局前缀 =====
    int global_prefix = (bid > 0) ? block_sums[bid - 1] : 0;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] += global_prefix;
    }

    // Blocked → padded smem → int4 向量化 store to global
    block.sync();
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        s_items[tid * SMEM_STRIDE + i] = items[i];
    }
    block.sync();
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
//  主机封装
// ============================================================================
void grid_sync_inclusive_scan(const int* d_in, int* d_out, int N) {
    if (N <= 0) return;
    int tile_size = BLOCK_SIZE * ITEMS_PER_THREAD;

    // 查询最大可同时驻留的 block 数
    int max_blocks_per_sm = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, grid_sync_scan, BLOCK_SIZE, 0));
    int max_blocks = max_blocks_per_sm * prop.multiProcessorCount;

    int needed_blocks = (N + tile_size - 1) / tile_size;
    int num_blocks = std::min(needed_blocks, max_blocks);

    // 如果 block 数受限，需要每个 block 处理多个 tile
    if (num_blocks < needed_blocks) {
        printf("Warning: need %d blocks but can only co-resident %d, "
               "data too large for grid-sync approach\n", needed_blocks, num_blocks);
        return;
    }

    int* d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));

    void* args[] = {(void*)&d_in, (void*)&d_out, (void*)&N, (void*)&d_block_sums};
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
    printf("=== Correctness Test (Grid-Sync CG) ===\n");

    // grid-sync 能处理的数据量有限，用较小的 N 测试
    const int N = 1000000;
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

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool pass = check_result(h_out_cpu, h_out_gpu, N);

    if (pass) {
        printf("PASSED\n\n");
    } else {
        printf("FAILED\n\n");
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
    printf("=== Performance Test (Grid-Sync CG) ===\n");

    // 查询最大 block 数
    int max_blocks_per_sm = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, grid_sync_scan, BLOCK_SIZE, 0));
    int max_blocks = max_blocks_per_sm * prop.multiProcessorCount;
    int max_elements = max_blocks * TILE_SIZE;

    printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);
    printf("Max co-resident blocks: %d (max_per_sm=%d)\n", max_blocks, max_blocks_per_sm);
    printf("Max elements with grid-sync: %d (~%.1f M)\n\n",
           max_elements, max_elements / 1e6);

    printf("%-14s %-12s %-10s %-14s %-14s\n",
           "Data Size", "Elements", "Blocks", "Time (ms)", "Bandwidth");
    printf("--------------------------------------------------------------\n");

    int test_sizes[] = {100000, 1000000, 10000000, max_elements};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        int N = test_sizes[t];
        int tile_size = BLOCK_SIZE * ITEMS_PER_THREAD;
        int needed_blocks = (N + tile_size - 1) / tile_size;
        if (needed_blocks > max_blocks) {
            printf("%-14d %-12d SKIPPED (needs %d blocks > %d max)\n",
                   N, N, needed_blocks, max_blocks);
            continue;
        }

        int *d_in, *d_out;
        int *h_data = new int[N];
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

        for (int i = 0; i < N; i++) h_data[i] = rand() % 100;
        CUDA_CHECK(cudaMemcpy(d_in, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

        for (int i = 0; i < 10; i++) {
            grid_sync_inclusive_scan(d_in, d_out, N);
        }

        const int num_iter = 100;
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
        float bw_gb = (2.0f * N * sizeof(int) / (avg_ms / 1000.0f)) / 1e9f;

        printf("%-14d %-12d %-10d %-14.3f %.2f GB/s\n",
               N, N, needed_blocks, avg_ms, bw_gb);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        delete[] h_data;
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }
}

// ============================================================================
//  Main
// ============================================================================
int main() {
    printf("=== Cooperative Groups Grid-Sync Scan ===\n\n");

    if (!test_correctness()) {
        printf("Correctness test failed, aborting.\n");
        return EXIT_FAILURE;
    }

    test_performance();

    printf("\nDone.\n");
    return 0;
}
