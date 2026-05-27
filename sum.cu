#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <algorithm>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;

// warp 内归约
__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Naive: 一个线程一个元素，无 stride。warp reduce + block reduce + atomicAdd
__global__ void sum_kernel_naive(const float* d_in,
                                 float* d_out,
                                 int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;

    int global_tid = blockIdx.x * blockDim.x + tid;
    float val = (global_tid < N) ? d_in[global_tid] : 0.0f;

    float warp_sum = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            atomicAdd(d_out, block_sum);
        }
    }
}

// Naive Vec4: 无 stride，float4 加载，一个线程处理 4 个元素
__global__ void sum_kernel_vec4(const float* d_in,
                                      float* d_out,
                                      int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;

    int global_tid = blockIdx.x * blockDim.x + tid;

    int N4 = N / 4;
    const float4* d_in4 = reinterpret_cast<const float4*>(d_in);

    float val = 0.0f;
    if (global_tid < N4) {
        float4 v = d_in4[global_tid];
        val = v.x + v.y + v.z + v.w;
    }

    int tail_idx = N4 * 4 + global_tid;
    if (tail_idx < N) {
        val += d_in[tail_idx];
    }

    float warp_sum = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            atomicAdd(d_out, block_sum);
        }
    }
}

// Stride loop + warp reduce + block reduce + atomicAdd
__global__ void sum_kernel_stride(const float* d_in,
                           float* d_out,
                           int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;

    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = global_tid; i < N; i += stride) {
        local_sum += d_in[i];
    }

    float warp_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            atomicAdd(d_out, block_sum);
        }
    }
}

// Vectorized: float4 加载 + stride + warp reduce + block reduce + atomicAdd
__global__ void sum_kernel_stride_vec4(const float* d_in,
                                float* d_out,
                                int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid  = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp = tid / WARP_SIZE;

    int global_tid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // float4 向量化部分
    int N4 = N / 4;
    const float4* d_in4 = reinterpret_cast<const float4*>(d_in);

    float local_sum = 0.0f;
    for (int i = global_tid; i < N4; i += stride) {
        float4 v = d_in4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }

    int tail_idx = N4 * 4 + global_tid;
    if (tail_idx < N) {
        local_sum += d_in[tail_idx];
    }

    float warp_sum = warp_reduce_sum(local_sum);

    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            atomicAdd(d_out, block_sum);
        }
    }
}


// Tensor Core TF32: 两次矩阵乘求 sum
//
// 数据排成 16×16 矩阵 A（256 个 float）
// 右乘一个全1矩阵再左乘一个全1矩阵得到sum
constexpr int TF32_M = 16, TF32_N = 16, TF32_K = 8;
constexpr int TF32_TILE = TF32_M * TF32_N; // 256 floats per tile

__global__ void sum_kernel_tf32_two_wmma(const float* __restrict__ d_in,
                                         float* __restrict__ d_out,
                                         int N) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    // 每个 warp 需要 256 floats 临时空间存放中间矩阵 C
    __shared__ float smem_c[WARPS_PER_BLOCK][TF32_M * TF32_N];

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int total_warps = gridDim.x * WARPS_PER_BLOCK;

    // ==== Step 1: 右乘全1，累积行和 ====
    // A[16×8] × Ones[8×16] → 累加到 C[16×16]
    wmma::fragment<wmma::matrix_a, TF32_M, TF32_N, TF32_K,
                   wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TF32_M, TF32_N, TF32_K,
                   wmma::precision::tf32, wmma::row_major> b_ones;
    wmma::fragment<wmma::accumulator, TF32_M, TF32_N, TF32_K, float> c_frag;

    wmma::fill_fragment(b_ones, 1.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    int num_tiles = N / TF32_TILE;

    for (int t = global_warp_id; t < num_tiles; t += total_warps) {
        // 把 256 个连续 float 视为 16×16 row-major 矩阵，ld=16
        const float* base = d_in + t * TF32_TILE;

        // K=8：先加载左半部分（列 0-7），再加载右半部分（列 8-15）
        wmma::load_matrix_sync(a_frag, base, 16);      // 列 0-7
        wmma::mma_sync(c_frag, a_frag, b_ones, c_frag);

        wmma::load_matrix_sync(a_frag, base + 8, 16);  // 列 8-15
        wmma::mma_sync(c_frag, a_frag, b_ones, c_frag);
    }

    // 此时 c_frag[i][j] = 第 i 行在所有 tile 中的累计行和（每列值相同）

    // ==== Step 2: 左乘全1，把 16 个行和归约为标量 ====
    // D[16×16] = Ones[16×8] × C[8×16]
    // K=8：需要 2 次覆盖 C 的 16 行（上 8 行 + 下 8 行）

    // 先把 C 写到 shared memory（16×16 row-major）
    wmma::store_matrix_sync(smem_c[warp_id], c_frag, 16, wmma::mem_row_major);

    // 准备第二次乘法的 fragments
    wmma::fragment<wmma::matrix_a, TF32_M, TF32_N, TF32_K,
                   wmma::precision::tf32, wmma::row_major> a_ones;
    wmma::fragment<wmma::matrix_b, TF32_M, TF32_N, TF32_K,
                   wmma::precision::tf32, wmma::row_major> c_as_b;
    wmma::fragment<wmma::accumulator, TF32_M, TF32_N, TF32_K, float> d_frag;

    wmma::fill_fragment(a_ones, 1.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    // 加载 C 的上半部分（行 0-7）作为 matrix_b [8×16]，ld=16
    wmma::load_matrix_sync(c_as_b, smem_c[warp_id], 16);
    wmma::mma_sync(d_frag, a_ones, c_as_b, d_frag);

    // 加载 C 的下半部分（行 8-15）作为 matrix_b [8×16]，ld=16
    wmma::load_matrix_sync(c_as_b, smem_c[warp_id] + 8 * 16, 16);
    wmma::mma_sync(d_frag, a_ones, c_as_b, d_frag);

    // d_frag 的每个元素都 = total_sum，直接取一个
    if (lane == 0) {
        atomicAdd(d_out, d_frag.x[0]);
    }
}

// 原始 host 封装
float gpu_sum(const float* h_in, int N) {
    float *d_in = nullptr, *d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    int block = BLOCK_SIZE;
    int grid = std::min((N + block - 1) / block, 1024);

    sum_kernel_stride<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_out = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return h_out;
}
enum KernelType { NAIVE, NAIVE_VEC4, STRIDE, VEC4, TENSOR_CORE };

const char* kernel_name(KernelType k) {
    switch (k) {
        case NAIVE:       return "Naive (1 elem/thread)";
        case NAIVE_VEC4:  return "Naive Vec4 (4 elem/thread)";
        case STRIDE:      return "Stride (N elem/thread)";
        case VEC4:        return "Vec4+Stride (float4 load)";
        case TENSOR_CORE: return "TensorCore TF32 (2x WMMA)";
    }
    return "";
}

void launch_kernel(KernelType k, const float* d_in, float* d_out, int N, int grid, int block) {
    switch (k) {
        case NAIVE:
            sum_kernel_naive<<<grid, block>>>(d_in, d_out, N);
            break;
        case NAIVE_VEC4:
            sum_kernel_vec4<<<grid, block>>>(d_in, d_out, N);
            break;
        case STRIDE:
            sum_kernel_stride<<<grid, block>>>(d_in, d_out, N);
            break;
        case VEC4:
            sum_kernel_stride_vec4<<<grid, block>>>(d_in, d_out, N);
            break;
        case TENSOR_CORE:
            sum_kernel_tf32_two_wmma<<<grid, block>>>(d_in, d_out, N);
            break;
    }
}

int compute_grid(KernelType k, int N, int block) {
    if (k == NAIVE) {
        return (N + block - 1) / block;
    }
    if (k == NAIVE_VEC4) {
        return (N / 4 + block - 1) / block;
    }
    if (k == TENSOR_CORE) {
        int warps_per_block = block / WARP_SIZE;
        int num_tiles = N / TF32_TILE;
        int needed_blocks = (num_tiles + warps_per_block - 1) / warps_per_block;
        return std::min(needed_blocks, 1024);
    }
    return std::min((N + block - 1) / block, 1024);
}

void test_performance() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("=== Performance Test ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Block Size: %d threads\n\n", BLOCK_SIZE);

    long long test_sizes[] = {
        50000000LL,
        100000000LL,
        200000000LL,
        500000000LL
    };
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    KernelType kernels[] = { NAIVE, NAIVE_VEC4, STRIDE, VEC4, TENSOR_CORE };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);

    for (int ki = 0; ki < num_kernels; ki++) {
        KernelType k = kernels[ki];
        printf("--- %s ---\n", kernel_name(k));
        printf("┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐\n");
        printf("│ %-12s │ %-10s │ %-8s │ %-12s │ %-12s │ %-12s │\n",
               "Data Size", "Elements", "Blocks", "Time (ms)", "Bandwidth", "Throughput");
        printf("│              │            │          │              │   (GB/s)     │ (Melem/s)    │\n");
        printf("├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤\n");

        for (int t = 0; t < num_sizes; t++) {
            int N = (int)test_sizes[t];
            int block = BLOCK_SIZE;
            int grid  = compute_grid(k, N, block);

            std::vector<float> h_data(N, 1.0f);

            float *d_in = nullptr, *d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_in, h_data.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

            for (int i = 0; i < 10; i++) {
                CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
                launch_kernel(k, d_in, d_out, N, grid, block);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            const int num_iter = 50;
            float total_ms = 0.0f;

            for (int i = 0; i < num_iter; i++) {
                CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

                CUDA_CHECK(cudaEventRecord(start));
                launch_kernel(k, d_in, d_out, N, grid, block);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));

                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                total_ms += ms;
            }

            float avg_ms = total_ms / num_iter;

            double total_bytes = (double)N * sizeof(float);
            double bw_gb_s = total_bytes / (avg_ms / 1000.0) / 1.0e9;
            double throughput = N / (avg_ms * 1000.0);

            char size_str[20];
            double bytes = (double)N * sizeof(float);
            if (bytes >= 1e9) snprintf(size_str, sizeof(size_str), "%.1f GB", bytes / 1e9);
            else if (bytes >= 1e6) snprintf(size_str, sizeof(size_str), "%.0f MB", bytes / 1e6);
            else snprintf(size_str, sizeof(size_str), "%.0f KB", bytes / 1e3);

            char bw_str[20];
            if (bw_gb_s >= 1000.0) snprintf(bw_str, sizeof(bw_str), "%.2f TB/s", bw_gb_s / 1000.0);
            else snprintf(bw_str, sizeof(bw_str), "%.2f GB/s", bw_gb_s);

            printf("│ %-12s │ %-10d │ %-8d │ %-12.3f │ %-12s │ %-12.2f │\n",
                   size_str, N, grid, avg_ms, bw_str, throughput);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));
        }

        printf("└──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘\n\n");
    }
}


int main() {
    int N = 1 << 20;
    std::vector<float> h_in(N);

    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;
    }

    float gpu_res = gpu_sum(h_in.data(), N);

    float cpu_res = 0.0f;
    for (int i = 0; i < N; ++i) {
        cpu_res += h_in[i];
    }

    printf("GPU sum = %.1f\n", gpu_res);
    printf("CPU sum = %.1f\n\n", cpu_res);

    test_performance();

    return 0;
}

// (base) root@autodl-container-9ce94bbb39-7afd8cfb:~# ./sum 
// GPU sum = 1048576.0
// CPU sum = 1048576.0

// === Performance Test ===
// GPU: NVIDIA GeForce RTX 5090
// Block Size: 256 threads

// --- Naive (1 elem/thread) ---
// ┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐
// │ Data Size    │ Elements   │ Blocks   │ Time (ms)    │ Bandwidth    │ Throughput   │
// │              │            │          │              │   (GB/s)     │ (Melem/s)    │
// ├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤
// │ 200 MB       │ 50000000   │ 195313   │ 0.251        │ 796.19 GB/s  │ 199048.14    │
// │ 400 MB       │ 100000000  │ 390625   │ 0.498        │ 803.78 GB/s  │ 200945.76    │
// │ 800 MB       │ 200000000  │ 781250   │ 0.985        │ 811.84 GB/s  │ 202960.17    │
// │ 2.0 GB       │ 500000000  │ 1953125  │ 2.455        │ 814.51 GB/s  │ 203627.63    │
// └──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘

// --- Naive Vec4 (4 elem/thread) ---
// ┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐
// │ Data Size    │ Elements   │ Blocks   │ Time (ms)    │ Bandwidth    │ Throughput   │
// │              │            │          │              │   (GB/s)     │ (Melem/s)    │
// ├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤
// │ 200 MB       │ 50000000   │ 48829    │ 0.130        │ 1.54 TB/s    │ 384127.54    │
// │ 400 MB       │ 100000000  │ 97657    │ 0.252        │ 1.59 TB/s    │ 396958.53    │
// │ 800 MB       │ 200000000  │ 195313   │ 0.491        │ 1.63 TB/s    │ 407095.61    │
// │ 2.0 GB       │ 500000000  │ 488282   │ 1.196        │ 1.67 TB/s    │ 417958.85    │
// └──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘

// --- Stride (N elem/thread) ---
// ┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐
// │ Data Size    │ Elements   │ Blocks   │ Time (ms)    │ Bandwidth    │ Throughput   │
// │              │            │          │              │   (GB/s)     │ (Melem/s)    │
// ├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤
// │ 200 MB       │ 50000000   │ 1024     │ 0.141        │ 1.42 TB/s    │ 354799.20    │
// │ 400 MB       │ 100000000  │ 1024     │ 0.277        │ 1.45 TB/s    │ 361323.49    │
// │ 800 MB       │ 200000000  │ 1024     │ 0.538        │ 1.49 TB/s    │ 371841.93    │
// │ 2.0 GB       │ 500000000  │ 1024     │ 1.321        │ 1.51 TB/s    │ 378510.30    │
// └──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘

// --- Vec4+Stride (float4 load) ---
// ┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐
// │ Data Size    │ Elements   │ Blocks   │ Time (ms)    │ Bandwidth    │ Throughput   │
// │              │            │          │              │   (GB/s)     │ (Melem/s)    │
// ├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤
// │ 200 MB       │ 50000000   │ 1024     │ 0.127        │ 1.57 TB/s    │ 392951.19    │
// │ 400 MB       │ 100000000  │ 1024     │ 0.248        │ 1.62 TB/s    │ 403769.76    │
// │ 800 MB       │ 200000000  │ 1024     │ 0.488        │ 1.64 TB/s    │ 409470.92    │
// │ 2.0 GB       │ 500000000  │ 1024     │ 1.210        │ 1.65 TB/s    │ 413349.98    │
// └──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘

// --- TensorCore TF32 (2x WMMA) ---
// ┌──────────────┬────────────┬──────────┬──────────────┬──────────────┬──────────────┐
// │ Data Size    │ Elements   │ Blocks   │ Time (ms)    │ Bandwidth    │ Throughput   │
// │              │            │          │              │   (GB/s)     │ (Melem/s)    │
// ├──────────────┼────────────┼──────────┼──────────────┼──────────────┼──────────────┤
// │ 200 MB       │ 50000000   │ 1024     │ 0.126        │ 1.58 TB/s    │ 395900.31    │
// │ 400 MB       │ 100000000  │ 1024     │ 0.246        │ 1.63 TB/s    │ 406259.96    │
// │ 800 MB       │ 200000000  │ 1024     │ 0.487        │ 1.64 TB/s    │ 410709.75    │
// │ 2.0 GB       │ 500000000  │ 1024     │ 1.216        │ 1.64 TB/s    │ 411106.76    │
// └──────────────┴────────────┴──────────┴──────────────┴──────────────┴──────────────┘

