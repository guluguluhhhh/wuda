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
        #pragma unroll
        for (int phase = 1; phase <= 16; phase <<= 1) {
            #pragma unroll
            for (int stride = phase; stride >= 1; stride >>= 1) {
                warp_compare_swap(vals[0], stride);
            }
        }
    }
}

// 5. WarpSelect（Algorithm 3）
//    K = top-k 数量（必须是 64 的倍数）
//    T = 每个 lane 的 thread queue 大小
template <int K, int T>
struct WarpSelect {
    float thread_queue[T];        // 升序 lane-stride（[T-1] 最大，用作门槛）
    float warp_queue[K / 32];     // lane-stride，升序

    __device__ void init() {
        #pragma unroll
        for (int i = 0; i < T; i++) thread_queue[i] = __FLT_MAX__;
        #pragma unroll
        for (int i = 0; i < K / 32; i++) warp_queue[i] = __FLT_MAX__;
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
            float warp_max = __shfl_sync(0xFFFFFFFF, warp_queue[K / 32 - 1], 31);
            if (__any_sync(0xFFFFFFFF, thread_queue[T - 1] < warp_max)) {
                restore();
            }
        }
    }

    __device__ void restore() {
        sort_odd<32 * T>(thread_queue);
        merge_odd<K, 32 * T>(warp_queue, thread_queue);
    }

    __device__ void finalize() { restore(); }

    __device__ void write(float* out) {
        int lane_id = threadIdx.x & 31;
        #pragma unroll
        for (int r = 0; r < K / 32; r++) {
            out[r * 32 + lane_id] = warp_queue[r];
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

    for (int i = global_tid; i < n; i += stride) {
        sel.add(input[i]);
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

// ===== 测试 =====

int main() {
    const int N = 1000000;
    constexpr int K = 64;

    float* h_input = new float[N];
    srand(42);
    for (int i = 0; i < N; i++) h_input[i] = (float)rand() / RAND_MAX;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    topk<K>(d_input, N, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_output[K];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, K * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 验证
    std::sort(h_output, h_output + K);
    float* h_sorted = new float[N];
    memcpy(h_sorted, h_input, N * sizeof(float));
    std::partial_sort(h_sorted, h_sorted + K, h_sorted + N);

    printf("GPU top-%d vs CPU top-%d:\n", K, K);
    bool pass = true;
    for (int i = 0; i < K; i++) {
        if (fabsf(h_output[i] - h_sorted[i]) > 1e-6f) {
            printf("  MISMATCH at %d: GPU=%.6f CPU=%.6f\n", i, h_output[i], h_sorted[i]);
            pass = false;
        }
    }
    if (pass) printf("  PASS!\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_input;
    delete[] h_sorted;
    return 0;
}


