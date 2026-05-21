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
constexpr int largest_pow2_below(int n) {
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
    if constexpr (LEN <= 32) {
        if constexpr (LEN == 32) {
            #pragma unroll
            for (int stride = 16; stride >= 1; stride >>= 1) {
                warp_compare_swap(vals[0], stride);
            }
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
__device__ void merge_odd(float L_vals[], float R_vals[]);

// 4. Odd-size sort（Algorithm 2）
//    对一个 lane-stride register array 做全排序
template <int LEN>
__device__ void sort_odd(float vals[]);
