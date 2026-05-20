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
__device__ void warp_compare_swap(float &val, int &idx, int stride) {
    float other_val = __shfl_xor_sync(0xFFFFFFFF, val, stride);
    int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, stride);

    // 低位 lane 留 min，高位 lane 留 max
    bool keep_min = ((threadIdx.x & 31) & stride) == 0;
    bool should_swap = keep_min ? (other_val < val) : (other_val > val);
    if (should_swap) {
        val = other_val;
        idx = other_idx;
    }
}

// 2. Odd-size merge（Algorithm 1）
//    合并两个已排序的 lane-stride register array
//    L 和 R 都是升序，合并后整体升序
//    L_len 和 R_len 不要求是 2 的幂
template <int L_LEN, int R_LEN>
__device__ void merge_odd(float L_vals[], int L_indices[],
                          float R_vals[], int R_indices[]);

// 辅助：编译期计算最大的 2^k < n
constexpr int largest_pow2_below(int n) {
    int h = 1;
    while (h < n) h <<= 1;
    return h >> 1;
}

// 辅助：本地 compare-swap（stride >= 32 时，两个值在同一 lane 的不同寄存器中）
__device__ void local_compare_swap(float &a_val, int &a_idx, float &b_val, int &b_idx) {
    if (a_val > b_val) {
        float tv = a_val; a_val = b_val; b_val = tv;
        int ti = a_idx; a_idx = b_idx; b_idx = ti;
    }
}

// 3. merge_odd_continue（Algorithm 1 的递归部分）
//    LEN = 子数组的总元素数（必须是 32 的倍数）
//    IsLeft = true 表示 dummy 在左边（对应论文中 p=left）
//    vals[]/indices[] 是每个 lane 持有的 LEN/32 个寄存器
template <int LEN, bool IsLeft>
__device__ void merge_odd_continue(float vals[], int indices[]) {
    if constexpr (LEN <= 32) {
        // LEN=32: 单个寄存器层，全用 warp shuffle
        // 对 32 元素做 stride=16,8,4,2,1 的 compare-swap
        // XOR butterfly 天然让各子问题并行处理
        if constexpr (LEN == 32) {
            #pragma unroll
            for (int stride = 16; stride >= 1; stride >>= 1) {
                warp_compare_swap(vals[0], indices[0], stride);
            }
        }
    } else {
        // LEN > 32: H >= 32，compare-swap 在同一 lane 的不同寄存器间完成
        constexpr int H = largest_pow2_below(LEN);
        constexpr int REG_STRIDE = H / 32;
        constexpr int NUM_PAIRS = (LEN - H) / 32;

        // compare-swap(x[i], x[i+H]) for i = 0..LEN-H-1
        // 对应：每个 lane 的 reg[r] 和 reg[r + H/32] 比较
        #pragma unroll
        for (int r = 0; r < NUM_PAIRS; r++) {
            local_compare_swap(vals[r], indices[r],
                               vals[r + REG_STRIDE], indices[r + REG_STRIDE]);
        }

        // 递归处理两个子数组
        if constexpr (IsLeft) {
            // 左子数组 [0 : LEN-H], 右子数组 [LEN-H : LEN]
            constexpr int LEFT_LEN = LEN - H;
            constexpr int RIGHT_LEN = H;
            merge_odd_continue<LEFT_LEN, true>(vals, indices);
            merge_odd_continue<RIGHT_LEN, false>(vals + LEFT_LEN / 32, indices + LEFT_LEN / 32);
        } else {
            // 左子数组 [0 : H], 右子数组 [H : LEN]
            constexpr int LEFT_LEN = H;
            constexpr int RIGHT_LEN = LEN - H;
            merge_odd_continue<LEFT_LEN, true>(vals, indices);
            merge_odd_continue<RIGHT_LEN, false>(vals + LEFT_LEN / 32, indices + LEFT_LEN / 32);
        }
    }
}

// 4. Odd-size sort（Algorithm 2）
//    对一个 lane-stride register array 做全排序
template <int LEN>
__device__ void sort_odd(float vals[], int indices[]);
