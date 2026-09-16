# RMSNorm & Softmax

两个都是 **一个 block 处理一行** 的 memory-bound 算子：读一行 → 归约 → 逐元素写回。核心是 block 内两级 reduce（warp shuffle + shared memory）。

```cpp
// 两个 kernel 共用：warp 内 shfl 归约
__device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;   // 全 lane 拿到同一个和（xor 蝶形，不是 down）
}
__device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// block 内两级归约：warp 内 shfl → 每 warp 写 smem → **每个 warp 都再归约一遍**
// （各 warp 读的是同一份 s[0..nwarp-1]，算出来必然相同 → 全 block 拿到同一个值）
// 切忌：不能写 if (warp == 0) v = op(v)，那样只有 warp0 对；__shfl_sync 只能
// 在 warp 内广播，跨不了 warp。而 RMSNorm/Softmax 的缩放阶段**所有线程**都要用。
template <class Op>
__device__ float block_reduce(float v, float init, Op op) {
    __shared__ float s[32];                 // 最多 32 个 warp
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    v = op(v);                              // 第一级：xor 蝶形 → warp 内全 lane 得和
    if (lane == 0) s[warp] = v;
    __syncthreads();
    v = (lane < nwarp) ? s[lane] : init;    // 每个 warp 都读全部 partial
    v = op(v);                              // 第二级：各 warp 冗余归约一遍，结果一致
    __syncthreads();                        // 下一次复用 s[] 前的保护
    return v;                               // 全 block 每个线程都拿到总结果
}
```

---

1. RMSNorm

$$y_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2 + \epsilon}} \cdot \gamma_i$$

比 LayerNorm 省一个均值：**不减均值、无 bias**，只用平方和。

```cpp
// grid = 行数，block = 256/512；一行长 N
__global__ void rmsnorm(const float* x, const float* gamma, float* y,
                        int N, float eps) {
    int row = blockIdx.x;
    const float* xr = x + row * N;
    float* yr = y + row * N;

    // 1) 局部平方和（grid-stride 覆盖 N > blockDim 的情况）
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        ss += xr[i] * xr[i];

    // 2) block 归约 → 全 block 拿到同一个 rms
    ss = block_reduce(ss, 0.f, warp_reduce_sum);
    float rms = rsqrtf(ss / N + eps);       // rsqrt 比 1/sqrt 快

    // 3) 逐元素缩放写回
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        yr[i] = xr[i] * rms * gamma[i];
}
```

要点

- **读两遍 x**：第一遍算平方和，第二遍缩放。N 不大时可把 x 暂存寄存器/smem 省第二遍 HBM 读。
- `eps` 加在 **mean 之内**（`ss/N + eps`），不是加在 sqrt 之外。
- 精度：累加用 **fp32**，即使输入 bf16/fp16 也要先转 fp32 再平方，否则大 N 下平方和溢出/丢精度。
- 归约维 = N（hidden dim），**天然在一个 block/rank 内**，所以 TP 下 per-head norm 不需要跨 rank 通信。

```cpp
// grid = 行数(token 数)，block = 256；一行长 N
__global__ void rmsnorm_quant_i8(const half* x, int8_t* yq, float* ys,
                                 int N, float eps) {
    int row = blockIdx.x;
    const half* xr = x  + row * N;
    int8_t*     yr = yq + row * N;

    // 1) 一遍访存喂两个归约：平方和 + max|x|
    float ss = 0.f, amax = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(xr[i]);        // 累加一律先转 fp32
        ss   = fmaf(v, v, ss);
        amax = fmaxf(amax, fabsf(v));
    }
    ss   = block_reduce(ss,   0.f, warp_reduce_sum);
    amax = block_reduce(amax, 0.f, warp_reduce_max);

    // 2) rstd 是全行相同的正标量 → max|y| = rstd · max|x|
    //    所以不必先算完整个 y 再多读一遍求最大值
    float rstd  = rsqrtf(ss / N + eps);
    float scale = fmaxf(amax * rstd, 1e-12f) / 127.f;   // 防全零行出 scale=0
    float inv   = 1.f / scale;                          // 每行一次除法，可忽略
    if (threadIdx.x == 0) ys[row] = scale;              // 下游反量化要用

    // 3) 第二遍：normalize + quant，中间 fp16 不落 HBM
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int q = __float2int_rn(__half2float(xr[i]) * rstd * inv);
        yr[i] = (int8_t)min(max(q, -127), 127);         // 截到 ±127 保对称
    }
}
```


```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// q: [rows, hidden] INT8，按行对称量化，x = q * scale[row]
// gamma: [hidden] FP16；y: [rows, hidden] FP16
// y = x * rsqrt(mean(x^2) + eps) * gamma

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

// 固定使用 256 threads/block，每个 block 处理一行
__global__ void dequant_rmsnorm(
    const int8_t* __restrict__ q,
    const float* __restrict__ scale,
    const half* __restrict__ gamma,
    half* __restrict__ y,
    int hidden,
    float eps)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const size_t base = (size_t)row * hidden;
    const float s = scale[row];

    __shared__ float partial[8];
    __shared__ float inv_rms;

    float sum = 0.0f;
    for (int col = tid; col < hidden; col += 256) {
        float x = float(q[base + col]) * s;
        sum = fmaf(x, x, sum);
    }

    sum = warp_sum(sum);
    if ((tid & 31) == 0)
        partial[tid >> 5] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < 8) ? partial[tid] : 0.0f;
        sum = warp_sum(sum);
        if (tid == 0)
            inv_rms = rsqrtf(sum / float(hidden) + eps);
    }
    __syncthreads();

    for (int col = tid; col < hidden; col += 256) {
        float x = float(q[base + col]) * s;
        float g = __half2float(gamma[col]);
        y[base + col] = __float2half_rn(x * inv_rms * g);
    }
}

// 调用：rows > 0，hidden > 0
// dequant_rmsnorm<<<rows, 256, 0, stream>>>(
//     q, scale, gamma, y, hidden, eps);
```
---

2. Softmax

Safe Softmax（减 max 防溢出）

$$y_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}$$

减 `max` 不改变结果（分子分母同乘 $e^{-\max}$），但避免 $e^{x}$ 上溢。**朴素实现读三遍**：求 max → 求 exp 和 → 归一化。

```cpp
// grid = 行数，block = 256；一行长 N
__global__ void softmax(const float* x, float* y, int N) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * N;
    float* yr = y + (size_t)row * N;

    // 1) 行最大值
    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = block_reduce(m, -INFINITY, warp_reduce_max);

    // 2) Σ exp(x - m)
    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += __expf(xr[i] - m);
    s = block_reduce(s, 0.f, warp_reduce_sum);
    float inv = 1.f / s;

    // 3) 归一化写回
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        yr[i] = __expf(xr[i] - m) * inv;
}
```

要点

- **减 max 是为了数值稳定**，不是为了正确性；不减会 `exp` 溢出成 inf/nan。
- 减 max 不改变结果：分子分母同乘 $e^{-\max}$ 抵消。
- 三遍访存（max → exp 和 → 归一化），N 不大时可把 `exp(x-m)` 暂存省第三遍重算。
- `__expf` 是快速近似指数（SFU 硬件），精度够 softmax 用；要高精度用 `expf`。

---

3. Online Softmax

把「求 max」和「求 exp 和」压到**同一遍扫描**：维护一个 running pair $(m, d)$，$m$ 是已见元素的最大值，$d = \sum_{j \le i} e^{x_j - m}$。来一个新元素 $x$：

$$m' = \max(m, x), \qquad d' = d \cdot e^{m - m'} + e^{x - m'}$$

$d$ 乘上 $e^{m-m'}$ 就是把它从「以旧 $m$ 为基」换到「以新 $m$ 为基」。因为 $m' \ge m$，这个因子恒 $\le 1$，永不上溢。

两个部分结果的合并（归约用的就是这个）：

$$m = \max(m_1, m_2), \qquad d = d_1 e^{m_1 - m} + d_2 e^{m_2 - m}$$

它**可结合、可交换**，所以能塞进任意归约树（shuffle / smem / 跨 block）——FlashAttention 能分块算 attention 就靠这条性质。

(m, d) 的两级归约：要同时归约两个量，复用不了上面的 `block_reduce`，得自己写一套

```cpp
// grid = 行数，block = 256；N > 0，输入为有限值
__global__ void online_softmax(const half* x, half* y, int N) {
    size_t base = (size_t)blockIdx.x * N;
    const half* xr = x + base;
    half* yr = y + base;

    // 1) 每个线程在线更新自己的 (m, s)
    float m = -CUDART_INF_F;
    float s = 0.f;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(xr[i]);
        float next_m = fmaxf(m, v);

        // 最大值变化时，将旧指数和换算到新的基准
        s = s * __expf(m - next_m) + __expf(v - next_m);
        m = next_m;
    }

    // 2) 合并各线程的统计量
    float M = block_reduce(m, -CUDART_INF_F, warp_reduce_max);

    // s 原来以局部 m 为基准，现在统一换算到全行 M
    // 没有读到元素的线程：m = -inf，s = 0，贡献为 0
    float local_s = s * __expf(m - M);
    float S = block_reduce(local_s, 0.f, warp_reduce_sum);

    // 3) 第二遍读取并输出
    float inv_S = 1.f / S;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(xr[i]);
        yr[i] = __float2half_rn(__expf(v - M) * inv_S);
    }
}
```

---

对比

| | 归约算子 | 遍数 | 关键陷阱 |
|---|---|---|---|
| RMSNorm | 1 个 sum(x²) | 2（算和 + 缩放） | eps 在 mean 内；fp32 累加 |
| RMSNorm + quant | sum(x²) + max\|x\| | 2 | max\|y\| = rstd·max\|x\|，别为求 max 再读一遍 |
| dequant + RMSNorm | 1 个 sum(q²)，int 域 | 2 | scale 被约掉，只通过 eps 起作用 |
| Safe Softmax | max + sum | 3 | 必须减 max |
| Online Softmax | (m, d) 融合归约 | 2 | init 不能用 -inf；全 mask 行 d=0 出 nan |


4. transpose
```cpp
template<int Bm, int Bn>
__global__ void transposeShared(float* A, float* B, const int M, const int N) {
  __shared__ float tile[Bm][Bn];

  /* -------- 读取阶段 -------- */
  // (r0, c0) 表示 tile 内左上角元素在 matrixA 中的坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // thread y 方向负责：矩阵 A 的行，shared memory 的行
  // thread x 方向负责：矩阵 A 的列，shared memory 的列
  // shared memory 中的元素 tile[y][x] = A[r0 + y, c0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bm; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int r = r0 + y;
    if (r >= M) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bn; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int c = c0 + x;
      if (c < N) {
        tile[y][x] = A[r * N + c];  // 将 A[r0 + y, c0 + x] 写入 tile[y][x]
        //tile[y][x ^ y] = A[r * N + c];
      }
    }
  }

  __syncthreads();  // 同步线程块

/* -------- 写入阶段 -------- */
// (c0, r0) 表示 tile 内左上角元素在 matrixB 中的坐标
// thread y 方向负责：矩阵 B 的行，shared memory 的列
// thread x 方向负责：矩阵 B 的列，shared memory 的行
// shared memory 中的元素 tile[x][y] = B[c0 + y, r0 + x]
#pragma unroll
  for (int y = threadIdx.y; y < Bn; y += blockDim.y) {  // 在 y 方向，每次跨度为 blockDim.y
    int c = c0 + y;
    if (c >= N) break;

#pragma unroll
    for (int x = threadIdx.x; x < Bm; x += blockDim.x) {  // 在 x 方向，每次跨度为 blockDim.x
      int r = r0 + x;
      if (r < M) { B[c * M + r] = tile[x][y]; }  // 将 tile[x][y] 写入 B[c0 + y, r0 + x]
      //if (r < M) { B[c * M + r] = tile[x][x ^ y]; }
    }
  }
}
```


quant
```cpp
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cub/block/block_reduce.cuh>
#include <stdint.h>

template<bool E8M0>
__device__ float encode_scale(float amax, uint8_t& bits) {
    float t = amax > 0.f ? amax / 448.f : 1.f;

    if constexpr (E8M0) {
        int e;
        float f = frexpf(t, &e);        // t = f * 2^e, f in [0.5, 1)
        e -= (f == 0.5f);               // ceil(log2(t))
        bits = static_cast<uint8_t>(e + 127);
        return ldexpf(1.f, e);
    } else {
        __nv_fp8_e4m3 s(t);             // 默认最近偶数舍入
        if (float(s) < t) ++s.__x;      // 正数编码递增，改为向上取整
        bits = s.__x;
        return float(s);
    }
}

// 每个 block 处理一行中的 B 个元素，blockDim.x 必须等于 B
// x: [M, N]；q: [Mp, Np]；sc: [Mp, Sp]，均为 row-major
// Np 为 B 的倍数且 >= N；Mp >= M；Sp >= Np / B
template<int B, bool E8M0>
__global__ void quant_fp8(
    const half* x, uint8_t* q, uint8_t* sc,
    int M, int N, int Np, int Sp)
{
    __shared__ float s;

    int row = blockIdx.y, g = blockIdx.x;
    int col = g * B + threadIdx.x;

    // padding 不读输入；补零也不会改变 amax
    float v = (row < M && col < N)
            ? __half2float(x[(size_t)row * N + col]) : 0.f;
    float amax = block_max(fabsf(v));

    if (threadIdx.x == 0) {
        uint8_t bits;
        s = encode_scale<E8M0>(amax, bits);
        sc[(size_t)row * Sp + g] = bits;
    }
    __syncthreads();

    if (col < Np)
        q[(size_t)row * Np + col] = __nv_fp8_e4m3(v / s).__x;
}

// 例如 B = 32；grid = dim3(Sp, Mp)，block = B
// quant_fp8<32, true ><<<grid, 32>>>(x, q, sc, M, N, Np, Sp); // UE8M0
// quant_fp8<32, false><<<grid, 32>>>(x, q, sc, M, N, Np, Sp); // UE4M3
```