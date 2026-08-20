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

// block 内两级归约：warp 内 shfl → 每 warp 写 smem → warp0 再归约 → 广播
template <class Op>
__device__ float block_reduce(float v, float init, Op op) {
    __shared__ float s[32];                 // 最多 32 个 warp
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    v = op(v);                              // warp_reduce_sum / _max
    if (lane == 0) s[warp] = v;
    __syncthreads();
    v = (lane < nwarp) ? s[lane] : init;    // warp0 收集各 warp 结果
    if (warp == 0) v = op(v);
    __syncthreads();                         // 复用 smem 前的保护
    return __shfl_sync(0xffffffff, v, 0);   // 只有 warp0 lane0 有效，广播给全 block
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
    const float* xr = x + (size_t)row * N;
    float* yr = y + (size_t)row * N;

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

对比

| | 归约算子 | 遍数 | 关键陷阱 |
|---|---|---|---|
| RMSNorm | 1 个 sum(x²) | 2（算和 + 缩放） | eps 在 mean 内；fp32 累加 |
| Safe Softmax | max + sum | 3 | 必须减 max |
