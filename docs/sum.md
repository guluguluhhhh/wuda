# Sum Reduction

通用归约框架

四种 kernel 共享同一归约结构：

```
线程局部累加 → warp shuffle reduce → shared memory → block reduce → atomicAdd
```

- **Warp Reduce**：`__shfl_down_sync` 做 5 轮归约，32 线程 → 1 个 warp sum
- **Block Reduce**：每个 warp 的 lane 0 写入 `warp_sums[8]`，warp 0 再做一次 warp reduce
- **跨 Block 归约**：`atomicAdd` 将各 block 结果累加到全局输出

---

四种实现策略

1. Naive — 一线程一元素

```
grid = ceil(N / 256)    // N=500M → 1,953,125 blocks
每线程读 1 个 float
```

**慢的根因**：

1. **访存指令效率低**：每线程发一条 LDG.32 只搬 4B，相比 float4 的 LDG.128搬 512B，搬运同等数据需要 4 倍指令数，SM 的访存指令发射吞吐成为瓶颈。
2. **Block 过多导致 atomicAdd 竞争**：每个block处理的数据少导致block数量过多，带来原子操作竞争。

2. Naive Vec4 — float4 向量化加载

```
grid = ceil(N/4 / 256)    // block 数降为 1/4
每线程读 1 个 float4（16B），本地累加 4 个分量
尾部元素标量处理
```

`reinterpret_cast<const float4*>` 做 128-bit 对齐加载，一条 LDG.128 指令读 16B，warp 一次发 32 × 16B = 512B，减少访存指令数 4 倍。

3. Stride — Grid_stride loop

```
grid = min(ceil(N/256), 1024)    // 设置 block 数上限 1024
每线程以 stride = blockDim.x * gridDim.x 循环累加多个元素
```

**为什么用 stride 而不是连续分块**：如果每线程处理连续一段数据，warp 内 32 个线程同时访问的地址相距很远，无法合并访存。Stride 模式下，每次迭代 warp 内 32 线程访问连续 32 个 float，合并为一次 128B 事务。

4. Vec4 + Stride

```
每线程以 stride 循环读 float4，本地累加
尾部标量处理
```


---

Benchmark 分析（RTX 5090, 理论带宽 ~1.79 TB/s）

| Kernel | Blocks (N=500M) | Time (ms) | 带宽 | 峰值占比 |
|---|---|---|---|---|
| Naive | 1,953,125 | 2.455 | 814 GB/s | 45% |
| Naive Vec4 | 488,282 | 1.196 | 1.67 TB/s | **93%** |
| Stride | 1,024 | 1.321 | 1.51 TB/s | 84% |
| Vec4+Stride | 1,024 | 1.210 | 1.65 TB/s | **92%** |

关键结论

1. **Vec4 是最大加速因素**：814 GB/s → 1.67 TB/s（2.05 倍），LDG.128 减少 4 倍指令数直接打满带宽。
2. **Stride 收益来自减少 block 数**：195 万 → 1024，降低调度开销和 atomicAdd 竞争。
3. **Vec4+Stride ≈ Naive Vec4**：Vec4 已接近带宽峰值，stride 的额外收益有限。
4. **典型 memory-bound 算子**：归约部分开销可忽略，瓶颈完全在 global memory 读取。

---

面试手写版

四个变体**只有 load 部分不同，归约尾部完全一致**，所以白板只写最终版（Vec4+Stride）+ 归约框架即可。

```cpp
// ---- Warp Reduce：shfl_down 5 轮，全 warp 拿到 warp_sum ----
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int d = 16; d >= 1; d >>= 1)          // 16, 8, 4, 2, 1
        val += __shfl_down_sync(0xffffffffu, val, d);
    return val;
}

// ---- Vec4 + Stride（最终版）：线程局部累加 → warp → block → atomicAdd ----
__global__ void sum_vec4_stride(const float* in, float* out, int N) {
    __shared__ float warp_sums[8];             // BLOCK_SIZE / 32
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;       // 注意：gid 只用于首元素，步长是全局的

    // ① 局部累加：grid-stride + float4 向量化
    float local = 0.0f;
    const float4* in4 = reinterpret_cast<const float4*>(in);
    int N4 = N / 4;
    for (int i = gid; i < N4; i += stride) {
        float4 v = in4[i];
        local += v.x + v.y + v.z + v.w;
    }
    if (N4 * 4 + gid < N)                      // 尾部 ≤ 3 个元素，标量处理
        local += in[N4 * 4 + gid];

    // ② warp reduce：32 -> 1
    float wsum = warp_reduce_sum(local);
    if (lane == 0) warp_sums[warp] = wsum;
    __syncthreads();

    // ③ block reduce：warp0 再归约 8 个 warp_sum
    if (warp == 0) {
        float bsum = (lane < 8) ? warp_sums[lane] : 0.0f;
        bsum = warp_reduce_sum(bsum);
        // ④ 跨 block 归约：原子加（block 数只有 1024，竞争不是瓶颈）
        if (lane == 0) atomicAdd(out, bsum);
    }
}

// host：grid 限制 1024，内存够 block 铺满 SM 即可
// grid = min(ceil(N/4 / 256), 1024);
// out 先 cudaMemset 为 0，atomicAdd 累加其上。
```

**四个变体的差异只在这一个 loop 上**（归约尾部完全相同）：

| 变体 | load 循环 | 一句话 |
|---|---|---|
| Naive | `gid < N`，每线程 1 个 float | 4B/条 LDG.32，指令发射成瓶颈 |
| Naive Vec4 | `gid < N/4`，每线程 1 个 float4 | 一条 LDG.128 掰 16B，指令数 ÷4 |
| Stride | `i < N`，`i += stride`，float | 铺满 SM，去掉多余 block |
| Vec4+Stride | `i < N/4`，`i += stride`，float4 | 两者叠加，接近带宽峰值 |

**上面刻意不写、只口述的部分**：

| 略掉的 | 一句话说法 |
|---|---|
| stride 为什么是 `blockDim.x * gridDim.x` | 每次迭代 warp 内 32 线程访问连续 32 个 float4，合并成一条 512B 事务；连续分块会让同 warp 线程地址相隔整段数据，无法合并 |
| float4 对齐 | `reinterpret_cast<const float4*>` 要求 16B 对齐，`cudaMalloc` 保证；尾部靠 `N4*4 + gid < N` 标量补齐（最多 3 个） |
| 归约为什么不优化 | memory-bound：归约只碰一次数据、耗时占比 ~1%，shfl_down 5 轮已足够 |
| 为什么 `grid=1024` | block 数 = 铺满 SM 的整数倍即可，多余 block 只增加 atomicAdd 竞争和调度开销 |
| atomicAdd 竞争 | 一个 block 一次 atomicAdd、共 1024 次，相对 500M 次访存可忽略；`out` 需预先清零 |

记忆主线：**归约框架固定不变（局部累加 → warp shuffle → smem → block → atomicAdd），变体只改 load 循环**；sum 是典型 memory-bound，性能差异全部来自访存指令密度和 block 数。
