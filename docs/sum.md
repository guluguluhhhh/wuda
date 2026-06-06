# Sum Reduction

## 通用归约框架

四种 kernel 共享同一归约结构：

```
线程局部累加 → warp shuffle reduce → shared memory → block reduce → atomicAdd
```

- **Warp Reduce**：`__shfl_down_sync` 做 5 轮归约，32 线程 → 1 个 warp sum
- **Block Reduce**：每个 warp 的 lane 0 写入 `warp_sums[8]`，warp 0 再做一次 warp reduce
- **跨 Block 归约**：`atomicAdd` 将各 block 结果累加到全局输出

---

## 四种实现策略

### 1. Naive — 一线程一元素

```
grid = ceil(N / 256)    // N=500M → 1,953,125 blocks
每线程读 1 个 float
```

**慢的根因**：

1. **访存指令效率低**：每线程发一条 LDG.32 只搬 4B，相比 float4 的 LDG.128搬 512B，搬运同等数据需要 4 倍指令数，SM 的访存指令发射吞吐成为瓶颈。
2. **Block 过多导致 atomicAdd 竞争**：每个block处理的数据少导致block数量过多，带来原子操作竞争。

### 2. Naive Vec4 — float4 向量化加载

```
grid = ceil(N/4 / 256)    // block 数降为 1/4
每线程读 1 个 float4（16B），本地累加 4 个分量
尾部元素标量处理
```

`reinterpret_cast<const float4*>` 做 128-bit 对齐加载，一条 LDG.128 指令读 16B，warp 一次发 32 × 16B = 512B，减少访存指令数 4 倍。

### 3. Stride — Grid-stride loop

```
grid = min(ceil(N/256), 1024)    // 设置 block 数上限 1024
每线程以 stride = blockDim.x * gridDim.x 循环累加多个元素
```

**为什么用 stride 而不是连续分块**：如果每线程处理连续一段数据，warp 内 32 个线程同时访问的地址相距很远，无法合并访存。Stride 模式下，每次迭代 warp 内 32 线程访问连续 32 个 float，合并为一次 128B 事务。

### 4. Vec4 + Stride

```
每线程以 stride 循环读 float4，本地累加
尾部标量处理
```


---

## Benchmark 分析（RTX 5090, 理论带宽 ~1.79 TB/s）

| Kernel | Blocks (N=500M) | Time (ms) | 带宽 | 峰值占比 |
|---|---|---|---|---|
| Naive | 1,953,125 | 2.455 | 814 GB/s | 45% |
| Naive Vec4 | 488,282 | 1.196 | 1.67 TB/s | **93%** |
| Stride | 1,024 | 1.321 | 1.51 TB/s | 84% |
| Vec4+Stride | 1,024 | 1.210 | 1.65 TB/s | **92%** |

### 关键结论

1. **Vec4 是最大加速因素**：814 GB/s → 1.67 TB/s（2.05 倍），LDG.128 减少 4 倍指令数直接打满带宽。
2. **Stride 收益来自减少 block 数**：195 万 → 1024，降低调度开销和 atomicAdd 竞争。
3. **Vec4+Stride ≈ Naive Vec4**：Vec4 已接近带宽峰值，stride 的额外收益有限。
4. **典型 memory-bound 算子**：归约部分开销可忽略，瓶颈完全在 global memory 读取。
