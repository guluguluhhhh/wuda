# GEMM (Half-Precision Matrix Multiplication)

## 问题定义与任务分解

计算 C[M, N] = A[M, K] × B^T[N, K]（TN layout：A row-major, B row-major 即 B 转置后参与计算）。

### 分层并行结构

```
Grid 级：每个 block 负责 C 上一个 [BlockM, BlockN] 的输出 tile
Block 级：block 内多个 warp 瓜分 tile，每个 warp 负责 [WarpM, WarpN]
Warp 级：warp 内 32 线程协作执行 Tensor Core mma 指令
K 维循环：沿 K 方向分段累加，每段长 WarpK
```

一个 block 的执行流程：

```
for k in range(0, K, WarpK):
    1. Prelogue: 全 block 协同搬运 A[BlockM, WarpK] 和 B[BlockN, WarpK] → smem
    2. MMA:     各 warp 从 smem 加载 fragment, 执行 tensor core 指令累加到寄存器
    3. Sync:    等搬运和计算都完成，进入下一个 K 段
Epilogue: 累加器写回 smem → 全 block 协同写回 global C
```

---

## 实现与优化过程

### 1. Warp MMA 封装 — WMMA API [`b5cf1b2`](https://github.com/guluguluhhhh/wuda/commit/b5cf1b2)

先实现 warp 级计算单元 `WarpHMMA_f16<WarpM, WarpN, WarpK>`：

- 用 nvcuda `wmma::fragment` 管理寄存器中的矩阵碎片
- `ldmatrix`：从 smem 加载 A、B fragment（A row-major，B col-major）
- `forward`：沿 K 方向每 16 元素调一次 `wmma::mma_sync`，fp32 累加
- `stmatrix`：fp32 → fp16 转换后写回 smem

Warp tile 的分法：block 内 `WarpCountM × WarpCountN` 个 warp 网格状瓜分 `[BlockM, BlockN]`，每个 warp 负责 `[WarpM, WarpN] = [BlockM/WarpCountM, BlockN/WarpCountN]`。

### 2. Basic Block GEMM [`b579c7b`](https://github.com/guluguluhhhh/wuda/commit/b579c7b)

组装完整 kernel：

```
配置: TPB=256, BlockM=64, BlockN=64, WarpM=16, WarpN=32, WarpK=32
      WarpCountM=4, WarpCountN=2 (共 8 个 warp)
Grid: dim3{M/BlockM, N/BlockN}
```

- **Prelogue**：全 block 用 `uint4`（16B）向量化拷贝把 A/B 的 K 切片搬到 smem
- **MMA**：各 warp 调用 `forward()` 累加
- **Epilogue**：累加器 → smem → 全 block 向量化写回 global

### 3. Padding 消除 Bank Conflict [`0fbb990`](https://github.com/guluguluhhhh/wuda/commit/0fbb990)

smem 布局从 `[BlockM, WarpK]` 改为 `[BlockM, WarpK + 8]`（PAD=8）。

**原因**：`wmma::load_matrix_sync` 内部多线程同时访问 smem，若 stride 恰好是 bank 数的倍数（32 half = 64B = 32 banks），不同行的同一列落在同一 bank，产生冲突。PAD 8 个 half 使 stride 变为 40，错开 bank 映射。

### 4. Double Buffering — cp.async [`08f9cf2`](https://github.com/guluguluhhhh/wuda/commit/08f9cf2)

基础版每个 K 步都要 `__syncthreads()` 等拷贝完才能算，计算和访存完全串行。

Double buffering 用 `Kstage=2` 段 smem 缓冲 + `cp.async`（SM80+ 异步拷贝指令，GMEM → SMEM 绕过寄存器）：

```
while (load_kidx < t_tile_max || mma_kidx < t_tile_max):
    if load_kidx < max: 发射 cp.async 到 stage[load_kidx % 2]
    if mma_kidx >= 0:   等前一组完成, 消费 stage[mma_kidx % 2]
    load_kidx++, mma_kidx++
```

第 k 步的计算与第 k+1 步的数据搬运重叠执行。`cp.async.wait_group<N>` 精确控制等待哪一组完成。

同时引入 **smem A/C 别名复用**：mainloop 用 smem_A + smem_B，epilogue 用 smem_C，两者生命期不重叠，共享同一块物理 shared memory，节省 smem 用量。

### 5. Block Swizzle [`eeb4061`](https://github.com/guluguluhhhh/wuda/commit/eeb4061)

默认的 `blockIdx.x/y` 线性映射下，相邻 block 可能访问相距很远的 B 列，L2 cache 命中率低。

Block swizzle 重映射 block → tile 的对应关系：N 方向按 `swizzle_stride` 分段，段内 block 访问相邻的 B 列，通过 `gridDim.z` 实现分段。效果：相邻被调度的 block 共享 B 的 L2 cache line。

### 6. MMA PTX 替换 WMMA [`47899bd`](https://github.com/guluguluhhhh/wuda/commit/47899bd)

WMMA API 是黑盒，无法控制寄存器布局。替换为手写 PTX：

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16`：从 smem 直接加载到 mma 所需的寄存器布局（一条指令 4 个 32-bit 寄存器）
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`：两条凑成 m16n16k16

**优势**：
1. 精确控制 smem 地址计算，为后续 swizzle 优化铺路
2. 避免 WMMA 可能的冗余数据搬运
3. 一个 warp tile（如 64×64）拆成多个 16×16 mma 指令，用 `#pragma unroll` 全展开

### 7. Reuse A smem for C [`a3398ad`](https://github.com/guluguluhhhh/wuda/commit/a3398ad)

扩大 tile 为 `BlockM=128, BlockN=128, WarpM=64, WarpN=64`（4 warp × 128 TPB），提升计算密度。smem_C 直接别名到 smem_A 起始地址，`__syncthreads()` 保证 mainloop 结束后才写 epilogue。

### 8. TMA — Tensor Memory Accelerator [`c18753d`](https://github.com/guluguluhhhh/wuda/commit/c18753d)

SM90+ 硬件 DMA 引擎，特点：
- **单线程发起**：thread 0 一条指令启动整块数据搬运，无需全 block 参与
- **mbarrier 同步**：硬件 barrier，TMA 完成时自动 arrive，consumer warp wait
- **2D tile descriptor**：host 端构造 `CUtensorMap`，描述全局 tensor 形状和 tile 大小

```
Host: make_tma_2d_desc(A, M, K, BlockM, WarpK, SWIZZLE_64B)
Device:
  thread 0: mbarrier_arrive_expect_tx → cp_async_bulk_tensor_2d
  all warps: mbarrier_wait → mma
```

相比 cp.async（全 block 协同 + 显式地址计算），TMA 代码更简洁，硬件自动处理 2D 地址和对齐。

### 9. SMEM Swizzle [`2be4d6b`](https://github.com/guluguluhhhh/wuda/commit/2be4d6b)

TMA 硬件支持在写入 smem 时自动做 swizzle 重排。使用 `CU_TENSOR_MAP_SWIZZLE_64B`：

```
一行 = 4 个 atom (每 atom = 16B = 8 halves)
物理 atom = 逻辑 atom XOR ((row >> 1) & 3)
周期 = 8 行
```

**效果**：消除 bank conflict（无需 padding），同时 ldmatrix 地址计算需适配 swizzle 规则 → 新增 `WarpHMMA_f16_sw64` 结构体，`sw_atom()` 函数计算物理 atom 偏移。

### 10. Warp Specialize [`59f3c92`](https://github.com/guluguluhhhh/wuda/commit/59f3c92)

将 block 内线程分为两个角色：

```
Producer warp (1 个): 发 TMA, 等 empty barrier
Consumer warps (4 个): 跑 mma, 等 full barrier
两套 mbarrier: full (producer→consumer), empty (consumer→producer)
```

**解耦流程**：

```
Producer:                         Consumer:
  wait(empty[stage])               wait(full[stage])
  arrive_expect_tx(full[stage])    mma(smem[stage])
  cp_async_bulk(smem[stage])       arrive(empty[stage])
```

Producer 不参与计算，专注搬运；Consumer 不参与搬运，专注计算。两者通过 mbarrier 握手，重叠程度最大化。

TPB 从 128 变为 160（5 warps = 4 consumer + 1 producer）。

---

## Benchmark（RTX 5090, M=N=K, fp16, TFLOPS）

| MNK | cuBLAS | basic | db+big_tile | tma | tma_ws |
|---|---|---|---|---|---|
| 512 | 41.5 | 31.0 (74.6%) | 17.8 (42.8%) | 20.9 (50.3%) | 23.8 (57.4%) |
| 1024 | 127.0 | 100.7 (79.3%) | 77.8 (61.3%) | 110.4 (86.9%) | 124.6 (98.1%) |
| 2048 | 283.8 | 165.9 (58.5%) | 258.8 (91.2%) | 301.6 (106.3%) | 303.1 (106.8%) |
| 4096 | 364.4 | 193.7 (53.1%) | 279.5 (76.7%) | 369.0 (101.2%) | 371.6 (102.0%) |
| 8192 | 400.1 | 170.2 (42.5%) | 325.2 (81.3%) | 401.3 (100.3%) | 392.5 (98.1%) |

（括号内为 cuBLAS 百分比）

### 关键结论

1. **Basic → db+big_tile**：小矩阵反而变慢（tile 过大 occupancy 下降），大矩阵 58% → 91%，double buffering 隐藏访存延迟 + 大 tile 提升计算密度。
2. **db → tma**：TMA + swizzle 消除 bank conflict + 硬件搬运，大矩阵达到 100%+ cuBLAS。
3. **tma → tma_ws**：Warp specialize 在中等矩阵（1024~2048）有显著收益（87% → 98%~107%），大矩阵两者持平（计算已饱和）。
4. **最终 tma/tma_ws 版本在 2048+ 规模稳定达到 cuBLAS 的 100%±5%**，个别点略超（cuBLAS 对特定 shape 未必最优调度）。
5. **小矩阵（512）所有自定义 kernel 都不如 cuBLAS**：block 数少、SM 利用率低、kernel launch 开销占比大。

