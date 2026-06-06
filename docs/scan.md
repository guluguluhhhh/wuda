# Scan (Prefix Sum)

## scan_basic — Recursive Reduce_then_Scan

### 整体结构

三级递归：block 内 scan → block sum 的 scan → 前缀回写。

```
Phase 1: block_scan_then_fan    每个 block 内部做 inclusive scan，输出 block_sum
Phase 2: 递归 scan block_sum    对所有 block_sum 再做一次全局 scan（block 数 > 256 时递归）
Phase 3: add_block_prefix       每个元素加上前面所有 block 的累计和
```

### Warp 内 Scan — Kogge_Stone

```
offset = 1:   每线程加上左邻 1 位的值
offset = 2:   每线程加上左邻 2 位的值
offset = 4:   ...
offset = 8:
offset = 16:  5 轮后 32 个线程的 inclusive scan 完成
```

`__shfl_up_sync` 从低 lane 向高 lane 传值，5 轮后每个 lane 都持有各自的 inclusive 前缀和。

### Block 内 Scan — Scan_then_Fan

三阶段，以 256 线程（8 warps）为例：

| 阶段 | 操作 | 数据流 |
|---|---|---|
| Phase 1 | 各 warp 内 Kogge-Stone scan | 产出warp内局部前缀和 + 1 个 warp_sum |
| Phase 2 | warp 0 对 8 个 warp_sum 做 scan | 得到 warp 间的前缀和 |
| Phase 3 | warp 1-7 的每个线程加上前一个 warp 的前缀 | 全 block 256 个前缀和完成 |


### 全局 Scan — 递归

```
N 个元素 → ceil(N/256) 个 block
         → 每 block 产出 1 个 block_sum
         → 对 block_sum 数组递归做 scan
         → 将 block 前缀加回各元素
```

递归终止条件：block_sum 数量 ≤ 256，一个 block 就能处理。

### scan_basic的局限

**多次 kernel launch 导致多遍全局读写**：block 间存在数据依赖（每个 block 的前缀依赖前面所有 block 的总和），而同一 kernel 内 block 间无法通信，只能分步 launch：block 内 scan → block_sum scan → 前缀回写，每步都要完整读写一遍全局内存。

加上scan_basic版本没有引入向量化等trick，导致带宽只有42%

---

## scan_cooperative_groups — Grid Sync

用 `grid.sync()` 替代多次 kernel launch，三个 pass 写在同一个 kernel 内，逻辑线性，编程简单。

```
Pass 1: 每个 block 做 tile 内 scan，写出 block_sum
        grid.sync()
Pass 2: block 0 串行扫描所有 block_sum
        grid.sync()
Pass 3: 每个 block 加上全局前缀
```

CG 版本比 basic 还略慢：`grid.sync()` 本质仍是全局屏障，且 `cudaLaunchCooperativeKernel` 要求所有 block 同时驻留（仅 340 个），tile 数远超 block 数时需循环处理，增加读写遍数。

**结论**：CG 优势在于编程简单，但 `grid.sync()` 是全局屏障，不适合 scan 这种只有前向依赖的算法。

---

## scan_single_pass — Decoupled Lookback

参考论文：[Single-pass Parallel Prefix Scan with Decoupled Look-back (NVIDIA, 2016)](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf)

### 算法核心思想

scan_basic 的根本问题是 block 间有数据依赖却无法在同一 kernel 内通信，只能靠 kernel launch 边界做全局同步——**所有** block 的 block_scan 必须全部结束，才能启动 block_sum 的 scan。Decoupled Lookback 的核心洞察：**每个 block 只需向前回溯到一个已算好全局前缀的 block 即可，不需要全局同步所有block。**

算法需要在 global memory 上维护三个数组供 block 间通信：

| 数组 | 作用 |
|---|---|
| `g_status[num_blocks]` | 每个 block 的状态标志（0=未就绪，1=partial，2=prefix） |
| `g_partial[num_blocks]` | 存储每个 block 自身的 block_sum |
| `g_prefix[num_blocks]` | 存储每个 block 的全局前缀和（= 本 block 及之前所有 block 的总和） |

每个 block 经历三种状态：

| status | 含义 |
|---|---|
| 0 | 未就绪，block 还没算完内部 scan |
| 1 (partial) | 内部 scan 完成，block_sum 已写入 `g_partial` |
| 2 (prefix) | 全局前缀已确定，inclusive 前缀和已写入 `g_prefix` |

**Lookback 流程**：

```
1. 完成 block 内 scan，发布 block_sum → g_partial[my_id], status = 1
2. 从 my_id-1 开始向前回溯：
   - 轮询 g_status[look]，等待 ≠ 0
   - 若 status == 2：累加 g_prefix[look]，结束（该 block 已包含更前面所有 block 的和）
   - 若 status == 1：累加 g_partial[look]，继续向前看
3. 发布 g_prefix[my_id] = prefix_sum + block_sum, status = 2
```

**为什么需要 `atomicAdd(g_counter, 1)` 动态分配 block ID**：GPU 不保证 block 按 `blockIdx.x` 顺序调度。这种方式让当前block按数据的顺序去领取自己负责的区域，确保 `my_id` 反映实际执行顺序，前面的 block 一定已在执行，lookback 不会死锁。

**为什么需要 `__threadfence()`**：保证 `g_partial` 写入对其他 block 全局可见后，才将 `g_status` 设为 1。否则其他 block 可能看到 status=1 但读到旧的 partial 值。

### V0：最基础的 single_pass [`233f708`](https://github.com/guluguluhhhh/wuda/commit/233f708)

每线程处理 1 个元素，block 内 scan 与 basic 完全相同。Lookback 由 block 内最后一个有效线程**单线程串行**执行：逐个向前轮询。

这是算法最直接的实现，性能不好但逻辑最清晰。

### 优化 1：32 线程并行 Lookback [`b817c02`](https://github.com/guluguluhhhh/wuda/commit/b817c02)

单线程串行 lookback 是瓶颈：每次只查一个 block，回溯深度大时延迟高。

改为最后一个 warp（32 线程）并行 lookback：每次迭代 32 个 lane 同时查 32 个前驱 block，用 `__ballot_sync` 判断是否有 lane 遇到 status==2，用 `__shfl_down_sync` 归约求和。一次迭代覆盖 32 个 block，回溯速度提升 32 倍。

### 优化 2：Tile — 每线程多元素 [`a36ba6e`](https://github.com/guluguluhhhh/wuda/commit/a36ba6e)

每线程从 1 个元素增加到 `ITEMS_PER_THREAD=8` 个，每个 block 处理 `TILE_SIZE = 256 × 8 = 2048` 个元素。

一个 block 负责输入数组中**连续的** 2048 个元素。问题是怎么让每个线程拿到自己负责的连续 8 个元素，同时保证全局内存合并访存。

直接让 thread 0 读 `d_in[0..7]`、thread 1 读 `d_in[8..15]` 会导致 warp 内线程访问不连续地址，无法合并。所以分两步，**协作加载 + 各取所需**：

```
第一步：所有线程合作，以交错方式把整块 2048 个元素从 global 搬入 smem（合并访存）
  thread 0 读 d_in[base+0],   d_in[base+256], ...  写入 smem[0], smem[256], ...
  thread 1 读 d_in[base+1],   d_in[base+257], ...  写入 smem[1], smem[257], ...
  → warp 内 32 个线程每次读连续 32 个地址，完美合并

第二步：每个线程从 smem 中取自己负责的连续 8 个元素到寄存器，做串行 scan
  thread 0 读 smem[0..7]   等同于 d_in[base+0 ~ base+7]
  thread 1 读 smem[8..15]  等同于 d_in[base+8 ~ base+15]
```

写回时反过来：每线程把结果写入 smem 各自位置，再所有线程合作以交错方式写回 global。

### 优化 3：int4 向量化访存 [`021cbbc`](https://github.com/guluguluhhhh/wuda/commit/021cbbc)

`ITEMS_PER_THREAD` 从 8 增加到 32。用 `int4`（128-bit）做全局内存加载和写回，每条指令搬 16B，访存指令数减少 4 倍。

### 优化 4：Padding 消除 bank conflict [`013662f`](https://github.com/guluguluhhhh/wuda/commit/013662f)

Blocked read 时 `smem[tid * 32 ... tid * 32 + 31]`，同一 warp 内相邻线程起始地址间距 32，刚好是 bank 数，**每次访问全部 32 个线程命中同一 bank**。

解决：`SMEM_STRIDE = 33`（而非 32），每行多填充 1 个 int。相邻线程起始地址间距变为 33，错开 bank 映射。

## Benchmark 对比（RTX 5090, N=2G, 理论带宽 ~1.79 TB/s）
详细bench数据可以看对应文件底部

| 版本 | Time | 带宽 | 峰值占比 |
|---|---|---|---|
| basic | 21.437 ms | 746 GB/s | 42% |
| cooperative_groups | 25.805 ms | 620 GB/s | 35% |
| single_pass | 10.543 ms | 1.52 TB/s | 85% |


