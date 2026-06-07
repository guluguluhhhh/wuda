# TopK

## topk_radix_select — Radix Select

### 算法思想

不排序，逐 bit 缩小候选范围，确定第 K 大的值。

**核心变量**：

- `desired`：正在逐 bit 构建的第 K 大的值。每轮确定一个 bit，32 轮后 `desired` 就是精确的第 K 大。
- `desired_mask`：记录哪些 bit 已经确定。用 `(val & desired_mask) == desired` 过滤出当前候选集。
- `remaining_k`：在当前候选集中，目标是第几大。每次排除掉一批比目标大的元素时，remaining_k 相应减小。

从最高位（bit 31）开始，每轮统计满足已确定高位的元素中，当前 bit 为 1 的有多少个：
- 若 count ≥ remaining_k：说明第 K 大的数在当前 bit 为 1 的这批里，确定该 bit 为 1
- 若 count < remaining_k：第 K 大在该 bit 为 0 的那批里，remaining_k -= count，该 bit 为 0

32 轮后精确确定第 K 大的值（`desired`），最后收集所有 ≥ desired 的元素。

### V0：基础版本 [`633a251`](https://github.com/guluguluhhhh/wuda/commit/633a251)

每线程处理 1 个元素，每轮用 `__ballot_sync` + `__popc` 统计 warp 内满足条件且当前 bit 为 1 的数量，block reduce 后 atomicAdd 到全局 `state->count`。

block 间同步采用 **Last Block 决策模式**：

```
1. 每个 block 完成计数后 atomicAdd(&block_finished, 1)
2. 最后一个完成的 block（finished == gridDim.x - 1）负责决策：
   - 根据 count 和 remaining_k 更新 desired
   - 清零 count，递增 generation
3. 其他 block 轮询 generation，等待决策完成后进入下一轮
```

**为什么是 Last Block**：只有最后一个 block 能确保全局 count 已完整汇总。

### 优化 1：Grid-stride loop [`8b194c7`](https://github.com/guluguluhhhh/wuda/commit/8b194c7)

基础版本每线程只处理 1 个元素，需要 N/1024 个 block。改为 grid-stride loop：固定 block 数，每线程循环处理多个元素，本地累加 count 后再做一次 block reduce。

减少了 block 数，降低了 atomicAdd 和 block 间同步的压力。

### 优化 2：8-bit 分桶 [`67f1aff`](https://github.com/guluguluhhhh/wuda/commit/67f1aff)

从每轮 1 bit 改为每轮 8 bit，32 轮变 4 轮，数据只需扫描 4 遍。

每轮对当前 8 bit 分成 256 个桶，用 smem 局部直方图 + atomicAdd 汇总到全局 `hist[256]`。Last block 从大到小累计 hist 找到第 K 大落在哪个桶，更新 desired。

代价是需要 256 个桶的 smem atomicAdd（有竞争），但总遍历次数从 32 降到 4。

### Benchmark（RTX 5090, K=1000）

| N | Blocks | Time (ms) | 带宽 |
|---|---|---|---|
| 100M | 170 | 1.292 | 310 GB/s |
| 500M | 170 | 6.094 | 328 GB/s |
| 1G | 170 | 12.122 | 330 GB/s |
| 2G | 170 | 24.125 | 332 GB/s |

### 分析

有效带宽 ~330 GB/s（峰值 18%），但实际 4 轮各扫描一遍全部数据（条件过滤无法跳过读取），有效读取量约 4×N×4B。修正后实际带宽约 1.3 TB/s，接近峰值的 73%。

---

## topk_radix_select_cg — Cooperative Groups 版本

原版 block 间同步逻辑：每个 block 计数完后 `atomicAdd(&block_finished, 1)`，最后一个完成的 block 检测到自己是 last block，负责决策并递增 `generation`；其他 block 用 `while (generation < ...)` 忙等（`generation` 是全局计数器，记录已完成几轮 pass，本质是手写的屏障信号）。需要手写 `block_finished` 计数、last block 判断、`generation` 轮询，逻辑分散且容易出错。

CG 版本直接用 `grid.sync()` 替代整套同步逻辑：所有 block 计数完 → `grid.sync()` → block 0 决策 → `grid.sync()` → 进入下一轮。不需要 `block_finished`、`generation` 字段，代码更直观。

### 性能对比（RTX 5090, K=1000, N=2G）

| 版本 | Time | 带宽 |
|---|---|---|
| radix_select | 24.125 ms | 332 GB/s |
| radix_select_cg | 27.928 ms | 286 GB/s |

CG 版本略慢约 16%。但 radix select 本身就需要全局同步（所有 block 计数完才能决策），`grid.sync()` 替代的正是这个全局屏障，和手写 last block 模式做的是同一件事，所以性能损失不大。与 scan 不同（scan 只有前向依赖，不需要全局屏障），topk 的全局同步是算法固有需求，CG 在这类场景下是合理的选择。
