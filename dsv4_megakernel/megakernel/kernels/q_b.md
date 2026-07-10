# wq_b_proj GEMM (tcgen05 / Blackwell) 实现讲解

## 1. 问题定义

`wq_b` 投影是一个 **极扁的 GEMM**：

```
D[M, N] = A[M, K] @ B[N, K]^T
  A = 激活 (activation)   [M, 1536]     bf16
  B = 权重 (weight)       [65536, 1536] bf16     ← 201 MB
  D = 输出                [M, 65536]    fp32
  M ∈ {32, 64, ..., 256} (32 对齐)   K = 1536   N = 65536 (128 heads × 512 dim)
```

形状特征决定了实现策略：
- **M 极小（32~256），N 极大（65536）**，K 中等（1536）。
- **权重 B 是绝对大头**：201 MB，每 byte 只读一次、无复用 → 这是一个 **访存受限 (memory-bound)** 算子。理论下界 ≈ 把权重从 HBM 读一遍的时间（≈34 µs），实现的核心目标就是**把 HBM 带宽喂满**。
- 输出 FP32（M=256 时 67 MB），对大 M 也是可观流量。

当前实现全部 warm 测试 **正确性 PASS（cos_sim ≈ 1.0，max_diff ~5e-7）**，性能：

| M | 延迟 (µs) | 带宽 (GB/s, FP32 输出计) | vs cuBLAS |
|---|---|---|---|
| 32  | 38.4 | 5468 | 85% |
| 64  | 39.3 | 5557 | 88% |
| 128 | 43.0 | 5471 | 92% |
| 256 | 51.4 | 5233 | 86% |

> 说明：cuBLAS 基线输出 **BF16**（输出字节只有我们的一半），而本算子按需求输出 **FP32**，因此单看延迟对 cuBLAS 天然吃亏；按"相同字节数"折算后已非常接近硬件上限。带宽按 `权重 + 激活 + FP32 输出` 全部字节计算。

---

## 2. 整体架构

面向 Blackwell (SM100) 的第五代张量核，采用四条主线：

1. **tcgen05 UMMA + 2SM MMA (`cta_group::2`)**：两个 CTA 组成一个 cluster，协作完成一条 `UMMA_M=256` 的矩阵乘指令，累加器放在专用的 **Tensor Memory (TMEM)**。
2. **Warp specialization（warp 分工）**：一个 block 256 线程分成三类角色（搬运 / 计算 / 写回），异步流水、互不阻塞。
3. **Persistent kernel（持久化）**：只发 `grid = SM 数` 个 block，每个 block 用一个循环吃掉多个 N-tile，消除重复启动与调度开销。
4. **TMA（Tensor Memory Accelerator）**：descriptor 驱动的异步批量搬运，配合 mbarrier 做多级软件流水；输出也走 TMA store。

### 配置（统一用于所有 32 对齐的 M）

```
BLOCK_M = M            (单个 M 块，M 直接作为 tile 尺寸，无 padding)
BLOCK_N = 128          (每个 CTA 负责的 N 宽度)
CLUSTER_BLOCK_N = 256  (一个 cluster 覆盖的 N = 2 × 128 = UMMA_M)
BLOCK_K = 64           NUM_K_TILES = 1536 / 64 = 24
UMMA_M = 128 × 2 = 256 (沿 N)   UMMA_N = M (沿 M)   UMMA_K = 16
cluster = (2,1,1)，kNumMulticast = 2，cluster_n = 2
流水级数 NUM_STAGES: M=32→11, 64→10, 128→8, 256→6 (由 SMEM 预算自动求解)
```

---

## 3. 用到的优化（核心）

### 优化① swap_ab —— 让张量核算力精确匹配真实 M

这是本 kernel 最关键的设计。2SM MMA 的输出行数 `UMMA_M` 固定是 `128 × 2 = 256`。如果直接把这条 256 的轴对齐到问题的 M 维（M 只有 32~64），张量核每条指令要算 256 行、其中大部分是浪费，会把**张量核管线**顶成瓶颈，反而喂不满 HBM。

**swap_ab** 把矩阵乘的 A/B 操作数在 MMA 内部对调，使得：

```
UMMA_M(256) 沿大维度 N → 打满这条大轴
UMMA_N       沿小维度 M = M(真实) → 张量核算力随 M 精确缩小，无浪费
```

于是每条 MMA 只算 `256(N) × M(真实) × 16(K)`，计算量正比于真实 M，kernel 稳定在**访存受限**区间。实测张量核管线利用率降到 ~34%、DRAM 利用率成为头号（~60%），说明瓶颈正确地落在了显存带宽上。

### 优化② 2SM MMA + 权重按 N 切分 —— 权重零冗余读取

一个 cluster 的两个 CTA 协作一条 `cta_group::2` UMMA：**每个 CTA 只加载权重的一半 N（各 128 行），拼成 UMMA_M=256**。整个 kernel 里 201 MB 权重被**恰好读一遍**，没有任何重复读取——这是达到访存下界的前提。

### 优化③ 多级软件流水 (multi-stage pipeline)

SMEM 里为 A/B 各开一个 6~11 级的环形缓冲。生产者持续把后续 K-block 的 TMA 预取进来，让 **多条 TMA 同时在飞**，用流水深度隐藏 HBM 的长延迟。级数按 SMEM 预算自动求最大值（小 M 富余 → 11 级，大 M 紧张 → 6 级）。

### 优化④ Warp specialization + 三条解耦流水

搬运、计算、写回是三条**独立的持久化循环**，通过 mbarrier 解耦并行推进，而不是串行的 load→compute→store。TMA 搬运、张量核计算、TMEM 写回三者在时间上完全重叠。

### 优化⑤ TMEM 累加器双缓冲

TMEM 累加器开 2 份（`NUM_EPI_STAGES=2`）：MMA 在写第 N+1 个 tile 的累加器时，epilogue 可以并行读回第 N 个 tile，**计算与写回重叠**。

### 优化⑥ 无 padding

`BLOCK_M = M` 直接把真实 M 作为 tile 尺寸（M 已 32 对齐、每 CTA 分 M/2 是 16 对齐，满足 MMA/swizzle 约束）。既不浪费激活加载和输出写回带宽，也省掉了对 M 补零的额外 kernel。

### 优化⑦ TMA 128B swizzle + 持久化调度

- A/B/D 三个张量都用 **128B swizzle** 的 TMA descriptor，保证 SMEM 无 bank conflict 的读写（epilogue 实测 bank conflict ≈1.6-way）。
- 持久化调度让 `grid = SM 数`，每个 cluster 循环处理 `num_tiles = N/256 = 256` 个 tile 中属于自己的部分，块间共享 store 流水、零重复启动。

### 优化⑧ TMA L2 promotion 256B —— 让激活常驻 L2 复用

TMA descriptor 的 L2 promotion 粒度设为 **256B**（`CU_TENSOR_MAP_L2_PROMOTION_L2_256B`）。

机理：激活 `[M,1536]`（M=32 时仅 96KB）被**所有 74 个 cluster、每个 cluster 的每个 N-tile 反复读取**。256B 的促进粒度能把这块小激活留在 L2 里跨 cluster 复用，而 128B 粒度留不住、每次都打 DRAM。实测把 L2 命中率从 **4.6% 提到 26.7%**、DRAM 利用率从 58.7% 提到 ~60%，M=32 延迟 40.2 → 38.4 µs。**一行改动、免费收益。**

---

## 4. 两个 CTA 如何协作（2SM 的数据分布）

一条 leader 发起的 `cta_group::2` UMMA 把 cluster 的两个 CTA 视作 `ThrID=2`，数据分布如下（这是正确性的关键）：

- **MMA 的 A 操作数与累加器 (TMEM) 按 M 切分**：CTA0 拿 MMA-M 的 `[0:128]`，CTA1 拿 `[128:256]`，各存各的 SMEM/TMEM。
- **MMA 的 B 操作数按 N/2 切分**，在两个 CTA 的 datapath 间交换，因此**每个 CTA 的 TMEM 都拿到完整的 UMMA_N**。
- **2SM 的 TMA load 是"每个 CTA 各自按自己的坐标加载到自己的 SMEM"**；切分/共享完全由软件传入的坐标决定，硬件只负责把完成字节记到 leader 的 mbarrier。

在 swap 布局下（MMA-A = 权重，MMA-B = 激活），每个 CTA 具体加载：

| 操作数 | rank 0 (leader) | rank 1 (peer) | 在 MMA 中的角色 |
|---|---|---|---|
| 权重 (`desc_B`) | N 行 `[t·256 : +128]` | N 行 `[t·256+128 : +128]` | MMA-A，沿 UMMA_M 按 M 切 128/128 |
| 激活 (`desc_A`) | M 行 `[0 : M/2]` | M 行 `[M/2 : M]` | MMA-B，沿 UMMA_N 按 N/2 切、交换 |

结果：一个 cluster 每个 tile 算出 `256(N) × M` 的转置输出 `Dᵀ`，**CTA0 拥有输出 N 的前半、CTA1 拥有后半**，所以 epilogue 写回时各自按 `cta_rank` 给 N 加偏移，两个 CTA 合起来覆盖完整的 256 列 N。

---

## 5. Warp 分工与流水（256 线程 = 8 warp）

```
warp 0        : TMA 生产者 (producer)         —— 两个 CTA 都跑
warp 1        : MMA 消费者 (仅 leader CTA)
warp 2        : TMEM 分配 (tcgen05.alloc)
warp 4~7 (128): epilogue（转置 + TMA store）   —— 两个 CTA 都跑
```

三条持久化循环通过 mbarrier 环形握手：

1. **生产者 (warp 0)**：对每个 tile 的 24 个 K-block，等 `empty_barrier[stage]`（槽位空闲）→ 发两条 2SM TMA（激活 + 权重）到 SMEM 环形缓冲的 `stage` 槽 → `arrive_and_expect_tx`（leader 登记两个 CTA 到达的字节数）→ 推进流水。

2. **MMA (warp 1, leader)**：等 `full_barrier[stage]`（TMA 到齐）→ warp-shuffle 取出该 stage 描述符 → 对 `BLOCK_K/UMMA_K = 4` 个子步发 `tcgen05.mma.cta_group::2`（累加到 TMEM）→ `umma_arrive` 释放 `empty_barrier`（让生产者复用槽位），末 K 时通知 `tmem_full`。

3. **Epilogue (warp 4~7)**：等 `tmem_full` → 从 TMEM 读回 → 转置写入 SMEM → TMA store 到全局 D → 释放 `tmem_empty`（让 MMA 复用累加器）。

barrier 计数：`full_barriers` init(2)（两个 CTA 到达 + TMA 字节）；`tmem_empty_barriers` init(2 × 128)（两个 CTA 的 128 个 epilogue 线程各 arrive 一次）。

---

## 6. Epilogue：转置 store

swap 布局下 TMEM 里存的是 `Dᵀ`（datapath 轴 = N，列 = M），写回行主序的 `D[M, N]` 需要转置：

- 每个 store 阶段处理 16 行 M（`STORE_BLOCK_M=16`），共 `M/16` 个阶段。
- 用 `tcgen05.ld.32x32b.x8`（每 lane 取 8 个 FP32）从 TMEM 读；硬件按 warp 自动选 datapath 段（warp i 读 N 行 `[i·32 : +32]`）。
- 按 `(col ^ row)` 的 128B swizzle 公式写入 SMEM，完成 N↔M 转置并对齐 TMA store 的 swizzle 布局。
- 4 个 warp 各写一个 32 宽的 N-atom；再由 warp0 发 4 条 `TMA STORE 2D`（box = 16(M)×32(N)，128B swizzle）写回全局。
- store 用满 128 线程（4 warp），SMEM 双缓冲（`NUM_TMA_STORE_STAGES=2`）与 TMA 重叠。

---

## 7. 优化清单速查

| # | 优化 | 作用 |
|---|---|---|
| ① | **swap_ab** | 张量核算力随真实 M 缩放，避免浪费，kernel 稳定在访存受限 |
| ② | **2SM MMA + 权重按 N 切分** | 201 MB 权重恰好读一遍，零冗余 |
| ③ | **多级软件流水 (6~11 级)** | 多条 TMA 在飞，隐藏 HBM 长延迟 |
| ④ | **Warp specialization 三路解耦** | 搬运/计算/写回完全重叠 |
| ⑤ | **TMEM 累加器双缓冲** | 计算与 epilogue 重叠 |
| ⑥ | **无 padding (BLOCK_M=M)** | 不浪费激活加载与输出写回带宽 |
| ⑦ | **128B swizzle + 持久化调度** | 无 bank conflict、零重复启动 |
| ⑧ | **TMA L2 promotion 256B** | 小激活常驻 L2 复用，L2 命中 4.6%→26.7% |

---

## 附：关键文件

- `kernels/wq_b_proj_gemm_tcgen05.cu`   —— kernel + host 启动 + PyTorch 绑定
- `include/wq_b_proj_gemm_tcgen05.cuh`  —— 配置常量、描述符构造、PTX 封装、TMA 封装
- `test/test_wq_b_proj_tcgen05.py`       —— 正确性 + benchmark（JIT 编译，sm_100+）

---

## 8. 融合 vs 不融合（GEMM + per-head RMSNorm）端到端对比

输出统一为 **BF16**。三种实现：

- **纯 GEMM**：本 kernel 只做 GEMM（B300 实测）。
- **融合**：GEMM + per-head RMSNorm 融进一个 kernel。
- **不融合端到端**：本纯 GEMM kernel → 独立 RMSNorm kernel链式串起来计时。

带宽三列**同分子**（`weight + 激活 + 一个 q 输出` 的逻辑最小字节，bf16）→ 可直接横比；只有时间不同。

| M | 纯GEMM(µs) | 融合(µs) | 不融合端到端(µs) | cuBLAS(µs) | 纯GEMM_BW | 融合_BW | 不融合_BW | 更快 |
|---|---|---|---|---|---|---|---|---|
| 32  | 37.5 | 37.9 | 41.9 | 32.4 | 5489 | 5419 | 4911 | **融合** −10% |
| 64  | 37.6 | 41.0 | 43.2 | 34.1 | 5577 | 5116 | 4860 | **融合** −5% |
| 96  | 38.9 | 44.1 | 45.8 | 35.5 | 5512 | 4857 | 4678 | **融合** −4% |
| 128 | 39.5 | 49.2 | 47.5 | 39.4 | 5526 | 4441 | 4596 | 不融合 −3% |
| 160 | 41.0 | 63.6 | 51.5 | 40.8 | 5429 | 3505 | 4328 | 不融合 −19% |
| 192 | 43.0 | 69.6 | 53.7 | 42.2 | 5276 | 3263 | 4225 | 不融合 −23% |
| 224 | 45.1 | 76.0 | 56.9 | 46.7 | 5131 | 3045 | 4069 | 不融合 −25% |
| 256 | 47.8 | 81.8 | 61.2 | 44.6 | 4933 | 2883 | 3853 | 不融合 −25% |

> "更快"列 = 融合 vs 不融合端到端 的延迟对比（百分比为较快一方的领先幅度）。

### 结论

- **纯 GEMM 已达 cuBLAS 水平**：%cuBLAS 86→**99.6%（M=128）→103.7%（M=224）**，带宽 ~5000-5580 GB/s。
- **交叉点在 M≈112**：
  - **小 M（≤96,decode 热点）：融合更快**（省掉 q 的额外读写 + 第二次 launch,-4~10%）。
  - **大 M（≥128）：不融合更快**,大 M 差距悬殊（M=256 融合 82µs vs 不融合 61µs,慢 25%）。融合在大 M 崩是因为 epilogue 要扣着 TMEM 做归约,EPI=1 断崖。
- **前提**：独立 RMSNorm kernel 近峰值带宽（M=256 ~8000 GB/s ≈ 95% HBM peak,仅 ~9µs）,使得"拆开"在大 M 非常划算。
- **建议**：按 M 门控——**小 M 用融合,大 M 用纯 GEMM + 独立 RMSNorm**；或图省事全用不融合（仅小 M 亏 1-4µs）。

