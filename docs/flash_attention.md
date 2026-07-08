# FlashAttention (Half-Precision Self-Attention)

计算单个 head 的注意力：

```
S = Q @ K^T · scale          Q,K,V: [N, D]   scale = 1/√D
P = softmax(S, axis=-1)       S,P:   [N, N]
O = P @ V                     O:     [N, D]
```

本仓库实现前向、fp16 输入 / fp32 统计、`D = 128`，layout 为 `[B, H, N, D]` row-major，一个 head 即一次 `[N,D]×[N,D]→[N,D]` 的问题。

---

## 为什么需要 FlashAttention

朴素实现分三个 kernel：`S=QK^T` → `P=softmax(S)` → `O=PV`。中间矩阵 `S`、`P` 都是 `[N, N]`，必须落回 HBM 再读回：

- **显存爆炸**：`N=8192` 时单个 `[N,N]` fp16 就是 128 MB，且随 `N²` 增长。
- **访存瓶颈 (memory bound)**：attention 的算术强度低，`S`、`P` 的 HBM 往返（写 N²、读 N²、再写 N²）远多于实际计算，kernel 卡在带宽上，Tensor Core 大量空转。

**FlashAttention 的核心**：把 `S`、`P` 永远留在片上（smem / 寄存器），用 **tiling + online softmax** 分块流式计算，中间结果不落 HBM。HBM 上只读 Q/K/V、写 O，访存量从 `O(N²)` 降到 `O(N·D)`。

### Online Softmax（算法基石）

softmax 需要「先求整行最大值和总和，再归一化」，天然要求看到整行——这与分块流式冲突。**online softmax** 用「运行时最大值 + 运行时和 + 追溯修正」解决：每来一个新分块，就地更新统计量，并把已累加的输出按比例修正。

设已扫描过的分块给出运行统计 `(m, l, O)`（行最大值、指数和、加权输出），新分块 `S_t` 到来时：

```
m_new = max(m_old, rowmax(S_t))            更新行最大值
P_t   = exp(S_t·scale − m_new)             用新 max 重算本块概率
l_tile= rowsum(P_t)
α     = exp(m_old − m_new)                 旧统计的修正因子 (m 变大 → α<1)
l_new = α·l_old + l_tile                   修正旧和后累加
O_new = α·O_old + P_t @ V_t                修正旧输出后累加
```

扫完所有 KV 分块后，一次性 `O ← O / l` 得到最终结果。

**为什么减 `m` 数值安全**：`S·scale` 可能很大，`exp` 直接溢出 fp32。减去运行最大值把指数参数压到 `≤ 0`，`exp ∈ (0,1]`，绝不溢出；`m` 每步单调增，`α = exp(m_old − m_new) ≤ 1` 也不会溢出。这就是 "safe softmax"。所有统计量 `m/l/α` 用 **fp32** 保存，避免 fp16 累加误差。

---

## 1. FlashAttention-1：O/m/l 常驻 HBM  [`1a25bc8`](https://github.com/guluguluhhhh/wuda/commit/1a25bc8)

对应论文 Algorithm 1，见 [flash_attention_v1.cu](../flash_attention/flash_attention_v1.cu)。

```
grid = (B*H,)                    一个 block 负责一个 (b, h) 的整块 attention
外层循环 KV (Tc = N/Bc 段)
  内层循环 Q (Tr = N/Br 段)
    O, m, l 全程存 HBM, 每个内层 iter: HBM 读 → smem 算 → 写回 HBM
```

关键在归一化时机——FA1 **每一步都把 O 完整归一化**（P 预乘 `1/l_new`），末尾无需再除：

```
m_new   = max(m_old, m_tile)
P       = exp(S·scale − m_new)
l_tile  = rowsum(P)
α       = exp(m_old − m_new)
l_new   = α·l_old + l_tile
O_new   = (α·l_old/l_new)·O_old + (1/l_new)·(P@V)
```

**FA1 的两个致命开销**（正是 FA2 要解决的）：

1. **O/m/l 反复读写 HBM**：外层 KV、内层 Q 的循环顺序下，每个 `(i, j)` 组合都要把 `O_i, m_i, l_i` 从 HBM 读出、更新、写回。同一 Q 行块被外层每一段 KV 反复 R-M-W。
2. **每步一次除法归一化**：`(1/l_new)` 每个 KV 步都乘进 O，非矩阵乘的标量运算量大。

FA1 在本仓库仅作对照基线，用来凸显 FA2 的循环顺序改进。

---

## 2. FlashAttention-2：交换循环，O 常驻片上  [`3acf45b`](https://github.com/guluguluhhhh/wuda/commit/3acf45b)

见 [flash_attention_v2.cu](../flash_attention/flash_attention_v2.cu)。FA2 相对 FA1 做了三处结构性改动：

### (1) 交换循环顺序，Q 上并行

```
grid = (N/Br, B*H)               一个 block 独占一个 Q 行块 [Br, D]
块内: 外层循环 KV (Tc = N/Bc 段)
```

一个 block 全程只负责固定的 `Q_i`，O/m/l 的运行统计**常驻 smem**，不再进出 HBM。不同 Q 行块之间完全独立 → 天然沿序列维并行，`grid.x = N/Br` 直接喂满 SM（FA1 的 grid 只有 `B*H`，长序列下 SM 利用率低）。

### (2) 延迟归一化

FA2 循环内**只**做 `O ← α·O + P@V`（不除 l），把 `1/l` 归一化推迟到最后一次性完成。省掉每步的除法，减少非 matmul FLOP：

```
循环内: O_old *= α;  O += P @ V
末尾:   O *= 1/l
```

### (3) Split-Q：warp 间分行、共享 KV

```
TPB = WarpCountM · 32           沿 Br 切 WarpCountM 个 warp
每 warp 负责 WarpM = Br/WarpCountM 行, 固定 WarpM = 16 (= mma 的 M)
KV 在所有 warp 间共享
```

一个 block 内每步的执行流程（K/V 复用同一块 smem，见 [`75f6265`](https://github.com/guluguluhhhh/wuda/commit/75f6265)：K 用完才加载 V，生命周期不重叠）：

```
1. load K_t → smem_KV
2. S = Q @ K^T      WarpHMMA_f16 (TN: K row-major 当 col_major B 加载即 K^T)
                    输出 fp16 到 smem_SP
3. online softmax   每 warp WarpM 行, lane 0..15 各处理一行
                    m_new / P / l_tile / α;  O_old *= α
4. load V_t → smem_KV (复用 K 的 buffer)
5. O += P @ V       WarpHMMA_NN_f16 (NN: P,V 都 row-major), fp32 累加在 smem_O
末尾: O *= 1/l → 写回 gmem
```

此时 O 用 **fp32 smem** 累加保精度，S/P 在 `smem_SP` 上原地覆盖（softmax 读 S 写 P 同一 buffer）。这是功能正确的基线，后续所有优化都在 `flash_attention_performance.cu` 上展开。

---

## 3. 性能优化：从 WMMA 基线到寄存器驻留 + 流水

见 [flash_attention_performance.cu](../flash_attention/flash_attention_performance.cu)。下面按提交顺序拆解每一步优化的动机与手段。

### 3.1 MMA PTX 替换 WMMA  [`50ef078`](https://github.com/guluguluhhhh/wuda/commit/50ef078)

WMMA API 是黑盒，fragment 的寄存器布局不可见，无法在寄存器上直接操作累加器。换成手写 PTX（`ldmatrix.sync.m8n8.x4` + `mma.sync.m16n8k16.f16`，见 [mma_ptx.cuh](../gemm/warp/mma_ptx.cuh)）后能精确掌控每个 lane 持有哪些元素——这是后续「O 常驻寄存器」「按行 rescale 累加器」的前提。

此步先把 QK 用 `WarpHMMA_f16`（B col-major）、PV 用 `WarpHMMA_Trans_f16`（B row-major，`ldmatrix.trans`）搭起来，O 仍落 fp16 smem，功能与基线等价。

### 3.2 Warp 协作 softmax  [`5af305d`](https://github.com/guluguluhhhh/wuda/commit/5af305d)

基线 softmax 只有 `lane < 16` 干活、其余 16 个 lane 空转，且串行遍历整行。改为**全 32 lane 协作处理一行**：每 lane 取 `Bc/32` 个元素求局部 max/sum，再用 `__shfl_xor_sync` butterfly 归约得到行 max/sum。消除半个 warp 的空转，行内归约走寄存器 shuffle 而非串行。

### 3.3 O 累加器常驻寄存器  [`c6fbb9c`](https://github.com/guluguluhhhh/wuda/commit/c6fbb9c)

基线每个 KV 步都：`load_C`(smem→reg) → `mma` → `stmatrix`(reg→smem)，softmax 里再逐元素 rescale smem 上的 O。O 在 smem ↔ reg 之间反复搬运。

优化：把 PV 的累加器 `mma_pv.frag_C` **跨整个 KV 循环持久保留在寄存器里**，全程不落 smem：

- 循环内直接 `mma_pv.forward(P, V)` 累加到 `frag_C`；
- rescale 改为直接在寄存器上乘 α（`rescale_frag_O`）；
- 末尾才 `stmatrix` 一次写回。

**面试常考——为什么每 lane 只需 2 个 α 因子**：`mma.m16n8k16` 的 fp16 累加器里，每个 lane 持有一个 m16n16 tile 的 4 个 uint32（= 8 个 half，两条 m16n8 拼接）。这 8 个元素只落在**两行**上：上半行 `row0 = lane>>2`（0..7）对应 `frag_C[·][·][0]`、`[2]`，下半行 `row0+8`（8..15）对应 `[1]`、`[3]`。α 是 per-row 的，所以每 lane 只需 `a_upper`、`a_lower` 两个因子。rescale 用 `__hmul2` 一条指令处理一对 half（`hmul2_pack`）。

α 由 softmax 算出后写入 `smem_alpha`（`__syncwarp` 保证 lane 0 写齐），rescale 再从 `smem_alpha` 按行读回。

### 3.4 Q 常驻寄存器  [`d90c8b3`](https://github.com/guluguluhhhh/wuda/commit/d90c8b3)

`Q_i` 在整个 KV 循环里不变，却每步都要作为 mma 的 A 操作数从 smem `ldmatrix` 一次。优化：kernel 开头一次性 `load_full_A` 把整块 Q 的所有 K-iter fragment 读进 per-lane 寄存器 `frag_Q`，之后 `forward_with_A` 每步只 `ldmatrix` K，不再读 Q。

副作用：`smem_Q` 可删——Q 只在初始化时借 `smem_KV` 暂存、`ldmatrix` 进寄存器后，`smem_KV` 立刻能被 K/V 覆写。至此 **Q 常驻寄存器、O 常驻寄存器**，smem 只剩 KV staging + S/P。

### 3.5 向量化 + `exp2f` + 树形归约 softmax  [`acc0fa9`](https://github.com/guluguluhhhh/wuda/commit/acc0fa9)

softmax 是 attention 里唯一的非 matmul 热点，逐 fp16 访存 + `expf` 依赖链很贵。四管齐下：

- **向量化访存**：每 lane 用一条 `LDS.128`（`uint4`）读 8 个 fp16、`STS.128` 写回，访存指令数 ÷4（`smem_SP` 加 `__align__(16)`）。为喂满 128-bit，把一个 warp 拆成 2 个 16-lane threadgroup，各算一行（`Bc=128` → 16 lane × 8 fp16 = 一整行）。
- **`exp2f` 替 `expf`**：把 `log2e` 折进 scale（`scale·log2e`），`exp(x) = exp2(x·log2e)`，映射到单条 `MUFU.EX2` 硬件指令，省掉 `expf` 的软件展开。
- **树形归约**：8 个元素的 local max/sum 用平衡树（依赖链深度 3 而非 8），butterfly 只需 `off ≤ 8`（threadgroup 内自治，不跨 lane16 边界）。
- **指令级并行**：`STS.128` 写 P、`exp2f` 算 α（SFU pipe）、`__shfl` 求和（MIO pipe）走不同硬件端口，可重叠发射。

### 3.6 D=128 + 动态 smem  [`c5eb97f`](https://github.com/guluguluhhhh/wuda/commit/c5eb97f)

把 head dim 提到实用的 `D=128`。此时静态 smem（`smem_KV` + `smem_SP`）已顶到 48 KB 静态上限，于是把小而杂的 `m/l/alpha`（各 `Br` 个 float）挪到 **dynamic shared memory**（`extern __shared__` + `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 显式 opt-in），给静态区腾地方。

### 3.7 Split-N：加一维 warp 切分提 TLP  [`cb3c45b`](https://github.com/guluguluhhhh/wuda/commit/cb3c45b)

原本只沿 Br 切 warp（`WarpCountM`），warp 数偏少、并行度不足。新增 `WarpCountN` 维，把 warp grid 变成 `WarpCountM × WarpCountN`：

- **QK 阶段**：`warp_n_id` 沿 Bc 切 K 的行，每 warp 算 S 的 `[WarpM, Bc/WarpCountN]` slice；
- **PV 阶段**：`warp_n_id` 沿 D 切 V 的列，每 warp 算 O 的 `[WarpM, D/WarpCountN]` slice（`ldb=D` 不变，只加列偏移）；
- **softmax**：`smem_SP` 已含整行，行数平均分给所有 `WarpCount` 个 warp。

warp 数 ×`WarpCountN` → TLP 翻倍、每 warp 的 `frag_C` 减半（寄存器压力降低，利于 occupancy）。**代价**：同一 M-group 的 N-warps 持有相同 Q 行，`frag_Q` 在寄存器里重复（smem 是 broadcast 读，不额外耗带宽）。α rescale 后改用 `__syncthreads`（M-group 内所有 N-warp 都要读 α）。默认 `WarpCountM=4, WarpCountN=2`，8 warp/block。

### 3.8 cp.async 预取 V，与 softmax 重叠  [`d4dcc15`](https://github.com/guluguluhhhh/wuda/commit/d4dcc15)

用 `cp.async`（SM80+ 异步拷贝，GMEM→SMEM 绕过寄存器，见 [prelogue.cuh](../gemm/block/prelogue.cuh) 的 `block_mma_prelogue_f16_async`）做计算/访存重叠。

关键观察：QK GEMM 做完后 `smem_KV` 里的 K 已无人再读，可以**提前发射 V 的异步加载**，让 V 的搬运与随后的 **softmax + rescale** 计算重叠：

```
QK GEMM (读 smem_KV 里的 K)
cp.async 发射 V → smem_KV           ← V load START
online softmax   (只碰 smem_SP)     ← 与 V 传输并行
rescale frag_C   (只碰寄存器)        ← 与 V 传输并行
cp_async_wait_group<0>()            ← V load WAIT
O += P @ V
```

softmax/rescale 阶段完全不碰 `smem_KV`，V 的搬运被这段计算「免费」掩盖。

### 3.9 Padding 消除 bank conflict（约 2×）  [`d625baf`](https://github.com/guluguluhhhh/wuda/commit/d625baf)

`D = Bc = 128` half 的行 stride 恰好是 32 个 bank 的整数倍。`ldmatrix`/`LDS.128` 同时访问多行同一列时，不同行的同一列落进**同一 bank**，访问被串行化。

解决：每行 padding 8 个 half，stride 从 128 → **136**（`KV_STRIDE = D+PAD`、`SP_STRIDE = Bc+PAD`），错开 bank 映射消除 conflict。PAD=8 保证 16B 对齐（`ldmatrix` 要求）。这是**单步收益最大的优化，约 2×**——印证此前 kernel 的瓶颈已从访存/计算转移到 smem bank conflict 上。

> O 写回的 staging 段仍用 `stride=D`（无 padding）——staging 只被 `stmatrix` 顺序写、`eplogue` 顺序读，无跨行同列冲突，不必浪费 smem。`SP_ELEMS = Br·max(SP_STRIDE, D)` 取二者较大者。

### 3.10 Adaptive Bc + occupancy  [`9a18ae4`](https://github.com/guluguluhhhh/wuda/commit/9a18ae4)

前面 softmax 硬编码「一个 warp 算 2 行、16 lane/行」，绑死 `Bc=128`。泛化 `warp_online_softmax`：`kLanesPerRow = Bc/8`、`kRowsPerWarp = 32/kLanesPerRow`，每 lane 始终吃 8 fp16，归约范围随 Bc 自适应（`Bc=64` → 8 lane/行、4 行/warp）。

解锁后把 `Bc` 从 128 缩到 **64**：单 block 的 smem 从占满降到约 27 KB，**一个 SM 能同时驻留 2 个 block**，occupancy 翻倍、延迟隐藏更好。这是 tile 大小与 occupancy 的经典权衡——大 tile 计算密度高但挤占 smem/寄存器压低并发，缩 tile 反而更快。

---

## 优化路径总览

| # | 优化 | 手段 | 消除的瓶颈 |
|---|---|---|---|
| — | FA1 → FA2 | 交换循环 + 延迟归一化 + split-Q | O/m/l 的 HBM 往返、每步除法 |
| 3.1 | MMA PTX | 手写 `ldmatrix`+`mma.sync` | WMMA 黑盒，无法操作累加器 |
| 3.2 | Warp softmax | 32 lane 协作 + shuffle 归约 | 半 warp 空转、串行归约 |
| 3.3 | O 驻寄存器 | `frag_C` 跨循环持久 + 寄存器 rescale | O 的 smem↔reg 往返 |
| 3.4 | Q 驻寄存器 | `load_full_A` 一次性 + `forward_with_A` | Q 每步重复 ldmatrix |
| 3.5 | 向量化 softmax | LDS/STS.128 + `exp2f` + 树形归约 | softmax 访存 & SFU 延迟 |
| 3.6 | D=128 + dyn smem | m/l/α 移入 dynamic smem | 48 KB 静态上限 |
| 3.7 | Split-N | 加 `WarpCountN` 维 | TLP 不足、寄存器压力 |
| 3.8 | cp.async | 异步预取 V，掩盖于 softmax | 访存/计算串行 |
| 3.9 | Padding | 行 stride 128→136 | smem bank conflict（≈2×） |
| 3.10 | Adaptive Bc | 泛化 softmax + 缩 Bc=64 | occupancy 偏低 |

**最终配置**：`Br=64, Bc=64, D=128, WarpCountM=4, WarpCountN=2`（8 warp/block）。Q/O 全程驻寄存器，KV 用 cp.async 预取并与 softmax 重叠，smem padding 消 bank conflict，2 blocks/SM。

---

## 关键实现细节（深入）

**三块矩阵乘的 layout**：QK 是 `Q[Br,D] @ K^T`，K row-major 作为 col-major B 加载即天然取到 `K^T`（TN，用 `WarpHMMA_f16`）；PV 是 `P[Br,Bc] @ V[Bc,D]`，P、V 都 row-major（NN，用 `WarpHMMA_Trans_f16`，B 走 `ldmatrix.trans` 转置）。

**S/P 原地覆盖**：`smem_SP` 里 softmax 读入 S（QK 的输出）、写出 P（PV 的输入），同一 buffer 原地覆盖，省一块 smem；末尾再复用为 O 写回 gmem 前的 staging。

**smem 别名复用**：`smem_KV` 依次承载 Q-init → K tile → V tile（生命周期不重叠）；`smem_SP` 承载 S/P → O staging。片上只留两大块 buffer，是 D=128 能压进 smem 预算的关键。

**统计量精度**：`m/l/α` 全程 fp32；PV 累加器虽是 fp16（Tensor Core f16.f16.f16），但 attention 权重 P∈(0,1]、V 有界，配合 per-tile max 归一，精度足够（bench 对 fp32 CPU 参考验证）。

---

## Benchmark

见 [bench.cu](../flash_attention/bench.cu)。正确性对 fp32 CPU 参考实现验证；性能测试配置 `B=1, H=32, D=128`，`N ∈ {512, 1024, 2048, 4096, 8192}`，指标 `TFLOPS = 4·B·H·N²·D / time`（QK 与 PV 各贡献 `2·B·H·N²·D`）。三个 kernel（`fa_v1` / `fa_v2` / `fa_perf`）同表对比，可直接编译运行复现每一步优化的增量收益。
