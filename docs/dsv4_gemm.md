# 按 Shape 选写法：三种小 M GEMM 的优化剖析（SM100 / tcgen05）

三个 GEMM 的共同前提：M（batch）很小（1~256），算术强度低，全部 memory-bound。**第一步永远是找出最大的那条数据流，用 流量 ÷ HBM 带宽 算出时间下限，然后让这条流永不断**。在此之上，N/K 的形状决定完全不同的三种写法：

- §1 N 巨大 + K 短 → persistent 网格 + 深流水 + TMEM 双缓冲
- §2 N 极瘦 + K 巨大 → 放弃 tile 范式，DeepGEMM split-K + fused reduce
- §3 N 中等 + K 大 + 混合精度 → 一次性网格，一 cluster 一 tile

---

## 1. 超大 N · 短 K：persistent 权重流 GEMM（FP8 block-scale）

**Shape**：实际 kernel 计算的是 merged projection：`x_fp8[M,1536] @ w_merged[73728,1536]^T`。前 `N_IDX=8192` 行是 indexer-q，后 `N_TOTAL=65536` 行是 main-q；最终 `y` 只保存 main-q 的 `[M,65536]` BF16，indexer tile 先 drain 到 `iq_scratch[M,64,128]`，再做 RoPE/Hadamard/FP8 或 FP4 transform。权重约 113MB，仍然是**权重带宽墙**。

**Tile 策略**：merged N 按 cluster-N=256 切成 `73728/256=288` 个 tile；一个 tile 覆盖整个 padded M（`M_pad∈{32,64,96,128}`），K 只沿 12 个 K128 stage 滚动。B300 上 GEMM 部分使用约 72 个 2-CTA clusters（144 CTA），每个 cluster 处理约 4 个 N tile；fused transform 为保持满 grid 另保留 2 个 transform-only clusters。所有 tile 的权重条带只读一次，激活因 M 小而主要由 L2 吸收。

- **swap-AB + 2SM**：`UMMA_M=256` 对应 merged N，权重是 MMA-A；`UMMA_N=M_pad` 对应 batch，激活是 MMA-B。每个 CTA 装 128 个权重行和 `M_pad/2` 个激活行，leader CTA 发 `cta_group::2` MMA。
- **persistent + M 相关深流水**：`SwapDims<M>` 根据 SMEM 预算决定 `NUM_STAGES`；M 越小，activation stage 和 TMEM accumulator 越小，可放更多 stage。每个 stage 有 `full/empty/with_sf_full` 三类 barrier，TMEM accumulator 另有双缓冲 `tmem_full/tmem_empty`。所有 barrier 都只靠 `(stage, parity)` 对齐、不存任何 tile 信息：`full/with_sf_full/tmem_full` 等 `wait(ph)`，`empty/tmem_empty` 等 `wait(ph ^ 1)`——后者利用 `try_wait.parity` 查询“上一个 phase”会立即通过，使所有 stage 初始天然是空的，不需要 warmup arrive。
- **SF 通道**：激活 scale 是 1×128，权重 scale 是 128×128；warp2 把连续 4 个 K128 的 UE8M0 字节作为一个对齐 u32 写入 SMEM，warp1 每组只做一次 UTCCP，MMA 用 `sf_id=0..3` 选择组内字节。
- **merged 特有的 indexer overlap**：N 的前 8192 列先被 epilogue drain；indexer 不走 main-q 的 TMA store，而是写 `iq_scratch` 后释放 TMEM。512-thread fused 版本的 warps 8..15 依次执行 ready-flag handshake、无依赖的 window/compressor chain、跨 CTA 的 SPREAD；plain/mock 版本可只启动 256-thread GEMM。
- **main-q epilogue**：TMEM FP32 先转 BF16，经 32×32 shared-memory transpose 和 128B swizzled TMA store 写 `y`；可选 `head_ssq` 在 epilogue 中用 `red.relaxed.global.add.f32` 累加，避免额外的 SSQ kernel。
- **PDL 与非对齐 M**：前置 qnorm/quant producer 写 `x_fp8/x_sf`，GEMM 在 barrier/TMEM prologue 后做 `cudaGridDependencySynchronize`；TMA descriptor 的 globalDim 用真实 M，box 用 M_pad，OOB 行零填/裁剪。


### §1 形态的面试手撕版（persistent FP8 GEMM 骨架）

只写面试需要表达的结构；`tma`、`mma`、`mbarrier`、`tmem` 都用短函数代表底层 PTX。

```cpp
template<int M_TILE>
__global__ void persistent_merged(/* X/W/SF/output/indexer 参数 */,
                                  int M, int num_gemm_clusters) {
    constexpr int K = 1536, BK = 128, NTILE = 288;
    int warp = threadIdx.x / 32, rank = cluster_rank();
    int cluster = blockIdx.x / 2;
    bool leader = rank == 0, fused_post = blockDim.x == 512;
    int first_tile = cluster < num_gemm_clusters ? cluster : NTILE;
    Shared s;

    // full/empty 管 A/B，sf_ready 表示 scale 已打包；双 TMEM accum 管 persistent tile。
    // arrive 计数：full/empty/tmem_full 都是 1（TMA 只有一个 elected lane；commit 的
    // multicast 给两个 CTA 各投一次），sf_ready = 2*32、tmem_empty = 2*128（两个 CTA
    // 的 warp2 / store 线程都远程 arrive 到 rank0 那一份）。
    if (warp == 1) init_barriers_and_double_tmem(s);
    cluster_sync();
    cudaGridDependencySynchronize();

    // (stage, phase) 就地从单调递增的 iter 算：st = iter % NSTAGE，
    // ph = (iter / NSTAGE) & 1（走完一圈 NSTAGE 翻一次 parity）。iter 跨 tile
    // 连续、不在 tile 边界重置——K-stage 环本就横跨 tile。barrier 里不存
    // tile 信息，各角色靠“iter 走得一样多”自动对齐。
    // 等“满”用 wait(ph)，等“空”用 wait(ph ^ 1)：try_wait.parity 查上一个 phase
    // 立即通过，所以第 0 圈 NSTAGE 个 stage 天然全空，不需要任何 warmup arrive。
    if (warp == 0)                      // 持久化 TMA 生产者
        for (int tile = first_tile, iter = 0; tile < NTILE; tile += num_gemm_clusters)
            for (int k = 0; k < K / BK; ++k, ++iter) {
                int st = iter % NSTAGE, ph = (iter / NSTAGE) & 1;
                s.empty[st].wait(ph ^ 1);
                tma_load_XW(tile, k, rank, s.a[st], s.b[st]);
                s.full[st].arrive_expect_tx();  // 一个 barrier 收两条 TMA，按字节总数计
            }

    if (warp == 2)                      // scale 生产者
        for (int tile = first_tile, iter = 0; tile < NTILE; tile += num_gemm_clusters)
            for (int k = 0; k < K / BK; ++k, ++iter) {
                int st = iter % NSTAGE, ph = (iter / NSTAGE) & 1;
                s.full[st].wait(ph);            // 只等本 CTA 的 TMA
                if ((k & 3) == 0) pack_four_ue8m0(tile, k, s.sf[st]);
                fence_proxy_async_shared();     // st.shared -> UTCCP 的异步代理可见
                // 两 CTA x 32 lane 汇聚到 rank0：把 per-CTA 的 full 中继成 cluster 级就绪
                s.sf_ready[st].arrive(/*to=*/0);
            }

    if (warp == 1 && leader) {          // 2SM MMA 消费者
        int iter = 0, it = 0;           // iter 管 K-stage 环，it 管 accum 环（每 tile 一次）
        for (int tile = first_tile; tile < NTILE; tile += num_gemm_clusters, ++it) {
            int acc = it % NUM_EPI, acc_ph = (it / NUM_EPI) & 1;  // accum 环深 NUM_EPI
            s.tmem_empty[acc].wait(acc_ph ^ 1);   // 同理：两个 accum 初始都是空的
            for (int k = 0; k < K / BK; ++k, ++iter) {
                int st = iter % NSTAGE, ph = (iter / NSTAGE) & 1;
                s.sf_ready[st].wait(ph);          // 等两个 CTA 的 operands + SF
                if ((k & 3) == 0) utccp_sf(s.sf[st]);
                mma_2sm_fp8(acc, s.a[st], s.b[st], /*sf_id=*/k & 3);
                commit_2sm(s.empty[st]);          // MMA 退休才投递，两 CTA 各收 1 次
            }
            commit_2sm(s.tmem_full[acc]);
        }
        // 末个 accum 的 release 没有下一轮 wait 去消费；显式收口后才能安全 dealloc TMEM。
        if (it > 0) s.tmem_empty[(it - 1) % NUM_EPI].wait(((it - 1) / NUM_EPI) & 1);
    }

    if (warp >= 4 && warp < 8) {        // 双缓冲输出阶段
        int it = 0;
        for (int tile = first_tile; tile < NTILE; tile += num_gemm_clusters, ++it) {
            int acc = it % NUM_EPI, acc_ph = (it / NUM_EPI) & 1;
            s.tmem_full[acc].wait(acc_ph);
            if (tile < 8192 / 256) {
                drain_indexer_to_scratch(acc, tile, rank, iq_scratch);
                s.tmem_empty[acc].arrive(/*to=*/0);  // drain-first：先释放 accum
                if (fused_post) publish_iq_ready(tile, rank, iq_ready);
            } else {
                tmem_to_bf16_smem(acc);           // 32x32 transpose
                s.tmem_empty[acc].arrive(/*to=*/0);  // 不等 store，尽早放走下个 tile 的 MMA
                tma_store_main_q(tile, rank, M);  // 大输出走 TMA store
            }
        }
    }

    // publish_iq_ready / wait_acquire 不是 mbarrier，所以没有 parity：CTA 内是 named
    // barrier（兼作 scratch 写的内存栅栏），跨 CTA 是 release/acquire 的全局 flag。
    if (fused_post && warp >= 8) {      // CUDA-core 变换 worker
        run_window_and_compressor_without_dependency();
        for (int head = worker_head(); head < IDX_NUM_HEADS; head += worker_stride()) {
            wait_acquire(iq_ready[head]);
            run_indexer_rope_hadamard_quant(iq_scratch, head);
        }
    }

    cluster_sync();
    if (warp == 0) tmem_free();
}

// M_pad 只取 {32,64,96,128}；2-CTA cluster，persistent grid 约占满全部 SM。
launch<M_pad><<<max_clusters * 2, fused_post ? 512 : 256>>>(...);
```

**面试讲解顺序**：先说 merged N 的 indexer-first 调度；再说 swap-AB 和 M-dependent stage；然后按 `TMA → SF-ready → MMA → TMEM` 讲主流水，顺手交代 `(stage, phase)` 的对齐方式和 `wait(ph ^ 1)` 为什么能省掉 warmup arrive；最后说明 indexer drain 释放 TMEM、异步 worker 用 ready flag 做 SPREAD。这些是它区别于普通 persistent FP8 GEMM 的关键。

---

## 1b. 同一 shape 的变体：BF16 + kernel 内完整 RMSNorm

**Shape 与 §1 同源**：`y[M,65536] = x[M,1536] @ w[65536,1536]^T`，BF16 进 BF16 出，`M=32~256`（对齐到 32），`BLOCK_K=64` → 24 个 K tile。同样 swap-AB（`UMMA_M=256` 在 N、`UMMA_N=M` 在 batch）、同样 2-CTA cluster + persistent。

**唯一的新需求：把每个 head 的 weightless RMSNorm 完整做完**（`rsqrt(mean(q²))` 沿 `HEAD_DIM=512` 归约）。就这一条需求，把 tile 策略、TMEM 布局、epilogue 形态全部改写了。

### norm 反过来决定 tile 策略

归约长度是 512，而一次 2SM MMA 只覆盖 `CLUSTER_BLOCK_N=256`。要让归约保持 **cluster-local**（不走 global），就必须让一个 cluster 同时持有整个 head：

- persistent 循环枚举的不再是 N tile，而是 **`NUM_HEAD_TILES = 65536/512 = 128` 个 head**；
- 每个 head = `SUBTILES_PER_HEAD = 2` 个 256 宽的 sub-tile，**两个 accumulator 同时常驻 TMEM**，最后一个 sub-tile 的最后一个 k 才发 `tmem_full`；
- 于是 TMEM 预算变成 `NUM_EPI_STAGES × 2 × M` 列（向上取 2 的幂），512 列的预算下 **只有 `M<=128` 能双缓冲**（`NUM_EPI_STAGES=2`），`M=256` 退化成单缓冲——直接失去了 §1 那种“drain 与下个 tile 的 MMA 重叠”。
- 激活 A 每个 sub-tile 重读一次（命中 L2），换取 sub-tile 之间复用同一条 SMEM 流水。

### barrier：比 §1 多一类 DSMEM barrier

| barrier | 计数 | 谁 arrive |
|---|---|---|
| `full[NS]` | **2** | 2SM TMA 把 tx 重定向到 leader；leader 发 `arrive_and_expect_tx(2×bytes)`，follower 发 `arrive(0)` |
| `empty[NS]` | 1 | `tcgen05.commit` multicast，两 CTA 各收 1 次 |
| `tmem_full[EPI]` | 1 | 同上，但只在 head 的**最后一个 sub-tile** 发 |
| `tmem_empty[EPI]` | 2×128 | 两 CTA 的 store 线程全体，最后一个 `st` 才 arrive |
| **`dsmem[EPI]`** | **1** | 本地 1 次 `arrive_and_expect_tx(M×4B)`；字节由**对端 CTA** 的 `store_shared_remote` 推过来 |

### 两趟 epilogue + DSMEM 跨 CTA 归约

RMSNorm 必须先拿到整个 head 的平方和才能缩放任何一个元素，而 512 个列横跨 **2 个 sub-tile × 2 个 CTA**，所以 TMEM 必须读两趟：

```
PASS 1  读两个 sub-tile 的 TMEM → 每行平方和
        → warp_reduce_sum32（沿 32 个 N lane）→ smem_warp_sq[4][M]  ← per-warp 槽位，免 atomic/预清零
        → NamedBarrier #1
        → store_shared_remote 把本 CTA 的 partial 推进对端 smem（DSMEM）
        → dsmem_barrier.wait → local + peer → rms = rsqrt(sum/512 + eps)
        → NamedBarrier #2
PASS 2  重读 TMEM（16dp256b 布局）→ ×rms → bf16 → stmatrix.trans → 合并 TMA store
```

三个值得单独记住的点：

1. **跨 CTA 归约走 DSMEM 而不是 global**：`cute::store_shared_remote(val, peer_smem_addr, bar_addr, peer_rank)` 直接写对端共享内存，并把字节数计入对端的 transaction barrier。**两个 CTA 对称互推**，因此两边都拿到完整和、各自算 `rms`，没有 leader/follower 不对称，也没有任何 HBM 往返。
2. **两趟之间 TMEM 不能释放**：PASS 2 还要再读一次，所以 `tmem_empty` 拖到最后一个 `st` 才 arrive——这正是“norm 融合拉长了 accumulator 占用周期”的代价。
3. **两个 sub-tile 合并 store**：一个 TMA-store stage 里并排放 `SUBTILES_PER_HEAD` 份区域，两边在**同一对 barrier** 下写出，epilogue barrier 数量减半。

### §1b 面试手写版

```cpp
template<int M>
__global__ void wq_b_bf16_rmsnorm(/* A/B/D descriptor */, int num_blocks, float eps) {
    constexpr int K = 1536, BK = 64, NK = K / BK;      // 24 个 K tile
    constexpr int HEADS = 128, SUB = 2;                // 512/256：一个 head 两个 sub-tile
    int warp = threadIdx.x / 32, lane = lane_id(), rank = cluster_rank();
    int cluster = blockIdx.x / 2, nclu = num_blocks / 2;
    bool leader = rank == 0;
    Shared s;

    // 比 §1 多一类 dsmem barrier（跨 CTA 推平方和）。EPI = (2*SUB*M <= 512) ? 2 : 1。
    if (warp == 1) init_barriers(s);   // full=2, empty=1, tmem_full=1, tmem_empty=2*128, dsmem=1
    if (warp == 2) tmem_alloc_2sm(pow2(EPI * SUB * M));
    cluster_sync();
    cudaGridDependencySynchronize();

    // (stage, phase) 同 §1 就地算：st = iter % NS, ph = (iter / NS) & 1。
    if (warp == 0)                                     // TMA 生产者：2SM multicast load
        for (int h = cluster, iter = 0; h < HEADS; h += nclu)
            for (int sub = 0; sub < SUB; ++sub)         // 激活每 sub-tile 重读（命中 L2）
                for (int k = 0; k < NK; ++k, ++iter) {
                    int st = iter % NS, ph = (iter / NS) & 1;
                    s.empty[st].wait(ph ^ 1);
                    tma_load_2sm(A, h, sub, k, rank, s, st);   // 内部算 m_base / n_base
                    tma_load_2sm(B, h, sub, k, rank, s, st);
                    if (leader) s.full[st].arrive_expect_tx(2 * (act + wgt));
                    else        s.full[st].arrive(0);          // 2SM TMA 把 tx 全计到 leader
                }

    if (warp == 1 && leader) {                         // MMA：一个 head 两个 accumulator
        int iter = 0, it = 0;                          // iter 管 K-stage 环，it 管 accum 环
        for (int h = cluster; h < HEADS; h += nclu, ++it) {
            int acc = it % EPI, acc_ph = (it / EPI) & 1;
            s.tmem_empty[acc].wait(acc_ph ^ 1);
            for (int sub = 0; sub < SUB; ++sub) {
                int tmem_c = acc * (SUB * M) + sub * M;         // 两份 accum 并存
                for (int k = 0; k < NK; ++k, ++iter) {
                    int st = iter % NS, ph = (iter / NS) & 1;
                    s.full[st].wait(ph);
                    for (int j = 0; j < BK / 16; ++j)           // BF16：K16 一条 kind::f16
                        mma_2sm_bf16(tmem_c, s, st, j, /*accum=*/k || j);
                    commit_2sm(s.empty[st]);
                    // 只有整个 head 算完才发布，epilogue 靠这一下拿到全 512 列
                    if (sub == SUB - 1 && k == NK - 1) commit_2sm(s.tmem_full[acc]);
                }
            }
        }
        if (it > 0) s.tmem_empty[(it - 1) % EPI].wait(((it - 1) / EPI) & 1);
    }

    if (warp >= 4) {                                   // 4 个 epilogue warp，两趟
        for (int h = cluster, it = 0; h < HEADS; h += nclu, ++it) {
            int acc = it % EPI, acc_ph = (it / EPI) & 1;
            s.tmem_full[acc].wait(acc_ph);

            // ---- PASS 1：head_dim 平方和（本 CTA 的 256 列）----
            for (int st = 0; st < M / 16; ++st)
                for (int i = 0; i < 2; ++i) {
                    float sq[8] = {0};
                    for (int sub = 0; sub < SUB; ++sub)         // 两个 sub-tile 一起累
                        accumulate_squares(sq, acc, sub, st, i);
                    for (int r = 0; r < 8; ++r) {
                        float v = warp_reduce_sum32(sq[r]);     // 沿 32 个 N lane
                        if (lane == 0) s.warp_sq[warp - 4][row(st,i,r)] = v;
                    }
                }
            named_barrier(128, 0);                             // #1：4 个 warp 的槽位可见

            // ---- 跨 CTA：DSMEM 对称互推，两边各自算 rms（不过 HBM）----
            if (tid_in_wg == 0) s.dsmem[acc].arrive_expect_tx(M * 4);
            for (int m = tid_in_wg; m < M; m += 128)
                store_shared_remote(sum4(s.warp_sq, m),         // 写对端 smem + 计入对端 barrier
                                    &s.peer_sq[acc][m], &s.dsmem[acc], rank ^ 1);
            s.dsmem[acc].wait(acc_ph);
            for (int m = tid_in_wg; m < M; m += 128)
                s.rms[m] = rsqrtf((sum4(s.warp_sq, m) + s.peer_sq[acc][m]) / 512.f + eps);
            named_barrier(128, 0);                             // #2：rms 可见

            // ---- PASS 2：重读 TMEM × rms → bf16，两个 sub-tile 合并 store ----
            for (int st = 0; st < M / 16; ++st) {
                if (epi_warp == 0) tma_store_wait<1>();
                named_barrier(128, 0);
                for (int sub = 0; sub < SUB; ++sub)
                    tmem_scale_bf16_stmatrix(acc, sub, st, s.rms, s.cd);  // 16dp256b + trans
                if (st == M / 16 - 1) s.tmem_empty[acc].arrive(0);  // 两趟都读完才释放
                tma_store_fence(); named_barrier(128, 0);
                if (epi_warp == 0) tma_store_both_subtiles(D, h, st, rank);
            }
        }
    }

    cluster_sync();
    if (warp == 0) tmem_free_2sm();
}

// persistent：枚举 128 个 head（不是 N tile），M 只取 32 的倍数。
launch<M><<<num_clusters * 2, 256>>>(...);
```

**只口述的部分**：

| 略掉的 | 一句话说法 |
|---|---|
| `full` 计数为何是 2 | 2SM TMA（`SM100_TMA_2SM_LOAD_2D`）把两个 CTA 的 tx 全部重定向到 leader 的 mbarrier，leader expect 双倍字节、follower 只补一次 arrive |
| PASS 1 / PASS 2 的 TMEM 布局不同 | PASS 1 用 `32dp32b.x8`（算平方和方便），PASS 2 用 `16dp256b.x1` ×2（`stmatrix.trans` 要求的寄存器布局） |
| rms 怎么对应到 lane | 16dp256b 下每 lane 的 8 个值只落在两个 M 行：`m0 = 2*(lane%4)`、`m1 = m0+1`，偶位→m0 奇位→m1 |
| `warp_sq[4][M]` 为何不用 atomic | per-warp 独占槽位 + 固定顺序求和，既免 atomic 也免预清零，还保证 bitwise 可重现 |
| `M=256` 为何变慢 | `EPI` 退化为 1，epilogue 与下个 head 的 MMA 不再重叠 |
| 输出精度 | 存 BF16（model-faithful：DSV4 的 q 就是 bf16），但 ssq 用 TMEM 里的 FP32 累加器算，精度优于先舍入再平方 |

### 对照：同一个 projection 的两种 norm 融合策略

| | §1（FP8, `head_ssq`） | §1b（BF16, 完整 RMSNorm） |
|---|---|---|
| kernel 内做多少 | 只累 `Σq²`，`rsqrt` 和缩放推给消费方 | 全做完，输出已 normalize |
| 跨 CTA 归约通道 | `red.relaxed.gpu.global.add.f32` fire-and-forget | **DSMEM `store_shared_remote` + txn barrier** |
| tile 自由度 | 高：按 N tile 枚举，一个 128 列 tile 必在一个 head 内即可 | 低：**整个 head 必须驻留一个 cluster** |
| TMEM | 1 accum/stage，双缓冲至 M=128 | 2 accum/stage（一个 head），双缓冲仅 M≤128 |
| epilogue | 单趟，drain 完就释放 | **两趟**，TMEM 占用翻倍 |
| 代价 | 消费方要自己 finalize | 无，下游直接用 |

记忆主线：**归约长度（head_dim=512）> 一次 MMA 的 N 覆盖（256），就强迫“一个 cluster 持一个 head”**，由此连锁出：persistent 枚举 head、两份 accumulator 常驻、TMEM 双缓冲只到 M=128、epilogue 必须两趟、跨 CTA 那一半平方和走 DSMEM 互推。想要 tile 自由度，就只能像 §1 那样只算 ssq、把 norm 拆给消费方。

---

## 2. 极瘦 N · 超长 K：DeepGEMM split-K + fused reduce（tf32）

**Shape**：`mix[M,24] = X[M,28672] @ W[24,28672]^T`，其中 `X` 是 BF16、`W` 是 FP32，乘法在 SM100 上以 TF32 执行，`M ≤ 256`。`N=24` 填不满常规 MMA 的 N 维，算术量很小，主要问题是把长 K 维读完并填满 SM。生产路径不是一个“大 GEMM 加一个简单求和”，而是两个真实算子：

1. DeepGEMM `tf32_hc_prenorm_gemm`：TF32 split-K GEMM，同时在 BF16→TF32 cast 时计算输入平方和；
2. 本仓库 `hc_reduce_fuse_out`：归约 partial，完成第一次 RMSNorm、门控、Sinkhorn、collapse，并可继续融合 attention RMSNorm 和 FP8 quant。

`ws[S,M,24]` 保存 GEMM partial，`sq[S,M]` 保存同一 split 的 `Σ X²` partial；`S ≈ SM_count / ceil(M/64)`，最多 112，使 `ceil(M/64) × S` 接近一波 SM。

### 2.1 算子一：DeepGEMM `tf32_hc_prenorm_gemm`

DeepGEMM 的真实调用约定是 `A/B` K-major、`D` N-major；`B` 的逻辑形状为 `[24,28672]`，但计算含义仍是 `X @ W^T`。当前 SM100 配置固定 `BLOCK_M=64`、`BLOCK_K=64`、`BLOCK_N=align(24,16)=32`，一个 block 负责一个 `(m_tile, k_split)`。

```cpp
// grid=ceil(M/64)*S；BLOCK=64x32x64；输出 ws[S,M,24]、sq[S,M]。
__global__ void splitk_tf32(int M, int S, float* ws, float* sq) {
    int m0 = (blockIdx.x / S) * 64;
    int split = blockIdx.x % S;
    int k0 = split_k_begin(split), nk = split_k_blocks(split);
    int warp = threadIdx.x / 32;
    Shared s;

    // TMA<->cast: full/empty；cast<->MMA: full_cast/empty_cast。phase 约定同 §1：
    // 等“满”wait(ph)，等“空”wait(ph ^ 1)；tmem_full 一个 split 只用一次，只有 phase 0。
    if (warp == 1) init_barriers_and_tmem(s);
    __syncthreads();
    cudaGridDependencySynchronize();

    if (warp == 0)                      // TMA 生产者
        for (int k = 0; k < nk; ++k) {
            int st = k % NSTAGE, ph = (k / NSTAGE) & 1;
            s.empty[st].wait(ph ^ 1);
            tma_load_XW(s.a[st], s.b[st], m0, k0 + k * 64);
            s.full[st].arrive_expect_tx();
        }

    if (warp >= 4) {                    // BF16->TF32 cast + Σx²
        float ssq = 0.f;
        for (int k = 0; k < nk; ++k) {
            int st = k % NSTAGE, ph = (k / NSTAGE) & 1;   // TMA 流水
            int cs = k % 2,      cph = (k / 2) & 1;       // cast 换手是 2 深环
            s.full[st].wait(ph);
            auto a = bf16_to_tf32(ldmatrix(s.a[st]));
            ssq += dot(a, a);
            s.empty_cast[cs].wait(cph ^ 1); tcgen05_st(/*TMEM A*/, a);
            s.full_cast[cs].arrive();
        }
        write_sq_without_atomic(sq, split, m0, warp_reduce(ssq));
    }

    if (warp == 1) {                    // TS-MMA: A=TMEM, B=SMEM
        for (int k = 0; k < nk; ++k) {
            int st = k % NSTAGE, cs = k % 2, cph = (k / 2) & 1;
            s.full_cast[cs].wait(cph);
            tcgen05_mma_tf32_ts(/*TMEM A*/, s.b[st], /*TMEM C*/);
            tcgen05_commit(s.empty_cast[cs], s.empty[st]);   // 两条 commit：还 TMEM-A、还 smem-B
        }
        s.tmem_full.arrive();
    }

    if (warp < 4) {                     // TMEM -> workspace
        s.tmem_full.wait(0);            // 一次性 accum：没有 parity 翻转
        tmem_to_smem(); named_barrier_sync(128);
        if (warp == 0) tma_store(ws[split][m0]);
        if (warp == 1) tmem_free();
    }
}
```

区别于普通 GEMM 的关键是：cast warp 在把 BF16 A 写入 TMEM 供 TS-MMA 使用时顺便生成 `sq`，所以不会为了第一次 norm 再扫一遍 `X`。

### 2.2 算子二：`hc_reduce_fuse_out`

该 kernel 是 `grid=M`、`block=256`。每个 block 处理一行，8 个 warp 同时归约 split-K partial：每个 warp 负责 3 个输出列，lane 以 `split` 为索引，避免 `atomicAdd`。`with_post_comb=true` 时 warp 0 做 Sinkhorn，其他 warp 同时做 collapse；生产 lite 路径则 8 个 warp 全部做 collapse。

```cpp
// grid=M，block=256；FULL=false 是生产 lite+norm+quant 路径。
template<bool FULL>
__global__ void reduce_fuse(/* hidden, ws, sq, gate 与输出指针 */, int M, int S) {
    int m = blockIdx.x, tid = threadIdx.x;
    int warp = tid / 32, lane = tid % 32;
    Shared s;                         // rms、mix[24]、pre、warp_ssq[8]

    asm volatile("griddepcontrol.launch_dependents;");
    init_shared(s);
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // reduce_split：lane 沿 split 走并做 warp_sum；8 warp 各负责 3 列。
    if (warp == 0) {
        float v = reduce_split(sq, m, lane, S);
        if (lane == 0) s.rms = rsqrtf(v / 28672.f + rms_eps);
    }
    for (int c = warp * 3; c < min(warp * 3 + 3, 24); ++c) {
        float v = reduce_split(ws, m, c, lane, S);
        if (lane == 0) s.mix[c] = v;
    }
    __syncthreads();

    // 第一次 norm：RMSNorm(X) @ W 等价于 (X @ W) * rms。
    if (tid < 24) {
        s.mix[tid] *= s.rms;
        if (mix_out) mix_out[m][tid] = s.mix[tid];
    }
    if (tid < 4)
        s.pre[tid] = sigmoid(s.mix[tid] * scale[0] + base[tid]) + hc_eps;
    __syncthreads();

    if constexpr (FULL) {
        // warp0: post + 4x4 softmax/Sinkhorn；warp1..7: 直接 collapse。
        if (warp == 0) sinkhorn_4x4(s.mix + 4, post[m], comb[m], base, scale);
        if (warp > 0)
            collapse_bf16(hidden[m], s.pre, collapsed[m], tid - 32, 224);
        return;
    }

    // 生产 lite 路径，两阶段完成 collapse + 第二次 norm。
    bf16 keep[4][8];                  // 每线程最多保存 4 组 BF16 x_b
    float local_ssq = 0.f;
    int seg = 0;
    for (int d = tid * 8; d < 7168; d += 256 * 8, ++seg) {
        float x[8] = collapse4(hidden[m], s.pre, d);
        keep[seg] = bf16_round(x);    // 必须先按 BF16 materialize
        local_ssq += dot(keep[seg], keep[seg]);
    }
    block_reduce_to_shared(local_ssq, s.warp_ssq); // warp shuffle + 8 个 partial
    __syncthreads();
    if (tid == 0) s.r2 = rsqrtf(sum8(s.warp_ssq) / 7168.f + attn_eps);
    __syncthreads();

    seg = 0;
    for (int d = tid * 8; d < 7168; d += 256 * 8, ++seg) {
        bf16 out[8] = bf16_round((float(keep[seg]) * s.r2) * gamma[d:d+8]);
        collapsed[m][d:d+8] = out;

        // 16 个线程覆盖连续 128 元素，直接发 FP8 + UE8M0 scale。
        float amax = max_16_threads(max_abs(out));
        int e = ceil_log2(max(amax / 448.f, 1e-4f));
        xq_out[m][d:d+8] = fp8(out * exp2f(-e));
        if ((tid & 15) == 0) xsf_out[m][d / 128] = e + 127;
    }
}
```

这里的两次 norm 不是同一件事：第一次用 DeepGEMM cast 阶段生成的 `sq`，对 `X[M,28672]` 做 RMSNorm 并缩放 24 维 `mix`；第二次在生产 `FULL=false` 路径中对 collapse 后的 `[M,7168]` 做 attention RMSNorm。第二次 norm 的 BF16 结果直接 quant 成 `xq_out[M,7168]` 和 `xsf_out[M,56]`，不再启动独立 norm/quant kernel。

### 为什么拆成这两个算子

- GEMM 的并行度来自 K split；`ws/sq` 只承载很小的 partial，避免 N=24 的空 tile。
- reduce 是 latency-bound：每 warp 负责 3 列、lane 遍历 split，读请求充分并发且没有原子冲突。
- PDL 让 reduce 在 DeepGEMM 尾部开始准备，并让 front 先做 prologue；`griddepcontrol.wait` 只放在真正消费 `ws/sq` 之前。
- 代价是一次 `[S,M,24]` FP32 workspace 写回和读取，但比重新扫描 28672 维 `X` 做 norm 小得多。

---

## 3. 中等 N · 长 K · 混合精度：CSA swap-AB front

**Shape**：`Y[M,4672] = X[M,7168] @ W[4672,7168]^T`，`1 ≤ M ≤ 256`，输出 bf16。权重前 2048 列是 FP8，后 2624 列是 BF16；权重约 52.3 MB，因此在 B300 上主要受 HBM 权重流限制。

### Tile 与 dispatch

- 两种精度统一使用 `UMMA_M=128`，问题的 batch 维放到 `UMMA_N=BatchN`，即权重作为 MMA-A、激活作为 MMA-B，这就是 swap-AB。一个 tile 由 2 个 CTA 组成 cluster：每个 CTA 负责 64 个 feature 行和 `BatchN/2` 个 batch 行，leader CTA 发 `cta_group::2` MMA。
- 每个 batch tile 固定 37 个 feature task：`task 0..15` 是 FP8 的 16 个 N128 tile；`task 16..36` 是 BF16 的 21 个 N128 tile。最后一个 BF16 tile 只有 64 个有效列，TMA OOB 填零，epilogue 按 `out_col < N` 裁剪。
- host 选择物理 `BatchN`：
  ```text
  M <= 48   : ceil(M / 16) * 16
  M <= 192  : ceil(M / 32) * 16
  M > 192   : 128
  ```
  取值只落在 `{16,32,48,64,80,96,128}`。`batch_tiles=ceil(M/BatchN)`，grid 为 `batch_tiles * 37 * 2` 个 CTA。一个 batch tile 只有 37 个 cluster、占 74 个 SM；`M>48` 时固定为两片 batch，得到 74 个 cluster/148 个 CTA，填满 B300。切 M 会让权重被两片各读一次，但这里用有限的重复读换取并行度；FP8 tile 不需要做成 BF16 tile 的两倍。

### Mainloop

- K 不切。FP8/BF16 的 `BLOCK_K` 实际相同：`BatchN<64` 时为 256，否则为 128；只有 TMA 子搬运粒度不同，FP8 为 K128、BF16 为 K64。对应的 stage 数是：FP8 为 `5/10/12`（`BN<64/64<=BN<128/BN=128`），BF16 为 `5/9/8/7`（`BN<=48/64/80..96/128`）。FP8、BF16 的 stage storage 互斥复用，实际分配取较大者。
- warp 0 是 TMA producer：发 `cp.async.bulk.tensor.cta_group::2` + `multicast::cluster`，每个 CTA 用 `self_mask = 1<<rank` 只把自己的 64-feature/`BatchN/2`-batch 半块搬进自己的 smem。但 barrier 地址被 `TMA_2SM_PEER_MASK` 抹掉 rank 位，**两个 CTA 的完成都汇报到 rank0 的 `full[st]`**：所以只有 rank0 发一次 `arrive_and_expect_tx(2 × (act + wgt))`，也只有 rank0 的 MMA warp 等 `full`；rank1 的 producer 只等本地 `empty[st]`（`tcgen05.commit` 的 multicast 会给两个 CTA 各投一次）。
- FP8 的 56 个 K128 scale tile 在 mainloop 前由 warp 3 搬入 shared memory，4 个 scale 打包成一个 u32，共 14 组；leader 每组只做一次 `UTCCP`，随后用 `sf_id=0..3` 发 MMA。**但两侧 scale 的 UTCCP 形态不同**：权重侧一个 cluster 只对应一个 128-feature scale block，两个 64-feature CTA 读的是同一行，用 `4x32.warpx4` 广播到四个 subpartition；激活侧走 M128 的 2×2 datapath，两个 `BatchN/2` 半区在 TMEM 里必须保持**各自不同**的 pattern，用 `64x128b.warpx2::01_23`，因此 `sf_act` 的 smem 是 `sf_weight` 的**两倍**。FP8 每个 K128 含 4 条 K32 `kind::mxf8f6f4.block_scale` MMA；BF16 每 K16 发一条 `kind::f16` MMA。最后一次 commit 发布 `tmem_full`。
- 开启 PDL 时，kernel 在初始化 barrier/TMEM 后执行 `griddepcontrol.wait`，再开始读激活、scale 和权重；可选的 warp 5 同时运行 HC/Sinkhorn tail，并在 epilogue 前通过 named barrier 汇合。

### TMEM 与 epilogue

每个 cluster 只有一次结果消费，所以只有一个 accumulator、一个 `tmem_full`（只用到 phase 0），不需要 §1 persistent kernel 的 accumulator 双缓冲。`NUM_TMEM_COLS=128` 用于 `BatchN<=96`，`BatchN=128` 时扩到 256。FP8 的 scale 列排在 accumulator 之后：`BatchN<=64` 时固定落在 64/68，更大的 `BatchN` 则随 accumulator 宽度推到 `BatchN`/`BatchN+4`。

`tmem_full` 后，普通路径由 epilogue warps 直接把 TMEM 的 fp32 结果转成 bf16 写 `out`。**两种精度共用同一套 2×2 象限解码**：cluster rank 选 64-feature 半区，epilogue warp 的低位选其中的 32-feature 块，高位选 `BatchN/2` 行半区（物理 warps 4..7）。production 路径按输出用途分流：FP8 的 `[1536,2048)` 写 `win_y2`，BF16 尾部 `[4608,4672)` 写 `w64`（该 tile 只有 64 个有效列，rank1 的列越界被裁）；其余 BF16 列先做 32×32 transpose——swap-AB 的 TMEM drain 天然是「一 lane 一列」，而 state ring 要按行写对齐 32B 向量，转置缓冲**直接借用 `tmem_full` 之后已经死掉的 pipeline smem**（每个 epilogue warp 独占 4KB）——再按 `state_row` 写 main/index state，并在需要时叠加 APE。所有路径都对真实 `M,N` 做边界裁剪。

### §3 面试手写版

只写白板上真要写的骨架（生产细节全在上面两节，面试时**口述**即可，见代码后的清单）。和 §1 的 prologue 相比只多一个二维解码：§1 是 persistent、grid 只枚举 N tile（M 是一整块），一维就够；§3 是一次性网格且 M 被切成 `batch_tile`，grid 必须枚举 batch × N-tile。其余寻址（feature 行 / batch 行 / 输出列）和 §1 一样藏进 `tma_load_2sm` / `store_tile`。

```cpp
template<int BatchN>                          // {16,32,48,64,80,96,128}
__global__ void mixed_swapab(/* X/W/SF/out 参数 */, int M) {
    constexpr int K = 7168, TASKS = 37;       // 16 个 FP8 + 21 个 BF16，都是 N128 tile
    int warp = threadIdx.x / 32, rank = cluster_rank();
    int cluster = blockIdx.x / 2;
    int batch_tile = cluster / TASKS, nt = cluster % TASKS;   // 一次性网格：二维解码
    bool fp8 = nt < 16, leader = rank == 0;   // fp8 是 CTA-uniform：两条流水静态分流
    Shared s;                                 // 两种精度 stage 存储互斥复用

    if (warp == 1) init_barriers(s);          // full/empty[NS] + 单个 tmem_full，均 init(1)
    if (warp == 2) tmem_alloc_2sm(NUM_TMEM_COLS<BatchN>());
    if (warp == 3 && fp8) preload_scales(s.sf, batch_tile, nt, M);  // K 单遍 => scale 常驻 smem
    asm volatile("griddepcontrol.wait;" ::: "memory");
    cluster_sync();

    int NS = stages<BatchN>(fp8), BK = block_k<BatchN>();

    if (warp == 0)                            // TMA 生产者，两种精度同一条流水
        for (int k = 0; k < K / BK; ++k) {
            int st = k % NS, ph = (k / NS) & 1;
            s.empty[st].wait(ph ^ 1);
            if (leader) s.full[st].arrive_expect_tx(2 * (act + wgt));  // 汇聚到 rank0
            tma_load_2sm(fp8, batch_tile, nt, k, /*mask=*/1 << rank, s, st);  // 内部算 w_row/x_row
        }

    if (warp == 5) hc_tail_run(hc);           // 主循环期间空闲，用来藏 HC/Sinkhorn 尾巴

    if (warp == 1 && leader)                  // rank0 发 2SM MMA，单 accumulator
        for (int k = 0; k < K / BK; ++k) {
            int st = k % NS, ph = (k / NS) & 1;
            s.full[st].wait(ph);
            if (fp8) {
                if ((k & 3) == 0) utccp_scales(s.sf, k / 4);   // 每 4 个 scale 一组
                mma_2sm_fp8(s, st, /*sf_id=*/k & 3);
            } else {
                mma_2sm_bf16(s, st);
            }
            commit_2sm(s.empty[st]);
            if (k == K / BK - 1) commit_2sm(s.tmem_full);
        }

    if (warp == 0) s.tmem_full.wait(0);       // K 单遍 => 只有 phase 0
    if (warp == 0 || warp >= 4) named_barrier(160, /*TMEM ready=*/0);

    if (warp >= 4)                            // 4 个输出 warp 各 drain 一个 2x2 象限
        store_tile(out, batch_tile, nt, fp8, rank, warp - 4, M); // 内部算列/行 + production 分流

    if (warp == 2 || warp >= 4) named_barrier(160, /*store done=*/1);
    if (warp == 2) tmem_free_2sm();
}

// M<=48 只一片（37 cluster/74 SM），M>48 用两片凑成 74 cluster/148 CTA 填满 B300。
int BatchN = M <= 48  ? ceil_div(M, 16) * 16
           : M <= 192 ? ceil_div(M, 32) * 16 : 128;
launch<BatchN><<<ceil_div(M, BatchN) * 37 * 2, 256>>>(...);
```

**上面刻意不写、只口述的部分**（被追问时再展开，细节见 Mainloop / TMEM 两节）：

| 略掉的 | 一句话说法 |
|---|---|
| swap-AB 寻址（`tma_load_2sm`/`store_tile` 内部） | A=权重按 feature 切（每 CTA 64 行），B=激活按 batch 切（每 CTA `BatchN/2` 行）；输出列 = `nt*128 (+2048 if BF16)` |
| `full` 为何只由 rank0 expect | barrier 地址抹掉 rank 位，两个 CTA 的 TMA 都汇报到 rank0 那一份 |
| TMA 子搬运粒度 | 一个 `BLOCK_K` 拆成几次 TMA：FP8 按 K128、BF16 按 K64 |
| SF 两侧不对称 | 权重 `4x32.warpx4` 广播，激活 `64x128b.warpx2::01_23` 保持两半区独立、smem 双倍 |
| FP8 内层 MMA | 每个 K128 子块发 4 条 K32 block-scale MMA；BF16 每 K16 一条 |
| 输出四路分流（`store_tile` 内部） | `win_y2` / `w64` / state+APE / 普通 `out`，按输出列区间静态选；2x2 象限：rank 选 64 半、`q&1` 选 32 块、`q>>1` 选 `BatchN/2` 行半 |
| 32×32 转置 | swap-AB drain 是“一 lane 一列”而 state ring 要按行写，转置缓冲借用已死的 pipeline smem |
| 尾 tile | BF16 `nt=36`（段内 20）只 64 行有效，rank1 整个越界，靠 TMA OOB 填零 + `col < N` 裁剪 |

记忆主线：`UMMA_M=128` 保持两种精度的 tile 形状一致（因此 epilogue 可以共用同一套 2×2 象限解码）；`BatchN` 只切到能填满 SM 的两片；TMA producer、scale preload、MMA consumer 由 mbarrier 串成流水（`full` 汇聚到 rank0，`empty` 由 commit multicast 回两个 CTA）；权重 scale 广播、激活 scale 走 2×2 且 smem 双倍；一次性 K 遍历只需一个 accumulator，epilogue 再按 plain/production 路径分流。

---

## 总结：按 shape 定写法

| 形状特征 | 代表 shape | 写法 |
|---|---|---|
| N tile 数 ≫ SM 数，K 短 | [M,1536]×[73728,1536]ᵀ | persistent + cluster 步进；深流水让整条 K 在飞；TMEM accum 双缓冲让 store 与下个 tile 的 MMA 重叠；epilogue 走 smem swizzle + TMA store |
| N < 最小 tile，K 巨大 | [M,28672]×[24,28672]ᵀ | 放弃 tile 范式：split-K 到 ≈SM 数个 K 段，partial 落 workspace；独立 reduce kernel（lane=split、无原子、拼 in-flight load）；两 kernel 用 PDL 缝合 |
| N tile 数 ≈ SM 数，K 长 | [M,7168]×[4672,7168]ᵀ | 一次性网格，一 cluster 一 tile，K 单遍；SF/元数据常驻 smem；单 accum 无双缓冲；输出小 → 直接 st.global |

跨形状通用的手段：

1. **小 M 一律 swap-AB**：大维（N/特征）放 A/M 侧吃满 128/256 档，小 batch 放 B/N 侧按模板取 {16..128}；模板化 M 的红利是 smem/TMEM 随 M 缩小、流水深度自动加深。
2. **cluster 大小由 tile 大小定**：256 宽的大 tile 用 2SM 劈半（TMA 流量与 smem 减半）；64×32 的小 tile 用 1SM——2SM 不是默认答案。
3. **非对齐 M/N 交给 TMA OOB 语义**：真实尺寸进 globalDim、padded 尺寸进 box，加载零填、store 裁剪，host 侧零 pad。
4. **低精度输入 × 高精度 MMA 用 cast 警组**：操作数必须过 CUDA-core 寄存器时（如 bf16→tf32 进 TMEM 的 TS-MMA），把 cast 做成独立 warpgroup、与 MMA 双 barrier 换手，cast 藏进上一 tile 的 MMA 影子。
5. **FP8 block-scale 的 SF 通道单独设计**：UE8M0 按 4 打包成 u32、UTCCP 按组摊销、`sf_id` 选字节；K 单遍 → 常驻 smem，K 多遍 → per-stage 流水 + 专职 warp。
6. **混合精度按 N 分段**：同 kernel 两套指令描述符/TMA 描述符，任务 id 静态分流，两条流水线互斥复用同一块 smem。
7. **流水深度不拍脑袋**：`(smem 容量 − 常驻开销) / 每 stage 字节` 编译期推导 + clamp + static_assert。
8. **epilogue 通路按输出体量选**：MB 级以上走寄存器→smem swizzle→TMA store（合并写 + 与 MMA 重叠需 accum 双缓冲）；KB~低 MB 级直接 st.global；split-K partial 直写 workspace。
9. **多 kernel 模式用 PDL 缝合**：secondary 的 `griddepcontrol.wait` 放在自身 prologue 之后，启动延迟与初始化藏进 primary 尾部。
10. **归约类 kernel 按 latency-bound 设计**：并行维放进 lane（最大化 in-flight load）、shuffle 树收口、避开共享地址原子。
11. **mbarrier 只存 parity，不存进度**：每个角色本地维护 `(stage, phase)`，`stage` 环回时翻 `phase`；等“满”`wait(ph)`、等“空”`wait(ph ^ 1)`。`try_wait.parity` 查上一个 phase 立即通过，`^1` 因此免费实现“所有 stage 初始为空”，不需要 warmup arrive。多深环（K-stage / TMEM accum / cast 换手）各算各的 parity：环深 D 的 parity 是 `(iter / D) & 1`；K 单遍的一次性 accumulator 只用到 phase 0。

---
