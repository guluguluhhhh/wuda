# DSV4 Attention Kernels

DeepSeek-V4 Attention 部分的 CUDA 算子实现。按数据流拆分为两个独立 kernel。

## Kernel 1: hc_fused_kernel

**功能**: mHC Pre-Block Mixing（Manifold-Constrained Hyper-Connections 的输入映射）

**计算流程**:
1. RMSNorm (无 weight): 对 [HC*D=28672] 做归一化
2. GEMV [28672, 24]: 生成 pre/post/comb 的原始参数
3. Activation: sigmoid(pre), 2*sigmoid(post), softmax(comb)
4. Sinkhorn (20 iterations): comb 矩阵双随机归一化
5. Collapse: pre 加权求和 HC 份 → 1 份
6. attn_norm: 带 weight 的 RMSNorm

**硬件选择**: 纯 CUDA Core。GEMV 的 N=24 太窄，Tensor Core 无收益。

**优化**: GEMV 占整个 kernel 90%+ 延迟，核心优化：
- 2-block cluster + DSMEM: 两个 block 分别负责 K/2 维度（各 14336 元素），通过 distributed shared memory 交换 partial sum，将 GEMV 带宽翻倍
- Split-K 并行: 1024 threads/block × 2 blocks 协作规约，每个 thread 负责 14 个 K 元素
- 权重预转置 [24, 28672] → [28672, 24]: 使同一 K 位置的 24 个输出权重在内存中连续，每个 thread 用 3 次 int4 (128-bit) 向量化加载取回全部 24 个 bf16 权重，替代 24 次标量加载

**性能** (B300, sm_103, HBM_PEAK=8000 GB/s):

| B×S | Latency(us) | Throughput | BW (GB/s) | HBM Util |
|-----|-------------|------------|-----------|----------|
| 1 | 32.8 | 0.03 M pos/s | 44.1 | 0.6% |
| 4 | 32.8 | 0.12 M pos/s | 176.6 | 2.2% |
| 16 | 32.8 | 0.49 M pos/s | 706.4 | 8.8% |
| 64 | 33.1 | 1.94 M pos/s | 2804.0 | 35.1% |
| 128 | 57.4 | 2.23 M pos/s | 3229.5 | 40.4% |
| 256 | 106.5 | 2.40 M pos/s | 3481.0 | 43.5% |
| 512 | 178.6 | 2.87 M pos/s | 4151.7 | 51.9% |
| 1024 | 346.5 | 2.96 M pos/s | 4280.0 | 53.5% |
| 2048 | 689.1 | 2.97 M pos/s | 4304.0 | 53.8% |
| 4096 | 1361.6 | 3.01 M pos/s | 4356.3 | 54.5% |

## Kernel 2: qa_kv_proj_gemm

**功能**: 小投影 wq_a + wkv（合并为单次 GEMM）

**计算**: `[M, 7168] × [7168, 2048] → [M, 2048]`，输出 split 为 qr[1536] + kv[512]

**精度链路**: FP8(E4M3) input × FP8(E4M3) weight → FP32 累加 → BF16 输出，per-32-element E8M0 block scaling

**硬件选择**: SM100 TCGEN05 Block-Scaled MMA (`tcgen05.mma.block_scale`)。

**实现**: 调用 CUTLASS device API (`GemmUniversalAdapter`)，自动处理:
- TMA 异步加载 (global → smem)
- TMEM 累加器
- Warp Specialization (5 warp 角色分工)
- Block-scaled MMA (scale factor 内建于指令)
- Persistent tile scheduling (CLC)

**优化**:
- wq_a + wkv 权重合并: 省 1 次 kernel launch (~16 us)
- 输出用 view (narrow) 而非 contiguous: 省 memcpy (~10 us)
- KernelScheduleAuto: CUTLASS 自动选择最优 tile/pipeline 配置

**性能** (B300, sm_103, HBM_PEAK=8000 GB/s):

| M | Latency(us) | TFLOPS | BW(GB/s) | HBM Util |
|---|-------------|--------|-----------|----------|
| 16 | 17.0 | 27.63 | 874.2 | 10.9% |
| 32 | 16.9 | 55.73 | 892.1 | 11.2% |
| 64 | 16.2 | 115.78 | 949.0 | 11.9% |
| 128 | 16.5 | 227.31 | 975.1 | 12.2% |
| 256 | 17.1 | 440.67 | 1029.7 | 12.9% |
| 512 | 18.3 | 819.81 | 1115.1 | 13.9% |
| 1024 | 20.6 | 1460.50 | 1273.5 | 15.9% |
| 2048 | 32.9 | 1826.68 | 1146.8 | 14.3% |
