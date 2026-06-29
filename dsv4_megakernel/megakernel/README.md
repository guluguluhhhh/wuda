# DSV4 Attention Megakernel

DeepSeek-V4 Attention 的 CUDA kernel 实现。

## 运行

```bash
cd /home/admin/workspace/aop_lab/app_source/jinhua/dsv4_megakernel/megakernel/test

# Phase 1 (mHC): benchmark
python test_hc_fused.py --benchmark

# Phase 1 (mHC): intra-kernel profile (生成 Chrome Trace JSON)
python test_hc_fused.py --profile

# Phase 2 (投影 TC GEMM): benchmark
python test_phase2_proj.py --benchmark
```

