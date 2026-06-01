#pragma once

#include <cuda.h>          // CUtensorMap, cuTensorMapEncodeTiled
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// mbarrier PTX wrappers (sm_90+)
// ============================================================

__device__ __forceinline__
void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                 :: "r"(addr), "r"(count));
}

// arrive 一次 + 期望 N 字节后续 tx_count（来自 TMA bulk）
__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(addr), "r"(bytes));
}

// 等待 phase 翻转（即一次"完整 barrier"完成）
__device__ __forceinline__
void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        "  .reg .pred  P1;\n"
        "  LAB_WAIT:\n"
        "    mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "    @P1 bra DONE;\n"
        "    bra LAB_WAIT;\n"
        "  DONE:\n"
        "}\n"
        :: "r"(addr), "r"(phase)
    );
}

// ============================================================
// TMA cp.async.bulk.tensor.2d (sm_90+)
//   smem  : destination smem ptr (16B 对齐)
//   desc  : CUtensorMap, 通常通过 __grid_constant__ 传入
//   coord0: 最快变化维（global tensor 的列方向）
//   coord1: 慢维（行方向）
//   bar   : mbarrier 对象，TMA 完成时自动更新其 tx_count
// ============================================================
__device__ __forceinline__
void cp_async_bulk_tensor_2d(
    void* smem, const CUtensorMap* desc,
    int32_t coord0, int32_t coord1,
    uint64_t* bar
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t bar_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(smem_addr), "l"(desc), "r"(coord0), "r"(coord1), "r"(bar_addr)
        : "memory"
    );
}

// ============================================================
// Host helper: 构造 [rows, cols] row-major tensor 的 2D tile descriptor
//   box = [box_rows, box_cols]
//   swizzle: NONE / 32B / 64B / 128B
// ============================================================
inline CUtensorMap make_tma_2d_desc(
    const __half* gptr, int32_t rows, int32_t cols,
    int32_t box_rows, int32_t box_cols,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE
) {
    CUtensorMap desc{};
    // TMA 维度顺序：[0] = 最快变化（列），[1] = 慢（行）
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(__half)};  // row stride，bytes
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,                              // rank
        (void*)gptr,
        globalDim, globalStride,
        boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (err != CUDA_SUCCESS) {
        const char* errstr = nullptr;
        cuGetErrorString(err, &errstr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errstr);
    }
    return desc;
}
