#pragma once
// ============================================================
// hc_fused_kernel_tc.cuh
// Shared constants + tiny device helpers for the MHC fused epilogue
// (kernels/hc_fused_kernel_tc.cu).
//
// HISTORY: this header used to own a full tcgen05 tf32 split-K GEMM engine
// for the mix projection X[M,28672] @ W[24,28672]^T. That path was DELETED
// when the production form switched to deep_gemm.tf32_hc_prenorm_gemm
// (identical split-K workspace layout, measured 2-3us faster on B300); the
// engine lives in git history if ever needed again.
// ============================================================

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace hc_tc {

// ---- problem constants (shared with the epilogue) ----
static constexpr int HC = 4;
static constexpr int DIM = 7168;
static constexpr int K_DIM = HC * DIM;          // 28672
static constexpr int N_OUT = 24;
static constexpr int SINKHORN_ITERS = 20;
static constexpr int EPILOGUE_THREADS = 256;

// clock64 profiling stamp slots (block 0; see the epilogue kernel).
static constexpr int PROF_SLOTS = 8;

namespace ptx {

__device__ __forceinline__ long long rdclock() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
    return t;
}

}  // namespace ptx

}  // namespace hc_tc
