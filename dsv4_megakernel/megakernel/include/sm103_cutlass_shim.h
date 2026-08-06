#pragma once
// sm_103a (B300) CUTLASS arch-feature shim -- a no-op on CUTLASS >= 4.2.
//
// WHAT CUTLASS DOES
//   cutlass/arch/config.h maps the compiling arch to feature flags:
//     __CUDA_ARCH__ == 1030  ->  CUTLASS_ARCH_MMA_SM103A_ENABLED
//   and cute/arch/config.hpp forwards those onto the macros that gate every
//   inline-asm wrapper:
//     ... || defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || ...
//       -> CUTE_ARCH_TMA_SM90_ENABLED / CUTE_ARCH_TMA_SM100_ENABLED
//          CUTE_ARCH_TCGEN05_{TMEM,TF32_MMA,F16F32_MMA,MXF8F6F4_MMA,...}_ENABLED
//          CUTE_ARCH_{LDSM,STSM}_SM100A_ENABLED ...
//   With the gate off, each wrapper compiles to CUTE_INVALID_CONTROL_PATH,
//   i.e. `assert(0); printf(...)` -- it links fine and dies at runtime.
//
// WHY THIS FILE
//   The SM103A branch only exists from CUTLASS 4.2. On 4.0/4.1 an sm_103a pass
//   matches no branch, so TMA and tcgen05 are silently disabled; the observed
//   failure is
//     cute/arch/copy_sm90_desc.hpp:315 prefetch_tma_descriptor():
//     Assertion `0 && "... without CUTE_ARCH_TMA_SM90_ENABLED"` failed
//   B300 (SM103) implements the tcgen05/TMA instruction set CUTLASS gates
//   behind SM100A, so on older CUTLASS we opt into the SM100A feature set for
//   the sm_103a pass. CUTLASS writes its guards as `#if (!defined(X) && ...)`,
//   so predefining X is its own supported extension point, not a patch.
//
// TWO INVARIANTS -- both were found the hard way:
//
//   1. DEVICE PASS ONLY. cute's CUTE_HOST_DEVICE copy() helpers call
//      __device__-only synclog hooks from inside
//      `#if defined(CUTE_ARCH_TMA_SM90_ENABLED)`. Enabling that unconditionally
//      (a plain -DCUTLASS_ARCH_MMA_SM100A_ENABLED=1 on the command line) makes
//      the HOST pass illegal: "calling a __device__ function from a
//      __host__ __device__ function is not allowed" (32 errors in
//      copy_sm90_tma.hpp). Hence the __CUDA_ARCH__ guard below.
//
//   2. VERSION GATED. On CUTLASS >= 4.2 this file must do nothing. Defining
//      SM100A on top of CUTLASS's own SM103A would additionally light up
//      CUTE_ARCH_TCGEN05_S8_MMA_ENABLED and CUTE_ARCH_FLOAT2_MATH_ENABLED
//      (f32x2 PTX codegen), which are NOT part of the SM103A feature set --
//      a gratuitous codegen change. Gating on CUTLASS_VERSION keeps the
//      4.2 build bit-identical to before this file existed.
//
// USAGE: force-included ahead of every other header via
//        `-include .../include/sm103_cutlass_shim.h`, so it lands before any
//        CUTLASS header can latch its own gates. Only needed by the
//        CUTLASS-dependent kernels (front_mixed_gemm, wq_b_fp8_gemm,
//        mqa_logits_fp4).
#include <cutlass/version.h>

// CUTLASS_VERSION = MAJOR*100 + MINOR*10 + PATCH  (4.0.0 -> 400, 4.2.1 -> 421)
#if CUTLASS_VERSION < 420
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
#    ifndef CUTLASS_ARCH_MMA_SM100A_ENABLED
#      define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#    endif
#    ifndef CUTLASS_ARCH_MMA_SM100_ENABLED
#      define CUTLASS_ARCH_MMA_SM100_ENABLED 1
#    endif
#  endif
#endif
