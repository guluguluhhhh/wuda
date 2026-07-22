#pragma once
// ============================================================
// mqa_logits_fp4.cuh — SELF-CONTAINED DSV4 score-attention (Sparse Top-K
// Indexer index-score / "lightning indexer" logits).
//
// This is DeepGEMM's sm100_fp4_mqa_logits kernel + its minimal helper closure,
// INLINED into a single header so it depends ONLY on the repo's CUTLASS/CuTe
// (megakernel/cutlass) — no `deep_gemm` package, no include/dg tree — same
// self-contained style as w1_merged_fp8_gemm.cuh. The deep_gemm::* namespaces
// are kept verbatim (cosmetic; not a package dependency).
//
// Three AOT edits vs upstream, marked `// [MEGAKERNEL EDIT]`:
//   1. kernel `kNumSMs` template param -> gridDim.x
//   2. cudaGridDependencySynchronize() (PDL-only) neutralized
//   3. decode tile-pool scheduler (`kTilePool`): the GLOBAL pool of
//      Σ_b cdiv(ke[b]-ks[b], BLOCK_KV) KV tiles is split into gridDim.x balanced
//      contiguous chunks (may cross token boundaries) — inline equivalent of
//      DeepGEMM's paged-path metadata schedule, for decode where B < #SMs.
//   4. math register diet (224 -> <=128/thread, BIT-EXACT results): weights are
//      read as float2 from smem inside the reduce (no register cache) and TMEM is
//      consumed in two 32-head passes reusing one accum[32] — prepares the future
//      TPB=512 CUDA-core tail (65536/512 = 128 architectural register cap).
//   5. TPB 384 -> 512: CUDA-core tail warpgroup (warps 12-15) hides the wq_b
//      per-head WEIGHTLESS RMSNorm (head_dim=512) under the KV stream — the
//      gemm_fuse_norm_b TC/CC dual-path pattern. Fully decoupled (no shared
//      barriers); idle when rms_y == nullptr.
//
// Math: logits[t] = Σ_h relu(<iq[h,:],kvc[t,:]>)·weights[h]  (fp4 UMMA + cuda reduce)
// Host launcher + PyTorch binding: kernels/mqa_logits_fp4.cu
//
// Only the helper symbols this kernel actually uses are inlined below. The unused
// remainder of the vendored DeepGEMM helper closure is parked (not compiled) under
// include/mqa_unused/ — restore from there if ever needed.
// ============================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/numeric_types.h>

// ============================================================
// inlined from deep_gemm/common/compile.cuh
// ============================================================

#include <cutlass/detail/helper_macros.hpp>

#if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or defined(__CLION_IDE__)
#define DG_IN_CUDA_COMPILATION
#endif

#if defined(__NVCC__) || (defined(__clang__) and defined(__CUDA__))
#define CUTLASS_HOST_DEVICE_NOINLINE  __device__ __host__
#define CUTLASS_DEVICE_NOINLINE __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE_NOINLINE __device__
#define CUTLASS_DEVICE_NOINLINE __device__
#else
#define CUTLASS_HOST_DEVICE_NOINLINE
#define CUTLASS_DEVICE_NOINLINE
#endif

// ============================================================
// inlined from deep_gemm/common/exception.cuh
// ============================================================

#include <cuda/std/cstdint>

#ifdef __CLION_IDE__

CUTLASS_HOST_DEVICE void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_UNIFIED_ASSERT
#ifdef DG_IN_CUDA_COMPILATION
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) DG_HOST_ASSERT(cond)
#endif
#endif

// ============================================================
// inlined from deep_gemm/common/math.cuh
// ============================================================

#include <cuda/std/cstdint>

namespace deep_gemm::math {

/// Math functions
template <typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T, bool kDoCeilAlignment = true>
CUTLASS_HOST_DEVICE T align(T a, T b) {
    return (kDoCeilAlignment ? ceil_div(a, b) : (a / b)) * b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}


} // namespace deep_gemm

// ============================================================
// inlined from deep_gemm/common/utils.cuh
// ============================================================

#include <cuda/std/cstdint>


namespace deep_gemm::utils {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    CUTLASS_HOST_DEVICE
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    CUTLASS_HOST_DEVICE
    auto operator [](const uint32_t& i) const {
        return func(i);
    }
};

template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

template <uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols() {
    DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <=  32) return  32;
    if constexpr (kNumCols <=  64) return  64;
    if constexpr (kNumCols <= 128) return 128;
    if constexpr (kNumCols <= 256) return 256;
    return 512;
}

} // namespace deep_gemm::utils

// ============================================================
// inlined from deep_gemm/common/cute_tie.cuh
// ============================================================

#include <cute/int_tuple.hpp>

namespace cute {

struct ignore_t {
    template <typename T>
    constexpr const ignore_t& operator=(T&&) const noexcept {
        return *this;
    }
};

inline constexpr ignore_t ignore{};

} // namespace cute

#define CUTE_TIE_CONCAT_IMPL(A, B) A##B
#define CUTE_TIE_CONCAT(A, B) CUTE_TIE_CONCAT_IMPL(A, B)

#define CUTE_TIE_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define CUTE_TIE_COUNT_ARGS(...) \
    CUTE_TIE_GET_NTH_ARG(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

#define CUTE_TIE_OP_DECL(I, TUPLE, VAR) auto VAR = ::cute::get<I>(TUPLE)
#define CUTE_TIE_OP_ASSIGN(I, TUPLE, VAR) VAR = ::cute::get<I>(TUPLE)

#define CUTE_TIE_APPLY_OP_1(OP, T, V1) OP(0, T, V1);
#define CUTE_TIE_APPLY_OP_2(OP, T, V1, V2) OP(0, T, V1); OP(1, T, V2);
#define CUTE_TIE_APPLY_OP_3(OP, T, V1, V2, V3) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3);
#define CUTE_TIE_APPLY_OP_4(OP, T, V1, V2, V3, V4) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4);
#define CUTE_TIE_APPLY_OP_5(OP, T, V1, V2, V3, V4, V5) OP(0, T, V1); OP(1, T, V2); OP(2, T, V3); OP(3, T, V4); OP(4, T, V5);

#define CUTE_TIE_DECL(TUPLE_EXPR, ...) \
    auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
    CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
        CUTE_TIE_OP_DECL, \
        CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), \
        __VA_ARGS__ \
    )

#define CUTE_TIE(TUPLE_EXPR, ...) \
    do { \
        auto&& CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__) = (TUPLE_EXPR); \
        CUTE_TIE_CONCAT(CUTE_TIE_APPLY_OP_, CUTE_TIE_COUNT_ARGS(__VA_ARGS__)) ( \
            CUTE_TIE_OP_ASSIGN, \
            CUTE_TIE_CONCAT(cute_tie__temp_tuple_, __LINE__), \
            __VA_ARGS__ \
        ); \
    } while (0)

// ============================================================
// inlined from deep_gemm/ptx/utils.cuh
// ============================================================

#include <cuda/std/cstdint>
#include <cuda_bf16.h>


namespace deep_gemm::ptx {

CUTLASS_DEVICE uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

// [MEGAKERNEL EDIT] Device-wide nanosecond timer (%globaltimer): ONE clock for the
// whole GPU, so stamps from different warps/CTAs are directly comparable — used to
// SEE the score-attention path and the CUDA-core RMSNorm tail overlap (the
// gemm_fuse_norm_b profiling pattern), instead of inferring it from time deltas.
// NOTE the "memory" clobber: without it the compiler may hoist the (independent)
// first loads ABOVE the timer read — the stamp then records "first data arrived"
// instead of "reached this line" (observed as a fake ~5us late tail start under
// the post-L2-flush DRAM writeback storm). The clobber pins the stamp in program
// order; prof-guarded and outside hot loops, so codegen of real work is untouched.
CUTLASS_DEVICE unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) : : "memory");
    return t;
}

CUTLASS_DEVICE void sync_aligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

CUTLASS_DEVICE void sync_unaligned(const uint32_t& num_threads, const uint32_t& barrier_idx) {
    asm volatile("barrier.sync %0, %1;" : : "r"(barrier_idx), "r"(num_threads));
}

template <typename dtype_t>
CUTLASS_DEVICE dtype_t exchange(dtype_t ptr, const uint32_t& src_lane_idx) {
    DG_STATIC_ASSERT(sizeof(dtype_t) % sizeof(uint32_t) == 0, "");
    const auto send_int_values = reinterpret_cast<uint32_t*>(&ptr);
    dtype_t recv_dtype;
    auto recv_int_values = reinterpret_cast<uint32_t*>(&recv_dtype);
    #pragma unroll
    for (uint32_t i = 0; i < sizeof(dtype_t) / sizeof(uint32_t); ++ i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], static_cast<int>(src_lane_idx));
    return recv_dtype;
}

CUTLASS_DEVICE void accumulate(float2& a, nv_bfloat162 b) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    // Use `add.rn.f32.bf16` instruction to perform fused (cast + add) operation on SM100
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.x) : "h"(*reinterpret_cast<uint16_t*>(&b.x)));
    asm("add.rn.f32.bf16 %0, %1, %0;\n" : "+f"(a.y) : "h"(*reinterpret_cast<uint16_t*>(&b.y)));
#else
    const auto [x, y] = __bfloat1622float2(b);
    a.x += x, a.y += y;
#endif
}

} // namespace deep_gemm::ptx

// ============================================================
// inlined from deep_gemm/ptx/ld_st.cuh
// ============================================================

#include <cuda/std/cstdint>
#include <cuda_bf16.h>

namespace deep_gemm::ptx {
/// Shared memory
CUTLASS_DEVICE uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(__cvta_generic_to_shared(ptr)));
    return ret;
}

CUTLASS_DEVICE void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val));
}

CUTLASS_DEVICE void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "f"(val.x), "f"(val.y));
}

CUTLASS_DEVICE void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y));
}

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

CUTLASS_DEVICE void st_shared(const __int128_t* ptr, __int128_t val) {
    asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(ptr)), "q"(val));
}
} // namespace deep_gemm::ptx

// ============================================================
// inlined from deep_gemm/ptx/tcgen05.cuh
// ============================================================

namespace deep_gemm::ptx {

struct SM100_MMA_MXF4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

/// Tensor memory operations
CUTLASS_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTLASS_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

} // namespace deep_gemm::ptx

// ============================================================
// inlined from deep_gemm/common/tma_copy.cuh
// ============================================================

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cutlass/arch/barrier.h>


namespace deep_gemm::tma {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();

    if constexpr (not kIs3DTMA) {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                             inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                      inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                    }
                }
            #endif
        }
    } else {
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                            inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
            }
        } else {
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                // 2-CTA function will send signals to the leader CTA only
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                      inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                }
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                    }
                }
            #endif
        }
    }
}

} // namespace deep_gemm::tma

// ============================================================
// inlined from deep_gemm/mma/sm100.cuh
// ============================================================

#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>


namespace deep_gemm::mma::sm100 {

/// Shared memory descriptor
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_smem_desc(cute::UMMA::LayoutType layout, void* smem_ptr,
                                          const uint32_t& stride_byte_offset, const uint32_t& leading_byte_offset) {
    cute::UMMA::SmemDescriptor desc;

    // Set the version for SM100
    desc.version_ = 1;

    // Legacy mode
    desc.lbo_mode_ = 0;

    // Layout
    desc.layout_type_ = static_cast<uint8_t>(layout);

    // Start address
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);

    // Base offset
    desc.base_offset_ = 0;

    // SBO and LBO
    desc.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;

    return desc;
}

CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr) {
    // NOTES: the UTCCP layout is K-major by default
    // Atom size: 8 x 128 bits
    // {SBO, LBO} means the byte stride between atoms on {MN, K}
    // Since the UTCCP we used is 128b-wide (only 1 atom on K), so LBO can be zero
    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

CUTLASS_DEVICE
void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr) {
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}
CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

} // namespace deep_gemm::mma::sm100

// ============================================================
// inlined from deep_gemm/impls/sm100_fp4_mqa_logits.cuh
// ============================================================
// ============================================================
// Kernel body from DeepGEMM (deep_gemm/impls/sm100_fp4_mqa_logits.cuh); its
// helper closure is inlined above in this same header, so the DSV4 score-attention
// (index score / "lightning indexer" logits) depends only on the repo's
// CUTLASS/CuTe — no deep_gemm package, no separate include tree.
//
// Three surgical edits for standalone ahead-of-time (nvcc) build, all marked
// with `// [MEGAKERNEL EDIT]` below:
//   1. template param `kNumSMs` removed -> use `gridDim.x` (JIT baked it in
//      DeepGEMM; AOT can't, and gridDim.x always equals the launch grid).
//   2. `cudaGridDependencySynchronize()` neutralized (PDL-only; we launch
//      standalone, not as a programmatic-dependent-launch megakernel).
//   3. `kTilePool` decode schedule: all 4 warp roles enumerate tasks through a
//      shared `for_each_task` closure; kTilePool=false reproduces the original
//      per-q-block enumeration byte-identically (faithful single-seq path).
// Everything else is byte-identical to the proven DeepGEMM kernel.
// ============================================================

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>


namespace deep_gemm {

// ============================================================
// [MEGAKERNEL EDIT] Decode tile-pool scheduler (kTilePool=true path).
//
// Decode-only geometry: seq_len = B tokens (BLOCK_Q == 1), token b owns the KV
// window [ks[b], ke[b]) inside the flat kv[B*T]. The GLOBAL pool of
// Σ_b cdiv(window_b, BLOCK_KV) tiles is split into gridDim.x balanced contiguous
// chunks (per-CTA imbalance <= 1 tile) that may cross token boundaries — the
// inline equivalent of DeepGEMM's paged-path metadata kernel (which emits per-SM
// (q_token_idx, kv_split_idx) starts), but DEVICE-ONLY and launch-free
// (megakernel-ready: no host/Python in the loop).
//
// Cost model: ONE warp per CTA builds the tile prefix sum into smem with a
// warp-parallel scan (O(B/32) rounds, see the kernel prologue), published by the
// existing __syncthreads(); each thread then locates its chunk with a log2(B)
// binary search on smem. This replaces the previous per-thread O(B) serial
// global-memory walk, which dominated small-T large-B decode (measured
// ~0.11us/token slope vs DeepGEMM's metadata-driven ~0.02).
// ============================================================
static constexpr uint32_t kNumMaxTilePoolTokens = 512;   // decode B cap (serving: 32-256)

template <uint32_t BLOCK_KV>
struct TilePoolScheduler {
    const uint32_t* ks;
    const uint32_t* ke;
    const uint32_t* tile_prefix;     // smem, [num_tokens + 1], built by the warp scan
    uint32_t num_tokens, seq_len_kv;
    uint32_t chunk_cur, chunk_end;   // this CTA's global tile ids: [chunk_cur, chunk_end)
    uint32_t token;                  // current token (tile_prefix[token] <= chunk_cur)

    CUTLASS_DEVICE TilePoolScheduler(const uint32_t& cta_idx, const uint32_t& num_ctas,
                                     const uint32_t& num_tokens, const uint32_t& seq_len_kv,
                                     const uint32_t* ks, const uint32_t* ke,
                                     const uint32_t* tile_prefix):
            ks(ks), ke(ke), tile_prefix(tile_prefix), num_tokens(num_tokens), seq_len_kv(seq_len_kv) {
        // Balanced contiguous partition: first `total % num_ctas` CTAs take one extra tile
        const uint32_t total = tile_prefix[num_tokens];
        const uint32_t per = total / num_ctas, rem = total % num_ctas;
        chunk_cur = cta_idx * per + cute::min(cta_idx, rem);
        chunk_end = chunk_cur + per + (cta_idx < rem ? 1u : 0u);
        // Binary search: largest token with tile_prefix[token] <= chunk_cur
        uint32_t lo = 0, hi = num_tokens;
        while (lo < hi) {
            const uint32_t mid = (lo + hi + 1) / 2;
            tile_prefix[mid] <= chunk_cur ? (lo = mid) : (hi = mid - 1);
        }
        token = lo;
    }

    // Emits one (token, contiguous KV tile sub-range) task per call. Deterministic:
    // every warp role constructs its own scheduler and sees the SAME task sequence,
    // keeping the Q/KV/TMEM pipelines in lock-step (as with the legacy enumeration).
    CUTLASS_DEVICE bool next(uint32_t& q_idx, uint32_t& kv_start, uint32_t& num_kv_blocks,
                             uint32_t& seq_k_start, uint32_t& seq_k_end) {
        while (chunk_cur < chunk_end and token < num_tokens) {
            const uint32_t tile_base = tile_prefix[token];
            const uint32_t tile_end  = cute::min(tile_prefix[token + 1], chunk_end);
            if (tile_end > chunk_cur) {
                q_idx = token;
                seq_k_start = cute::min(ks[token], seq_len_kv);
                seq_k_end = cute::min(ke[token], seq_len_kv);
                kv_start = seq_k_start / 4 * 4 + (chunk_cur - tile_base) * BLOCK_KV;
                num_kv_blocks = tile_end - chunk_cur;
                chunk_cur = tile_end;
                ++ token;
                return true;
            }
            ++ token;   // empty-window token (or fully before the chunk); skip
        }
        return false;
    }
};

template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsCompressedLogits, bool kTilePool,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          // [MEGAKERNEL EDIT] `kNumSMs` template param removed; see gridDim.x below.
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          // [MEGAKERNEL EDIT] CUDA-core tail warpgroup (0 disables the branch entirely)
          uint32_t kNumTailThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads + kNumTailThreads, 1)
void sm100_fp4_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t max_seqlen_k,
                          const uint32_t logits_stride,
                          const uint32_t* cu_seq_len_k_start,
                          const uint32_t* cu_seq_len_k_end,
                          logits_dtype_t* logits,
                          // [MEGAKERNEL EDIT] hidden wq_b per-head RMSNorm (tail warps);
                          // rms_y == nullptr -> tail idles. groups = numel / 512.
                          const float* rms_y, nv_bfloat16* rms_out,
                          const uint32_t rms_num_groups, const float rms_eps,
                          // [MEGAKERNEL EDIT] tail head-start experiment knob: the tail
                          // spins this long (ns) before touching memory, letting the KV
                          // pipeline fill uncontended. 0 = start immediately.
                          const uint32_t rms_delay_ns,
                          // [MEGAKERNEL EDIT] globaltimer stamps [gridDim.x][4]:
                          // 0=attention-path start (post-prologue), 1=attention end,
                          // 2=tail start (at CTA start), 3=tail end.
                          // nullptr = off (gemm_fuse_norm_b pattern).
                          unsigned long long* prof,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Utils
    const auto sm_idx = blockIdx.x;
    // [MEGAKERNEL EDIT] was a template param; AOT build uses the launch grid.
    const uint32_t kNumSMs = gridDim.x;
    // [MEGAKERNEL EDIT] legacy grid.y KV carve (kTilePool=false only; decode launches
    // grid.y=1 and uses the tile-pool scheduler instead). logits[t] are per-slot
    // independent (no cross-kv reduction / softmax), so split writes go direct — no combine.
    const uint32_t kv_split      = blockIdx.y;
    const uint32_t num_kv_splits = gridDim.y;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // [MEGAKERNEL EDIT] CUDA-core tail warpgroup (warps 12-15), hoisted ABOVE the
    // whole prologue — gemm_fuse_norm_b discipline: the attention path and the tail
    // NEVER share a block-wide barrier; each side handles its own sync. The tail
    // touches no smem / mbarrier / TMEM state, so it starts the instant the CTA
    // does: it must NOT wait for barrier init, TMEM alloc, or the tile-prefix scan
    // warp (whose serial ks/ke global reads take ~us on a cold L2). The remaining
    // 384 threads publish the prologue via a role-scoped NamedBarrier below.
    // Work: wq_b per-head WEIGHTLESS RMSNorm  q *= rsqrt(mean(q^2) + eps),
    // head_dim = 512, fp32 in -> bf16 out, hidden under this kernel's KV stream.
    // Tuned for the two MEASURED bottlenecks:
    //  * solo bandwidth is in-flight-bytes limited (Little's law): 4 groups in
    //    flight = 8KB/warp outstanding, ~2x the 2-group ceiling of ~4TB/s;
    //  * the long-T interference tax is write-turnaround driven: bf16 results
    //    are packed 4-per-lane into ONE 8B st.v2 (256B full-line bursts per
    //    warp instruction) instead of 16 half-line 64B scalar stores per group.
    // Layout: lane owns 4 CONSECUTIVE elements per 128-elem segment
    // (elem = e*128 + lane*4 + i) -> float4 loads, still perfectly coalesced.
    // ~90 regs/thread (64 data + control), under the 128 cap. Remainder groups
    // load a duplicated (discarded) partner; their stores are guarded off.
    if (kNumTailThreads > 0 and warp_idx >= kSpecWarpStart + 4) {
        const bool prof_lane = prof != nullptr and warp_idx == kSpecWarpStart + 4 and lane_idx == 0;
        if (prof_lane)
            prof[blockIdx.x * 4 + 2] = ptx::globaltimer();
        // Optional deferred start: park the tail (no memory traffic) so the KV
        // pipeline's latency-critical FILL phase runs uncontended — tests the
        // "tail head-start slows attention" hypothesis / doubles as a pacing knob.
        if (rms_delay_ns > 0) {
            const unsigned long long t_go = ptx::globaltimer() + rms_delay_ns;
            while (ptx::globaltimer() < t_go)
                ;
        }
        if (rms_y != nullptr) {
            constexpr uint32_t kRmsHeadDim = 512;
            constexpr uint32_t kInFlight = 4;      // groups in flight per warp
            const uint32_t tail_warp = warp_idx - (kSpecWarpStart + 4);
            const uint32_t num_tail_warps = kNumTailThreads / 32;
            const uint32_t stride = gridDim.x * num_tail_warps;
            for (uint32_t g0 = blockIdx.x * num_tail_warps + tail_warp; g0 < rms_num_groups;
                 g0 += kInFlight * stride) {
                float4 v[kInFlight][4];
                uint32_t g[kInFlight];
                bool valid[kInFlight];
                // Issue ALL 16 float4 loads before any dependent math
                #pragma unroll
                for (uint32_t p = 0; p < kInFlight; ++ p) {
                    g[p] = g0 + p * stride;
                    valid[p] = g[p] < rms_num_groups;
                    const auto src = reinterpret_cast<const float4*>(
                        rms_y + (uint64_t)(valid[p] ? g[p] : g0) * kRmsHeadDim) + lane_idx;
                    #pragma unroll
                    for (uint32_t e = 0; e < 4; ++ e)
                        v[p][e] = __ldcs(src + e * 32);   // evict-first reads
                }
                float sq[kInFlight];
                #pragma unroll
                for (uint32_t p = 0; p < kInFlight; ++ p) {
                    sq[p] = 0.f;
                    #pragma unroll
                    for (uint32_t e = 0; e < 4; ++ e) {
                        const float4& f = v[p][e];
                        sq[p] += f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
                    }
                }
                #pragma unroll
                for (uint32_t o = 16; o > 0; o >>= 1) {   // 4 interleaved shfl chains (ILP)
                    #pragma unroll
                    for (uint32_t p = 0; p < kInFlight; ++ p)
                        sq[p] += __shfl_xor_sync(0xffffffffu, sq[p], o);
                }
                #pragma unroll
                for (uint32_t p = 0; p < kInFlight; ++ p) {
                    if (not valid[p])
                        continue;
                    const float rms = rsqrtf(sq[p] / float(kRmsHeadDim) + rms_eps);
                    const auto dst = rms_out + (uint64_t)g[p] * kRmsHeadDim + lane_idx * 4;
                    #pragma unroll
                    for (uint32_t e = 0; e < 4; ++ e) {
                        const float4& f = v[p][e];
                        const auto ab = __floats2bfloat162_rn(f.x * rms, f.y * rms);
                        const auto cd = __floats2bfloat162_rn(f.z * rms, f.w * rms);
                        asm volatile("st.global.cs.v2.b32 [%0], {%1, %2};"
                                     :: "l"(dst + e * 128),
                                        "r"(*reinterpret_cast<const uint32_t*>(&ab)),
                                        "r"(*reinterpret_cast<const uint32_t*>(&cd)) : "memory");
                    }
                }
            }
        }
        if (prof_lane)
            prof[blockIdx.x * 4 + 3] = ptx::globaltimer();
        return;
    }

    // Prefetch TMA descriptors
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    // UMMA configs
    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = 64;
    static constexpr uint32_t kNumSFQ  = math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
    static constexpr uint32_t kNumSFKV = math::constexpr_align(BLOCK_KV, kNumUTCCPAlignedElems);
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarpGroups * UMMA_M and BLOCK_KV % kNumUTCCPAlignedElems == 0, "Invalid `BLOCK_KV`");
    DG_STATIC_ASSERT(not kTilePool or BLOCK_Q == 1, "Tile pool schedule assumes 1 decode token per q-block");

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE      = BLOCK_Q * kNumHeads * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE   = kNumSFQ * sizeof(int);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE     = BLOCK_KV * (kHeadDim / 2);
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE  = kNumSFKV * sizeof(int);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE  % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q and KV data on shared memory
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i;
    });
    const auto smem_sf_ptr = smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
    auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * i);
    });
    auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_sf_ptr + SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages + SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages
                                                    + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });

    // Barriers and TMEM pointer on shared memory
    const auto barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + kNumKVStages + i; });
    const auto tmem_barrier_ptr = barrier_ptr + kNumQStages * 2 + kNumKVStages * 2;
    auto full_tmem_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + i; });
    auto empty_tmem_barriers = utils::PatternVisitor([&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
    auto tmem_ptr_in_smem    = reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);
    // [MEGAKERNEL EDIT] tile-pool prefix scratch ([kNumMaxTilePoolTokens + 1] u32)
    // lives right after the TMEM pointer; sized in the host's compute_smem_bytes().
    auto smem_tile_prefix    = tmem_ptr_in_smem + 1;

    // Tensor memory configs
    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // Initialize barriers
    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            full_tmem_barriers[i]->init(1);
            empty_tmem_barriers[i]->init(128);
        }
        cutlass::arch::fence_barrier_init();
    }

    // Allocate tensor memory
    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);

    // [MEGAKERNEL EDIT] tile-pool metadata: one (otherwise idle) warp builds the tile
    // prefix sum in smem via a warp-parallel scan; the __syncthreads() below publishes
    // it to all roles. Device-only — the fused equivalent of DeepGEMM's separate
    // metadata kernel launch (no host/Python in the loop; megakernel-compatible).
    if constexpr (kTilePool) {
        if (warp_idx == kSpecWarpStart + 3) {
            DG_TRAP_ONLY_DEVICE_ASSERT(seq_len <= kNumMaxTilePoolTokens);
            uint32_t running = 0;
            if (lane_idx == 0)
                smem_tile_prefix[0] = 0;
            for (uint32_t base = 0; base < seq_len; base += 32) {
                const uint32_t b = base + lane_idx;
                uint32_t num_tiles = 0;
                if (b < seq_len) {
                    // Same tile geometry as the legacy path: base aligned down to 4 (SF TMA)
                    const uint32_t s = cute::min(cu_seq_len_k_start[b], seq_len_kv) / 4 * 4;
                    const uint32_t e = cute::min(cu_seq_len_k_end[b], seq_len_kv);
                    num_tiles = e > s ? math::ceil_div(e - s, BLOCK_KV) : 0u;
                }
                // Inclusive warp scan (Hillis-Steele over shfl_up)
                uint32_t prefix = num_tiles;
                #pragma unroll
                for (uint32_t d = 1; d < 32; d <<= 1) {
                    const uint32_t v = __shfl_up_sync(0xffffffffu, prefix, d);
                    if (lane_idx >= d)
                        prefix += v;
                }
                if (b < seq_len)
                    smem_tile_prefix[b + 1] = running + prefix;
                running += __shfl_sync(0xffffffffu, prefix, 31);
            }
        }
    }
    // [MEGAKERNEL EDIT] role-scoped prologue publish (barrier init / TMEM ptr /
    // tile prefix) for the 384 attention-path threads ONLY — the tail warpgroup
    // exited above and must not be made to wait here (gemm_fuse_norm_b: the two
    // sides never share a block-wide barrier, so __syncthreads is forbidden).
    // NamedBarrier id 1; id 0 is the math-only barrier at the epilogue.
    cutlass::arch::NamedBarrier(kNumSpecializedThreads + kNumMathThreads, 1).sync();

    // [MEGAKERNEL EDIT] attention-path start stamp (post-prologue; the tail's t2 is
    // stamped at CTA start above, so t2 <= t0 is expected on the timeline)
    if (prof != nullptr and threadIdx.x == 0)
        prof[blockIdx.x * 4 + 0] = ptx::globaltimer();

    // Scheduler
    const uint32_t num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    auto load_schedule = [&](const uint32_t& q_idx) -> cute::tuple<uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto row_idx = cute::min(q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cute::min(cu_seq_len_k_start[row_idx], seq_len_kv);
            seq_k_end[i] = cute::min(cu_seq_len_k_end[row_idx], seq_len_kv);
            start = cute::min(start, seq_k_start[i]);
            end = cute::max(end, seq_k_end[i]);
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        // [MEGAKERNEL EDIT] carve this CTA's contiguous KV-block sub-range out of the
        // full [start, end) run (whole BLOCK_KV blocks -> boundaries stay aligned; a
        // tail split with 0 blocks is safe). seq_k_start/end[i] stay the FULL per-row
        // ranges above, so the store guard / compressed offset remain correct.
        const uint32_t total_blocks = math::ceil_div(end - start, BLOCK_KV);
        const uint32_t bps = math::ceil_div(total_blocks, num_kv_splits);
        const uint32_t s0  = kv_split * bps;
        const uint32_t s1  = cute::min((kv_split + 1) * bps, total_blocks);
        return {start + s0 * BLOCK_KV, s1 > s0 ? (s1 - s0) : 0u};
    };

    // [MEGAKERNEL EDIT] unified task enumeration for all 4 warp roles:
    //  - kTilePool=false: one CTA per q-block via grid.x stride + grid.y KV carve
    //    (byte-identical schedule to the original kernel; faithful single-seq path).
    //  - kTilePool=true: decode; gridDim.x CTAs split the GLOBAL KV tile pool into
    //    balanced contiguous chunks (see TilePoolScheduler above). All roles see the
    //    same task sequence, so the Q/KV/TMEM pipelines stay in lock-step.
    auto for_each_task = [&](auto&& fn) {
        if constexpr (kTilePool) {
            TilePoolScheduler<BLOCK_KV> sched(sm_idx, kNumSMs, seq_len, seq_len_kv,
                                              cu_seq_len_k_start, cu_seq_len_k_end,
                                              smem_tile_prefix);
            uint32_t q_idx, kv_start, num_kv_blocks;
            while (sched.next(q_idx, kv_start, num_kv_blocks, seq_k_start[0], seq_k_end[0]))
                fn(q_idx, kv_start, num_kv_blocks);
        } else {
            for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
                CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);
                fn(q_idx, kv_start, num_kv_blocks);
            }
        }
    };

    // Make Q, KV and TMEM pipeline
    auto make_pipeline = [](const uint32_t& num_stages) {
        // Return current stage and phase, and advance pipeline by steps
        return [iter_idx = 0u, num_stages](const uint32_t& step = 1) mutable -> cute::tuple<uint32_t, uint32_t> {
            uint32_t current_idx = iter_idx;
            iter_idx += step;
            return {current_idx % num_stages, (current_idx / num_stages) & 1};
        };
    };
    auto advance_q_pipeline    = make_pipeline(kNumQStages);
    auto advance_kv_pipeline   = make_pipeline(kNumKVStages);
    auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

    // Register reconfigurations
    // [MEGAKERNEL EDIT] math target lowered 224 -> 128 after the register diet
    // (weights from smem + two-pass accum[32]); setmaxnreg.inc requires the
    // compiled per-thread count <= 128 — verify via `--ptxas-options=-v` in the
    // test build; if ptxas reports more, raise this to the next multiple of 8.
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 128;

    // Wait for primary kernel completion
    // [MEGAKERNEL EDIT] PDL-only; neutralized for standalone launch.
    // cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        // TMA warp for loading Q
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Enumerate assigned tasks (Q/SF/weights loaded once per token per CTA)
        if (cute::elect_one_sync()) {
            for_each_task([&](const uint32_t& q_idx, const uint32_t&, const uint32_t&) {
                // Wait Q consumer release
                CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                cute::SM90_TMA_LOAD_2D::copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[q_stage_idx]),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_q[q_stage_idx], 0, q_idx * BLOCK_Q * kNumHeads);
                tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, full_q_barriers[q_stage_idx], smem_sf_q[q_stage_idx], 0, q_idx * BLOCK_Q);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[q_stage_idx], smem_weights[q_stage_idx], 0, q_idx * BLOCK_Q);
                full_q_barriers[q_stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + kRealNumSFQ * sizeof(int) + SMEM_WEIGHT_SIZE_PER_STAGE);
            });
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        // TMA warp for loading KV cache
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if (cute::elect_one_sync()) {
            // Enumerate assigned (token, KV tile sub-range) tasks
            for_each_task([&](const uint32_t& q_idx, const uint32_t& kv_start, const uint32_t& num_kv_blocks) {
                // Enumerate KV blocks
                for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                    // Wait KV consumer release
                    CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    cute::SM90_TMA_LOAD_2D::copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                                                 static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                 smem_kv[kv_stage_idx], 0, kv_start + kv_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
                                              smem_sf_kv[kv_stage_idx],
                                              kv_start + kv_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
            });
        }
    } else if (warp_idx == kSpecWarpStart + 2) {
        // UMMA warp
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        // Make UMMA desc
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e2m1_t, cutlass::float_e2m1_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        // Enumerate assigned tasks
        for_each_task([&](const uint32_t& q_idx, const uint32_t&, const uint32_t& num_kv_blocks) {
            // Wait TMA Q arrivals
            CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Transpose and copy SF Q
            #pragma unroll
            for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                utccp_required_smem_warp_transpose(smem_ptr);
                cutlass::arch::fence_view_async_shared();
                mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                if (cute::elect_one_sync())
                    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                __syncwarp();
            }

            // Enumerate KV blocks
            for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                // Wait TMA KV arrivals
                CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Transpose
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                    cutlass::arch::fence_view_async_shared();
                }

                // UMMA with SF
                if (cute::elect_one_sync()) {
                    // Copy SF KV
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        // Wait TMEM release
                        CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        // Issue UMMA with SF
                        #pragma unroll
                        for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                            auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * 2, k * 2);
                            // TODO: generalize umma desc
                            DG_STATIC_ASSERT(kHeadDim == 128, "Invalid head dim");
                            auto a_desc = mma::sm100::make_smem_desc(
                                cute::UMMA::LayoutType::SWIZZLE_64B,
                                smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) + k * UMMA_K / 2,
                                8 * (kHeadDim / 2), 0);
                            auto b_desc = mma::sm100::make_smem_desc(
                                cute::UMMA::LayoutType::SWIZZLE_64B,
                                smem_q[q_stage_idx] + k * UMMA_K / 2,
                                8 * (kHeadDim / 2), 0);
                            ptx::SM100_MMA_MXF4_SS::fma(
                                a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                        }
                        // TODO: move this into `deep_gemm/ptx/tcgen05.cuh`
                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
            }

            // UMMA warp must also arrive on empty_q to prevent running ahead
            // of math warps in the Q pipeline. Without this, UMMA can consume
            // kNumQStages Q blocks before math warps release any, causing a
            // circular dependency: UMMA waits full_q -> TMA_Q waits empty_q
            // -> Math waits full_tmem -> UMMA (already moved on).
            empty_q_barriers[q_stage_idx]->arrive();
        });
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else if (warp_idx < kSpecWarpStart) {
        // Math warpgroups for reduce
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = threadIdx.x;

        // Helper lambda for loading tensor memory
        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr uint32_t N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Math warpgroups process TMEM stages alternately
        // Advance pipeline to align with the assigned stage
        advance_tmem_pipeline(math_warpgroup_idx);

        // Local register buffers
        // [MEGAKERNEL EDIT] register diet: no register-cached weights (the reduce reads
        // float2 pairs straight from smem — the Q stage stays valid until the empty_q
        // arrive below), and accum halved to 32 (TMEM consumed in two 32-head passes
        // reusing these registers). fp32 accumulation order per (sum_0, sum_1) chain is
        // identical to the previous single-pass form -> bit-exact.
        float accum[kNumHeads / 2];

        // Enumerate assigned tasks
        for_each_task([&](const uint32_t& q_idx, const uint32_t& kv_start, const uint32_t& num_kv_blocks) {
            // Wait TMA Q arrivals
            CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Enumerate KV blocks
            for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                // Calculate KV offset in advance
                auto kv_offset = kv_start + kv_idx * BLOCK_KV + math_thread_idx;

                // Advance pipeline by `kNumMathWarpGroups` steps
                // Wait UMMA arrival
                CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
                ptx::tcgen05_after_thread_sync();

                // Reduce over the head dim and store
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    const uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                    const auto w2 = reinterpret_cast<const float2*>(
                        smem_weights[q_stage_idx] + i * kNumHeads);

                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);

                    // Two 32-head passes reusing the same accum registers
                    #pragma unroll
                    for (uint32_t half = 0; half < 2; ++ half) {
                        // Load accumulator from TMEM
                        tmem_load(cute::Int<kNumHeads / 2>{}, tmem_addr + half * (kNumHeads / 2), accum);

                        // Release TMEM empty once ALL reads of this stage are done
                        if (half == 1 and i == BLOCK_Q - 1) {
                            ptx::tcgen05_before_thread_sync();
                            empty_tmem_barriers[tmem_stage_idx]->arrive();
                        }

                        // Accumulate weighted ReLU in parallel (weights via smem float2)
                        const uint32_t jb = half * (kNumHeads / 2);
                        const auto transform = [&](const uint32_t& j, const float2& sum) {
                            auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                            auto b = ptx::ld_shared(w2 + ((jb + j) >> 1));
                            return __ffma2_rn(a, b, sum);
                        };

                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads / 2; j += 4) {
                            sum_0 = transform(j, sum_0);
                            sum_1 = transform(j + 2, sum_1);
                        }
                    }

                    auto sum = __fadd2_rn(sum_0, sum_1);
                    auto result = static_cast<logits_dtype_t>(sum.x + sum.y);

                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    // TODO: optimize performance
                    const auto q_offset = (q_idx * BLOCK_Q + i) * static_cast<uint64_t>(logits_stride);
                    if constexpr (kIsCompressedLogits) {
                        if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
                            logits[q_offset + kv_offset - seq_k_start[i]] = result;
                    } else {
                        logits[q_offset + kv_offset] = result;
                    }
                    __syncwarp();
                }
            }

            // Release last Q empty
            empty_q_barriers[q_stage_idx]->arrive();
        });

        // [MEGAKERNEL EDIT] attention-path end stamp (math warpgroup 0 done consuming)
        if (prof != nullptr and threadIdx.x == 0)
            prof[blockIdx.x * 4 + 1] = ptx::globaltimer();

        // Free tensor memory
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace deep_gemm

// ============================================================
// DSV4 fixed configuration (config.json: index_n_heads=64, index_head_dim=128).
// ============================================================
namespace mqa_logits_fp4 {
static constexpr int NUM_HEADS = 64;
static constexpr int HEAD_DIM  = 128;
// Decode-only (per-sequence seqlen == 1): 1 query token per CTA -> UMMA_N = BLOCK_Q*64 = 64.
// This is swap-AB (KV slots on UMMA_M=128 via BLOCK_KV; query*head on the flexible N).
// BLOCK_Q=1 avoids the padding of single-token decode AND the ~2x KV overscan of packing
// two different batches into one q-block. (For MTP/prefill where >=2 real queries share a
// KV range, BLOCK_Q=2 -> UMMA_N=128 would pack them for 2x MMA throughput; not our case.)
static constexpr int BLOCK_Q   = 1;                 // UMMA_N = 64
static constexpr int BLOCK_KV  = 256;
static constexpr int NUM_Q_STAGES  = 3;
// DEFAULT KV pipeline depth (faithful single-seq path only; decode auto-selects 6,
// with {4,6,8,10} instantiated for explicit override). B300 sweep verdict: 6 wins
// or ties every BxT cell — depth only needs to cover the TMA latency-bandwidth
// product (~3-4 tiles in flight); deeper costs unified-L1 carveout and HBM
// outstanding-window pressure for zero extra coverage.
static constexpr int NUM_KV_STAGES = 6;
static constexpr int NUM_TMEM_STAGES = 3;           // hardcoded in the kernel
static constexpr int NUM_SPECIALIZED_THREADS = 128;
static constexpr int NUM_MATH_THREADS        = 2 * 128;
// CUDA-core tail warpgroup (warps 12-15): hides the wq_b per-head RMSNorm under
// the KV stream; idle when no rms work is passed. NOTE: TPB=512 caps the
// architectural register budget at 65536/512 = 128 — the math register diet
// (edit #4) is the prerequisite.
static constexpr int NUM_TAIL_THREADS        = 128;
static constexpr int TPB = NUM_SPECIALIZED_THREADS + NUM_MATH_THREADS + NUM_TAIL_THREADS;  // 512
}  // namespace mqa_logits_fp4
