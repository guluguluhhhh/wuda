#pragma once
// ============================================================
// mqa_logits_fp4.cuh — DSV4 score-attention kernel (FP4 MQA-logits), PAGED
// decode. Self-contained: DeepGEMM's sm100_fp4_mqa_logits + minimal helper
// closure inlined, depends only on the repo's CUTLASS/CuTe.
//
// Math: logits[t] = Σ_h relu(<iq[h,:],kvc[t,:]>)·weights[h] (fp4 UMMA + reduce)
// Host launcher / binding / usage: kernels/mqa_logits_fp4.cu.
//
// Deltas vs upstream, marked `// [MEGAKERNEL EDIT]` in the code: gridDim-based
// SM count; PDL neutralized; IN-KERNEL tile-pool scheduler (global Σ KV tiles
// balanced across CTAs -- replaces the metadata kernel); math register diet
// (224 -> <=128, bit-exact); TPB 512 with a CUDA-core tail warpgroup (fused
// MAIN compressor, idle when off); PAGED KV via block_table (fused pages:
// [PAGE_KV*64B fp4 | PAGE_KV*4B sf], per-token windows = context_lens).
// Unused remainder of the vendored closure is parked under include/mqa_unused/.
// ============================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>   // [MEGAKERNEL EDIT] fused MAIN compressor: real e4m3 quant
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
//   3. tile-pool decode schedule: all 4 warp roles enumerate tasks through a
//      shared `for_each_task` closure (the legacy per-q-block contiguous
//      enumeration has been removed together with the contiguous-KV path).
// Everything else is byte-identical to the proven DeepGEMM kernel.
// ============================================================

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>


namespace deep_gemm {

// ============================================================
// [MEGAKERNEL EDIT] Decode tile-pool scheduler (PAGED; the only schedule).
//
// PAGED decode geometry (RTP-LLM model): seq_len = B tokens (BLOCK_Q == 1), token b
// owns the KV window [0, context_lens[b]) addressed through its block_table row
// (fused pages, PAGE_KV tokens each). The GLOBAL pool of Σ_b cdiv(ctx_b, BLOCK_KV)
// tiles is split into gridDim.x balanced contiguous chunks (may cross token
// boundaries) — the inline equivalent of DeepGEMM's paged-path metadata kernel
// (which emits per-SM (q_token_idx, kv_split_idx) starts), but DEVICE-ONLY and
// launch-free (megakernel-ready: no host/Python in the loop, no schedule_meta,
// no per-step metadata launch).
//
// Cost model: ONE warp per CTA builds the tile prefix sum into smem with a
// warp-parallel scan (latency-flattened, see the kernel prologue); each thread
// then locates its chunk with a log2(B) binary search on smem.
// ============================================================
static constexpr uint32_t kNumMaxTilePoolTokens = 512;   // decode B cap (serving: 32-256)

template <uint32_t BLOCK_KV>
struct TilePoolScheduler {
    const uint32_t* lens;            // [num_tokens] per-token context length (paged)
    const uint32_t* tile_prefix;     // smem, [num_tokens + 1], built by the warp scan
    uint32_t num_tokens, max_ctx;    // max_ctx = capacity clamp (max_pages * PAGE_KV)
    uint32_t chunk_cur, chunk_end;   // this CTA's global tile ids: [chunk_cur, chunk_end)
    uint32_t token;                  // current token (tile_prefix[token] <= chunk_cur)

    CUTLASS_DEVICE TilePoolScheduler(const uint32_t& cta_idx, const uint32_t& num_ctas,
                                     const uint32_t& num_tokens, const uint32_t& max_ctx,
                                     const uint32_t* lens, const uint32_t* tile_prefix):
            lens(lens), tile_prefix(tile_prefix), num_tokens(num_tokens), max_ctx(max_ctx) {
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
    // [PAGED] kv_start is a TOKEN-LOCAL slot offset (tile index * BLOCK_KV); the
    // physical address comes from block_table in the KV TMA role. seq_k window is
    // [0, context_len) -- ks is identically 0 in the paged decode model.
    CUTLASS_DEVICE bool next(uint32_t& q_idx, uint32_t& kv_start, uint32_t& num_kv_blocks,
                             uint32_t& seq_k_start, uint32_t& seq_k_end) {
        while (chunk_cur < chunk_end and token < num_tokens) {
            const uint32_t tile_base = tile_prefix[token];
            const uint32_t tile_end  = cute::min(tile_prefix[token + 1], chunk_end);
            if (tile_end > chunk_cur) {
                q_idx = token;
                seq_k_start = 0;
                seq_k_end = cute::min(lens[token], max_ctx);
                kv_start = (chunk_cur - tile_base) * BLOCK_KV;
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

// [MEGAKERNEL EDIT] MAIN-compressor argument bundle (gemm_fuse_norm_b
// compressor_process_row, d=512 MAIN part ONLY; the indexer(d=128) part stays
// upstream). The [m, 8, 1024] state slots are assumed already written by the
// producer op; this kernel's tail only post-processes COMPRESS rows
// ((pos+1)%4 == 0): overlap-cat softmax aggregate -> weighted
// bf16 RMSNorm -> interleaved RoPE(last 64) -> fp8 e4m3 block-64 quant.
// kv == nullptr disables the whole section.
struct MainCompressorArgs {
    const long long* pos;      // [M] absolute token positions
    const float* norm;         // [512] RMSNorm weight (comp_norm)
    const float* cos_tab;      // [S, 32] RoPE tables, row = compressed position
    const float* sin_tab;      // [S, 32]
    // [B1] PING-PONG state window (READ-ONLY here; no physical shift): the 8 rows
    // are a pos-derived circular window, physical row = (4*(⌊pos/4⌋&1) + rr) & 7
    // for logical row rr. The producer writes the fresh token at logical 4+pos%4
    // under the SAME mapping, so "current" rows become next window's "previous"
    // rows by phase flip alone -- the old rows[0:4]=rows[4:8] copy (64KB/row, 4
    // latency waves) is gone. Anyone touching kv/sc must use this mapping.
    const float* kv;           // [M, 8, 1024] state (aggregate reads only)
    const float* sc;           // [M, 8, 1024] state scores
    uint8_t* q8;               // [M, 448] e4m3 output
    float* s8;                 // [M, 7]   per-block-64 scales (pow2, e8m0-exact)
    nv_bfloat16* rope;         // [M, 64]  bf16 rope-tail output
    // ---- kvcache design (kccache_design.png): persistent-slot state + DIRECT
    // Main-compressed MODEL1 page writes. slot_map[m] turns kv/sc into POOLS
    // [capacity,8,1024]; cmp_dst[m] = compressed-token page*64+off (-1 skip).
    const int* slot_map = nullptr;   // [M]
    uint8_t* cmp_cache  = nullptr;   // MODEL1 pages [P, 37440]B
    const int* cmp_dst  = nullptr;   // [M]
};

// MODEL1_FP8Sparse page geometry (FlashMLA quant.py, d=512): body = 64 x
// (448B e4m3 nope + 128B bf16 rope), tail = 64 x 8B (7 e8m0 scales + pad),
// page padded to a 576B multiple.
constexpr int M1_TOK_BODY   = 576;
constexpr int M1_TAIL_OFF   = 64 * M1_TOK_BODY;                    // 36864
constexpr int M1_PAGE_BYTES = (64 * (M1_TOK_BODY + 8) + 575) / 576 * 576;

// pow2-ceil scale helpers (bit-exact, delivery tile/quant.cuh semantics).
__device__ __forceinline__ int m1_flog2_ceil(float x) {
    const unsigned bits = __float_as_uint(x);
    const int exp_field = (int)((bits >> 23) & 0xFF);
    return exp_field - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
}
__device__ __forceinline__ float m1_fpow2(int e) {
    return __uint_as_float((unsigned)(e + 127) << 23);
}

// [C1] Cooperative MAIN-compressor row chain (d=512): FOUR warps split one row by
// 128-col group (g = warp index 0-3; lane owns cols g*128 + lane*4 .. +3), vs the
// old one-warp-per-row form: the aggregate's 4 SERIAL 16-float4 load waves become
// ONE, per-lane expf drops 128 -> 32, and norm/rope/quant spread 4-ways -- the
// compressor is the ONLY tail work now (rmsnorm deleted), so this per-row chain
// latency IS the tail wall. Cross-warp state: a 4-float static-smem slot for the
// RMSNorm sum of squares + two NamedBarrier(128, barrier_id) syncs per row (the
// trailing one also protects the smem slot across back-to-back rows). Per-column
// math is IDENTICAL to the old form; only the RMSNorm reduction ORDER changes
// (group partials summed g=0..3) -- tolerance-level, like any reduce-order change.
// SHARED by the fused tail (barrier_id=2; ids 0/1 belong to the attention path)
// and the standalone reference kernel (one 128-thread block per row, barrier_id=0).
// [B1] The state shift is GONE: rows are addressed through the pos-derived
// ping-pong mapping (see MainCompressorArgs); the state is read-only.
__device__ __forceinline__ void run_main_compressor_row(
    const MainCompressorArgs& comp, uint32_t m, long long p,
    uint32_t g, uint32_t lane_idx, float rms_eps,
    uint32_t barrier_id, unsigned long long* prof) {
    constexpr uint32_t D_M = 512, WK_M = 1024, RD = 64, RATIO = 4;
    __shared__ float ssq_sm[4];    // static smem, coexists with the extern dynamic smem
    // kvcache design: state pool addressed by the persistent request slot.
    const uint32_t srow = comp.slot_map ? (uint32_t)comp.slot_map[m] : m;
    const float* bkv = comp.kv + (uint64_t)srow * 8 * WK_M;
    const float* bsc = comp.sc + (uint64_t)srow * 8 * WK_M;
    // Ping-pong base: window index k = ⌊p/4⌋ (p is a compress step, p%4==3);
    // logical row rr lives at physical row (base + rr) & 7.
    const uint32_t base = 4u * ((uint32_t)(p >> 2) & 1u);
    // Phase stamps (group-0 warp's lane 0; last compress row wins per CTA)
    const bool cprof = prof != nullptr and g == 0 and lane_idx == 0;

    // Aggregate: per-column softmax over the 8 overlap-cat rows, THIS group's 128
    // contiguous cols only (rows<4 read col c, rows>=4 read col 512+c). All 16
    // float4 loads are in flight at once; each is a fully-coalesced 512B warp burst.
    // cmp[e] <-> column g*128 + lane*4 + e.
    float cmp[4];
    {
        const float4* bsc4 = reinterpret_cast<const float4*>(bsc);
        const float4* bkv4 = reinterpret_cast<const float4*>(bkv);
        const uint32_t c4 = g * 32 + lane_idx;      // float4 col; warp = 512B contiguous
        float4 s4[8], k4[8];
        #pragma unroll
        for (uint32_t rr = 0; rr < 8; ++ rr) {
            const uint32_t pr = (base + rr) & 7;        // [B1] ping-pong physical row
            const uint32_t col4 = (rr < RATIO) ? c4 : (D_M / 4 + c4);
            s4[rr] = bsc4[pr * (WK_M / 4) + col4];
            k4[rr] = bkv4[pr * (WK_M / 4) + col4];
        }
        #pragma unroll
        for (uint32_t e = 0; e < 4; ++ e) {
            float mx = (&s4[0].x)[e];
            #pragma unroll
            for (uint32_t rr = 1; rr < 8; ++ rr)
                mx = fmaxf(mx, (&s4[rr].x)[e]);
            float sm = 0.f, acc = 0.f;
            #pragma unroll
            for (uint32_t rr = 0; rr < 8; ++ rr) {
                const float ex = expf((&s4[rr].x)[e] - mx);
                sm += ex;
                acc += ex * (&k4[rr].x)[e];
            }
            cmp[e] = acc / sm;
        }
    }
    if (cprof)
        prof[blockIdx.x * 8 + 5] = ptx::globaltimer();   // aggregate done (group 0)

    // (bf16) weighted RMSNorm over d=512: warp-local group partial (shuffle tree)
    // -> 4-float smem exchange -> every warp sums the 4 partials in fixed order.
    // NOTE: __float2bfloat16/__bfloat162float instead of C-style casts —
    // torch extensions build with -D__CUDA_NO_BFLOAT16_CONVERSIONS__.
    float part = 0.f;
    #pragma unroll
    for (uint32_t e = 0; e < 4; ++ e) {
        const float vb = __bfloat162float(__float2bfloat16(cmp[e]));
        part += vb * vb;
    }
    #pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1)
        part += __shfl_xor_sync(0xffffffffu, part, o);
    if (lane_idx == 0)
        ssq_sm[g] = part;
    cutlass::arch::NamedBarrier(128, barrier_id).sync();
    const float total = ssq_sm[0] + ssq_sm[1] + ssq_sm[2] + ssq_sm[3];
    const float rms = rsqrtf(total / float(D_M) + rms_eps);
    const float4 ng = reinterpret_cast<const float4*>(comp.norm)[g * 32 + lane_idx];
    #pragma unroll
    for (uint32_t e = 0; e < 4; ++ e) {
        const float vb = __bfloat162float(__float2bfloat16(cmp[e]));
        cmp[e] = __bfloat162float(__float2bfloat16(vb * rms * (&ng.x)[e]));
    }

    // Interleaved RoPE on the last 64 dims: cols 448..511 = group 3's UPPER
    // half-warp (lanes 16..31, 4 consecutive cols each -> both pair elements
    // (2j, 2j+1) live in the same lane).
    const long long ri = p + 1 - RATIO;        // compressed-token position
    if (g == 3 and lane_idx >= 16) {
        const float* crow = comp.cos_tab + (uint64_t)ri * (RD / 2);
        const float* srow = comp.sin_tab + (uint64_t)ri * (RD / 2);
        #pragma unroll
        for (uint32_t e = 0; e < 4; e += 2) {
            const uint32_t j = (lane_idx - 16) * 2 + e / 2;
            const float ev = cmp[e], ov = cmp[e + 1];
            const float cc = crow[j], ss = srow[j];
            cmp[e]     = __bfloat162float(__float2bfloat16(ev * cc - ov * ss));
            cmp[e + 1] = __bfloat162float(__float2bfloat16(ev * ss + ov * cc));
        }
    }

    // fp8 e4m3 quant of this group's two 64-col blocks (half-warp width-16 shuffle
    // amax, same as before) + bf16 store of the roped tail (group 3, cols >= 448).
    // q8 stores are one u32 per lane at consecutive addresses -> coalesced bursts.
    {
        float mx = 0.f;
        #pragma unroll
        for (uint32_t e = 0; e < 4; ++ e)
            mx = fmaxf(mx, fabsf(cmp[e]));
        #pragma unroll
        for (uint32_t o = 1; o < 16; o <<= 1)      // all lanes participate
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o, 16));
        const uint32_t col0 = g * 128 + lane_idx * 4;
        // kvcache design: optional DIRECT write into the Main-compressed
        // MODEL1 page (same bytes/scales as the compact outputs).
        uint8_t* m1 = nullptr;
        if (comp.cmp_cache != nullptr && comp.cmp_dst != nullptr
            && comp.cmp_dst[m] >= 0)
            m1 = comp.cmp_cache
                + (uint64_t)(comp.cmp_dst[m] / 64) * M1_PAGE_BYTES;
        const int m1_off = (comp.cmp_dst != nullptr && comp.cmp_dst[m] >= 0)
            ? (comp.cmp_dst[m] % 64) : 0;
        if (col0 < D_M - RD) {                     // quant half-warps
            // MODEL1 scale: pow2-ceil(clamp(amax/448, 1e-4)) -- e8m0-exact
            // (was fp32 amax/448 pre-kvcache; FlashMLA MODEL1 contract).
            const int se = m1_flog2_ceil(fmaxf(mx * (1.0f / 448.0f), 1e-4f));
            const float scale = m1_fpow2(se);
            uint32_t packed = 0;
            #pragma unroll
            for (uint32_t e = 0; e < 4; ++ e) {
                const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(cmp[e] / scale);
                packed |= (uint32_t)f8.__x << (8 * e);
            }
            // COMPACT outputs optional in cache mode (double-write saver).
            if (comp.q8 != nullptr)
                *reinterpret_cast<uint32_t*>(
                    comp.q8 + (uint64_t)m * (D_M - RD) + col0) = packed;
            if ((lane_idx & 15) == 0 && comp.s8 != nullptr)
                comp.s8[(uint64_t)m * 7 + (col0 >> 6)] = scale;
            if (m1 != nullptr) {
                *reinterpret_cast<uint32_t*>(
                    m1 + m1_off * M1_TOK_BODY + col0) = packed;
                if ((lane_idx & 15) == 0)
                    m1[M1_TAIL_OFF + m1_off * 8 + (col0 >> 6)] =
                        (uint8_t)(se + 127);
            }
        } else {                                   // rope tail: pack 2 bf16 -> u32
            #pragma unroll
            for (uint32_t e = 0; e < 4; e += 2) {
                const auto b2 = __floats2bfloat162_rn(cmp[e], cmp[e + 1]);
                if (comp.rope != nullptr)
                    *reinterpret_cast<uint32_t*>(
                        comp.rope + (uint64_t)m * RD + (col0 - (D_M - RD)) + e) =
                        *reinterpret_cast<const uint32_t*>(&b2);
                if (m1 != nullptr)
                    *reinterpret_cast<uint32_t*>(
                        m1 + m1_off * M1_TOK_BODY + (D_M - RD)
                        + ((col0 - (D_M - RD)) + e) * 2) =
                        *reinterpret_cast<const uint32_t*>(&b2);
            }
        }
    }
    // Row-end sync: makes the slot-7 stamp a TRUE row end and protects ssq_sm
    // against the next row's partial publish (back-to-back rows on one CTA).
    cutlass::arch::NamedBarrier(128, barrier_id).sync();
    if (cprof)
        prof[blockIdx.x * 8 + 7] = ptx::globaltimer();   // row end (norm+rope+quant)
}

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV, uint32_t PAGE_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          // [MEGAKERNEL EDIT] `kNumSMs` template param removed; see gridDim.x below.
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          // [MEGAKERNEL EDIT] CUDA-core tail warpgroup (0 disables the branch entirely)
          uint32_t kNumTailThreads,
          typename logits_dtype_t,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads + kNumTailThreads, 1)
void sm100_fp4_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv,
                          const uint32_t logits_stride,
                          // [MEGAKERNEL EDIT / PAGED] per-token context lengths and the
                          // page table ([seq_len, block_table_stride] physical page ids;
                          // fused page layout, see header note 6). `seq_len_kv` carries
                          // the CAPACITY clamp (max_pages * PAGE_KV).
                          const uint32_t* context_lens,
                          const uint32_t* block_table,
                          const uint32_t block_table_stride,
                          logits_dtype_t* logits,
                          // [MEGAKERNEL EDIT] epsilon for the MAIN compressor's internal
                          // RMSNorm step (tail warps; unused when comp.kv == nullptr).
                          const float comp_eps,
                          // [MEGAKERNEL EDIT] globaltimer stamps [gridDim.x][8]
                          // (test_complex.cu phase-breakdown pattern):
                          // 0=attention-path start (post-prologue), 1=attention end,
                          // 2=tail start (at CTA start), 3=tail end,
                          // 4=retired (was the rms section end),
                          // and per compress row (last one per CTA):
                          // 5=aggregate done, 6=retired (was shift; B1),
                          // 7=row end (norm+rope+quant).
                          // nullptr = off.
                          unsigned long long* prof,
                          // [MEGAKERNEL EDIT] MAIN compressor bundle (tail warps; see
                          // MainCompressorArgs above). comp.kv == nullptr -> disabled.
                          const MainCompressorArgs comp,
                          // [MEGAKERNEL EDIT] benchmark tail_us: true -> the 384
                          // attention threads exit before the prologue, leaving the
                          // tail warpgroup running ALONE in its in-situ launch shape.
                          const bool attn_mock,
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
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

    // [MEGAKERNEL EDIT] CUDA-core tail warpgroup (warps 12-15), hoisted ABOVE the
    // whole prologue — gemm_fuse_norm_b discipline: the attention path and the tail
    // NEVER share a block-wide barrier; each side handles its own sync. The tail
    // touches no smem / mbarrier / TMEM state, so it starts the instant the CTA
    // does: it must NOT wait for barrier init, TMEM alloc, or the tile-prefix scan
    // warp. The remaining 384 threads publish the prologue via a role-scoped
    // NamedBarrier below.
    // Work: MAIN-indexer compressor rows ONLY (one warp per compress row; the
    // aggregate -> norm -> rope -> quant chain lives in run_main_compressor_row
    // above, shared with the standalone reference kernel). comp.kv == nullptr ->
    // the tail exits immediately.
    if (kNumTailThreads > 0 and warp_idx >= kSpecWarpStart + 4) {
        const bool prof_lane = prof != nullptr and warp_idx == kSpecWarpStart + 4 and lane_idx == 0;
        if (prof_lane)
            prof[blockIdx.x * 8 + 2] = ptx::globaltimer();
        // ---- MAIN compressor rows: [C1] all FOUR tail warps cooperate on ONE row
        // (one 128-col group per warp), rows stride by CTA -- the per-row chain,
        // which IS the tail wall, is ~4x shorter than one-warp-per-row. All warps
        // iterate the same m sequence and take the same compress-row branch, so
        // the in-row NamedBarrier(128, 2) is always fully attended.
        if (comp.kv != nullptr) {
            DG_STATIC_ASSERT(kNumTailThreads == 128, "coop compressor assumes 4 tail warps");
            const uint32_t tail_warp = warp_idx - (kSpecWarpStart + 4);
            for (uint32_t m = blockIdx.x; m < seq_len; m += gridDim.x) {
                const long long p = comp.pos[m];
                if (((p + 1) & 3) != 0)
                    continue;                              // not a compress row
                run_main_compressor_row(comp, m, p, tail_warp, lane_idx, comp_eps,
                                        /*barrier_id=*/2, prof);
            }
        }
        // Tail end = the LATEST tail warp (atomicMax): a single-warp stamp would
        // under-report compress CTAs by the whole compressor duration.
        if (prof != nullptr and lane_idx == 0)
            atomicMax(prof + blockIdx.x * 8 + 3, ptx::globaltimer());
        return;
    }
    // [MEGAKERNEL EDIT] benchmark tail_us: attention mocked out -- the remaining
    // 384 threads leave BEFORE any prologue state (mbarrier init / TMEM alloc /
    // scan), so the kernel degenerates to the tail warpgroup alone, in situ.
    if (attn_mock)
        return;

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
    DG_STATIC_ASSERT(BLOCK_Q == 1, "Paged decode schedule assumes 1 token per q-block");
    DG_STATIC_ASSERT(PAGE_KV > 0 and BLOCK_KV % PAGE_KV == 0, "KV tile must be whole pages");

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
    // prefix sum in smem via a warp-parallel scan; the NamedBarrier below publishes
    // it to all roles. Device-only — the fused equivalent of DeepGEMM's separate
    // metadata kernel launch (no host/Python in the loop; megakernel-compatible).
    if (warp_idx == kSpecWarpStart + 3) {
        DG_TRAP_ONLY_DEVICE_ASSERT(seq_len <= kNumMaxTilePoolTokens);
        // [MEGAKERNEL EDIT] latency-flattened scan: phase 1 issues ALL context_lens
        // loads back-to-back into registers (fully unrolled + predicated -> every
        // load in flight, ONE cold-L2 latency wave); phase 2 scans from registers
        // (the `running` carry chain is shfl-only). The earlier fused load+scan
        // form paid one full HBM latency per 32 tokens -- the measured B-linear
        // deficit vs DeepGEMM's untimed-metadata paged path.
        constexpr uint32_t kMaxScanRounds = kNumMaxTilePoolTokens / 32;   // 16
        uint32_t lv[kMaxScanRounds];
        #pragma unroll
        for (uint32_t r = 0; r < kMaxScanRounds; ++ r) {
            const uint32_t b = r * 32 + lane_idx;
            lv[r] = 0;
            if (b < seq_len)
                lv[r] = __ldg(context_lens + b);
        }
        uint32_t running = 0;
        if (lane_idx == 0)
            smem_tile_prefix[0] = 0;
        #pragma unroll
        for (uint32_t r = 0; r < kMaxScanRounds; ++ r) {
            if (r * 32 >= seq_len)
                break;
            const uint32_t b = r * 32 + lane_idx;
            uint32_t num_tiles = 0;
            if (b < seq_len) {
                // [PAGED] tiles_b = cdiv(ctx_b, BLOCK_KV); ks == 0 and page starts
                // are PAGE_KV-aligned, so the old /4 SF alignment fixup is gone.
                const uint32_t e = cute::min(lv[r], seq_len_kv);
                num_tiles = math::ceil_div(e, BLOCK_KV);
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
    // [MEGAKERNEL EDIT] role-scoped prologue publish (barrier init / TMEM ptr /
    // tile prefix) for the 384 attention-path threads ONLY — the tail warpgroup
    // exited above and must not be made to wait here (gemm_fuse_norm_b: the two
    // sides never share a block-wide barrier, so __syncthreads is forbidden).
    // NamedBarrier id 1; id 0 is the math-only barrier at the epilogue.
    cutlass::arch::NamedBarrier(kNumSpecializedThreads + kNumMathThreads, 1).sync();

    // [MEGAKERNEL EDIT] attention-path start stamp (post-prologue; the tail's t2 is
    // stamped at CTA start above, so t2 <= t0 is expected on the timeline)
    if (prof != nullptr and threadIdx.x == 0)
        prof[blockIdx.x * 8 + 0] = ptx::globaltimer();

    // Scheduler state: per-task KV window (paged model: [0, ctx); BLOCK_Q == 1)
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    // [MEGAKERNEL EDIT] unified task enumeration for all 4 warp roles: gridDim.x CTAs
    // split the GLOBAL KV tile pool into balanced contiguous chunks (see
    // TilePoolScheduler above). All roles construct their own scheduler and see the
    // SAME task sequence, so the Q/KV/TMEM pipelines stay in lock-step.
    auto for_each_task = [&](auto&& fn) {
        TilePoolScheduler<BLOCK_KV> sched(sm_idx, kNumSMs, seq_len, seq_len_kv,
                                          context_lens, smem_tile_prefix);
        uint32_t q_idx, kv_start, num_kv_blocks;
        while (sched.next(q_idx, kv_start, num_kv_blocks, seq_k_start[0], seq_k_end[0]))
            fn(q_idx, kv_start, num_kv_blocks);
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

        // [MEGAKERNEL EDIT / PAGED] whole-WARP task walk (DeepGEMM paged pattern):
        // the warp cooperatively caches 32 consecutive block_table entries in ONE
        // coalesced L2 read; per tile the BLOCK_KV/PAGE_KV page ids are register
        // SHUFFLES instead of serial __ldg on the elect_one lane (that serial chain
        // sits on the KV issue critical path). Every lane runs its own scheduler /
        // pipeline copy (deterministic, same sequence); only the barrier wait + TMA
        // issue stay elect_one-scoped.
        {
            constexpr uint32_t kPagesPerBlock = BLOCK_KV / PAGE_KV;
            DG_STATIC_ASSERT(kPagesPerBlock <= 32, "tile spans more pages than a warp can cache");
            uint32_t cached_page_base = 0, cached_page_coord = 0;
            // Enumerate assigned (token, KV tile sub-range) tasks
            for_each_task([&](const uint32_t& q_idx, const uint32_t& kv_start, const uint32_t& num_kv_blocks) {
                const uint32_t* bt_row = block_table + q_idx * static_cast<uint64_t>(block_table_stride);
                const uint32_t num_kv_pages = math::ceil_div(seq_k_end[0], PAGE_KV);
                // New token row -> the cached window is stale
                cached_page_base = cute::numeric_limits<uint32_t>::max();
                // Enumerate KV blocks
                for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++ kv_idx) {
                    const uint32_t page_base = kv_start / PAGE_KV + kv_idx * kPagesPerBlock;
                    // Refill the 32-entry window when the tile leaves it (whole warp,
                    // one coalesced read; OOB offsets clamp to page 0 -- the math
                    // store guard masks those slots)
                    if (page_base < cached_page_base
                        or page_base + kPagesPerBlock > cached_page_base + 32) {
                        cached_page_base = (page_base / 32) * 32;
                        const uint32_t po = cached_page_base + lane_idx;
                        cached_page_coord = po < num_kv_pages ? __ldg(bt_row + po) : 0u;
                    }
                    uint32_t pages[kPagesPerBlock];
                    #pragma unroll
                    for (uint32_t i = 0; i < kPagesPerBlock; ++ i)
                        pages[i] = __shfl_sync(0xffffffffu, cached_page_coord,
                                               static_cast<int>(page_base - cached_page_base + i));

                    // Wait KV consumer release
                    CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
                    if (cute::elect_one_sync()) {
                        empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
                        // [PAGED] one TMA per PAGE_KV page. Page smem offsets are
                        // whole SWIZZLE_64B atoms (PAGE_KV*64B), so the assembled
                        // stage is byte-identical in layout to one contiguous load.
                        #pragma unroll
                        for (uint32_t i = 0; i < kPagesPerBlock; ++ i) {
                            cute::SM90_TMA_LOAD_3D::copy(&tensor_map_kv,
                                reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                                static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                smem_kv[kv_stage_idx] + i * PAGE_KV * (kHeadDim / 2),
                                0, 0, pages[i]);
                            cute::SM90_TMA_LOAD_2D::copy(&tensor_map_sf_kv,
                                reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                                static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                smem_sf_kv[kv_stage_idx] + i * PAGE_KV,
                                0, pages[i]);
                        }
                        full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                    }
                    __syncwarp();
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
                    // Compressed self-clean store (paged model: window is [0, ctx))
                    if (kv_offset < seq_k_end[i])
                        logits[q_offset + kv_offset] = result;
                    __syncwarp();
                }
            }

            // Release last Q empty
            empty_q_barriers[q_stage_idx]->arrive();
        });

        // [MEGAKERNEL EDIT] attention-path end stamp (math warpgroup 0 done consuming)
        if (prof != nullptr and threadIdx.x == 0)
            prof[blockIdx.x * 8 + 1] = ptx::globaltimer();

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
// [PAGED] fused-page size in tokens (RTP-LLM tokens_per_block; page layout per page:
// [PAGE_KV * 64B fp4 | PAGE_KV * 4B packed-ue8m0 sf] -> 68B/token, stride 4352B).
// Must divide BLOCK_KV; page smem offsets stay whole SWIZZLE_64B atoms.
static constexpr int PAGE_KV   = 64;
static constexpr int NUM_Q_STAGES  = 3;
// DEFAULT KV pipeline depth. PAGED B300 sweep verdict: 8 wins at long chunks --
// each 256-slot tile is now EIGHT page-granular TMAs (4 data + 4 SF) instead of
// two, so the issue-latency x bandwidth product needs a deeper in-flight window
// than the contiguous producer's 6. {4,6,8,10} stay instantiated for override.
static constexpr int NUM_KV_STAGES = 8;
static constexpr int NUM_TMEM_STAGES = 3;           // hardcoded in the kernel
static constexpr int NUM_SPECIALIZED_THREADS = 128;
static constexpr int NUM_MATH_THREADS        = 2 * 128;
// CUDA-core tail warpgroup (warps 12-15): hides the MAIN-indexer compressor rows
// under the KV stream; idle when comp.kv == nullptr. NOTE: TPB=512 caps the
// architectural register budget at 65536/512 = 128 — the math register diet
// (edit #4) is the prerequisite.
static constexpr int NUM_TAIL_THREADS        = 128;
static constexpr int TPB = NUM_SPECIALIZED_THREADS + NUM_MATH_THREADS + NUM_TAIL_THREADS;  // 512
}  // namespace mqa_logits_fp4
