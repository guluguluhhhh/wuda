#pragma once
// ============================================================
// mqa_logits_fp4.cuh — vendored DeepGEMM HELPER CLOSURE (math/utils/ptx/tma/mma,
// verbatim, single-header, depends only on the repo's CUTLASS/CuTe) + the DSV4
// MAIN-indexer compressor (MainCompressorArgs / run_main_compressor_row).
//
// The score-attention kernel itself is the VERBATIM DeepGEMM sm100 paged one in
// include/dg_paged_mqa_logits.cuh (which includes this header for the closure).
// Host launcher + PyTorch binding: kernels/mqa_logits_fp4.cu
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

template <typename T>
CUTLASS_DEVICE void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

/// Reduction
CUTLASS_DEVICE uint32_t warp_inclusive_sum(uint32_t value, const uint32_t& lane_idx) {
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const uint32_t synced = __shfl_up_sync(0xffffffff, value, offset);
        if (lane_idx >= offset)
            value += synced;
    }
    return value;
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

CUTLASS_DEVICE uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
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

struct SM100_MMA_MXF8F6F4_SS {
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
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p; \n\t"
          "}\n"
          :
          : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
            "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

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

CUTLASS_DEVICE
static uint32_t get_atom_base(const cute::UMMA::LayoutType& layout_type) {
    return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

/// UMMA descriptors
// ReSharper disable once CppNotAllPathsReturnValue
template <cute::UMMA::Major kMajorMode, uint32_t kSwizzleMode, bool kUseBase32, typename dtype_t>
constexpr static cute::UMMA::LayoutType to_umma_layout_type() {
    DG_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or
                     kSwizzleMode == 32 or kSwizzleMode == 64 or
                     kSwizzleMode == 128, "Invalid swizzling mode");
    // A special case
    if constexpr ((cute::is_same_v<dtype_t, float> and kMajorMode == cute::UMMA::Major::MN) or kUseBase32) {
        DG_STATIC_ASSERT(kUseBase32, "Invalid swizzling base");
        return cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B;
    }

    // Normal cases
    if constexpr (kSwizzleMode == 0)   return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 16)  return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)  return cute::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)  return cute::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128) return cute::UMMA::LayoutType::SWIZZLE_128B;
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE
constexpr uint32_t get_umma_desc_stride_k() {
    return kMajorMode == cute::UMMA::Major::K ? 1 : tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

template <typename dtype_t>
CUTLASS_DEVICE
constexpr uint32_t get_umma_desc_pack_factor() {
    // Packed FP4 stores two logical elements per byte in SMEM.
    if constexpr (cute::is_same_v<dtype_t, cutlass::float_e2m1_t>) {
        return 2;
    } else {
        return 1;
    }
}

template <cute::UMMA::Major kMajorMode, uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode,
          bool kUseBase32 = false, typename dtype_t>
CUTLASS_DEVICE
cute::UMMA::SmemDescriptor make_umma_desc(dtype_t* base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
    // NOTES: `base_smem_ptr` must use the logical SMEM element type used by UMMA descriptors.
    constexpr uint32_t kPackFactor = get_umma_desc_pack_factor<dtype_t>();
    DG_STATIC_ASSERT(kPackFactor == 1 or sizeof(dtype_t) == 1, "Packing expects a 1-byte storage type");
    const uint32_t stride_k = get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
    const auto layout_type = to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
    const auto num_non_contiguous = 128 / get_atom_base(layout_type);
    if constexpr (kMajorMode == cute::UMMA::Major::K) {
        // NOTES: for K-major layout, the swizzle must be the same as `BLOCK_K` elements in bytes
        // also, atom index must be 0, so that each block has exactly one swizzle atom on the K axis
        DG_STATIC_ASSERT(kSwizzleMode * kPackFactor == BLOCK_K * sizeof(dtype_t), "Unexpected value");

        // Atom size: 8 x `kSwizzleMode` (in bytes, on K)
        // {SBO, LBO} means the byte stride between atoms on {MN, K}
        // NOTES: on K, there is only 1 atom as asserted previously, so LBO can be 0
        const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype_t) / kPackFactor;
        const uint32_t leading_byte_offset = 0;
        const auto byte_ptr = reinterpret_cast<uint8_t*>(base_smem_ptr) +
                              (mn_idx * BLOCK_K + k_idx * stride_k) * sizeof(dtype_t) / kPackFactor;
        return make_smem_desc(layout_type, byte_ptr, stride_byte_offset, leading_byte_offset);
    } else {
        DG_STATIC_ASSERT(kPackFactor <= 1, "Packing only supports K-major");
        constexpr uint32_t BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();

        // Must have no in-atom MN-idx
        // NOTES: no worries for the runtime assert, the `mn_idx` are constants at compilation time
        DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
        DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

        // Atom size: `kSwizzleMode` (in bytes, on MN) x 8
        // NOTES: `kSwizzleMode == 16` mean non-swizzling but interleaving
        // {SBO, LBO} means the byte stride between atoms on {K, MN} for swizzling
        // {SBO, LBO} means the byte stride between atoms on {MN, K} for non-swizzling
        uint32_t stride_byte_offset = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
        uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
        if constexpr (kSwizzleMode == 16)
            math::swap(stride_byte_offset, leading_byte_offset);
        return make_smem_desc(layout_type,
                              base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                              stride_byte_offset, leading_byte_offset);
    }
}

CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc, const uint32_t& sfa_id, const uint32_t& sfb_id) {
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

} // namespace deep_gemm::mma::sm100

// ============================================================
// DSV4 MAIN-indexer compressor row chain, SHARED by the standalone kernel
// (kernels/mqa_logits_fp4.cu) and the fused tail warpgroup of the DeepGEMM
// paged attention kernel (include/dg_paged_mqa_logits.cuh, TPB=512).
// ============================================================

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>


namespace deep_gemm {

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
    float* s8;                 // [M, 7]   per-block-64 scales
    nv_bfloat16* rope;         // [M, 64]  bf16 rope-tail output
};

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
// SHARED barrier discipline: caller passes the NamedBarrier id (the standalone
// kernel launches one 128-thread block per row and uses id 0; the fused tail
// warpgroup uses id 2 -- ids 0/1 belong to the attention path there).
// [B1] The state shift is GONE: rows are addressed through the pos-derived
// ping-pong mapping (see MainCompressorArgs); the state is read-only.
__device__ __forceinline__ void run_main_compressor_row(
    const MainCompressorArgs& comp, uint32_t m, long long p,
    uint32_t g, uint32_t lane_idx, float rms_eps,
    uint32_t barrier_id) {
    constexpr uint32_t D_M = 512, WK_M = 1024, RD = 64, RATIO = 4;
    __shared__ float ssq_sm[4];    // static smem, coexists with the extern dynamic smem
    const float* bkv = comp.kv + (uint64_t)m * 8 * WK_M;
    const float* bsc = comp.sc + (uint64_t)m * 8 * WK_M;
    // Ping-pong base: window index k = ⌊p/4⌋ (p is a compress step, p%4==3);
    // logical row rr lives at physical row (base + rr) & 7.
    const uint32_t base = 4u * ((uint32_t)(p >> 2) & 1u);

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
        if (col0 < D_M - RD) {                     // quant half-warps
            const float scale = fmaxf(mx, 1e-4f) * (1.0f / 448.0f);
            uint32_t packed = 0;
            #pragma unroll
            for (uint32_t e = 0; e < 4; ++ e) {
                const __nv_fp8_e4m3 f8 = __nv_fp8_e4m3(cmp[e] / scale);
                packed |= (uint32_t)f8.__x << (8 * e);
            }
            *reinterpret_cast<uint32_t*>(
                comp.q8 + (uint64_t)m * (D_M - RD) + col0) = packed;
            if ((lane_idx & 15) == 0)
                comp.s8[(uint64_t)m * 7 + (col0 >> 6)] = scale;
        } else {                                   // rope tail: pack 2 bf16 -> u32
            #pragma unroll
            for (uint32_t e = 0; e < 4; e += 2) {
                const auto b2 = __floats2bfloat162_rn(cmp[e], cmp[e + 1]);
                *reinterpret_cast<uint32_t*>(
                    comp.rope + (uint64_t)m * RD + (col0 - (D_M - RD)) + e) =
                    *reinterpret_cast<const uint32_t*>(&b2);
            }
        }
    }
    // Row-end sync: publishes the row AND protects ssq_sm across back-to-back rows.
    cutlass::arch::NamedBarrier(128, barrier_id).sync();
}

} // namespace deep_gemm

