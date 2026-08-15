#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace wuda::tp2 {

// Compact TP2 specialization of DeepGEMM's SymBuffer address mapping.
struct SymmetricView {
    uintptr_t local_base = 0;
    uintptr_t remote_base = 0;
    uint32_t rank = 0;

    template <typename T = void>
    __host__ __device__ T* local() const {
        return reinterpret_cast<T*>(local_base);
    }

    template <typename T = void>
    __host__ __device__ T* peer_base() const {
        return reinterpret_cast<T*>(remote_base);
    }

    __host__ __device__ uint32_t peer_rank() const {
        return rank ^ 1u;
    }

    __host__ __device__ explicit operator bool() const {
        return local_base != 0;
    }
};

inline SymmetricView make_symmetric_view(
        const std::vector<int64_t>& pointers, uint32_t rank) {
    return {
        static_cast<uintptr_t>(pointers[rank]),
        static_cast<uintptr_t>(pointers[rank ^ 1u]),
        rank,
    };
}

__device__ __forceinline__ uint32_t load_relaxed_sys(
        const uint32_t* ptr) {
    uint32_t value;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ uint32_t load_acquire_sys(
        const uint32_t* ptr) {
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ void store_relaxed_sys(
        uint32_t* ptr, uint32_t value) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::
                 "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ void store_release_sys(
        uint32_t* ptr, uint32_t value) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::
                 "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ void store_relaxed_sys_v4_u32(
        void* ptr, const uint4& value) {
    asm volatile(
        "st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::
        "l"(ptr), "r"(value.x), "r"(value.y),
        "r"(value.z), "r"(value.w) : "memory");
}

__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void fence_acquire_sys() {
    asm volatile("fence.acquire.sys;" ::: "memory");
}

__device__ __forceinline__ void spin_pause() {
    asm volatile("nanosleep.u32 128;");
}

}  // namespace wuda::tp2
