#pragma once

#include <cuda_bf16.h>

namespace wuda_query_rms_rope {

constexpr int kHidden = 512;
constexpr int kRope = 64;
constexpr int kWarp = 32;
constexpr int kElemsPerVec = 8;
constexpr int kVecsPerThread = kHidden / (kWarp * kElemsPerVec);

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float vec_sumsq(const uint4& value) {
    const nv_bfloat162* halves = reinterpret_cast<const nv_bfloat162*>(&value);
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = __bfloat1622float2(halves[i]);
        result = fmaf(pair.x, pair.x, result);
        result = fmaf(pair.y, pair.y, result);
    }
    return result;
}

__device__ __forceinline__ uint4 scale_vec(
        const uint4& value, float scale) {
    const nv_bfloat162* input = reinterpret_cast<const nv_bfloat162*>(&value);
    uint4 result;
    nv_bfloat162* output = reinterpret_cast<nv_bfloat162*>(&result);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = __bfloat1622float2(input[i]);
        output[i] = __floats2bfloat162_rn(pair.x * scale, pair.y * scale);
    }
    return result;
}

__device__ __forceinline__ uint4 scale_rope_vec(
        const uint4& value, float scale, const float* cos,
        const float* sin, int pair_base) {
    const nv_bfloat162* input = reinterpret_cast<const nv_bfloat162*>(&value);
    uint4 result;
    nv_bfloat162* output = reinterpret_cast<nv_bfloat162*>(&result);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = __bfloat1622float2(input[i]);
        const float even = pair.x * scale;
        const float odd = pair.y * scale;
        output[i] = __floats2bfloat162_rn(
            even * cos[pair_base + i] - odd * sin[pair_base + i],
            even * sin[pair_base + i] + odd * cos[pair_base + i]);
    }
    return result;
}

} // namespace wuda_query_rms_rope
