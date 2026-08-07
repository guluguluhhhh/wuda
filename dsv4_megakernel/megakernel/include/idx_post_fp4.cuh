#pragma once
// ============================================================
// idx_post_fp4.cuh — DSV4 indexer-q post-processing, device chain + kernel.
//
// One head row (128 fp32): round bf16 -> RoPE(tail 64) -> Hadamard-128
// (* 128^-1/2, bf16) -> per-32 MXFP4 (scale = 2^ceil(log2(amax/6)), ue8m0)
// -> packed fp4 + sf word, exactly the score-attention q/sf_q layout; rounding
// matches the golden chain bit-for-bit.
//
// 8 threads per row, 16 cols each; FWHT long strides = 3 in-warp shfl_xor
// levels. LOAD/COMPUTE split lets low-TLP callers software-pipeline batches.
// SHARED single implementation: fused path (wq_b_fp8_gemm.cu xform warpgroup)
// and standalone kernel (idx_post_fp4.cu) stay bitwise-comparable.
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace idx_post {
constexpr int NUM_HEADS = 64;    // indexer heads
constexpr int HEAD_DIM  = 128;   // per-head width (rope on the tail 64)
}  // namespace idx_post

struct IdxRowIn {
    float4 raw[4];        // 16 fp32 columns
    float c8[8], s8[8];   // rope cos/sin (loaded for e >= 4 only)
};

__device__ __forceinline__ void idx_row_load(
    const float* __restrict__ src_row,  // this (m, head)'s 128-col fp32 row
    uint32_t e,                         // 16-col block 0..7
    int m,
    const int* __restrict__ q_pos,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    IdxRowIn& d) {
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        d.raw[i] = *reinterpret_cast<const float4*>(src_row + e * 16 + i * 4);
    if (e >= 4) {
        // this lane's 8 rotary pairs are ids (e-4)*8 .. +7: two float4 loads per
        // table instead of 16 scalar __ldg on the dependency chain
        const int p = __ldg(q_pos + m);
        const float4 c0 = *reinterpret_cast<const float4*>(rope_cos + p * 32 + (e - 4) * 8);
        const float4 c1 = *reinterpret_cast<const float4*>(rope_cos + p * 32 + (e - 4) * 8 + 4);
        const float4 s0 = *reinterpret_cast<const float4*>(rope_sin + p * 32 + (e - 4) * 8);
        const float4 s1 = *reinterpret_cast<const float4*>(rope_sin + p * 32 + (e - 4) * 8 + 4);
        d.c8[0] = c0.x; d.c8[1] = c0.y; d.c8[2] = c0.z; d.c8[3] = c0.w;
        d.c8[4] = c1.x; d.c8[5] = c1.y; d.c8[6] = c1.z; d.c8[7] = c1.w;
        d.s8[0] = s0.x; d.s8[1] = s0.y; d.s8[2] = s0.z; d.s8[3] = s0.w;
        d.s8[4] = s1.x; d.s8[5] = s1.y; d.s8[6] = s1.z; d.s8[7] = s1.w;
    }
}

__device__ __forceinline__ void idx_row_compute(
    const IdxRowIn& d, uint32_t e, int m, int head,
    uint8_t* __restrict__ iq_fp4, int* __restrict__ iq_sf,
    int num_heads = idx_post::NUM_HEADS,
    bool store_ok = true) {   // false = padding row: full compute (warp-converged
                              // shuffles), stores suppressed
    // Paired bf16 rounding: one cvt.rn.bf16x2.f32 per 2 elements (each component
    // rounds RN exactly like the scalar cvt -> bit-identical), ~25% fewer cvt
    // instructions on the transform chain
    const auto bf16r2 = [](float& a, float& b) {
        const float2 f = __bfloat1622float2(__floats2bfloat162_rn(a, b));
        a = f.x; b = f.y;
    };
    float v[16];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) {
        v[i * 4 + 0] = d.raw[i].x; v[i * 4 + 1] = d.raw[i].y;
        v[i * 4 + 2] = d.raw[i].z; v[i * 4 + 3] = d.raw[i].w;
        bf16r2(v[i * 4 + 0], v[i * 4 + 1]);
        bf16r2(v[i * 4 + 2], v[i * 4 + 3]);
    }
    // RoPE on the tail 64 columns (e >= 4): interleaved (even, odd) pairs
    if (e >= 4) {
        #pragma unroll
        for (uint32_t j = 0; j < 8; ++ j) {
            const float c = d.c8[j], s = d.s8[j];
            const float ev = v[j * 2], o = v[j * 2 + 1];
            v[j * 2]     = ev * c - o * s;
            v[j * 2 + 1] = ev * s + o * c;
            bf16r2(v[j * 2], v[j * 2 + 1]);
        }
    }
    // FWHT-128: strides 1..8 inside the thread; strides 16/32/64 across the
    // 8-lane group (butterfly stages commute; fixed small-to-large order)
    #pragma unroll
    for (uint32_t s = 1; s <= 8; s <<= 1) {
        #pragma unroll
        for (uint32_t i = 0; i < 16; ++ i) {
            if ((i & s) == 0) {
                const float a = v[i], b = v[i + s];
                v[i] = a + b;
                v[i + s] = a - b;
            }
        }
    }
    #pragma unroll
    for (uint32_t x = 1; x <= 4; x <<= 1) {
        const bool upper = (e & x) != 0;
        #pragma unroll
        for (uint32_t i = 0; i < 16; ++ i) {
            const float other = __shfl_xor_sync(0xffffffffu, v[i], static_cast<int>(x));
            v[i] = upper ? (other - v[i]) : (v[i] + other);
        }
    }
    // * 128^-1/2 + bf16 rounding (hadamard_transform returns bf16), then the
    // power-of-2 scale (golden fast_round_scale): 2^ceil(log2(amax/6)) over the
    // 32-col quant block = this lane's 16 + its xor-1 partner's 16
    constexpr float kHadamardScale = 0.08838834764831845f;   // 1/sqrt(128)
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 2) {
        v[i]     *= kHadamardScale;
        v[i + 1] *= kHadamardScale;
        bf16r2(v[i], v[i + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(v[i]), fabsf(v[i + 1])));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, 7.052966400779935e-38f);              // 6 * 2^-126
    const uint32_t rbits = __float_as_uint(amax * (1.0f / 6.0f));
    const int32_t sexp = static_cast<int32_t>((rbits >> 23) & 0xff) - 127
                         + (((rbits & 0x7fffffu) != 0) ? 1 : 0);
    const float inv_scale = __uint_as_float(static_cast<uint32_t>(127 - sexp) << 23);
    const uint32_t e8m0 = static_cast<uint32_t>(sexp + 127);

    // 16 values -> 8 packed fp4 bytes (low nibble = even element)
    uint32_t packed[2];
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++ i) {
        uint32_t w = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++ j) {
            const float lo = v[i * 8 + j * 2] * inv_scale;
            const float hi = v[i * 8 + j * 2 + 1] * inv_scale;
            uint32_t b32;
            asm volatile("{\n\t.reg .b8 b;\n\tcvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n\t"
                         "cvt.u32.u8 %0, b;\n\t}"
                         : "=r"(b32) : "f"(lo), "f"(hi));
            w |= b32 << (j * 8);
        }
        packed[i] = w;
    }
    if (store_ok) {
        const uint2 pk = make_uint2(packed[0], packed[1]);
        const int64_t off = ((int64_t)m * num_heads + head) * (idx_post::HEAD_DIM / 2);
        reinterpret_cast<uint2*>(iq_fp4 + off)[e] = pk;
    }

    // packed-ue8m0 SF word (byte b = 32-col block b = lane pair e/2); the two
    // xor levels fold all 4 even-lane bytes into e == 0 (odd lanes carry 0)
    uint32_t sf = (e % 2 == 0) ? (e8m0 << ((e / 2) * 8)) : 0u;
    sf |= __shfl_xor_sync(0xffffffffu, sf, 2);
    sf |= __shfl_xor_sync(0xffffffffu, sf, 4);
    if (e == 0 && store_ok) {
        iq_sf[(int64_t)m * num_heads + head] = static_cast<int>(sf);
    }
}

// load + compute in one step (no pipelining)
__device__ __forceinline__ void idx_postprocess_row(
    const float* __restrict__ src_row, uint32_t e, int m, int head,
    const int* __restrict__ q_pos,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    uint8_t* __restrict__ iq_fp4, int* __restrict__ iq_sf,
    int num_heads = idx_post::NUM_HEADS, bool store_ok = true) {
    IdxRowIn d;
    idx_row_load(src_row, e, m, q_pos, rope_cos, rope_sin, d);
    idx_row_compute(d, e, m, head, iq_fp4, iq_sf, num_heads, store_ok);
}

// ======================== Standalone post-processing kernel ========================
// Grid-parallel over all M x 64 rows, 8 lanes per row, 32 rows per 256-thread
// CTA. Out-of-range threads CLAMP to the last row (keeps warps converged for
// the full-mask shuffles) and suppress their stores. `static`: each including
// TU / extension module carries its own instantiation.
static __global__ void __launch_bounds__(256, 1)
idx_post_kernel(
    const float* __restrict__ iq_f32,   // [M, 64, 128]
    const int* __restrict__ q_pos,
    const float* __restrict__ rope_cos, const float* __restrict__ rope_sin,
    uint8_t* __restrict__ iq_fp4, int* __restrict__ iq_sf,
    int total_rows, int num_heads)
{
    const int raw = blockIdx.x * (256 / 8) + (int)(threadIdx.x >> 3);  // m*64+head
    const int row = raw < total_rows ? raw : total_rows - 1;
    idx_postprocess_row(iq_f32 + (int64_t)row * idx_post::HEAD_DIM,
                        threadIdx.x & 7,
                        row / num_heads, row % num_heads,
                        q_pos, rope_cos, rope_sin, iq_fp4, iq_sf,
                        num_heads, raw < total_rows);
}
