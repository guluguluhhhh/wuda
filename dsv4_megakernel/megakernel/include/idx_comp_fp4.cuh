#pragma once
// ============================================================
// idx_comp_fp4.cuh — fused CUDA-core post-processing for the wq_b merged GEMM
// (warps 8..11, WARP-LEVEL: 32 threads/row, 4 rows in flight per CTA):
//   * indexer COMPRESSOR:
//       [compress row] read the framework's 8-entry state ring -> overlap-cat
//       softmax aggregate -> bf16 RMSNorm(128) -> RoPE(last 64), then either
//       RTP FP8 (no Hadamard, one FP32 scale) or legacy FWHT-128 + block-32
//       FP4. FRONT-EMIT has already published the fresh state row.
//   * LOCAL KV WINDOW post-processing (full chain, i.e. through the fp8
//     quant): weighted RMSNorm(512) -> RoPE(last 64) -> per-64 fp8
//     e4m3 quant of [0,448) (main-compressor semantics: scale =
//     max(amax,1e-4)/448, fp32) + bf16 rope tail [448,512).
// NO split-K: FRONT-EMIT publishes reduced indexer state and win_y2 [M,512]
// arrives pre-reduced.
//
// Why warp-level: a 128-thread/row version serialized ~12 NamedBarrier phases
// of raw-latency global traffic per row (~12us/row); here lane l owns a
// contiguous column slice, every phase is register-resident + warp shuffles.
// ============================================================
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

namespace idx_comp {

constexpr int RATIO = 4;      // compress every 4 tokens
constexpr int SROWS = 8;      // state ring rows (2*RATIO, overlap window)
constexpr int D_I   = 128;    // compressed head dim
constexpr int RD    = 64;     // rope dims (last RD of D_I)
constexpr int WK_I  = 256;    // wkv / wgate width each (overlap-cat halves)
constexpr int D_W   = 512;    // local-window kv dim (winkv row width)
constexpr int NF8_W = D_W - RD;   // 448 fp8-quantized cols (rope tail stays bf16)

// Both pools group every entry's data before the per-entry scale records.
// RTP-LLM supplies logical entries-per-block independently from the physical
// byte stride, which may include FlashMLA TMA padding.
constexpr int M1_TOK_BODY    = NF8_W + 2 * RD;                        // 576

// All DEVICE pointers, caller-owned. state == nullptr disables the compressor;
// win_y2 == nullptr disables the local-window chain. pos/cos_tab/sin_tab are
// shared by both chains. COMPACT outputs (q4/s4, win_q8/s8/rope) are OPTIONAL
// once the paged direct writes are on (null = skip: saves ~600B/row of
// double-write bandwidth in the production form).
struct Args {
    const long long* pos = nullptr;   // [M] absolute token positions
    // (y4/ape are GONE: the FRONT-EMIT epilogue publishes the fresh idx
    // state row -- fp32 accum, +idx_ape on the score half -- straight into
    // the pools below; this chain is now a pure POOL READER.)
    const float* norm_w  = nullptr;   // [D_I] compressor RMSNorm weight
    const float* cos_tab = nullptr;   // [S, RD/2] rope tables (S > max pos)
    const float* sin_tab = nullptr;
    // Framework scratch state: one ring per request, each entry is
    // [kv(WK_I) | score(WK_I)]. state_row is already folded to
    // block_id * state_ring_entries + pos % state_ring_entries.
    const float* state = nullptr;     // [blocks, ring, 2 * WK_I] fp32
    const int* state_row = nullptr;   // [M], -1 skips the row
    int state_ring_entries = 0;
    uint8_t* q4 = nullptr;            // [M, D_I/2]  packed e2m1 (compress rows)
    uint8_t* s4 = nullptr;            // [M, D_I/32] block-32 ue8m0 exponents
    // ---- local kv window (winkv) ----
    const float* win_y2   = nullptr;  // [M, D_W] pre-reduced wkv projection
    const float* win_norm = nullptr;  // [D_W] kv RMSNorm weight
    uint8_t* win_q8   = nullptr;      // [M, NF8_W] e4m3
    float*   win_s8   = nullptr;      // [M, NF8_W/64] per-64 fp32 scales (pow2)
    __nv_bfloat16* win_rope = nullptr;// [M, RD] rope tail (bf16)
    // Flat destinations use block_id * entries_per_block + offset. Logical
    // geometry and physical page stride are deliberately independent.
    uint8_t* idx_cache  = nullptr;
    const int* idx_dst  = nullptr;    // [M] compressed-token dst (compress rows)
    int idx_entries_per_block = 0;
    size_t idx_block_stride_bytes = 0;
    uint8_t* swa_cache  = nullptr;
    const int* swa_dst  = nullptr;    // [M] token dst (every row)
    int swa_entries_per_block = 0;
    size_t swa_block_stride_bytes = 0;
};

// bf16 round-trip via intrinsics (torch extensions define
// __CUDA_NO_BFLOAT16_CONVERSIONS__, so the (nv_bfloat16)float cast is unavailable)
__device__ __forceinline__ float bf16r(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

// ceil(log2(x)) for x > 0 / 2^e — exact bit tricks (delivery tile/quant.cuh)
__device__ __forceinline__ int flog2_ceil(float x) {
    const unsigned bits = __float_as_uint(x);
    const int exp_field = (int)((bits >> 23) & 0xFF);
    return exp_field - 127 + (((bits & 0x7FFFFFu) != 0u) ? 1 : 0);
}
__device__ __forceinline__ float fpow2(int e) {
    return __uint_as_float((unsigned)(e + 127) << 23);
}

// One token row, ONE warp (lane = 0..31); lane l owns cols {4l..4l+3}.
template <bool kFp8>
__device__ inline void process_row(const Args& a, int m, int lane,
                                   float eps = 1e-6f) {
    const long long p = a.pos[m];
    // FRONT-EMIT published this token's fresh row before this kernel launches.
    if (((p + 1) & (RATIO - 1)) != 0)
        return;                            // not a compress row (row-uniform)

    const int srow = a.state_row[m];
    if (srow < 0)
        return;
    const int c0 = 4 * lane;

    // Logical row rr of the overlap window ending at p lives at ring row
    // (p - (SROWS - 1) + rr) % ring. The first four rows read the overlap half
    // and the final four rows read the current half of each state entry.
    float v[4];
    {
        const int ring = a.state_ring_entries;
        const int rcur = srow % ring;
        const int r0 = (rcur >= SROWS - 1) ? rcur - (SROWS - 1)
                                           : rcur - (SROWS - 1) + ring;
        const float* base = a.state + (size_t)(srow - rcur) * (2 * WK_I);
        float4 sc8[SROWS], kv8[SROWS];
        #pragma unroll
        for (int rr = 0; rr < SROWS; ++rr) {
            int pr = r0 + rr;
            if (pr >= ring)
                pr -= ring;
            const int col = (rr < RATIO) ? c0 : (D_I + c0);
            const float* entry = base + (size_t)pr * (2 * WK_I);
            sc8[rr] = *reinterpret_cast<const float4*>(entry + WK_I + col);
            kv8[rr] = *reinterpret_cast<const float4*>(entry + col);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s0[SROWS], k0[SROWS];
            #pragma unroll
            for (int rr = 0; rr < SROWS; ++rr) {
                s0[rr] = (&sc8[rr].x)[j];
                k0[rr] = (&kv8[rr].x)[j];
            }
            float max_logit = s0[0];
            #pragma unroll
            for (int rr = 1; rr < SROWS; ++rr) max_logit = fmaxf(max_logit, s0[rr]);
            float denom = 0.f, wsum = 0.f;
            #pragma unroll
            for (int rr = 0; rr < SROWS; ++rr) {
                const float w = expf(s0[rr] - max_logit);
                denom += w;
                wsum  += w * k0[rr];
            }
            v[j] = wsum / denom;
        }
    }

    // ---- bf16 RMSNorm over D_I (lane 4 + 5-level shuffle reduction) ----
    {
        float sumsq = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = bf16r(v[j]);
            sumsq += b * b;
        }
        #pragma unroll
        for (int x = 16; x >= 1; x >>= 1)
            sumsq += __shfl_xor_sync(0xffffffffu, sumsq, x);
        const float rms = rsqrtf(sumsq / float(D_I) + eps);
        const float4 nw = *reinterpret_cast<const float4*>(a.norm_w + c0);
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = bf16r(bf16r(v[j]) * rms * (&nw.x)[j]);
    }

    // ---- interleaved RoPE on the last RD dims: lanes 16..31 hold cols 64..127,
    // each lane owns TWO adjacent (even,odd) pairs -- fully in-lane. ----
    if (lane >= (D_I - RD) / 4) {
        const long long rope_pos = p + 1 - RATIO;            // >= 0 on compress rows
        const int jp = (c0 - (D_I - RD)) / 2;                // first pair index
        const float2 cs0 = make_float2(a.cos_tab[(size_t)rope_pos * (RD / 2) + jp],
                                       a.sin_tab[(size_t)rope_pos * (RD / 2) + jp]);
        const float2 cs1 = make_float2(a.cos_tab[(size_t)rope_pos * (RD / 2) + jp + 1],
                                       a.sin_tab[(size_t)rope_pos * (RD / 2) + jp + 1]);
        const float e0 = v[0], o0 = v[1], e1 = v[2], o1 = v[3];
        v[0] = bf16r(e0 * cs0.x - o0 * cs0.y);
        v[1] = bf16r(e0 * cs0.y + o0 * cs0.x);
        v[2] = bf16r(e1 * cs1.x - o1 * cs1.y);
        v[3] = bf16r(e1 * cs1.y + o1 * cs1.x);
    }

    if constexpr (kFp8) {
        // RTP Indexer cache: no Hadamard. One ordinary FP32 scale covers the
        // whole 128-vector and is stored after all K bytes in the physical page.
        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])),
                           fmaxf(fabsf(v[2]), fabsf(v[3])));
        #pragma unroll
        for (int x = 16; x >= 1; x >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, x));
        const float scale = fmaxf(amax * (1.0f / 448.0f), 1.0e-12f);
        alignas(4) uint8_t q[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            q[j] = __nv_fp8_e4m3(v[j] / scale).__x;
        if (a.q4 != nullptr)
            *reinterpret_cast<uint32_t*>(a.q4 + (size_t)m * D_I + lane * 4) =
                *reinterpret_cast<const uint32_t*>(q);
        if (lane == 0 && a.s4 != nullptr)
            *reinterpret_cast<float*>(a.s4 + (size_t)m * sizeof(float)) = scale;
        if (a.idx_cache != nullptr && a.idx_dst != nullptr && a.idx_dst[m] >= 0) {
            const int dst = a.idx_dst[m];
            const int epb = a.idx_entries_per_block;
            uint8_t* page = a.idx_cache
                + (size_t)(dst / epb) * a.idx_block_stride_bytes;
            const int off = dst % epb;
            *reinterpret_cast<uint32_t*>(page + off * D_I + lane * 4) =
                *reinterpret_cast<const uint32_t*>(q);
            if (lane == 0)
                *reinterpret_cast<float*>(page + (size_t)epb * D_I
                                          + off * sizeof(float)) = scale;
        }
        return;
    }

    // ---- FWHT-128, natural order: value index = 4*lane + j.
    // h=1,2 in-thread; h=4..64 via shfl_xor(lane, h/4). ----
    {
        float t0 = v[0] + v[1], t1 = v[0] - v[1];            // h = 1
        float t2 = v[2] + v[3], t3 = v[2] - v[3];
        v[0] = t0 + t2; v[1] = t1 + t3;                      // h = 2
        v[2] = t0 - t2; v[3] = t1 - t3;
        #pragma unroll
        for (int k = 0; k < 5; ++k) {                        // h = 4..64
            const int hl = 1 << k;
            const bool hi = (lane & hl) != 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float pv = __shfl_xor_sync(0xffffffffu, v[j], hl);
                v[j] = hi ? (pv - v[j]) : (v[j] + pv);
            }
        }
        constexpr float inv_sqrt = 0.088388347648318447f;    // 128^-1/2
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = bf16r(v[j] * inv_sqrt);
    }

    // ---- fp4(e2m1) block-32 quant: block = 8 lanes; xor-shuffles <= 4 stay
    // inside the 8-lane group. scale = 2^ceil(log2(amax/6)), ue8m0. ----
    {
        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])),
                           fmaxf(fabsf(v[2]), fabsf(v[3])));
        #pragma unroll
        for (int x = 4; x >= 1; x >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, x));
        amax = fmaxf(amax, 6.0f * 1.1754944e-38f);
        const int se = flog2_ceil(amax * (1.0f / 6.0f));
        const float scale = fpow2(se);
        if ((lane & 7) == 0 && a.s4 != nullptr)
            a.s4[(size_t)m * (D_I / 32) + (lane >> 3)] = (uint8_t)(se + 127);
        const __nv_fp4x2_e2m1 lo(make_float2(v[0] / scale, v[1] / scale));
        const __nv_fp4x2_e2m1 hi(make_float2(v[2] / scale, v[3] / scale));
        const uint16_t packed = (uint16_t)((uint8_t)lo.__x | ((uint16_t)(uint8_t)hi.__x << 8));
        if (a.q4 != nullptr)
            *reinterpret_cast<uint16_t*>(a.q4 + (size_t)m * (D_I / 2) + lane * 2) = packed;
        // Direct store into the RTP physical indexer page.
        if (a.idx_cache != nullptr && a.idx_dst != nullptr && a.idx_dst[m] >= 0) {
            const int dst = a.idx_dst[m];
            const int epb = a.idx_entries_per_block;
            uint8_t* page = a.idx_cache
                + (size_t)(dst / epb) * a.idx_block_stride_bytes;
            const int off = dst % epb;
            *reinterpret_cast<uint16_t*>(page + off * (D_I / 2) + lane * 2) = packed;
            if ((lane & 7) == 0)
                page[(size_t)epb * (D_I / 2) + off * 4 + (lane >> 3)] =
                    (uint8_t)(se + 127);
        }
    }
}

// One LOCAL KV WINDOW row (CSA stage 4 full chain), ONE warp; lane l owns
// cols {16l..16l+15} of the 512-wide row. Delivery numerics: sum-of-squares
// over the RAW fp32 row; quant region [0,448) is bf16-rounded AFTER the
// weighted normalize (dtail semantic) then per-64 fp8 (scale = max(amax,1e-4)
// /448, fp32); rope pairs [448,512) are weighted-normalized in fp32, rotated
// by the token's OWN position, then bf16. Runs on EVERY row (no compress gate).
__device__ inline void process_win_row(const Args& a, int m, int lane,
                                       float eps = 1e-6f) {
    float v[16];
    {
        const float4* src = reinterpret_cast<const float4*>(
            a.win_y2 + (size_t)m * D_W) + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 f = src[i];
            v[4 * i + 0] = f.x; v[4 * i + 1] = f.y;
            v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
        }
    }
    // ---- RMSNorm scale from the raw fp32 row (16/lane + 5-level shuffle) ----
    float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) ss += v[j] * v[j];
    #pragma unroll
    for (int x = 16; x >= 1; x >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, x);
    const float rms = rsqrtf(ss / float(D_W) + eps);
    {
        const float4* wv = reinterpret_cast<const float4*>(a.win_norm) + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 w4 = wv[i];
            v[4 * i + 0] *= rms * w4.x; v[4 * i + 1] *= rms * w4.y;
            v[4 * i + 2] *= rms * w4.z; v[4 * i + 3] *= rms * w4.w;
        }
    }
    if (lane < NF8_W / 16) {                    // lanes 0..27: fp8 quant region
        #pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = bf16r(v[j]);
        float amax = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(v[j]));
        // mask = lanes 0..27 ONLY: lanes 28..31 are in the rope branch and never
        // reach this shuffle -- a full mask would deadlock the warp. xor <= 3
        // stays inside the 4-lane block group.
        #pragma unroll
        for (int x = 2; x >= 1; x >>= 1)        // 4-lane group == one 64 block
            amax = fmaxf(amax, __shfl_xor_sync(0x0fffffffu, amax, x));
        // MODEL1 scale: pow2-ceil of clamp(amax/448, 1e-4) -- e8m0-exact (the
        // FlashMLA MODEL1_FP8Sparse contract; was fp32 amax/448 pre-kvcache).
        const int se = flog2_ceil(fmaxf(amax * (1.0f / 448.0f), 1e-4f));
        const float scale = fpow2(se);
        if ((lane & 3) == 0 && a.win_s8 != nullptr)
            a.win_s8[(size_t)m * (NF8_W / 64) + (lane >> 2)] = scale;
        alignas(16) uint8_t q[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            q[j] = __nv_fp8_e4m3(v[j] / scale).__x;
        if (a.win_q8 != nullptr)
            *reinterpret_cast<uint4*>(a.win_q8 + (size_t)m * NF8_W + lane * 16) =
                *reinterpret_cast<const uint4*>(q);
        // kvcache design: DIRECT store into the SWA MODEL1 page (body nope
        // bytes + tail e8m0 scale; same quant chain as the compact outputs).
        if (a.swa_cache != nullptr && a.swa_dst != nullptr && a.swa_dst[m] >= 0) {
            const int dst = a.swa_dst[m];
            const int epb = a.swa_entries_per_block;
            uint8_t* page = a.swa_cache
                + (size_t)(dst / epb) * a.swa_block_stride_bytes;
            const int off = dst % epb;
            *reinterpret_cast<uint4*>(page + off * M1_TOK_BODY + lane * 16) =
                *reinterpret_cast<const uint4*>(q);
            if ((lane & 3) == 0)
                page[(size_t)epb * M1_TOK_BODY + off * 8 + (lane >> 2)] =
                    (uint8_t)(se + 127);
        }
    } else {                                    // lanes 28..31: rope tail (bf16)
        const long long p = a.pos[m];
        const float* cos_row = a.cos_tab + (size_t)p * (RD / 2);
        const float* sin_row = a.sin_tab + (size_t)p * (RD / 2);
        const int j0 = (lane - NF8_W / 16) * 8;        // first pair index
        alignas(16) __nv_bfloat16 out[16];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float c = cos_row[j0 + k], s = sin_row[j0 + k];
            const float e = v[2 * k], o = v[2 * k + 1];
            out[2 * k]     = __float2bfloat16(e * c - o * s);
            out[2 * k + 1] = __float2bfloat16(e * s + o * c);
        }
        if (a.win_rope != nullptr) {
            *reinterpret_cast<uint4*>(a.win_rope + (size_t)m * RD + j0 * 2) =
                *reinterpret_cast<const uint4*>(out);
            *reinterpret_cast<uint4*>(a.win_rope + (size_t)m * RD + j0 * 2 + 8) =
                *reinterpret_cast<const uint4*>(out + 8);
        }
        // kvcache design: rope tail into the SWA MODEL1 body (cols 448..511).
        if (a.swa_cache != nullptr && a.swa_dst != nullptr && a.swa_dst[m] >= 0) {
            const int dst = a.swa_dst[m];
            const int epb = a.swa_entries_per_block;
            uint8_t* base = a.swa_cache
                + (size_t)(dst / epb) * a.swa_block_stride_bytes
                + (size_t)(dst % epb) * M1_TOK_BODY + NF8_W;
            *reinterpret_cast<uint4*>(base + j0 * 4) =
                *reinterpret_cast<const uint4*>(out);
            *reinterpret_cast<uint4*>(base + j0 * 4 + 16) =
                *reinterpret_cast<const uint4*>(out + 8);
        }
    }
}

} // namespace idx_comp
