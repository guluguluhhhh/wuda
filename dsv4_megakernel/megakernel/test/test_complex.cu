// ============================================================
//  test_cgemm_norm.cu — complex_gemm STEP 1: 1-SM swap-AB bf16 GEMM FUSED WITH
//  per-token RMSNorm (operator under test: complex_a.cuh).
//  Target: B300 (SM100), decode regime. Shape: M=1..64, N=4608, K=7168.
//      y  = A * B            (full [M,N] split-K GEMM)
//      D[:, 0:1536] = rmsnorm(y[:, 0:1536], rms_w[0:1536], eps)   <- STEP 2 (y1 only)
//      D[:, 1536:N] : NOT written here (y2/y3/y4 partials stay in ws for step-3)
//  A=(M,K) activation, B=(N,K) weight (both bf16, row-major), rms_w length N (fp32,
//  only the first FUSENORM_NORM_DIM=1536 entries are used).
//
//  Four sections:
//    (1) Correctness: fused output vs an fp32 GOLDEN (cuBLASLt fp32 GEMM ->
//        fp32 RMSNorm over the FIRST 1536 cols -> bf16); both normalize
//        FULL-precision GEMM, so the only residual gap is the final bf16 round.
//        Only the first 1536 columns are compared (the rest are unwritten).
//    (2) Performance: fused single launch vs the realistic bf16 pipeline
//        (cuBLASLt bf16 GEMM + separate fast CUDA RMSNorm), M-sweep over 1..64.
//    (3) Full pipeline (op A -> op B = gemm_fuse_norm_b): op A's y1 (D[:,0:1536],
//        pitch 4608) feeds op B via a pitched TMA read (lda=4608, zero-copy);
//        op B does y1 @ w2[65536,1536].T + per-head(512) weightless RMSNorm.
//        Checked vs an fp32 cuBLAS golden at M in {32..256}.
//    (4) Pipeline PERFORMANCE: op A / op B / A+B latency vs a naive cuBLAS-bf16
//        pipeline (GEMM1 + rmsnorm + slice + GEMM2 + head-norm) + effective BW.
//
//  编译:
//    nvcc -std=c++17 -gencode arch=compute_100f,code=sm_100 -O3 \
//         --expt-relaxed-constexpr --expt-extended-lambda \
//         -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//         -o test_cgemm_norm test_cgemm_norm.cu -lcublas -lcublasLt -lcuda -Xptxas -v
// ============================================================

#include "complex_a.cuh"          // op A: FuseNormCtx / fusenorm_setup/run/free, bf16_t (STEP 1+2)
#include "complex_b.cuh"      // op B: gfnb::wq_b_proj_run (STEP 3, tcgen05 2SM GEMM + head RMSNorm)
#include "cublas_baseline.cuh"   // Test 4: self-contained cuBLAS baseline (GEMM + CSA compressor(y3)+indexer(y4) post-proc + real quant)
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cmath>
#include <cstdio>

// ============================================================
//  Test 4 helpers: load raw little-endian dumps from test5_data/ (see
//  test5_compressor_golden.py). Shapes are fixed by that script's config.
// ============================================================
#define TEST5_DIR "test5_data/"
static void* t5_load(const char* name, size_t expect_bytes) {
    char path[512];
    snprintf(path, sizeof(path), "%s%s.bin", TEST5_DIR, name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("  [Test4] MISSING %s (run: python test5_compressor_golden.py)\n", path); return nullptr; }
    fseek(f, 0, SEEK_END); size_t n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    if (expect_bytes && n != expect_bytes)
        printf("  [Test4] WARN %s: %zu bytes, expected %zu\n", name, n, expect_bytes);
    void* h = malloc(n);
    size_t rd = fread(h, 1, n, f); (void)rd; fclose(f);
    return h;
}
// upload host buffer -> fresh device buffer of the same byte count
static void* t5_dev(const void* host, size_t bytes) {
    void* d = nullptr; cudaMalloc(&d, bytes); cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice); return d;
}
static size_t t5_fsize(const char* name) {
    char path[512]; snprintf(path, sizeof(path), "%s%s.bin", TEST5_DIR, name);
    FILE* f = fopen(path, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); size_t n = (size_t)ftell(f); fclose(f); return n;
}

// ============================================================
//  Test-data init (values in [-1,1]; A=(M,K), B=(N,K) row-major).
// ============================================================
static void init_test_data(bf16_t* dA, bf16_t* dB, int M, int N, int K, int seed) {
    {
        size_t sz = (size_t)M * K;
        bf16_t* h = (bf16_t*)malloc(sz * sizeof(bf16_t));
        srand(seed);
        for (size_t i = 0; i < sz; i++)
            h[i] = __float2bfloat16((float)(rand() % 200 - 100) / 100.0f);
        cudaMemcpy(dA, h, sz * sizeof(bf16_t), cudaMemcpyHostToDevice);
        free(h);
    }
    {
        size_t sz = (size_t)N * K;
        bf16_t* h = (bf16_t*)malloc(sz * sizeof(bf16_t));
        for (size_t i = 0; i < sz; i++)
            h[i] = __float2bfloat16((float)(rand() % 200 - 100) / 100.0f);
        cudaMemcpy(dB, h, sz * sizeof(bf16_t), cudaMemcpyHostToDevice);
        free(h);
    }
}

// RMSNorm gain (rms_w): fp32, length N. Realistic gamma ~ 1 -> draw in [0.5, 1.5].
static void init_rms_w(float* d_rms_w, int N, int seed) {
    float* h = (float*)malloc((size_t)N * sizeof(float));
    srand(seed);
    for (int i = 0; i < N; i++) h[i] = 0.5f + (float)(rand() % 1000) / 1000.0f;   // [0.5,1.5)
    cudaMemcpy(d_rms_w, h, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

// ============================================================
//  Standalone CUDA RMSNorm kernel (templated on input element type).
//  D is (M,stride) row-major (token-major): each block owns ONE token row m,
//  reduces the row's sum-of-squares over its FIRST norm_len features, then writes
//      out[m,n] = in[m,n] * rsqrt(mean(row[0:norm_len]^2) + eps) * rms_w[n]  (bf16)
//  for n < norm_len only (n in [norm_len, stride) untouched). This matches the
//  fused op's STEP-2 semantics (only y1 = first 1536 columns normalized).
//  Two uses (same math, different input precision):
//    - TIn=float  : the CORRECTNESS golden (reads the fp32 cuBLAS GEMM result, so
//                   it normalizes FULL-precision values exactly like the fused op).
//    - TIn=bf16_t : the realistic "cuBLAS bf16 GEMM + separate rmsnorm" PERF
//                   baseline (reads the bf16 GEMM output).
// ============================================================
__device__ __forceinline__ float rmsnorm_ld(const __nv_bfloat16& x) { return __bfloat162float(x); }
__device__ __forceinline__ float rmsnorm_ld(const float& x) { return x; }

template <class TIn>
static __global__ void rmsnorm_ref_kernel(const TIn* __restrict__ in,
                                          bf16_t* __restrict__ out,
                                          const float* __restrict__ rms_w,
                                          float eps, int norm_len, int stride) {
    const int    row  = blockIdx.x;              // one block per token
    const size_t base = (size_t)row * stride;    // physical row stride (== full N)
    __shared__ float red[32];                    // per-warp partials (blockDim<=1024)
    float local = 0.f;
    for (int c = threadIdx.x; c < norm_len; c += blockDim.x) {
        float v = rmsnorm_ld(in[base + c]);
        local += v * v;
    }
    // block reduction (warp shuffle + shared broadcast)
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = local;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    float ss = (threadIdx.x < nwarps) ? red[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[0] = ss;
    }
    __syncthreads();
    const float rms = rsqrtf(red[0] / (float)norm_len + eps);
    for (int c = threadIdx.x; c < norm_len; c += blockDim.x) {
        float v = rmsnorm_ld(in[base + c]);
        out[base + c] = __float2bfloat16(v * rms * rms_w[c]);
    }
}

template <class TIn>
static void rmsnorm_ref_run(const TIn* in, bf16_t* out, const float* rms_w,
                            float eps, int M, int norm_len, int stride, cudaStream_t stream = 0) {
    int tpb = 256;
    if (tpb > norm_len) { tpb = ((norm_len + 31) / 32) * 32; if (tpb < 32) tpb = 32; }
    rmsnorm_ref_kernel<TIn><<<M, tpb, 0, stream>>>(in, out, rms_w, eps, norm_len, stride);
}

// ------------------------------------------------------------
//  Fast bf16 RMSNorm (PERF baseline). Mirrors the fused kernel's shape: one block
//  per token row, each thread owns ONE 8-feature group -> vectorized uint4 load
//  (8 bf16 = 16 B) ONCE into registers, block-reduce the sum-of-squares, then
//  normalize + uint4 store from registers (no second global read). Only the first
//  norm_len features (y1) are normalized/written; stride is the full row pitch.
//  REQUIRES norm_len % 8 == 0 (1536 % 8 == 0 holds); blockDim = round_up(norm_len/8, 32).
// ------------------------------------------------------------
static __global__ void rmsnorm_fast_kernel(const bf16_t* __restrict__ in,
                                           bf16_t* __restrict__ out,
                                           const float* __restrict__ rms_w,
                                           float eps, int norm_len, int stride) {
    const int    col    = threadIdx.x << 3;              // 8 features per thread
    const bool   active = (col < norm_len);              // threads beyond y1 width idle
    const size_t base   = (size_t)blockIdx.x * stride + col;
    float s[8];
    if (active) {
        uint4 raw = *reinterpret_cast<const uint4*>(in + base);   // 8 bf16 = 16 B, one load
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
        float2 f0 = __bfloat1622float2(h[0]), f1 = __bfloat1622float2(h[1]);
        float2 f2 = __bfloat1622float2(h[2]), f3 = __bfloat1622float2(h[3]);
        s[0]=f0.x; s[1]=f0.y; s[2]=f1.x; s[3]=f1.y;
        s[4]=f2.x; s[5]=f2.y; s[6]=f3.x; s[7]=f3.y;
    } else {
        #pragma unroll
        for (int e = 0; e < 8; ++e) s[e] = 0.f;
    }
    float ss = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) ss += s[e] * s[e];
    // block reduction (warp shuffle + shared broadcast; blockDim is a warp multiple)
    __shared__ float red[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nwarps) ? red[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    const float rms = rsqrtf(red[0] / (float)norm_len + eps);
    if (!active) return;
    float4 w0 = *reinterpret_cast<const float4*>(rms_w + col);
    float4 w1 = *reinterpret_cast<const float4*>(rms_w + col + 4);
    __nv_bfloat162 o2[4];
    o2[0] = __floats2bfloat162_rn(s[0]*rms*w0.x, s[1]*rms*w0.y);
    o2[1] = __floats2bfloat162_rn(s[2]*rms*w0.z, s[3]*rms*w0.w);
    o2[2] = __floats2bfloat162_rn(s[4]*rms*w1.x, s[5]*rms*w1.y);
    o2[3] = __floats2bfloat162_rn(s[6]*rms*w1.z, s[7]*rms*w1.w);
    *reinterpret_cast<uint4*>(out + base) = *reinterpret_cast<uint4*>(o2);
}

static void rmsnorm_fast_run(const bf16_t* in, bf16_t* out, const float* rms_w,
                             float eps, int M, int norm_len, int stride, cudaStream_t stream = 0) {
    const int tpb = ((norm_len >> 3) + 31) & ~31;   // round norm_len/8 up to a warp multiple
    rmsnorm_fast_kernel<<<M, tpb, 0, stream>>>(in, out, rms_w, eps, norm_len, stride);
}

// ------------------------------------------------------------
//  GOLDEN for op B's CUDA-core tail (legacy helper; not currently invoked). Given the FULL fp32 op-A GEMM
//  y = x @ w1.T [M, N1], produce the expected op-A D tail (bf16, cols [1536,4352)):
//    - y2 (cols [1536,2048)) : weighted RMSNorm over the 512-wide y2 (gain rms_w2),
//      then (if rope_cs) interleaved RoPE on the NORMALIZED last 64 dims [1984,2048)
//    - y3/y4 (cols [2048,4352)): reduce-only (identity fp32 -> bf16)
//  Mirrors the op B tail exactly (norm -> rope, pairs (2j,2j+1)). One block per row.
// ------------------------------------------------------------
static __global__ void tail_golden_kernel(const float* __restrict__ y, bf16_t* __restrict__ out,
                                          const float* __restrict__ rms_w2, float eps, int N1,
                                          const float2* __restrict__ rope_cs) {
    constexpr int Y2_LO = 1536, Y2_HI = 2048, ROW_HI = 4352, Y2_W = Y2_HI - Y2_LO;
    constexpr int ROPE_DIM = 64, ROPE_PAIRS = ROPE_DIM / 2, ROPE_LO = Y2_HI - ROPE_DIM;  // 1984
    const bool   do_rope = (rope_cs != nullptr);
    const int    m    = blockIdx.x;
    const size_t base = (size_t)m * N1;
    __shared__ float red[32];
    float local = 0.f;
    for (int n = Y2_LO + threadIdx.x; n < Y2_HI; n += blockDim.x) { float v = y[base + n]; local += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = local;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    float ss = (threadIdx.x < nwarps) ? red[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) red[0] = ss;
    }
    __syncthreads();
    const float rms = rsqrtf(red[0] / (float)Y2_W + eps);
    for (int n = Y2_LO + threadIdx.x; n < ROW_HI; n += blockDim.x) {
        if (do_rope && n >= ROPE_LO && n < Y2_HI) continue;   // rope region handled pairwise below
        float v = y[base + n];
        if (n < Y2_HI) v = v * rms * rms_w2[n - Y2_LO];
        out[base + n] = __float2bfloat16(v);
    }
    if (do_rope && threadIdx.x < ROPE_PAIRS) {
        const int j = threadIdx.x, ne = ROPE_LO + 2 * j, no = ne + 1;
        float e = y[base + ne] * rms * rms_w2[ne - Y2_LO];
        float o = y[base + no] * rms * rms_w2[no - Y2_LO];
        const float2 cs = rope_cs[(size_t)m * ROPE_PAIRS + j];
        out[base + ne] = __float2bfloat16(e * cs.x - o * cs.y);
        out[base + no] = __float2bfloat16(e * cs.y + o * cs.x);
    }
}

// Baseline "correct" y2 op (Test 3): over cols [1536,2048) of a [M,stride] bf16 buffer,
// weighted 512-wide RMSNorm then interleaved RoPE on the last 64 dims. In-place; one
// block per row. Represents the y2 work op B folds into its CUDA-core tail, so the
// cuBLAS baseline computes the SAME outputs (fair apples-to-apples).
static __global__ void y2_normrope_bf16_kernel(bf16_t* __restrict__ d, const float* __restrict__ rms_w2,
                                               const float2* __restrict__ rope_cs, float eps, int stride) {
    // Fast y2 op (mirrors rmsnorm_fast_kernel): each thread reads its 8 features ONCE via
    // uint4 (16 B) into registers, block-reduces the sum-of-squares, then weighted-
    // normalizes + (for the last-64 rope dims) rotates + uint4-stores FROM REGISTERS -- no
    // second global read (the old version read y2 three times, scalar). 64 threads cover
    // the 512-wide y2 (8 cols each). RoPE pairs (2j,2j+1) are adjacent and land inside one
    // thread's uint4, so the rotation is register-local.
    constexpr int Y2_LO = 1536, Y2_W = 512, ROPE_LO = 1984, ROPE_PAIRS = 32;
    const bool   do_rope = (rope_cs != nullptr);
    const int    m       = blockIdx.x;
    const size_t base    = (size_t)m * stride;
    const int    cl      = threadIdx.x << 3;              // y2-local col (0..504 step 8)
    const int    cg      = Y2_LO + cl;                    // global col
    const bool   active  = (cl < Y2_W);
    float s[8];
    if (active) {
        uint4 raw = *reinterpret_cast<const uint4*>(d + base + cg);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
        float2 f0 = __bfloat1622float2(h[0]), f1 = __bfloat1622float2(h[1]);
        float2 f2 = __bfloat1622float2(h[2]), f3 = __bfloat1622float2(h[3]);
        s[0]=f0.x; s[1]=f0.y; s[2]=f1.x; s[3]=f1.y;
        s[4]=f2.x; s[5]=f2.y; s[6]=f3.x; s[7]=f3.y;
    } else {
        #pragma unroll
        for (int e = 0; e < 8; ++e) s[e] = 0.f;
    }
    float ss = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) ss += s[e] * s[e];
    __shared__ float red[32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    float tt = (threadIdx.x < nwarps) ? red[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) tt += __shfl_down_sync(0xffffffffu, tt, o);
        if (lane == 0) red[0] = tt;
    }
    __syncthreads();
    const float rms = rsqrtf(red[0] / (float)Y2_W + eps);
    if (!active) return;
    // weighted normalize all 8 features (rope block included -> normalize BEFORE rotate)
    #pragma unroll
    for (int e = 0; e < 8; ++e) s[e] = s[e] * rms * rms_w2[cl + e];
    // RoPE on cols >= ROPE_LO (last 64 dims): the 4 pairs of this uint4 rotate in place.
    if (do_rope && cg >= ROPE_LO) {
        #pragma unroll
        for (int pp = 0; pp < 4; ++pp) {
            const int    j  = (cg - ROPE_LO) / 2 + pp;
            const float2 cs = rope_cs[(size_t)m * ROPE_PAIRS + j];
            const float  ev = s[2*pp], ov = s[2*pp+1];
            s[2*pp]   = ev * cs.x - ov * cs.y;
            s[2*pp+1] = ev * cs.y + ov * cs.x;
        }
    }
    __nv_bfloat162 o2[4];
    o2[0] = __floats2bfloat162_rn(s[0], s[1]);
    o2[1] = __floats2bfloat162_rn(s[2], s[3]);
    o2[2] = __floats2bfloat162_rn(s[4], s[5]);
    o2[3] = __floats2bfloat162_rn(s[6], s[7]);
    *reinterpret_cast<uint4*>(d + base + cg) = *reinterpret_cast<uint4*>(o2);
}

// Fills a device [M,32] float2 cos/sin table with random per-(row,pair) angles. Tests
// the RoPE rotation math without needing the real YaRN freqs; kernel & golden share it.
static void init_rope_cs(float2* d_cs, int M, int seed) {
    float2* h = (float2*)malloc((size_t)M * 32 * sizeof(float2));
    srand(seed);
    for (int i = 0; i < M * 32; i++) {
        float th = (float)(rand() % 62832) / 10000.0f;   // angle in [0, ~2pi)
        h[i].x = cosf(th); h[i].y = sinf(th);
    }
    cudaMemcpy(d_cs, h, (size_t)M * 32 * sizeof(float2), cudaMemcpyHostToDevice);
    free(h);
}

// ============================================================
//  cuBLASLt reference GEMM (fair baseline; A=(M,K)/B=(N,K)/D=(M,N) row-major).
// ============================================================
#define CUBLAS_LT_WORKSPACE_SIZE (32 * 1024 * 1024)  // 32 MB

struct CublasCtx {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t layout_a, layout_b, layout_d;
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t heuristic;
    void* workspace;
    int M, N, K;
    bool valid;
};

void cublas_setup(CublasCtx& ctx, int M, int N, int K, cudaDataType d_type = CUDA_R_16BF, int lda = -1) {
    ctx.M = M; ctx.N = N; ctx.K = K;
    ctx.valid = true;
    cublasLtCreate(&ctx.handle);
    cublasLtMatmulDescCreate(&ctx.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t trans_a = CUBLAS_OP_T, trans_b = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(ctx.desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(ctx.desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cublasLtMatmulDescSetAttribute(ctx.desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &num_sms, sizeof(num_sms));
    cublasLtMatrixLayoutCreate(&ctx.layout_a, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&ctx.layout_b, CUDA_R_16BF, K, M, (lda < 0 ? K : lda));   // A(activation) leading dim; lda>K => pitched read (lets GEMM skip a slice copy)
    cublasLtMatrixLayoutCreate(&ctx.layout_d, d_type, N, M, N);   // bf16 (perf) or fp32 (golden)
    cudaMalloc(&ctx.workspace, CUBLAS_LT_WORKSPACE_SIZE);
    cublasLtMatmulPreferenceCreate(&ctx.pref);
    size_t ws_bytes = CUBLAS_LT_WORKSPACE_SIZE;
    cublasLtMatmulPreferenceSetAttribute(ctx.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws_bytes, sizeof(ws_bytes));
    uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    cublasLtMatmulPreferenceSetAttribute(ctx.pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                          &reduction_mask, sizeof(reduction_mask));
    int num_results = 0;
    cublasLtMatmulAlgoGetHeuristic(ctx.handle, ctx.desc,
                                    ctx.layout_a, ctx.layout_b, ctx.layout_d, ctx.layout_d,
                                    ctx.pref, 1, &ctx.heuristic, &num_results);
    if (num_results == 0) {
        printf("WARNING: cuBLASLt found no algorithm for M=%d N=%d K=%d\n", M, N, K);
        ctx.valid = false;
    }
}

// ===== fp8 VEC32 MXFP8 cuBLASLt baseline for GEMM2 (activation quant + block-scale fp8 GEMM) =====
namespace fp8b {
static constexpr int kVec32TileRows = 128;
__device__ __forceinline__ int vec32_byte_offset(int outer, int inner, int inner_groups){
    int outer_tile    = outer / kVec32TileRows;
    int outer_in_tile = outer % kVec32TileRows;
    int inner_tile    = inner / 4;
    int inner_in_tile = inner % 4;
    int tile = outer_tile * inner_groups + inner_tile;
    return tile * 512 + (outer_in_tile & 31) * 16 + (outer_in_tile >> 5) * 4 + inner_in_tile;
}
// quantize row-major [rows,K] bf16 (leading dim in_stride) -> packed fp8 e4m3 [rows,K]
// + VEC32 UE8M0 scale (one exponent byte per 32 contiguous K elements).
__global__ void quant_vec32_kernel(const bf16_t* __restrict__ in, int rows, int K, int in_stride,
                                   __nv_fp8_e4m3* __restrict__ out, uint8_t* __restrict__ scale){
    int nblk = K >> 5;                                        // K / 32 (one UE8M0 per 32 elems)
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // one warp handles one 32-block
    int lane = threadIdx.x & 31;
    if (warp >= rows * nblk) return;
    int row = warp / nblk;
    int blk = warp % nblk;
    float v = __bfloat162float(in[(size_t)row * in_stride + blk * 32 + lane]);   // coalesced
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int E = -127;
    if (a > 0.f){ E = (int)ceilf(log2f(a / 448.0f)); E = E < -127 ? -127 : (E > 127 ? 127 : E); }
    float inv = exp2f(-(float)E);
    out[(size_t)row * K + blk * 32 + lane] = (__nv_fp8_e4m3)(v * inv);           // coalesced
    if (lane == 0) scale[vec32_byte_offset(row, blk, nblk >> 2)] = (uint8_t)(E + 127);
}
struct Fp8Plan {
    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    void*  ws = nullptr;
    size_t wsBytes = 32u * 1024u * 1024u;
    bool   valid = false;
};
static void fp8_plan_create(Fp8Plan& p, cublasLtHandle_t h, int M, int N, int K, const void* dummy){
    cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opa = CUBLAS_OP_N, opb = CUBLAS_OP_T;
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    int8_t fa = 0;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dummy, sizeof(dummy));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dummy, sizeof(dummy));
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8F_E4M3, M, K, K);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8F_E4M3, N, K, K);
    cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF,    M, N, N);
    cublasLtMatrixLayoutCreate(&p.d, CUDA_R_16BF,    M, N, N);
    cublasLtOrder_t ro = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(p.a, CUBLASLT_MATRIX_LAYOUT_ORDER, &ro, sizeof(ro));
    cublasLtMatrixLayoutSetAttribute(p.b, CUBLASLT_MATRIX_LAYOUT_ORDER, &ro, sizeof(ro));
    cublasLtMatrixLayoutSetAttribute(p.c, CUBLASLT_MATRIX_LAYOUT_ORDER, &ro, sizeof(ro));
    cublasLtMatrixLayoutSetAttribute(p.d, CUBLASLT_MATRIX_LAYOUT_ORDER, &ro, sizeof(ro));
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t wsb = p.wsBytes;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
    cublasLtMatmulHeuristicResult_t hr{}; int ret = 0;
    cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, p.b, p.c, p.d, pref, 1, &hr, &ret);
    cublasLtMatmulPreferenceDestroy(pref);
    if (ret > 0){ p.algo = hr.algo; p.valid = true; cudaMalloc(&p.ws, p.wsBytes); }
    else printf("WARNING: fp8 cuBLASLt found no block-scaled algo for M=%d N=%d K=%d\n", M, N, K);
}
static void fp8_gemm_run(Fp8Plan& p, cublasLtHandle_t h, const void* A_act, const void* B_w,
                         void* D, cudaStream_t s){
    if (!p.valid) return;
    float alpha = 1.f, beta = 0.f;
    const void* dummy_c = A_act;   // beta=0 => C never read (matches PyTorch scaled_gemm)
    cublasLtMatmul(h, p.op, &alpha, A_act, p.a, B_w, p.b, &beta, dummy_c, p.c, D, p.d,
                   &p.algo, p.ws, p.wsBytes, s);
}
// Autotune the block-scaled fp8 plan: heuristic[0] is often NOT the fastest algo for
// skinny-M decode shapes (M<=128), so query up to 16 candidates and time each on the
// REAL buffers, keeping the winner. This makes the baseline as fast as cuBLASLt can
// go -> a fair speedup denominator. Buffer CONTENTS are irrelevant for GEMM speed
// (caller just memsets them so the candidates run on defined data).
static void fp8_autotune(Fp8Plan& p, cublasLtHandle_t h, const void* A_act, const void* B_w, void* D){
    if (!p.valid) return;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t wsb = p.wsBytes;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
    cublasLtMatmulHeuristicResult_t cand[16]; int ncand = 0;
    cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, p.b, p.c, p.d, pref, 16, cand, &ncand);
    cublasLtMatmulPreferenceDestroy(pref);
    if (ncand <= 1) return;
    float alpha = 1.f, beta = 0.f;
    cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);
    float best = 1e30f; int besti = -1;
    for (int i = 0; i < ncand; i++) {
        auto run1 = [&]{ return cublasLtMatmul(h, p.op, &alpha, A_act, p.a, B_w, p.b, &beta,
                                               A_act, p.c, D, p.d, &cand[i].algo, p.ws, p.wsBytes, 0); };
        if (run1() != CUBLAS_STATUS_SUCCESS) continue;   // candidate rejected at run time
        run1();                                          // warm
        cudaDeviceSynchronize();
        cudaEventRecord(ea);
        for (int r = 0; r < 5; r++) run1();
        cudaEventRecord(eb); cudaEventSynchronize(eb);
        float e; cudaEventElapsedTime(&e, ea, eb);
        if (e < best) { best = e; besti = i; }
    }
    cudaEventDestroy(ea); cudaEventDestroy(eb);
    if (besti >= 0) p.algo = cand[besti].algo;
}
static void fp8_plan_destroy(Fp8Plan& p){
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    if (p.a)  cublasLtMatrixLayoutDestroy(p.a);
    if (p.b)  cublasLtMatrixLayoutDestroy(p.b);
    if (p.c)  cublasLtMatrixLayoutDestroy(p.c);
    if (p.d)  cublasLtMatrixLayoutDestroy(p.d);
    if (p.ws) cudaFree(p.ws);
    p = Fp8Plan{};
}
} // namespace fp8b

void cublas_run(CublasCtx& ctx, const bf16_t* A, const bf16_t* B, void* C, cudaStream_t stream = 0) {
    if (!ctx.valid) return;
    const float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(ctx.handle, ctx.desc, &alpha,
                   B, ctx.layout_a, A, ctx.layout_b, &beta,
                   C, ctx.layout_d, C, ctx.layout_d,
                   &ctx.heuristic.algo, ctx.workspace, CUBLAS_LT_WORKSPACE_SIZE, stream);
}

// Autotune a bf16 CublasCtx: like fp8b::fp8_autotune, time up to 16 heuristic
// candidates on the real buffers and keep the fastest (heuristic[0] is often
// suboptimal for skinny-M decode shapes). Call AFTER cublas_setup, with the same
// buffers the timed runs will use; only used for PERF-baseline contexts (the
// correctness goldens don't need it -- any algo gives the same fp32 result).
static void cublas_autotune(CublasCtx& ctx, const bf16_t* A, const bf16_t* B, void* C) {
    if (!ctx.valid) return;
    cublasLtMatmulHeuristicResult_t cand[16]; int ncand = 0;
    cublasLtMatmulAlgoGetHeuristic(ctx.handle, ctx.desc, ctx.layout_a, ctx.layout_b,
                                   ctx.layout_d, ctx.layout_d, ctx.pref, 16, cand, &ncand);
    if (ncand <= 1) return;
    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);
    float best = 1e30f; int besti = -1;
    for (int i = 0; i < ncand; i++) {
        auto run1 = [&]{ return cublasLtMatmul(ctx.handle, ctx.desc, &alpha,
                                               B, ctx.layout_a, A, ctx.layout_b, &beta,
                                               C, ctx.layout_d, C, ctx.layout_d,
                                               &cand[i].algo, ctx.workspace, CUBLAS_LT_WORKSPACE_SIZE, 0); };
        if (run1() != CUBLAS_STATUS_SUCCESS) continue;
        run1();
        cudaDeviceSynchronize();
        cudaEventRecord(ea);
        for (int r = 0; r < 5; r++) run1();
        cudaEventRecord(eb); cudaEventSynchronize(eb);
        float e; cudaEventElapsedTime(&e, ea, eb);
        if (e < best) { best = e; besti = i; }
    }
    cudaEventDestroy(ea); cudaEventDestroy(eb);
    if (besti >= 0) ctx.heuristic = cand[besti];
}

void cublas_free(CublasCtx& ctx) {
    cublasLtMatmulPreferenceDestroy(ctx.pref);
    cublasLtMatrixLayoutDestroy(ctx.layout_a);
    cublasLtMatrixLayoutDestroy(ctx.layout_b);
    cublasLtMatrixLayoutDestroy(ctx.layout_d);
    cublasLtMatmulDescDestroy(ctx.desc);
    cublasLtDestroy(ctx.handle);
    cudaFree(ctx.workspace);
}

// ============================================================
//  Correctness metrics (host-side; reference = cuBLAS GEMM + CUDA rmsnorm)
// ============================================================
struct CorrMetrics { double cos_sim; double rel_l2; float max_abs_err; bool pass; };

CorrMetrics compute_metrics(const bf16_t* d_ours, const bf16_t* d_ref, int M, int cmp_cols, int stride) {
    // Copies full rows (stride wide) but compares ONLY the first cmp_cols per row,
    // since the fused op writes just y1 = [0, cmp_cols); [cmp_cols, stride) is unwritten.
    size_t elems = (size_t)M * stride;
    float* h_ours = (float*)malloc(elems * sizeof(float));
    float* h_ref  = (float*)malloc(elems * sizeof(float));
    bf16_t* h_tmp = (bf16_t*)malloc(elems * sizeof(bf16_t));
    cudaMemcpy(h_tmp, d_ours, elems * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < elems; i++) h_ours[i] = __bfloat162float(h_tmp[i]);
    cudaMemcpy(h_tmp, d_ref, elems * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < elems; i++) h_ref[i] = __bfloat162float(h_tmp[i]);
    free(h_tmp);

    double dot = 0, norm_a = 0, norm_b = 0, diff_sq = 0;
    float max_ae = 0.0f;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < cmp_cols; n++) {
            size_t i = (size_t)m * stride + n;
            float a = h_ours[i], b = h_ref[i];
            if (isnan(a) || isnan(b)) { free(h_ours); free(h_ref); return {0.0, 1e9, 1e9f, false}; }
            dot += (double)a * b; norm_a += (double)a * a; norm_b += (double)b * b;
            float ae = fabsf(a - b); if (ae > max_ae) max_ae = ae;
            diff_sq += (double)(a - b) * (a - b);
        }
    }
    double cos_s = (norm_a > 0 && norm_b > 0) ? dot / (sqrt(norm_a) * sqrt(norm_b)) : 0.0;
    double rel = (norm_b > 0) ? sqrt(diff_sq) / sqrt(norm_b) : 1e9;
    free(h_ours); free(h_ref);
    bool pass = (cos_s > 0.9999) && (rel < 0.01) && (max_ae < 1.0f);
    return {cos_s, rel, max_ae, pass};
}

// Same as compute_metrics but compares a COLUMN RANGE [col_lo, col_hi) per row (for
// op B's CUDA-core tail, which writes cols [1536,4352) of op A's D buffer). The pass
// criterion drops the absolute max-error bound: y3/y4 are UN-normalized GEMM outputs
// (magnitude ~ sqrt(K)), so a bf16 round there is naturally ~O(0.1..1). cos + rel_l2
// are scale-invariant and are the reliable signals; max_ae is reported for info.
CorrMetrics compute_metrics_range(const bf16_t* d_ours, const bf16_t* d_ref,
                                  int M, int col_lo, int col_hi, int stride) {
    size_t elems = (size_t)M * stride;
    float* h_ours = (float*)malloc(elems * sizeof(float));
    float* h_ref  = (float*)malloc(elems * sizeof(float));
    bf16_t* h_tmp = (bf16_t*)malloc(elems * sizeof(bf16_t));
    cudaMemcpy(h_tmp, d_ours, elems * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < elems; i++) h_ours[i] = __bfloat162float(h_tmp[i]);
    cudaMemcpy(h_tmp, d_ref, elems * sizeof(bf16_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < elems; i++) h_ref[i] = __bfloat162float(h_tmp[i]);
    free(h_tmp);
    double dot = 0, na = 0, nb = 0, dsq = 0; float mae = 0.f;
    for (int m = 0; m < M; m++)
        for (int n = col_lo; n < col_hi; n++) {
            size_t i = (size_t)m * stride + n;
            float a = h_ours[i], b = h_ref[i];
            if (isnan(a) || isnan(b)) { free(h_ours); free(h_ref); return {0.0, 1e9, 1e9f, false}; }
            dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            float ae = fabsf(a - b); if (ae > mae) mae = ae; dsq += (double)(a - b) * (a - b);
        }
    double cs = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    double rel = (nb > 0) ? sqrt(dsq) / sqrt(nb) : 1e9;
    free(h_ours); free(h_ref);
    bool pass = (cs > 0.9999) && (rel < 0.01);
    return {cs, rel, mae, pass};
}

// ============================================================
//  Main benchmark
// ============================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("complex_gemm STEP 1: FUSED 1-SM swap-AB bf16 GEMM + per-token RMSNorm (eps=1e-6)\n");
    printf("  Shape: M=1..64, N=4608, K=7168\n");
    printf("  Correctness golden: cuBLASLt fp32 GEMM -> fp32 RMSNorm -> bf16 (full-precision)\n");
    printf("  Perf baseline: realistic bf16 pipeline (cuBLAS bf16 GEMM + fast CUDA RMSNorm)\n\n");

    const float eps = 1e-6f;

    struct TestCase { int M, N, K; };
    TestCase cases[] = {
        {1,  4608, 7168}, {2,  4608, 7168}, {4,  4608, 7168}, {7,  4608, 7168},
        {8,  4608, 7168}, {16, 4608, 7168}, {17, 4608, 7168}, {31, 4608, 7168},
        {32, 4608, 7168}, {48, 4608, 7168}, {64, 4608, 7168},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    int warmup = 100, repeats = 50;

    // ---- Test 1: Correctness ----
    printf("==================== Test 1: Correctness (fused GEMM + RMSNorm) ====================\n");
    printf("+-------+-------+-------+-----+-----+----+------+----------+----------+---------+------+\n");
    printf("|     M |     N |     K |  BM |  BN | ks | blk  |  cos_sim |   rel_l2 | max_abs | Pass |\n");
    printf("+-------+-------+-------+-----+-----+----+------+----------+----------+---------+------+\n");

    bool all_pass = true;
    for (int c = 0; c < num_cases; c++) {
        int M = cases[c].M, N = cases[c].N, K = cases[c].K;
        const int nd = (N < FUSENORM_NORM_DIM) ? N : FUSENORM_NORM_DIM;  // y1 norm width (compared range)

        bf16_t *dA, *dB, *dC_ours, *dC_ref;
        float  *dC_f32, *d_rms_w;
        cudaMalloc(&dA, (size_t)M * K * sizeof(bf16_t));
        cudaMalloc(&dB, (size_t)N * K * sizeof(bf16_t));
        cudaMalloc(&dC_ours, (size_t)M * N * sizeof(bf16_t));
        cudaMalloc(&dC_f32,  (size_t)M * N * sizeof(float));    // cuBLAS fp32 GEMM (golden input)
        cudaMalloc(&dC_ref,  (size_t)M * N * sizeof(bf16_t));   // golden after fp32 rmsnorm
        cudaMalloc(&d_rms_w, (size_t)N * sizeof(float));
        cudaMemset(dC_ours, 0, (size_t)M * N * sizeof(bf16_t));

        init_test_data(dA, dB, M, N, K, 42 + c);
        init_rms_w(d_rms_w, N, 1234 + c);

        FuseNormCtx sctx;
        fusenorm_setup(sctx, dA, dB, M, N, K);
        CublasCtx cubctx;
        cublas_setup(cubctx, M, N, K, CUDA_R_32F);   // fp32 output => fair (full-precision) golden

        // ours: fused GEMM + RMSNorm in one pipeline (normalizes fp32 accumulators).
        fusenorm_run(sctx, dC_ours, d_rms_w, eps);
        // golden: cuBLAS fp32 GEMM -> fp32 RMSNorm -> bf16. Both sides normalize
        // FULL-precision GEMM, so the only residual gap is the single final bf16
        // round (no intermediate bf16 round-trip like the perf baseline below).
        cublas_run(cubctx, dA, dB, dC_f32);
        rmsnorm_ref_run<float>(dC_f32, dC_ref, d_rms_w, eps, M, nd, N);
        cudaDeviceSynchronize();

        if (!sctx.valid || !cubctx.valid) {
            printf("| %5d | %5d | %5d | %3d | %3d | %2d | %4d |      N/A |      N/A |     N/A | SKIP |\n",
                   M, N, K, sctx.block_m, sctx.block_n, sctx.ks, sctx.num_blocks);
        } else {
            CorrMetrics m = compute_metrics(dC_ours, dC_ref, M, nd, N);
            if (!m.pass) all_pass = false;
            printf("| %5d | %5d | %5d | %3d | %3d | %2d | %4d | %8.6f | %8.6f | %7.4f | %s |\n",
                   M, N, K, sctx.block_m, sctx.block_n, sctx.ks, sctx.num_blocks,
                   m.cos_sim, m.rel_l2, m.max_abs_err, m.pass ? "PASS" : "FAIL");
        }

        fusenorm_free(sctx);
        cublas_free(cubctx);
        cudaFree(dA); cudaFree(dB); cudaFree(dC_ours); cudaFree(dC_f32);
        cudaFree(dC_ref); cudaFree(d_rms_w);
    }
    printf("+-------+-------+-------+-----+-----+----+------+----------+----------+---------+------+\n");
    printf("Overall: %s\n\n", all_pass ? "ALL PASS" : "SOME FAILED");

    // ---- Global GPU warm-up: drive clocks to sustained boost BEFORE any timing ----
    {
        int wM = 256, wN = 1536, wK = 7168;
        bf16_t *wA, *wB, *wC;
        cudaMalloc(&wA, (size_t)wM * wK * sizeof(bf16_t));
        cudaMalloc(&wB, (size_t)wN * wK * sizeof(bf16_t));
        cudaMalloc(&wC, (size_t)wM * wN * sizeof(bf16_t));
        init_test_data(wA, wB, wM, wN, wK, 7);
        CublasCtx wctx; cublas_setup(wctx, wM, wN, wK);
        if (wctx.valid) { for (int i = 0; i < 5000; i++) cublas_run(wctx, wA, wB, wC); cudaDeviceSynchronize(); }
        cublas_free(wctx);
        cudaFree(wA); cudaFree(wB); cudaFree(wC);
    }

    // ================= Test 2: Performance (N=4608, K=7168) =================
    // complex_gemm step-1 shape: fixed N=4608; sweep M over 1..64 (dense primes +
    // powers of two + boundaries) profiling the fused op vs the realistic bf16
    // pipeline (cuBLAS bf16 GEMM + fast CUDA RMSNorm). Runs AFTER the global warm-up.
    printf("============ Test 2: Performance (N=4608, K=7168): fused vs cuBLAS-GEMM + fast-RMSNorm ============\n");
    printf("+-------+-------+-----+-----+----+-------+-----------+----------+-------------+--------+\n");
    printf("|     M |     N |  BM |  BN | ks | blk   | fused(us) | BW(TB/s) | cuB+rms(us) | Ratio  |\n");
    printf("+-------+-------+-----+-----+----+-------+-----------+----------+-------------+--------+\n");

    const int sweepK = 7168;
    const int sweepMs[] = {
        1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 32,
        37, 41, 43, 47, 53, 59, 61, 64
    };
    const int nM = (int)(sizeof(sweepMs) / sizeof(sweepMs[0]));

    for (int p = 0; p < nM; p++) {
        int M = sweepMs[p], N = 4608, K = sweepK;
        const int nd = (N < FUSENORM_NORM_DIM) ? N : FUSENORM_NORM_DIM;  // y1 norm width

        bf16_t *dA, *dB, *dC, *dC_gemm, *dC_ref;
        float  *d_rms_w;
        cudaMalloc(&dA, (size_t)M * K * sizeof(bf16_t));
        cudaMalloc(&dB, (size_t)N * K * sizeof(bf16_t));
        cudaMalloc(&dC,      (size_t)M * N * sizeof(bf16_t));
        cudaMalloc(&dC_gemm, (size_t)M * N * sizeof(bf16_t));
        cudaMalloc(&dC_ref,  (size_t)M * N * sizeof(bf16_t));
        cudaMalloc(&d_rms_w, (size_t)N * sizeof(float));
        init_test_data(dA, dB, M, N, K, 42 + p);
        init_rms_w(d_rms_w, N, 1234 + p);

        CublasCtx cubctx; cublas_setup(cubctx, M, N, K);
        cublas_autotune(cubctx, dA, dB, dC_gemm);   // pick cuBLASLt's actual-fastest algo (fair baseline)

        // baseline timing: cuBLAS GEMM + standalone fast CUDA RMSNorm (sequential).
        float base_us = 0.0f;
        if (cubctx.valid) {
            cublas_run(cubctx, dA, dB, dC_gemm);
            rmsnorm_fast_run(dC_gemm, dC_ref, d_rms_w, eps, M, nd, N);
            cudaDeviceSynchronize();
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            for (int i = 0; i < warmup; i++) {
                cublas_run(cubctx, dA, dB, dC_gemm);
                rmsnorm_fast_run(dC_gemm, dC_ref, d_rms_w, eps, M, nd, N);
            }
            cudaDeviceSynchronize();
            cudaEventRecord(t0);
            for (int i = 0; i < repeats; i++) {
                cublas_run(cubctx, dA, dB, dC_gemm);
                rmsnorm_fast_run(dC_gemm, dC_ref, d_rms_w, eps, M, nd, N);
            }
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float e; cudaEventElapsedTime(&e, t0, t1);
            base_us = e / repeats * 1000.0f;
            cudaEventDestroy(t0); cudaEventDestroy(t1);
        }

        // fused operator timing.
        FuseNormCtx sctx; fusenorm_setup(sctx, dA, dB, M, N, K);
        if (!sctx.valid || !cubctx.valid) {
            printf("| %5d | %5d | %3d | %3d | %2d | %5d |       N/A |      N/A | %11.2f |    N/A |\n",
                   M, N, sctx.block_m, sctx.block_n, sctx.ks, sctx.num_blocks, base_us);
            fusenorm_free(sctx); cublas_free(cubctx);
            cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_gemm);
            cudaFree(dC_ref); cudaFree(d_rms_w);
            continue;
        }

        fusenorm_run(sctx, dC, d_rms_w, eps);
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int i = 0; i < warmup; i++) fusenorm_run(sctx, dC, d_rms_w, eps);
        cudaDeviceSynchronize();
        cudaEventRecord(t0);
        for (int i = 0; i < repeats; i++) fusenorm_run(sctx, dC, d_rms_w, eps);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float e; cudaEventElapsedTime(&e, t0, t1);
        float us = e / repeats * 1000.0f;
        double ratio = (base_us > 0 && us > 0) ? base_us / us * 100.0 : 0.0;
        // Effective DRAM traffic of the FUSED op -- ESSENTIAL I/O only: read A(M,K)
        // + B(N,K) bf16, read rms_w(y1=nd) fp32, write D(M,nd) bf16 (only y1 is
        // written). The ks fp32 split-K ws round-trip is an IMPLEMENTATION artifact
        // (not useful traffic), so it is NOT counted -- BW measures how close we get
        // to the essential-bytes floor.
        double bytes = (double)M * K * 2.0 + (double)N * K * 2.0
                     + (double)M * nd * 2.0 + (double)nd * 4.0;
        double bw = (us > 0) ? bytes / us / 1.0e6 : 0.0;   // bytes/(us*1e-6)/1e12 = bytes/us/1e6

        printf("| %5d | %5d | %3d | %3d | %2d | %5d | %9.2f | %8.2f | %11.2f | %5.1f%% |\n",
               M, N, sctx.block_m, sctx.block_n, sctx.ks, sctx.num_blocks, us, bw, base_us, ratio);

        fusenorm_free(sctx); cublas_free(cubctx);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_gemm);
        cudaFree(dC_ref); cudaFree(d_rms_w);
    }
    printf("+-------+-------+-----+-----+----+-------+-----------+----------+-------------+--------+\n");
    printf("(Ratio > 100%% => fused is faster than cuBLAS-GEMM + separate RMSNorm)\n");
    printf("(BW = fused-op ESSENTIAL DRAM traffic [A+B+rms_w read + D write] / time; excludes split-K ws)\n\n");

    // ============================================================
    //  Test 3: PIPELINE PERFORMANCE  (op A + op B  vs  a CORRECT cuBLAS-bf16 pipeline)
    //    fused    : fusenorm_run (op A, N=4608) -> gfnb::wq_b_proj_run (op B, pitched read; the
    //               op B call folds op A's y2 RMSNorm+RoPE AND the full y3/y4 compressor tail).
    //    baseline : cuBLAS bf16 GEMM1[M,4608] -> rmsnorm_fast(y1) -> y2 RMSNorm&RoPE ->
    //               slice y1[:, :1536] -> cuBLAS bf16 GEMM2[M,65536] -> head-norm. NOTE: the
    //               baseline does NOT do the y3/y4 compressor post-proc, so speedup is conservative.
    //    speedup  : base_sum / AB_bb, where base_sum = SUM of each baseline op's OWN
    //               back-to-back exec (per-op launch overhead removed -> fair vs a 2-launch
    //               fused pipeline). base_seq (single-shot sequential launch latency) is
    //               reported too. BW is on ESSENTIAL bytes only: x + w1 + w2 + q + rms_w1
    //               (excludes the y1 HBM round-trip and op A's split-K ws).
    // ============================================================
    {
        const int K1 = 7168, N1 = 4608;
        const int KB = gfnb::wq_b::K_DIM;        // 1536
        const int NB = gfnb::wq_b::N_TOTAL;      // 65536
        const int HD = gfnb::wq_b::HEAD_DIM;     // 512
        const float epsP = 1e-6f;
        const int warmup = 10, repeats = 10;
        // op B requires 16-aligned M in [16,256]; all cases below are <=128 so the
        // activation-SF block (SF_BLOCK_M=128) fully covers M (no padded rows).
        const int NM3 = 7;
        int Ms[NM3] = {16, 32, 48, 64, 80, 96, 128};

        printf("============ Test 3: pipeline performance A->B (N1=4608,K1=7168 | N2=65536,K2=1536) ============\n");
        printf("  (fused op A+B vs a FULL logic-equivalent baseline: cuBLAS GEMMs + y1/y2 norms + the complete\n");
        printf("   y3/y4 compressor+indexer decode [Test-4-verified byte-exact kernels] -> a fair, apples-to-apples denominator)\n");

        // ---- M-independent weights ----
        bf16_t *w1=nullptr, *w2=nullptr; float *rms_w1=nullptr, *d_ones=nullptr;
        cudaMalloc(&w1, (size_t)N1*K1*sizeof(bf16_t));
        cudaMalloc(&w2, (size_t)NB*KB*sizeof(bf16_t));
        cudaMalloc(&rms_w1, (size_t)N1*sizeof(float));
        cudaMalloc(&d_ones, (size_t)HD*sizeof(float));
        {   bf16_t* h = (bf16_t*)malloc((size_t)NB*KB*sizeof(bf16_t));
            srand(1234);
            size_t s1=(size_t)N1*K1; for(size_t i=0;i<s1;i++) h[i]=__float2bfloat16((float)(rand()%200-100)/100.0f);
            cudaMemcpy(w1,h,s1*sizeof(bf16_t),cudaMemcpyHostToDevice);
            size_t s2=(size_t)NB*KB; for(size_t i=0;i<s2;i++) h[i]=__float2bfloat16((float)(rand()%200-100)/100.0f);
            cudaMemcpy(w2,h,s2*sizeof(bf16_t),cudaMemcpyHostToDevice);
            free(h); }
        {   float* h=(float*)malloc(HD*sizeof(float)); for(int i=0;i<HD;i++) h[i]=1.0f;
            cudaMemcpy(d_ones,h,HD*sizeof(float),cudaMemcpyHostToDevice); free(h); }
        init_rms_w(rms_w1, N1, 42);

        // ---- fused-compressor (y3/y4) M-INDEPENDENT inputs: ape/norm/rope-table.
        //      Perf test => values only need to be finite/non-degenerate (the softmax
        //      subtracts the row max, so the ~sqrt(K) GEMM magnitudes never overflow). ----
        const int RATIO=cbl::RATIO, RD=cbl::RD, SROWS=cbl::SROWS;
        const int D_M=cbl::D_M, WK_M=cbl::WK_M, D_I=cbl::D_I, WK_I=cbl::WK_I;
        const int NF8=D_M-RD, SEQ=256;   // NF8=448; SEQ >= max(pos)=163
        const int NW    = 2*WK_M + 2*WK_I;   // 2560 = the y3/y4 block width (wkv_m|wgate_m|wkv_i|wgate_i)
        const int Y34_LO = 2048;             // y3/y4 start column inside the [M,4608] GEMM output ([2048,4608))
        float *d_cape=nullptr,*d_iape=nullptr,*d_cnorm=nullptr,*d_inorm=nullptr,*d_cos=nullptr,*d_sin=nullptr;
        cudaMalloc(&d_cape,(size_t)RATIO*WK_M*4); cudaMalloc(&d_iape,(size_t)RATIO*WK_I*4);
        cudaMalloc(&d_cnorm,(size_t)D_M*4);       cudaMalloc(&d_inorm,(size_t)D_I*4);
        cudaMalloc(&d_cos,(size_t)SEQ*(RD/2)*4);  cudaMalloc(&d_sin,(size_t)SEQ*(RD/2)*4);
        cudaMemset(d_cape,0,(size_t)RATIO*WK_M*4); cudaMemset(d_iape,0,(size_t)RATIO*WK_I*4);
        {   float* h=(float*)malloc((size_t)D_M*4); for(int i=0;i<D_M;i++) h[i]=1.0f;
            cudaMemcpy(d_cnorm,h,(size_t)D_M*4,cudaMemcpyHostToDevice);
            cudaMemcpy(d_inorm,h,(size_t)D_I*4,cudaMemcpyHostToDevice); free(h); }
        {   int nel=SEQ*(RD/2); float* h=(float*)malloc((size_t)nel*4);
            for(int i=0;i<nel;i++) h[i]=1.0f; cudaMemcpy(d_cos,h,(size_t)nel*4,cudaMemcpyHostToDevice);
            for(int i=0;i<nel;i++) h[i]=0.0f; cudaMemcpy(d_sin,h,(size_t)nel*4,cudaMemcpyHostToDevice); free(h); }

        // (tables are printed AFTER the sweep: main comparison + per-operator breakdown + our-op timeline)

        // op B internal timeline probe: one instrumented run per M stamps WHEN the
        // tensor-core path (warps 0-7) and the CUDA-core tail (warps 8-15) start/finish.
        // grid <= 148 CTAs; over-allocate 256*4 u64 and ignore unwritten (0) slots.
        unsigned long long* d_prof = nullptr;
        cudaMalloc(&d_prof, (size_t)256 * 16 * sizeof(unsigned long long));
        double r_tc_beg[NM3], r_tc_end[NM3], r_cc_beg[NM3], r_cc_end[NM3];
        // compressor-tail phase stamps for the critical compress-row CTA (relative to t0, us):
        double r_p_beg[NM3], r_p_red[NM3], r_p_agg[NM3], r_p_mnq[NM3], r_p_idx[NM3];
        double r_p_s8[NM3], r_p_s9[NM3], r_p_s10[NM3];
        for (int i = 0; i < NM3; i++) {
            r_tc_beg[i]=r_tc_end[i]=r_cc_beg[i]=r_cc_end[i]=-1;
            r_p_beg[i]=r_p_red[i]=r_p_agg[i]=r_p_mnq[i]=r_p_idx[i]=-1;
            r_p_s8[i]=r_p_s9[i]=r_p_s10[i]=-1;
        }
        // per-op baseline exec times [mi], for the vertical per-operator table below (us).
        // main path: GEMM1 y1norm y2norm+rope GEMM2 headnorm ; compressor: aggregate main-nr idx-nr hadamard fp8 fp4.
        double op_g1[NM3]={0}, op_y1[NM3]={0}, op_y2[NM3]={0}, op_g2[NM3]={0}, op_hn[NM3]={0};
        double op_g2q[NM3]={0}, op_g2m[NM3]={0}, op_bqf[NM3]={0};
        double op_g1q[NM3]={0}, op_g1m[NM3]={0};
        double op_ag[NM3]={0}, op_mq[NM3]={0}, op_iq[NM3]={0};
        double r_ours[NM3]={0}, r_base[NM3]={0}, r_sp[NM3]={0}, r_bw[NM3]={0};
        double r_opA[NM3]={0}, r_opB[NM3]={0}, r_opBnt[NM3]={0};

        for (int mi = 0; mi < NM3; mi++) {
            int M = Ms[mi];
            bf16_t *x=nullptr,*dA_out=nullptr,*dB_out=nullptr,*dG1=nullptr,*dG2=nullptr;
            cudaMalloc(&x,      (size_t)M*K1*sizeof(bf16_t));
            cudaMalloc(&dA_out, (size_t)M*N1*sizeof(bf16_t));
            cudaMalloc(&dB_out, (size_t)M*NB*sizeof(bf16_t));
            cudaMalloc(&dG1,    (size_t)M*N1*sizeof(bf16_t));  // baseline GEMM1 out [M,4608] (bf16 y1|y2|y3|y4, one shot)
            cudaMalloc(&dG2,    (size_t)M*NB*sizeof(bf16_t));   // baseline GEMM2 out [M,65536]
            float2* d_rope_cs=nullptr; cudaMalloc(&d_rope_cs,(size_t)M*32*sizeof(float2)); init_rope_cs(d_rope_cs,M,555+M);
            {   bf16_t* h=(bf16_t*)malloc((size_t)M*K1*sizeof(bf16_t)); srand(7+M);
                for(size_t i=0;i<(size_t)M*K1;i++) h[i]=__float2bfloat16((float)(rand()%200-100)/100.0f);
                cudaMemcpy(x,h,(size_t)M*K1*sizeof(bf16_t),cudaMemcpyHostToDevice); free(h); }

            // ---- fused-compressor per-M buffers: positions (pos%4==3 => 1/4 rows compress),
            //      state (in/out: op B writes the current slot then shifts), quant outputs. ----
            long long* d_pos=nullptr; cudaMalloc(&d_pos,(size_t)M*8);
            {   long long* h=(long long*)malloc((size_t)M*8); for(int i=0;i<M;i++) h[i]=100+i;
                cudaMemcpy(d_pos,h,(size_t)M*8,cudaMemcpyHostToDevice); free(h); }
            float *d_ckv=nullptr,*d_csc=nullptr,*d_ikv=nullptr,*d_isc=nullptr;
            cudaMalloc(&d_ckv,(size_t)M*SROWS*WK_M*4); cudaMalloc(&d_csc,(size_t)M*SROWS*WK_M*4);
            cudaMalloc(&d_ikv,(size_t)M*SROWS*WK_I*4); cudaMalloc(&d_isc,(size_t)M*SROWS*WK_I*4);
            cudaMemset(d_ckv,0,(size_t)M*SROWS*WK_M*4); cudaMemset(d_csc,0,(size_t)M*SROWS*WK_M*4);
            cudaMemset(d_ikv,0,(size_t)M*SROWS*WK_I*4); cudaMemset(d_isc,0,(size_t)M*SROWS*WK_I*4);
            uint8_t *d_q8=nullptr,*d_q4=nullptr,*d_s4=nullptr; float* d_s8=nullptr; bf16_t* d_crope=nullptr;
            cudaMalloc(&d_q8,(size_t)M*NF8); cudaMalloc(&d_s8,(size_t)M*(NF8/64)*4); cudaMalloc(&d_crope,(size_t)M*RD*sizeof(bf16_t));
            cudaMalloc(&d_q4,(size_t)M*(D_I/2)); cudaMalloc(&d_s4,(size_t)M*(D_I/32));
            // ---- baseline compressor scratch: aggregate outputs (fp32) + normed/roped (bf16). y3/y4 are
            //      read straight from dG1[:, 2048:4608] (bf16) -- no separate compressor GEMM/fp32 buffer. ----
            float  *d_cmain=nullptr, *d_cidx=nullptr; bf16_t *d_cmbf=nullptr, *d_cibf=nullptr;
            cudaMalloc(&d_cmain,(size_t)M*D_M*4);           // [M,512] main aggregate (fp32)
            cudaMalloc(&d_cidx, (size_t)M*D_I*4);           // [M,128] idx aggregate (fp32)
            cudaMalloc(&d_cmbf, (size_t)M*D_M*sizeof(bf16_t)); // [M,512] main normed+roped (bf16)
            cudaMalloc(&d_cibf, (size_t)M*D_I*sizeof(bf16_t)); // [M,128] idx normed+roped (bf16)

            FuseNormCtx actx; fusenorm_setup(actx, x, w1, M, N1, K1);
            CublasCtx cub2; cublas_setup(cub2, M, NB, KB, CUDA_R_16BF, N1);   // GEMM2 pitched-reads y1 from dG1[:, :1536] (lda=N1); CORRECTNESS golden only (not timed)
            // ---- fp8 VEC32 MXFP8 cuBLASLt baseline for GEMM2 (activation quant + block-scale fp8 GEMM) ----
            static cublasLtHandle_t s_ltB2 = nullptr;
            static __nv_fp8_e4m3* s_w2_fp8 = nullptr; static uint8_t* s_w2_sf = nullptr; static bf16_t* s_w2_src = nullptr;
            if (!s_ltB2) cublasLtCreate(&s_ltB2);
            if (!s_w2_fp8){ cudaMalloc(&s_w2_fp8,(size_t)NB*KB); cudaMalloc(&s_w2_sf,(size_t)NB*(KB/32)); }
            if (s_w2_src != w2){ int wtot=NB*(KB/32); fp8b::quant_vec32_kernel<<<(wtot*32+255)/256,256>>>(w2, NB, KB, KB, s_w2_fp8, s_w2_sf); s_w2_src=w2; }
            __nv_fp8_e4m3* d_y1_fp8=nullptr; uint8_t* d_y1_sf=nullptr;
            cudaMalloc(&d_y1_fp8,(size_t)128*KB); cudaMalloc(&d_y1_sf,(size_t)128*(KB/32));
            cudaMemset(d_y1_fp8, 0, (size_t)128*KB); cudaMemset(d_y1_sf, 0, (size_t)128*(KB/32));
            fp8b::Fp8Plan planB2; fp8b::fp8_plan_create(planB2, s_ltB2, M, NB, KB, s_w2_sf);
            cublasLtMatmulDescSetAttribute(planB2.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_y1_sf, sizeof(void*));
            cublasLtMatmulDescSetAttribute(planB2.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &s_w2_sf, sizeof(void*));
            static __nv_fp8_e4m3* s_qx_fp8=nullptr; static uint8_t* s_qx_sf=nullptr;
            if (!s_qx_fp8){ cudaMalloc(&s_qx_fp8,(size_t)128*KB); cudaMalloc(&s_qx_sf,(size_t)128*gfnb::wq_b::NUM_K_TILES); }
            // ---- fp8 VEC32 MXFP8 cuBLASLt baseline for GEMM1 (activation quant x + block-scale fp8 GEMM, K=7168) ----
            static __nv_fp8_e4m3* s_w1_fp8=nullptr; static uint8_t* s_w1_sf=nullptr; static bf16_t* s_w1_src=nullptr;
            if (!s_w1_fp8){ cudaMalloc(&s_w1_fp8,(size_t)N1*K1); cudaMalloc(&s_w1_sf,(size_t)N1*(K1/32)); }
            if (s_w1_src != w1){ int w1tot=N1*(K1/32); fp8b::quant_vec32_kernel<<<(w1tot*32+255)/256,256>>>(w1, N1, K1, K1, s_w1_fp8, s_w1_sf); s_w1_src=w1; }
            __nv_fp8_e4m3* d_x_fp8=nullptr; uint8_t* d_x_sf=nullptr;
            cudaMalloc(&d_x_fp8,(size_t)128*K1); cudaMalloc(&d_x_sf,(size_t)128*(K1/32));
            cudaMemset(d_x_fp8, 0, (size_t)128*K1); cudaMemset(d_x_sf,0,(size_t)128*(K1/32));
            fp8b::Fp8Plan planB1; fp8b::fp8_plan_create(planB1, s_ltB2, M, N1, K1, s_w1_sf);
            cublasLtMatmulDescSetAttribute(planB1.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_x_sf, sizeof(void*));
            cublasLtMatmulDescSetAttribute(planB1.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &s_w1_sf, sizeof(void*));
            // Autotune BOTH fp8 baseline GEMMs on the real buffers (zero-filled: contents don't
            // affect speed). Without this the baseline runs heuristic[0], which for these skinny-M
            // shapes can be well below cuBLASLt's best -> inflated speedup. Runs once per M case.
            fp8b::fp8_autotune(planB1, s_ltB2, d_x_fp8,  s_w1_fp8, dG1);
            fp8b::fp8_autotune(planB2, s_ltB2, d_y1_fp8, s_w2_fp8, dG2);
            bool ok = actx.valid && cub2.valid;
            if (!ok) {
                printf("| %5d |   setup failed (fusenorm/cuBLAS) - skipped                    |\n", M);
                fusenorm_free(actx); cublas_free(cub2);
                fp8b::fp8_plan_destroy(planB1); fp8b::fp8_plan_destroy(planB2);
                cudaFree(d_y1_fp8); cudaFree(d_y1_sf); cudaFree(d_x_fp8); cudaFree(d_x_sf);
                cudaFree(x); cudaFree(dA_out); cudaFree(dB_out);
                cudaFree(dG1); cudaFree(dG2); cudaFree(d_rope_cs);
                cudaFree(d_cmain); cudaFree(d_cidx); cudaFree(d_cmbf); cudaFree(d_cibf);
                continue;
            }
            // Two timing modes (a kernel launch costs ~4-6us of CPU overhead here):
            //   exec : queue ALL iters, ONE sync, /N. The CPU runs ahead so back-to-back
            //          kernels leave NO launch gap on the GPU -> pure GPU execution time.
            //          Used for per-op (opA/opB) so their cost/BW is not polluted by the
            //          CPU launch latency. (Same-stream kernels never overlap, so this is
            //          still latency, not a throughput trick.)
            //   step : sync after EACH iter -> one isolated launch+execute = the realistic
            //          SINGLE decode-step latency (CPU launch latency INCLUDED, since you
            //          pay it once per step). Used for A+B and base. Draining per step also
            //          avoids the back-to-back PDL streaming stall that op A's Programmatic-
            //          StreamSerialization reduce accumulates when many pairs are queued.
            auto time_exec = [&](auto&& fn) -> float {
                for (int i = 0; i < warmup; i++) fn();
                cudaDeviceSynchronize();
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a);
                for (int i = 0; i < repeats; i++) fn();
                cudaEventRecord(b); cudaEventSynchronize(b);
                float e; cudaEventElapsedTime(&e, a, b);
                cudaEventDestroy(a); cudaEventDestroy(b);
                return e / repeats * 1000.0f;
            };
            auto time_step = [&](auto&& fn) -> float {
                for (int i = 0; i < warmup; i++) fn();
                cudaDeviceSynchronize();
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                float total = 0.f;
                for (int i = 0; i < repeats; i++) {
                    cudaEventRecord(a); fn(); cudaEventRecord(b);
                    cudaEventSynchronize(b);
                    float e; cudaEventElapsedTime(&e, a, b); total += e;
                }
                cudaEventDestroy(a); cudaEventDestroy(b);
                return total / repeats * 1000.0f;
            };

            // op B WITH the full fused-compressor tail: reduces op A's y2/y3/y4 split-K
            // partials, writes the y3/y4 state slot, then (on compress rows) runs the
            // compressor (aggregate/shift/RMSNorm/RoPE/hadamard/real fp8+fp4 quant).
            auto runB = [&](unsigned long long* prof) {
                return gfnb::wq_b_proj_run(
                    dB_out, dA_out, w2, M, epsP, N1, 0,
                    actx.plan.d_ws, actx.plan.ks, d_ones, dA_out, N1, d_rope_cs, prof,
                    d_pos, d_cape, d_iape, d_cnorm, d_inorm, d_cos, d_sin,
                    d_ckv, d_csc, d_ikv, d_isc,
                    d_q8, d_s8, d_crope, d_q4, d_s4);
            };

            // [profile] one instrumented op B run (WITH tail) -> TC/CC start/end stamps.
            // op A must run first to fill the split-K partials the tail reduces. Warm up
            // first so the stamps reflect STEADY-STATE, not the cold first launch (cold L2/
            // TLB + module load) -- keeps this timeline comparable to opB in the table.
            for (int i = 0; i < warmup; i++) {
                fusenorm_run(actx, dA_out, rms_w1, epsP);
                runB(nullptr);
            }
            cudaDeviceSynchronize();
            cudaMemset(d_prof, 0, (size_t)256 * 16 * sizeof(unsigned long long));
            fusenorm_run(actx, dA_out, rms_w1, epsP);
            runB(d_prof);
            cudaDeviceSynchronize();
            {
                static unsigned long long hp[256 * 16];
                cudaMemcpy(hp, d_prof, (size_t)256 * 16 * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost);
                unsigned long long tcb=~0ull, tce=0, ccb=~0ull, cce=0;
                for (int b = 0; b < 256; b++) {
                    unsigned long long a=hp[b*16+0], e=hp[b*16+1], o=hp[b*16+2], f=hp[b*16+3];
                    if (a && a < tcb) tcb = a;   // TC start: earliest MMA-leader entry
                    if (e > tce) tce = e;        // TC end  : latest epilogue store
                    if (o && o < ccb) ccb = o;   // CC start: earliest tail entry
                    if (f > cce) cce = f;        // CC end  : latest tail finish
                }
                unsigned long long t0 = (tcb < ccb) ? tcb : ccb;   // earliest of the two paths
                r_tc_beg[mi] = (double)(tcb - t0) / 1000.0;
                r_tc_end[mi] = (double)(tce - t0) / 1000.0;
                r_cc_beg[mi] = (double)(ccb - t0) / 1000.0;
                r_cc_end[mi] = (double)(cce - t0) / 1000.0;
                // critical compress-row CTA = the one with a phase[4] stamp and the LATEST phase[7].
                unsigned long long best = 0; int bb = -1;
                for (int b = 0; b < 256; b++) if (hp[b*16+4] && hp[b*16+7] > best) { best = hp[b*16+7]; bb = b; }
                if (bb >= 0) {
                    r_p_beg[mi] = (double)(hp[bb*16+2] - t0) / 1000.0;   // this CTA's tail start
                    r_p_red[mi] = (double)(hp[bb*16+4] - t0) / 1000.0;   // after reduce + state write
                    r_p_agg[mi] = (double)(hp[bb*16+5] - t0) / 1000.0;   // after main aggregate + shift
                    r_p_mnq[mi] = (double)(hp[bb*16+6] - t0) / 1000.0;   // after main RMSNorm+RoPE+fp8 quant
                    r_p_idx[mi] = (double)(hp[bb*16+7] - t0) / 1000.0;   // after indexer (compressor end)
                    r_p_s8[mi]  = (double)(hp[bb*16+8]  - t0) / 1000.0;
                    r_p_s9[mi]  = (double)(hp[bb*16+9]  - t0) / 1000.0;
                    r_p_s10[mi] = (double)(hp[bb*16+10] - t0) / 1000.0;
                }
            }

            float usA = time_exec([&]{ fusenorm_run(actx, dA_out, rms_w1, epsP); });
            float usB = time_exec([&]{ runB(nullptr); });
            // op B WITHOUT the CUDA-core tail: 6-arg call -> ws_tail=null -> the tail branch
            // is skipped, so op B does only the GEMM + head-norm (tail warps stay idle, SAME
            // 512-thread launch). opB - opBnt = the exposed (un-hidden) cost of the folded
            // y2/y3/y4 reduce + y2 RMSNorm + y2 RoPE tail work.
            float usB_nt = time_exec([&]{ gfnb::wq_b_proj_run(dB_out, dA_out, w2, M, epsP, N1); });
            // A+B back-to-back (batched, undrained): CPU runs ahead so the two launches
            // leave no GPU gap -> pure GPU exec of op A + op B.
            float usT_bb = time_exec([&]{ fusenorm_run(actx, dA_out, rms_w1, epsP);
                                          runB(nullptr); });
            // ================= FULL baseline (logic-equivalent to op A + op B) =================
            //  ONE bf16 GEMM1[M,4608] = x @ w1^T produces y1|y2|y3|y4 (== op A's output). Main path then does
            //  y1 RMSNorm -> y2 RMSNorm&RoPE -> GEMM2[M,65536] -> head-norm; the compressor reads y3/y4 straight
            //  from dG1[:, 2048:4608] (bf16): aggregate+shift(K1) -> FUSED main Norm+RoPE+fp8-quant(K2) ->
            //  FUSED idx Norm+RoPE+FWHT+fp4-quant(K3). GEMMs use cuBLASLt (== our tcgen05); post-proc reuses
            //  the Test-4-verified byte-exact cbl kernels (the fused K2/K3 are byte-identical to the old
            //  5-kernel chain and are what Test 4 now verifies). A per-op sequential launch pays ~4-6us CPU
            //  overhead, so the FAIR denominator (base_sum) sums each op's OWN back-to-back exec; base_seq/base_cg too.
            auto runK = [&](cudaStream_t st){
                cbl::k_update_aggregate<<<M,256,0,st>>>(dG1, d_pos, d_cape, d_iape,
                    d_ckv, d_csc, d_ikv, d_isc, d_cmain, d_cidx, M, N1, Y34_LO);   // y3/y4 = dG1[:, 2048:4608] bf16
                cbl::k_main_norm_rope_quant<<<M,512,0,st>>>(d_cmain, d_cnorm, d_cos, d_sin, d_pos,
                    d_cmbf, d_q8, d_s8, d_crope, M);
                cbl::k_idx_norm_rope_had_quant<<<M,256,0,st>>>(d_cidx, d_inorm, d_cos, d_sin, d_pos,
                    d_cibf, d_q4, d_s4, M);
            };
            // fp8 baseline GEMM1 = activation quant (x -> fp8 VEC32) + cuBLASLt MXFP8 block-scale GEMM
            float bG1q = time_exec([&]{ fp8b::quant_vec32_kernel<<<(M*(K1/32)*32+255)/256,256>>>(x, M, K1, K1, d_x_fp8, d_x_sf); });
            float bG1m = time_exec([&]{ fp8b::fp8_gemm_run(planB1, s_ltB2, d_x_fp8, s_w1_fp8, dG1, 0); });
            float bG1  = bG1q + bG1m;   // total fp8 GEMM1 (quant + matmul)
            float bY1 = time_exec([&]{ rmsnorm_fast_run(dG1, dG1, rms_w1, epsP, M, KB, N1); });
            float bY2 = time_exec([&]{ y2_normrope_bf16_kernel<<<M,64>>>(dG1, d_ones, d_rope_cs, epsP, N1); });
            // fp8 baseline GEMM2 = activation quant (y1 -> fp8 VEC32) + cuBLASLt MXFP8 block-scale GEMM
            float bG2q = time_exec([&]{ fp8b::quant_vec32_kernel<<<(M*(KB/32)*32+255)/256,256>>>(dG1, M, KB, N1, d_y1_fp8, d_y1_sf); });
            float bG2m = time_exec([&]{ fp8b::fp8_gemm_run(planB2, s_ltB2, d_y1_fp8, s_w2_fp8, dG2, 0); });
            float bG2  = bG2q + bG2m;   // total fp8 GEMM2 (quant + matmul)
            // OUR megakernel's activation quant (native DSV4 1x128 UE8M0), standalone latency
            // reference. In op B this is FUSED into the kernel prologue (zero extra launch).
            float bQfused = time_exec([&]{ gfnb::quant_act_gran128<<<dim3(M, gfnb::wq_b::NUM_K_TILES),128>>>(dA_out, M, N1, s_qx_fp8, s_qx_sf); });
            float bHN = time_exec([&]{ rmsnorm_fast_run(dG2, dG2, d_ones, epsP, M*(NB/HD), HD, HD); });
            // compressor post-proc (3 cbl kernels; y3/y4 read from dG1[:, 2048:4608], timed individually)
            float bAg = time_exec([&]{ cbl::k_update_aggregate<<<M,256>>>(dG1, d_pos, d_cape, d_iape,
                                          d_ckv, d_csc, d_ikv, d_isc, d_cmain, d_cidx, M, N1, Y34_LO); });
            float bMq = time_exec([&]{ cbl::k_main_norm_rope_quant<<<M,512>>>(d_cmain, d_cnorm, d_cos, d_sin,
                                          d_pos, d_cmbf, d_q8, d_s8, d_crope, M); });
            float bIq = time_exec([&]{ cbl::k_idx_norm_rope_had_quant<<<M,256>>>(d_cidx, d_inorm, d_cos, d_sin,
                                          d_pos, d_cibf, d_q4, d_s4, M); });
            float base_sum = bG1 + bY1 + bY2 + bG2 + bHN + bAg + bMq + bIq;
            float base_seq = time_step([&]{
                fp8b::quant_vec32_kernel<<<(M*(K1/32)*32+255)/256,256>>>(x, M, K1, K1, d_x_fp8, d_x_sf);
                fp8b::fp8_gemm_run(planB1, s_ltB2, d_x_fp8, s_w1_fp8, dG1, 0);
                rmsnorm_fast_run(dG1, dG1, rms_w1, epsP, M, KB, N1);
                y2_normrope_bf16_kernel<<<M,64>>>(dG1, d_ones, d_rope_cs, epsP, N1);
                fp8b::quant_vec32_kernel<<<(M*(KB/32)*32+255)/256,256>>>(dG1, M, KB, N1, d_y1_fp8, d_y1_sf);
                fp8b::fp8_gemm_run(planB2, s_ltB2, d_y1_fp8, s_w2_fp8, dG2, 0);
                rmsnorm_fast_run(dG2, dG2, d_ones, epsP, M*(NB/HD), HD, HD);
                runK(0);
            });

            // ---- CUDA-graph baseline: capture ALL 10 ops onto a stream, replay as ONE graph launch.
            //   base_cg = single-shot latency with per-op CPU launch overhead folded into one submit.
            float base_cg = -1.f;
            {
                cudaStream_t cs; cudaStreamCreate(&cs);
                auto run_base_on = [&](cudaStream_t st){
                    fp8b::quant_vec32_kernel<<<(M*(K1/32)*32+255)/256,256,0,st>>>(x, M, K1, K1, d_x_fp8, d_x_sf);
                    fp8b::fp8_gemm_run(planB1, s_ltB2, d_x_fp8, s_w1_fp8, dG1, st);
                    rmsnorm_fast_run(dG1, dG1, rms_w1, epsP, M, KB, N1, st);
                    y2_normrope_bf16_kernel<<<M,64,0,st>>>(dG1, d_ones, d_rope_cs, epsP, N1);
                    fp8b::quant_vec32_kernel<<<(M*(KB/32)*32+255)/256,256,0,st>>>(dG1, M, KB, N1, d_y1_fp8, d_y1_sf);
                    fp8b::fp8_gemm_run(planB2, s_ltB2, d_y1_fp8, s_w2_fp8, dG2, st);
                    rmsnorm_fast_run(dG2, dG2, d_ones, epsP, M*(NB/HD), HD, HD, st);
                    runK(st);
                };
                run_base_on(cs); cudaStreamSynchronize(cs);   // warmup: keep cuBLASLt lazy init OUT of capture
                cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
                if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                    run_base_on(cs);
                    if (cudaStreamEndCapture(cs, &g) == cudaSuccess &&
                        cudaGraphInstantiate(&ge, g, 0) == cudaSuccess) {
                        for (int i=0;i<warmup;i++) cudaGraphLaunch(ge, cs);
                        cudaStreamSynchronize(cs);
                        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
                        float total=0.f;
                        for (int i=0;i<repeats;i++){
                            cudaEventRecord(a, cs); cudaGraphLaunch(ge, cs); cudaEventRecord(b, cs);
                            cudaEventSynchronize(b); float e; cudaEventElapsedTime(&e,a,b); total+=e;
                        }
                        cudaEventDestroy(a); cudaEventDestroy(b);
                        base_cg = total/repeats*1000.0f;
                        cudaGraphExecDestroy(ge);
                    }
                    if (g) cudaGraphDestroy(g);
                }
                cudaStreamDestroy(cs);
            }

            // ---- effective BW (essential I/O only) on our AB_bb; speedup vs the CUDA-graph baseline ----
            double bytes = (double)M*K1*2.0 + (double)N1*K1*2.0 + (double)NB*KB*1.0
                         + (double)M*NB*2.0 + (double)KB*4.0
                         + (double)(NB/128)*(KB/128)*1.0 /* w2 fp8 blk-scale SF: native UE8M0 128x128 blocks */
                         + (double)M*(KB/128)*1.0        /* x fp8 blk-scale SF: native UE8M0 1x128 rows */;
            double bw = (usT_bb > 0) ? bytes/usT_bb/1.0e6 : 0.0;
            double base_ref = (base_cg > 0.f) ? (double)base_cg : (double)base_sum;  // CUDA-graph baseline (fallback: base_sum)
            double sp = (usT_bb > 0) ? base_ref/usT_bb : 0.0;
            // stash for the summary + per-operator tables printed after the sweep
            op_g1[mi]=bG1; op_y1[mi]=bY1; op_y2[mi]=bY2; op_g2[mi]=bG2; op_hn[mi]=bHN;
            op_g2q[mi]=bG2q; op_g2m[mi]=bG2m; op_bqf[mi]=bQfused;
            op_g1q[mi]=bG1q; op_g1m[mi]=bG1m;
            op_ag[mi]=bAg; op_mq[mi]=bMq; op_iq[mi]=bIq;
            r_ours[mi]=usT_bb; r_base[mi]=base_ref; r_sp[mi]=sp; r_bw[mi]=bw;
            r_opA[mi]=usA; r_opB[mi]=usB; r_opBnt[mi]=usB_nt;

            // ================= op B head-GEMM sanity (fp8 DSV4-quant vs bf16 reference) =================
            // reference: cuBLAS bf16 GEMM2 over the SAME y1 op B consumes (dA_out[:, :1536], op A's
            // output) @ w2^T (UNquantized bf16) -> per-head(512) weightless RMSNorm.
            // ours     : fused op B main output dB_out, computed in e4m3 with DSV4 granularity
            //            (activation 1x128, weight 128x128, UE8M0 scales).
            // The residual is therefore the genuine e4m3 QUANTIZATION error, NOT a kernel bug:
            // e4m3 has ~3 mantissa bits (~12.5% relative step) and the weight shares one UE8M0 per
            // 128x128 block, so on random-uniform weights rel_l2 ~ a few % and cos ~ 0.999 are EXPECTED.
            // This is a gross-breakage guard (a broken GEMM gives cos<<0.99 / NaN), not a bit-exact test.
            // (A bit-exact test would compare against a dequantized-input reference; see notes.)
            cublas_run(cub2, dA_out, w2, dG2);   // reference reads the SAME y1 as op B (dA_out[:, :1536])
            rmsnorm_fast_run(dG2, dG2, d_ones, epsP, M*(NB/HD), HD, HD);
            runB(nullptr);   // refresh ours (dB_out); op A partials are already resident
            cudaDeviceSynchronize();
            {
                CorrMetrics mb = compute_metrics(dB_out, dG2, M, NB, NB);
                bool pb = (mb.cos_sim > 0.998) && (mb.rel_l2 < 0.06);   // e4m3 quant tolerance (not bit-exact)
                printf("  [opB-corr M=%3d] fp8 head-GEMM vs bf16 ref: cos=%8.6f rel_l2=%8.6f max_ae=%7.4f  %s (e4m3 quant err)\n",
                       M, mb.cos_sim, mb.rel_l2, mb.max_abs_err, pb ? "PASS" : "FAIL");
            }

            fusenorm_free(actx); cublas_free(cub2);
            fp8b::fp8_plan_destroy(planB1); fp8b::fp8_plan_destroy(planB2);   // incl. their 32MB workspaces
            cudaFree(d_y1_fp8); cudaFree(d_y1_sf); cudaFree(d_x_fp8); cudaFree(d_x_sf);
            cudaFree(x); cudaFree(dA_out); cudaFree(dB_out);
            cudaFree(dG1); cudaFree(dG2); cudaFree(d_rope_cs);
            cudaFree(d_pos); cudaFree(d_ckv); cudaFree(d_csc); cudaFree(d_ikv); cudaFree(d_isc);
            cudaFree(d_q8); cudaFree(d_s8); cudaFree(d_crope); cudaFree(d_q4); cudaFree(d_s4);
            cudaFree(d_cmain); cudaFree(d_cidx); cudaFree(d_cmbf); cudaFree(d_cibf);
        }
        // ============================ TABLE 1: main comparison ============================
        // ours(A+B) = pure GPU back-to-back exec of op A + op B (launch latency excluded).
        // baseline  = the FULL logic-equivalent pipeline replayed as ONE CUDA graph (base_cg).
        printf("---- Test 3 summary: fused op (A+B) vs the FULL logic-equivalent baseline ----\n");
        printf("+-------+-------------+-------------+----------+-----------+\n");
        printf("|     M |  ours(A+B)  |  baseline   | speedup  |  BW TB/s  |\n");
        printf("|       |  us (bb)    |  us (graph) |          | (fused)   |\n");
        printf("+-------+-------------+-------------+----------+-----------+\n");
        for (int mi = 0; mi < NM3; mi++)
            printf("| %5d | %11.2f | %11.2f | %7.2fx | %9.3f |\n",
                   Ms[mi], r_ours[mi], r_base[mi], r_sp[mi], r_bw[mi]);
        printf("+-------+-------------+-------------+----------+-----------+\n");
        printf("  ours(A+B) = op A + op B back-to-back GPU exec (our fused pipeline; the compressor tail is\n");
        printf("              folded into op B and hidden under its GEMM). baseline = CUDA-graph replay of the\n");
        printf("              10-op reference pipeline below. speedup = baseline / ours(A+B).\n");
        printf("  BW = (x + w1 + w2 + w2_scale + q + rms_w1) essential DRAM traffic / ours(A+B)  [excludes y1 rt & split-K ws].\n\n");

        // ==================== TABLE 2: per-operator baseline breakdown ====================
        // Every op is its OWN back-to-back GPU exec (launch overhead removed) -> a fair, additive
        // decomposition of the baseline. GEMMs are cuBLASLt; compressor kernels are the byte-exact cbl ops.
        // Operators are rows; the NM3 swept M-values are columns. Column widths are computed (not
        // hand-drawn) so a long operator label can never knock the numeric columns out of alignment.
        printf("---- baseline per-operator latency (each = its own back-to-back GPU exec, us) ----\n");
        constexpr int NAME_W = 33;   // operator-label field width (>= the longest label, 32 chars)
        auto t2_dashes = [](int n){ for (int i = 0; i < n; i++) putchar('-'); };
        auto t2_sep = [&](){
            putchar('+'); t2_dashes(4); putchar('+'); t2_dashes(NAME_W + 2); putchar('+');
            for (int mi = 0; mi < NM3; mi++) { t2_dashes(10); putchar('+'); }
            putchar('\n'); };
        auto t2_row = [&](const char* tag, const char* name, const double* a){
            printf("| %-2s | %-*s |", tag, NAME_W, name);
            for (int mi = 0; mi < NM3; mi++) printf(" %8.2f |", a[mi]);
            printf("\n"); };
        t2_sep();
        printf("| %-2s | %-*s |", "#", NAME_W, "operator");
        for (int mi = 0; mi < NM3; mi++) printf(" M=%-5d  |", Ms[mi]);
        printf("\n");
        t2_sep();
        t2_row("1",  "GEMM1 proj [M,4608] K=7168 (fp8)", op_g1);
        t2_row("1a", "  - activation quant (fp8 VEC32)", op_g1q);
        t2_row("1b", "  - cuBLASLt MXFP8 block  GEMM",   op_g1m);
        t2_row("2",  "RMSNorm  y1 (1536)",               op_y1);
        t2_row("3",  "Norm+RoPE  y2 (512)",              op_y2);
        t2_row("4",  "GEMM2 head proj [M,65536] (fp8)",  op_g2);
        t2_row("4a", "  - activation quant (fp8 VEC32)", op_g2q);
        t2_row("4b", "  - cuBLASLt MXFP8 block  GEMM",   op_g2m);
        t2_row("5",  "RMSNorm  per-head (512)",          op_hn);
        t2_row("6",  "Aggregate+Shift  win-kv state",    op_ag);
        t2_row("7",  "Main Norm+RoPE+fp8quant (fused)",  op_mq);
        t2_row("8",  "Idx Norm+RoPE+FWHT+fp4 (fused)",   op_iq);
        t2_sep();
        t2_row("Qf", "(ref) act-quant [FUSED in opB]",   op_bqf);
        t2_sep();
        printf("  ops 1-5 = main path (y1/y2 proj + norms + head GEMM);  ops 6-11 = CSA compressor(y3)+indexer(y4).\n");
        printf("  ONE bf16 GEMM1[M,4608] produces y1|y2|y3|y4 (== op A's output, no recompute); the compressor reads\n");
        printf("  y3/y4 straight from dG1[:, 2048:4608]. Sum of all 11 == base_sum; the CUDA-graph total (baseline\n");
        printf("  above) is lower because the graph overlaps independent ops and folds launch overhead into one submit.\n");
        printf("  Qf = our activation quant measured as a STANDALONE kernel (reference only). It is NOT summed into the\n");
        printf("  baseline and NOT a separate op in our pipeline: op B fuses it into its prologue, so this ~us (mostly a\n");
        printf("  tiny-kernel launch cost) is what the fusion AVOIDS paying separately -- it is hidden inside op B's TC span.\n\n");

        // ---- op A / op B raw exec (our side, for reference) ----
        printf("---- our fused ops raw exec (pure GPU, launch latency excluded, us) ----\n");
        printf("+-------+----------+----------+-----------+----------+\n");
        printf("|     M |  opA     |  opB     | opB-notl  |  AB_bb   |\n");
        printf("+-------+----------+----------+-----------+----------+\n");
        for (int mi = 0; mi < NM3; mi++)
            printf("| %5d | %8.2f | %8.2f | %9.2f | %8.2f |\n",
                   Ms[mi], r_opA[mi], r_opB[mi], r_opBnt[mi], r_ours[mi]);
        printf("+-------+----------+----------+-----------+----------+\n");
        printf("  opA = fused GEMM+y1-norm;  opB = head GEMM + the FULL folded tail;  opB-notl = op B with the\n");
        printf("  CUDA-core tail disabled (GEMM+head-norm only);  opB - opB-notl = the exposed (un-hidden) tail cost.\n");
        printf("  opB folds op A's y2/y3/y4 reduce + y2 norm/rope + the y3/y4 compressor onto CUDA-core warps 8-15\n");
        printf("  (decoupled from GEMM warps 0-7); the timeline below shows whether that tail hides under the GEMM.\n\n");

        // ---- op B internal timeline: tensor-core (GEMM) span vs CUDA-core (tail) span ----
        printf("---- op B internal timeline (GPU %%globaltimer, us relative to the earliest path start) ----\n");
        printf("+-------+----------+----------+----------+----------+----------+----------+\n");
        printf("|     M |  TC_beg  |  TC_end  |  TC_dur  |  CC_beg  |  CC_end  |  CC_dur  |\n");
        printf("+-------+----------+----------+----------+----------+----------+----------+\n");
        for (int mi = 0; mi < NM3; mi++) {
            if (r_tc_beg[mi] < 0) { printf("| %5d |   (skipped)                                              |\n", Ms[mi]); continue; }
            printf("| %5d | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f |\n",
                   Ms[mi], r_tc_beg[mi], r_tc_end[mi], r_tc_end[mi]-r_tc_beg[mi],
                   r_cc_beg[mi], r_cc_end[mi], r_cc_end[mi]-r_cc_beg[mi]);
        }
        printf("+-------+----------+----------+----------+----------+----------+----------+\n");
        printf("  TC = tensor-core path (warps 0-7: fused act-quant + TMA + MMA + epilogue);  CC = CUDA-core tail (warps 8-15).\n");
        printf("  Both START stamps are taken before the fused activation-quant prologue, so both spans include it\n");
        printf("  (TC also absorbs the grid-sync + producer TMA as stall). If [CC_beg,CC_end] sits inside [TC_beg,TC_end] the tail is fully hidden.\n\n");

        // ---- compressor tail phase breakdown (critical compress-row CTA, durations in us) ----
        printf("---- compressor tail phase breakdown (critical compress-row CTA, us) ----\n");
        printf("+-------+----------+----------+----------+----------+----------+----------+\n");
        printf("|     M | tailBeg  | reduce   | mainAgg  | mainNQ   | indexer  | tailTot  |\n");
        printf("+-------+----------+----------+----------+----------+----------+----------+\n");
        for (int mi = 0; mi < NM3; mi++) {
            if (r_p_red[mi] < 0) { printf("| %5d |   (no compress-row phase stamp captured)                 |\n", Ms[mi]); continue; }
            printf("| %5d | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f |\n",
                   Ms[mi], r_p_beg[mi],
                   r_p_red[mi]-r_p_beg[mi],   // reduce (pass1+pass2+pass2b) + y3/y4 state-slot write
                   r_p_agg[mi]-r_p_red[mi],   // main compressor: 8-row softmax aggregate + in-place state shift
                   r_p_mnq[mi]-r_p_agg[mi],   // main compressor: RMSNorm + RoPE(64) + fp8 block64 quant
                   r_p_idx[mi]-r_p_mnq[mi],   // indexer: aggregate/shift/RMSNorm/RoPE/128-pt hadamard/fp4 quant
                   r_p_idx[mi]-r_p_beg[mi]);  // whole CUDA-core tail on this compress-row CTA
        }
        printf("+-------+----------+----------+----------+----------+----------+----------+\n");
        printf("  tailBeg = this CTA's tail start (us from earliest path start); columns below are DURATIONS.\n");
        printf("  reduce  = y2/y3/y4 split-K(4) reduce + y2 rms/rope store + y3/y4 state-slot write.\n");
        printf("  mainAgg = main compressor 8-row softmax aggregate + in-place state shift.\n");
        printf("  mainNQ  = main compressor RMSNorm + RoPE(64) + fp8 block64 quant.\n");
        printf("  indexer = full indexer path (aggregate/shift/RMSNorm/RoPE/128-pt hadamard/fp4 quant).\n");
        printf("  tailTot = reduce+mainAgg+mainNQ+indexer == this CTA's whole CUDA-core tail (~ CC_dur).\n\n");
        printf("---- reduce-internal breakdown (critical compress-row CTA, us) ----\n");
        printf("+-------+----------+----------+----------+----------+----------+\n");
        printf("|     M | p1_load  | p1_rms   | p2_load  | store    | reduce   |\n");
        printf("+-------+----------+----------+----------+----------+----------+\n");
        for (int mi = 0; mi < NM3; mi++) {
            if (r_p_red[mi] < 0) { printf("| %5d |  (no stamp)\n", Ms[mi]); continue; }
            printf("| %5d | %8.3f | %8.3f | %8.3f | %8.3f | %8.3f |\n", Ms[mi],
                   r_p_s8[mi]-r_p_beg[mi], r_p_s9[mi]-r_p_s8[mi], r_p_s10[mi]-r_p_s9[mi],
                   r_p_red[mi]-r_p_s10[mi], r_p_red[mi]-r_p_beg[mi]);
        }
        printf("+-------+----------+----------+----------+----------+----------+\n");
        printf("  p1_load=y2 sumsq loads; p1_rms=2 barriers+reduce+rsqrt; p2_load=comp/idx reduce loads; store=y2 norm/rope + y3/y4 state writes.\n\n");
        cudaFree(d_prof);
        cudaFree(w1); cudaFree(w2); cudaFree(rms_w1); cudaFree(d_ones);
        cudaFree(d_cape); cudaFree(d_iape); cudaFree(d_cnorm); cudaFree(d_inorm); cudaFree(d_cos); cudaFree(d_sin);
    }

    // ================= Test 4: CSA compressor(y3) + indexer(y4) decode baseline =================
    // cuBLAS bf16 GEMM (x @ w_comp^T -> y[M,2560] fp32) + straightforward CUDA post-proc
    // kernels (state update / overlap-cat aggregate / RMSNorm+RoPE / hadamard / REAL fp8+fp4
    // quant), compared byte/elt-wise against the Python golden in test5_data/.
    {
        using cbl::D_M; using cbl::WK_M; using cbl::D_I; using cbl::WK_I;
        using cbl::RD; using cbl::SROWS; using cbl::NW; using cbl::RATIO;
        const int M5 = 64, DIM = 7168;
        const int NF8 = D_M - RD;                 // 448
        printf("\n============ Test 4: CSA compressor(y3)+indexer(y4) decode + real fp8/fp4 quant ============\n");

        // ---- load inputs/weights ----
        void* hx      = t5_load("x",        (size_t)M5*DIM*2);
        void* hw      = t5_load("w_comp",   (size_t)NW*DIM*2);
        void* hcape   = t5_load("comp_ape", (size_t)RATIO*WK_M*4);
        void* hiape   = t5_load("idx_ape",  (size_t)RATIO*WK_I*4);
        void* hcnorm  = t5_load("comp_norm",(size_t)D_M*4);
        void* hinorm  = t5_load("idx_norm", (size_t)D_I*4);
        size_t cos_b  = t5_fsize("cos");
        void* hcos    = t5_load("cos", cos_b);
        void* hsin    = t5_load("sin", cos_b);
        void* hpos    = t5_load("pos",      (size_t)M5*8);
        void* hckv0   = t5_load("comp_kv0", (size_t)M5*SROWS*WK_M*4);
        void* hcsc0   = t5_load("comp_sc0", (size_t)M5*SROWS*WK_M*4);
        void* hikv0   = t5_load("idx_kv0",  (size_t)M5*SROWS*WK_I*4);
        void* hisc0   = t5_load("idx_sc0",  (size_t)M5*SROWS*WK_I*4);

        if (!hx || !hw || !hpos || !hcos) {
            printf("  [Test4] SKIPPED (missing dumps). Generate with: python test5_compressor_golden.py\n");
        } else {
            // ---- upload ----
            bf16_t* dx = (bf16_t*)t5_dev(hx, (size_t)M5*DIM*2);
            bf16_t* dw = (bf16_t*)t5_dev(hw, (size_t)NW*DIM*2);
            float*  d_cape  = (float*)t5_dev(hcape,  (size_t)RATIO*WK_M*4);
            float*  d_iape  = (float*)t5_dev(hiape,  (size_t)RATIO*WK_I*4);
            float*  d_cnorm = (float*)t5_dev(hcnorm, (size_t)D_M*4);
            float*  d_inorm = (float*)t5_dev(hinorm, (size_t)D_I*4);
            float*  d_cos   = (float*)t5_dev(hcos, cos_b);
            float*  d_sin   = (float*)t5_dev(hsin, cos_b);
            long long* d_pos = (long long*)t5_dev(hpos, (size_t)M5*8);
            float*  d_ckv = (float*)t5_dev(hckv0, (size_t)M5*SROWS*WK_M*4);
            float*  d_csc = (float*)t5_dev(hcsc0, (size_t)M5*SROWS*WK_M*4);
            float*  d_ikv = (float*)t5_dev(hikv0, (size_t)M5*SROWS*WK_I*4);
            float*  d_isc = (float*)t5_dev(hisc0, (size_t)M5*SROWS*WK_I*4);

            // ---- GEMM + post-proc are now owned by the baseline (see header) ----
            cbl::QWCBaseline bl; bl.setup(M5, DIM);

            // ---- output buffers ----
            uint8_t* d_q8   = nullptr; cudaMalloc(&d_q8,   (size_t)M5*NF8);
            float*   d_s8   = nullptr; cudaMalloc(&d_s8,   (size_t)M5*(NF8/64)*4);
            cbl::bf16* d_rope = nullptr; cudaMalloc(&d_rope, (size_t)M5*RD*2);
            uint8_t* d_q4   = nullptr; cudaMalloc(&d_q4,   (size_t)M5*(D_I/2));
            uint8_t* d_s4   = nullptr; cudaMalloc(&d_s4,   (size_t)M5*(D_I/32));
            cudaMemset(d_q8, 0, (size_t)M5*NF8);       cudaMemset(d_s8, 0, (size_t)M5*(NF8/64)*4);
            cudaMemset(d_rope, 0, (size_t)M5*RD*2);    cudaMemset(d_q4, 0, (size_t)M5*(D_I/2));
            cudaMemset(d_s4, 0, (size_t)M5*(D_I/32));

            bl.run(dx, dw, d_cape, d_iape, d_cnorm, d_inorm, d_cos, d_sin, d_pos,
                   d_ckv, d_csc, d_ikv, d_isc,
                   d_q8, d_s8, d_rope, d_q4, d_s4);
            cudaDeviceSynchronize();
            cudaError_t kerr = cudaGetLastError();
            if (kerr != cudaSuccess) printf("  [Test4] kernel error: %s\n", cudaGetErrorString(kerr));

            // ---- load golden ----
            uint8_t* g_should = (uint8_t*)t5_load("should",   (size_t)M5);
            uint8_t* g_q8   = (uint8_t*)t5_load("comp_q8",   (size_t)M5*NF8);
            float*   g_s8   = (float*)  t5_load("comp_s8",   (size_t)M5*(NF8/64)*4);
            uint16_t* g_rope= (uint16_t*)t5_load("comp_rope",(size_t)M5*RD*2);
            uint8_t* g_q4   = (uint8_t*)t5_load("idx_q4",    (size_t)M5*(D_I/2));
            uint8_t* g_s4   = (uint8_t*)t5_load("idx_s4",    (size_t)M5*(D_I/32));
            float*   g_ckv  = (float*)  t5_load("comp_kv_out",(size_t)M5*SROWS*WK_M*4);
            float*   g_csc  = (float*)  t5_load("comp_sc_out",(size_t)M5*SROWS*WK_M*4);
            float*   g_ikv  = (float*)  t5_load("idx_kv_out", (size_t)M5*SROWS*WK_I*4);
            float*   g_isc  = (float*)  t5_load("idx_sc_out", (size_t)M5*SROWS*WK_I*4);

            // ---- copy outputs back ----
            uint8_t* h_q8   = (uint8_t*)malloc((size_t)M5*NF8);
            float*   h_s8   = (float*)  malloc((size_t)M5*(NF8/64)*4);
            uint16_t* h_rope= (uint16_t*)malloc((size_t)M5*RD*2);
            uint8_t* h_q4   = (uint8_t*)malloc((size_t)M5*(D_I/2));
            uint8_t* h_s4   = (uint8_t*)malloc((size_t)M5*(D_I/32));
            float*   h_ckv  = (float*)  malloc((size_t)M5*SROWS*WK_M*4);
            float*   h_csc  = (float*)  malloc((size_t)M5*SROWS*WK_M*4);
            float*   h_ikv  = (float*)  malloc((size_t)M5*SROWS*WK_I*4);
            float*   h_isc  = (float*)  malloc((size_t)M5*SROWS*WK_I*4);
            cudaMemcpy(h_q8, d_q8, (size_t)M5*NF8, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_s8, d_s8, (size_t)M5*(NF8/64)*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_rope, d_rope, (size_t)M5*RD*2, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_q4, d_q4, (size_t)M5*(D_I/2), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_s4, d_s4, (size_t)M5*(D_I/32), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ckv, d_ckv, (size_t)M5*SROWS*WK_M*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_csc, d_csc, (size_t)M5*SROWS*WK_M*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ikv, d_ikv, (size_t)M5*SROWS*WK_I*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_isc, d_isc, (size_t)M5*SROWS*WK_I*4, cudaMemcpyDeviceToHost);

            // ---- compare ----
            int nc = 0; for (int m=0;m<M5;m++) nc += (g_should[m]!=0);
            // byte-exact quant outputs, only on compress rows
            auto cmp_bytes = [&](const char* nm, const uint8_t* a, const uint8_t* b, int per_row){
                int mism=0, tot=0;
                for (int m=0;m<M5;m++){ if(!g_should[m]) continue;
                    for (int j=0;j<per_row;j++){ tot++; if(a[(size_t)m*per_row+j]!=b[(size_t)m*per_row+j]) mism++; } }
                printf("  %-12s bytes: %6d/%6d mismatch%s\n", nm, mism, tot, mism==0?"  OK":"");
                return mism; };
            // max abs diff over f32
            auto cmp_f32 = [&](const char* nm, const float* a, const float* b, size_t n, bool compress_only, int per_row){
                double mx=0; size_t at=0;
                for (size_t i=0;i<n;i++){ if(compress_only && !g_should[i/per_row]) continue;
                    double d=fabs((double)a[i]-(double)b[i]); if(d>mx){mx=d;at=i;} }
                printf("  %-12s max|d|: %.3e (idx %zu)\n", nm, mx, at); return mx; };
            // bf16 (uint16) compare as float
            auto cmp_bf16 = [&](const char* nm, const uint16_t* a, const uint16_t* b, int per_row){
                double mx=0;
                for (int m=0;m<M5;m++){ if(!g_should[m]) continue;
                    for (int j=0;j<per_row;j++){ uint32_t ua=(uint32_t)a[(size_t)m*per_row+j]<<16, ub=(uint32_t)b[(size_t)m*per_row+j]<<16;
                        float fa=*(float*)&ua, fb=*(float*)&ub; double d=fabs((double)fa-(double)fb); if(d>mx)mx=d; } }
                printf("  %-12s max|d|: %.3e\n", nm, mx); return mx; };

            printf("  compress rows (should==1): %d/%d\n", nc, M5);
            printf("  -- main compressor (y3) --\n");
            cmp_bytes("comp_q8",  h_q8, g_q8, NF8);
            cmp_f32  ("comp_s8",  h_s8, g_s8, (size_t)M5*(NF8/64), true, NF8/64);
            cmp_bf16 ("comp_rope", h_rope, g_rope, RD);
            printf("  -- indexer (y4) --\n");
            cmp_bytes("idx_q4",   h_q4, g_q4, D_I/2);
            cmp_bytes("idx_s4",   h_s4, g_s4, D_I/32);
            printf("  -- states (all rows) --\n");
            cmp_f32("comp_kv_out", h_ckv, g_ckv, (size_t)M5*SROWS*WK_M, false, 0);
            cmp_f32("comp_sc_out", h_csc, g_csc, (size_t)M5*SROWS*WK_M, false, 0);
            cmp_f32("idx_kv_out",  h_ikv, g_ikv, (size_t)M5*SROWS*WK_I, false, 0);
            cmp_f32("idx_sc_out",  h_isc, g_isc, (size_t)M5*SROWS*WK_I, false, 0);
            printf("  (quant outputs expect 0 byte-mismatch; states/rope expect ~0 up to fp32/bf16 rounding)\n");

            bl.free();
            cudaFree(dx); cudaFree(dw); cudaFree(d_cape); cudaFree(d_iape);
            cudaFree(d_cnorm); cudaFree(d_inorm); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_pos);
            cudaFree(d_ckv); cudaFree(d_csc); cudaFree(d_ikv); cudaFree(d_isc);
            cudaFree(d_q8); cudaFree(d_s8); cudaFree(d_rope); cudaFree(d_q4); cudaFree(d_s4);
            free(h_q8); free(h_s8); free(h_rope); free(h_q4); free(h_s4);
            free(h_ckv); free(h_csc); free(h_ikv); free(h_isc);
            free(g_should); free(g_q8); free(g_s8); free(g_rope); free(g_q4); free(g_s4);
            free(g_ckv); free(g_csc); free(g_ikv); free(g_isc);
        }
        free(hx); free(hw); free(hcape); free(hiape); free(hcnorm); free(hinorm);
        free(hcos); free(hsin); free(hpos); free(hckv0); free(hcsc0); free(hikv0); free(hisc0);
    }

    // ============ Test 5: END-TO-END fused pipeline op A(N=4608) -> op B(fused compressor) ============
    // Same golden inputs as Test 4, but the y3/y4 GEMM + post-proc are produced by the
    // REAL operators: op A (complex_a) runs x@w1^T with w1[2048:4608]==w_comp so op A's
    // split-K partials cols [2048,4608) ARE golden's y3/y4; op B's CUDA-core tail then
    // reduces those partials into the compressor state and runs the fused compressor
    // (aggregate/shift/RMSNorm/RoPE/hadamard/real fp8+fp4 quant) in-place. Compared vs the
    // SAME Python golden as Test 4 -> directly comparable to Test 4's baseline numbers.
    // (op A uses tcgen05 split-K bf16 MMA, a different accumulation order than the golden's
    //  single fp32 matmul, so a few fp8 boundary byte-flips are expected -- see Test 4.)
    {
        using cbl::D_M; using cbl::WK_M; using cbl::D_I; using cbl::WK_I;
        using cbl::RD; using cbl::SROWS; using cbl::NW; using cbl::RATIO;
        const int M6 = 64, DIM = 7168;
        const int N1 = 4608;                     // op A output width (fused: qr|kv|comp|idx)
        const int K1 = DIM;                      // op A contraction (== golden DIM)
        const int KB = gfnb::wq_b::K_DIM;        // 1536  (op B contraction = y1 width)
        const int NB = gfnb::wq_b::N_TOTAL;      // 65536 (op B head-GEMM output width)
        const int HD = gfnb::wq_b::HEAD_DIM;     // 512   (== y2 width / rms_w2 length)
        const int NF8 = D_M - RD;                // 448
        const float epsP = 1e-6f;
        printf("\n============ Test 5: end-to-end fused A(N=4608)->B(compressor) vs golden ============\n");

        // ---- load golden inputs/outputs (own copies; Test 4 freed its own) ----
        void* hx      = t5_load("x",        (size_t)M6*DIM*2);
        void* hw      = t5_load("w_comp",   (size_t)NW*DIM*2);
        void* hcape   = t5_load("comp_ape", (size_t)RATIO*WK_M*4);
        void* hiape   = t5_load("idx_ape",  (size_t)RATIO*WK_I*4);
        void* hcnorm  = t5_load("comp_norm",(size_t)D_M*4);
        void* hinorm  = t5_load("idx_norm", (size_t)D_I*4);
        size_t cos_b  = t5_fsize("cos");
        void* hcos    = t5_load("cos", cos_b);
        void* hsin    = t5_load("sin", cos_b);
        void* hpos    = t5_load("pos",      (size_t)M6*8);
        void* hckv0   = t5_load("comp_kv0", (size_t)M6*SROWS*WK_M*4);
        void* hcsc0   = t5_load("comp_sc0", (size_t)M6*SROWS*WK_M*4);
        void* hikv0   = t5_load("idx_kv0",  (size_t)M6*SROWS*WK_I*4);
        void* hisc0   = t5_load("idx_sc0",  (size_t)M6*SROWS*WK_I*4);

        if (!hx || !hw || !hpos || !hcos) {
            printf("  [Test6] SKIPPED (missing dumps). Generate with: python test5_compressor_golden.py\n");
        } else {
            // ---- upload golden inputs ----
            bf16_t* dx = (bf16_t*)t5_dev(hx, (size_t)M6*DIM*2);
            float*  d_cape  = (float*)t5_dev(hcape,  (size_t)RATIO*WK_M*4);
            float*  d_iape  = (float*)t5_dev(hiape,  (size_t)RATIO*WK_I*4);
            float*  d_cnorm = (float*)t5_dev(hcnorm, (size_t)D_M*4);
            float*  d_inorm = (float*)t5_dev(hinorm, (size_t)D_I*4);
            float*  d_cos   = (float*)t5_dev(hcos, cos_b);
            float*  d_sin   = (float*)t5_dev(hsin, cos_b);
            long long* d_pos = (long long*)t5_dev(hpos, (size_t)M6*8);
            float*  d_ckv = (float*)t5_dev(hckv0, (size_t)M6*SROWS*WK_M*4);
            float*  d_csc = (float*)t5_dev(hcsc0, (size_t)M6*SROWS*WK_M*4);
            float*  d_ikv = (float*)t5_dev(hikv0, (size_t)M6*SROWS*WK_I*4);
            float*  d_isc = (float*)t5_dev(hisc0, (size_t)M6*SROWS*WK_I*4);

            // ---- build op A weight w1[4608,7168]: rows[0,2048) random (qr|kv, don't-care),
            //      rows[2048,4608) == golden w_comp so op A cols [2048,4608) reproduce y3/y4 ----
            bf16_t *w1=nullptr, *w2=nullptr; float *rms_w1=nullptr, *d_ones=nullptr;
            cudaMalloc(&w1, (size_t)N1*K1*sizeof(bf16_t));
            cudaMalloc(&w2, (size_t)NB*KB*sizeof(bf16_t));
            cudaMalloc(&rms_w1, (size_t)N1*sizeof(float));
            cudaMalloc(&d_ones, (size_t)HD*sizeof(float));
            {   // top 2048 rows random; then splice golden w_comp (hw) into rows [2048,4608)
                size_t topN = 2048; bf16_t* htop = (bf16_t*)malloc(topN*K1*sizeof(bf16_t));
                srand(6060); for (size_t i=0;i<topN*(size_t)K1;i++) htop[i]=__float2bfloat16((float)(rand()%200-100)/100.0f);
                cudaMemcpy(w1, htop, topN*K1*sizeof(bf16_t), cudaMemcpyHostToDevice); free(htop);
                cudaMemcpy(w1 + topN*(size_t)K1, hw, (size_t)NW*K1*sizeof(bf16_t), cudaMemcpyHostToDevice);
            }
            {   bf16_t* h=(bf16_t*)malloc((size_t)NB*KB*sizeof(bf16_t)); srand(1234);
                for(size_t i=0;i<(size_t)NB*KB;i++) h[i]=__float2bfloat16((float)(rand()%200-100)/100.0f);
                cudaMemcpy(w2,h,(size_t)NB*KB*sizeof(bf16_t),cudaMemcpyHostToDevice); free(h); }
            {   float* h=(float*)malloc(HD*sizeof(float)); for(int i=0;i<HD;i++) h[i]=1.0f;
                cudaMemcpy(d_ones,h,HD*sizeof(float),cudaMemcpyHostToDevice); free(h); }
            init_rms_w(rms_w1, N1, 42);

            // ---- op A/B output buffers + compressor outputs ----
            bf16_t *dA_out=nullptr, *dB_out=nullptr;
            cudaMalloc(&dA_out, (size_t)M6*N1*sizeof(bf16_t));   // op A out [M,4608] (== dtail)
            cudaMalloc(&dB_out, (size_t)M6*NB*sizeof(bf16_t));   // op B head-GEMM out [M,65536]
            uint8_t* d_q8=nullptr; cudaMalloc(&d_q8, (size_t)M6*NF8);
            float*   d_s8=nullptr; cudaMalloc(&d_s8, (size_t)M6*(NF8/64)*4);
            bf16_t*  d_rope=nullptr; cudaMalloc(&d_rope, (size_t)M6*RD*sizeof(bf16_t));
            uint8_t* d_q4=nullptr; cudaMalloc(&d_q4, (size_t)M6*(D_I/2));
            uint8_t* d_s4=nullptr; cudaMalloc(&d_s4, (size_t)M6*(D_I/32));
            cudaMemset(d_q8,0,(size_t)M6*NF8); cudaMemset(d_s8,0,(size_t)M6*(NF8/64)*4);
            cudaMemset(d_rope,0,(size_t)M6*RD*sizeof(bf16_t)); cudaMemset(d_q4,0,(size_t)M6*(D_I/2));
            cudaMemset(d_s4,0,(size_t)M6*(D_I/32));

            // ---- RUN: op A (fills split-K partials) -> op B (reduces partials + compressor) ----
            FuseNormCtx actx; fusenorm_setup(actx, dx, w1, M6, N1, K1);
            cudaError_t berr = cudaSuccess;
            if (actx.valid) {
                fusenorm_run(actx, dA_out, rms_w1, epsP);
                berr = gfnb::wq_b_proj_run(
                    dB_out, dA_out, w2, M6, epsP, N1, 0,
                    actx.plan.d_ws, actx.plan.ks, d_ones, dA_out, N1, nullptr, nullptr,
                    d_pos, d_cape, d_iape, d_cnorm, d_inorm, d_cos, d_sin,
                    d_ckv, d_csc, d_ikv, d_isc,
                    d_q8, d_s8, d_rope, d_q4, d_s4);
            }
            cudaDeviceSynchronize();
            cudaError_t kerr = cudaGetLastError();
            if (!actx.valid) printf("  [Test6] op A setup FAILED\n");
            if (berr != cudaSuccess) printf("  [Test6] op B launch error: %s\n", cudaGetErrorString(berr));
            if (kerr != cudaSuccess) printf("  [Test6] kernel error: %s\n", cudaGetErrorString(kerr));

            // ---- load golden outputs ----
            uint8_t* g_should=(uint8_t*)t5_load("should",(size_t)M6);
            uint8_t* g_q8=(uint8_t*)t5_load("comp_q8",(size_t)M6*NF8);
            float*   g_s8=(float*)t5_load("comp_s8",(size_t)M6*(NF8/64)*4);
            uint16_t*g_rope=(uint16_t*)t5_load("comp_rope",(size_t)M6*RD*2);
            uint8_t* g_q4=(uint8_t*)t5_load("idx_q4",(size_t)M6*(D_I/2));
            uint8_t* g_s4=(uint8_t*)t5_load("idx_s4",(size_t)M6*(D_I/32));
            float*   g_ckv=(float*)t5_load("comp_kv_out",(size_t)M6*SROWS*WK_M*4);
            float*   g_csc=(float*)t5_load("comp_sc_out",(size_t)M6*SROWS*WK_M*4);
            float*   g_ikv=(float*)t5_load("idx_kv_out",(size_t)M6*SROWS*WK_I*4);
            float*   g_isc=(float*)t5_load("idx_sc_out",(size_t)M6*SROWS*WK_I*4);

            // ---- copy op B outputs back ----
            uint8_t* h_q8=(uint8_t*)malloc((size_t)M6*NF8);
            float*   h_s8=(float*)malloc((size_t)M6*(NF8/64)*4);
            uint16_t*h_rope=(uint16_t*)malloc((size_t)M6*RD*2);
            uint8_t* h_q4=(uint8_t*)malloc((size_t)M6*(D_I/2));
            uint8_t* h_s4=(uint8_t*)malloc((size_t)M6*(D_I/32));
            float*   h_ckv=(float*)malloc((size_t)M6*SROWS*WK_M*4);
            float*   h_csc=(float*)malloc((size_t)M6*SROWS*WK_M*4);
            float*   h_ikv=(float*)malloc((size_t)M6*SROWS*WK_I*4);
            float*   h_isc=(float*)malloc((size_t)M6*SROWS*WK_I*4);
            cudaMemcpy(h_q8,d_q8,(size_t)M6*NF8,cudaMemcpyDeviceToHost);
            cudaMemcpy(h_s8,d_s8,(size_t)M6*(NF8/64)*4,cudaMemcpyDeviceToHost);
            cudaMemcpy(h_rope,d_rope,(size_t)M6*RD*2,cudaMemcpyDeviceToHost);
            cudaMemcpy(h_q4,d_q4,(size_t)M6*(D_I/2),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_s4,d_s4,(size_t)M6*(D_I/32),cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ckv,d_ckv,(size_t)M6*SROWS*WK_M*4,cudaMemcpyDeviceToHost);
            cudaMemcpy(h_csc,d_csc,(size_t)M6*SROWS*WK_M*4,cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ikv,d_ikv,(size_t)M6*SROWS*WK_I*4,cudaMemcpyDeviceToHost);
            cudaMemcpy(h_isc,d_isc,(size_t)M6*SROWS*WK_I*4,cudaMemcpyDeviceToHost);

            // ---- compare (same criteria as Test 4) ----
            int nc=0; for(int m=0;m<M6;m++) nc += (g_should && g_should[m]!=0);
            auto cmp_bytes=[&](const char* nm,const uint8_t* a,const uint8_t* b,int per_row){
                int mism=0,tot=0; for(int m=0;m<M6;m++){ if(!g_should[m]) continue;
                    for(int j=0;j<per_row;j++){tot++; if(a[(size_t)m*per_row+j]!=b[(size_t)m*per_row+j]) mism++;} }
                printf("  %-12s bytes: %6d/%6d mismatch%s\n", nm, mism, tot, mism==0?"  OK":""); return mism; };
            auto cmp_f32=[&](const char* nm,const float* a,const float* b,size_t n,bool conly,int per_row){
                double mx=0; size_t at=0; for(size_t i=0;i<n;i++){ if(conly && !g_should[i/per_row]) continue;
                    double d=fabs((double)a[i]-(double)b[i]); if(d>mx){mx=d;at=i;} }
                printf("  %-12s max|d|: %.3e (idx %zu)\n", nm, mx, at); return mx; };
            auto cmp_bf16=[&](const char* nm,const uint16_t* a,const uint16_t* b,int per_row){
                double mx=0; for(int m=0;m<M6;m++){ if(!g_should[m]) continue;
                    for(int j=0;j<per_row;j++){ uint32_t ua=(uint32_t)a[(size_t)m*per_row+j]<<16, ub=(uint32_t)b[(size_t)m*per_row+j]<<16;
                        float fa=*(float*)&ua, fb=*(float*)&ub; double d=fabs((double)fa-(double)fb); if(d>mx)mx=d; } }
                printf("  %-12s max|d|: %.3e\n", nm, mx); return mx; };

            printf("  compress rows (should==1): %d/%d\n", nc, M6);
            printf("  -- main compressor (y3) --\n");
            cmp_bytes("comp_q8", h_q8, g_q8, NF8);
            cmp_f32  ("comp_s8", h_s8, g_s8, (size_t)M6*(NF8/64), true, NF8/64);
            cmp_bf16 ("comp_rope", h_rope, g_rope, RD);
            printf("  -- indexer (y4) --\n");
            cmp_bytes("idx_q4", h_q4, g_q4, D_I/2);
            cmp_bytes("idx_s4", h_s4, g_s4, D_I/32);
            printf("  -- states (all rows) --\n");
            cmp_f32("comp_kv_out", h_ckv, g_ckv, (size_t)M6*SROWS*WK_M, false, 0);
            cmp_f32("comp_sc_out", h_csc, g_csc, (size_t)M6*SROWS*WK_M, false, 0);
            cmp_f32("idx_kv_out",  h_ikv, g_ikv, (size_t)M6*SROWS*WK_I, false, 0);
            cmp_f32("idx_sc_out",  h_isc, g_isc, (size_t)M6*SROWS*WK_I, false, 0);
            printf("  (fused op B tail: y3/y4 from op A split-K partials -> compressor. vs Test 4, quant\n");
            printf("   byte-flips may differ slightly [tcgen05 split-K order]; states/rope expect ~1e-5.)\n");

            fusenorm_free(actx);
            cudaFree(dx); cudaFree(w1); cudaFree(w2); cudaFree(rms_w1); cudaFree(d_ones);
            cudaFree(d_cape); cudaFree(d_iape); cudaFree(d_cnorm); cudaFree(d_inorm);
            cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_pos);
            cudaFree(d_ckv); cudaFree(d_csc); cudaFree(d_ikv); cudaFree(d_isc);
            cudaFree(dA_out); cudaFree(dB_out);
            cudaFree(d_q8); cudaFree(d_s8); cudaFree(d_rope); cudaFree(d_q4); cudaFree(d_s4);
            free(h_q8); free(h_s8); free(h_rope); free(h_q4); free(h_s4);
            free(h_ckv); free(h_csc); free(h_ikv); free(h_isc);
            free(g_should); free(g_q8); free(g_s8); free(g_rope); free(g_q4); free(g_s4);
            free(g_ckv); free(g_csc); free(g_ikv); free(g_isc);
        }
        free(hx); free(hw); free(hcape); free(hiape); free(hcnorm); free(hinorm);
        free(hcos); free(hsin); free(hpos); free(hckv0); free(hcsc0); free(hikv0); free(hisc0);
    }

    return 0;
}
