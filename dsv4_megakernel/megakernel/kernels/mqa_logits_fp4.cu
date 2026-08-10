// ============================================================
// mqa_logits_fp4.cu — DSV4 score attention (FP4 MQA-logits, PAGED decode).
//
// logits[b,t] = Σ_h relu(<q[b,h,:], kv[b,t,:]>) · weights[b,h],  t < ctx_b
//
// Usage (one launch, in-kernel tile-pool schedule -- no metadata kernel):
//   logits = mqa_logits_fp4_decode(
//       q [B,64,64] i8 fp4, sf_q [B,64] i32,          # = idx_post outputs
//       kv_cache [num_blocks, 4352] u8,               # fused pages:
//                                                     # [64tok*64B fp4 | 64tok*4B sf]
//       weights [B,64] f32, context_lens [B] i32, block_table [B,max_pages] i32,
//       max_context_len, out_dtype)                   # -> [B, max_ctx], tail -inf
//   _out variant writes a preallocated buffer; num_kv_stages=0 -> auto.
//
// Kernel body: include/mqa_logits_fp4.cuh (self-contained, CUTLASS-only).
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/bfloat16.h>

#include "mqa_logits_fp4.cuh"   // self-contained kernel (only CUTLASS/CuTe)

namespace {

using namespace mqa_logits_fp4;

static int host_align_up(int a, int b) { return (a + b - 1) / b * b; }

// ---- "each op as its OWN kernel" reference (benchmark sep_us column only) ----
// Well-written standalone version of the op the fused tail hides: same math and
// lane layout as the tail, but with the whole GPU to itself (one warp per row,
// full grid). sep_us = base + this one ≙ test_complex.cu's base_sum.
__global__ void standalone_compressor_kernel(deep_gemm::MainCompressorArgs comp,
                                             uint32_t seq_len, float eps) {
    // [C1] one 128-thread block per row, same 4-warp cooperative chain as the tail
    const uint32_t m = blockIdx.x;
    if (m >= seq_len)
        return;
    const long long p = comp.pos[m];
    if (((p + 1) & 3) != 0)
        return;
    deep_gemm::run_main_compressor_row(comp, m, p, threadIdx.x >> 5,
                                       threadIdx.x & 31, eps, /*barrier_id=*/0, nullptr);
}

// One CTA per SM (queried once; matches DeepGEMM's device_runtime->get_num_sms()).
static int get_num_sms() {
    static int num_sms = -1;
    if (num_sms < 0) {
        int device = 0;
        TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
        TORCH_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess,
                    "cudaDeviceGetAttribute(MultiProcessorCount) failed");
    }
    return num_sms;
}

static CUtensorMapSwizzle to_swizzle(int mode) {
    switch (mode) {
        case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default:  return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

// Generic 2D TMA descriptor. Replicates DeepGEMM runtime_utils.hpp
// make_tma_2d_desc semantics, including the packed-FP4 smem-inner fixup.
static CUtensorMap make_tma_2d(const char* name, void* ptr, CUtensorMapDataType dtype,
                               int elem_size, int gmem_inner, int gmem_outer,
                               int smem_inner, int smem_outer, int gmem_outer_stride_elems,
                               int swizzle_mode, bool is_fp4 = false,
                               bool fp4_unpacked_smem = true) {
    if (swizzle_mode != 0)
        smem_inner = swizzle_mode / elem_size;
    if (is_fp4 && !fp4_unpacked_smem && swizzle_mode != 0)
        smem_inner = swizzle_mode * 2;

    CUtensorMap tm{};
    cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gstr[1]  = {(cuuint64_t)gmem_outer_stride_elems * elem_size};
    cuuint32_t estr[2]  = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tm, dtype, 2, ptr, gdims, gstr, sdims, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, to_swizzle(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(res, &msg);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    (msg ? msg : "unknown"), " [gmem=", gmem_inner, "x", gmem_outer,
                    " smem=", smem_inner, "x", smem_outer, " stride=", gmem_outer_stride_elems,
                    " swizzle=", swizzle_mode, "]");
    }
    return tm;
}

// Generic 3D TMA descriptor (paged KV: outermost dim = physical page id).
// dim0 is contiguous; stride1 = bytes between rows, stride2 = bytes between pages
// (the fused page stride, so the fp4 view simply SKIPS each page's SF tail).
static CUtensorMap make_tma_3d(const char* name, void* ptr, CUtensorMapDataType dtype,
                               int elem_size, int g0, int g1, int g2,
                               int s0, int s1, int s2,
                               int row_stride_elems, int64_t page_stride_bytes,
                               int swizzle_mode, bool is_fp4 = false,
                               bool fp4_unpacked_smem = true) {
    int smem_inner = s0;
    if (swizzle_mode != 0)
        smem_inner = swizzle_mode / elem_size;
    if (is_fp4 && !fp4_unpacked_smem && swizzle_mode != 0)
        smem_inner = swizzle_mode * 2;

    CUtensorMap tm{};
    cuuint64_t gdims[3] = {(cuuint64_t)g0, (cuuint64_t)g1, (cuuint64_t)g2};
    cuuint32_t sdims[3] = {(cuuint32_t)smem_inner, (cuuint32_t)s1, (cuuint32_t)s2};
    cuuint64_t gstr[2]  = {(cuuint64_t)row_stride_elems * elem_size,
                           (cuuint64_t)page_stride_bytes};
    cuuint32_t estr[3]  = {1, 1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tm, dtype, 3, ptr, gdims, gstr, sdims, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, to_swizzle(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(res, &msg);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    (msg ? msg : "unknown"), " [gmem=", g0, "x", g1, "x", g2,
                    " smem=", smem_inner, "x", s1, "x", s2,
                    " strides=", row_stride_elems, ",", page_stride_bytes,
                    " swizzle=", swizzle_mode, "]");
    }
    return tm;
}

// Per-stage / total dynamic shared-memory bytes for a given KV pipeline depth
// (mirrors the wrapper; KV stages are now a dispatch dimension, see pick paths below).
static int compute_smem_bytes(int num_kv_stages) {
    const int smem_q   = BLOCK_Q * NUM_HEADS * (HEAD_DIM / 2);            // 8192
    const int smem_sfq = host_align_up(BLOCK_Q * NUM_HEADS, 128) * 4;     // 512
    const int smem_kv  = BLOCK_KV * (HEAD_DIM / 2);                        // 16384
    const int smem_sfkv= host_align_up(BLOCK_KV, 128) * 4;                // 1024
    const int smem_w   = BLOCK_Q * NUM_HEADS * 4;                          // 512
    const int barriers = (NUM_Q_STAGES + num_kv_stages + NUM_TMEM_STAGES) * 2 * 8;
    const int tmem_ptr = 4;
    // tile-pool prefix scratch (after the TMEM ptr; see smem_tile_prefix in the kernel)
    const int tile_prefix = ((int)deep_gemm::kNumMaxTilePoolTokens + 1) * 4;
    return NUM_Q_STAGES * (smem_q + smem_sfq + smem_w) +
           num_kv_stages * (smem_kv + smem_sfkv) + barriers + tmem_ptr + tile_prefix;
}

// KV pipeline depth (used when num_kv_stages=0). PAGED B300 sweep over {4,6,8,10}
// x the full BxT grid: the winner follows the per-CTA chunk length — each
// 256-slot tile is now EIGHT page-granular TMAs (4 data + 4 SF), so long chunks
// need a deeper in-flight window to cover the issue latency; short chunks never
// fill any pipeline and prefer the smaller L1 carveout.
static int auto_kv_stages(int B, int max_context_len, int num_ctas) {
    if (num_ctas < 1) num_ctas = get_num_sms();
    const int64_t tiles = (int64_t)B * ((max_context_len + BLOCK_KV - 1) / BLOCK_KV);
    const int64_t per_cta = (tiles + num_ctas - 1) / num_ctas;
    if (per_cta <= 8)  return 4;
    if (per_cta <= 16) return 6;
    return 8;
}

template <typename logits_dtype_t, int kKVStages, int kPageKV>
static void launch_typed(int seq_len, int seq_len_kv, int stride_logits,
                         const int* context_lens, const int* block_table, int bt_stride,
                         void* logits,
                         const unsigned int* iq_ready, const unsigned int* iq_gen,
                         int iq_world, int iq_skip,
                         float comp_eps, unsigned long long* prof,
                         const deep_gemm::MainCompressorArgs& comp,
                         bool attn_mock,
                         const CUtensorMap& dQ, const CUtensorMap& dSFQ,
                         const CUtensorMap& dKV, const CUtensorMap& dSFKV,
                         const CUtensorMap& dW, dim3 grid, int smem,
                         cudaStream_t stream) {
    auto kernel = &deep_gemm::sm100_fp4_mqa_logits<
        NUM_HEADS, HEAD_DIM,
        BLOCK_Q, BLOCK_KV, kPageKV, NUM_Q_STAGES, kKVStages,
        NUM_SPECIALIZED_THREADS, NUM_MATH_THREADS, NUM_TAIL_THREADS, logits_dtype_t>;

    // cudaFuncSetAttribute is PER DEVICE state, so this cache is indexed by device
    // ordinal: a single `static bool` lets the first card configure the function and
    // every launch on the second one fail, which is why one process could never
    // drive two GPUs.
    static constexpr int kMaxDevices = 16;
    static bool configured[kMaxDevices] = {};
    int cfg_dev = 0;
    TORCH_CHECK(cudaGetDevice(&cfg_dev) == cudaSuccess, "cudaGetDevice failed");
    TORCH_CHECK(cfg_dev >= 0 && cfg_dev < kMaxDevices,
                "device ordinal ", cfg_dev, " exceeds kMaxDevices=", kMaxDevices);
    if (!configured[cfg_dev]) {
        auto e = cudaFuncSetAttribute((void*)kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute: ", cudaGetErrorString(e),
                    " smem=", smem);
        configured[cfg_dev] = true;
    }

    // PDL: allow this grid to start while the producer (wq_b) is still draining
    // its tail. cudaGridDependencySynchronize() inside the kernel is what actually
    // orders the producer's stores; without the attribute that call is a no-op wait
    // that returns immediately, and with a non-PDL predecessor it is also harmless
    // (stream order already separates them). iq_ready != nullptr is the signal that
    // a cross-rank producer is in play and the overlap is worth having.
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(TPB, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    int nattr = 0;
    if (iq_ready != nullptr) {
        attrs[nattr].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[nattr].val.programmaticStreamSerializationAllowed = 1;
        ++nattr;
    }
    config.attrs = attrs;
    config.numAttrs = nattr;
    auto lerr = cudaLaunchKernelEx(
        &config, kernel,
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t)stride_logits,
        reinterpret_cast<const uint32_t*>(context_lens),
        reinterpret_cast<const uint32_t*>(block_table),
        (uint32_t)bt_stride,
        reinterpret_cast<logits_dtype_t*>(logits),
        iq_ready, iq_gen, (uint32_t)iq_world, (uint32_t)iq_skip,
        comp_eps, prof,
        comp, attn_mock,
        dQ, dSFQ, dKV, dSFKV, dW);
    TORCH_CHECK(lerr == cudaSuccess, "mqa_logits_fp4 launch failed: ",
                cudaGetErrorString(lerr));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mqa_logits_fp4 launch failed: ", cudaGetErrorString(err));
}

// Build the 5 TMA descriptors + grid, then dispatch (out_dtype x KV stages) and
// launch INTO a caller-provided logits buffer `lp`. No allocation here (so callers
// can hoist it out of a timed loop, matching the repo's *_out convention).
//   q [B,H,D/2] i8, sf_q [B,H] i32, weights [B,H] f32
//   kv_cache = fused pages [num_blocks, PAGE_KV*(D/2+4)] bytes
//   context_lens [B] i32, block_table [B, max_pages] i32 (physical page ids)
static void dispatch_launch(torch::Tensor q, torch::Tensor sf_q,
                            torch::Tensor kv_cache, torch::Tensor weights,
                            const int* context_lens, const int* block_table, int bt_stride,
                            int num_blocks, int kv_entries_per_block,
                            int64_t kv_block_stride_bytes,
                            int B, int max_context_len,
                            at::ScalarType out_dtype, int stride_logits,
                            int num_ctas, int num_kv_stages,
                            float comp_eps, unsigned long long* prof,
                            const deep_gemm::MainCompressorArgs& comp,
                            bool attn_mock,
                            void* lp,
                            const unsigned int* iq_ready = nullptr,
                            const unsigned int* iq_gen = nullptr,
                            int iq_world = 0, int iq_skip = -1) {
    constexpr int H = NUM_HEADS, D = HEAD_DIM;
    auto stream = at::cuda::getCurrentCUDAStream();

    const int q_elem  = (int)q.element_size();    // 1 (int8-packed fp4)
    const int sf_elem = (int)sf_q.element_size(); // 4 (int32)
#if CUDA_VERSION >= 12080
    const CUtensorMapDataType FP4_DT = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
#else
    #error "FP4 packed TMA (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B) requires CUDA >= 12.8"
#endif

    CUtensorMap dQ = make_tma_2d("q", q.data_ptr(), FP4_DT, q_elem,
                                 D, B * H, D, BLOCK_Q * H,
                                 (int)q.stride(1), D / 2, /*is_fp4=*/true, /*unpacked=*/false);
    CUtensorMap dSFQ = make_tma_2d("sf_q", sf_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, sf_elem,
                                   H, B, H, BLOCK_Q, (int)sf_q.stride(0), 0);
    CUtensorMap dW = make_tma_2d("weights", weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                                 H, B, H, BLOCK_Q, (int)weights.stride(0), 0);
    // Fused page cache split into strided views (DeepGEMM attention.hpp from_blob
    // pattern): KV = 3D fp4 view (outermost dim = physical page id, page stride
    // skips each page's SF tail); SF begins after every entry's FP4 body.
    CUtensorMap dKV = make_tma_3d("kv_paged", kv_cache.data_ptr(), FP4_DT, q_elem,
                                  D, kv_entries_per_block, num_blocks,
                                  D, kv_entries_per_block, 1,
                                  /*row stride=*/D / 2,
                                  /*page stride=*/kv_block_stride_bytes,
                                  D / 2, /*is_fp4=*/true, /*unpacked=*/false);
    CUtensorMap dSFKV = make_tma_2d("sf_kv_paged",
                                    static_cast<uint8_t*>(kv_cache.data_ptr())
                                        + (int64_t)kv_entries_per_block * (D / 2),
                                    CU_TENSOR_MAP_DATA_TYPE_INT32, sf_elem,
                                    kv_entries_per_block, num_blocks,
                                    kv_entries_per_block, 1,
                                    kv_block_stride_bytes / 4, 0);

    // Grid: num_ctas CTAs (default = #SMs) split the global KV tile pool; empty
    // chunks exit immediately, so grid.x never over-subscribes.
    if (num_ctas < 1) num_ctas = get_num_sms();
    dim3 grid((unsigned)num_ctas, 1, 1);
    if (num_kv_stages == 0)
        num_kv_stages = auto_kv_stages(B, max_context_len, num_ctas);
    const int smem = compute_smem_bytes(num_kv_stages);
    // Capacity clamp for context_lens: whatever the block_table can address.
    const int seq_len_kv_cap = bt_stride * kv_entries_per_block;

    // Instantiated combos: {4,6,8,10} KV stages x {f32, bf16} logits (AOT stand-in
    // for DeepGEMM's JIT per-shape configs).
    #define MQA_LT_PAGE(dtype_t, stages_, page_) \
        launch_typed<dtype_t, stages_, page_>(B, seq_len_kv_cap, stride_logits, \
                                              context_lens, block_table, bt_stride, lp, \
                                              iq_ready, iq_gen, iq_world, iq_skip, \
                                              comp_eps, prof, comp, attn_mock, \
                                              dQ,dSFQ,dKV,dSFKV,dW, grid, smem, stream)
    #define MQA_LT(dtype_t, stages_) \
        do { \
            switch (kv_entries_per_block) { \
                case 32: MQA_LT_PAGE(dtype_t, stages_, 32); break; \
                case 64: MQA_LT_PAGE(dtype_t, stages_, 64); break; \
                default: MQA_LT_PAGE(dtype_t, stages_, 128); break; \
            } \
        } while (0)
    TORCH_CHECK(num_kv_stages == 4 || num_kv_stages == 6 || num_kv_stages == 8 || num_kv_stages == 10,
                "num_kv_stages must be 0 (auto) or one of 4/6/8/10, got ", num_kv_stages);
    if (out_dtype == torch::kFloat) {
        switch (num_kv_stages) {
            case 4:  MQA_LT(float, 4);  break;
            case 6:  MQA_LT(float, 6);  break;
            case 8:  MQA_LT(float, 8);  break;
            default: MQA_LT(float, 10); break;
        }
    } else {
        switch (num_kv_stages) {
            case 4:  MQA_LT(cutlass::bfloat16_t, 4);  break;
            case 6:  MQA_LT(cutlass::bfloat16_t, 6);  break;
            case 8:  MQA_LT(cutlass::bfloat16_t, 8);  break;
            default: MQA_LT(cutlass::bfloat16_t, 10); break;
        }
    }
    #undef MQA_LT
    #undef MQA_LT_PAGE
}

// Shared checks for the PAGED decode entries. Returns (B, num_blocks, max_pages).
static std::tuple<int, int, int> check_paged(
    const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table, at::ScalarType out_dtype,
    int kv_entries_per_block, int64_t& kv_block_stride_bytes) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt8 && q.dim() == 3
                && q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM / 2 && q.is_contiguous(),
                "q must be CUDA int8-packed fp4 [B,H,D/2] contiguous");
    const int B = (int)q.size(0);
    TORCH_CHECK(B <= (int)deep_gemm::kNumMaxTilePoolTokens,
                "decode B must be <= ", (int)deep_gemm::kNumMaxTilePoolTokens,
                " (tile-pool smem prefix cap)");
    TORCH_CHECK(sf_q.is_cuda() && sf_q.scalar_type() == torch::kInt32 && sf_q.is_contiguous()
                && sf_q.sizes() == torch::IntArrayRef({B, NUM_HEADS}), "sf_q [B,H] i32");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat
                && weights.sizes() == torch::IntArrayRef({B, NUM_HEADS})
                && weights.stride(1) == 1, "weights [B,H] f32");
    TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kBFloat16, "out_dtype float/bf16");
    TORCH_CHECK(kv_entries_per_block == 32 || kv_entries_per_block == 64 ||
                kv_entries_per_block == 128,
                "kv_entries_per_block must be one of 32/64/128");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.is_contiguous() && kv_cache.dim() == 2
                && (kv_cache.scalar_type() == torch::kInt8 || kv_cache.scalar_type() == torch::kUInt8),
                "kv_cache must be CUDA (u)int8 physical pages [blocks,stride]");
    const int num_blocks = (int)kv_cache.size(0);
    if (kv_block_stride_bytes == 0)
        kv_block_stride_bytes = kv_cache.size(1);
    const int64_t payload = (int64_t)kv_entries_per_block * (HEAD_DIM / 2 + 4);
    TORCH_CHECK(kv_block_stride_bytes >= payload && kv_block_stride_bytes % 16 == 0,
                "kv_block_stride_bytes must cover the FP4 payload and be 16B aligned");
    TORCH_CHECK(kv_cache.numel() == (int64_t)num_blocks * kv_block_stride_bytes,
                "kv_cache must expose exactly kv_block_stride_bytes per block");
    TORCH_CHECK(context_lens.is_cuda() && context_lens.scalar_type() == torch::kInt32
                && context_lens.numel() == B && context_lens.is_contiguous(),
                "context_lens [B] i32");
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32
                && block_table.dim() == 2 && block_table.size(0) == B
                && block_table.is_contiguous(), "block_table [B, max_pages] i32 contiguous");
    return {B, num_blocks, (int)block_table.size(1)};
}

}  // namespace

// ======================== PyTorch bindings ========================

// Multi-batch PAGED decode, COMPRESSED + self-clean: allocating wrapper.
// ONE launch for all B tokens; grid.x = num_ctas (default #SMs), tile-pool
// schedule inside the kernel (NO metadata kernel, no schedule_meta).
//   q [B,H,D/2] i8, sf_q [B,H] i32, weights [B,H] f32
//   kv_cache fused pages [num_blocks, PAGE_KV*(D/2+4)] bytes
//   context_lens [B] i32, block_table [B, max_pages] i32 (physical page ids)
// Returns logits [B, max_context_len]; entries >= ctx_b are -inf.
static torch::Tensor mqa_logits_fp4_decode(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv_cache,
    torch::Tensor weights, torch::Tensor context_lens, torch::Tensor block_table,
    int max_context_len, at::ScalarType out_dtype, int num_ctas = 0,
    int num_kv_stages = 0, int kv_entries_per_block = PAGE_KV,
    int64_t kv_block_stride_bytes = 0) {
    auto [B, num_blocks, max_pages] = check_paged(q, sf_q, kv_cache, weights,
                                                  context_lens, block_table, out_dtype,
                                                  kv_entries_per_block,
                                                  kv_block_stride_bytes);
    TORCH_CHECK(max_context_len > 0 &&
                max_context_len <= max_pages * kv_entries_per_block,
                "max_context_len exceeds block-table capacity");

    const int stride_logits = host_align_up(max_context_len, BLOCK_KV);
    torch::Tensor buf = torch::full({B, stride_logits},
                                    -std::numeric_limits<float>::infinity(),
                                    q.options().dtype(out_dtype));
    dispatch_launch(q, sf_q, kv_cache, weights,
                    context_lens.data_ptr<int>(), block_table.data_ptr<int>(), max_pages,
                    num_blocks, kv_entries_per_block, kv_block_stride_bytes,
                    B, max_context_len, out_dtype, stride_logits,
                    num_ctas, num_kv_stages,
                    /*comp_eps=*/1e-6f, /*prof=*/nullptr,
                    /*comp=*/deep_gemm::MainCompressorArgs{},
                    /*attn_mock=*/false,
                    buf.data_ptr());
    return buf.index({torch::indexing::Slice(0, B),
                      torch::indexing::Slice(0, max_context_len)});
}

// Preallocated-output PAGED decode (repo *_out convention): the timed region is
// just 5 descriptors + ONE launch — no per-call alloc/-inf-fill. Caller provides
// `logits` preallocated as [>=B, align(max_ctx,BLOCK_KV)] pre-filled with -inf ONCE
// (the kernel only overwrites each row's [0,ctx) so the -inf tail persists across
// reuse). Writes into `logits` in place; caller slices [:B, :max_ctx].
static void mqa_logits_fp4_decode_out(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv_cache,
    torch::Tensor weights, torch::Tensor context_lens, torch::Tensor block_table,
    torch::Tensor logits, int num_ctas, int num_kv_stages,
    double comp_eps, c10::optional<torch::Tensor> prof,
    c10::optional<torch::Tensor> cmp_pos, c10::optional<torch::Tensor> comp_norm,
    c10::optional<torch::Tensor> cos_tab, c10::optional<torch::Tensor> sin_tab,
    c10::optional<torch::Tensor> comp_state,
    c10::optional<torch::Tensor> comp_q8, c10::optional<torch::Tensor> comp_s8,
    c10::optional<torch::Tensor> comp_rope, bool mock_attn,
    c10::optional<torch::Tensor> comp_state_row,
    c10::optional<torch::Tensor> cmp_cache,
    c10::optional<torch::Tensor> cmp_dst,
    int64_t kv_entries_per_block = PAGE_KV,
    int64_t kv_block_stride_bytes = 0,
    int64_t comp_state_ring_entries = 0,
    int64_t cmp_entries_per_block = 0,
    int64_t cmp_block_stride_bytes = 0,
    // Cross-rank iq readiness (TP/DP direct-write handoff from wq_b). Only the Q
    // TMA warp waits, so the KV pipeline (99.85% of this kernel's traffic) starts
    // immediately and hides the peer skew -- that is why this belongs here rather
    // than in a separate wait kernel on the critical path.
    c10::optional<torch::Tensor> iq_ready = c10::nullopt,
    c10::optional<torch::Tensor> iq_gen = c10::nullopt,
    // iq_skip is the rank slot NOT to wait on. It defaults to -1 (wait on every
    // rank, including ourselves) because PDL releases this kernel before the local
    // wq_b has finished its own SPREAD, so skipping our own slot is wrong too. A
    // default of 0 silently made every rank skip slot 0, i.e. rank 1 never waited
    // for rank 0's data at all.
    int64_t iq_world = 0, int64_t iq_skip = -1) {
    auto [B, num_blocks, max_pages] = check_paged(q, sf_q, kv_cache, weights,
                                                  context_lens, block_table,
                                                  logits.scalar_type(),
                                                  (int)kv_entries_per_block,
                                                  kv_block_stride_bytes);
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= B
                && logits.size(1) % BLOCK_KV == 0 && logits.is_contiguous(),
                "logits must be [>=B, k*BLOCK_KV] contiguous");
    TORCH_CHECK(num_ctas >= 0, "num_ctas must be >= 0 (0 = one CTA per SM)");
    TORCH_CHECK(iq_ready.has_value() == iq_gen.has_value(),
                "iq_ready and iq_gen must be given together");
    if (iq_ready.has_value()) {
        TORCH_CHECK(iq_ready->is_cuda() && iq_ready->is_contiguous()
                    && iq_ready->scalar_type() == torch::kInt32
                    && iq_world > 0 && iq_world <= iq_ready->numel(),
                    "iq_ready must be CUDA i32 [>= iq_world]");
        TORCH_CHECK(iq_gen->is_cuda() && iq_gen->numel() == 1
                    && iq_gen->scalar_type() == torch::kInt32,
                    "iq_gen must be a 1-element CUDA i32");
    }

    // Optional globaltimer stamps [num_ctas, 8] i64 (test_complex.cu phase pattern):
    // 0=attn start, 1=attn end, 2=tail start, 3=tail end, 4=retired (was rms end),
    // 5=aggregate done, 6=retired (the RTP ring needs no state-shift phase),
    // 7=compress-row end (ns).
    unsigned long long* prof_ptr = nullptr;
    if (prof.has_value()) {
        auto& p = prof.value();
        TORCH_CHECK(p.is_cuda() && p.scalar_type() == torch::kInt64 && p.is_contiguous(),
                    "prof must be CUDA int64 contiguous");
        const int need = 8 * (num_ctas > 0 ? num_ctas : get_num_sms());
        TORCH_CHECK(p.numel() >= need, "prof needs >= ", need, " elements");
        prof_ptr = reinterpret_cast<unsigned long long*>(p.data_ptr());
    }

    // Optional MAIN-compressor bundle (gemm_fuse_norm_b d=512 part; tail warps).
    // All-or-none: gated on the framework state ring.
    deep_gemm::MainCompressorArgs comp{};
    if (comp_state.has_value()) {
        const int64_t comp_B = cmp_pos.has_value() ? cmp_pos->numel() : 0;
        const bool cache_mode = cmp_cache.has_value() && cmp_cache->numel() > 0;
        // Compact q8/s8/rope optional in cache mode (double-write saver).
        TORCH_CHECK(cmp_pos && comp_norm && cos_tab && sin_tab && comp_state_row
                    && (cache_mode || (comp_q8 && comp_s8 && comp_rope)),
                    "main-compressor tensors must be given together");
        auto chk = [](const torch::Tensor& t, at::ScalarType ty, const char* n) {
            TORCH_CHECK(t.is_cuda() && t.scalar_type() == ty && t.is_contiguous(),
                        n, " must be CUDA contiguous of the right dtype");
        };
        chk(*cmp_pos, torch::kInt64, "cmp_pos");   chk(*comp_norm, torch::kFloat, "comp_norm");
        chk(*cos_tab, torch::kFloat, "cos_tab");   chk(*sin_tab, torch::kFloat, "sin_tab");
        chk(*comp_state, torch::kFloat, "comp_state");
        chk(*comp_state_row, torch::kInt32, "comp_state_row");
        TORCH_CHECK(comp_B >= B
                    && comp_B <= (int64_t)deep_gemm::kNumMaxTilePoolTokens
                    && comp_norm->numel() == 512
                    && comp_state->dim() >= 2 && comp_state->size(-1) == 2048
                    && comp_state_row->numel() >= comp_B,
                    "main-compressor tensor shapes");
        TORCH_CHECK(comp_state_ring_entries >= 8,
                    "comp_state_ring_entries must cover the 8-row window");
        if (comp_q8.has_value() && comp_q8->numel() > 0) {
            chk(*comp_q8, torch::kUInt8, "comp_q8");
            TORCH_CHECK(comp_q8->numel() >= comp_B * 448, "comp_q8 shape");
            comp.q8 = comp_q8->data_ptr<uint8_t>();
        }
        if (comp_s8.has_value() && comp_s8->numel() > 0) {
            chk(*comp_s8, torch::kFloat, "comp_s8");
            TORCH_CHECK(comp_s8->numel() >= comp_B * 7, "comp_s8 shape");
            comp.s8 = comp_s8->data_ptr<float>();
        }
        if (comp_rope.has_value() && comp_rope->numel() > 0) {
            chk(*comp_rope, torch::kBFloat16, "comp_rope");
            TORCH_CHECK(comp_rope->numel() >= comp_B * 64, "comp_rope shape");
            comp.rope = reinterpret_cast<nv_bfloat16*>(comp_rope->data_ptr());
        }
        // int64_t is `long` on linux; the device struct uses `long long` (same width)
        comp.pos = reinterpret_cast<const long long*>(cmp_pos->data_ptr<int64_t>());
        comp.norm = comp_norm->data_ptr<float>();
        comp.cos_tab = cos_tab->data_ptr<float>();
        comp.sin_tab = sin_tab->data_ptr<float>();
        comp.state = comp_state->data_ptr<float>();
        comp.state_row = comp_state_row->data_ptr<int>();
        comp.state_ring_entries = (int)comp_state_ring_entries;
        comp.seq_len = (uint32_t)comp_B;
        if (cmp_cache.has_value() && cmp_cache->numel() > 0) {
            TORCH_CHECK(cmp_dst.has_value(), "cmp_cache requires cmp_dst");
            TORCH_CHECK(cmp_entries_per_block > 0,
                        "cmp_cache requires cmp_entries_per_block");
            const int64_t payload = cmp_entries_per_block *
                (deep_gemm::M1_TOK_BODY + deep_gemm::M1_SCALE_RECORD_BYTES);
            TORCH_CHECK(cmp_block_stride_bytes >= payload,
                        "cmp_block_stride_bytes must cover the MODEL1 payload");
            TORCH_CHECK(cmp_cache->is_cuda() && cmp_cache->is_contiguous() &&
                        cmp_cache->scalar_type() == torch::kUInt8 &&
                        cmp_cache->numel() % cmp_block_stride_bytes == 0,
                        "cmp_cache must be uint8 physical pages with supplied stride");
            TORCH_CHECK(cmp_dst->is_cuda() && cmp_dst->is_contiguous() &&
                        cmp_dst->scalar_type() == torch::kInt32 &&
                        cmp_dst->numel() >= comp_B,
                        "cmp_dst must cover all compressor rows");
            comp.cmp_cache = reinterpret_cast<uint8_t*>(cmp_cache->data_ptr());
            comp.cmp_dst = cmp_dst->data_ptr<int>();
            comp.cmp_entries_per_block = (int)cmp_entries_per_block;
            comp.cmp_block_stride_bytes = (size_t)cmp_block_stride_bytes;
        }
    }

    dispatch_launch(q, sf_q, kv_cache, weights,
                    context_lens.data_ptr<int>(), block_table.data_ptr<int>(), max_pages,
                    num_blocks, (int)kv_entries_per_block, kv_block_stride_bytes,
                    B, /*max_context_len=*/(int)logits.size(1),
                    logits.scalar_type(), /*stride_logits=*/(int)logits.size(1),
                    num_ctas, num_kv_stages,
                    (float)comp_eps, prof_ptr,
                    comp,
                    /*attn_mock=*/mock_attn,
                    logits.data_ptr(),
                    iq_ready.has_value()
                        ? reinterpret_cast<const unsigned int*>(iq_ready->data_ptr())
                        : nullptr,
                    iq_gen.has_value()
                        ? reinterpret_cast<const unsigned int*>(iq_gen->data_ptr())
                        : nullptr,
                    (int)iq_world, (int)iq_skip);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mqa_compressor_standalone",
          [](torch::Tensor cmp_pos, torch::Tensor comp_norm,
             torch::Tensor cos_tab, torch::Tensor sin_tab,
             torch::Tensor comp_state, torch::Tensor comp_state_row,
             int64_t comp_state_ring_entries,
             torch::Tensor comp_q8, torch::Tensor comp_s8,
             torch::Tensor comp_rope, double eps) {
              deep_gemm::MainCompressorArgs comp{};
              comp.pos = reinterpret_cast<const long long*>(cmp_pos.data_ptr<int64_t>());
              comp.norm = comp_norm.data_ptr<float>();
              comp.cos_tab = cos_tab.data_ptr<float>();
              comp.sin_tab = sin_tab.data_ptr<float>();
              comp.state = comp_state.data_ptr<float>();
              comp.state_row = comp_state_row.data_ptr<int>();
              comp.state_ring_entries = (int)comp_state_ring_entries;
              comp.q8 = comp_q8.data_ptr<uint8_t>();
              comp.s8 = comp_s8.data_ptr<float>();
              comp.rope = reinterpret_cast<nv_bfloat16*>(comp_rope.data_ptr());
              const uint32_t B = (uint32_t)cmp_pos.numel();
              standalone_compressor_kernel<<<B, 128, 0,
                                             at::cuda::getCurrentCUDAStream()>>>(
                  comp, B, (float)eps);
          },
          "standalone MAIN-compressor reference (benchmark sep_us column)",
          py::arg("cmp_pos"), py::arg("comp_norm"), py::arg("cos_tab"), py::arg("sin_tab"),
          py::arg("comp_state"), py::arg("comp_state_row"),
          py::arg("comp_state_ring_entries"), py::arg("comp_q8"), py::arg("comp_s8"),
          py::arg("comp_rope"), py::arg("eps") = 1e-6);
    m.def("mqa_compressor_paged",
          [](torch::Tensor cmp_pos, torch::Tensor comp_norm,
             torch::Tensor cos_tab, torch::Tensor sin_tab,
             torch::Tensor comp_state, torch::Tensor comp_state_row,
             int64_t comp_state_ring_entries,
             torch::Tensor cmp_cache, torch::Tensor cmp_dst,
             int64_t cmp_entries_per_block, int64_t cmp_block_stride_bytes,
             double eps) {
              const int64_t B = cmp_pos.numel();
              TORCH_CHECK(B >= 1 && cmp_pos.is_cuda() && cmp_pos.is_contiguous()
                          && cmp_pos.scalar_type() == torch::kInt64,
                          "cmp_pos must be contiguous CUDA int64 [B]");
              TORCH_CHECK(comp_norm.is_cuda() && comp_norm.is_contiguous()
                          && comp_norm.scalar_type() == torch::kFloat32
                          && comp_norm.numel() == 512,
                          "comp_norm must be contiguous CUDA fp32 [512]");
              TORCH_CHECK(cos_tab.is_cuda() && cos_tab.is_contiguous()
                          && cos_tab.scalar_type() == torch::kFloat32
                          && cos_tab.dim() == 2 && cos_tab.size(1) == 32
                          && sin_tab.is_cuda() && sin_tab.is_contiguous()
                          && sin_tab.scalar_type() == torch::kFloat32
                          && sin_tab.sizes() == cos_tab.sizes(),
                          "cos_tab/sin_tab must be contiguous CUDA fp32 [S,32]");
              TORCH_CHECK(comp_state.is_cuda() && comp_state.is_contiguous()
                          && comp_state.scalar_type() == torch::kFloat32
                          && comp_state.dim() >= 2 && comp_state.size(-1) == 2048,
                          "comp_state must be contiguous CUDA fp32 [...,2048]");
              TORCH_CHECK(comp_state_row.is_cuda() && comp_state_row.is_contiguous()
                          && comp_state_row.scalar_type() == torch::kInt32
                          && comp_state_row.numel() >= B
                          && comp_state_ring_entries >= 8,
                          "comp_state_row/ring geometry is invalid");
              const int64_t payload = cmp_entries_per_block
                  * (deep_gemm::M1_TOK_BODY + deep_gemm::M1_SCALE_RECORD_BYTES);
              TORCH_CHECK(cmp_entries_per_block > 0
                          && cmp_block_stride_bytes >= payload
                          && cmp_cache.is_cuda() && cmp_cache.is_contiguous()
                          && cmp_cache.scalar_type() == torch::kUInt8
                          && cmp_cache.numel() % cmp_block_stride_bytes == 0,
                          "cmp_cache physical page geometry is invalid");
              TORCH_CHECK(cmp_dst.is_cuda() && cmp_dst.is_contiguous()
                          && cmp_dst.scalar_type() == torch::kInt32
                          && cmp_dst.numel() >= B,
                          "cmp_dst must be contiguous CUDA int32 [>=B]");

              deep_gemm::MainCompressorArgs comp{};
              comp.pos = reinterpret_cast<const long long*>(cmp_pos.data_ptr<int64_t>());
              comp.norm = comp_norm.data_ptr<float>();
              comp.cos_tab = cos_tab.data_ptr<float>();
              comp.sin_tab = sin_tab.data_ptr<float>();
              comp.state = comp_state.data_ptr<float>();
              comp.state_row = comp_state_row.data_ptr<int>();
              comp.state_ring_entries = (int)comp_state_ring_entries;
              comp.seq_len = (uint32_t)B;
              comp.cmp_cache = reinterpret_cast<uint8_t*>(cmp_cache.data_ptr());
              comp.cmp_dst = cmp_dst.data_ptr<int>();
              comp.cmp_entries_per_block = (int)cmp_entries_per_block;
              comp.cmp_block_stride_bytes = (size_t)cmp_block_stride_bytes;
              standalone_compressor_kernel<<<(uint32_t)B, 128, 0,
                  at::cuda::getCurrentCUDAStream()>>>(comp, (uint32_t)B,
                                                      (float)eps);
          },
          "Standalone MAIN compressor writing RTP MODEL1 physical pages",
          py::arg("cmp_pos"), py::arg("comp_norm"), py::arg("cos_tab"),
          py::arg("sin_tab"), py::arg("comp_state"),
          py::arg("comp_state_row"), py::arg("comp_state_ring_entries"),
          py::arg("cmp_cache"), py::arg("cmp_dst"),
          py::arg("cmp_entries_per_block"),
          py::arg("cmp_block_stride_bytes"), py::arg("eps") = 1e-6);
    m.def("mqa_logits_fp4_decode", &mqa_logits_fp4_decode,
          "DSV4 FP4 MQA-logits (multi-batch PAGED decode, compressed, ONE launch, "
          "in-kernel tile-pool schedule — no metadata kernel): fused page cache "
          "with runtime logical entries/physical stride + block_table; num_ctas=0 -> one CTA per SM; "
          "num_kv_stages=0 -> auto (chunk-length heuristic), or force 4/6/8/10",
          py::arg("q"), py::arg("sf_q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("max_context_len"),
          py::arg("out_dtype"), py::arg("num_ctas") = 0, py::arg("num_kv_stages") = 0,
          py::arg("kv_entries_per_block") = PAGE_KV,
          py::arg("kv_block_stride_bytes") = 0);
    m.def("mqa_logits_fp4_decode_out", &mqa_logits_fp4_decode_out,
          "DSV4 FP4 PAGED decode into a preallocated buffer (repo *_out convention; "
          "hoists alloc/-inf-fill out of the timed path); num_ctas=0 -> one CTA per SM; "
          "num_kv_stages=0 -> auto (chunk-length heuristic), or force 4/6/8/10; "
          "optional MAIN-indexer compressor fused into the tail warpgroup (cmp_* / comp_*)",
          py::arg("q"), py::arg("sf_q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("logits"),
          py::arg("num_ctas") = 0,
          py::arg("num_kv_stages") = 0,
          py::arg("comp_eps") = 1e-6, py::arg("prof") = c10::nullopt,
          py::arg("cmp_pos") = c10::nullopt, py::arg("comp_norm") = c10::nullopt,
          py::arg("cos_tab") = c10::nullopt, py::arg("sin_tab") = c10::nullopt,
          py::arg("comp_state") = c10::nullopt,
          py::arg("comp_q8") = c10::nullopt, py::arg("comp_s8") = c10::nullopt,
          py::arg("comp_rope") = c10::nullopt, py::arg("mock_attn") = false,
          py::arg("comp_state_row") = c10::nullopt,
          py::arg("cmp_cache") = c10::nullopt,
          py::arg("cmp_dst") = c10::nullopt,
          py::arg("kv_entries_per_block") = PAGE_KV,
          py::arg("kv_block_stride_bytes") = 0,
          py::arg("comp_state_ring_entries") = 0,
          py::arg("cmp_entries_per_block") = 0,
          py::arg("cmp_block_stride_bytes") = 0,
          py::arg("iq_ready") = c10::nullopt, py::arg("iq_gen") = c10::nullopt,
          py::arg("iq_world") = 0, py::arg("iq_skip") = -1);
}
