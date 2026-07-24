// ============================================================
// mqa_logits_fp4.cu — Host launcher + PyTorch binding for the migrated
// DeepGEMM FP4 MQA-logits (DSV4 Sparse Top-K Indexer score attention).
//
// Kernel body + helper closure are inlined self-contained (only CUTLASS/CuTe) in
// megakernel/include/mqa_logits_fp4.cuh (modulo 2 documented AOT edits).
//
// Two entry points (PAGED decode only, VERBATIM DeepGEMM sm100 paged kernel --
// see include/dg_paged_mqa_logits.cuh):
//   get_paged_mqa_logits_metadata — DG's per-step schedule kernel ([num_sms+1,2] i32).
//   mqa_logits_fp4_decode(_out)   — DG's fp8_fp4_paged_mqa_logits, mxfp4 / H=64 /
//                                   D=128 / next_n=1 / page 64 specialization;
//                                   clean_logits=False semantics (RAW row tails).
// decode_out optionally FUSES the DSV4 MAIN-indexer compressor rows into a
// CUDA-core tail warpgroup (TPB=512, hidden under the KV stream); the compressor
// also remains available as a standalone kernel.
//
// Host TMA setup mirrors DeepGEMM sm100_mqa_logits.hpp + attention.hpp;
// launch pattern mirrors kernels/w1_merged_fp8_gemm.cu.
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>
#include <tuple>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/bfloat16.h>

#include "dg_paged_mqa_logits.cuh"   // vendored DeepGEMM paged kernel (+ shared helpers)

namespace {

using namespace dg_paged_mqa_logits;

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
                                       threadIdx.x & 31, eps, /*barrier_id=*/0);
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

// Generic 3D TMA descriptor (paged KV: batch dim = physical page id). Mirrors
// DeepGEMM runtime_utils.hpp make_tma_3d_desc semantics incl. the packed-FP4 fixup.
static CUtensorMap make_tma_3d(const char* name, void* ptr, CUtensorMapDataType dtype,
                               int elem_size, int gd0, int gd1, int gd2,
                               int sd0, int sd1, int sd2,
                               int gstride0_elems, int gstride1_elems,
                               int swizzle_mode, bool is_fp4 = false,
                               bool fp4_unpacked_smem = true) {
    if (swizzle_mode != 0)
        sd0 = swizzle_mode / elem_size;
    if (is_fp4 && !fp4_unpacked_smem && swizzle_mode != 0)
        sd0 = swizzle_mode * 2;

    CUtensorMap tm{};
    cuuint64_t gdims[3] = {(cuuint64_t)gd0, (cuuint64_t)gd1, (cuuint64_t)gd2};
    cuuint32_t sdims[3] = {(cuuint32_t)sd0, (cuuint32_t)sd1, (cuuint32_t)sd2};
    cuuint64_t gstr[2]  = {(cuuint64_t)gstride0_elems * elem_size,
                           (cuuint64_t)gstride1_elems * elem_size};
    cuuint32_t estr[3]  = {1, 1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tm, dtype, 3, ptr, gdims, gstr, sdims, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, to_swizzle(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        const char* msg = nullptr; cuGetErrorString(res, &msg);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    (msg ? msg : "unknown"), " [gmem=", gd0, "x", gd1, "x", gd2,
                    " strides=", gstride0_elems, ",", gstride1_elems,
                    " swizzle=", swizzle_mode, "]");
    }
    return tm;
}

// Per DeepGEMM: dynamic smem = sizeof(MQALogitsSharedStorage) for the given config.
static int compute_smem_bytes() {
    return (int)sizeof(SharedStorage);
}

// [DG-ALIGNED] metadata kernel launch (deep_gemm.get_paged_mqa_logits_metadata):
// context_lens 2D [B, next_n], schedule_meta [num_sms + 1, 2] i32.
static void dg_metadata_launch(const int* context_lens, int* schedule_meta,
                               int num_requests, int num_sms, cudaStream_t stream) {
    auto kernel = &deep_gemm::sched::sm100_paged_mqa_logits_metadata<
        NEXT_N, /*BLOCK_Q(placeholder)=*/1, SPLIT_KV>;
    const int smem = 2 * num_requests * (int)sizeof(int);
    kernel<<<1, 256, smem, stream>>>(
        (uint32_t)num_requests, (uint32_t)(num_requests * NEXT_N), (uint32_t)num_sms,
        reinterpret_cast<const uint32_t*>(context_lens),
        reinterpret_cast<uint32_t*>(schedule_meta));
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "metadata launch failed: ", cudaGetErrorString(err));
}

// [DG-ALIGNED] main paged kernel launch (mxfp4, H=64, D=128, next_n=1) with the
// fused MAIN-compressor tail warpgroup (TPB=512; comp.kv==nullptr -> tail idles).
template <typename logits_dtype_t>
static void dg_paged_launch(int num_q_tokens_total, int logits_stride, int bt_stride,
                            const int* context_lens, void* logits,
                            const int* block_table, const int* schedule_meta,
                            float comp_eps, const deep_gemm::MainCompressorArgs& comp,
                            bool attn_mock,
                            const CUtensorMap& dQ, const CUtensorMap& dSFQ,
                            const CUtensorMap& dKV, const CUtensorMap& dSFKV,
                            const CUtensorMap& dW, int num_sms, cudaStream_t stream) {
    auto kernel = &deep_gemm::sm100_paged_mqa_logits<
        NEXT_N, NUM_HEADS, HEAD_DIM, PAGE_KV,
        NUM_Q_STAGES, NUM_KV_STAGES, SPLIT_KV, SPLITS_PER_CHUNK,
        NUM_SPECIALIZED_THREADS, NUM_MATH_THREADS, NUM_TAIL_THREADS,
        cutlass::float_e2m1_t, logits_dtype_t>;

    const int smem = compute_smem_bytes();
    static bool configured = false;   // per-instantiation (template static local)
    if (!configured) {
        auto e = cudaFuncSetAttribute((void*)kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute: ", cudaGetErrorString(e),
                    " smem=", smem);
        configured = true;
    }

    kernel<<<dim3((unsigned)num_sms, 1, 1), dim3(TPB, 1, 1), smem, stream>>>(
        (uint32_t)num_q_tokens_total,
        (uint32_t)logits_stride, (uint32_t)bt_stride,
        reinterpret_cast<const uint32_t*>(context_lens),
        reinterpret_cast<logits_dtype_t*>(logits),
        reinterpret_cast<const uint32_t*>(block_table),
        reinterpret_cast<const uint32_t*>(schedule_meta),
        comp_eps, comp, attn_mock,
        dQ, dSFQ, dKV, dSFKV, dW);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "paged_mqa_logits launch failed: ", cudaGetErrorString(err));
}

// Build the 5 TMA descriptors exactly as DeepGEMM's sm100_mqa_logits.hpp paged
// host does, then launch the vendored kernel into a caller-provided buffer.
//   q [B,H,D/2] i8 (== DG's [B,1,H,D/2] with next_n=1), sf_q [B,H] i32,
//   kv_cache = fused pages [num_blocks, PAGE_KV, 1, D/2+4] bytes, weights [B,H] f32
static void dispatch_launch(torch::Tensor q, torch::Tensor sf_q,
                            torch::Tensor kv_cache, torch::Tensor weights,
                            const int* context_lens, const int* block_table, int bt_stride,
                            int num_blocks, int B, int stride_logits,
                            at::ScalarType out_dtype,
                            const int* schedule_meta, int num_sms,
                            void* lp,
                            float comp_eps, const deep_gemm::MainCompressorArgs& comp,
                            bool attn_mock) {
    constexpr int H = NUM_HEADS, D = HEAD_DIM;
    constexpr int PAGE_STRIDE = PAGE_KV * (HEAD_DIM / 2 + 4);   // fused page bytes (4352)
    auto stream = at::cuda::getCurrentCUDAStream();

    const int q_elem  = (int)q.element_size();    // 1 (int8-packed fp4)
    const int sf_elem = (int)sf_q.element_size(); // 4 (int32)
#if CUDA_VERSION >= 12080
    const CUtensorMapDataType FP4_DT = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
#else
    #error "FP4 packed TMA (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B) requires CUDA >= 12.8"
#endif

    // Mirrors DeepGEMM: q desc smem tile = BLOCK_Q*H rows (BLOCK_Q = 128/H = 2);
    // sf_q / weights desc smem tile = BLOCK_Q rows.
    CUtensorMap dQ = make_tma_2d("q", q.data_ptr(), FP4_DT, q_elem,
                                 D, B * H, D, BLOCK_Q * H,
                                 (int)q.stride(1), D / 2, /*is_fp4=*/true, /*unpacked=*/false);
    CUtensorMap dSFQ = make_tma_2d("sf_q", sf_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, sf_elem,
                                   H, B, H, BLOCK_Q, (int)sf_q.stride(0), 0);
    CUtensorMap dW = make_tma_2d("weights", weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                                 H, B, H, BLOCK_Q, (int)weights.stride(0), 0);
    // Fused page cache split into strided views (DG attention.hpp from_blob pattern):
    // KV 3D (batch dim = physical page id), SF 2D at byte offset PAGE_KV*(D/2).
    CUtensorMap dKV = make_tma_3d("kv_paged", kv_cache.data_ptr(), FP4_DT, q_elem,
                                  D, PAGE_KV, num_blocks,
                                  D, PAGE_KV, 1,
                                  /*row stride=*/D / 2, /*page stride=*/PAGE_STRIDE,
                                  D / 2, /*is_fp4=*/true, /*unpacked=*/false);
    CUtensorMap dSFKV = make_tma_2d("sf_kv_paged",
                                    static_cast<uint8_t*>(kv_cache.data_ptr()) + (int64_t)PAGE_KV * (D / 2),
                                    CU_TENSOR_MAP_DATA_TYPE_INT32, 4,
                                    PAGE_KV, num_blocks, PAGE_KV, 1,
                                    PAGE_STRIDE / 4, 0);

    if (out_dtype == torch::kFloat)
        dg_paged_launch<float>(B * NEXT_N, stride_logits, bt_stride, context_lens, lp,
                               block_table, schedule_meta, comp_eps, comp, attn_mock,
                               dQ, dSFQ, dKV, dSFKV, dW, num_sms, stream);
    else
        dg_paged_launch<cutlass::bfloat16_t>(B * NEXT_N, stride_logits, bt_stride, context_lens, lp,
                                             block_table, schedule_meta, comp_eps, comp, attn_mock,
                                             dQ, dSFQ, dKV, dSFKV, dW, num_sms, stream);
}

// Shared checks for the PAGED decode entries (exact deep_gemm.fp8_fp4_paged_mqa_logits
// shape conventions, next_n=1 specialization). Returns (B, num_blocks, max_pages).
static std::tuple<int, int, int> check_paged(
    const torch::Tensor& q, const torch::Tensor& sf_q, const torch::Tensor& kv_cache,
    const torch::Tensor& weights, const torch::Tensor& context_lens,
    const torch::Tensor& block_table, at::ScalarType out_dtype) {
    constexpr int PAGE_BYTES = PAGE_KV * (HEAD_DIM / 2 + 4);   // 4352
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt8 && q.dim() == 3
                && q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM / 2 && q.is_contiguous(),
                "q must be CUDA int8-packed fp4 [B,H,D/2] contiguous");
    const int B = (int)q.size(0);
    TORCH_CHECK(sf_q.is_cuda() && sf_q.scalar_type() == torch::kInt32 && sf_q.is_contiguous()
                && sf_q.sizes() == torch::IntArrayRef({B, NUM_HEADS}), "sf_q [B,H] i32");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat
                && weights.sizes() == torch::IntArrayRef({B, NUM_HEADS})
                && weights.stride(1) == 1, "weights [B,H] f32");
    TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kBFloat16, "out_dtype float/bf16");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.is_contiguous()
                && (kv_cache.scalar_type() == torch::kInt8 || kv_cache.scalar_type() == torch::kUInt8),
                "kv_cache must be CUDA (u)int8 fused pages");
    const int num_blocks = (int)kv_cache.size(0);
    TORCH_CHECK(kv_cache.numel() == (int64_t)num_blocks * PAGE_BYTES,
                "kv_cache must be [num_blocks, ", PAGE_BYTES, "] bytes per page "
                "(fused: PAGE_KV*(D/2) fp4 then PAGE_KV*4 sf)");
    TORCH_CHECK(context_lens.is_cuda() && context_lens.scalar_type() == torch::kInt32
                && context_lens.numel() == B && context_lens.is_contiguous(),
                "context_lens [B] i32 (== DG's 2D [B, next_n=1])");
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == torch::kInt32
                && block_table.dim() == 2 && block_table.size(0) == B
                && block_table.is_contiguous(), "block_table [B, max_pages] i32 contiguous");
    return {B, num_blocks, (int)block_table.size(1)};
}

}  // namespace

// ======================== PyTorch bindings ========================

// deep_gemm.get_paged_mqa_logits_metadata equivalent: ONE 256-thread block builds
// the per-SM (q_token_idx, kv_split_idx) starts. Returns [num_sms + 1, 2] i32.
static torch::Tensor get_paged_mqa_logits_metadata(torch::Tensor context_lens, int num_sms) {
    TORCH_CHECK(context_lens.is_cuda() && context_lens.scalar_type() == torch::kInt32
                && context_lens.is_contiguous() && context_lens.numel() > 0,
                "context_lens must be CUDA i32 contiguous");
    if (num_sms < 1) num_sms = get_num_sms();
    const int B = (int)context_lens.numel();
    auto meta = torch::empty({num_sms + 1, 2},
                             torch::TensorOptions().dtype(torch::kInt32)
                                                   .device(context_lens.device()));
    dg_metadata_launch(context_lens.data_ptr<int>(), meta.data_ptr<int>(),
                       B, num_sms, at::cuda::getCurrentCUDAStream());
    return meta;
}

// Multi-batch PAGED decode, DG-ALIGNED (== deep_gemm.fp8_fp4_paged_mqa_logits with
// clean_logits=False, next_n=1): allocating wrapper that also runs the metadata
// kernel. Returns RAW logits [B, max_context_len]; entries >= ctx_b are GARBAGE
// (DG semantics -- downstream topk masks by lengths).
//   q [B,H,D/2] i8, sf_q [B,H] i32, weights [B,H] f32
//   kv_cache fused pages [num_blocks, PAGE_KV*(D/2+4)] bytes
//   context_lens [B] i32, block_table [B, max_pages] i32 (physical page ids)
static torch::Tensor mqa_logits_fp4_decode(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv_cache,
    torch::Tensor weights, torch::Tensor context_lens, torch::Tensor block_table,
    int max_context_len, at::ScalarType out_dtype) {
    auto [B, num_blocks, max_pages] = check_paged(q, sf_q, kv_cache, weights,
                                                  context_lens, block_table, out_dtype);
    TORCH_CHECK(max_context_len > 0 && max_context_len <= max_pages * PAGE_KV,
                "max_context_len must be in (0, max_pages*PAGE_KV]");
    const int num_sms = get_num_sms();

    // DG allocation: row stride = align(align(max_ctx, SPLIT_KV), 1024B); RAW rows,
    // no -inf fill (kIsCompressedLogits=false writes the whole scanned range).
    const int align_elems = 1024 / (int)c10::elementSize(out_dtype);
    const int stride_logits = host_align_up(host_align_up(max_context_len, SPLIT_KV), align_elems);
    torch::Tensor logits = torch::empty({B, stride_logits}, q.options().dtype(out_dtype));

    auto meta = get_paged_mqa_logits_metadata(context_lens, num_sms);
    deep_gemm::MainCompressorArgs no_comp{};   // tail idles (allocating wrapper is attention-only)
    dispatch_launch(q, sf_q, kv_cache, weights,
                    context_lens.data_ptr<int>(), block_table.data_ptr<int>(), max_pages,
                    num_blocks, B, stride_logits, out_dtype,
                    meta.data_ptr<int>(), num_sms,
                    logits.data_ptr(),
                    /*comp_eps=*/1e-6f, no_comp, /*attn_mock=*/false);
    return logits.index({torch::indexing::Slice(0, B),
                         torch::indexing::Slice(0, max_context_len)});
}

// Preallocated-output PAGED decode (repo *_out convention; DG-aligned): the timed
// region is exactly DeepGEMM's -- 5 descriptors + ONE kernel launch. schedule_meta
// comes from get_paged_mqa_logits_metadata (precompute it outside timed loops for
// DG-table parity, or per step for end-to-end truth). Writes RAW rows in place.
// Optional cmp_* bundle: fuses the DSV4 MAIN-indexer compressor rows into the tail
// warpgroup (hidden under the KV stream); omit it for the attention-only launch.
static void mqa_logits_fp4_decode_out(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv_cache,
    torch::Tensor weights, torch::Tensor context_lens, torch::Tensor block_table,
    torch::Tensor schedule_meta, torch::Tensor logits,
    c10::optional<torch::Tensor> cmp_pos, c10::optional<torch::Tensor> comp_norm,
    c10::optional<torch::Tensor> cos_tab, c10::optional<torch::Tensor> sin_tab,
    c10::optional<torch::Tensor> comp_kv, c10::optional<torch::Tensor> comp_sc,
    c10::optional<torch::Tensor> comp_q8, c10::optional<torch::Tensor> comp_s8,
    c10::optional<torch::Tensor> comp_rope, double comp_eps, bool mock_attn) {
    auto [B, num_blocks, max_pages] = check_paged(q, sf_q, kv_cache, weights,
                                                  context_lens, block_table,
                                                  logits.scalar_type());
    const int num_sms = get_num_sms();
    TORCH_CHECK(schedule_meta.is_cuda() && schedule_meta.scalar_type() == torch::kInt32
                && schedule_meta.is_contiguous()
                && schedule_meta.sizes() == torch::IntArrayRef({num_sms + 1, 2}),
                "schedule_meta must be [num_sms+1, 2] i32 (get_paged_mqa_logits_metadata)");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= B
                && logits.size(1) % SPLIT_KV == 0 && logits.is_contiguous(),
                "logits must be [>=B, k*SPLIT_KV] contiguous (DG row-stride alignment)");

    deep_gemm::MainCompressorArgs comp{};
    if (comp_kv.has_value()) {
        TORCH_CHECK(cmp_pos && comp_norm && cos_tab && sin_tab && comp_sc
                    && comp_q8 && comp_s8 && comp_rope,
                    "fused compressor needs the full cmp_* bundle");
        TORCH_CHECK((int)cmp_pos->numel() == B, "cmp_pos must be [B] i64");
        comp.pos = reinterpret_cast<const long long*>(cmp_pos->data_ptr<int64_t>());
        comp.norm = comp_norm->data_ptr<float>();
        comp.cos_tab = cos_tab->data_ptr<float>();
        comp.sin_tab = sin_tab->data_ptr<float>();
        comp.kv = comp_kv->data_ptr<float>();
        comp.sc = comp_sc->data_ptr<float>();
        comp.q8 = comp_q8->data_ptr<uint8_t>();
        comp.s8 = comp_s8->data_ptr<float>();
        comp.rope = reinterpret_cast<nv_bfloat16*>(comp_rope->data_ptr());
    }

    dispatch_launch(q, sf_q, kv_cache, weights,
                    context_lens.data_ptr<int>(), block_table.data_ptr<int>(), max_pages,
                    num_blocks, B, (int)logits.size(1), logits.scalar_type(),
                    schedule_meta.data_ptr<int>(), num_sms,
                    logits.data_ptr(),
                    (float)comp_eps, comp, mock_attn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mqa_compressor_standalone",
          [](torch::Tensor cmp_pos, torch::Tensor comp_norm,
             torch::Tensor cos_tab, torch::Tensor sin_tab,
             torch::Tensor comp_kv, torch::Tensor comp_sc,
             torch::Tensor comp_q8, torch::Tensor comp_s8,
             torch::Tensor comp_rope, double eps) {
              deep_gemm::MainCompressorArgs comp{};
              comp.pos = reinterpret_cast<const long long*>(cmp_pos.data_ptr<int64_t>());
              comp.norm = comp_norm.data_ptr<float>();
              comp.cos_tab = cos_tab.data_ptr<float>();
              comp.sin_tab = sin_tab.data_ptr<float>();
              comp.kv = comp_kv.data_ptr<float>();
              comp.sc = comp_sc.data_ptr<float>();
              comp.q8 = comp_q8.data_ptr<uint8_t>();
              comp.s8 = comp_s8.data_ptr<float>();
              comp.rope = reinterpret_cast<nv_bfloat16*>(comp_rope.data_ptr());
              const uint32_t B = (uint32_t)cmp_pos.numel();
              standalone_compressor_kernel<<<B, 128, 0,
                                             at::cuda::getCurrentCUDAStream()>>>(
                  comp, B, (float)eps);
          },
          "standalone MAIN-compressor kernel (DSV4 tail op; also fusable into the"
          " attention kernel's tail warpgroup via mqa_logits_fp4_decode_out cmp_*)",
          py::arg("cmp_pos"), py::arg("comp_norm"), py::arg("cos_tab"), py::arg("sin_tab"),
          py::arg("comp_kv"), py::arg("comp_sc"), py::arg("comp_q8"), py::arg("comp_s8"),
          py::arg("comp_rope"), py::arg("eps") = 1e-6);
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata,
          "DG-aligned schedule metadata kernel: context_lens [B] i32 -> [num_sms+1, 2] i32",
          py::arg("context_lens"), py::arg("num_sms") = 0);
    m.def("mqa_logits_fp4_decode", &mqa_logits_fp4_decode,
          "DSV4 FP4 paged MQA-logits decode, verbatim DeepGEMM sm100 paged kernel "
          "(mxfp4, H=64, D=128, next_n=1, page 64, clean_logits=False semantics); "
          "allocating wrapper that also runs the metadata kernel",
          py::arg("q"), py::arg("sf_q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("max_context_len"),
          py::arg("out_dtype"));
    m.def("mqa_logits_fp4_decode_out", &mqa_logits_fp4_decode_out,
          "DG-aligned preallocated-output decode: 5 descriptors + one launch "
          "(schedule_meta from get_paged_mqa_logits_metadata). Optional cmp_* bundle "
          "fuses the MAIN-indexer compressor into the tail warpgroup; mock_attn=True "
          "benchmarks the tail alone in its in-situ launch shape",
          py::arg("q"), py::arg("sf_q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("logits"),
          py::arg("cmp_pos") = py::none(), py::arg("comp_norm") = py::none(),
          py::arg("cos_tab") = py::none(), py::arg("sin_tab") = py::none(),
          py::arg("comp_kv") = py::none(), py::arg("comp_sc") = py::none(),
          py::arg("comp_q8") = py::none(), py::arg("comp_s8") = py::none(),
          py::arg("comp_rope") = py::none(),
          py::arg("comp_eps") = 1e-6, py::arg("mock_attn") = false);
}
