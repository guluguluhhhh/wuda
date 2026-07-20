// ============================================================
// mqa_logits_fp4.cu — Host launcher + PyTorch binding for the migrated
// DeepGEMM FP4 MQA-logits (DSV4 Sparse Top-K Indexer score attention).
//
// Kernel body + helper closure are inlined self-contained (only CUTLASS/CuTe) in
// megakernel/include/mqa_logits_fp4.cuh (modulo 2 documented AOT edits).
//
// Two entry points:
//   mqa_logits_fp4         — faithful single-sequence, NON-compressed. Returns RAW
//                            logits [S, Skv]; caller must mask outside [ks,ke) (the
//                            kernel writes real values across the scanned union, like
//                            DeepGEMM before its clean_logits pass). Used for golden
//                            parity vs ref_fp8_mqa_logits.
//   mqa_logits_fp4_decode  — MULTI-BATCH decode, COMPRESSED + self-clean. Packs B
//                            decode tokens as seq_len=B and the contiguous
//                            idx_kv_cache [B,T,D] as one flat kv[B*T,D]; per-token
//                            KV window ks[b]=b*T, ke[b]=b*T+valid_b restricts each
//                            token to its own batch. ONE launch, grid.x = #SMs:
//                            the kernel's tile-pool scheduler balances the global
//                            Σ_b cdiv(window_b,256) KV tiles across all CTAs
//                            (chunks may cross token boundaries), so B < #SMs and
//                            mixed context lengths both fill the machine.
//                            Output [B, T]; invalid tail (>= valid_b) is -inf.
//
// Host TMA setup mirrors DeepGEMM smxx_fp8_fp4_mqa_logits.hpp + attention.hpp;
// launch pattern mirrors kernels/w1_merged_fp8_gemm.cu.
// ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/bfloat16.h>

#include "mqa_logits_fp4.cuh"   // self-contained kernel (only CUTLASS/CuTe)

namespace {

using namespace mqa_logits_fp4;

static int host_align_up(int a, int b) { return (a + b - 1) / b * b; }

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

// TMA-aligned length (16-byte alignment for the SF vector). Mirrors DeepGEMM
// utils/math.hpp::get_tma_aligned_size.
static int get_tma_aligned_size(int x, int element_size) {
    constexpr int kNumTMAAlignmentBytes = 16;
    TORCH_CHECK(kNumTMAAlignmentBytes % element_size == 0, "bad SF element size");
    return host_align_up(x, kNumTMAAlignmentBytes / element_size);
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

// Per-stage / total dynamic shared-memory bytes (mirrors the wrapper).
static int compute_smem_bytes() {
    const int smem_q   = BLOCK_Q * NUM_HEADS * (HEAD_DIM / 2);            // 8192
    const int smem_sfq = host_align_up(BLOCK_Q * NUM_HEADS, 128) * 4;     // 512
    const int smem_kv  = BLOCK_KV * (HEAD_DIM / 2);                        // 16384
    const int smem_sfkv= host_align_up(BLOCK_KV, 128) * 4;                // 1024
    const int smem_w   = BLOCK_Q * NUM_HEADS * 4;                          // 512
    const int barriers = (NUM_Q_STAGES + NUM_KV_STAGES + NUM_TMEM_STAGES) * 2 * 8;
    const int tmem_ptr = 4;
    return NUM_Q_STAGES * (smem_q + smem_sfq + smem_w) +
           NUM_KV_STAGES * (smem_kv + smem_sfkv) + barriers + tmem_ptr;
}

template <typename logits_dtype_t, bool kCompressed, bool kTilePool>
static void launch_typed(int seq_len, int seq_len_kv, int stride_logits,
                         const int* ks, const int* ke, void* logits,
                         const CUtensorMap& dQ, const CUtensorMap& dSFQ,
                         const CUtensorMap& dKV, const CUtensorMap& dSFKV,
                         const CUtensorMap& dW, dim3 grid, int smem,
                         cudaStream_t stream) {
    auto kernel = &deep_gemm::sm100_fp4_mqa_logits<
        NUM_HEADS, HEAD_DIM, kCompressed, kTilePool,
        BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SPECIALIZED_THREADS, NUM_MATH_THREADS, logits_dtype_t>;

    static bool configured = false;   // per-instantiation (template static local)
    if (!configured) {
        auto e = cudaFuncSetAttribute((void*)kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute: ", cudaGetErrorString(e),
                    " smem=", smem);
        configured = true;
    }

    kernel<<<grid, dim3(TPB, 1, 1), smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv, /*max_seqlen_k (unused)=*/0u,
        (uint32_t)stride_logits,
        reinterpret_cast<const uint32_t*>(ks),
        reinterpret_cast<const uint32_t*>(ke),
        reinterpret_cast<logits_dtype_t*>(logits),
        dQ, dSFQ, dKV, dSFKV, dW);
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mqa_logits_fp4 launch failed: ", cudaGetErrorString(err));
}

// Build the 5 TMA descriptors + grid, then dispatch (out_dtype x schedule) and
// launch INTO a caller-provided logits buffer `lp`. No allocation here (so callers
// can hoist it out of a timed loop, matching the repo's *_out convention).
// `decode_tile_pool` selects the decode schedule (compressed + global tile pool,
// grid.x = num_ctas CTAs); false = faithful single-seq legacy schedule.
//   q [S,H,D/2] i8, sf_q [S,H] i32, kv [Skv,D/2] i8, sf_kv [Skv] i32, weights [S,H] f32
static void dispatch_launch(torch::Tensor q, torch::Tensor sf_q,
                            torch::Tensor kv, torch::Tensor sf_kv, torch::Tensor weights,
                            const int* ks, const int* ke, int seq_len, int seq_len_kv,
                            at::ScalarType out_dtype, bool decode_tile_pool, int stride_logits,
                            int num_ctas, void* lp) {
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
                                 D, seq_len * H, D, BLOCK_Q * H,
                                 (int)q.stride(1), D / 2, /*is_fp4=*/true, /*unpacked=*/false);
    CUtensorMap dKV = make_tma_2d("kv", kv.data_ptr(), FP4_DT, q_elem,
                                  D, seq_len_kv, D, BLOCK_KV,
                                  (int)kv.stride(0), D / 2, /*is_fp4=*/true, /*unpacked=*/false);
    CUtensorMap dSFQ = make_tma_2d("sf_q", sf_q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, sf_elem,
                                   H, seq_len, H, BLOCK_Q, (int)sf_q.stride(0), 0);
    CUtensorMap dW = make_tma_2d("weights", weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                                 H, seq_len, H, BLOCK_Q, (int)weights.stride(0), 0);
    CUtensorMap dSFKV = make_tma_2d("sf_kv", sf_kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_INT32, sf_elem,
                                    get_tma_aligned_size(seq_len_kv, sf_elem), 1, BLOCK_KV, 1, 0, 0);

    // Grid:
    //  - faithful single-seq (legacy): one CTA per q-block, grid.y = 1 (no KV carve).
    //  - decode tile pool: grid.x CTAs (default = #SMs) split the global KV tile pool;
    //    empty chunks exit immediately, so grid.x never over-subscribes.
    dim3 grid(1, 1, 1);
    if (decode_tile_pool) {
        if (num_ctas < 1) num_ctas = get_num_sms();
        grid.x = (unsigned)num_ctas;
    } else {
        grid.x = (unsigned)(host_align_up(seq_len, BLOCK_Q) / BLOCK_Q);
    }
    const int smem = compute_smem_bytes();

    // Only the two used (schedule x compression) combos are instantiated:
    // decode = compressed + tile pool; faithful = raw + legacy.
    if (out_dtype == torch::kFloat) {
        if (decode_tile_pool) launch_typed<float, true,  true >(seq_len, seq_len_kv, stride_logits, ks, ke, lp, dQ,dSFQ,dKV,dSFKV,dW, grid, smem, stream);
        else                  launch_typed<float, false, false>(seq_len, seq_len_kv, stride_logits, ks, ke, lp, dQ,dSFQ,dKV,dSFKV,dW, grid, smem, stream);
    } else {
        if (decode_tile_pool) launch_typed<cutlass::bfloat16_t, true,  true >(seq_len, seq_len_kv, stride_logits, ks, ke, lp, dQ,dSFQ,dKV,dSFKV,dW, grid, smem, stream);
        else                  launch_typed<cutlass::bfloat16_t, false, false>(seq_len, seq_len_kv, stride_logits, ks, ke, lp, dQ,dSFQ,dKV,dSFKV,dW, grid, smem, stream);
    }
}

// Allocating wrapper: makes the padded [align(seq_len,BLOCK_Q), stride_logits] buffer
// (decode/compressed pre-filled -inf; faithful RAW) then dispatch_launch into it.
static torch::Tensor run_mqa(torch::Tensor q, torch::Tensor sf_q,
                             torch::Tensor kv, torch::Tensor sf_kv,
                             torch::Tensor weights, const int* ks, const int* ke,
                             int seq_len, int seq_len_kv, at::ScalarType out_dtype,
                             bool decode_tile_pool, int stride_logits, int num_ctas) {
    const int aligned_seq_len = host_align_up(seq_len, BLOCK_Q);
    torch::Tensor logits_buf = decode_tile_pool
        ? torch::full({aligned_seq_len, stride_logits},
                      -std::numeric_limits<float>::infinity(), q.options().dtype(out_dtype))
        : torch::empty({aligned_seq_len, stride_logits}, q.options().dtype(out_dtype));
    dispatch_launch(q, sf_q, kv, sf_kv, weights, ks, ke, seq_len, seq_len_kv,
                    out_dtype, decode_tile_pool, stride_logits, num_ctas, logits_buf.data_ptr());
    return logits_buf;
}

static void check_qkv(const torch::Tensor& q, const torch::Tensor& sf_q,
                      const torch::Tensor& kv, const torch::Tensor& sf_kv,
                      const torch::Tensor& weights, at::ScalarType out_dtype) {
    constexpr int H = NUM_HEADS, D = HEAD_DIM;
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kInt8, "q must be CUDA int8-packed fp4");
    TORCH_CHECK(kv.is_cuda() && kv.scalar_type() == torch::kInt8, "kv must be CUDA int8-packed fp4");
    TORCH_CHECK(sf_q.is_cuda() && sf_q.scalar_type() == torch::kInt32, "sf_q must be CUDA int32");
    TORCH_CHECK(sf_kv.is_cuda() && sf_kv.scalar_type() == torch::kInt32, "sf_kv must be CUDA int32");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat, "weights must be CUDA float32");
    TORCH_CHECK(out_dtype == torch::kFloat || out_dtype == torch::kBFloat16, "out_dtype float/bf16");
    TORCH_CHECK(q.size(-1) == D / 2 && kv.size(-1) == D / 2, "last dim must be head_dim/2=", D/2);
    (void)H;
}

}  // namespace

// ======================== PyTorch bindings ========================

// Faithful single-sequence, NON-compressed. Returns RAW logits [S, Skv]
// (caller masks outside [ks,ke); matches DeepGEMM before clean_logits).
static torch::Tensor mqa_logits_fp4_forward(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv, torch::Tensor sf_kv,
    torch::Tensor weights, torch::Tensor cu_seq_len_k_start,
    torch::Tensor cu_seq_len_k_end, at::ScalarType out_dtype) {
    check_qkv(q, sf_q, kv, sf_kv, weights, out_dtype);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == NUM_HEADS, "q must be [S,H,D/2]");
    TORCH_CHECK(kv.dim() == 2, "kv must be [Skv,D/2]");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous");
    const int seq_len = q.size(0), seq_len_kv = kv.size(0);
    TORCH_CHECK(sf_q.sizes() == torch::IntArrayRef({seq_len, NUM_HEADS}), "sf_q [S,H]");
    TORCH_CHECK(sf_kv.dim() == 1 && sf_kv.size(0) == seq_len_kv, "sf_kv [Skv]");
    TORCH_CHECK(weights.sizes() == torch::IntArrayRef({seq_len, NUM_HEADS}) && weights.stride(1) == 1, "weights [S,H]");
    TORCH_CHECK(cu_seq_len_k_start.scalar_type() == torch::kInt32 && cu_seq_len_k_start.numel() == seq_len, "ks [S] i32");
    TORCH_CHECK(cu_seq_len_k_end.scalar_type() == torch::kInt32 && cu_seq_len_k_end.numel() == seq_len, "ke [S] i32");

    const int stride_logits = host_align_up(seq_len_kv + BLOCK_KV, 8);
    // Faithful single-sequence path: legacy per-q-block schedule, behavior unchanged.
    auto buf = run_mqa(q, sf_q, kv, sf_kv, weights,
                       cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
                       seq_len, seq_len_kv, out_dtype, /*decode_tile_pool=*/false, stride_logits,
                       /*num_ctas=*/0);
    return buf.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});
}

// Multi-batch decode, COMPRESSED + self-clean. ONE launch for all B tokens;
// grid.x = num_ctas (default #SMs), tile-pool schedule inside the kernel.
//   q [B,H,D/2] i8, sf_q [B,H] i32, kv [B,T,D/2] i8, sf_kv [B,T] i32, weights [B,H] f32
//   valid_len [B] i32 optional (per-batch valid KV length; default T).
// Returns logits [B, T]; entries >= valid_b are -inf.
static torch::Tensor mqa_logits_fp4_decode(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv, torch::Tensor sf_kv,
    torch::Tensor weights, c10::optional<torch::Tensor> valid_len,
    at::ScalarType out_dtype, int num_ctas = 0) {
    check_qkv(q, sf_q, kv, sf_kv, weights, out_dtype);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == NUM_HEADS, "q must be [B,H,D/2]");
    TORCH_CHECK(kv.dim() == 3, "kv must be [B,T,D/2]");
    const int B = q.size(0), T = kv.size(1);
    TORCH_CHECK(kv.size(0) == B, "kv batch must match q");
    TORCH_CHECK(sf_q.sizes() == torch::IntArrayRef({B, NUM_HEADS}), "sf_q [B,H]");
    TORCH_CHECK(sf_kv.sizes() == torch::IntArrayRef({B, T}), "sf_kv [B,T]");
    TORCH_CHECK(weights.sizes() == torch::IntArrayRef({B, NUM_HEADS}) && weights.stride(1) == 1, "weights [B,H]");

    // Flatten the contiguous [B,T,*] indexer cache into one varlen kv[B*T,*].
    auto q_c  = q.contiguous();
    auto kv_c = kv.reshape({B * T, HEAD_DIM / 2}).contiguous();
    auto sfkv_c = sf_kv.reshape({B * T}).contiguous();
    auto sfq_c  = sf_q.contiguous();
    auto w_c    = weights.contiguous();

    // Per-token KV window: token b sees kv[b*T : b*T + valid_b].
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
    torch::Tensor ks = torch::arange(B, i32) * T;                 // [B]
    torch::Tensor ke;
    if (valid_len.has_value()) {
        auto vl = valid_len.value();
        TORCH_CHECK(vl.scalar_type() == torch::kInt32 && vl.numel() == B, "valid_len [B] i32");
        ke = ks + vl.to(q.device());
    } else {
        ke = ks + T;
    }

    const int seq_len = B, seq_len_kv = B * T;
    const int stride_logits = host_align_up(T, BLOCK_KV);   // compressed: align(max_seqlen_k, block_kv)
    // Tile-pool schedule: the kernel balances Σ_b cdiv(window_b, BLOCK_KV) global KV
    // tiles across num_ctas CTAs (default #SMs); chunks may cross token boundaries,
    // so B < #SMs and mixed valid_len both keep every SM busy (imbalance <= 1 tile).
    auto buf = run_mqa(q_c, sfq_c, kv_c, sfkv_c, w_c,
                       ks.data_ptr<int>(), ke.data_ptr<int>(),
                       seq_len, seq_len_kv, out_dtype, /*decode_tile_pool=*/true, stride_logits,
                       num_ctas);
    return buf.index({torch::indexing::Slice(0, B), torch::indexing::Slice(0, T)});
}

// Preallocated-output decode (repo *_out convention): the timed region is just
// reshape(views) + 5 descriptors + launch — no per-call alloc/-inf-fill/arange.
// Caller provides: ks/ke [B] i32 (precomputed), and `logits` preallocated as
// [align(B,BLOCK_Q), align(T,BLOCK_KV)] pre-filled with -inf ONCE (the kernel only
// overwrites each row's [0,valid) so the -inf tail persists across reuse).
// Writes into `logits` in place; caller slices [:B, :T].
static void mqa_logits_fp4_decode_out(
    torch::Tensor q, torch::Tensor sf_q, torch::Tensor kv, torch::Tensor sf_kv,
    torch::Tensor weights, torch::Tensor ks, torch::Tensor ke,
    torch::Tensor logits, int num_ctas) {
    check_qkv(q, sf_q, kv, sf_kv, weights, logits.scalar_type());
    const int B = q.size(0), T = kv.size(1);
    TORCH_CHECK(kv.dim() == 3 && kv.size(0) == B, "kv must be [B,T,D/2]");
    TORCH_CHECK(ks.scalar_type() == torch::kInt32 && ks.numel() == B, "ks [B] i32");
    TORCH_CHECK(ke.scalar_type() == torch::kInt32 && ke.numel() == B, "ke [B] i32");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) >= host_align_up(B, BLOCK_Q),
                "logits must be [>=align(B,BLOCK_Q), stride]");
    TORCH_CHECK(num_ctas >= 0, "num_ctas must be >= 0 (0 = one CTA per SM)");

    auto kv_c   = kv.reshape({B * T, HEAD_DIM / 2});   // view (cache is contiguous)
    auto sfkv_c = sf_kv.reshape({B * T});
    dispatch_launch(q, sf_q, kv_c, sfkv_c, weights,
                    ks.data_ptr<int>(), ke.data_ptr<int>(),
                    /*seq_len=*/B, /*seq_len_kv=*/B * T, logits.scalar_type(),
                    /*decode_tile_pool=*/true, /*stride_logits=*/(int)logits.size(1),
                    num_ctas, logits.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mqa_logits_fp4", &mqa_logits_fp4_forward,
          "DSV4 FP4 MQA-logits (single-seq, non-compressed RAW logits) — migrated from DeepGEMM",
          py::arg("q"), py::arg("sf_q"), py::arg("kv"), py::arg("sf_kv"),
          py::arg("weights"), py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("out_dtype"));
    m.def("mqa_logits_fp4_decode", &mqa_logits_fp4_decode,
          "DSV4 FP4 MQA-logits (multi-batch decode, compressed, one launch, tile-pool schedule) — "
          "varlen packing of [B,T,D] cache; num_ctas=0 -> one CTA per SM",
          py::arg("q"), py::arg("sf_q"), py::arg("kv"), py::arg("sf_kv"),
          py::arg("weights"), py::arg("valid_len") = c10::nullopt,
          py::arg("out_dtype"), py::arg("num_ctas") = 0);
    m.def("mqa_logits_fp4_decode_out", &mqa_logits_fp4_decode_out,
          "DSV4 FP4 MQA-logits decode into a preallocated buffer (repo *_out convention; "
          "hoists alloc/-inf-fill/arange out of the timed path); num_ctas=0 -> one CTA per SM",
          py::arg("q"), py::arg("sf_q"), py::arg("kv"), py::arg("sf_kv"), py::arg("weights"),
          py::arg("ks"), py::arg("ke"), py::arg("logits"), py::arg("num_ctas") = 0);
}
