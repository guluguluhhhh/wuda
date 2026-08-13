// DSV4 FP8 paged MQA logits with the MAIN compressor fused into an independent
// tail warpgroup. The FP4 implementation remains in mqa_logits_fp4.cu.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "mqa_logits_fp8.cuh"

namespace {

constexpr int kNumHeads = 64;
constexpr int kHeadDim = 128;
constexpr int kBlockQ = 2;
constexpr int kSplitKV = 256;
constexpr int kSplitsPerChunk = 16;
constexpr int kNumQStages = 3;
constexpr int kNumSpecializedThreads = 128;
constexpr int kNumMathThreads = 256;
constexpr int kNumTailThreads = 128;
constexpr int kThreads = 512;

static int get_num_sms() {
    int device = 0, sms = 0;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                       device) == cudaSuccess,
                "cudaDeviceGetAttribute(MultiProcessorCount) failed");
    return sms;
}

static CUtensorMapSwizzle to_swizzle(int bytes) {
    switch (bytes) {
        case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default: return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

static CUtensorMap make_tma_2d(const char* name, void* pointer,
                               CUtensorMapDataType dtype, int element_size,
                               int global_inner, int global_outer,
                               int shared_inner, int shared_outer,
                               int global_outer_stride_elements,
                               int swizzle_bytes) {
    if (swizzle_bytes != 0)
        shared_inner = swizzle_bytes / element_size;
    CUtensorMap map{};
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(global_inner),
        static_cast<cuuint64_t>(global_outer)};
    cuuint32_t shared_dims[2] = {
        static_cast<cuuint32_t>(shared_inner),
        static_cast<cuuint32_t>(shared_outer)};
    cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(global_outer_stride_elements) * element_size};
    cuuint32_t element_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &map, dtype, 2, pointer, global_dims, global_strides, shared_dims,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        to_swizzle(swizzle_bytes), CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    message ? message : "unknown");
    }
    return map;
}

static CUtensorMap make_tma_3d(const char* name, void* pointer,
                               CUtensorMapDataType dtype, int element_size,
                               int global_0, int global_1, int global_2,
                               int shared_0, int shared_1, int shared_2,
                               int row_stride_elements,
                               int64_t page_stride_bytes,
                               int swizzle_bytes) {
    if (swizzle_bytes != 0)
        shared_0 = swizzle_bytes / element_size;
    CUtensorMap map{};
    cuuint64_t global_dims[3] = {
        static_cast<cuuint64_t>(global_0),
        static_cast<cuuint64_t>(global_1),
        static_cast<cuuint64_t>(global_2)};
    cuuint32_t shared_dims[3] = {
        static_cast<cuuint32_t>(shared_0),
        static_cast<cuuint32_t>(shared_1),
        static_cast<cuuint32_t>(shared_2)};
    cuuint64_t global_strides[2] = {
        static_cast<cuuint64_t>(row_stride_elements) * element_size,
        static_cast<cuuint64_t>(page_stride_bytes)};
    cuuint32_t element_strides[3] = {1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &map, dtype, 3, pointer, global_dims, global_strides, shared_dims,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        to_swizzle(swizzle_bytes), CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    message ? message : "unknown");
    }
    return map;
}

template <int kKVStages>
static int shared_memory_bytes() {
    using Storage = deep_gemm::layout::MQALogitsSharedStorage<
        kNumHeads, kHeadDim, false, kBlockQ, kSplitKV,
        kNumQStages, kKVStages, 3, cutlass::float_e4m3_t, float>;
    return static_cast<int>(sizeof(Storage));
}

template <int kKVStages, int kPageKV>
static void launch_typed(
    int batch, int logits_stride, int block_table_stride,
    const int* context_lens, float* logits, const int* block_table,
    const int* schedule_meta,
    const wuda_fp8_mqa::MainCompressorArgs& compressor,
    const wuda_fp8_mqa::QueryRmsRopeArgs& query,
    float compressor_eps,
    const CUtensorMap& q_map, const CUtensorMap& sf_q_map,
    const CUtensorMap& kv_map, const CUtensorMap& sf_kv_map,
    const CUtensorMap& weights_map, cudaStream_t stream, bool pdl) {
    using KernelPtr = void (*)(
        uint32_t, uint32_t, uint32_t, const uint32_t*, float*,
        const uint32_t*, const uint32_t*, const uint32_t*,
        wuda_fp8_mqa::MainCompressorArgs, float,
        wuda_fp8_mqa::QueryRmsRopeArgs,
        cute::TmaDescriptor, cute::TmaDescriptor, cute::TmaDescriptor,
        cute::TmaDescriptor, cute::TmaDescriptor);
    KernelPtr kernel;
    const bool query_ablation = query.work_flag != nullptr;
    if (query.x == nullptr) {
        kernel = &deep_gemm::sm100_fp8_paged_mqa_logits_fused<
            1, kNumHeads, kHeadDim, kPageKV,
            true, false, kNumQStages, kKVStages,
            kSplitKV, kSplitsPerChunk,
            kNumSpecializedThreads, kNumMathThreads, kNumTailThreads,
            float, kNumMathThreads / 128, false, false, false>;
    } else if (pdl) {
        kernel = query_ablation
            ? &deep_gemm::sm100_fp8_paged_mqa_logits_fused<
                1, kNumHeads, kHeadDim, kPageKV,
                true, false, kNumQStages, kKVStages,
                kSplitKV, kSplitsPerChunk,
                kNumSpecializedThreads, kNumMathThreads, kNumTailThreads,
                float, kNumMathThreads / 128, true, true, true>
            : &deep_gemm::sm100_fp8_paged_mqa_logits_fused<
                1, kNumHeads, kHeadDim, kPageKV,
                true, false, kNumQStages, kKVStages,
                kSplitKV, kSplitsPerChunk,
                kNumSpecializedThreads, kNumMathThreads, kNumTailThreads,
                float, kNumMathThreads / 128, true, true, false>;
    } else {
        kernel = query_ablation
            ? &deep_gemm::sm100_fp8_paged_mqa_logits_fused<
                1, kNumHeads, kHeadDim, kPageKV,
                true, false, kNumQStages, kKVStages,
                kSplitKV, kSplitsPerChunk,
                kNumSpecializedThreads, kNumMathThreads, kNumTailThreads,
                float, kNumMathThreads / 128, true, false, true>
            : &deep_gemm::sm100_fp8_paged_mqa_logits_fused<
                1, kNumHeads, kHeadDim, kPageKV,
                true, false, kNumQStages, kKVStages,
                kSplitKV, kSplitsPerChunk,
                kNumSpecializedThreads, kNumMathThreads, kNumTailThreads,
                float, kNumMathThreads / 128, true, false, false>;
    }

    constexpr int kMaxDevices = 16;
    static bool configured[kMaxDevices][2][2][2] = {};
    int device = 0;
    TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    TORCH_CHECK(device >= 0 && device < kMaxDevices,
                "device ordinal exceeds local configuration cache");
    const int smem = shared_memory_bytes<kKVStages>();
    const int query_idx = query.x != nullptr ? 1 : 0;
    const int ablation_idx = query_ablation ? 1 : 0;
    const int wait_idx = query.x != nullptr && pdl ? 1 : 0;
    if (!configured[device][query_idx][ablation_idx][wait_idx]) {
        const cudaError_t attr = cudaFuncSetAttribute(
            reinterpret_cast<void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        TORCH_CHECK(attr == cudaSuccess, "cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(attr), " smem=", smem);
        configured[device][query_idx][ablation_idx][wait_idx] = true;
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(get_num_sms(), 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    if (pdl) {
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
    }
    const cudaError_t launch = cudaLaunchKernelEx(
        &config, kernel,
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(logits_stride),
        static_cast<uint32_t>(block_table_stride),
        reinterpret_cast<const uint32_t*>(context_lens), logits,
        reinterpret_cast<const uint32_t*>(block_table),
        static_cast<const uint32_t*>(nullptr),
        reinterpret_cast<const uint32_t*>(schedule_meta),
        compressor, compressor_eps, query,
        q_map, sf_q_map, kv_map, sf_kv_map, weights_map);
    TORCH_CHECK(launch == cudaSuccess, "mqa_logits_fp8 launch failed: ",
                cudaGetErrorString(launch));
}

static wuda_fp8_mqa::MainCompressorArgs make_compressor_args(
    int attention_batch,
    const c10::optional<torch::Tensor>& cmp_pos,
    const c10::optional<torch::Tensor>& comp_norm,
    const c10::optional<torch::Tensor>& cos_tab,
    const c10::optional<torch::Tensor>& sin_tab,
    const c10::optional<torch::Tensor>& comp_state,
    const c10::optional<torch::Tensor>& comp_state_row,
    int64_t comp_state_ring_entries,
    const c10::optional<torch::Tensor>& comp_q8,
    const c10::optional<torch::Tensor>& comp_s8,
    const c10::optional<torch::Tensor>& comp_rope,
    const c10::optional<torch::Tensor>& cmp_cache,
    const c10::optional<torch::Tensor>& cmp_dst,
    int64_t cmp_entries_per_block,
    int64_t cmp_block_stride_bytes) {
    wuda_fp8_mqa::MainCompressorArgs args{};
    if (!comp_state.has_value())
        return args;

    const int64_t rows = cmp_pos.has_value() ? cmp_pos->numel() : 0;
    const bool cache_mode = cmp_cache.has_value() && cmp_cache->numel() > 0;
    TORCH_CHECK(cmp_pos && comp_norm && cos_tab && sin_tab && comp_state_row
                && (cache_mode || (comp_q8 && comp_s8 && comp_rope)),
                "MAIN compressor tensors must be provided together");
    auto check = [](const torch::Tensor& tensor, at::ScalarType dtype,
                    const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
                    && tensor.scalar_type() == dtype,
                    name, " must be contiguous CUDA with the expected dtype");
    };
    check(*cmp_pos, torch::kInt64, "cmp_pos");
    check(*comp_norm, torch::kFloat32, "comp_norm");
    check(*cos_tab, torch::kFloat32, "cos_tab");
    check(*sin_tab, torch::kFloat32, "sin_tab");
    check(*comp_state, torch::kFloat32, "comp_state");
    check(*comp_state_row, torch::kInt64, "comp_state_row");
    TORCH_CHECK(rows >= attention_batch && comp_norm->numel() == 512
                && cos_tab->dim() == 2 && cos_tab->size(1) == 32
                && sin_tab->sizes() == cos_tab->sizes()
                && comp_state->dim() >= 2 && comp_state->size(-1) == 2048
                && comp_state_row->numel() >= rows
                && comp_state_ring_entries >= 8,
                "invalid MAIN compressor shapes or ring geometry");

    args.pos = reinterpret_cast<const long long*>(cmp_pos->data_ptr<int64_t>());
    args.norm = comp_norm->data_ptr<float>();
    args.cos_tab = cos_tab->data_ptr<float>();
    args.sin_tab = sin_tab->data_ptr<float>();
    args.state = comp_state->data_ptr<float>();
    args.state_row = comp_state_row->data_ptr<int64_t>();
    args.state_ring_entries = static_cast<int>(comp_state_ring_entries);
    args.seq_len = static_cast<uint32_t>(rows);

    if (comp_q8.has_value() && comp_q8->numel() > 0) {
        check(*comp_q8, torch::kUInt8, "comp_q8");
        TORCH_CHECK(comp_q8->numel() >= rows * 448, "comp_q8 is too small");
        args.q8 = comp_q8->data_ptr<uint8_t>();
    }
    if (comp_s8.has_value() && comp_s8->numel() > 0) {
        check(*comp_s8, torch::kFloat32, "comp_s8");
        TORCH_CHECK(comp_s8->numel() >= rows * 7, "comp_s8 is too small");
        args.s8 = comp_s8->data_ptr<float>();
    }
    if (comp_rope.has_value() && comp_rope->numel() > 0) {
        check(*comp_rope, torch::kBFloat16, "comp_rope");
        TORCH_CHECK(comp_rope->numel() >= rows * 64, "comp_rope is too small");
        args.rope = reinterpret_cast<nv_bfloat16*>(comp_rope->data_ptr());
    }

    if (cache_mode) {
        TORCH_CHECK(cmp_dst && cmp_entries_per_block > 0,
                    "cmp_cache requires cmp_dst and entries_per_block");
        check(*cmp_cache, torch::kUInt8, "cmp_cache");
        check(*cmp_dst, torch::kInt64, "cmp_dst");
        const int64_t payload = cmp_entries_per_block
            * (wuda_fp8_mqa::kM1TokenBodyBytes
               + wuda_fp8_mqa::kM1ScaleRecordBytes);
        TORCH_CHECK(cmp_block_stride_bytes >= payload
                    && cmp_cache->numel() % cmp_block_stride_bytes == 0
                    && cmp_dst->numel() >= rows,
                    "invalid MAIN compressed-cache geometry");
        args.cmp_cache = cmp_cache->data_ptr<uint8_t>();
        args.cmp_dst = cmp_dst->data_ptr<int64_t>();
        args.cmp_entries_per_block = static_cast<int>(cmp_entries_per_block);
        args.cmp_block_stride_bytes = static_cast<size_t>(cmp_block_stride_bytes);
    }
    return args;
}

static wuda_fp8_mqa::QueryRmsRopeArgs make_query_args(
    const c10::optional<torch::Tensor>& query_x,
    const c10::optional<torch::Tensor>& query_positions,
    const c10::optional<torch::Tensor>& query_cos,
    const c10::optional<torch::Tensor>& query_sin,
    const c10::optional<torch::Tensor>& query_out,
    int64_t query_input_heads, double query_eps,
    const c10::optional<torch::Tensor>& query_work_flag) {
    wuda_fp8_mqa::QueryRmsRopeArgs args{};
    if (!query_x.has_value())
        return args;

    TORCH_CHECK(query_positions && query_cos && query_sin && query_out,
                "query RMSNorm+RoPE tensors must be provided together");
    auto check = [](const torch::Tensor& tensor, at::ScalarType dtype,
                    const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous()
                    && tensor.scalar_type() == dtype,
                    name, " must be contiguous CUDA with the expected dtype");
    };
    check(*query_x, torch::kBFloat16, "query_x");
    check(*query_positions, torch::kInt64, "query_positions");
    check(*query_cos, torch::kFloat32, "query_cos");
    check(*query_sin, torch::kFloat32, "query_sin");
    check(*query_out, torch::kBFloat16, "query_out");
    TORCH_CHECK(query_input_heads > 0
                && query_x->numel() % (query_input_heads * 512) == 0,
                "query_x must contain complete [B,input_heads,512] rows");
    const int64_t output_batch = query_positions->numel();
    TORCH_CHECK(output_batch > 0
                && query_out->numel() % (output_batch * 512) == 0,
                "query_out must contain complete [B,output_heads,512] rows");
    const int64_t input_batch =
        query_x->numel() / (query_input_heads * 512);
    const int64_t output_heads = query_out->numel() / (output_batch * 512);
    TORCH_CHECK(output_batch <= input_batch && output_heads > 0
                && output_heads % query_input_heads == 0,
                "invalid fused query batch/head geometry");
    TORCH_CHECK((output_heads == 64 || output_heads == 128)
                && (query_input_heads == 64 || query_input_heads == 128),
                "fused MODEL1 query requires output_heads=64 or 128 and "
                "query_input_heads=64 or 128");
    TORCH_CHECK(output_batch <= UINT32_MAX / output_heads,
                "fused query row count exceeds uint32 range");
    TORCH_CHECK(query_cos->dim() == 2 && query_cos->size(1) == 32
                && query_sin->sizes() == query_cos->sizes(),
                "query cos/sin tables must be float32 [max_pos,32]");
    TORCH_CHECK(query_x->get_device() == query_out->get_device()
                && query_positions->get_device() == query_x->get_device()
                && query_cos->get_device() == query_x->get_device()
                && query_sin->get_device() == query_x->get_device(),
                "fused query tensors must be on one device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(query_x->data_ptr()) % 16 == 0
                && reinterpret_cast<uintptr_t>(query_out->data_ptr()) % 16 == 0,
                "query_x and query_out must be 16-byte aligned");

    args.x = reinterpret_cast<const nv_bfloat16*>(query_x->data_ptr());
    args.positions = reinterpret_cast<const long long*>(
        query_positions->data_ptr<int64_t>());
    args.cos_tab = query_cos->data_ptr<float>();
    args.sin_tab = query_sin->data_ptr<float>();
    args.out = reinterpret_cast<nv_bfloat16*>(query_out->data_ptr());
    args.input_heads = static_cast<int>(query_input_heads);
    args.output_heads = static_cast<int>(output_heads);
    args.output_rows = static_cast<uint32_t>(output_batch * output_heads);
    args.eps = static_cast<float>(query_eps);
    if (query_work_flag.has_value()) {
        check(*query_work_flag, torch::kInt32, "query_work_flag");
        TORCH_CHECK(query_work_flag->numel() == 1
                    && query_work_flag->get_device() == query_x->get_device(),
                    "query_work_flag must be a CUDA int32 scalar on query_x device");
        args.work_flag = query_work_flag->data_ptr<int>();
    }
    return args;
}

static void mqa_logits_fp8_decode_out(
    torch::Tensor q, torch::Tensor kv_cache, torch::Tensor weights,
    torch::Tensor context_lens, torch::Tensor block_table,
    torch::Tensor schedule_meta, torch::Tensor logits,
    int64_t kv_entries_per_block, int64_t kv_block_stride_bytes,
    int64_t num_kv_stages, double comp_eps,
    c10::optional<torch::Tensor> cmp_pos,
    c10::optional<torch::Tensor> comp_norm,
    c10::optional<torch::Tensor> cos_tab,
    c10::optional<torch::Tensor> sin_tab,
    c10::optional<torch::Tensor> comp_state,
    c10::optional<torch::Tensor> comp_state_row,
    int64_t comp_state_ring_entries,
    c10::optional<torch::Tensor> comp_q8,
    c10::optional<torch::Tensor> comp_s8,
    c10::optional<torch::Tensor> comp_rope,
    c10::optional<torch::Tensor> cmp_cache,
    c10::optional<torch::Tensor> cmp_dst,
    int64_t cmp_entries_per_block,
    int64_t cmp_block_stride_bytes,
    c10::optional<torch::Tensor> query_x,
    c10::optional<torch::Tensor> query_positions,
    c10::optional<torch::Tensor> query_cos,
    c10::optional<torch::Tensor> query_sin,
    c10::optional<torch::Tensor> query_out,
    int64_t query_input_heads,
    double query_eps,
    c10::optional<torch::Tensor> query_work_flag,
    bool pdl) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous()
                && q.scalar_type() == torch::kFloat8_e4m3fn
                && q.dim() == 3 && q.size(1) == kNumHeads
                && q.size(2) == kHeadDim,
                "q must be contiguous CUDA FP8 [B,64,128]");
    const int batch = static_cast<int>(q.size(0));
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous()
                && weights.scalar_type() == torch::kFloat32
                && weights.sizes() == torch::IntArrayRef({batch, kNumHeads}),
                "weights must be contiguous CUDA FP32 [B,64]");
    TORCH_CHECK(context_lens.is_cuda() && context_lens.is_contiguous()
                && context_lens.scalar_type() == torch::kInt32
                && context_lens.numel() == batch,
                "context_lens must be contiguous CUDA int32 [B] or [B,1]");
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous()
                && block_table.scalar_type() == torch::kInt32
                && block_table.dim() == 2 && block_table.size(0) == batch,
                "block_table must be contiguous CUDA int32 [B,max_pages]");
    TORCH_CHECK(schedule_meta.is_cuda() && schedule_meta.is_contiguous()
                && schedule_meta.scalar_type() == torch::kInt32
                && schedule_meta.dim() == 2 && schedule_meta.size(0) == get_num_sms() + 1
                && schedule_meta.size(1) == 2,
                "schedule_meta must be int32 [num_sms+1,2]");
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous()
                && logits.scalar_type() == torch::kFloat32
                && logits.dim() == 2 && logits.size(0) >= batch
                && logits.size(1) % kSplitKV == 0,
                "logits must be contiguous CUDA FP32 [>=B,k*256]");
    TORCH_CHECK(kv_entries_per_block == 32 || kv_entries_per_block == 64
                || kv_entries_per_block == 128,
                "kv_entries_per_block must be 32, 64, or 128");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.is_contiguous()
                && kv_cache.scalar_type() == torch::kUInt8
                && kv_cache.dim() == 2,
                "kv_cache must be contiguous CUDA uint8 physical pages");
    if (kv_block_stride_bytes == 0)
        kv_block_stride_bytes = kv_cache.size(1);
    const int64_t kv_payload = kv_entries_per_block * (kHeadDim + 4);
    TORCH_CHECK(kv_block_stride_bytes >= kv_payload
                && kv_block_stride_bytes % 16 == 0
                && kv_cache.size(1) == kv_block_stride_bytes,
                "invalid FP8 KV page stride");
    TORCH_CHECK(num_kv_stages == 0 || num_kv_stages == 3
                || num_kv_stages == 4 || num_kv_stages == 5,
                "num_kv_stages must be 0 or one of 3/4/5");
    if (num_kv_stages == 0) {
        // Use only host-known geometry: reading context_lens.max() here would
        // insert a decode-path device synchronization. Short schedules prefer
        // shallower pipelines; long schedules match DeepGEMM's five stages.
        const int64_t total_splits = static_cast<int64_t>(batch)
            * ((logits.stride(0) + kSplitKV - 1) / kSplitKV);
        const int64_t splits_per_cta =
            (total_splits + get_num_sms() - 1) / get_num_sms();
        num_kv_stages = splits_per_cta <= 1 ? 3
                      : (splits_per_cta <= 8 ? 4 : 5);
    }

    const int num_blocks = static_cast<int>(kv_cache.size(0));
    const int block_table_stride = static_cast<int>(block_table.stride(0));
    const int logits_stride = static_cast<int>(logits.stride(0));
    const auto compressor = make_compressor_args(
        batch, cmp_pos, comp_norm, cos_tab, sin_tab, comp_state,
        comp_state_row, comp_state_ring_entries, comp_q8, comp_s8,
        comp_rope, cmp_cache, cmp_dst, cmp_entries_per_block,
        cmp_block_stride_bytes);
    const auto query = make_query_args(
        query_x, query_positions, query_cos, query_sin, query_out,
        query_input_heads, query_eps, query_work_flag);

    CUtensorMap q_map = make_tma_2d(
        "q", q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        kHeadDim, batch * kNumHeads, kHeadDim, kBlockQ * kNumHeads,
        static_cast<int>(q.stride(1)), 128);
    CUtensorMap kv_map = make_tma_3d(
        "kv", kv_cache.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        kHeadDim, static_cast<int>(kv_entries_per_block), num_blocks,
        kHeadDim, static_cast<int>(kv_entries_per_block), 1,
        kHeadDim, kv_block_stride_bytes, 128);
    void* scale_base = kv_cache.data_ptr<uint8_t>()
        + kv_entries_per_block * kHeadDim;
    CUtensorMap sf_kv_map = make_tma_2d(
        "sf_kv", scale_base, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
        static_cast<int>(kv_entries_per_block), num_blocks,
        static_cast<int>(kv_entries_per_block), 1,
        static_cast<int>(kv_block_stride_bytes / 4), 0);
    CUtensorMap weights_map = make_tma_2d(
        "weights", weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
        kNumHeads, batch, kNumHeads, kBlockQ,
        static_cast<int>(weights.stride(0)), 0);
    const CUtensorMap sf_q_map = sf_kv_map;  // FP8 core does not consume Q scales.

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_PAGE(STAGES, PAGE) \
    launch_typed<STAGES, PAGE>(batch, logits_stride, block_table_stride, \
        context_lens.data_ptr<int>(), logits.data_ptr<float>(), \
        block_table.data_ptr<int>(), schedule_meta.data_ptr<int>(), \
        compressor, query, static_cast<float>(comp_eps), q_map, sf_q_map, \
        kv_map, sf_kv_map, weights_map, stream, pdl)
#define DISPATCH_PAGE(STAGES) \
    do { \
        if (kv_entries_per_block == 32) LAUNCH_PAGE(STAGES, 32); \
        else if (kv_entries_per_block == 64) LAUNCH_PAGE(STAGES, 64); \
        else LAUNCH_PAGE(STAGES, 128); \
    } while (0)
    if (num_kv_stages == 3) DISPATCH_PAGE(3);
    else if (num_kv_stages == 4) DISPATCH_PAGE(4);
    else DISPATCH_PAGE(5);
#undef DISPATCH_PAGE
#undef LAUNCH_PAGE
}

__global__ void standalone_compressor_kernel(
    wuda_fp8_mqa::MainCompressorArgs compressor, float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= compressor.seq_len)
        return;
    const long long position = compressor.pos[row];
    if (((position + 1) & 3) == 0)
        wuda_fp8_mqa::run_main_compressor_row(
            compressor, row, position, threadIdx.x >> 5,
            threadIdx.x & 31, eps, /*barrier_id=*/0);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "mqa_logits_fp8_decode_out", &mqa_logits_fp8_decode_out,
        "FP8 paged MQA logits with optional fused MAIN compressor",
        py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
        py::arg("context_lens"), py::arg("block_table"),
        py::arg("schedule_meta"), py::arg("logits"),
        py::arg("kv_entries_per_block") = 64,
        py::arg("kv_block_stride_bytes") = 0,
        py::arg("num_kv_stages") = 0,
        py::arg("comp_eps") = 1e-6,
        py::arg("cmp_pos") = c10::nullopt,
        py::arg("comp_norm") = c10::nullopt,
        py::arg("cos_tab") = c10::nullopt,
        py::arg("sin_tab") = c10::nullopt,
        py::arg("comp_state") = c10::nullopt,
        py::arg("comp_state_row") = c10::nullopt,
        py::arg("comp_state_ring_entries") = 0,
        py::arg("comp_q8") = c10::nullopt,
        py::arg("comp_s8") = c10::nullopt,
        py::arg("comp_rope") = c10::nullopt,
        py::arg("cmp_cache") = c10::nullopt,
        py::arg("cmp_dst") = c10::nullopt,
        py::arg("cmp_entries_per_block") = 0,
        py::arg("cmp_block_stride_bytes") = 0,
        py::arg("query_x") = c10::nullopt,
        py::arg("query_positions") = c10::nullopt,
        py::arg("query_cos") = c10::nullopt,
        py::arg("query_sin") = c10::nullopt,
        py::arg("query_out") = c10::nullopt,
        py::arg("query_input_heads") = 0,
        py::arg("query_eps") = 1e-6,
        py::arg("query_work_flag") = c10::nullopt,
        py::arg("pdl") = false);

    module.def(
        "mqa_compressor_fp8_standalone",
        [](torch::Tensor cmp_pos, torch::Tensor comp_norm,
           torch::Tensor cos_tab, torch::Tensor sin_tab,
           torch::Tensor comp_state, torch::Tensor comp_state_row,
           int64_t comp_state_ring_entries,
           torch::Tensor comp_q8, torch::Tensor comp_s8,
           torch::Tensor comp_rope, double eps) {
            const auto args = make_compressor_args(
                static_cast<int>(cmp_pos.numel()), cmp_pos, comp_norm,
                cos_tab, sin_tab, comp_state, comp_state_row,
                comp_state_ring_entries, comp_q8, comp_s8, comp_rope,
                c10::nullopt, c10::nullopt, 0, 0);
            standalone_compressor_kernel<<<
                static_cast<uint32_t>(cmp_pos.numel()), 128, 0,
                at::cuda::getCurrentCUDAStream()>>>(args,
                                                    static_cast<float>(eps));
        },
        py::arg("cmp_pos"), py::arg("comp_norm"), py::arg("cos_tab"),
        py::arg("sin_tab"), py::arg("comp_state"),
        py::arg("comp_state_row"), py::arg("comp_state_ring_entries"),
        py::arg("comp_q8"), py::arg("comp_s8"), py::arg("comp_rope"),
        py::arg("eps") = 1e-6);
}
