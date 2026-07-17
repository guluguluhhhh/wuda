// DeepSeek-V4 MHC fused forward for NVIDIA Blackwell/B300.
//
// The projection is a very tall-K, narrow-N BF16 GEMM:
//     [M, 28672] x [24, 28672]^T -> [M, 24].
// A one-SM tcgen05 tile avoids the 8-10x N padding of a 2-SM swap-AB tile.
// Split-K partials stay FP32; the reduction is fused with RMSNorm, gates,
// Sinkhorn, and collapse in the second kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

namespace hc_tc {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

static constexpr int HC = 4;
static constexpr int DIM = 7168;
static constexpr int K_DIM = HC * DIM;
static constexpr int N_OUT = 24;
static constexpr int SINKHORN_ITERS = 20;

static constexpr int BLOCK_K = 64;
static constexpr int UMMA_K = 16;
static constexpr int NUM_K_TILES = K_DIM / BLOCK_K;
static constexpr int NUM_TMEM_COLS = 32;
static constexpr int GEMM_THREADS = 256;
static constexpr int EPILOGUE_THREADS = 256;
static constexpr int MIN_K_TILES_PER_SPLIT = 8;

static_assert(K_DIM % BLOCK_K == 0, "K must be tiled exactly");

namespace ptx {

__device__ __forceinline__ bool elect_one_sync() {
    uint32_t pred;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "elect.sync _|p, 0xffffffff;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ void tcgen05_alloc_1sm(uint32_t smem_addr, uint32_t cols) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_addr), "r"(cols));
}

__device__ __forceinline__ void tcgen05_dealloc_1sm(uint32_t tmem_addr, uint32_t cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_addr), "r"(cols));
}

__device__ __forceinline__ void tcgen05_fence_before_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tcgen05_mma_1sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint64_t runtime_idesc, uint32_t accumulate) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 "
        "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
           "r"(static_cast<uint32_t>(runtime_idesc >> 32)), "r"(accumulate),
           "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

__device__ __forceinline__ void tcgen05_commit_1sm(Barrier* barrier) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    if (elect_one_sync()) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"(addr) : "memory");
    }
}

__device__ __forceinline__ void tmem_load_32dp32b8x(
    uint32_t addr,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3,
    uint32_t& v4, uint32_t& v5, uint32_t& v6, uint32_t& v7) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3),
          "=r"(v4), "=r"(v5), "=r"(v6), "=r"(v7)
        : "r"(addr));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace ptx

namespace mma_desc {

__device__ __forceinline__ cute::UMMA::SmemDescriptor make_k_major(void* ptr) {
    cute::UMMA::SmemDescriptor desc;
    desc.version_ = 1;
    desc.lbo_mode_ = 0;
    desc.layout_type_ = static_cast<uint8_t>(cute::UMMA::LayoutType::SWIZZLE_128B);
    const auto smem_ptr = cute::cast_smem_ptr_to_uint(ptr);
    desc.start_address_ = static_cast<uint16_t>(smem_ptr >> 4);
    constexpr uint32_t stride_bytes = 8u * BLOCK_K * sizeof(__nv_bfloat16);
    desc.stride_byte_offset_ = stride_bytes >> 4;
    desc.leading_byte_offset_ = 0;
    desc.base_offset_ = 0;
    return desc;
}

template <int M_TILE, int N_TILE>
__device__ __forceinline__ uint64_t make_runtime_idesc() {
    auto idesc = cute::UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        M_TILE, N_TILE, cute::UMMA::Major::K, cute::UMMA::Major::K>();
    return cute::UMMA::make_runtime_instr_desc(idesc);
}

__device__ __forceinline__ uint32_t advance_k(uint32_t lo, int kk) {
    return lo + kk * (UMMA_K * sizeof(__nv_bfloat16) / 16);
}

}  // namespace mma_desc

template <int M_TILE, int N_TILE, int STAGES>
struct GemmSharedStorage {
    static constexpr int A_STAGE_ELEMS = M_TILE * BLOCK_K;
    static constexpr int B_STAGE_ELEMS = N_TILE * BLOCK_K;

    alignas(1024) __nv_bfloat16 smem_a[STAGES * A_STAGE_ELEMS];
    alignas(1024) __nv_bfloat16 smem_b[STAGES * B_STAGE_ELEMS];
    alignas(16) Barrier full[STAGES];
    alignas(16) Barrier empty[STAGES];
    alignas(16) Barrier tmem_full;
    alignas(16) uint32_t tmem_base;
};

__device__ __forceinline__ void tma_load_2d(
    const void* desc, Barrier* barrier, void* smem,
    int coord0, int coord1) {
    cute::SM90_TMA_LOAD_2D::copy(
        desc, reinterpret_cast<uint64_t*>(barrier),
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
        smem, coord0, coord1);
}

template <int M_TILE, int N_TILE, int STAGES>
__global__ void __launch_bounds__(GEMM_THREADS, 1)
hc_gemm_splitk_kernel(
    const __grid_constant__ CUtensorMap desc_x,
    const __grid_constant__ CUtensorMap desc_w,
    int num_positions,
    int num_m_tiles,
    int num_n_tiles,
    int num_splits,
    int k_tiles_per_split,
    float* __restrict__ workspace) {
    using Storage = GemmSharedStorage<M_TILE, N_TILE, STAGES>;
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    Storage& s = *reinterpret_cast<Storage*>(smem_raw);

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    int task = static_cast<int>(blockIdx.x);
    const int m_tile = task % num_m_tiles;
    task /= num_m_tiles;
    const int n_tile = task % num_n_tiles;
    const int split = task / num_n_tiles;
    if (split >= num_splits) return;

    const int k_begin = split * k_tiles_per_split;
    const int k_end = min(k_begin + k_tiles_per_split, NUM_K_TILES);
    const int k_count = k_end - k_begin;
    const int m_base = m_tile * M_TILE;
    const int n_base = n_tile * N_TILE;

    if (warp_id == 0 && ptx::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&desc_x);
        cute::prefetch_tma_descriptor(&desc_w);
    }
    if (warp_id == 1 && ptx::elect_one_sync()) {
        #pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
            s.full[stage].init(1);
            s.empty[stage].init(1);
        }
        s.tmem_full.init(1);
        cutlass::arch::fence_barrier_init();
    }
    if (warp_id == 2) {
        const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_1sm(addr, NUM_TMEM_COLS);
    }
    __syncthreads();

    if (warp_id == 0 && ptx::elect_one_sync()) {
        int stage = 0;
        int phase = 0;
        #pragma unroll 1
        for (int kt = k_begin; kt < k_end; ++kt) {
            s.empty[stage].wait(phase ^ 1);
            auto* a_dst = s.smem_a + stage * Storage::A_STAGE_ELEMS;
            auto* b_dst = s.smem_b + stage * Storage::B_STAGE_ELEMS;
            const int k_offset = kt * BLOCK_K;
            tma_load_2d(&desc_x, &s.full[stage], a_dst, k_offset, m_base);
            tma_load_2d(&desc_w, &s.full[stage], b_dst, k_offset, n_base);
            constexpr uint32_t tx_bytes =
                (Storage::A_STAGE_ELEMS + Storage::B_STAGE_ELEMS) * sizeof(__nv_bfloat16);
            s.full[stage].arrive_and_expect_tx(tx_bytes);
            if (++stage == STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }
    } else if (warp_id == 1) {
        const auto a_desc = mma_desc::make_k_major(s.smem_a);
        const auto b_desc = mma_desc::make_k_major(s.smem_b);
        const uint32_t a_stage_lo = (lane_id < STAGES)
            ? a_desc.lo + lane_id * (Storage::A_STAGE_ELEMS * sizeof(__nv_bfloat16) / 16)
            : 0u;
        const uint32_t b_stage_lo = (lane_id < STAGES)
            ? b_desc.lo + lane_id * (Storage::B_STAGE_ELEMS * sizeof(__nv_bfloat16) / 16)
            : 0u;
        const uint64_t runtime_idesc = mma_desc::make_runtime_idesc<M_TILE, N_TILE>();
        const uint32_t tmem_c = s.tmem_base;

        int stage = 0;
        int phase = 0;
        #pragma unroll 1
        for (int ki = 0; ki < k_count; ++ki) {
            s.full[stage].wait(phase);
            ptx::tcgen05_fence_after_sync();
            const uint32_t a_base = __shfl_sync(0xffffffffu, a_stage_lo, stage);
            const uint32_t b_base = __shfl_sync(0xffffffffu, b_stage_lo, stage);
            if (ptx::elect_one_sync()) {
                #pragma unroll
                for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                    const uint32_t a_lo = mma_desc::advance_k(a_base, kk);
                    const uint32_t b_lo = mma_desc::advance_k(b_base, kk);
                    const uint64_t a = (static_cast<uint64_t>(a_desc.hi) << 32) | a_lo;
                    const uint64_t b = (static_cast<uint64_t>(b_desc.hi) << 32) | b_lo;
                    const uint32_t accumulate = (ki != 0 || kk != 0) ? 1u : 0u;
                    ptx::tcgen05_mma_1sm(tmem_c, a, b, runtime_idesc, accumulate);
                }
            }
            __syncwarp();
            ptx::tcgen05_commit_1sm(&s.empty[stage]);
            if (ki == k_count - 1) {
                ptx::tcgen05_commit_1sm(&s.tmem_full);
            }
            __syncwarp();
            if (++stage == STAGES) {
                stage = 0;
                phase ^= 1;
            }
        }
    } else if (warp_id >= 4) {
        const int epi_warp = warp_id - 4;
        // M64 Layout F has four 16-row groups at DP 0/32/64/96; M128
        // Layout D uses all 32 lanes of each datapath partition.
        const int rows_per_warp = M_TILE == 64 ? 16 : 32;
        const int row_local = epi_warp * rows_per_warp + lane_id;
        const int row = m_base + row_local;
        s.tmem_full.wait(0);
        ptx::tcgen05_fence_after_sync();

        #pragma unroll
        for (int ng = 0; ng < N_TILE / 8; ++ng) {
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
            ptx::tmem_load_32dp32b8x(
                s.tmem_base + ng * 8, v0, v1, v2, v3, v4, v5, v6, v7);
            cutlass::arch::fence_view_async_tmem_load();
            if ((M_TILE == 128 || lane_id < 16) && row < num_positions) {
                const uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
                const int col0 = n_base + ng * 8;
                float* dst = workspace
                    + (static_cast<int64_t>(split) * num_positions + row) * N_OUT + col0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (col0 + j < N_OUT) dst[j] = __uint_as_float(vals[j]);
                }
            }
        }
        ptx::tcgen05_fence_before_sync();
    }

    __syncthreads();
    if (warp_id == 2) {
        ptx::tcgen05_dealloc_1sm(s.tmem_base, NUM_TMEM_COLS);
    }
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void __launch_bounds__(EPILOGUE_THREADS, 2)
hc_reduce_and_fuse_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const float* __restrict__ workspace,
    const float* __restrict__ base,
    const float* __restrict__ scale,
    float hc_eps,
    float rms_eps,
    int num_positions,
    int num_splits,
    __nv_bfloat16* __restrict__ collapsed_out,
    float* __restrict__ pre_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out) {
    const int pos = static_cast<int>(blockIdx.x);
    if (pos >= num_positions) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* hidden_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* scratch = reinterpret_cast<float*>(hidden_smem + K_DIM);
    float* warp_sums = scratch;              // 8
    float* rms_smem = warp_sums + 8;         // 1
    float* mix_smem = rms_smem + 1;          // 24
    float* pre_smem = mix_smem + N_OUT;      // 4
    float* post_smem = pre_smem + HC;        // 4
    float* comb_smem = post_smem + HC;       // 16

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const auto* src = hidden_states + static_cast<int64_t>(pos) * K_DIM;

    constexpr int NUM_I4 = K_DIM / 8;
    const int4* src_i4 = reinterpret_cast<const int4*>(src);
    int4* dst_i4 = reinterpret_cast<int4*>(hidden_smem);
    float sq_sum = 0.0f;
    for (int i = tid; i < NUM_I4; i += EPILOGUE_THREADS) {
        const int4 v = src_i4[i];
        dst_i4[i] = v;
        const auto* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(p[j]);
            sq_sum += f.x * f.x + f.y * f.y;
        }
    }
    sq_sum = ptx::warp_sum(sq_sum);
    if (lane_id == 0) warp_sums[warp_id] = sq_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = lane_id < 8 ? warp_sums[lane_id] : 0.0f;
        v = ptx::warp_sum(v);
        if (lane_id == 0) rms_smem[0] = rsqrtf(v / static_cast<float>(K_DIM) + rms_eps);
    }
    __syncthreads();

    if (tid < N_OUT) {
        float sum = 0.0f;
        const int64_t split_stride = static_cast<int64_t>(num_positions) * N_OUT;
        const float* partial = workspace + static_cast<int64_t>(pos) * N_OUT + tid;
        #pragma unroll 1
        for (int split = 0; split < num_splits; ++split) {
            sum += partial[static_cast<int64_t>(split) * split_stride];
        }
        mix_smem[tid] = sum * rms_smem[0];
    }
    __syncthreads();

    if (tid < HC) {
        pre_smem[tid] = fast_sigmoid(mix_smem[tid] * scale[0] + base[tid]) + hc_eps;
        post_smem[tid] = 2.0f * fast_sigmoid(
            mix_smem[HC + tid] * scale[1] + base[HC + tid]);
    }
    if (tid < HC * HC) {
        comb_smem[tid] = mix_smem[2 * HC + tid] * scale[2] + base[2 * HC + tid];
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = lane_id < HC * HC ? comb_smem[lane_id] : 0.0f;
        float max_v = v;
        #pragma unroll
        for (int offset = 1; offset < HC; offset <<= 1) {
            max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffffu, max_v, offset));
        }
        const float e = __expf(v - max_v);
        float row_sum = e;
        #pragma unroll
        for (int offset = 1; offset < HC; offset <<= 1) {
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, offset);
        }
        v = e / row_sum + hc_eps;

        float col_sum = v;
        #pragma unroll
        for (int offset = HC; offset < HC * HC; offset <<= 1) {
            col_sum += __shfl_xor_sync(0xffffffffu, col_sum, offset);
        }
        v /= col_sum + hc_eps;

        #pragma unroll 1
        for (int iter = 0; iter < SINKHORN_ITERS - 1; ++iter) {
            row_sum = v;
            #pragma unroll
            for (int offset = 1; offset < HC; offset <<= 1) {
                row_sum += __shfl_xor_sync(0xffffffffu, row_sum, offset);
            }
            v /= row_sum + hc_eps;
            col_sum = v;
            #pragma unroll
            for (int offset = HC; offset < HC * HC; offset <<= 1) {
                col_sum += __shfl_xor_sync(0xffffffffu, col_sum, offset);
            }
            v /= col_sum + hc_eps;
        }
        if (lane_id < HC * HC) comb_smem[lane_id] = v;
    }

    if (tid < HC) {
        pre_out[pos * HC + tid] = pre_smem[tid];
        post_out[pos * HC + tid] = post_smem[tid];
    }
    if (tid < HC * HC) {
        comb_out[pos * HC * HC + tid] = comb_smem[tid];
    }
    __syncthreads();

    auto* collapsed = collapsed_out + static_cast<int64_t>(pos) * DIM;
    for (int d = tid; d < DIM; d += EPILOGUE_THREADS) {
        float value = 0.0f;
        #pragma unroll
        for (int h = 0; h < HC; ++h) {
            value += pre_smem[h] * __bfloat162float(hidden_smem[h * DIM + d]);
        }
        collapsed[d] = __float2bfloat16_rn(value);
    }
}

struct SplitConfig {
    int m_tile;
    int n_tile;
    int num_m_tiles;
    int num_n_tiles;
    int num_splits;
    int k_tiles_per_split;
    int grid;
    int num_sms;
};

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static int choose_n_tile(int m) {
    if (m <= 4) return 8;
    if (m <= 16) return 16;
    return 32;
}

static int choose_m_tile(int m) {
    return m >= 4096 ? 128 : 64;
}

static SplitConfig make_split_config(int m) {
    int device = 0;
    int num_sms = 0;
    auto err = cudaGetDevice(&device);
    TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
    err = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));

    const int m_tile = choose_m_tile(m);
    const int n_tile = choose_n_tile(m);
    const int num_m_tiles = ceil_div(m, m_tile);
    const int num_n_tiles = ceil_div(N_OUT, n_tile);
    const int mn_tiles = num_m_tiles * num_n_tiles;
    const int target_splits = std::max(ceil_div(num_sms, mn_tiles), 1);
    const int k_tiles_per_split = std::max(
        ceil_div(NUM_K_TILES, target_splits), MIN_K_TILES_PER_SPLIT);
    const int num_splits = ceil_div(NUM_K_TILES, k_tiles_per_split);
    return {m_tile, n_tile, num_m_tiles, num_n_tiles, num_splits,
            k_tiles_per_split, mn_tiles * num_splits, num_sms};
}

static CUtensorMap make_tma_bf16_2d(
    const char* name,
    const __nv_bfloat16* ptr,
    int rows,
    int cols,
    int box_rows) {
    CUtensorMap desc{};
    cuuint64_t global_dims[2] = {
        static_cast<cuuint64_t>(cols), static_cast<cuuint64_t>(rows)};
    cuuint64_t global_strides[1] = {
        static_cast<cuuint64_t>(cols) * sizeof(__nv_bfloat16)};
    cuuint32_t box_dims[2] = {
        static_cast<cuuint32_t>(BLOCK_K), static_cast<cuuint32_t>(box_rows)};
    cuuint32_t elem_strides[2] = {1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16*>(ptr), global_dims, global_strides,
        box_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        TORCH_CHECK(false, "cuTensorMapEncodeTiled(", name, ") failed: ",
                    message ? message : "unknown", " rows=", rows,
                    " cols=", cols, " box_rows=", box_rows);
    }
    return desc;
}

struct TmaCache {
    const void* x_ptr = nullptr;
    const void* w_ptr = nullptr;
    int m = -1;
    int m_tile = -1;
    int n_tile = -1;
    CUtensorMap x{};
    CUtensorMap w{};
};

template <int M_TILE, int N_TILE, int STAGES>
static void launch_gemm(
    const CUtensorMap& desc_x,
    const CUtensorMap& desc_w,
    const SplitConfig& cfg,
    int m,
    float* workspace,
    cudaStream_t stream) {
    using Storage = GemmSharedStorage<M_TILE, N_TILE, STAGES>;
    void* kernel = reinterpret_cast<void*>(&hc_gemm_splitk_kernel<M_TILE, N_TILE, STAGES>);
    const int smem_bytes = static_cast<int>(sizeof(Storage));

    static bool configured = false;
    if (!configured) {
        const auto attr_err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess,
                    "cudaFuncSetAttribute(gemm) failed: ", cudaGetErrorString(attr_err),
                    " smem=", smem_bytes, " tile=", M_TILE, "x", N_TILE,
                    " stages=", STAGES);
        configured = true;
    }

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(cfg.grid, 1, 1);
    config.blockDim = dim3(GEMM_THREADS, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    void* args[] = {
        const_cast<CUtensorMap*>(&desc_x), const_cast<CUtensorMap*>(&desc_w),
        &m, const_cast<int*>(&cfg.num_m_tiles), const_cast<int*>(&cfg.num_n_tiles),
        const_cast<int*>(&cfg.num_splits), const_cast<int*>(&cfg.k_tiles_per_split),
        &workspace};
    const auto launch_err = cudaLaunchKernelExC(&config, kernel, args);
    TORCH_CHECK(launch_err == cudaSuccess,
                "hc tcgen05 GEMM launch failed: ", cudaGetErrorString(launch_err));
}

static std::vector<torch::Tensor> hc_fused_forward_full(
    torch::Tensor hidden_states,
    torch::Tensor attn_hc_fn,
    torch::Tensor attn_hc_base,
    torch::Tensor attn_hc_scale,
    double hc_eps,
    double rms_norm_eps) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16,
                "hidden_states must be bf16");
    TORCH_CHECK(hidden_states.dim() == 2 || hidden_states.dim() == 3,
                "hidden_states must be [HC,DIM] or [M,HC,DIM]");
    if (hidden_states.dim() == 2) {
        TORCH_CHECK(hidden_states.size(0) == HC && hidden_states.size(1) == DIM,
                    "2D hidden_states must be [4,7168]");
    } else {
        TORCH_CHECK(hidden_states.size(1) == HC && hidden_states.size(2) == DIM,
                    "3D hidden_states must be [M,4,7168]");
    }
    TORCH_CHECK(attn_hc_fn.is_cuda() && attn_hc_fn.scalar_type() == torch::kBFloat16,
                "attn_hc_fn must be CUDA bf16");
    TORCH_CHECK(attn_hc_fn.dim() == 2 && attn_hc_fn.size(0) == N_OUT &&
                attn_hc_fn.size(1) == K_DIM,
                "attn_hc_fn must be [24,28672]");
    TORCH_CHECK(attn_hc_base.is_cuda() && attn_hc_base.scalar_type() == torch::kFloat32 &&
                attn_hc_base.numel() == N_OUT,
                "attn_hc_base must be CUDA fp32 [24]");
    TORCH_CHECK(attn_hc_scale.is_cuda() && attn_hc_scale.scalar_type() == torch::kFloat32 &&
                attn_hc_scale.numel() == 3,
                "attn_hc_scale must be CUDA fp32 [3]");
    TORCH_CHECK(hidden_states.get_device() == attn_hc_fn.get_device() &&
                hidden_states.get_device() == attn_hc_base.get_device() &&
                hidden_states.get_device() == attn_hc_scale.get_device(),
                "all inputs must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    auto hs = hidden_states.contiguous().view({-1, K_DIM});
    auto weight = attn_hc_fn.contiguous();
    auto base = attn_hc_base.contiguous();
    auto scale = attn_hc_scale.contiguous();
    const int m = static_cast<int>(hs.size(0));
    TORCH_CHECK(m > 0, "num_positions must be positive");

    const SplitConfig cfg = make_split_config(m);
    auto fp32_opts = hs.options().dtype(torch::kFloat32);
    auto bf16_opts = hs.options().dtype(torch::kBFloat16);
    auto workspace = torch::empty({cfg.num_splits, m, N_OUT}, fp32_opts);
    auto collapsed = torch::empty({m, DIM}, bf16_opts);
    auto pre = torch::empty({m, HC}, fp32_opts);
    auto post = torch::empty({m, HC}, fp32_opts);
    auto comb = torch::empty({m, HC, HC}, fp32_opts);

    const auto* x_ptr = reinterpret_cast<const __nv_bfloat16*>(hs.data_ptr());
    const auto* w_ptr = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    thread_local TmaCache cache;
    if (cache.x_ptr != x_ptr || cache.w_ptr != w_ptr ||
        cache.m != m || cache.m_tile != cfg.m_tile || cache.n_tile != cfg.n_tile) {
        cache.x = make_tma_bf16_2d("hidden", x_ptr, m, K_DIM, cfg.m_tile);
        cache.w = make_tma_bf16_2d("weight", w_ptr, N_OUT, K_DIM, cfg.n_tile);
        cache.x_ptr = x_ptr;
        cache.w_ptr = w_ptr;
        cache.m = m;
        cache.m_tile = cfg.m_tile;
        cache.n_tile = cfg.n_tile;
    }

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    float* workspace_ptr = workspace.data_ptr<float>();
    if (cfg.m_tile == 128) {
        TORCH_CHECK(cfg.n_tile == 32, "M128 path requires N32");
        launch_gemm<128, 32, 8>(cache.x, cache.w, cfg, m, workspace_ptr, stream);
    } else {
        switch (cfg.n_tile) {
            case 8:
                launch_gemm<64, 8, 4>(cache.x, cache.w, cfg, m, workspace_ptr, stream);
                break;
            case 16:
                launch_gemm<64, 16, 4>(cache.x, cache.w, cfg, m, workspace_ptr, stream);
                break;
            case 32:
                if (cfg.k_tiles_per_split <= 16) {
                    launch_gemm<64, 32, 4>(
                        cache.x, cache.w, cfg, m, workspace_ptr, stream);
                } else {
                    launch_gemm<64, 32, 12>(
                        cache.x, cache.w, cfg, m, workspace_ptr, stream);
                }
                break;
            default:
                TORCH_CHECK(false, "unsupported N tile: ", cfg.n_tile);
        }
    }

    constexpr int scratch_floats = 8 + 1 + N_OUT + HC + HC + HC * HC;
    constexpr int fuse_smem_bytes = K_DIM * sizeof(__nv_bfloat16)
        + scratch_floats * sizeof(float);
    static bool fuse_configured = false;
    if (!fuse_configured) {
        const auto attr_err = cudaFuncSetAttribute(
            reinterpret_cast<void*>(&hc_reduce_and_fuse_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, fuse_smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess,
                    "cudaFuncSetAttribute(fuse) failed: ", cudaGetErrorString(attr_err),
                    " smem=", fuse_smem_bytes);
        fuse_configured = true;
    }

    hc_reduce_and_fuse_kernel<<<m, EPILOGUE_THREADS, fuse_smem_bytes, stream>>>(
        x_ptr, workspace_ptr, base.data_ptr<float>(), scale.data_ptr<float>(),
        static_cast<float>(hc_eps), static_cast<float>(rms_norm_eps),
        m, cfg.num_splits,
        reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr()),
        pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>());
    const auto fuse_err = cudaGetLastError();
    TORCH_CHECK(fuse_err == cudaSuccess,
                "HC reduce/fuse launch failed: ", cudaGetErrorString(fuse_err));

    if (hidden_states.dim() == 2) {
        return {collapsed.squeeze(0), pre.squeeze(0), post.squeeze(0), comb.squeeze(0)};
    }
    return {collapsed, pre, post, comb};
}

static std::vector<int64_t> hc_fused_tc_config(int64_t num_positions) {
    TORCH_CHECK(num_positions > 0 && num_positions <= INT32_MAX,
                "num_positions is out of range");
    const SplitConfig cfg = make_split_config(static_cast<int>(num_positions));
    return {cfg.n_tile, cfg.num_splits, cfg.k_tiles_per_split,
            cfg.grid, cfg.num_m_tiles, cfg.num_n_tiles, cfg.num_sms, cfg.m_tile};
}

}  // namespace hc_tc

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_fused_forward_full", &hc_tc::hc_fused_forward_full,
          "MHC fused forward (BF16 tcgen05 + split-K + fused HC epilogue)");
    m.def("hc_fused_tc_config", &hc_tc::hc_fused_tc_config,
          "Return [N tile, split-K, K tiles/split, grid, M tiles, N tiles, SMs, M tile]");
}
