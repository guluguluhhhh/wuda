// ============================================================
// wq_b_proj_gemm_tcgen05.cu
// tcgen05 BF16 GEMM — Kernel + Host + PyTorch Binding
// Aligned with DeepGEMM sm100_bf16_gemm.cuh architecture
//
// M=32~256, K=1536, N=65536, BF16 → FP32 output
// 2SM MMA (cta_group::2), Cluster=(2,1,1), Persistent, Warp-Specialized
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "wq_b_proj_gemm_tcgen05.cuh"

using namespace wq_b;
using Barrier = mma_desc::Barrier;

// ======================== Shared Memory Layout ========================
template <int BLOCK_M_TPL>
struct SharedStorage {
    static constexpr int LOAD_BLOCK_M_T = BLOCK_M_TPL;
    static constexpr int STORE_BLOCK_M_T = BLOCK_M_TPL;  // min(128, BLOCK_M) but BLOCK_M<=128
    static constexpr int NUM_STORE_THREADS_T = STORE_BLOCK_M_T;
    static constexpr int SMEM_A_PER_STAGE_T = LOAD_BLOCK_M_T * BLOCK_K * sizeof(nv_bfloat16);
    static constexpr int SMEM_CD_PER_STAGE_T = STORE_BLOCK_M_T * STORE_BLOCK_N * sizeof(float);
    static constexpr int SMEM_CD_TOTAL_T = SMEM_CD_PER_STAGE_T * NUM_TMA_STORE_STAGES;

    // Dynamic NUM_STAGES: fit within 232448 bytes
    // DeepGEMM formula: (capacity - smem_extra) / smem_per_stage
    // smem_extra = smem_cd + barriers(32*8*3 + 2*8*2 + 8 = 808) + tmem_ptr(4)
    static constexpr int SMEM_PER_STAGE_T = SMEM_A_PER_STAGE_T + SMEM_B_PER_STAGE;
    static constexpr int SMEM_BARRIERS = 32 * 8 * 3 + 2 * 8 * 2 + 8;  // 808
    static constexpr int SMEM_OVERHEAD = SMEM_CD_TOTAL_T + SMEM_BARRIERS + 4;
    static constexpr int NUM_STAGES_COMPUTED = (232448 - SMEM_OVERHEAD) / SMEM_PER_STAGE_T;
    static constexpr int NUM_STAGES_T = (NUM_STAGES_COMPUTED > 8) ? 8 : NUM_STAGES_COMPUTED;

    alignas(1024) uint8_t smem_cd[SMEM_CD_TOTAL_T];
    alignas(1024) uint8_t smem_a[NUM_STAGES_T * SMEM_A_PER_STAGE_T];
    alignas(1024) uint8_t smem_b[NUM_STAGES_T * SMEM_B_PER_STAGE];

    // Barriers
    alignas(16) Barrier full_barriers[NUM_STAGES_T];
    alignas(16) Barrier empty_barriers[NUM_STAGES_T];
    alignas(16) Barrier tmem_full_barriers[NUM_EPI_STAGES];
    alignas(16) Barrier tmem_empty_barriers[NUM_EPI_STAGES];

    // TMEM base address
    alignas(16) uint32_t tmem_base;
};

// ======================== Kernel ========================
template <int BLOCK_M_TPL>
__global__ void __launch_bounds__(TPB, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    const __grid_constant__ CUtensorMap desc_D,
    int M, int N, int K,
    int num_m_blocks, int num_n_tiles, int total_tiles, int num_blocks)
{
    // Local constants derived from template parameter
    constexpr int LOAD_BLOCK_M_T = BLOCK_M_TPL;
    constexpr int STORE_BLOCK_M_T = BLOCK_M_TPL;
    constexpr int NUM_STORE_THREADS_T = STORE_BLOCK_M_T;
    constexpr int SMEM_A_PER_STAGE_T = LOAD_BLOCK_M_T * BLOCK_K * sizeof(nv_bfloat16);

    using Storage = SharedStorage<BLOCK_M_TPL>;
    constexpr int NUM_STAGES_T = Storage::NUM_STAGES_T;
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    Storage& s = *reinterpret_cast<Storage*>(smem_buf);

    const uint32_t warp_id = threadIdx.x / 32;
    const uint32_t lane_id = ptx::get_lane_idx();
    const uint32_t cta_rank = ptx::block_rank_in_cluster();
    const bool is_leader = (cta_rank == 0);

    // ================================================================
    // INITIALIZATION (DeepGEMM L101-L172)
    // ================================================================

    // 1. Cluster sync BEFORE 2-CTA TMEM allocation (DeepGEMM L102)
    ptx::cluster_sync();

    // 2. Prefetch TMA descriptors (DeepGEMM L109-L114)
    if (warp_id == 0) {
        cute::prefetch_tma_descriptor(&desc_A);
        cute::prefetch_tma_descriptor(&desc_B);
        cute::prefetch_tma_descriptor(&desc_D);
    }

    // 3. Initialize barriers (DeepGEMM L148-L167)
    if (warp_id == 1 && ptx::elect_one_sync()) {
        for (int i = 0; i < NUM_STAGES_T; ++i) {
            s.full_barriers[i].init(NUM_MULTICAST);
            s.empty_barriers[i].init(1);
        }
        for (int i = 0; i < NUM_EPI_STAGES; ++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(NUM_MULTICAST * NUM_STORE_THREADS_T);
        }
        cutlass::arch::fence_barrier_init();
    }

    // 4. Allocate TMEM (DeepGEMM L168-L171)
    if (warp_id == 2) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_2sm(addr, NUM_TMEM_COLS);
    }

    // 5. Cluster sync AFTER init (DeepGEMM L172)
    ptx::cluster_sync();

    // Wait for primary kernel completion (DeepGEMM L175)
    cudaGridDependencySynchronize();

    // ================================================================
    // SMEM DESCRIPTOR PRE-COMPUTATION (DeepGEMM warp-shuffle trick)
    // ================================================================
    auto* smem_a_base = reinterpret_cast<nv_bfloat16*>(s.smem_a);
    auto* smem_b_base = reinterpret_cast<nv_bfloat16*>(s.smem_b);

    auto a_desc = mma_desc::make_smem_desc_k_major(smem_a_base);
    auto b_desc = mma_desc::make_smem_desc_k_major(smem_b_base);

    uint32_t a_desc_lo = (lane_id < NUM_STAGES_T)
        ? a_desc.lo + lane_id * (SMEM_A_PER_STAGE_T / 16) : 0u;
    uint32_t b_desc_lo = (lane_id < NUM_STAGES_T)
        ? b_desc.lo + lane_id * (SMEM_B_PER_STAGE / 16) : 0u;

    // Instruction descriptor
    uint64_t runtime_idesc = mma_desc::make_runtime_instr_desc();

    // ================================================================
    // THREE-WAY INDEPENDENT PERSISTENT LOOPS (DeepGEMM architecture)
    // Tile scheduling: DeepGEMM L117-152 get_swizzled_block_idx
    // For kIsMulticastOnA=false: grouping on M, interleave M first
    // m_block_idx = block_idx % num_m_blocks
    // n_block_idx = block_idx / num_m_blocks
    // This makes adjacent blocks share B's L2 cache lines.
    // ================================================================
    int num_clusters = num_blocks / CLUSTER_M;
    int cluster_id = blockIdx.x / CLUSTER_M;
    int num_tiles_total = total_tiles;

    // ======== WARP 0: TMA PRODUCER (independent persistent loop) ========
    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = (stage_idx + 1) % NUM_STAGES_T;
            if (stage_idx == 0) phase ^= 1;
        };

        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            // DeepGEMM swizzle: M-major for L2 reuse of B
            int m_block = tile_id % num_m_blocks;
            int n_tile = tile_id / num_m_blocks;
            int n_offset = n_tile * BLOCK_N + cta_rank * LOAD_BLOCK_N;

            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.empty_barriers[stage_idx].wait(phase ^ 1);

                int k_offset = k * BLOCK_K;
                auto* smem_a_dst = reinterpret_cast<nv_bfloat16*>(
                    s.smem_a + stage_idx * SMEM_A_PER_STAGE_T);
                auto* smem_b_dst = reinterpret_cast<nv_bfloat16*>(
                    s.smem_b + stage_idx * SMEM_B_PER_STAGE);

                tma::copy_2sm_2d(&desc_A, &s.full_barriers[stage_idx],
                                 smem_a_dst, k_offset, m_block * BLOCK_M_TPL);
                tma::copy_2sm_2d(&desc_B, &s.full_barriers[stage_idx],
                                 smem_b_dst, k_offset, n_offset);

                constexpr uint32_t kNumArrivalBytes = SMEM_A_PER_STAGE_T + SMEM_B_PER_STAGE;
                if (is_leader) {
                    s.full_barriers[stage_idx].arrive_and_expect_tx(
                        kNumArrivalBytes * NUM_MULTICAST);
                } else {
                    s.full_barriers[stage_idx].arrive(0u);
                }

                advance_pipeline();
            }
        }
    }

    // ======== WARP 1: MMA CONSUMER (independent persistent loop, leader only) ========
    else if (warp_id == 1 && is_leader) {
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = (stage_idx + 1) % NUM_STAGES_T;
            if (stage_idx == 0) phase ^= 1;
        };

        uint32_t persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;

            // Wait for TMEM to be empty
            s.tmem_empty_barriers[accum_stage].wait(accum_phase ^ 1);
            ptx::tcgen05_fence_after_sync();

            uint32_t tmem_c = accum_stage * UMMA_N;

            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_fence_after_sync();

                uint32_t a_base = __shfl_sync(0xffffffff, a_desc_lo, stage_idx);
                uint32_t b_base = __shfl_sync(0xffffffff, b_desc_lo, stage_idx);

                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                        uint32_t a_lo = mma_desc::advance_desc_lo_for_k(a_base, kk);
                        uint32_t b_lo = mma_desc::advance_desc_lo_for_k(b_base, kk);

                        uint64_t a_full = (static_cast<uint64_t>(a_desc.hi) << 32) | a_lo;
                        uint64_t b_full = (static_cast<uint64_t>(b_desc.hi) << 32) | b_lo;

                        uint32_t accum_flag = (k > 0 || kk > 0) ? 1 : 0;
                        ptx::tcgen05_mma_2sm(tmem_c, a_full, b_full,
                                             runtime_idesc, accum_flag);
                    }
                }
                __syncwarp();

                // Signal MMA done + optional TMEM full
                constexpr uint16_t CTA_MASK = (1 << NUM_MULTICAST) - 1;
                ptx::umma_arrive_multicast_2sm(
                    reinterpret_cast<uint64_t*>(&s.empty_barriers[stage_idx]), CTA_MASK);
                if (k == NUM_K_TILES - 1) {
                    ptx::umma_arrive_multicast_2sm(
                        reinterpret_cast<uint64_t*>(&s.tmem_full_barriers[accum_stage]),
                        CTA_MASK);
                }
                __syncwarp();

                advance_pipeline();
            }

            persistent_iter++;
        }

        // Wait last epilogue before dealloc (DeepGEMM L392-397)
        if (persistent_iter > 0) {
            uint32_t last_iter = persistent_iter - 1;
            uint32_t last_accum_stage = last_iter % NUM_EPI_STAGES;
            uint32_t last_accum_phase = (last_iter / NUM_EPI_STAGES) & 1;
            s.tmem_empty_barriers[last_accum_stage].wait(last_accum_phase);
        }
    }

    // ======== EPILOGUE WARPS (independent persistent loop) ========
    else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
             warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS_T) / 32) {
        uint32_t epi_warp_idx = warp_id - (NUM_NON_EPI_THREADS / 32);
        uint32_t tma_store_idx = 0;

        uint32_t persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            // DeepGEMM swizzle: M-major for L2 reuse of B
            int m_block = tile_id % num_m_blocks;
            int n_tile = tile_id / num_m_blocks;

            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;

            // Wait for MMA to fill TMEM
            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_fence_after_sync();

            uint32_t tmem_base_addr = accum_stage * UMMA_N;
            int out_m = m_block * BLOCK_M_TPL;
            int out_n = n_tile * BLOCK_N;

            constexpr int NUM_N_STORES = BLOCK_N / STORE_BLOCK_N;
            constexpr int NUM_BANK_GROUP_BYTES = 16;
            constexpr int ELEMS_PER_BG = NUM_BANK_GROUP_BYTES / sizeof(float);

            for (int ns = 0; ns < NUM_N_STORES; ++ns, tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES) {
                auto* smem_cd_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * Storage::SMEM_CD_PER_STAGE_T);

                if (epi_warp_idx == 0) {
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                }
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS_T, 0);

                for (int i = 0; i < STORE_BLOCK_N / ELEMS_PER_BG; ++i) {
                    uint32_t tmem_addr = tmem_base_addr + ns * STORE_BLOCK_N + i * ELEMS_PER_BG;

                    uint32_t v0, v1, v2, v3;
                    ptx::tmem_load_32dp32b4x(tmem_addr, v0, v1, v2, v3);
                    cutlass::arch::fence_view_async_tmem_load();

                    uint32_t bank_group_idx = i + lane_id * (SWIZZLE_CD / NUM_BANK_GROUP_BYTES);
                    constexpr bool kHasShortcut = (SWIZZLE_CD / NUM_BANK_GROUP_BYTES) == 8;
                    uint32_t row = kHasShortcut ? (i / 8 + lane_id) : (bank_group_idx / 8);
                    uint32_t col = kHasShortcut ? i : (bank_group_idx % 8);
                    col ^= row % (SWIZZLE_CD / 16);

                    auto* smem_dst = smem_cd_ptr
                        + epi_warp_idx * 32 * SWIZZLE_CD
                        + row * (NUM_BANK_GROUP_BYTES * 8) + col * NUM_BANK_GROUP_BYTES;

                    ptx::st_shared_v4(smem_dst, v0, v1, v2, v3);
                }

                if (ns == NUM_N_STORES - 1) {
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS_T, 0);

                if (epi_warp_idx == 0 && ptx::elect_one_sync()) {
                    int store_n = out_n + ns * STORE_BLOCK_N;
                    tma::store_2d(&desc_D, smem_cd_ptr, store_n, out_m);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

            persistent_iter++;
        }
    }

    // ================================================================
    // CLEANUP (DeepGEMM L454-L456)
    // ================================================================
    ptx::cluster_sync();
    if (warp_id == 0) {
        ptx::tcgen05_dealloc_2sm(0, NUM_TMEM_COLS);
    }
}

// ======================== Host: TMA Descriptor ========================
static CUtensorMap make_tma_desc_bf16_2d(
    const nv_bfloat16* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2] = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(nv_bfloat16)};
    uint32_t boxDim[2] = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2] = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

static CUtensorMap make_tma_desc_fp32_2d(
    const float* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2] = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(float)};
    uint32_t boxDim[2] = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2] = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

// ======================== PyTorch Binding ========================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);

    const int M = x.size(0);
    TORCH_CHECK(x.size(1) == K_DIM);
    TORCH_CHECK(w.size(0) == N_TOTAL && w.size(1) == K_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0);

    // Select BLOCK_M based on M (DeepGEMM sm100.hpp L62-64)
    // M<=64 -> BLOCK_M=64, M>64 -> BLOCK_M=128
    const int block_m = (M <= 64) ? 64 : 128;
    const int M_padded = ((M + block_m - 1) / block_m) * block_m;

    torch::Tensor x_padded = x;
    if (M < M_padded) {
        x_padded = torch::zeros({M_padded, K_DIM}, x.options());
        x_padded.slice(0, 0, M).copy_(x);
    }

    auto out = torch::empty({M_padded, N_TOTAL}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_ptr = reinterpret_cast<const nv_bfloat16*>(x_padded.data_ptr());
    auto w_ptr = reinterpret_cast<const nv_bfloat16*>(w.data_ptr());
    auto out_ptr = reinterpret_cast<float*>(out.data_ptr());

    // TMA descriptors
    const int load_block_m = block_m;  // = BLOCK_M_TPL
    CUtensorMap desc_A = make_tma_desc_bf16_2d(x_ptr, M_padded, K_DIM, load_block_m, BLOCK_K);
    CUtensorMap desc_B = make_tma_desc_bf16_2d(w_ptr, N_TOTAL, K_DIM, LOAD_BLOCK_N, BLOCK_K);
    const int store_block_m = block_m;  // = STORE_BLOCK_M_T
    CUtensorMap desc_D = make_tma_desc_fp32_2d(out_ptr, M_padded, N_TOTAL, store_block_m, STORE_BLOCK_N);

    // Grid: persistent
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    int num_m_blocks = M_padded / block_m;
    int num_n_tiles = N_TOTAL / BLOCK_N;
    int total_tiles_val = num_m_blocks * num_n_tiles;
    // grid_size must be multiple of cluster_dim.x=2
    int grid_size = min(num_SMs, total_tiles_val);
    grid_size = (grid_size / 2) * 2;  // round down to multiple of 2
    if (grid_size < 2) grid_size = 2;

    int smem_bytes = (block_m == 64)
        ? static_cast<int>(sizeof(SharedStorage<64>))
        : static_cast<int>(sizeof(SharedStorage<128>));
    auto kernel_64 = &wq_b_proj_kernel<64>;
    auto kernel_128 = &wq_b_proj_kernel<128>;
    void* kernel_ptr = (block_m == 64) ? (void*)kernel_64 : (void*)kernel_128;

    auto attr_err = cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(attr_err),
                " smem_bytes=", smem_bytes);

    // Launch with cluster=(2,1,1)
    {
        dim3 grid(grid_size, 1, 1);
        dim3 block(TPB, 1, 1);

        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem_bytes;
        config.stream = stream;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = 2;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;

        // Use cudaLaunchKernelExC (C API) to pass CUtensorMap by value
        int M_val = M_padded;
        int N_val = N_TOTAL;
        int K_val = K_DIM;
        void* ptr_args[] = {
            &desc_A, &desc_B, &desc_D,
            &M_val, &N_val, &K_val,
            &num_m_blocks, &num_n_tiles, &total_tiles_val, &grid_size
        };
        auto err = cudaLaunchKernelExC(
            &config,
            kernel_ptr,
            ptr_args);
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    }

    return out.slice(0, 0, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj (tcgen05 2SM MMA, DeepGEMM-style, Blackwell)");
}
