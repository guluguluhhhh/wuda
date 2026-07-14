// ============================================================
// wq_b_gemm.cu
// tcgen05 BF16 GEMM — Kernel + Host + PyTorch Binding (SWAP-AB)
// Aligned with DeepGEMM sm100_bf16_gemm.cuh (swap_ab path)
//
// M=32~256 (32-aligned), K=1536, N=65536, BF16 -> FP32 output
// swap_ab=1: UMMA_M=256 along N, UMMA_N=M along M.
// 2SM MMA (cta_group::2), Cluster=(2,1,1) [cluster_n=2],
// kIsMulticastOnA=true, Persistent, Warp-Specialized.
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "wq_b_gemm.cuh"

using namespace wq_b;
using Barrier = mma_desc::Barrier;

// ======================== Shared Memory Layout ========================
template <int M_TPL>
struct SharedStorage {
    using D = SwapDims<M_TPL>;
    static constexpr int NUM_STAGES_T       = D::NUM_STAGES;
    static constexpr int SMEM_A_PER_STAGE_T = D::SMEM_A_PER_STAGE;

    alignas(1024) uint8_t smem_cd[SMEM_CD_TOTAL];
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
template <int M_TPL, bool kProfile>
__global__ void __launch_bounds__(TPB, 1)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,   // activation [M, K], K-major
    const __grid_constant__ CUtensorMap desc_B,   // weight     [N, K], K-major
    const __grid_constant__ CUtensorMap desc_D,   // output     [M, N], FP32 row-major
    int num_blocks,
    int64_t* prof)                                // clock64 timing buffer (nullptr if disabled)
{
    using Dims = SwapDims<M_TPL>;
    constexpr int NUM_STAGES_T       = Dims::NUM_STAGES;
    constexpr int SMEM_A_PER_STAGE_T = Dims::SMEM_A_PER_STAGE;
    constexpr int LOAD_BLOCK_M_T     = Dims::LOAD_BLOCK_M;   // M/2
    constexpr int UMMA_N_T           = Dims::UMMA_N;         // M
    constexpr int NUM_TMEM_COLS_T    = Dims::NUM_TMEM_COLS;  // 2*M

    constexpr int NUM_TILES_TOTAL = NUM_N_TILES; // single M block -> tiles = N tiles

    using Storage = SharedStorage<M_TPL>;
    extern __shared__ __align__(1024) uint8_t smem_buf[];
    Storage& s = *reinterpret_cast<Storage*>(smem_buf);

    const uint32_t warp_id  = threadIdx.x / 32;
    const uint32_t lane_id  = ptx::get_lane_idx();
    const uint32_t cta_rank = ptx::block_rank_in_cluster();
    const bool is_leader    = (cta_rank == 0);

    // ================================================================
    // INITIALIZATION
    // ================================================================
    ptx::cluster_sync();

    if (warp_id == 0) {
        cute::prefetch_tma_descriptor(&desc_A);
        cute::prefetch_tma_descriptor(&desc_B);
        cute::prefetch_tma_descriptor(&desc_D);
    }

    if (warp_id == 1 && ptx::elect_one_sync()) {
        for (int i = 0; i < NUM_STAGES_T; ++i) {
            s.full_barriers[i].init(NUM_MULTICAST);   // arrivals from both CTAs (+ tx to leader)
            s.empty_barriers[i].init(1);
        }
        for (int i = 0; i < NUM_EPI_STAGES; ++i) {
            s.tmem_full_barriers[i].init(1);
            s.tmem_empty_barriers[i].init(NUM_MULTICAST * NUM_STORE_THREADS);
        }
        cutlass::arch::fence_barrier_init();
    }

    if (warp_id == 2) {
        uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(&s.tmem_base));
        ptx::tcgen05_alloc_2sm(addr, NUM_TMEM_COLS_T);
    }

    ptx::cluster_sync();
    cudaGridDependencySynchronize();

    // ================================================================
    // SMEM DESCRIPTOR PRE-COMPUTATION (warp-shuffle trick)
    // swap-AB: MMA A-operand = weight (smem_b), B-operand = activation (smem_a)
    // ================================================================
    auto* smem_a_base = reinterpret_cast<nv_bfloat16*>(s.smem_a);   // activation
    auto* smem_b_base = reinterpret_cast<nv_bfloat16*>(s.smem_b);   // weight

    auto act_desc = mma_desc::make_smem_desc_k_major(smem_a_base);  // B-operand
    auto wgt_desc = mma_desc::make_smem_desc_k_major(smem_b_base);  // A-operand

    uint32_t act_desc_lo = (lane_id < NUM_STAGES_T)
        ? act_desc.lo + lane_id * (SMEM_A_PER_STAGE_T / 16) : 0u;
    uint32_t wgt_desc_lo = (lane_id < NUM_STAGES_T)
        ? wgt_desc.lo + lane_id * (SMEM_B_PER_STAGE / 16) : 0u;

    uint64_t runtime_idesc = mma_desc::make_runtime_instr_desc<UMMA_N_T>();

    // ================================================================
    // Persistent tile scheduling (single M block; iterate N tiles)
    // Each cluster processes CLUSTER_BLOCK_N (=256) columns of N per tile,
    // CTA rank r owns N[tile*256 + r*128 : +128].
    // ================================================================
    int num_clusters      = num_blocks / CLUSTER_SIZE;
    int cluster_id        = blockIdx.x / CLUSTER_SIZE;
    int num_tiles_total   = NUM_TILES_TOTAL;

    // ======== WARP 0: TMA PRODUCER (both CTAs) ========
    if (warp_id == 0 && ptx::elect_one_sync()) {
        uint32_t stage_idx = 0, phase = 0;
        auto advance_pipeline = [&]() {
            stage_idx = (stage_idx + 1) % NUM_STAGES_T;
            if (stage_idx == 0) phase ^= 1;
        };

        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            // Weight N base for this CTA (split across the 2 CTAs).
            int n_base = tile_id * CLUSTER_BLOCK_N + cta_rank * LOAD_BLOCK_N;
            // Activation M base for this CTA (split across the 2 CTAs; single M block).
            int m_base = cta_rank * LOAD_BLOCK_M_T;

            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.empty_barriers[stage_idx].wait(phase ^ 1);

                int k_offset = k * BLOCK_K;
                auto* smem_a_dst = reinterpret_cast<nv_bfloat16*>(
                    s.smem_a + stage_idx * SMEM_A_PER_STAGE_T);
                auto* smem_b_dst = reinterpret_cast<nv_bfloat16*>(
                    s.smem_b + stage_idx * SMEM_B_PER_STAGE);

                // activation (A): outer = m_base ; weight (B): outer = n_base
                tma::copy_2sm_2d(&desc_A, &s.full_barriers[stage_idx],
                                 smem_a_dst, k_offset, m_base);
                tma::copy_2sm_2d(&desc_B, &s.full_barriers[stage_idx],
                                 smem_b_dst, k_offset, n_base);

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

    // ======== WARP 1: MMA CONSUMER (leader only) ========
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

            s.tmem_empty_barriers[accum_stage].wait(accum_phase ^ 1);
            ptx::tcgen05_fence_after_sync();

            // [PROFILE] MMA warp: start of this iteration's issue window.
            long long prof_mma_t0 = 0;
            if (kProfile && cluster_id == 0 && lane_id == 0)
                prof_mma_t0 = ptx::rdclock();

            uint32_t tmem_c = accum_stage * UMMA_N_T;

            for (int k = 0; k < NUM_K_TILES; ++k) {
                s.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_fence_after_sync();

                uint32_t w_base = __shfl_sync(0xffffffff, wgt_desc_lo, stage_idx);
                uint32_t a_base = __shfl_sync(0xffffffff, act_desc_lo, stage_idx);

                if (ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int kk = 0; kk < BLOCK_K / UMMA_K; ++kk) {
                        uint32_t w_lo = mma_desc::advance_desc_lo_for_k(w_base, kk);
                        uint32_t a_lo = mma_desc::advance_desc_lo_for_k(a_base, kk);

                        uint64_t w_full = (static_cast<uint64_t>(wgt_desc.hi) << 32) | w_lo;
                        uint64_t a_full = (static_cast<uint64_t>(act_desc.hi) << 32) | a_lo;

                        uint32_t accum_flag = (k > 0 || kk > 0) ? 1 : 0;
                        // A-operand = weight (UMMA_M=256), B-operand = activation (UMMA_N=M)
                        ptx::tcgen05_mma_2sm(tmem_c, w_full, a_full,
                                             runtime_idesc, accum_flag);
                    }
                }
                __syncwarp();

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

            // [PROFILE] MMA warp: end of this iteration's issue window.
            if (kProfile && cluster_id == 0 && lane_id == 0) {
                prof[persistent_iter * 4 + 0] = prof_mma_t0;
                prof[persistent_iter * 4 + 1] = ptx::rdclock();
            }

            persistent_iter++;
        }

        if (persistent_iter > 0) {
            uint32_t last_iter        = persistent_iter - 1;
            uint32_t last_accum_stage = last_iter % NUM_EPI_STAGES;
            uint32_t last_accum_phase = (last_iter / NUM_EPI_STAGES) & 1;
            s.tmem_empty_barriers[last_accum_stage].wait(last_accum_phase);
        }
    }

    // ======== EPILOGUE WARPS (both CTAs, 128 threads / 4 warps) ========
    else if (warp_id >= NUM_NON_EPI_THREADS / 32 &&
             warp_id < (NUM_NON_EPI_THREADS + NUM_STORE_THREADS) / 32) {
        uint32_t epi_warp_idx = warp_id - (NUM_NON_EPI_THREADS / 32);  // 0..3
        uint32_t tma_store_idx = 0;

        constexpr int NUM_STORES         = M_TPL / STORE_BLOCK_M;              // M/16
        constexpr int NUM_TMEM_SUBROWS   = STORE_BLOCK_M / 8;                  // 2
        constexpr int NUM_N_STORE_ATOMS  = STORE_BLOCK_N / STORE_BLOCK_N_ATOM; // 4
        constexpr int SMEM_CD_PER_STAGE_T = SMEM_CD_PER_STAGE;                 // 8192

        uint32_t persistent_iter = 0;
        for (int tile_id = cluster_id; tile_id < num_tiles_total; tile_id += num_clusters) {
            uint32_t accum_stage = persistent_iter % NUM_EPI_STAGES;
            uint32_t accum_phase = (persistent_iter / NUM_EPI_STAGES) & 1;

            s.tmem_full_barriers[accum_stage].wait(accum_phase);
            ptx::tcgen05_fence_after_sync();

            // [PROFILE] Epilogue (leader CTA): start of this iteration's readback+store.
            long long prof_epi_t0 = 0;
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0)
                prof_epi_t0 = ptx::rdclock();

            uint32_t tmem_base = accum_stage * UMMA_N_T;
            // Output N base for this CTA (owns 128 columns of N).
            int base_n = tile_id * CLUSTER_BLOCK_N + cta_rank * BLOCK_N;
            int base_m = 0;

            for (int st = 0; st < NUM_STORES; ++st,
                 tma_store_idx = (tma_store_idx + 1) % NUM_TMA_STORE_STAGES) {
                auto* smem_cd_ptr = reinterpret_cast<uint8_t*>(
                    s.smem_cd + tma_store_idx * SMEM_CD_PER_STAGE_T);

                if (epi_warp_idx == 0)
                    cute::tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                // ---- Read TMEM, transpose into SMEM (DeepGEMM swap FP32 path) ----
                #pragma unroll
                for (int i = 0; i < NUM_TMEM_SUBROWS; ++i) {
                    uint32_t tmem_addr = tmem_base + st * STORE_BLOCK_M + i * 8;

                    uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
                    ptx::tmem_load_32dp32b8x(tmem_addr, v0, v1, v2, v3, v4, v5, v6, v7);
                    cutlass::arch::fence_view_async_tmem_load();
                    uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};

                    // one 32-wide N atom per epilogue warp
                    uint8_t* smem_base_ptr = smem_cd_ptr
                        + epi_warp_idx * (STORE_BLOCK_M * SWIZZLE_CD)   // outer atom offset
                        + i * (8 * SWIZZLE_CD);                        // inner atom offset
                    uint32_t col = lane_id / 4;
                    #pragma unroll
                    for (uint32_t row = 0; row < 8; ++row) {
                        uint8_t* smem_ptr = smem_base_ptr
                            + row * (16 * 8)                // kNumBankGroupBytes(16) * 8
                            + (col ^ row) * 16
                            + (lane_id % 4) * sizeof(float);
                        ptx::st_shared_u32(smem_ptr, vals[row]);
                    }
                }

                if (st == NUM_STORES - 1) {
                    ptx::tcgen05_fence_before_sync();
                    s.tmem_empty_barriers[accum_stage].arrive(0u);
                }

                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(NUM_STORE_THREADS, 0);

                if (epi_warp_idx == 0 && ptx::elect_one_sync()) {
                    #pragma unroll
                    for (int i = 0; i < NUM_N_STORE_ATOMS; ++i) {
                        auto* smem_ptr = reinterpret_cast<float*>(smem_cd_ptr)
                            + i * (STORE_BLOCK_M * STORE_BLOCK_N_ATOM);
                        int n_idx = base_n + i * STORE_BLOCK_N_ATOM;
                        int m_idx = base_m + st * STORE_BLOCK_M;
                        tma::store_2d(&desc_D, smem_ptr, n_idx, m_idx);
                    }
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }

            // [PROFILE] Epilogue (leader CTA): end of this iteration's readback+store.
            if (kProfile && cluster_id == 0 && cta_rank == 0 &&
                epi_warp_idx == 0 && lane_id == 0) {
                prof[persistent_iter * 4 + 2] = prof_epi_t0;
                prof[persistent_iter * 4 + 3] = ptx::rdclock();
            }

            persistent_iter++;
        }
    }

    // ================================================================
    // CLEANUP
    // ================================================================
    ptx::cluster_sync();
    if (warp_id == 0) {
        ptx::tcgen05_dealloc_2sm(0, NUM_TMEM_COLS_T);
    }
}

// ======================== Host: TMA Descriptors ========================
static CUtensorMap make_tma_desc_bf16_2d(
    const nv_bfloat16* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(nv_bfloat16)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

static CUtensorMap make_tma_desc_fp32_2d(
    const float* ptr, int rows, int cols, int box_rows, int box_cols)
{
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(float)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2, (void*)ptr, globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return desc;
}

// ======================== Kernel / SMEM selectors ========================
template <bool kProfile>
static void* get_kernel_ptr(int M) {
    switch (M) {
        case 32:  return (void*)&wq_b_proj_kernel<32,  kProfile>;
        case 64:  return (void*)&wq_b_proj_kernel<64,  kProfile>;
        case 96:  return (void*)&wq_b_proj_kernel<96,  kProfile>;
        case 128: return (void*)&wq_b_proj_kernel<128, kProfile>;
        case 160: return (void*)&wq_b_proj_kernel<160, kProfile>;
        case 192: return (void*)&wq_b_proj_kernel<192, kProfile>;
        case 224: return (void*)&wq_b_proj_kernel<224, kProfile>;
        case 256: return (void*)&wq_b_proj_kernel<256, kProfile>;
        default:  return nullptr;
    }
}

static int get_smem_bytes(int M) {
    switch (M) {
        case 32:  return (int)sizeof(SharedStorage<32>);
        case 64:  return (int)sizeof(SharedStorage<64>);
        case 96:  return (int)sizeof(SharedStorage<96>);
        case 128: return (int)sizeof(SharedStorage<128>);
        case 160: return (int)sizeof(SharedStorage<160>);
        case 192: return (int)sizeof(SharedStorage<192>);
        case 224: return (int)sizeof(SharedStorage<224>);
        case 256: return (int)sizeof(SharedStorage<256>);
        default:  return 0;
    }
}

// Common launch. When `profile` is true, allocates a [max_iters, 4] int64 timing
// tensor and returns it as the 2nd element. Each row is one persistent iteration
// of cluster 0's leader CTA:
//   col 0 = MMA warp start   (clock64)
//   col 1 = MMA warp end
//   col 2 = epilogue start
//   col 3 = epilogue end
// MMA and epilogue here are on the SAME CTA (same SM) -> clock64 directly comparable.
static std::vector<torch::Tensor> run_wq_b(torch::Tensor x, torch::Tensor w, bool profile)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);

    const int M = x.size(0);
    TORCH_CHECK(x.size(1) == K_DIM);
    TORCH_CHECK(w.size(0) == N_TOTAL && w.size(1) == K_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0);

    // swap-AB: BLOCK_M = M exactly (single M block), no padding.
    auto out = torch::empty({M, N_TOTAL}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_ptr   = reinterpret_cast<const nv_bfloat16*>(x.data_ptr());
    auto w_ptr   = reinterpret_cast<const nv_bfloat16*>(w.data_ptr());
    auto out_ptr = reinterpret_cast<float*>(out.data_ptr());

    // TMA descriptors
    const int load_block_m = M / NUM_MULTICAST; // activation rows per CTA
    CUtensorMap desc_A = make_tma_desc_bf16_2d(x_ptr, M, K_DIM, load_block_m, BLOCK_K);
    CUtensorMap desc_B = make_tma_desc_bf16_2d(w_ptr, N_TOTAL, K_DIM, LOAD_BLOCK_N, BLOCK_K);
    // Output D: box = STORE_BLOCK_M (M rows) x STORE_BLOCK_N_ATOM (N cols), 128B swizzle
    CUtensorMap desc_D = make_tma_desc_fp32_2d(out_ptr, M, N_TOTAL, STORE_BLOCK_M, STORE_BLOCK_N_ATOM);

    // Grid: persistent, cluster of 2 CTAs.
    static const int num_SMs = []() {
        int n = 0;
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
        return n;
    }();
    int total_cta = NUM_N_TILES * CLUSTER_SIZE;   // max useful CTAs
    int grid_size = min(num_SMs, total_cta);
    grid_size = (grid_size / CLUSTER_SIZE) * CLUSTER_SIZE;  // multiple of cluster size
    if (grid_size < CLUSTER_SIZE) grid_size = CLUSTER_SIZE;

    const int num_clusters = grid_size / CLUSTER_SIZE;
    const int max_iters = (NUM_N_TILES + num_clusters - 1) / num_clusters;

    // Profiling timing buffer (only cluster 0's leader CTA writes into it).
    int64_t* prof_dev = nullptr;
    torch::Tensor timing;
    if (profile) {
        timing = torch::zeros({max_iters, 4}, x.options().dtype(torch::kInt64));
        prof_dev = reinterpret_cast<int64_t*>(timing.data_ptr());
    }

    void* kernel_ptr = profile ? get_kernel_ptr<true>(M) : get_kernel_ptr<false>(M);
    int smem_bytes = get_smem_bytes(M);
    TORCH_CHECK(kernel_ptr != nullptr && smem_bytes > 0, "Unsupported M=", M);

    // Configure max dynamic SMEM once per (profile, M) variant.
    static bool smem_configured[2][9] = {{false}};
    const int m_idx = M / 32;
    const int p_idx = profile ? 1 : 0;
    if (!smem_configured[p_idx][m_idx]) {
        auto attr_err = cudaFuncSetAttribute(kernel_ptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        TORCH_CHECK(attr_err == cudaSuccess, "cudaFuncSetAttribute failed: ",
                    cudaGetErrorString(attr_err), " smem_bytes=", smem_bytes);
        smem_configured[p_idx][m_idx] = true;
    }

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
        attrs[0].val.clusterDim.x = CLUSTER_SIZE;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;

        void* ptr_args[] = { &desc_A, &desc_B, &desc_D, &grid_size, &prof_dev };
        auto err = cudaLaunchKernelExC(&config, kernel_ptr, ptr_args);
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
    }

    if (profile) return {out, timing};
    return {out};
}

// ======================== PyTorch Binding ========================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps)
{
    (void)rms_w; (void)eps;   // unused in the pure-GEMM (FP32 output) path
    return run_wq_b(x, w, /*profile=*/false)[0];
}

// Profiled variant: returns (out, timing[max_iters, 4]) where timing holds
// clock64 timestamps of the MMA / epilogue windows on cluster 0's leader CTA.
std::vector<torch::Tensor> wq_b_proj_gemm_profiled(torch::Tensor x, torch::Tensor w)
{
    return run_wq_b(x, w, /*profile=*/true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj (tcgen05 2SM MMA, DeepGEMM swap-AB, Blackwell)");
    m.def("wq_b_proj_gemm_profiled", &wq_b_proj_gemm_profiled,
          "wq_b proj + clock64 MMA/epilogue overlap timing -> (out, timing[max_iters,4])");
}
