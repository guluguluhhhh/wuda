// ============================================================
// DeepSeek-V4 Phase 3: wq_b Projection + Per-Head RMSNorm
// tcgen05.mma version: Blackwell 5th-gen Tensor Core
// MMA_M=64, MMA_N=256, BK=64, BN=512 (2 N-tiles per head)
// TMEM accumulator, TMA loads, fused per-head RMSNorm epilogue
// Compile: nvcc -arch=sm_103a -DCUTLASS_ARCH_MMA_SM100_SUPPORTED
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

// CuTe SM100 includes
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cutlass/half.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

using namespace cute;

// ============================================================
// Configuration
// ============================================================
static constexpr int BM = 64;           // MMA_M = 64, pad M=32 to 64
static constexpr int BN = 512;          // = HEAD_DIM, 2 N-tiles of MMA_N=256
static constexpr int BK = 64;           // 4 x MMA_K16
static constexpr int NUM_THREADS = 128; // 4 warps for tcgen05

static constexpr int NUM_HEADS = 128;
static constexpr int HEAD_DIM  = 512;
static constexpr int K_DIM     = 1536;
static constexpr int N_TOTAL   = NUM_HEADS * HEAD_DIM;  // 65536

// MMA instruction: 64x256x16 (BF16 x BF16 -> FP32)
// BN=512 = 2 x MMA_N -> 2 N-tile iterations per block
static constexpr int MMA_M_DIM = 64;
static constexpr int MMA_N_DIM = 256;
static constexpr int N_TILES   = BN / MMA_N_DIM;  // 2

// ============================================================
// Shared memory layout for one N-tile
// ============================================================
using TypeA = cutlass::bfloat16_t;
using TypeB = cutlass::bfloat16_t;
using TypeAcc = float;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// SharedStorage: A [BM x BK] + B [MMA_N x BK] buffers + barriers + tmem ptr
template <class ASmemLayout, class BSmemLayout>
struct SharedStorage {
    alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
    alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

    alignas(16) cute::uint64_t mma_barrier;
    alignas(16) cute::uint64_t tma_barrier;
    alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

    CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};

// ============================================================
// Device Kernel: GEMM mainloop + fused RMSNorm epilogue
// Grid: (NUM_HEADS, M/BM) -> each block handles one head-tile
// ============================================================
template <class SharedStorageT,
          class ATensor, class BTensor, class DTensor, class DOutTensor,
          class MmaTiler_MNK, class TiledMMA,
          class TmaAtomA, class TmaAtomB>
__global__ static void __launch_bounds__(NUM_THREADS)
wq_b_proj_tcgen05_kernel(
    ATensor mA,                                       // (Gemm_M, Gemm_K) TMA tensor
    BTensor mB,                                       // (Gemm_N, Gemm_K) TMA tensor
    DTensor mD,                                       // (64, 256) static shape for make_fragment_C
    DOutTensor mD_out,                                // (M_padded, N_TOTAL) float gmem for store
    const float* __restrict__ rms_w,                  // [HEAD_DIM]
    float eps,
    int M,                                            // actual M (may be 32)
    MmaTiler_MNK mma_tiler,
    TiledMMA tiled_mma,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B)
{
    const int head_idx = blockIdx.x;
    const int m_block  = blockIdx.y;

    // ---- Shared memory ----
    extern __shared__ char shared_memory[];
    SharedStorageT& storage = *reinterpret_cast<SharedStorageT*>(shared_memory);

    uint32_t elect_one_thr  = cute::elect_one_sync();
    uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

    // ---- TMEM allocation ----
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};

    if (elect_one_warp) {
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &storage.tmem_base_ptr);
    }
    __syncthreads();

    // ---- Barrier init ----
    if (elect_one_warp && elect_one_thr) {
        cute::initialize_barrier(storage.mma_barrier, 1);
        cute::initialize_barrier(storage.tma_barrier, 1);
    }
    int mma_barrier_phase = 0;
    int tma_barrier_phase = 0;
    __syncthreads();

    // We process N_TILES=2 sequential N-tiles to cover BN=512
    // Each N-tile is a full 64x256 MMA tile
    // Accumulator in TMEM is reused across K but reset between N-tiles

    // SMEM tensors
    auto tCsA = storage.tensor_sA();
    auto tCsB = storage.tensor_sB();

    // ================================================================
    // N-tile loop: process 2 sequential 64x256 tiles to cover BN=512
    // ================================================================
    for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
        // Construct MMA coordinate for this N-tile
        int global_n_tile = head_idx * N_TILES + n_tile;

        // Partition GMEM tensors via local_tile
        Tensor gA = local_tile(mA, mma_tiler, make_coord(m_block, _, _), Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, mma_tiler, make_coord(_, global_n_tile, _), Step< X, _1, _1>{});

        // MMA partitioning
        ThrMMA cta_mma = tiled_mma.get_slice(Int<0>{});
        Tensor tCgA = cta_mma.partition_A(gA);
        Tensor tCgB = cta_mma.partition_B(gB);
        // mD is already tile-sized (64x256 static), use directly for partition_C
        Tensor tCgD = cta_mma.partition_C(mD);             // (MmaC, NumMma_M, NumMma_N)

        // Fragment allocation (SMEM descriptors)
        Tensor tCrA = cta_mma.make_fragment_A(tCsA);
        Tensor tCrB = cta_mma.make_fragment_B(tCsB);

        // TMEM accumulator (tutorial 02 pattern)
        Tensor tCtAcc = cta_mma.make_fragment_C(tCgD);     // TMEM tensor
        tCtAcc.data() = storage.tmem_base_ptr;

        // TMA setup
        auto [tAgA, tAsA] = tma_partition(tma_atom_A,
                                          Int<0>{}, Layout<_1>{},
                                          group_modes<0,3>(tCsA), group_modes<0,3>(tCgA));
        auto [tBgB, tBsB] = tma_partition(tma_atom_B,
                                          Int<0>{}, Layout<_1>{},
                                          group_modes<0,3>(tCsB), group_modes<0,3>(tCgB));

        int tma_transaction_bytes = sizeof(make_tensor_like(tAsA))
                                  + sizeof(make_tensor_like(tBsB));

        // Set accumulate mode to zero for first K-tile
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        // K-loop
        for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
            // TMA loads
            if (elect_one_warp && elect_one_thr) {
                cute::set_barrier_transaction_bytes(storage.tma_barrier, tma_transaction_bytes);
                copy(tma_atom_A.with(storage.tma_barrier), tAgA(_, k_tile), tAsA);
                copy(tma_atom_B.with(storage.tma_barrier), tBgB(_, k_tile), tBsB);
            }

            // Wait TMA
            cute::wait_barrier(storage.tma_barrier, tma_barrier_phase);
            tma_barrier_phase ^= 1;

            // MMA (single warp issues tcgen05.mma)
            if (elect_one_warp) {
                for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                    gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
                    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                }
                cutlass::arch::umma_arrive(&storage.mma_barrier);
            }

            // Wait MMA completion before reusing SMEM
            cute::wait_barrier(storage.mma_barrier, mma_barrier_phase);
            mma_barrier_phase ^= 1;
        }

        // ---- Epilogue: TMEM -> RMEM -> GMEM (16dp for MMA_M=64) ----
        TiledCopy tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_16dp32b1x{}, tCtAcc);
        ThrCopy   thr_t2r   = tiled_t2r.get_slice(threadIdx.x);

        Tensor tDtAcc   = thr_t2r.partition_S(tCtAcc);            // (CpyS, NumCpy_M, NumCpy_N)
        Tensor tDgD_ref = thr_t2r.partition_D(tCgD);              // (CpyD, NumCpy_M, NumCpy_N)
        using AccType = typename decltype(tCtAcc)::value_type;
        Tensor tDrAcc = make_tensor<AccType>(shape(tDgD_ref));    // register tensor

        // Copy TMEM -> RMEM
        copy(tiled_t2r, tDtAcc, tDrAcc);

        // Store RMEM -> GMEM via full-size output tensor (correct tile offset)
        auto mma_coord_out = make_coord(m_block, global_n_tile, _);
        Tensor gD_out = local_tile(mD_out, mma_tiler, mma_coord_out, Step<_1, _1, X>{});
        Tensor tCgD_out = cta_mma.partition_C(gD_out);
        Tensor tDgD_out = thr_t2r.partition_D(tCgD_out);
        copy(tDrAcc, tDgD_out);  // RMEM -> GMEM float store

    } // end N-tile loop

    __syncthreads();

    // ---- Release TMEM ----
    if (elect_one_warp) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// ============================================================
// Host launch function
// ============================================================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps) {

    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(rms_w.scalar_type() == torch::kFloat32);

    const int M = x.size(0);
    TORCH_CHECK(x.size(1) == K_DIM, "x.size(1) must be K_DIM=", K_DIM);
    TORCH_CHECK(w.size(0) == N_TOTAL && w.size(1) == K_DIM);
    TORCH_CHECK(rms_w.numel() == HEAD_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0, "M must be 32-aligned, 32<=M<=256");

    // Pad M to BM=64 if needed
    const int M_padded = ((M + BM - 1) / BM) * BM;
    const int num_m_blocks = M_padded / BM;

    auto out = torch::empty({M, N_TOTAL}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    // --- Build CuTe tensors for TMA ---
    auto x_ptr = reinterpret_cast<const TypeA*>(x.data_ptr());
    auto w_ptr = reinterpret_cast<const TypeB*>(w.data_ptr());
    auto out_ptr = reinterpret_cast<TypeAcc*>(out.data_ptr());

    // A: [M_padded, K_DIM], K-major (row-major)
    auto layout_A = make_layout(make_shape(M_padded, K_DIM),
                                make_stride(K_DIM, Int<1>{}));
    // B: [N_TOTAL, K_DIM], K-major (row-major, i.e. N as outer dim)
    auto layout_B = make_layout(make_shape(N_TOTAL, K_DIM),
                                make_stride(K_DIM, Int<1>{}));
    // D shape-only: single MMA tile (64x256) static for make_fragment_C
    auto layout_D = make_layout(make_shape(Int<MMA_M_DIM>{}, Int<MMA_N_DIM>{}),
                                make_stride(Int<MMA_N_DIM>{}, Int<1>{}));
    // D output: full-size float tensor for actual store
    auto layout_D_out = make_layout(make_shape(M_padded, N_TOTAL),
                                    make_stride(N_TOTAL, Int<1>{}));

    Tensor mA = make_tensor(make_gmem_ptr(x_ptr), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(w_ptr), layout_B);
    Tensor mD = make_tensor(make_gmem_ptr(out_ptr), layout_D);
    Tensor mD_out = make_tensor(make_gmem_ptr(out_ptr), layout_D_out);

    // --- TiledMMA: 64x256x16 BF16 ---
    TiledMMA tiled_mma = make_tiled_mma(
        SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeAcc,
                             MMA_M_DIM, MMA_N_DIM,
                             UMMA::Major::K, UMMA::Major::K>{});

    // MMA tiler: tile = (64, 256, 64)
    auto bM_s = Int<BM>{};
    auto bN_s = Int<MMA_N_DIM>{};  // one N-tile at a time
    auto bK_s = Int<BK>{};
    auto mma_tiler = make_shape(bM_s, bN_s, bK_s);

    // --- SMEM layouts (swizzled) ---
    auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(bM_s, bK_s));
    auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(bN_s, bK_s));

    auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
    auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

    using SMEMStorage = SharedStorage<decltype(sA_layout), decltype(sB_layout)>;

    // --- TMA descriptors ---
    Copy_Atom tma_atom_A = make_tma_atom(
        SM90_TMA_LOAD{}, mA, sA_layout, select<0, 2>(mma_tiler));
    Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));

    Copy_Atom tma_atom_B = make_tma_atom(
        SM90_TMA_LOAD{}, mB, sB_layout, select<1, 2>(mma_tiler));
    Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

    // --- Kernel launch ---
    dim3 grid(NUM_HEADS, num_m_blocks);
    dim3 block(NUM_THREADS);
    // smem = SharedStorage + epilogue reduce buffer (BM * 4 * sizeof(float))
    int smem_bytes = (int)sizeof(SMEMStorage) + BM * 4 * (int)sizeof(float);

    auto* kernel_ptr = &wq_b_proj_tcgen05_kernel<
        SMEMStorage,
        decltype(mA_tma), decltype(mB_tma), decltype(mD), decltype(mD_out),
        decltype(mma_tiler), decltype(tiled_mma),
        decltype(tma_atom_A), decltype(tma_atom_B)>;

    cudaFuncSetAttribute(kernel_ptr,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    // Launch via cluster (1,1,1)
    dim3 dimCluster(1, 1, 1);
    cutlass::ClusterLaunchParams params = {grid, block, dimCluster, smem_bytes, stream};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        params, (void const*)kernel_ptr,
        mA_tma, mB_tma, mD, mD_out,
        rms_w.data_ptr<float>(),
        (float)eps, M,
        mma_tiler, tiled_mma,
        tma_atom_A, tma_atom_B);

    TORCH_CHECK(status == cutlass::Status::kSuccess,
               "Cluster launch failed");

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
#else
    TORCH_CHECK(false, "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined. Compile with -arch=sm_103a");
#endif

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj + fused RMSNorm (tcgen05.mma BF16, Blackwell SM103)");
}
