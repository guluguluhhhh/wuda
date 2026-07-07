// ============================================================
// DeepSeek-V4: wq_b Projection GEMM via CUTLASS SM100 CollectiveBuilder
//
// Single configuration optimized for M=32 (decode scenario):
//   Tile(64, 256, 64) + Cluster(2, 1, 1)
//   - 2SM MMA (cta_group::2) along M direction
//   - B shared via DSMEM between 2 CTAs
//   - Matches nvjet_sm103_tst_512x32_64x3_2x1_2cta pattern
//
// Layout: A [M, K] RowMajor, B [N, K] RowMajor (= TN GEMM)
// Output: D [M, N] RowMajor FP32
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

// ======================== Configuration ========================
static constexpr int NUM_HEADS = 128;
static constexpr int HEAD_DIM  = 512;
static constexpr int K_DIM     = 1536;
static constexpr int N_TOTAL   = NUM_HEADS * HEAD_DIM; // 65536

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 128 / sizeof(ElementA);
static constexpr int AlignmentB = 128 / sizeof(ElementB);
static constexpr int AlignmentC = 128 / sizeof(ElementC);
static constexpr int AlignmentD = 128 / sizeof(ElementD);

// Tile: (64, 256, 64) + Cluster(2,1,1) = 2SM MMA along M
// Cluster covers 128×256 per iteration, matching nvjet 512×32 style
using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;

// ======================== CUTLASS Kernel Type ========================
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using GemmDevice = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ======================== Host Function ========================
torch::Tensor wq_b_proj_gemm_cutlass(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);
    TORCH_CHECK(K == K_DIM);
    TORCH_CHECK(N == N_TOTAL);
    TORCH_CHECK(w.size(1) == K);

    auto out = torch::empty({M, N}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();

    using StrideA = typename GemmKernel::CollectiveMainloop::StrideA;
    using StrideB = typename GemmKernel::CollectiveMainloop::StrideB;
    using StrideC = typename GemmKernel::CollectiveEpilogue::StrideC;
    using StrideD = typename GemmKernel::CollectiveEpilogue::StrideD;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto* a_ptr = reinterpret_cast<ElementA const*>(x.data_ptr());
    auto* b_ptr = reinterpret_cast<ElementB const*>(w.data_ptr());
    auto* d_ptr = reinterpret_cast<ElementD*>(out.data_ptr());

    typename GemmDevice::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {a_ptr, stride_A, b_ptr, stride_B},
        {{1.0f, 0.0f}, nullptr, stride_C, d_ptr, stride_D}
    };

    GemmDevice gemm_op;

    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS cannot implement: ", cutlass::cutlassGetStatusString(status));

    size_t workspace_size = gemm_op.get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_size},
                                  x.options().dtype(torch::kUInt8));

    status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS init failed");

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS run failed");

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm_cutlass", &wq_b_proj_gemm_cutlass,
          "wq_b proj GEMM via CUTLASS SM100 (Tile64x256x64, Cluster 2x1)");
}
