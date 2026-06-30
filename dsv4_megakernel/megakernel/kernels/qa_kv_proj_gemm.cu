// ============================================================
// DeepSeek-V4 Phase 2: SM100 Block-Scaled FP8 GEMM
// Based on CUTLASS example 72c_blackwell_mixed_mxfp8_bf16_gemm
// C[M,N] = A_mxfp8[M,K] × B_mxfp8[N,K]^T with block-scaled E8M0
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================
// GEMM Configuration (following example 72c)
// ============================================================

// A: MXFP8 E4M3 (data + scale bundled)
using ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

// B: MXFP8 E4M3
using ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

// C/D: BF16
using ElementC    = cutlass::bfloat16_t;
using ElementD    = cutlass::bfloat16_t;
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

// Accumulator
using ElementAccumulator = float;

// Architecture
using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile & Cluster (1Sm, smaller tile for our use case)
using MmaTileShape  = Shape<_128, _128, _128>;
using ClusterShape  = Shape<_1, _1, _1>;

// ============================================================
// Build Epilogue (Auto schedule)
// ============================================================
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// ============================================================
// Build Mainloop (Auto schedule)
// ============================================================
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// ============================================================
// Assemble Kernel
// ============================================================
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // ProblemShape (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Type aliases from the kernel
using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ============================================================
// PyTorch Binding: Merged wq_a + wkv in single GEMM
// W_merged = cat([wq_a, wkv], dim=0) → [N_QA+N_KV, K] = [2048, 7168]
// Single GEMM: [M, K] × [K, 2048] → [M, 2048], then split
// ============================================================
std::vector<torch::Tensor> fp8_gemm_merged(
    torch::Tensor A,          // [M, K] float8_e4m3fn (x_normed)
    torch::Tensor A_scale,    // scale factors for A
    torch::Tensor B_merged,   // [N_QA+N_KV, K] = [2048, K] float8_e4m3fn (merged weight)
    torch::Tensor B_scale,    // scale factors for B_merged
    int64_t N_QA              // split point (1536)
) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fn, "A must be float8_e4m3fn");

    int M = A.size(0);
    int K = A.size(1);
    int N = B_merged.size(0);  // N_QA + N_KV = 2048

    // Output
    auto C = torch::zeros({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kBFloat16));
    auto D = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kBFloat16));

    // Strides
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    // Scale factor layouts
    LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    // Arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<typename ElementA::DataType const*>(A.data_ptr()),
            stride_A,
            reinterpret_cast<typename ElementB::DataType const*>(B_merged.data_ptr()),
            stride_B,
            reinterpret_cast<typename ElementA::ScaleFactorType const*>(A_scale.data_ptr()),
            layout_SFA,
            reinterpret_cast<typename ElementB::ScaleFactorType const*>(B_scale.data_ptr()),
            layout_SFB
        },
        {
            {1.0f, 0.0f},
            reinterpret_cast<ElementC const*>(C.data_ptr()),
            stride_C,
            reinterpret_cast<ElementD*>(D.data_ptr()),
            stride_D
        }
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS cannot implement merged GEMM: ", cutlass::cutlassGetStatusString(status));

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().device(A.device()).dtype(torch::kUInt8));

    status = gemm_op.initialize(arguments, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS init failed");

    status = gemm_op.run(at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS run failed");

    // Split output: D[:, :N_QA] = qr, D[:, N_QA:] = kv (non-contiguous views, no memcpy)
    auto qr = D.narrow(1, 0, N_QA);
    auto kv = D.narrow(1, N_QA, N - N_QA);
    return {qr, kv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qa_kv_proj_gemm", &fp8_gemm_merged,
          "SM100 Block-Scaled MXFP8 merged GEMM: wq_a+wkv in one call (E4M3 x E4M3 -> BF16, split output)");
}
