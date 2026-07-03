// ============================================================
// DeepSeek-V4 Phase 3: wq_b Projection + Per-Head RMSNorm
// BW-optimized: 256 threads (8 warps), STAGES=4, TMA SWIZZLE_64B
// All warps share load+compute, ldmatrix.x4 for A and B
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

// ============================================================
// Configuration
// ============================================================
constexpr int BN = 512;
constexpr int WARPK = 32;
constexpr int STAGES = 4;
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = 8;
constexpr int WARP_COUNT_N = 8;
constexpr int WARP_N = BN / WARP_COUNT_N;  // 64
constexpr int WARP_TILE_N = WARP_N / 16;   // 4

constexpr int TMA_B_ROWS = 256;

constexpr int NUM_HEADS = 128;
constexpr int HEAD_DIM = 512;
constexpr int K_DIM = 1536;
constexpr int N_TOTAL = NUM_HEADS * HEAD_DIM;
constexpr int NUM_K_TILES = K_DIM / WARPK;  // 48

constexpr int SMEM_B_STAGE = BN * WARPK;    // 16384 elements
constexpr int MBAR_REGION = 128;
constexpr int BYTES_B_HALF = TMA_B_ROWS * WARPK * 2;  // 16384

// ============================================================
// PTX: mbarrier + TMA (sm_90+)
// ============================================================
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* bar, uint32_t bytes) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n .reg .pred P1;\n"
        " LOOP: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        " @P1 bra DONE; bra LOOP; DONE:\n}\n"
        :: "r"(a), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d(
    void* smem, const CUtensorMap* desc, int32_t c0, int32_t c1, uint64_t* bar) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(s), "l"(desc), "r"(c0), "r"(c1), "r"(b) : "memory");
}

// ============================================================
// SWIZZLE_64B: atom XOR with (row >> 1) & 3
// Row index for swizzle increments every 128B (= 2 data rows at 64B each)
// Period = 8 data rows
// ============================================================
__device__ __forceinline__ uint32_t sw64_addr(
    const __nv_bfloat16* base, int row, int col_sector_start) {
    int sector = col_sector_start >> 3;
    int physical_sector = sector ^ ((row >> 1) & 3);
    int offset = row * WARPK + (physical_sector << 3);
    return static_cast<uint32_t>(__cvta_generic_to_shared(base + offset));
}

__device__ __forceinline__ __nv_bfloat16 sw64_load(
    const __nv_bfloat16* base, int row, int col) {
    int sector = col >> 3;
    int in_sector = col & 7;
    int physical_sector = sector ^ ((row >> 1) & 3);
    return base[row * WARPK + (physical_sector << 3) + in_sector];
}

// ============================================================
// PTX: ldmatrix.x4 + mma.sync.m16n8k16
// ============================================================
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    // D = A*B + D (in-place accumulate, "+f" constraint)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// ============================================================
// B fragment scalar load (bypass ldmatrix for correctness)
// mma.sync.m16n8k16.row.col B operand (.col = K-fast):
//   b[0] = pack(B_op[groupID, tidInGrp*2], B_op[groupID, tidInGrp*2+1])
//   b[1] = pack(B_op[groupID+8, tidInGrp*2], B_op[groupID+8, tidInGrp*2+1])
// where B_op[k, n] = sB[(n_start+n)*WARPK + k_start+k] (swizzled)
// ============================================================
__device__ __forceinline__ void load_B_frag(
    const __nv_bfloat16* sB, int n_start, int k_start,
    int group_id, int tid_in_grp, uint32_t& b0, uint32_t& b1) {
    // b[0] = {B_op[groupID, tidInGrp*2], B_op[groupID, tidInGrp*2+1]}  K=0..7
    // b[1] = {B_op[groupID+8, tidInGrp*2], B_op[groupID+8, tidInGrp*2+1]} K=8..15
    // B_op[k, n] = sB[(n_start+n)*WARPK + k_start+k]
    __nv_bfloat16 v0 = sw64_load(sB, n_start + tid_in_grp * 2,     k_start + group_id);
    __nv_bfloat16 v1 = sw64_load(sB, n_start + tid_in_grp * 2 + 1, k_start + group_id);
    __nv_bfloat16 v2 = sw64_load(sB, n_start + tid_in_grp * 2,     k_start + group_id + 8);
    __nv_bfloat16 v3 = sw64_load(sB, n_start + tid_in_grp * 2 + 1, k_start + group_id + 8);
    __nv_bfloat162 p0 = __halves2bfloat162(v0, v1);  // b[0]: K_low
    __nv_bfloat162 p1 = __halves2bfloat162(v2, v3);  // b[1]: K_high
    b0 = reinterpret_cast<const uint32_t&>(p0);
    b1 = reinterpret_cast<const uint32_t&>(p1);
}

// ============================================================
// Warp-level reduce within 4-lane group
// ============================================================
__device__ __forceinline__ float group4_reduce(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    return val;
}

// ============================================================
// Kernel (templated on BM)
// ============================================================
template <int BM>
__global__ void __launch_bounds__(NUM_THREADS)
wq_b_proj_kernel(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    const float* __restrict__ rms_w,
    __nv_bfloat16* __restrict__ gOut,
    float eps, int M
) {
    constexpr int SMEM_A_STAGE = BM * WARPK;
    constexpr int WARP_TILE_M = BM / 16;
    constexpr int BYTES_A = BM * WARPK * 2;
    constexpr int BYTES_PER_STAGE = BYTES_A + 2 * BYTES_B_HALF;

    const int head_idx = blockIdx.x;
    const int m_block  = blockIdx.y;
    const int warp_id  = threadIdx.x / 32;
    const int lane_id  = threadIdx.x % 32;
    const int group_id = lane_id / 4;
    const int tid_in_grp = lane_id % 4;

    // ================================================================
    // Shared memory: data FIRST (128B aligned), mbar at end
    // ================================================================
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_B = smem_A + STAGES * SMEM_A_STAGE;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        smem_raw + STAGES * (SMEM_A_STAGE + SMEM_B_STAGE) * sizeof(__nv_bfloat16));

    // Init mbarrier
    if (threadIdx.x < STAGES) mbar_init(&mbar[threadIdx.x], 1);
    __syncthreads();

    const int32_t tma_A_row = m_block * BM;
    const int32_t tma_B_row_lo = head_idx * HEAD_DIM;
    const int32_t tma_B_row_hi = tma_B_row_lo + TMA_B_ROWS;
    const int warp_n_start = warp_id * WARP_N;

    // Accumulator
    float frag_C[WARP_TILE_M][WARP_TILE_N][8];
    #pragma unroll
    for (int m = 0; m < WARP_TILE_M; ++m)
        #pragma unroll
        for (int n = 0; n < WARP_TILE_N; ++n)
            #pragma unroll
            for (int f = 0; f < 8; ++f)
                frag_C[m][n][f] = 0.0f;

    // ================================================================
    // Prologue
    // ================================================================
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            if (s < NUM_K_TILES) {
                mbar_arrive_tx(&mbar[s], BYTES_PER_STAGE);
                tma_load_2d(smem_A + s * SMEM_A_STAGE, &desc_A,
                            s * WARPK, tma_A_row, &mbar[s]);
                tma_load_2d(smem_B + s * SMEM_B_STAGE, &desc_B,
                            s * WARPK, tma_B_row_lo, &mbar[s]);
                tma_load_2d(smem_B + s * SMEM_B_STAGE + TMA_B_ROWS * WARPK,
                            &desc_B, s * WARPK, tma_B_row_hi, &mbar[s]);
            }
        }
    }

    // ================================================================
    // GEMM mainloop
    // ================================================================
    uint32_t phase[STAGES] = {};

    for (int kt = 0; kt < NUM_K_TILES; ++kt) {
        const int stg = kt % STAGES;
        mbar_wait(&mbar[stg], phase[stg]);
        phase[stg] ^= 1u;

        __nv_bfloat16* sA = smem_A + stg * SMEM_A_STAGE;
        __nv_bfloat16* sB = smem_B + stg * SMEM_B_STAGE;

        // 2 k-inner iterations (WarpK=32 -> 2 x k16)
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            const int k_atom_off = ki * 2;  // k=0→atoms 0,1; k=16→atoms 2,3
        
            // Load A via ldmatrix.x4 (16x16 tile, swizzled)
            // Thread mapping: a_row=lane&15, a_atom_local=(lane>>4)&1
            uint32_t a_frag[WARP_TILE_M][4];
            {
                int a_row_local = lane_id & 15;
                int a_atom_local = (lane_id >> 4) & 1;
                #pragma unroll
                for (int mt = 0; mt < WARP_TILE_M; ++mt) {
                    int row = mt * 16 + a_row_local;
                    int logical_atom = k_atom_off + a_atom_local;
                    uint32_t addr = sw64_addr(sA, row, logical_atom * 8);
                    ldmatrix_x4(addr, a_frag[mt][0], a_frag[mt][1],
                                      a_frag[mt][2], a_frag[mt][3]);
                }
            }
        
            // For each N tile: load B via ldmatrix.x4 (16x16), then 2 MMA calls
            #pragma unroll
            for (int nt = 0; nt < WARP_TILE_N; ++nt) {
                int b_n_base = warp_n_start + nt * 16;
        
                // Load B[16xK16] via ldmatrix.x4 (swizzled)
                // Thread mapping: b_row=(lane&7)+((lane>>4)<<3), b_atom_local=(lane>>3)&1
                uint32_t b_frag[4];
                {
                    int b_row_local = (lane_id & 7) + ((lane_id >> 4) << 3);
                    int b_atom_local = (lane_id >> 3) & 1;
                    int row = b_n_base + b_row_local;
                    int logical_atom = k_atom_off + b_atom_local;
                    uint32_t addr = sw64_addr(sB, row, logical_atom * 8);
                    ldmatrix_x4(addr, b_frag[0], b_frag[1], b_frag[2], b_frag[3]);
                }
        
                // MMA n8=0: uses b_frag[0:1]
                #pragma unroll
                for (int mt = 0; mt < WARP_TILE_M; ++mt) {
                    mma_m16n8k16(
                        frag_C[mt][nt][0], frag_C[mt][nt][1],
                        frag_C[mt][nt][2], frag_C[mt][nt][3],
                        a_frag[mt][0], a_frag[mt][1], a_frag[mt][2], a_frag[mt][3],
                        b_frag[0], b_frag[1]);
                }
        
                // MMA n8=1: uses b_frag[2:3]
                #pragma unroll
                for (int mt = 0; mt < WARP_TILE_M; ++mt) {
                    mma_m16n8k16(
                        frag_C[mt][nt][4], frag_C[mt][nt][5],
                        frag_C[mt][nt][6], frag_C[mt][nt][7],
                        a_frag[mt][0], a_frag[mt][1], a_frag[mt][2], a_frag[mt][3],
                        b_frag[2], b_frag[3]);
                }
            }
        }

        __syncthreads();

        // Issue next TMA
        int next_kt = kt + STAGES;
        if (next_kt < NUM_K_TILES && threadIdx.x == 0) {
            mbar_arrive_tx(&mbar[stg], BYTES_PER_STAGE);
            tma_load_2d(smem_A + stg * SMEM_A_STAGE, &desc_A,
                        next_kt * WARPK, tma_A_row, &mbar[stg]);
            tma_load_2d(smem_B + stg * SMEM_B_STAGE, &desc_B,
                        next_kt * WARPK, tma_B_row_lo, &mbar[stg]);
            tma_load_2d(smem_B + stg * SMEM_B_STAGE + TMA_B_ROWS * WARPK,
                        &desc_B, next_kt * WARPK, tma_B_row_hi, &mbar[stg]);
        }
    }

    // ================================================================
    // Epilogue: Fused RMSNorm (inside M-loop)
    // ================================================================
#ifndef SKIP_NORM
    // Step 1: partial sq_sum per row
    float my_sq[BM];
    #pragma unroll
    for (int r = 0; r < BM; ++r) my_sq[r] = 0.0f;

    #pragma unroll
    for (int mt = 0; mt < WARP_TILE_M; ++mt)
        #pragma unroll
        for (int nt = 0; nt < WARP_TILE_N; ++nt)
            #pragma unroll
            for (int fi = 0; fi < 8; ++fi) {
                int row = mt * 16 + group_id + ((fi & 2) ? 8 : 0);
                my_sq[row] += frag_C[mt][nt][fi] * frag_C[mt][nt][fi];
            }

    // Step 2: reduce within 4-lane group
    float row_sq[4];
    row_sq[0] = group4_reduce(my_sq[group_id]);
    row_sq[1] = group4_reduce(my_sq[group_id + 8]);
    row_sq[2] = group4_reduce(my_sq[16 + group_id]);
    row_sq[3] = group4_reduce(my_sq[24 + group_id]);

    // Step 3: write warp partials to smem
    float* warp_sq = reinterpret_cast<float*>(smem_raw);  // [BM][NUM_WARPS]
    __syncthreads();
    if (tid_in_grp == 0) {
        warp_sq[group_id * NUM_WARPS + warp_id]       = row_sq[0];
        warp_sq[(group_id + 8) * NUM_WARPS + warp_id] = row_sq[1];
        warp_sq[(16 + group_id) * NUM_WARPS + warp_id] = row_sq[2];
        warp_sq[(24 + group_id) * NUM_WARPS + warp_id] = row_sq[3];
    }
    __syncthreads();

    // Step 4: aggregate + rsqrt
    float* scale = warp_sq + BM * NUM_WARPS;
    if (threadIdx.x < BM) {
        float total = 0.0f;
        for (int w = 0; w < NUM_WARPS; ++w)
            total += warp_sq[threadIdx.x * NUM_WARPS + w];
        scale[threadIdx.x] = rsqrtf(total / (float)HEAD_DIM + eps);
    }
    __syncthreads();

    // Step 5: store normalized output
    const int out_base_m = m_block * BM;
    #pragma unroll
    for (int mt = 0; mt < WARP_TILE_M; ++mt)
        #pragma unroll
        for (int nt = 0; nt < WARP_TILE_N; ++nt) {
            int base_col = warp_id * WARP_N + nt * 16;
            #pragma unroll
            for (int fi = 0; fi < 8; ++fi) {
                int row = mt * 16 + group_id + ((fi & 2) ? 8 : 0);
                int col = base_col + (fi >= 4 ? 8 : 0) + tid_in_grp * 2 + (fi & 1);
                int gm = out_base_m + row;
                if (gm < M) {
                    float normed = frag_C[mt][nt][fi] * scale[row] * rms_w[col];
                    gOut[gm * N_TOTAL + head_idx * HEAD_DIM + col] = __float2bfloat16(normed);
                }
            }
        }

#else  // SKIP_NORM
    const int out_base_m = m_block * BM;
    #pragma unroll
    for (int mt = 0; mt < WARP_TILE_M; ++mt)
        #pragma unroll
        for (int nt = 0; nt < WARP_TILE_N; ++nt) {
            int base_col = warp_id * WARP_N + nt * 16;
            #pragma unroll
            for (int fi = 0; fi < 8; ++fi) {
                int row = mt * 16 + group_id + ((fi & 2) ? 8 : 0);
                int col = base_col + (fi >= 4 ? 8 : 0) + tid_in_grp * 2 + (fi & 1);
                int gm = out_base_m + row;
                if (gm < M) {
                    gOut[gm * N_TOTAL + head_idx * HEAD_DIM + col] =
                        __float2bfloat16(frag_C[mt][nt][fi]);
                }
            }
        }
#endif
}

// ============================================================
// Host: TMA descriptor (BF16, SWIZZLE_64B)
// ============================================================
static CUtensorMap make_tma_desc_bf16(
    const __nv_bfloat16* ptr, int rows, int cols,
    int box_rows, int box_cols) {
    CUtensorMap desc{};
    uint64_t globalDim[2]    = {(uint64_t)cols, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)cols * sizeof(__nv_bfloat16)};
    uint32_t boxDim[2]       = {(uint32_t)box_cols, (uint32_t)box_rows};
    uint32_t elemStride[2]   = {1, 1};
    CUresult err = cuTensorMapEncodeTiled(
        &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void*)ptr,
        globalDim, globalStride, boxDim, elemStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (err != CUDA_SUCCESS) {
        const char* s = nullptr; cuGetErrorString(err, &s);
        fprintf(stderr, "TMA desc failed: %s\n", s ? s : "?");
    }
    return desc;
}

// ============================================================
// PyTorch Binding
// ============================================================
torch::Tensor wq_b_proj_gemm(
    torch::Tensor x, torch::Tensor w, torch::Tensor rms_w, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(rms_w.scalar_type() == torch::kFloat32);

    const int M = x.size(0);
    TORCH_CHECK(x.size(1) == K_DIM);
    TORCH_CHECK(w.size(0) == N_TOTAL && w.size(1) == K_DIM);
    TORCH_CHECK(rms_w.numel() == HEAD_DIM);
    TORCH_CHECK(M >= 32 && M <= 256 && M % 32 == 0);

    auto out = torch::empty({M, NUM_HEADS, HEAD_DIM}, x.options().dtype(torch::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();

    auto x_ptr = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto w_ptr = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    auto out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());

    CUtensorMap desc_B = make_tma_desc_bf16(w_ptr, N_TOTAL, K_DIM, TMA_B_ROWS, WARPK);

    // Persistent kernel: grid = NUM_HEADS (128 blocks), M-loop inside kernel
    constexpr int BM = 32;
    constexpr int SMEM_A_STAGE = BM * WARPK;
    CUtensorMap desc_A = make_tma_desc_bf16(x_ptr, M, K_DIM, BM, WARPK);

    dim3 grid(NUM_HEADS, M / BM);  // parallel M-blocks
    dim3 block(NUM_THREADS);
    int smem = STAGES * (SMEM_A_STAGE + SMEM_B_STAGE) * (int)sizeof(__nv_bfloat16) + MBAR_REGION;

    cudaFuncSetAttribute(wq_b_proj_kernel<BM>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    wq_b_proj_kernel<BM><<<grid, block, smem, stream>>>(
        desc_A, desc_B, rms_w.data_ptr<float>(), out_ptr, (float)eps, M);

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wq_b_proj_gemm", &wq_b_proj_gemm,
          "wq_b proj + RMSNorm (TMA + SWIZZLE_64B + PTX mma.sync)");
}
