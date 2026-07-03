// ============================================================
// DeepSeek-V4 HC Fused Kernel - Tensor Core version (Cluster split-K)
// Uses CuTe SM80 m16n8k16 MMA for Phase 2 GEMM
// Fuses: RMSNorm + TC_GEMM + Activation + Sinkhorn + Collapse
//
// Split-K=2 via cluster (same as CC version):
//   - 2 blocks per cluster, each handles K/2=14336
//   - Phase 1: each block computes partial sq_sum, dsmem merge
//   - Phase 2: each block computes partial GEMM [BATCH, K/2] × [K/2, N_PAD], dsmem merge
//   - Phase 5: each block writes DIM/2 output elements
//
// Input:  bf16 [num_pos, HC*DIM] = [num_pos, 28672]
// Weight: bf16 [N_PAD, HC*DIM] = [32, 28672] (pre-padded, pre-transposed to [HC*DIM, N_PAD])
// Output: bf16 [num_pos, DIM] = [num_pos, 7168]
// Launch: cudaLaunchKernelEx, cluster_dim(2,1,1)
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"

#include "../include/hc_fused_kernel_tc.cuh"

using namespace cute;
namespace cg = cooperative_groups;

// ============================================================
// CuTe MMA Configuration
// ============================================================
// MMA atom: SM80_16x8x16_F32BF16BF16F32_TN
//   A: [M=16, K=16] col-major
//   B: [N=8,  K=16] col-major
//   C: [M=16, N=8]
using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;

// TiledMMA: cover full [BATCH=16, N_PAD=32] output
// Atom covers 16×8, so need 4× repeat along N to get 16×32
// Thread layout: <1, 4, 1> replicates atom across N using 4 warps (128 threads)
using TiledMma = TiledMMA<
    MmaAtom,
    Layout<Shape<_1, _4, _1>>  // 1 atom M, 4 atoms N, 1 atom K → 128 threads
>;

// ============================================================
// Main TC Kernel (Cluster split-K=2, 1024 threads/block)
// ============================================================
template <int BATCH, int HC, int DIM, int N_OUT, int N_PAD, int BK,
          int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __attribute__((cluster_dim(2, 1, 1))) __launch_bounds__(1024)
hc_fused_kernel_tc(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_pos, HC*DIM] bf16
    const __nv_bfloat16* __restrict__ weight_t,       // [HC*DIM, N_PAD] bf16 (transposed+padded)
    const float* __restrict__ attn_hc_base,           // [N_OUT] fp32
    const float* __restrict__ attn_hc_scale,          // [3] fp32
    float hc_eps,
    float rms_norm_eps,
    int num_positions,
    __nv_bfloat16* __restrict__ collapsed_out,        // [num_pos, DIM] bf16
    float* __restrict__ pre_out,                      // [num_pos, HC]
    float* __restrict__ post_out,                     // [num_pos, HC]
    float* __restrict__ comb_out                      // [num_pos, HC*HC]
) {
    cg::cluster_group cluster = cg::this_cluster();
    const int block_rank = cluster.block_rank();      // 0 or 1

    constexpr int HC_DIM_TOTAL = HC * DIM;            // 28672
    constexpr int HALF_K = HC_DIM_TOTAL / 2;          // 14336
    constexpr int ELEMS_PER_THR_NORM = HALF_K / BLOCK_SIZE;  // 14
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;       // 32

    const int tid = threadIdx.x;
    const int num_clusters = gridDim.x / 2;
    const int cluster_id = blockIdx.x / 2;

    for (int batch_group = cluster_id; batch_group * BATCH < num_positions; batch_group += num_clusters) {
    const int pos_base = batch_group * BATCH;
    const int actual_batch = min(BATCH, num_positions - pos_base);

    // ============================================================
    // Shared Memory Layout
    // ============================================================
    extern __shared__ char smem_raw[];
    float* reduce_smem = reinterpret_cast<float*>(smem_raw);
    float* partial_sq_slot = reduce_smem + NUM_WARPS;   // [1] for dsmem exchange

    // Phase 2: double-buffered smem (Kstage=2) with PAD on A to eliminate bank conflict
    // Stage layout: [sA_0 (padded)][sB_0][sA_1 (padded)][sB_1]
    // sA per stage: [BATCH+PAD, BK] col-major = (BATCH+PAD)*BK bf16 (PAD=8 for bank conflict)
    // sB per stage: [N_PAD, BK] col-major = N_PAD*BK bf16 (no conflict, stride=32=full bank cycle)
    constexpr int PAD_A = 8;  // 8 bf16 = 16 bytes = 8 banks offset
    constexpr int STAGE_A = (BATCH + PAD_A) * BK;  // 24*256 = 6144 bf16 = 12KB
    constexpr int STAGE_B = N_PAD * BK;            // 32*256 = 8192 bf16 = 16KB
    constexpr int STAGE_TOTAL = STAGE_A + STAGE_B; // 14336 bf16 = 28KB
    // 2 stages = 56KB > 48KB default! Need cudaFuncSetAttribute.
    // Actually: 2*28KB = 56KB. Let's reduce BK or accept extended smem.
    // sm_103 supports up to 228KB smem. Use cudaFuncSetAttribute in launcher.

    __nv_bfloat16* smem_base = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // Phase 3-5: reuse smem
    float* phase35_smem = reinterpret_cast<float*>(smem_raw);

    // ============================================================
    // Phase 1: RMSNorm - split-K, each block loads K/2
    // block_rank==0 -> [0, HALF_K), block_rank==1 -> [HALF_K, HC_DIM_TOTAL)
    // ============================================================
    int half_offset = block_rank * HALF_K;
    float rms_scales[BATCH];

    for (int b = 0; b < actual_batch; b++) {
        const __nv_bfloat16* row_ptr = hidden_states + (pos_base + b) * HC_DIM_TOTAL;

        float sq_sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < ELEMS_PER_THR_NORM; i++) {
            float v = __bfloat162float(row_ptr[half_offset + tid * ELEMS_PER_THR_NORM + i]);
            sq_sum += v * v;
        }

        float partial_sq = block_reduce_sum_tc<NUM_WARPS>(sq_sum, reduce_smem, tid);
        if (tid == 0) partial_sq_slot[0] = partial_sq;
        __syncthreads();
        cluster.sync();

        // Read other block's partial sum via dsmem
        float* remote_sq = cluster.map_shared_rank(partial_sq_slot, 1 - block_rank);
        float total_sq = partial_sq_slot[0] + *remote_sq;
        rms_scales[b] = rsqrtf(total_sq / (float)HC_DIM_TOTAL + rms_norm_eps);
        __syncthreads();
    }

    // ============================================================
    // Phase 2: TC GEMM with cp.async double buffer
    // RMS normalization moved to epilogue: result[b,n] *= rms_scale[b]
    // Both A and B loaded as pure memcpy (no per-element compute in K-loop)
    // BK=256, Kstage=2, K_ITERS=56 per block (HALF_K/BK)
    // ============================================================
    constexpr int K_ITERS = HALF_K / BK;  // 56

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid % MMA_THREADS);

    auto sA_layout = make_layout(make_shape(Int<BATCH>{}, Int<BK>{}),
                                  make_stride(Int<1>{}, Int<BATCH + PAD_A>{}));  // padded stride
    auto sB_layout = make_layout(make_shape(Int<N_PAD>{}, Int<BK>{}),
                                  make_stride(Int<1>{}, Int<N_PAD>{}));
    auto sC_layout = make_layout(make_shape(Int<BATCH>{}, Int<N_PAD>{}),
                                  make_stride(Int<1>{}, Int<BATCH>{}));
    auto sC_ref = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem_raw)), sC_layout);
    auto tCrC = thr_mma.partition_fragment_C(sC_ref);
    clear(tCrC);

    // Helper: load A tile to smem col-major with PAD (all 1024 threads, register-based)
    // A global: row-major [num_pos, HC_DIM_TOTAL], tile [BATCH, BK]
    // smem_A col-major padded: element (m,k) at addr m + k*(BATCH+PAD_A)
    auto load_A_tile = [&](int stage, int k_off) {
        __nv_bfloat16* sA = smem_base + stage * STAGE_TOTAL;
        constexpr int A_EPT = (BATCH * BK) / BLOCK_SIZE;  // 4096/1024 = 4
        #pragma unroll
        for (int i = 0; i < A_EPT; i++) {
            int flat = tid * A_EPT + i;
            int row = flat / BK;   // batch idx (0..15)
            int col = flat % BK;   // k idx (0..255)
            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (row < actual_batch)
                val = hidden_states[(pos_base + row) * HC_DIM_TOTAL + k_off + col];
            sA[row + col * (BATCH + PAD_A)] = val;  // padded col-major
        }
    };

    // Helper: cp.async B tile to smem col-major
    // B global: row-major [HC_DIM_TOTAL, N_PAD], tile at row k_off, cols [0, N_PAD)
    // For col-major smem [N_PAD, BK]: element (n,k) at addr n + k*N_PAD
    // Global B[(k_off+k)*N_PAD + n] maps to smem[n + k*N_PAD] — same linear order!
    // So we can cp.async contiguous 16B chunks along the N dimension
    auto load_B_tile_async = [&](int stage, int k_off) {
        __nv_bfloat16* sB = smem_base + stage * STAGE_TOTAL + STAGE_A;
        // Total bytes: N_PAD * BK * 2 = 32*256*2 = 16384B
        // 1024 threads, each copies 16B (= 8 bf16)
        // Total 16B chunks: 16384/16 = 1024 → exactly 1 chunk per thread
        constexpr int CHUNKS_TOTAL = (N_PAD * BK * 2) / 16;  // 1024
        constexpr int CHUNKS_PER_THR = CHUNKS_TOTAL / BLOCK_SIZE;  // 1
        #pragma unroll
        for (int i = 0; i < CHUNKS_PER_THR; i++) {
            int chunk_id = tid * CHUNKS_PER_THR + i;
            // Each chunk = 16 bytes = 8 bf16 elements
            // In the flat smem layout, chunk_id*8 gives the smem offset
            // In global, same linear offset since layout matches
            const void* gmem_ptr = reinterpret_cast<const void*>(
                &weight_t[k_off * N_PAD + chunk_id * 8]);
            void* smem_ptr = reinterpret_cast<void*>(&sB[chunk_id * 8]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         : : "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),
                             "l"(gmem_ptr));
        }
    };

    // === Prologue: load stage 0 ===
    load_A_tile(0, half_offset + 0 * BK);
    load_B_tile_async(0, half_offset + 0 * BK);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // === Mainloop: double-buffered K-loop ===
    for (int k_iter = 0; k_iter < K_ITERS; k_iter++) {
        int cur = k_iter % 2;
        int nxt = 1 - cur;
        int k_off_cur = half_offset + k_iter * BK;
        int k_off_nxt = half_offset + (k_iter + 1) * BK;

        // Issue next stage load (overlaps with MMA below)
        if (k_iter < K_ITERS - 1) {
            load_A_tile(nxt, k_off_nxt);
            load_B_tile_async(nxt, k_off_nxt);
            asm volatile("cp.async.commit_group;\n");
        }

        // MMA on current stage (first 128 threads)
        __nv_bfloat16* sA_cur = smem_base + cur * STAGE_TOTAL;
        __nv_bfloat16* sB_cur = smem_base + cur * STAGE_TOTAL + STAGE_A;
        if (tid < MMA_THREADS) {
            auto sA_tensor = make_tensor(make_smem_ptr(sA_cur), sA_layout);
            auto sB_tensor = make_tensor(make_smem_ptr(sB_cur), sB_layout);
            auto tCsA = thr_mma.partition_A(sA_tensor);
            auto tCsB = thr_mma.partition_B(sB_tensor);
            cute::gemm(tiled_mma, tCsA, tCsB, tCrC);
        }

        // Wait for next stage to be ready
        if (k_iter < K_ITERS - 1) {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
    }

    // === Epilogue: multiply acc by rms_scale (per-row) ===
    // Store acc to smem, apply scale, then keep in smem for Phase 3
    float* acc_slot = reinterpret_cast<float*>(smem_raw);  // reuse smem
    if (tid < MMA_THREADS) {
        auto sC_out = make_tensor(make_smem_ptr(acc_slot), sC_layout);
        auto tCsC = thr_mma.partition_C(sC_out);
        copy(tCrC, tCsC);
    }
    __syncthreads();

    // acc is col-major [BATCH, N_PAD]: element (b,n) at acc_slot[b + n*BATCH]
    // Multiply each row by rms_scales[b]
    for (int idx = tid; idx < actual_batch * N_PAD; idx += BLOCK_SIZE) {
        int b = idx % BATCH;    // col-major: b is fast dim
        int n = idx / BATCH;    // n is slow dim
        acc_slot[idx] *= rms_scales[b];
    }
    __syncthreads();

    // Merge with other cluster block via dsmem
    cluster.sync();

    // block_rank==0 merges both halves via dsmem
    float* gemm_result = phase35_smem;
    if (block_rank == 0) {
        float* remote_acc = cluster.map_shared_rank(acc_slot, 1);
        for (int idx = tid; idx < BATCH * N_PAD; idx += BLOCK_SIZE) {
            gemm_result[idx] = acc_slot[idx] + remote_acc[idx];
        }
    }
    if (block_rank == 0) __syncthreads();

    // Broadcast gemm_result to block_rank==1
    cluster.sync();
    if (block_rank == 1) {
        float* remote_result = cluster.map_shared_rank(gemm_result, 0);
        for (int idx = tid; idx < BATCH * N_PAD; idx += BLOCK_SIZE) {
            gemm_result[idx] = remote_result[idx];
        }
    }
    __syncthreads();

    // ============================================================
    // Phase 3: Activation (per position, independent)
    // gemm_result[b, 0:N_OUT] → pre[b,HC], post[b,HC], comb[b,HC*HC]
    // ============================================================
    float s_pre_scale, s_post_scale, s_comb_scale;
    // All threads read scales directly from global (avoids smem conflict with gemm_result)
    s_pre_scale = attn_hc_scale[0];
    s_post_scale = attn_hc_scale[1];
    s_comb_scale = attn_hc_scale[2];

    // Layout: gemm_result stored col-major → gemm_result[b + n * BATCH] for element (b, n)
    // pre[b,h]  = sigmoid(gemm[b,h] * scale[0] + base[h]) + eps,     h=0..3
    // post[b,h] = 2*sigmoid(gemm[b, HC+h] * scale[1] + base[HC+h]),  h=0..3
    // comb[b,i] = gemm[b, 2*HC+i] * scale[2] + base[2*HC+i],        i=0..15

    // Use threads to process batch×channels in parallel
    // Total work: BATCH * N_OUT = 16 * 24 = 384, threads=128 → 3 per thread
    float* pre_smem = phase35_smem + BATCH * N_PAD;    // [BATCH, HC]
    float* post_smem = pre_smem + BATCH * HC;          // [BATCH, HC]
    float* comb_smem = post_smem + BATCH * HC;         // [BATCH, HC*HC]

    // Each thread handles multiple (batch, channel) pairs
    // Col-major access: element (b, n) = gemm_result[b + n * BATCH]
    for (int idx = tid; idx < actual_batch * HC; idx += BLOCK_SIZE) {
        int b = idx / HC;
        int h = idx % HC;
        float val = gemm_result[b + h * BATCH];
        pre_smem[b * HC + h] = fast_sigmoid_tc(val * s_pre_scale + attn_hc_base[h]) + hc_eps;
    }
    for (int idx = tid; idx < actual_batch * HC; idx += BLOCK_SIZE) {
        int b = idx / HC;
        int h = idx % HC;
        float val = gemm_result[b + (HC + h) * BATCH];
        post_smem[b * HC + h] = 2.0f * fast_sigmoid_tc(val * s_post_scale + attn_hc_base[HC + h]);
    }
    for (int idx = tid; idx < actual_batch * HC * HC; idx += BLOCK_SIZE) {
        int b = idx / (HC * HC);
        int i = idx % (HC * HC);
        float val = gemm_result[b + (2 * HC + i) * BATCH];
        comb_smem[b * HC * HC + i] = val * s_comb_scale + attn_hc_base[2 * HC + i];
    }
    __syncthreads();

    // ============================================================
    // Phase 4: Softmax + Sinkhorn (per position)
    // Each position has a 4×4 comb matrix
    // With BATCH=16, use 16 threads (one per position) or tid<BATCH
    // ============================================================
    if (tid < actual_batch) {
        int b = tid;
        float* cm = comb_smem + b * HC * HC;

        // Row-wise softmax
        #pragma unroll
        for (int row = 0; row < HC; row++) {
            float max_val = cm[row * HC];
            for (int col = 1; col < HC; col++)
                max_val = fmaxf(max_val, cm[row * HC + col]);
            float row_sum = 0.0f;
            for (int col = 0; col < HC; col++) {
                cm[row * HC + col] = expf(cm[row * HC + col] - max_val);
                row_sum += cm[row * HC + col];
            }
            for (int col = 0; col < HC; col++)
                cm[row * HC + col] = cm[row * HC + col] / row_sum + hc_eps;
        }

        // Column normalization
        #pragma unroll
        for (int col = 0; col < HC; col++) {
            float col_sum = 0.0f;
            for (int row = 0; row < HC; row++) col_sum += cm[row * HC + col];
            for (int row = 0; row < HC; row++) cm[row * HC + col] /= (col_sum + hc_eps);
        }

        // Sinkhorn iterations
        for (int iter = 0; iter < SINKHORN_ITERS - 1; iter++) {
            for (int row = 0; row < HC; row++) {
                float row_sum = 0.0f;
                for (int col = 0; col < HC; col++) row_sum += cm[row * HC + col];
                for (int col = 0; col < HC; col++) cm[row * HC + col] /= (row_sum + hc_eps);
            }
            for (int col = 0; col < HC; col++) {
                float col_sum = 0.0f;
                for (int row = 0; row < HC; row++) col_sum += cm[row * HC + col];
                for (int row = 0; row < HC; row++) cm[row * HC + col] /= (col_sum + hc_eps);
            }
        }
    }
    __syncthreads();

    // Write gates to global memory
    for (int idx = tid; idx < actual_batch * HC; idx += BLOCK_SIZE) {
        int b = idx / HC, h = idx % HC;
        pre_out[(pos_base + b) * HC + h] = pre_smem[b * HC + h];
        post_out[(pos_base + b) * HC + h] = post_smem[b * HC + h];
    }
    for (int idx = tid; idx < actual_batch * HC * HC; idx += BLOCK_SIZE) {
        int b = idx / (HC * HC), i = idx % (HC * HC);
        comb_out[(pos_base + b) * HC * HC + i] = comb_smem[b * HC * HC + i];
    }

    // ============================================================
    // Phase 5: Collapse - split DIM, both blocks in parallel (like CC)
    // block_rank==0 -> output[0, DIM/2), block_rank==1 -> output[DIM/2, DIM)
    // ============================================================
    constexpr int HALF_DIM = DIM / 2;                    // 3584
    constexpr int THREADS_PER_HC = BLOCK_SIZE / HC;      // 256
    constexpr int ELEMS_COL = HALF_DIM / THREADS_PER_HC; // 14
    int h_idx = tid / THREADS_PER_HC;                    // 0..3
    int col_local = tid % THREADS_PER_HC;                // 0..255
    int d_offset = block_rank * HALF_DIM;                // 0 or 3584

    // Save pre gates to registers BEFORE col_buf overwrites smem
    float pre_regs[BATCH];
    for (int b = 0; b < actual_batch; b++) {
        pre_regs[b] = pre_smem[b * HC + h_idx];
    }
    __syncthreads();

    float* col_buf = reinterpret_cast<float*>(smem_raw);

    for (int b = 0; b < actual_batch; b++) {
        const __nv_bfloat16* hs_ptr = hidden_states + (pos_base + b) * HC_DIM_TOTAL;
        __nv_bfloat16* out_ptr = collapsed_out + (pos_base + b) * DIM;
        float pre_h = pre_regs[b];

        float col_contrib[ELEMS_COL];
        #pragma unroll
        for (int i = 0; i < ELEMS_COL; i++) {
            int d = d_offset + col_local * ELEMS_COL + i;
            col_contrib[i] = pre_h * __bfloat162float(hs_ptr[h_idx * DIM + d]);
        }

        if (h_idx == 0) {
            #pragma unroll
            for (int i = 0; i < ELEMS_COL; i++)
                col_buf[col_local * ELEMS_COL + i] = col_contrib[i];
        }
        __syncthreads();
        #pragma unroll
        for (int g = 1; g < HC; g++) {
            if (h_idx == g) {
                #pragma unroll
                for (int i = 0; i < ELEMS_COL; i++)
                    col_buf[col_local * ELEMS_COL + i] += col_contrib[i];
            }
            __syncthreads();
        }

        for (int d = tid; d < HALF_DIM; d += BLOCK_SIZE) {
            out_ptr[d_offset + d] = __float2bfloat16(col_buf[d]);
        }
        __syncthreads();
    }

    cluster.sync();  // end-of-iteration barrier
    } // end grid-stride loop
}

// ============================================================
// Host launcher (cudaLaunchKernelEx with cluster)
// ============================================================
void hc_fused_tc_launch(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* weight_t,
    const float* attn_hc_base,
    const float* attn_hc_scale,
    float hc_eps, float rms_norm_eps,
    int num_positions,
    __nv_bfloat16* collapsed_out,
    float* pre_out, float* post_out, float* comb_out,
    cudaStream_t stream
) {
    constexpr int BATCH = BATCH_DEFAULT;
    constexpr int BLOCK = BLOCK_SIZE_TC;

    int num_sms = get_num_sms_tc();
    int num_batch_groups = (num_positions + BATCH - 1) / BATCH;
    int num_clusters = std::min(2 * num_sms, num_batch_groups);
    int grid_size = num_clusters * 2;  // 2 blocks per cluster

    // smem: 56KB (2 stages × 28KB, padded A + B)
    int smem_size = SMEM_GEMM_BYTES;  // 56KB

    // Request extended smem (>48KB default)
    auto kernel_fn = hc_fused_kernel_tc<BATCH, HC_TC, DIM_TC, N_OUT_TC, N_PAD_TC, BK_TC, BLOCK, SINKHORN_TC>;
    cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid_size;
    config.blockDim = BLOCK;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, kernel_fn,
        hidden_states, weight_t, attn_hc_base, attn_hc_scale,
        hc_eps, rms_norm_eps, num_positions,
        collapsed_out, pre_out, post_out, comb_out);
}

// ============================================================
// PyTorch binding
// ============================================================
std::vector<torch::Tensor> hc_fused_tc_forward(
    torch::Tensor hidden_states,    // [num_pos, HC, DIM] or [num_pos, HC*DIM] bf16
    torch::Tensor attn_hc_fn,       // [N_OUT, HC*DIM] = [24, 28672] bf16 (original, will pad+transpose)
    torch::Tensor attn_hc_base,     // [N_OUT] fp32
    torch::Tensor attn_hc_scale,    // [3] fp32
    double hc_eps,
    double rms_norm_eps
) {
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(attn_hc_fn.scalar_type() == torch::kBFloat16);

    constexpr int HC = HC_TC, DIM = DIM_TC, N_OUT = N_OUT_TC, N_PAD = N_PAD_TC;

    auto hs_flat = hidden_states.contiguous().view({-1, HC * DIM});
    int num_pos = hs_flat.size(0);

    auto opts_bf16 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kBFloat16);
    auto opts_fp32 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kFloat32);

    auto collapsed = torch::empty({num_pos, DIM}, opts_bf16);
    auto pre = torch::empty({num_pos, HC}, opts_fp32);
    auto post = torch::empty({num_pos, HC}, opts_fp32);
    auto comb = torch::empty({num_pos, HC, HC}, opts_fp32);

    // Pad weight from [N_OUT=24, K=28672] to [N_PAD=32, K=28672] then transpose
    // Result: [K=28672, N_PAD=32] row-major (= [HC*DIM, N_PAD])
    auto w_padded = torch::zeros({N_PAD, HC * DIM}, opts_bf16);
    w_padded.narrow(0, 0, N_OUT).copy_(attn_hc_fn.contiguous());
    auto w_t = w_padded.t().contiguous();  // [HC*DIM, N_PAD]

    hc_fused_tc_launch(
        reinterpret_cast<const __nv_bfloat16*>(hs_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(w_t.data_ptr<at::BFloat16>()),
        attn_hc_base.contiguous().data_ptr<float>(),
        attn_hc_scale.contiguous().data_ptr<float>(),
        (float)hc_eps, (float)rms_norm_eps, num_pos,
        reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr<at::BFloat16>()),
        pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    );

    return {collapsed, pre, post, comb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_fused_tc_forward", &hc_fused_tc_forward,
          "HC fused forward - Tensor Core version (bf16, batched, N padded to 32)");
}
