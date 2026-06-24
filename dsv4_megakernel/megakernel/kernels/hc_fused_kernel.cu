// ============================================================
// DeepSeek-V4 HC Fused Kernel
// Fuses: RMSNorm(no weight) + GEMV + Activation + Sinkhorn + Collapse
// Input: bf16, Weight: bf16, Output: bf16 (matching origin/model.py)
// Launch: block=1024, grid=2*SM, grid-stride loop
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../include/hc_fused_kernel.cuh"

template <int HC, int D, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __launch_bounds__(1024)
hc_fused_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_pos, HC*D] bf16
    const __nv_bfloat16* __restrict__ attn_hc_fn,     // [N_OUT, HC*D] bf16
    const float* __restrict__ attn_hc_base,           // [N_OUT] fp32
    const float* __restrict__ attn_hc_scale,          // [3] fp32
    float hc_eps,
    float rms_norm_eps,
    int num_positions,
    __nv_bfloat16* __restrict__ collapsed_out,        // [num_pos, D] bf16
    float* __restrict__ pre_out,                      // [num_pos, HC]
    float* __restrict__ post_out,                     // [num_pos, HC]
    float* __restrict__ comb_out                      // [num_pos, HC*HC]
) {
    constexpr int HC_D_TOTAL = HC * D;
    constexpr int ELEMS_PER_THR = HC_D_TOTAL / BLOCK_SIZE;  // 7
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;              // 32

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    for (int pos_idx = blockIdx.x; pos_idx < num_positions; pos_idx += gridDim.x) {

    const __nv_bfloat16* hs_ptr = hidden_states + pos_idx * HC_D_TOTAL;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * D;
    float* pre_ptr = pre_out + pos_idx * HC;
    float* post_ptr = post_out + pos_idx * HC;
    float* comb_ptr = comb_out + pos_idx * HC * HC;

    extern __shared__ float smem[];
    float* reduce_smem = smem;
    float* gemv_smem = smem + NUM_WARPS;
    float* mix_smem = gemv_smem + NUM_WARPS * N_OUT;
    float* pre_smem = mix_smem + N_OUT;
    float* post_smem = pre_smem + HC;
    float* comb_smem = post_smem + HC;

    // ================================================================
    // Phase 1: Load bf16 input → fp32, RMSNorm (no weight)
    // ================================================================
    float orig_vals[ELEMS_PER_THR];
    float norm_vals[ELEMS_PER_THR];
    int base_idx = tid * ELEMS_PER_THR;

    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        float v = bf16_to_float(hs_ptr[base_idx + i]);
        orig_vals[i] = v;
        sq_sum += v * v;
    }

    float total_sq = block_reduce_sum<NUM_WARPS>(sq_sum, reduce_smem, tid);

    __shared__ float rms_scale;
    if (tid == 0) {
        rms_scale = rsqrtf(total_sq / (float)HC_D_TOTAL + rms_norm_eps);
    }
    __syncthreads();

    float scale = rms_scale;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        norm_vals[i] = orig_vals[i] * scale;
    }

    // ================================================================
    // Phase 2: GEMV - flat[7168] × attn_hc_fn^T → mix[24]
    // Weight is bf16, compute in fp32
    // ================================================================
    float acc[N_OUT];
    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = 0.0f;

    #pragma unroll
    for (int n = 0; n < N_OUT; n++) {
        const __nv_bfloat16* w_row = attn_hc_fn + n * HC_D_TOTAL + base_idx;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THR; i++) {
            acc[n] += norm_vals[i] * bf16_to_float(w_row[i]);
        }
    }

    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = warp_reduce_sum(acc[n]);

    if (lane_id == 0) {
        #pragma unroll
        for (int n = 0; n < N_OUT; n++) gemv_smem[warp_id * N_OUT + n] = acc[n];
    }
    __syncthreads();

    if (tid < N_OUT) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) sum += gemv_smem[w * N_OUT + tid];
        mix_smem[tid] = sum;
    }
    __syncthreads();

    // ================================================================
    // Phase 3: Activation → pre[4], post[4], comb logits[4,4]
    // ================================================================
    __shared__ float s_pre_scale, s_post_scale, s_comb_scale;
    if (tid == 0) {
        s_pre_scale = attn_hc_scale[0];
        s_post_scale = attn_hc_scale[1];
        s_comb_scale = attn_hc_scale[2];
    }
    __syncthreads();

    if (tid < HC) {
        pre_smem[tid] = fast_sigmoid(mix_smem[tid] * s_pre_scale + attn_hc_base[tid]) + hc_eps;
    } else if (tid < 2 * HC) {
        int idx = tid - HC;
        post_smem[idx] = 2.0f * fast_sigmoid(mix_smem[HC + idx] * s_post_scale + attn_hc_base[HC + idx]);
    }
    if (tid < HC * HC) {
        comb_smem[tid] = mix_smem[2 * HC + tid] * s_comb_scale + attn_hc_base[2 * HC + tid];
    }
    __syncthreads();

    // ================================================================
    // Phase 4: Softmax + Sinkhorn (tid==0, directly on comb_smem)
    // ================================================================
    if (tid == 0) {
        #pragma unroll
        for (int row = 0; row < HC; row++) {
            float max_val = comb_smem[row * HC];
            for (int col = 1; col < HC; col++)
                max_val = fmaxf(max_val, comb_smem[row * HC + col]);
            float row_sum = 0.0f;
            for (int col = 0; col < HC; col++) {
                comb_smem[row * HC + col] = expf(comb_smem[row * HC + col] - max_val);
                row_sum += comb_smem[row * HC + col];
            }
            for (int col = 0; col < HC; col++)
                comb_smem[row * HC + col] = comb_smem[row * HC + col] / row_sum + hc_eps;
        }
        #pragma unroll
        for (int col = 0; col < HC; col++) {
            float col_sum = 0.0f;
            for (int row = 0; row < HC; row++) col_sum += comb_smem[row * HC + col];
            for (int row = 0; row < HC; row++) comb_smem[row * HC + col] /= (col_sum + hc_eps);
        }
        for (int iter = 0; iter < SINKHORN_ITERS - 1; iter++) {
            for (int row = 0; row < HC; row++) {
                float row_sum = 0.0f;
                for (int col = 0; col < HC; col++) row_sum += comb_smem[row * HC + col];
                for (int col = 0; col < HC; col++) comb_smem[row * HC + col] /= (row_sum + hc_eps);
            }
            for (int col = 0; col < HC; col++) {
                float col_sum = 0.0f;
                for (int row = 0; row < HC; row++) col_sum += comb_smem[row * HC + col];
                for (int row = 0; row < HC; row++) comb_smem[row * HC + col] /= (col_sum + hc_eps);
            }
        }
    }
    __syncthreads();

    // Write pre, post, comb to global
    if (tid < HC) { pre_ptr[tid] = pre_smem[tid]; post_ptr[tid] = post_smem[tid]; }
    if (tid < HC * HC) { comb_ptr[tid] = comb_smem[tid]; }

    // ================================================================
    // Phase 5: Collapse = sum_h(pre[h] * orig_vals[h, d]) → bf16
    // ================================================================
    constexpr int THREADS_PER_HC = BLOCK_SIZE / HC;
    int h_idx = tid / THREADS_PER_HC;
    int local_tid = tid % THREADS_PER_HC;
    float pre_h = pre_smem[h_idx];
    __syncthreads();

    float* col_buf = smem;
    float local_contrib[ELEMS_PER_THR];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) local_contrib[i] = pre_h * orig_vals[i];

    if (h_idx == 0) {
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THR; i++) col_buf[local_tid * ELEMS_PER_THR + i] = local_contrib[i];
    }
    __syncthreads();
    #pragma unroll
    for (int g = 1; g < HC; g++) {
        if (h_idx == g) {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THR; i++) col_buf[local_tid * ELEMS_PER_THR + i] += local_contrib[i];
        }
        __syncthreads();
    }

    // Write output as bf16
    constexpr int OUT_PER_THR = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    #pragma unroll
    for (int i = 0; i < OUT_PER_THR; i++) {
        int idx = tid * OUT_PER_THR + i;
        if (idx < D) col_ptr[idx] = float_to_bf16(col_buf[idx]);
    }

    } // end grid-stride loop
}

// ============================================================
// Host launcher
// ============================================================
void hc_fused_launch(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* attn_hc_fn,
    const float* attn_hc_base,
    const float* attn_hc_scale,
    float hc_eps, float rms_norm_eps,
    int num_positions,
    __nv_bfloat16* collapsed_out,
    float* pre_out, float* post_out, float* comb_out,
    cudaStream_t stream
) {
    constexpr int BLOCK = BLOCK_SIZE_DEFAULT;
    int num_sms = get_num_sms();
    int grid_size = min(2 * num_sms, num_positions);
    int smem_size = D_DEFAULT * sizeof(float);

    hc_fused_kernel<HC_DEFAULT, D_DEFAULT, N_OUT_DEFAULT, BLOCK, SINKHORN_DEFAULT>
        <<<grid_size, BLOCK, smem_size, stream>>>(
        hidden_states, attn_hc_fn, attn_hc_base, attn_hc_scale,
        hc_eps, rms_norm_eps, num_positions,
        collapsed_out, pre_out, post_out, comb_out
    );
}

// ============================================================
// PyTorch binding
// ============================================================
std::vector<torch::Tensor> hc_fused_forward_full(
    torch::Tensor hidden_states,    // [num_pos, HC, D] or [HC, D] bf16
    torch::Tensor attn_hc_fn,       // [N_OUT, HC*D] bf16
    torch::Tensor attn_hc_base,     // [N_OUT] fp32
    torch::Tensor attn_hc_scale,    // [3] fp32
    double hc_eps,
    double rms_norm_eps
) {
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(attn_hc_fn.scalar_type() == torch::kBFloat16);

    const int HC = HC_DEFAULT, D = D_DEFAULT;
    auto hs_flat = hidden_states.contiguous().view({-1, HC * D});
    int num_pos = hs_flat.size(0);

    auto opts_bf16 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kBFloat16);
    auto opts_fp32 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kFloat32);

    auto collapsed = torch::empty({num_pos, D}, opts_bf16);
    auto pre = torch::empty({num_pos, HC}, opts_fp32);
    auto post = torch::empty({num_pos, HC}, opts_fp32);
    auto comb = torch::empty({num_pos, HC, HC}, opts_fp32);

    hc_fused_launch(
        reinterpret_cast<const __nv_bfloat16*>(hs_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_hc_fn.contiguous().data_ptr<at::BFloat16>()),
        attn_hc_base.contiguous().data_ptr<float>(),
        attn_hc_scale.contiguous().data_ptr<float>(),
        (float)hc_eps, (float)rms_norm_eps, num_pos,
        reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr<at::BFloat16>()),
        pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    );

    if (num_pos == 1) return {collapsed.squeeze(0), pre.squeeze(0), post.squeeze(0), comb.squeeze(0)};
    return {collapsed, pre, post, comb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_fused_forward_full", &hc_fused_forward_full,
          "HC fused forward (bf16 in/out, matching origin)");
}
