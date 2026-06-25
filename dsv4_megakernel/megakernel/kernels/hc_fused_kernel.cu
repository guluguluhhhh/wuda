// ============================================================
// DeepSeek-V4 HC Fused Kernel
// Fuses: RMSNorm(no weight) + GEMV + Activation + Sinkhorn + Collapse
// Input: bf16 [num_pos, HC*DIM] = [num_pos, 28672]
// Weight: bf16 [N_OUT, HC*DIM] = [24, 28672] (transposed to [28672, 24])
// Output: bf16 [num_pos, DIM] = [num_pos, 7168]
// Launch: block=1024, grid=2*SM, grid-stride loop
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../include/hc_fused_kernel.cuh"

template <int HC, int DIM, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __launch_bounds__(1024)
hc_fused_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_pos, HC*DIM] bf16
    const __nv_bfloat16* __restrict__ attn_hc_fn,     // [N_OUT, HC*DIM] bf16 row-major
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
    constexpr int HC_DIM_TOTAL = HC * DIM;            // 28672
    constexpr int ELEMS_PER_THR = HC_DIM_TOTAL / BLOCK_SIZE;  // 28
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;        // 32

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    for (int pos_idx = blockIdx.x; pos_idx < num_positions; pos_idx += gridDim.x) {

    // Per-position pointers
    const __nv_bfloat16* hs_ptr = hidden_states + pos_idx * HC_DIM_TOTAL;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * DIM;
    float* pre_ptr = pre_out + pos_idx * HC;
    float* post_ptr = post_out + pos_idx * HC;
    float* comb_ptr = comb_out + pos_idx * HC * HC;

    // Shared memory
    extern __shared__ float smem[];
    float* reduce_smem = smem;
    float* gemv_smem = smem + NUM_WARPS;
    float* mix_smem = gemv_smem + NUM_WARPS * N_OUT;
    float* pre_smem = mix_smem + N_OUT;
    float* post_smem = pre_smem + HC;
    float* comb_smem = post_smem + HC;

    // ================================================================
    // Phase 1: Load bf16 input [HC_DIM_TOTAL=28672] → fp32, RMSNorm
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
        rms_scale = rsqrtf(total_sq / (float)HC_DIM_TOTAL + rms_norm_eps);
    }
    __syncthreads();

    float scale = rms_scale;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        norm_vals[i] = orig_vals[i] * scale;
    }

    // ================================================================
    // Phase 2: GEMV  y[24] = W[24, 28672] @ x[28672]
    // W stored row-major [N_OUT, HC_DIM_TOTAL]. Simple scalar load.
    // ================================================================

    float acc[N_OUT];
    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = 0.0f;

    #pragma unroll
    for (int n = 0; n < N_OUT; n++) {
        const __nv_bfloat16* w_row = attn_hc_fn + n * HC_DIM_TOTAL + base_idx;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THR; i++) {
            acc[n] += norm_vals[i] * bf16_to_float(w_row[i]);
        }
    }

    // Warp reduce
    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = warp_reduce_sum(acc[n]);

    if (lane_id == 0) {
        #pragma unroll
        for (int n = 0; n < N_OUT; n++) gemv_smem[warp_id * N_OUT + n] = acc[n];
    }
    __syncthreads();

    // Cross-warp reduce
    if (tid < N_OUT) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) sum += gemv_smem[w * N_OUT + tid];
        mix_smem[tid] = sum;
    }
    __syncthreads();

    // ================================================================
    // Phase 3: Activation
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
    // Phase 4: Softmax + Sinkhorn
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

    // Write gates to global
    if (tid < HC) { pre_ptr[tid] = pre_smem[tid]; post_ptr[tid] = post_smem[tid]; }
    if (tid < HC * HC) { comb_ptr[tid] = comb_smem[tid]; }

    // ================================================================
    // Phase 5: Collapse  collapsed[d] = sum_h(pre[h] * x[h*DIM + d])
    // Thread mapping: thread i holds flat[i*28..(i+1)*28-1]
    //   h = flat_idx / DIM,  d = flat_idx % DIM
    // Use smem[DIM] to accumulate across HC channels.
    // ================================================================
    constexpr int THREADS_PER_HC = BLOCK_SIZE / HC;  // 256
    int h_idx = tid / THREADS_PER_HC;                // 0..3
    int local_tid = tid % THREADS_PER_HC;            // 0..255
    float pre_h = pre_smem[h_idx];
    __syncthreads();

    // Reuse smem as collapse buffer [DIM] = 7168 floats
    float* col_buf = smem;
    float local_contrib[ELEMS_PER_THR];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) local_contrib[i] = pre_h * orig_vals[i];

    // Each HC channel's threads cover DIM elements (256 threads * 28 elems = 7168 = DIM)
    // d position = local_tid * ELEMS_PER_THR + i
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

    // Write output [DIM] as bf16
    constexpr int OUT_PER_THR = (DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 7
    #pragma unroll
    for (int i = 0; i < OUT_PER_THR; i++) {
        int idx = tid * OUT_PER_THR + i;
        if (idx < DIM) {
            col_ptr[idx] = float_to_bf16(col_buf[idx]);
        }
    }

    } // end grid-stride loop
}

// ============================================================
// Profiled kernel (clock64 per phase)
// ============================================================
template <int HC, int DIM, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __launch_bounds__(1024)
hc_fused_kernel_profiled(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ attn_hc_fn,
    const float* __restrict__ attn_hc_base,
    const float* __restrict__ attn_hc_scale,
    float hc_eps, float rms_norm_eps, int num_positions,
    __nv_bfloat16* __restrict__ collapsed_out,
    float* __restrict__ pre_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    int64_t* __restrict__ timing_buf
) {
    constexpr int HC_DIM_TOTAL = HC * DIM;
    constexpr int ELEMS_PER_THR = HC_DIM_TOTAL / BLOCK_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int pos_idx = blockIdx.x;
    if (pos_idx >= num_positions) return;

    int64_t* my_t = timing_buf + blockIdx.x * 10;
    int64_t t0, t1;
    const __nv_bfloat16* hs_ptr = hidden_states + pos_idx * HC_DIM_TOTAL;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * DIM;
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

    // Phase 1
    if (tid==0) t0=clock64();
    float orig_vals[ELEMS_PER_THR], norm_vals[ELEMS_PER_THR];
    int base_idx = tid * ELEMS_PER_THR;
    float sq_sum = 0.0f;
    for(int i=0;i<ELEMS_PER_THR;i++){float v=bf16_to_float(hs_ptr[base_idx+i]);orig_vals[i]=v;sq_sum+=v*v;}
    float total_sq = block_reduce_sum<NUM_WARPS>(sq_sum, reduce_smem, tid);
    __shared__ float rms_scale;
    if(tid==0) rms_scale=rsqrtf(total_sq/(float)HC_DIM_TOTAL+rms_norm_eps);
    __syncthreads();
    float scale=rms_scale;
    for(int i=0;i<ELEMS_PER_THR;i++) norm_vals[i]=orig_vals[i]*scale;
    if(tid==0){t1=clock64();my_t[0]=t0;my_t[1]=t1;}
    __syncthreads();

    // Phase 2
    if(tid==0) t0=clock64();
    float acc[N_OUT]; for(int n=0;n<N_OUT;n++)acc[n]=0.0f;
    for(int n=0;n<N_OUT;n++){
        const __nv_bfloat16* w_row=attn_hc_fn+n*HC_DIM_TOTAL+base_idx;
        for(int i=0;i<ELEMS_PER_THR;i++) acc[n]+=norm_vals[i]*bf16_to_float(w_row[i]);
    }
    for(int n=0;n<N_OUT;n++) acc[n]=warp_reduce_sum(acc[n]);
    if(lane_id==0){for(int n=0;n<N_OUT;n++)gemv_smem[warp_id*N_OUT+n]=acc[n];}
    __syncthreads();
    if(tid<N_OUT){float s=0;for(int w=0;w<NUM_WARPS;w++)s+=gemv_smem[w*N_OUT+tid];mix_smem[tid]=s;}
    __syncthreads();
    if(tid==0){t1=clock64();my_t[2]=t0;my_t[3]=t1;}
    __syncthreads();

    // Phase 3
    if(tid==0) t0=clock64();
    __shared__ float sp,spo,sc;
    if(tid==0){sp=attn_hc_scale[0];spo=attn_hc_scale[1];sc=attn_hc_scale[2];}
    __syncthreads();
    if(tid<HC) pre_smem[tid]=fast_sigmoid(mix_smem[tid]*sp+attn_hc_base[tid])+hc_eps;
    else if(tid<2*HC){int idx=tid-HC;post_smem[idx]=2.0f*fast_sigmoid(mix_smem[HC+idx]*spo+attn_hc_base[HC+idx]);}
    if(tid<HC*HC) comb_smem[tid]=mix_smem[2*HC+tid]*sc+attn_hc_base[2*HC+tid];
    __syncthreads();
    if(tid==0){t1=clock64();my_t[4]=t0;my_t[5]=t1;}
    __syncthreads();

    // Phase 4
    if(tid==0) t0=clock64();
    if(tid==0){
        for(int r=0;r<HC;r++){float mx=comb_smem[r*HC];for(int c=1;c<HC;c++)mx=fmaxf(mx,comb_smem[r*HC+c]);
            float rs=0;for(int c=0;c<HC;c++){comb_smem[r*HC+c]=expf(comb_smem[r*HC+c]-mx);rs+=comb_smem[r*HC+c];}
            for(int c=0;c<HC;c++)comb_smem[r*HC+c]=comb_smem[r*HC+c]/rs+hc_eps;}
        for(int c=0;c<HC;c++){float cs=0;for(int r=0;r<HC;r++)cs+=comb_smem[r*HC+c];for(int r=0;r<HC;r++)comb_smem[r*HC+c]/=(cs+hc_eps);}
        for(int it=0;it<SINKHORN_ITERS-1;it++){
            for(int r=0;r<HC;r++){float rs=0;for(int c=0;c<HC;c++)rs+=comb_smem[r*HC+c];for(int c=0;c<HC;c++)comb_smem[r*HC+c]/=(rs+hc_eps);}
            for(int c=0;c<HC;c++){float cs=0;for(int r=0;r<HC;r++)cs+=comb_smem[r*HC+c];for(int r=0;r<HC;r++)comb_smem[r*HC+c]/=(cs+hc_eps);}}
    }
    __syncthreads();
    if(tid==0){t1=clock64();my_t[6]=t0;my_t[7]=t1;}
    if(tid<HC){pre_ptr[tid]=pre_smem[tid];post_ptr[tid]=post_smem[tid];}
    if(tid<HC*HC)comb_ptr[tid]=comb_smem[tid];

    // Phase 5
    if(tid==0) t0=clock64();
    constexpr int TPH=BLOCK_SIZE/HC;
    int h_idx=tid/TPH,local_tid=tid%TPH;
    float pre_h=pre_smem[h_idx];
    __syncthreads();
    float* col_buf=smem;
    float lc[ELEMS_PER_THR];for(int i=0;i<ELEMS_PER_THR;i++)lc[i]=pre_h*orig_vals[i];
    if(h_idx==0){for(int i=0;i<ELEMS_PER_THR;i++)col_buf[local_tid*ELEMS_PER_THR+i]=lc[i];}
    __syncthreads();
    for(int g=1;g<HC;g++){if(h_idx==g){for(int i=0;i<ELEMS_PER_THR;i++)col_buf[local_tid*ELEMS_PER_THR+i]+=lc[i];}__syncthreads();}
    constexpr int OPT=(DIM+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(int i=0;i<OPT;i++){int idx=tid*OPT+i;if(idx<DIM)col_ptr[idx]=float_to_bf16(col_buf[idx]);}
    if(tid==0){t1=clock64();my_t[8]=t0;my_t[9]=t1;}
}

void hc_fused_launch_profiled(
    const __nv_bfloat16* hs, const __nv_bfloat16* w,
    const float* base, const float* scale,
    float hc_eps, float rms_eps, int num_pos,
    __nv_bfloat16* col_out, float* pre_out, float* post_out, float* comb_out,
    int64_t* timing_buf, cudaStream_t stream
) {
    constexpr int BLOCK = BLOCK_SIZE_DEFAULT;
    int grid_size = std::min(2 * get_num_sms(), num_pos);
    int smem_size = DIM_DEFAULT * sizeof(float);
    hc_fused_kernel_profiled<HC_DEFAULT,DIM_DEFAULT,N_OUT_DEFAULT,BLOCK,SINKHORN_DEFAULT>
        <<<grid_size,BLOCK,smem_size,stream>>>(hs,w,base,scale,hc_eps,rms_eps,num_pos,col_out,pre_out,post_out,comb_out,timing_buf);
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
    // smem: max(Phase1-4 scratch, Phase5 collapse buf [DIM=7168 floats])
    int smem_size = DIM_DEFAULT * sizeof(float);

    hc_fused_kernel<HC_DEFAULT, DIM_DEFAULT, N_OUT_DEFAULT, BLOCK, SINKHORN_DEFAULT>
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
    torch::Tensor hidden_states,    // [num_pos, HC, DIM] or [HC, DIM] bf16
    torch::Tensor attn_hc_fn,       // [N_OUT, HC*DIM] = [24, 28672] bf16
    torch::Tensor attn_hc_base,     // [N_OUT] fp32
    torch::Tensor attn_hc_scale,    // [3] fp32
    double hc_eps,
    double rms_norm_eps
) {
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(attn_hc_fn.scalar_type() == torch::kBFloat16);

    const int HC = HC_DEFAULT, DIM = DIM_DEFAULT;
    auto hs_flat = hidden_states.contiguous().view({-1, HC * DIM});
    int num_pos = hs_flat.size(0);

    auto opts_bf16 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kBFloat16);
    auto opts_fp32 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kFloat32);

    auto collapsed = torch::empty({num_pos, DIM}, opts_bf16);
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
          "HC fused forward (bf16, row-major weight)");
    m.def("hc_fused_forward_profiled", [](torch::Tensor hs, torch::Tensor w,
          torch::Tensor base, torch::Tensor scale, double eps, double rms_eps) {
        TORCH_CHECK(hs.is_cuda() && hs.scalar_type() == torch::kBFloat16);
        const int HC = HC_DEFAULT, DIM = DIM_DEFAULT;
        auto hs_flat = hs.contiguous().view({-1, HC * DIM});
        int num_pos = hs_flat.size(0);
        int num_sms = get_num_sms();
        int grid_size = std::min(2 * num_sms, num_pos);

        auto opts_bf16 = torch::TensorOptions().device(hs.device()).dtype(torch::kBFloat16);
        auto opts_fp32 = torch::TensorOptions().device(hs.device()).dtype(torch::kFloat32);
        auto opts_i64 = torch::TensorOptions().device(hs.device()).dtype(torch::kInt64);

        auto collapsed = torch::empty({num_pos, DIM}, opts_bf16);
        auto pre = torch::empty({num_pos, HC}, opts_fp32);
        auto post = torch::empty({num_pos, HC}, opts_fp32);
        auto comb = torch::empty({num_pos, HC, HC}, opts_fp32);
        auto timing = torch::zeros({grid_size, 10}, opts_i64);

        hc_fused_launch_profiled(
            reinterpret_cast<const __nv_bfloat16*>(hs_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(w.contiguous().data_ptr<at::BFloat16>()),
            base.contiguous().data_ptr<float>(), scale.contiguous().data_ptr<float>(),
            (float)eps, (float)rms_eps, num_pos,
            reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr<at::BFloat16>()),
            pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
            timing.data_ptr<int64_t>(), at::cuda::getCurrentCUDAStream()
        );
        return timing;
    }, "HC fused profiled (returns timing buffer)");
}
