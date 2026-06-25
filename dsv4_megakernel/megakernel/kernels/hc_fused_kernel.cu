// ============================================================
// DeepSeek-V4 HC Fused Kernel (CuTe C++ version)
// Fuses: RMSNorm(no weight) + GEMV + Activation + Sinkhorn + Collapse
// Input: bf16, Weight: bf16 (transposed [HC_D, N_OUT]), Output: bf16
// Uses CuTe for tensor layout + vectorized copy
// Launch: block=1024, grid=2*SM, grid-stride loop
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../include/hc_fused_kernel.cuh"

using namespace cute;

template <int HC, int D, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __launch_bounds__(1024)
hc_fused_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_pos, HC*D] bf16
    const __nv_bfloat16* __restrict__ attn_hc_fn_t,   // [HC*D, N_OUT] bf16 transposed
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
    constexpr int HC_D_TOTAL = HC * D;                // 7168
    constexpr int ELEMS_PER_THR = HC_D_TOTAL / BLOCK_SIZE;  // 7
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;        // 32

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    for (int pos_idx = blockIdx.x; pos_idx < num_positions; pos_idx += gridDim.x) {

    // Per-position pointers
    const __nv_bfloat16* hs_ptr = hidden_states + pos_idx * HC_D_TOTAL;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * D;
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
    // Phase 1: Load bf16 input via CuTe → fp32, RMSNorm (no weight)
    // ================================================================

    // CuTe: 1D global tensor [HC_D_TOTAL] bf16
    auto gX = make_tensor(make_gmem_ptr(hs_ptr), make_shape(Int<HC_D_TOTAL>{}));

    float orig_vals[ELEMS_PER_THR];
    float norm_vals[ELEMS_PER_THR];
    int base_idx = tid * ELEMS_PER_THR;

    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        float v = bf16_to_float(gX(base_idx + i));  // CuTe tensor indexing
        orig_vals[i] = v;
        sq_sum += v * v;
    }

    // Block reduce for RMSNorm
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
    // Phase 2: GEMV using CuTe layout for vectorized weight access
    // Weight transposed: [HC_D, N_OUT] = [7168, 24] row-major
    // ================================================================

    // CuTe: 2D weight tensor [K, N]
    auto gW = make_tensor(
        make_gmem_ptr(attn_hc_fn_t),
        make_layout(make_shape(Int<HC_D_TOTAL>{}, Int<N_OUT>{}),
                    make_stride(Int<N_OUT>{}, Int<1>{}))  // row k: 24 contiguous bf16
    );

    float acc[N_OUT];
    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = 0.0f;

    // For each K position this thread owns, load 24 weights and FMA
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        int k = base_idx + i;
        float x = norm_vals[i];

        // CuTe: &gW(k, 0) gives the address of row k, load 24 bf16 via int4
        const int4* w_vec = reinterpret_cast<const int4*>(&gW(k, Int<0>{}));
        int4 v0 = w_vec[0], v1 = w_vec[1], v2 = w_vec[2];

        __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&v0);
        __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&v1);
        __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&v2);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[j*2]     += x * __bfloat162float(__low2bfloat16(p0[j]));
            acc[j*2+1]   += x * __bfloat162float(__high2bfloat16(p0[j]));
            acc[8+j*2]   += x * __bfloat162float(__low2bfloat16(p1[j]));
            acc[8+j*2+1] += x * __bfloat162float(__high2bfloat16(p1[j]));
            acc[16+j*2]  += x * __bfloat162float(__low2bfloat16(p2[j]));
            acc[16+j*2+1]+= x * __bfloat162float(__high2bfloat16(p2[j]));
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
    // Phase 5: Collapse via CuTe (output vectorized store)
    // ================================================================
    constexpr int THREADS_PER_HC = BLOCK_SIZE / HC;  // 256
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

    // CuTe: vectorized bf16 store for output
    // Define output tensor and partition
    auto gOut = make_tensor(make_gmem_ptr(col_ptr), make_shape(Int<D>{}));
    constexpr int OUT_PER_THR = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 2
    #pragma unroll
    for (int i = 0; i < OUT_PER_THR; i++) {
        int idx = tid * OUT_PER_THR + i;
        if (idx < D) {
            gOut(idx) = float_to_bf16(col_buf[idx]);
        }
    }

    } // end grid-stride loop
}

// ============================================================
// Profiled kernel (clock64 instrumentation per phase)
// ============================================================
template <int HC, int D, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __launch_bounds__(1024)
hc_fused_kernel_profiled(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ attn_hc_fn_t,
    const float* __restrict__ attn_hc_base,
    const float* __restrict__ attn_hc_scale,
    float hc_eps, float rms_norm_eps, int num_positions,
    __nv_bfloat16* __restrict__ collapsed_out,
    float* __restrict__ pre_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    int64_t* __restrict__ timing_buf  // [grid_size, 10]
) {
    constexpr int HC_D_TOTAL = HC * D;
    constexpr int ELEMS_PER_THR = HC_D_TOTAL / BLOCK_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int pos_idx = blockIdx.x;
    if (pos_idx >= num_positions) return;

    int64_t* my_t = timing_buf + blockIdx.x * 10;
    int64_t t0, t1;
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

    // Phase 1: RMSNorm
    if (tid==0) t0 = clock64();
    auto gX = make_tensor(make_gmem_ptr(hs_ptr), make_shape(Int<HC_D_TOTAL>{}));
    float orig_vals[ELEMS_PER_THR], norm_vals[ELEMS_PER_THR];
    int base_idx = tid * ELEMS_PER_THR;
    float sq_sum = 0.0f;
    for (int i=0;i<ELEMS_PER_THR;i++){float v=bf16_to_float(gX(base_idx+i));orig_vals[i]=v;sq_sum+=v*v;}
    float total_sq = block_reduce_sum<NUM_WARPS>(sq_sum, reduce_smem, tid);
    __shared__ float rms_scale;
    if (tid==0) rms_scale = rsqrtf(total_sq/(float)HC_D_TOTAL + rms_norm_eps);
    __syncthreads();
    float scale = rms_scale;
    for (int i=0;i<ELEMS_PER_THR;i++) norm_vals[i]=orig_vals[i]*scale;
    if (tid==0){t1=clock64();my_t[0]=t0;my_t[1]=t1;}
    __syncthreads();

    // Phase 2: GEMV
    if (tid==0) t0 = clock64();
    auto gW = make_tensor(make_gmem_ptr(attn_hc_fn_t),
        make_layout(make_shape(Int<HC_D_TOTAL>{},Int<N_OUT>{}),make_stride(Int<N_OUT>{},Int<1>{})));
    float acc[N_OUT]; for(int n=0;n<N_OUT;n++)acc[n]=0.0f;
    for(int i=0;i<ELEMS_PER_THR;i++){
        int k=base_idx+i;float x=norm_vals[i];
        const int4* wv=reinterpret_cast<const int4*>(&gW(k,Int<0>{}));
        int4 v0=wv[0],v1=wv[1],v2=wv[2];
        __nv_bfloat162* p0=reinterpret_cast<__nv_bfloat162*>(&v0);
        __nv_bfloat162* p1=reinterpret_cast<__nv_bfloat162*>(&v1);
        __nv_bfloat162* p2=reinterpret_cast<__nv_bfloat162*>(&v2);
        for(int j=0;j<4;j++){
            acc[j*2]+=x*__bfloat162float(__low2bfloat16(p0[j]));acc[j*2+1]+=x*__bfloat162float(__high2bfloat16(p0[j]));
            acc[8+j*2]+=x*__bfloat162float(__low2bfloat16(p1[j]));acc[8+j*2+1]+=x*__bfloat162float(__high2bfloat16(p1[j]));
            acc[16+j*2]+=x*__bfloat162float(__low2bfloat16(p2[j]));acc[16+j*2+1]+=x*__bfloat162float(__high2bfloat16(p2[j]));
        }
    }
    for(int n=0;n<N_OUT;n++)acc[n]=warp_reduce_sum(acc[n]);
    if(lane_id==0){for(int n=0;n<N_OUT;n++)gemv_smem[warp_id*N_OUT+n]=acc[n];}
    __syncthreads();
    if(tid<N_OUT){float s=0;for(int w=0;w<NUM_WARPS;w++)s+=gemv_smem[w*N_OUT+tid];mix_smem[tid]=s;}
    __syncthreads();
    if(tid==0){t1=clock64();my_t[2]=t0;my_t[3]=t1;}
    __syncthreads();

    // Phase 3: Activation
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

    // Phase 4: Sinkhorn
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

    // Phase 5: Collapse
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
    auto gOut=make_tensor(make_gmem_ptr(col_ptr),make_shape(Int<D>{}));
    constexpr int OPT=(D+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(int i=0;i<OPT;i++){int idx=tid*OPT+i;if(idx<D)gOut(idx)=float_to_bf16(col_buf[idx]);}
    if(tid==0){t1=clock64();my_t[8]=t0;my_t[9]=t1;}
}

void hc_fused_launch_profiled(
    const __nv_bfloat16* hs, const __nv_bfloat16* w_t,
    const float* base, const float* scale,
    float hc_eps, float rms_eps, int num_pos,
    __nv_bfloat16* col_out, float* pre_out, float* post_out, float* comb_out,
    int64_t* timing_buf, cudaStream_t stream
) {
    constexpr int BLOCK = BLOCK_SIZE_DEFAULT;
    int grid_size = min(2 * get_num_sms(), num_pos);
    int smem_size = D_DEFAULT * sizeof(float);
    hc_fused_kernel_profiled<HC_DEFAULT,D_DEFAULT,N_OUT_DEFAULT,BLOCK,SINKHORN_DEFAULT>
        <<<grid_size,BLOCK,smem_size,stream>>>(hs,w_t,base,scale,hc_eps,rms_eps,num_pos,col_out,pre_out,post_out,comb_out,timing_buf);
}

// ============================================================
// Host launcher
// ============================================================
void hc_fused_launch(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* attn_hc_fn_t,
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
        hidden_states, attn_hc_fn_t, attn_hc_base, attn_hc_scale,
        hc_eps, rms_norm_eps, num_positions,
        collapsed_out, pre_out, post_out, comb_out
    );
}

// ============================================================
// PyTorch binding
// ============================================================
std::vector<torch::Tensor> hc_fused_forward_full(
    torch::Tensor hidden_states,    // [num_pos, HC, D] or [HC, D] bf16
    torch::Tensor attn_hc_fn,       // [N_OUT, HC*D] bf16 (transposed internally)
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

    // Transpose weight for vectorized access: [N_OUT, HC*D] → [HC*D, N_OUT]
    auto attn_hc_fn_t = attn_hc_fn.contiguous().t().contiguous();

    auto opts_bf16 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kBFloat16);
    auto opts_fp32 = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kFloat32);

    auto collapsed = torch::empty({num_pos, D}, opts_bf16);
    auto pre = torch::empty({num_pos, HC}, opts_fp32);
    auto post = torch::empty({num_pos, HC}, opts_fp32);
    auto comb = torch::empty({num_pos, HC, HC}, opts_fp32);

    hc_fused_launch(
        reinterpret_cast<const __nv_bfloat16*>(hs_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(attn_hc_fn_t.data_ptr<at::BFloat16>()),
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
          "HC fused forward (CuTe C++, bf16, transposed weight)");
    m.def("hc_fused_forward_profiled", [](torch::Tensor hs, torch::Tensor w,
          torch::Tensor base, torch::Tensor scale, double eps, double rms_eps) {
        TORCH_CHECK(hs.is_cuda() && hs.scalar_type() == torch::kBFloat16);
        const int HC = HC_DEFAULT, D = D_DEFAULT;
        auto hs_flat = hs.contiguous().view({-1, HC * D});
        int num_pos = hs_flat.size(0);
        auto w_t = w.contiguous().t().contiguous();
        int num_sms = get_num_sms();
        int grid_size = min(2 * num_sms, num_pos);

        auto opts_bf16 = torch::TensorOptions().device(hs.device()).dtype(torch::kBFloat16);
        auto opts_fp32 = torch::TensorOptions().device(hs.device()).dtype(torch::kFloat32);
        auto opts_i64 = torch::TensorOptions().device(hs.device()).dtype(torch::kInt64);

        auto collapsed = torch::empty({num_pos, D}, opts_bf16);
        auto pre = torch::empty({num_pos, HC}, opts_fp32);
        auto post = torch::empty({num_pos, HC}, opts_fp32);
        auto comb = torch::empty({num_pos, HC, HC}, opts_fp32);
        auto timing = torch::zeros({grid_size, 10}, opts_i64);

        hc_fused_launch_profiled(
            reinterpret_cast<const __nv_bfloat16*>(hs_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(w_t.data_ptr<at::BFloat16>()),
            base.contiguous().data_ptr<float>(), scale.contiguous().data_ptr<float>(),
            (float)eps, (float)rms_eps, num_pos,
            reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr<at::BFloat16>()),
            pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
            timing.data_ptr<int64_t>(), at::cuda::getCurrentCUDAStream()
        );
        return timing;  // [grid_size, 10]
    }, "HC fused profiled (returns timing buffer)");
}
