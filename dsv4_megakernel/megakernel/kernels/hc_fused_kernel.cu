// ============================================================
// DeepSeek-V4 HC Fused Kernel (Cluster version: 2 blocks/cluster)
// Fuses: RMSNorm(no weight) + GEMV + Activation + Sinkhorn + Collapse
// Input: bf16 [num_pos, HC*DIM] = [num_pos, 28672]
// Weight: bf16 [N_OUT, HC*DIM] = [24, 28672] (transposed to [28672, 24])
// Output: bf16 [num_pos, DIM] = [num_pos, 7168]
// Launch: cudaLaunchKernelEx, cluster_dim(2,1,1), grid=2*SM clusters
// ============================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include "../include/hc_fused_kernel.cuh"

namespace cg = cooperative_groups;

template <int HC, int DIM, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __attribute__((cluster_dim(2, 1, 1))) __launch_bounds__(1024)
hc_fused_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_pos, HC*DIM] bf16
    const __nv_bfloat16* __restrict__ attn_hc_fn_t,    // [HC*DIM, N_OUT] bf16 TRANSPOSED
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
    const int block_rank = cluster.block_rank();

    constexpr int HC_DIM_TOTAL = HC * DIM;                    // 28672
    constexpr int HALF_K = HC_DIM_TOTAL / 2;                  // 14336
    constexpr int ELEMS_PER_THR = HALF_K / BLOCK_SIZE;        // 14
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;                // 32

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_clusters = gridDim.x / 2;
    const int cluster_id = blockIdx.x / 2;

    for (int pos_idx = cluster_id; pos_idx < num_positions; pos_idx += num_clusters) {

    // Per-position pointers
    const __nv_bfloat16* hs_ptr = hidden_states + pos_idx * HC_DIM_TOTAL;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * DIM;
    float* pre_ptr = pre_out + pos_idx * HC;
    float* post_ptr = post_out + pos_idx * HC;
    float* comb_ptr = comb_out + pos_idx * HC * HC;

    // Shared memory (identical layout in both blocks for dsmem)
    extern __shared__ float smem[];
    float* reduce_smem = smem;                            // [NUM_WARPS]
    float* partial_sq_slot = reduce_smem + NUM_WARPS;     // [1] dsmem exchange
    float* gemv_smem = partial_sq_slot + 1;               // [NUM_WARPS * N_OUT]
    float* acc_slot = gemv_smem + NUM_WARPS * N_OUT;      // [N_OUT] dsmem exchange
    float* mix_smem = acc_slot + N_OUT;                   // [N_OUT]
    float* pre_smem = mix_smem + N_OUT;                   // [HC]
    float* post_smem = pre_smem + HC;                     // [HC]
    float* comb_smem = post_smem + HC;                    // [HC*HC]

    // ================================================================
    // Phase 1: RMSNorm - each block loads K/2, merge sq_sum via dsmem
    // block_rank==0 -> [0, HALF_K), block_rank==1 -> [HALF_K, HC_DIM_TOTAL)
    // ================================================================
    int half_offset = block_rank * HALF_K;
    float orig_vals[ELEMS_PER_THR];
    int base_idx = tid * ELEMS_PER_THR;

    float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        float v = bf16_to_float(hs_ptr[half_offset + base_idx + i]);
        orig_vals[i] = v;
        sq_sum += v * v;
    }

    float partial_sq = block_reduce_sum<NUM_WARPS>(sq_sum, reduce_smem, tid);
    if (tid == 0) partial_sq_slot[0] = partial_sq;
    __syncthreads();

    cluster.sync();

    // Each block reads the other's partial sum via dsmem
    float* remote_sq = cluster.map_shared_rank(partial_sq_slot, 1 - block_rank);
    float total_sq = partial_sq_slot[0] + *remote_sq;
    float rms_scale_val = rsqrtf(total_sq / (float)HC_DIM_TOTAL + rms_norm_eps);

    float norm_vals[ELEMS_PER_THR];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        norm_vals[i] = orig_vals[i] * rms_scale_val;
    }

    // ================================================================
    // Phase 2: GEMV - each block computes K/2, cluster.sync() + dsmem merge
    // ================================================================

    float acc[N_OUT];
    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THR; i++) {
        int k = half_offset + base_idx + i;
        float x = norm_vals[i];
        const int4* w_vec = reinterpret_cast<const int4*>(attn_hc_fn_t + k * N_OUT);
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

    // Cross-warp reduce -> acc_slot[N_OUT]
    if (tid < N_OUT) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) sum += gemv_smem[w * N_OUT + tid];
        acc_slot[tid] = sum;
    }
    __syncthreads();

    cluster.sync();

    // block_rank==0 merges both halves via dsmem
    if (block_rank == 0 && tid < N_OUT) {
        float* remote_acc = cluster.map_shared_rank(acc_slot, 1);
        mix_smem[tid] = acc_slot[tid] + remote_acc[tid];
    }
    if (block_rank == 0) __syncthreads();

    // Broadcast mix_smem[24] to block_rank==1 via dsmem
    cluster.sync();
    if (block_rank == 1 && tid < N_OUT) {
        float* remote_mix = cluster.map_shared_rank(mix_smem, 0);
        mix_smem[tid] = remote_mix[tid];
    }
    __syncthreads();

    // ================================================================
    // Phase 3: Activation (both blocks independently)
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
    // Phase 4: Softmax + Sinkhorn (both blocks independently)
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

    // Write gates to global (block_rank==0 only)
    if (block_rank == 0 && tid < HC) { pre_ptr[tid] = pre_smem[tid]; post_ptr[tid] = post_smem[tid]; }
    if (block_rank == 0 && tid < HC * HC) { comb_ptr[tid] = comb_smem[tid]; }

    // ================================================================
    // Phase 5: Collapse - split output dims, both blocks in parallel
    // block_rank==0 -> output[0, DIM/2), block_rank==1 -> output[DIM/2, DIM)
    // Each block loads its half from HBM (14 bf16/thread) and accumulates HC=4
    // ================================================================
    constexpr int HALF_DIM = DIM / 2;                        // 3584
    constexpr int THREADS_PER_HC_C = BLOCK_SIZE / HC;        // 256
    constexpr int ELEMS_COL = HALF_DIM / THREADS_PER_HC_C;   // 14
    int h_idx = tid / THREADS_PER_HC_C;                      // 0..3
    int col_local = tid % THREADS_PER_HC_C;                  // 0..255
    int d_offset = block_rank * HALF_DIM;                    // 0 or 3584
    float pre_h = pre_smem[h_idx];
    __syncthreads();

    // Reuse smem as collapse buffer [HALF_DIM] = 3584 floats
    float* col_buf = smem;
    float col_contrib[ELEMS_COL];
    #pragma unroll
    for (int i = 0; i < ELEMS_COL; i++) {
        int d = d_offset + col_local * ELEMS_COL + i;
        col_contrib[i] = pre_h * bf16_to_float(hs_ptr[h_idx * DIM + d]);
    }

    if (h_idx == 0) {
        #pragma unroll
        for (int i = 0; i < ELEMS_COL; i++) col_buf[col_local * ELEMS_COL + i] = col_contrib[i];
    }
    __syncthreads();
    #pragma unroll
    for (int g = 1; g < HC; g++) {
        if (h_idx == g) {
            #pragma unroll
            for (int i = 0; i < ELEMS_COL; i++) col_buf[col_local * ELEMS_COL + i] += col_contrib[i];
        }
        __syncthreads();
    }

    // Write output [HALF_DIM] as bf16
    for (int d = tid; d < HALF_DIM; d += BLOCK_SIZE) {
        col_ptr[d_offset + d] = float_to_bf16(col_buf[d]);
    }

    cluster.sync();  // end-of-iteration barrier
    } // end grid-stride loop
}

// ============================================================
// Profiled kernel (cluster version, clock64 per phase)
// ============================================================
template <int HC, int DIM, int N_OUT, int BLOCK_SIZE, int SINKHORN_ITERS>
__global__ void __attribute__((cluster_dim(2, 1, 1))) __launch_bounds__(1024)
hc_fused_kernel_profiled(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ attn_hc_fn_t,
    const float* __restrict__ attn_hc_base,
    const float* __restrict__ attn_hc_scale,
    float hc_eps, float rms_norm_eps, int num_positions,
    __nv_bfloat16* __restrict__ collapsed_out,
    float* __restrict__ pre_out, float* __restrict__ post_out, float* __restrict__ comb_out,
    int64_t* __restrict__ timing_buf
) {
    cg::cluster_group cluster = cg::this_cluster();
    const int block_rank = cluster.block_rank();

    constexpr int HC_DIM_TOTAL = HC * DIM;
    constexpr int HALF_K = HC_DIM_TOTAL / 2;
    constexpr int ELEMS_PER_THR = HALF_K / BLOCK_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int cluster_id = blockIdx.x / 2;
    if (cluster_id >= num_positions) return;
    int pos_idx = cluster_id;

    int64_t* my_t = timing_buf + cluster_id * 10;
    int64_t t0, t1;
    const __nv_bfloat16* hs_ptr = hidden_states + pos_idx * HC_DIM_TOTAL;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * DIM;
    float* pre_ptr = pre_out + pos_idx * HC;
    float* post_ptr = post_out + pos_idx * HC;
    float* comb_ptr = comb_out + pos_idx * HC * HC;

    extern __shared__ float smem[];
    float* reduce_smem = smem;
    float* partial_sq_slot = reduce_smem + NUM_WARPS;
    float* gemv_smem = partial_sq_slot + 1;
    float* acc_slot = gemv_smem + NUM_WARPS * N_OUT;
    float* mix_smem = acc_slot + N_OUT;
    float* pre_smem = mix_smem + N_OUT;
    float* post_smem = pre_smem + HC;
    float* comb_smem = post_smem + HC;

    // Phase 1: RMSNorm (split K/2 + dsmem merge)
    if(block_rank==0 && tid==0) t0=clock64();
    int half_offset = block_rank * HALF_K;
    float orig_vals[ELEMS_PER_THR];
    int base_idx = tid * ELEMS_PER_THR;
    float sq_sum = 0.0f;
    for(int i=0;i<ELEMS_PER_THR;i++){float v=bf16_to_float(hs_ptr[half_offset+base_idx+i]);orig_vals[i]=v;sq_sum+=v*v;}
    float partial_sq = block_reduce_sum<NUM_WARPS>(sq_sum, reduce_smem, tid);
    if(tid==0) partial_sq_slot[0]=partial_sq;
    __syncthreads();
    cluster.sync();
    float* remote_sq = cluster.map_shared_rank(partial_sq_slot, 1 - block_rank);
    float total_sq = partial_sq_slot[0] + *remote_sq;
    float rms_scale_val = rsqrtf(total_sq/(float)HC_DIM_TOTAL+rms_norm_eps);
    float norm_vals[ELEMS_PER_THR];
    for(int i=0;i<ELEMS_PER_THR;i++) norm_vals[i]=orig_vals[i]*rms_scale_val;
    if(block_rank==0 && tid==0){t1=clock64();my_t[0]=t0;my_t[1]=t1;}
    __syncthreads();

    // Phase 2: GEMV (split K/2 + dsmem merge)
    if(block_rank==0 && tid==0) t0=clock64();
    float acc[N_OUT]; for(int n=0;n<N_OUT;n++)acc[n]=0.0f;
    for(int i=0;i<ELEMS_PER_THR;i++){
        int k=half_offset+base_idx+i; float x=norm_vals[i];
        const int4* wv=reinterpret_cast<const int4*>(attn_hc_fn_t+k*N_OUT);
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
    for(int n=0;n<N_OUT;n++) acc[n]=warp_reduce_sum(acc[n]);
    if(lane_id==0){for(int n=0;n<N_OUT;n++)gemv_smem[warp_id*N_OUT+n]=acc[n];}
    __syncthreads();
    if(tid<N_OUT){float s=0;for(int w=0;w<NUM_WARPS;w++)s+=gemv_smem[w*N_OUT+tid];acc_slot[tid]=s;}
    __syncthreads();
    cluster.sync();
    if(block_rank==0 && tid<N_OUT){float* ra=cluster.map_shared_rank(acc_slot,1);mix_smem[tid]=acc_slot[tid]+ra[tid];}
    if(block_rank==0) __syncthreads();
    if(block_rank==0 && tid==0){t1=clock64();my_t[2]=t0;my_t[3]=t1;}

    // Broadcast mix_smem to block_rank==1
    cluster.sync();
    if(block_rank==1 && tid<N_OUT){float* rm=cluster.map_shared_rank(mix_smem,0);mix_smem[tid]=rm[tid];}
    __syncthreads();

    // Phase 3: Activation (both blocks)
    if(block_rank==0 && tid==0) t0=clock64();
    __shared__ float sp,spo,sc;
    if(tid==0){sp=attn_hc_scale[0];spo=attn_hc_scale[1];sc=attn_hc_scale[2];}
    __syncthreads();
    if(tid<HC) pre_smem[tid]=fast_sigmoid(mix_smem[tid]*sp+attn_hc_base[tid])+hc_eps;
    else if(tid<2*HC){int idx=tid-HC;post_smem[idx]=2.0f*fast_sigmoid(mix_smem[HC+idx]*spo+attn_hc_base[HC+idx]);}
    if(tid<HC*HC) comb_smem[tid]=mix_smem[2*HC+tid]*sc+attn_hc_base[2*HC+tid];
    __syncthreads();
    if(block_rank==0 && tid==0){t1=clock64();my_t[4]=t0;my_t[5]=t1;}
    __syncthreads();

    // Phase 4: Sinkhorn (both blocks)
    if(block_rank==0 && tid==0) t0=clock64();
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
    if(block_rank==0 && tid==0){t1=clock64();my_t[6]=t0;my_t[7]=t1;}
    if(block_rank==0 && tid<HC){pre_ptr[tid]=pre_smem[tid];post_ptr[tid]=post_smem[tid];}
    if(block_rank==0 && tid<HC*HC)comb_ptr[tid]=comb_smem[tid];

    // Phase 5: Collapse (split output dims, both blocks parallel)
    if(block_rank==0 && tid==0) t0=clock64();
    constexpr int HALF_DIM_P=DIM/2;
    constexpr int TPH_C=BLOCK_SIZE/HC;
    constexpr int ELEMS_COL_P=HALF_DIM_P/TPH_C;
    int h_idx=tid/TPH_C,col_local=tid%TPH_C;
    int d_offset=block_rank*HALF_DIM_P;
    float pre_h=pre_smem[h_idx];
    __syncthreads();
    float* col_buf=smem;
    float cc[ELEMS_COL_P];
    for(int i=0;i<ELEMS_COL_P;i++){int d=d_offset+col_local*ELEMS_COL_P+i;cc[i]=pre_h*bf16_to_float(hs_ptr[h_idx*DIM+d]);}
    if(h_idx==0){for(int i=0;i<ELEMS_COL_P;i++)col_buf[col_local*ELEMS_COL_P+i]=cc[i];}
    __syncthreads();
    for(int g=1;g<HC;g++){if(h_idx==g){for(int i=0;i<ELEMS_COL_P;i++)col_buf[col_local*ELEMS_COL_P+i]+=cc[i];}__syncthreads();}
    for(int d=tid;d<HALF_DIM_P;d+=BLOCK_SIZE)col_ptr[d_offset+d]=float_to_bf16(col_buf[d]);
    if(block_rank==0 && tid==0){t1=clock64();my_t[8]=t0;my_t[9]=t1;}
    cluster.sync();  // ensure block_rank==0 finished before block_rank==1 exits
}

void hc_fused_launch_profiled(
    const __nv_bfloat16* hs, const __nv_bfloat16* w,
    const float* base, const float* scale,
    float hc_eps, float rms_eps, int num_pos,
    __nv_bfloat16* col_out, float* pre_out, float* post_out, float* comb_out,
    int64_t* timing_buf, cudaStream_t stream
) {
    constexpr int BLOCK = BLOCK_SIZE_DEFAULT;
    int num_sms = get_num_sms();
    int num_clusters = std::min(2 * num_sms, num_pos);
    int grid_size = num_clusters * 2;
    int smem_size = DIM_DEFAULT * sizeof(float);

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

    cudaLaunchKernelEx(&config,
        hc_fused_kernel_profiled<HC_DEFAULT,DIM_DEFAULT,N_OUT_DEFAULT,BLOCK,SINKHORN_DEFAULT>,
        hs,w,base,scale,hc_eps,rms_eps,num_pos,col_out,pre_out,post_out,comb_out,timing_buf);
}

// ============================================================
// Host launcher (cudaLaunchKernelEx with cluster)
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
    int num_clusters = std::min(2 * num_sms, num_positions);
    int grid_size = num_clusters * 2;  // 2 blocks per cluster
    int smem_size = DIM_DEFAULT * sizeof(float);

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

    cudaLaunchKernelEx(&config,
        hc_fused_kernel<HC_DEFAULT, DIM_DEFAULT, N_OUT_DEFAULT, BLOCK, SINKHORN_DEFAULT>,
        hidden_states, attn_hc_fn_t, attn_hc_base, attn_hc_scale,
        hc_eps, rms_norm_eps, num_positions,
        collapsed_out, pre_out, post_out, comb_out);
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
        reinterpret_cast<const __nv_bfloat16*>(attn_hc_fn.contiguous().t().contiguous().data_ptr<at::BFloat16>()),
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
        int num_clusters = std::min(2 * num_sms, num_pos);

        auto opts_bf16 = torch::TensorOptions().device(hs.device()).dtype(torch::kBFloat16);
        auto opts_fp32 = torch::TensorOptions().device(hs.device()).dtype(torch::kFloat32);
        auto opts_i64 = torch::TensorOptions().device(hs.device()).dtype(torch::kInt64);

        auto collapsed = torch::empty({num_pos, DIM}, opts_bf16);
        auto pre = torch::empty({num_pos, HC}, opts_fp32);
        auto post = torch::empty({num_pos, HC}, opts_fp32);
        auto comb = torch::empty({num_pos, HC, HC}, opts_fp32);
        auto timing = torch::zeros({num_clusters, 10}, opts_i64);

        hc_fused_launch_profiled(
            reinterpret_cast<const __nv_bfloat16*>(hs_flat.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(w.contiguous().t().contiguous().data_ptr<at::BFloat16>()),
            base.contiguous().data_ptr<float>(), scale.contiguous().data_ptr<float>(),
            (float)eps, (float)rms_eps, num_pos,
            reinterpret_cast<__nv_bfloat16*>(collapsed.data_ptr<at::BFloat16>()),
            pre.data_ptr<float>(), post.data_ptr<float>(), comb.data_ptr<float>(),
            timing.data_ptr<int64_t>(), at::cuda::getCurrentCUDAStream()
        );
        return timing;
    }, "HC fused profiled (returns timing buffer)");
}
