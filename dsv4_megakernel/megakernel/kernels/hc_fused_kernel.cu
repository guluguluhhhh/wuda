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

// ============================================================
// Optimized CC kernel (HC-group split so hidden is read from HBM ONCE)
//
//   block_rank==0 owns HC groups {0,1}, block_rank==1 owns {2,3}.
//   Each block streams its contiguous HALF_K = 2*DIM slice of hidden into
//   smem ONCE (vectorized 128-bit), then reuses it for:
//     Phase 1 RMSNorm (partial sq_sum, dsmem merge)
//     Phase 2 GEMV     (partial acc[N_OUT], dsmem merge)
//     Phase 5 Collapse (reads own + remote smem via dsmem, no HBM re-read)
//   Sinkhorn runs on a single warp with shuffles (no __syncthreads storm).
//   rms_scale is folded into the mix result (mathematically identical to
//   normalizing x, since it is a per-row scalar).
// ============================================================
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
    const int block_rank = cluster.block_rank();             // 0 or 1

    constexpr int HC_DIM_TOTAL = HC * DIM;                    // 28672
    constexpr int GROUPS_PER_BLOCK = HC / 2;                  // 2 HC groups per block
    constexpr int HALF_K = GROUPS_PER_BLOCK * DIM;            // 14336 (contiguous slice)
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;               // 32
    constexpr int HALF_K_I4 = HALF_K / 8;                    // 1792 int4 (128-bit)
    constexpr int HALF_DIM = DIM / 2;                         // 3584
    constexpr int HALF_DIM_I4 = HALF_DIM / 8;                // 448
    constexpr int DIM_I4 = DIM / 8;                          // 896

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_clusters = gridDim.x / 2;
    const int cluster_id = blockIdx.x / 2;

    // Shared memory (identical layout in both blocks for dsmem)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* hid_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);   // [HALF_K] bf16
    float* fsmem = reinterpret_cast<float*>(hid_smem + HALF_K);
    float* reduce_smem = fsmem;                           // [NUM_WARPS]
    float* partial_sq_slot = reduce_smem + NUM_WARPS;     // [1] dsmem exchange
    float* gemv_smem = partial_sq_slot + 1;               // [NUM_WARPS * N_OUT]
    float* acc_slot = gemv_smem + NUM_WARPS * N_OUT;      // [N_OUT] dsmem exchange
    float* mix_smem = acc_slot + N_OUT;                   // [N_OUT]
    float* pre_smem = mix_smem + N_OUT;                   // [HC]
    float* post_smem = pre_smem + HC;                     // [HC]
    float* comb_smem = post_smem + HC;                    // [HC*HC]

    const float s_pre_scale = attn_hc_scale[0];
    const float s_post_scale = attn_hc_scale[1];
    const float s_comb_scale = attn_hc_scale[2];

    for (int pos_idx = cluster_id; pos_idx < num_positions; pos_idx += num_clusters) {

    // block_rank r owns hidden[pos, r*HALF_K : (r+1)*HALF_K) = HC groups {2r, 2r+1}
    const __nv_bfloat16* hs_base = hidden_states + pos_idx * HC_DIM_TOTAL + block_rank * HALF_K;
    __nv_bfloat16* col_ptr = collapsed_out + pos_idx * DIM;

    // ================================================================
    // Phase 0+1: load hidden slice into smem (128-bit) + RMSNorm sq_sum
    // ================================================================
    const int4* hs_g = reinterpret_cast<const int4*>(hs_base);
    int4* hs_s = reinterpret_cast<int4*>(hid_smem);
    float sq_sum = 0.0f;
    for (int idx = tid; idx < HALF_K_I4; idx += BLOCK_SIZE) {
        int4 v = hs_g[idx];
        hs_s[idx] = v;
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(p[j]);
            sq_sum += f.x * f.x + f.y * f.y;
        }
    }

    // Block-local sq_sum reduce only; the cross-block exchange is DEFERRED and
    // fused with the GEMV acc exchange below (rms_scale is only needed at the
    // very end when folding into mix, so no early cluster barrier is required).
    float partial_sq = block_reduce_sum<NUM_WARPS>(sq_sum, reduce_smem, tid);
    if (tid == 0) partial_sq_slot[0] = partial_sq;
    __syncthreads();  // publish partial_sq_slot + make hid_smem visible for GEMV

    // ================================================================
    // Phase 2: GEMV over this block's HALF_K (raw x, from smem)
    // partial acc[N_OUT]; rms_scale folded into mix at the end.
    // ================================================================
    float acc[N_OUT];
    #pragma unroll
    for (int n = 0; n < N_OUT; n++) acc[n] = 0.0f;

    for (int k_local = tid; k_local < HALF_K; k_local += BLOCK_SIZE) {
        float x = __bfloat162float(hid_smem[k_local]);
        int k = block_rank * HALF_K + k_local;
        const int4* w_vec = reinterpret_cast<const int4*>(attn_hc_fn_t + k * N_OUT);
        int4 v0 = w_vec[0], v1 = w_vec[1], v2 = w_vec[2];
        __nv_bfloat162* p0 = reinterpret_cast<__nv_bfloat162*>(&v0);
        __nv_bfloat162* p1 = reinterpret_cast<__nv_bfloat162*>(&v1);
        __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&v2);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f0 = __bfloat1622float2(p0[j]);
            float2 f1 = __bfloat1622float2(p1[j]);
            float2 f2 = __bfloat1622float2(p2[j]);
            acc[j*2]      += x * f0.x;  acc[j*2+1]    += x * f0.y;
            acc[8+j*2]    += x * f1.x;  acc[8+j*2+1]  += x * f1.y;
            acc[16+j*2]   += x * f2.x;  acc[16+j*2+1] += x * f2.y;
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
        acc_slot[tid] = sum;
    }
    __syncthreads();

    // Single combined cluster exchange: partial_sq (-> rms_scale) AND acc.
    cluster.sync();
    float total_sq = partial_sq_slot[0] + *cluster.map_shared_rank(partial_sq_slot, 1 - block_rank);
    float rms_scale_val = rsqrtf(total_sq / (float)HC_DIM_TOTAL + rms_norm_eps);
    if (tid < N_OUT) {
        float* remote_acc = cluster.map_shared_rank(acc_slot, 1 - block_rank);
        mix_smem[tid] = (acc_slot[tid] + remote_acc[tid]) * rms_scale_val;
    }
    __syncthreads();

    // ================================================================
    // Phase 3: Activation. Both blocks compute pre[] (needed by collapse).
    // block_rank==0 additionally computes post[]/comb[] and writes gates.
    // ================================================================
    if (tid < HC) {
        pre_smem[tid] = fast_sigmoid(mix_smem[tid] * s_pre_scale + attn_hc_base[tid]) + hc_eps;
    }
    if (block_rank == 0) {
        if (tid < HC) {
            post_smem[tid] = 2.0f * fast_sigmoid(mix_smem[HC + tid] * s_post_scale + attn_hc_base[HC + tid]);
        }
        if (tid < HC * HC) {
            comb_smem[tid] = mix_smem[2 * HC + tid] * s_comb_scale + attn_hc_base[2 * HC + tid];
        }
    }
    __syncthreads();

    // ================================================================
    // Phase 4: Softmax + Sinkhorn on the 4x4 comb, warp 0 of block 0 only.
    // lane l holds comb[row=l/HC][col=l%HC]; reductions via shuffle.
    // ================================================================
    if (block_rank == 0 && warp_id == 0) {
        const unsigned FULL = 0xffffffffu;
        float v = (lane_id < HC * HC) ? comb_smem[lane_id] : 0.0f;

        // row-wise softmax (sum over cols == within groups of HC lanes: xor 1..HC/2)
        float m = v;
        #pragma unroll
        for (int o = 1; o < HC; o <<= 1) m = fmaxf(m, __shfl_xor_sync(FULL, m, o));
        float e = __expf(v - m);
        float rs = e;
        #pragma unroll
        for (int o = 1; o < HC; o <<= 1) rs += __shfl_xor_sync(FULL, rs, o);
        v = e / rs + hc_eps;

        // column normalize (sum over rows == lanes differing by HC: xor HC, 2*HC, ...)
        float cs = v;
        #pragma unroll
        for (int o = HC; o < HC * HC; o <<= 1) cs += __shfl_xor_sync(FULL, cs, o);
        v = v / (cs + hc_eps);

        #pragma unroll 1
        for (int iter = 0; iter < SINKHORN_ITERS - 1; iter++) {
            float rs2 = v;
            #pragma unroll
            for (int o = 1; o < HC; o <<= 1) rs2 += __shfl_xor_sync(FULL, rs2, o);
            v = v / (rs2 + hc_eps);
            float cs2 = v;
            #pragma unroll
            for (int o = HC; o < HC * HC; o <<= 1) cs2 += __shfl_xor_sync(FULL, cs2, o);
            v = v / (cs2 + hc_eps);
        }
        if (lane_id < HC * HC) comb_smem[lane_id] = v;
    }

    // Write gates to global (block_rank==0)
    if (block_rank == 0 && tid < HC) {
        pre_out[pos_idx * HC + tid] = pre_smem[tid];
        post_out[pos_idx * HC + tid] = post_smem[tid];
    }
    if (block_rank == 0 && tid < HC * HC) {
        comb_out[pos_idx * HC * HC + tid] = comb_smem[tid];
    }

    // ================================================================
    // Phase 5: Collapse. out[d] = sum_h pre[h] * hidden[h*DIM + d].
    // block r writes out[r*HALF_DIM : (r+1)*HALF_DIM); it holds groups
    // {2r,2r+1} locally and reads {2(1-r),2(1-r)+1} from remote smem.
    // No leading cluster.sync needed: the combined post-GEMV cluster.sync
    // already established that both blocks loaded hid_smem, and hid_smem is
    // untouched until next iteration's Phase 0 (gated by the end-of-iter sync).
    // ================================================================
    __nv_bfloat16* rem_smem = cluster.map_shared_rank(hid_smem, 1 - block_rank);

    const float pl0 = pre_smem[2 * block_rank];
    const float pl1 = pre_smem[2 * block_rank + 1];
    const float pr0 = pre_smem[2 * (1 - block_rank)];
    const float pr1 = pre_smem[2 * (1 - block_rank) + 1];

    const int4* loc_i4 = reinterpret_cast<const int4*>(hid_smem);
    const int4* rem_i4 = reinterpret_cast<const int4*>(rem_smem);
    int4* out_i4 = reinterpret_cast<int4*>(col_ptr);
    const int d0_i4 = block_rank * HALF_DIM_I4;   // int4 offset of this block's output range

    for (int t = tid; t < HALF_DIM_I4; t += BLOCK_SIZE) {
        int di4 = d0_i4 + t;                       // int4 index within [0, DIM_I4)
        int4 la = loc_i4[di4];                     // local group0 : hidden[0*DIM + d]
        int4 lb = loc_i4[DIM_I4 + di4];            // local group1 : hidden[1*DIM + d]
        int4 ra = rem_i4[di4];                     // remote group0
        int4 rb = rem_i4[DIM_I4 + di4];            // remote group1
        const __nv_bfloat162* pla = reinterpret_cast<const __nv_bfloat162*>(&la);
        const __nv_bfloat162* plb = reinterpret_cast<const __nv_bfloat162*>(&lb);
        const __nv_bfloat162* pra = reinterpret_cast<const __nv_bfloat162*>(&ra);
        const __nv_bfloat162* prb = reinterpret_cast<const __nv_bfloat162*>(&rb);
        int4 outv;
        __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&outv);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 a = __bfloat1622float2(pla[j]);
            float2 b = __bfloat1622float2(plb[j]);
            float2 c = __bfloat1622float2(pra[j]);
            float2 d = __bfloat1622float2(prb[j]);
            float o0 = pl0 * a.x + pl1 * b.x + pr0 * c.x + pr1 * d.x;
            float o1 = pl0 * a.y + pl1 * b.y + pr0 * c.y + pr1 * d.y;
            po[j] = __floats2bfloat162_rn(o0, o1);
        }
        out_i4[di4] = outv;
    }

    cluster.sync();  // end-of-iteration barrier (smem reused next iteration)
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
    // smem: hidden slice [HALF_K] bf16 + float scratch (reduce/gemv/mix/gates)
    constexpr int HALF_K = (HC_DEFAULT / 2) * DIM_DEFAULT;   // 14336
    constexpr int NUM_WARPS = BLOCK / 32;                    // 32
    int fscratch = (NUM_WARPS + 1 + NUM_WARPS * N_OUT_DEFAULT + N_OUT_DEFAULT
                    + N_OUT_DEFAULT + HC_DEFAULT + HC_DEFAULT + HC_DEFAULT * HC_DEFAULT);
    int smem_size = HALF_K * (int)sizeof(__nv_bfloat16) + fscratch * (int)sizeof(float);

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
