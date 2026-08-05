#pragma once

// mHC post: ported from mega_csa (Flash_DeepSeek_V4_Pro), include/mega/csa/
// mhc_post.cuh. The upstream namespace is kept so the port stays recognizable;
// the launch/validate split and the torch binding now live in one .cu (upstream
// keeps them apart because its runtime library must not depend on torch).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace mega::csa {

constexpr int kMhcResidualStreams = 4;
constexpr int kMhcHiddenDim = 7168;

struct MhcPostArgs {
  const __nv_bfloat16* attention_out = nullptr;  // [M,7168]
  const __nv_bfloat16* residual = nullptr;       // [M,4,7168]
  const float* post = nullptr;                    // [M,4]
  const float* comb = nullptr;                    // [M,4,4], [input,output]
  __nv_bfloat16* output = nullptr;                // [M,4,7168]
  int m = 0;
  bool use_pdl = true;
};

// Official DeepSeek-V4 mHC post:
// output[m,j,d] = post[m,j] * attention_out[m,d]
//               + sum_i comb[m,i,j] * residual[m,i,d].
// output is write-only and must not overlap attention_out, residual, post, or
// comb; later HC streams still read every input after earlier output stores.
cudaError_t mhc_post_validate(const MhcPostArgs& args);

cudaError_t mhc_post_run(const MhcPostArgs& args,
                         cudaStream_t stream = nullptr);

}  // namespace mega::csa
