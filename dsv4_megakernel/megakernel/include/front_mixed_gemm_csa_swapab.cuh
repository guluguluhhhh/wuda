#pragma once

// CSA front projection, swap-AB variant:
//   Y^T[4672,M] = W[4672,7168] @ X^T[7168,M]
// FP8 uses UMMA_M=256, BF16 uses UMMA_M=128. UMMA_N is the batch tile.
// Batch splits into AT MOST TWO tiles (M<=16 -> 1, else 2), so the batch
// ladder is 16/16x2/32x2/64x2/128x2 and N-parallelism carries the machine:
// 29 feature tiles (8 fp8 N256 + 21 bf16 N128, tail tile 64 valid rows)
// x batch_tiles -> 29 or 58 clusters.

#include "front_mixed_gemm_csa.cuh"

namespace front_mixed_csa_swapab {

namespace base = front_mixed_csa;
namespace fp8d = cluster_mma_fp8::detail;
using Barrier = base::Barrier;

constexpr int K = base::K;
constexpr int N = 4672;                  // CSA reordered layout
constexpr int N_FP8 = 2048;
constexpr int CLUSTER_SIZE = 2;
constexpr int TPB = 256;
constexpr int FP8_UMMA_M = 256;
constexpr int BF16_UMMA_M = 128;
constexpr int FP8_CTA_N = FP8_UMMA_M / CLUSTER_SIZE;      // 128 features/CTA
constexpr int BF16_CTA_N = BF16_UMMA_M / CLUSTER_SIZE;    // 64 features/CTA
constexpr int FP8_FEATURE_TILES = N_FP8 / FP8_UMMA_M;     // 8 clusters
constexpr int BF16_FEATURE_TILES =
    (N - N_FP8 + BF16_UMMA_M - 1) / BF16_UMMA_M;          // 21 (tail 64 rows)
constexpr int TASKS_PER_BATCH = FP8_FEATURE_TILES + BF16_FEATURE_TILES;  // 29
static_assert((N - N_FP8) == 20 * BF16_UMMA_M + 64,
              "bf16 tail tile has 64 valid feature rows (OOB fill + clip)");

constexpr int FP8_TMA_K = 128;
constexpr int BF16_TMA_K = 64;
constexpr int FP8_SCALE_TILES = K / FP8_TMA_K;            // 56 physical K128 tiles
constexpr int SF_GROUPS = FP8_SCALE_TILES / 4;             // packed 4 K128 scales
constexpr int SF_ROWS = 128;
constexpr int SF_BYTES = SF_GROUPS * SF_ROWS * sizeof(uint32_t);
constexpr uint32_t UE8M0_ONE4 = 0x7f7f7f7fu;

constexpr int TMEM_SF_ACT = 64;          // BatchN<=64 layout (accum first)
constexpr int TMEM_SF_WEIGHT = 68;
constexpr int TMEM_COLS = 128;
constexpr int STORE_M = 16;
constexpr int STORE_N = 32;
constexpr int TIMING_FIELDS = 13;

template <int BatchN>
struct Config {
  static_assert(BatchN == 16 || BatchN == 32 || BatchN == 64 ||
                BatchN == 128);
  static constexpr int BATCH_N = BatchN;
  static constexpr int FP8_BLOCK_K = BatchN >= 64 ? 128 : 256;
  static constexpr int BF16_BLOCK_K = BatchN >= 64 ? 128 : 256;
  static constexpr int FP8_K_TILES = K / FP8_BLOCK_K;
  static constexpr int BF16_K_TILES = K / BF16_BLOCK_K;
  // Stage budgets fill smem (227.5KB cap): BN128 act stages double, so the
  // rings shrink to 8 (fp8, 24KB/st) and 7 (bf16, 32KB/st).
  static constexpr int FP8_STAGES = BatchN == 128 ? 8 : (BatchN == 64 ? 10 : 5);
  static constexpr int BF16_STAGES = BatchN == 128 ? 7 : (BatchN == 64 ? 9 : 5);
  static constexpr int MAX_STAGES =
      FP8_STAGES > BF16_STAGES ? FP8_STAGES : BF16_STAGES;

  static constexpr int FP8_ACT_DATA_BYTES = BatchN * FP8_BLOCK_K / 2;
  static constexpr int FP8_ACT_STAGE_BYTES = BatchN >= 64
      ? FP8_ACT_DATA_BYTES : BatchN * FP8_BLOCK_K;
  static constexpr int FP8_WGT_STAGE_BYTES = FP8_CTA_N * FP8_BLOCK_K;
  static constexpr int BF16_ACT_STAGE_BYTES = BatchN * BF16_BLOCK_K;
  static constexpr int BF16_WGT_STAGE_BYTES =
      BF16_CTA_N * BF16_BLOCK_K * sizeof(__nv_bfloat16);

  static constexpr int FP8_PIPELINE_BYTES =
      FP8_STAGES * (FP8_ACT_STAGE_BYTES + FP8_WGT_STAGE_BYTES);
  static constexpr int BF16_PIPELINE_BYTES =
      BF16_STAGES * (BF16_ACT_STAGE_BYTES + BF16_WGT_STAGE_BYTES);
  static constexpr int FP8_STORAGE_BYTES = FP8_PIPELINE_BYTES + 2 * SF_BYTES;
  static constexpr int STORAGE_BYTES = FP8_STORAGE_BYTES > BF16_PIPELINE_BYTES
      ? FP8_STORAGE_BYTES : BF16_PIPELINE_BYTES;

  // BN128 accum spans TMEM cols 0..127 and would collide with the fixed
  // SF columns at 64/68 -> SFs move past the accum and the alloc doubles.
  static constexpr int SF_ACT_COL = BatchN == 128 ? 132 : TMEM_SF_ACT;
  static constexpr int SF_WGT_COL = BatchN == 128 ? 128 : TMEM_SF_WEIGHT;
  static constexpr int NUM_TMEM_COLS = BatchN == 128 ? 256 : TMEM_COLS;
};

template <int BatchN>
struct SharedStorage {
  using C = Config<BatchN>;
  // FP8 and BF16 tasks are mutually exclusive in a CTA, so their pipeline
  // storage can overlap. This lets BN64 use K256/5-stage for FP8 and
  // K128/9-stage for BF16 without splitting the launch.
  alignas(1024) uint8_t storage[C::STORAGE_BYTES];
  alignas(16) Barrier full[C::MAX_STAGES];
  alignas(16) Barrier empty[C::MAX_STAGES];
  alignas(16) Barrier tmem_full;
  alignas(16) uint32_t tmem_base;
};

template <int BatchN>
__device__ __forceinline__ uint64_t bf16_idesc() {
  return cute::UMMA::make_runtime_instr_desc<
      cutlass::bfloat16_t, cutlass::bfloat16_t, float,
      BF16_UMMA_M, BatchN, cute::UMMA::Major::K, cute::UMMA::Major::K>();
}

template <int BatchN>
__device__ __forceinline__ cute::UMMA::InstrDescriptorBlockScaled fp8_idesc() {
  return cute::UMMA::make_instr_desc_block_scaled<
      cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
      cutlass::float_ue8m0_t, FP8_UMMA_M, BatchN,
      cute::UMMA::Major::K, cute::UMMA::Major::K>();
}

__device__ __forceinline__ void mma_2sm_bf16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint64_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 "
      "[%0], %1, %2, %3, p;\n\t}"
      :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
         "r"(static_cast<uint32_t>(idesc >> 32)), "r"(accumulate));
}

__device__ __forceinline__ uint64_t read_globaltimer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

template <int BatchN, bool RecordTimestamps = false>
static __global__ void __launch_bounds__(TPB, 1) swapab_kernel(
    const __grid_constant__ CUtensorMap desc_x16,
    const __grid_constant__ CUtensorMap desc_w16,
    const __grid_constant__ CUtensorMap desc_x8,
    const __grid_constant__ CUtensorMap desc_w8,
    const uint8_t* __restrict__ x_sf,
    const uint8_t* __restrict__ w_sf,
    __nv_bfloat16* __restrict__ out,
    base::HcTailArgs hc,
    base::FrontEmitArgs emit,
    int problem_m,
    uint64_t* __restrict__ task_times) {
  using C = Config<BatchN>;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  auto& s = *reinterpret_cast<SharedStorage<BatchN>*>(smem_raw);

  const int warp = threadIdx.x >> 5;
  const int lane = base::lane_id();
  const int rank = base::cluster_rank();
  const int task = blockIdx.x / CLUSTER_SIZE;
  const int batch_tile = task / TASKS_PER_BATCH;
  const int task_in_batch = task % TASKS_PER_BATCH;
  const bool is_fp8 = task_in_batch < FP8_FEATURE_TILES;
  const int feature_tile = is_fp8
      ? task_in_batch : task_in_batch - FP8_FEATURE_TILES;
  const int batch_base = batch_tile * BatchN;
  const bool elected = base::elect_one();
  const int feature_base = is_fp8
      ? feature_tile * FP8_UMMA_M
      : N_FP8 + feature_tile * BF16_UMMA_M;
  const int local_w_row = is_fp8
      ? feature_base + rank * FP8_CTA_N
      : feature_tile * BF16_UMMA_M + rank * BF16_CTA_N;
  const int local_x_row =
      batch_base + rank * (BatchN / CLUSTER_SIZE);
  uint8_t* const smem_act = s.storage;
  uint8_t* const smem_wgt = smem_act + (is_fp8
      ? C::FP8_STAGES * C::FP8_ACT_STAGE_BYTES
      : C::BF16_STAGES * C::BF16_ACT_STAGE_BYTES);
  uint8_t* const sf_weight = s.storage + C::FP8_PIPELINE_BYTES;
  uint8_t* const sf_act = sf_weight + SF_BYTES;

  if constexpr (RecordTimestamps) {
    if (rank == 0 && warp == 0 && elected) {
      task_times[static_cast<size_t>(task) * TIMING_FIELDS] =
          read_globaltimer();
    }
  }

  if (warp == 0 && elected) {
    base::prefetch_tensormap(is_fp8 ? &desc_x8 : &desc_x16);
    base::prefetch_tensormap(is_fp8 ? &desc_w8 : &desc_w16);
  }
  if (warp == 1) {
    if (lane < C::MAX_STAGES) {
      s.full[lane].init(1);
      s.empty[lane].init(1);
    } else if (lane == C::MAX_STAGES) {
      s.tmem_full.init(1);
    }
    __syncwarp();
    if (lane == 0) {
      base::fence_barrier_init();
      if constexpr (RecordTimestamps) {
        if (rank == 0) {
          task_times[static_cast<size_t>(task) * TIMING_FIELDS + 1] =
              read_globaltimer();
        }
      }
    }
  }
  if (warp == 2) {
    uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&s.tmem_base));
    base::tmem_alloc_2sm(addr, C::NUM_TMEM_COLS);
    if constexpr (RecordTimestamps) {
      if (rank == 0 && lane == 0) {
        task_times[static_cast<size_t>(task) * TIMING_FIELDS + 2] =
            read_globaltimer();
      }
    }
  }

  // Resident FP8 scale factors. Each CTA owns 128 FP8 feature rows, exactly
  // one weight-scale block. Both CTAs replicate the full UMMA_N activation
  // scale tile; rows outside the logical batch are padded to 1.0.
  if (warp == 3 && is_fp8) {
    const int w_sf_row = local_w_row / 128;
    uint32_t packed_w[SF_GROUPS];
    #pragma unroll
    for (int g = 0; g < SF_GROUPS; ++g) {
      packed_w[g] = *reinterpret_cast<const uint32_t*>(
          w_sf + w_sf_row * FP8_SCALE_TILES + g * 4);
    }
    for (int l = lane; l < SF_ROWS; l += 32) {
      const int row = batch_base + l;
      const bool valid_x = l < BatchN && row < problem_m;
      const int slot = (l % 32) * 4 + l / 32;
      #pragma unroll
      for (int g = 0; g < SF_GROUPS; ++g) {
        const uint32_t av = valid_x
            ? *reinterpret_cast<const uint32_t*>(
                  x_sf + static_cast<size_t>(row) * FP8_SCALE_TILES + g * 4)
            : UE8M0_ONE4;
        base::st_shared_u32(sf_weight + (g * SF_ROWS + slot) * 4,
                            packed_w[g]);
        base::st_shared_u32(sf_act + (g * SF_ROWS + slot) * 4, av);
      }
    }
    base::fence_view_async_shared();
    if constexpr (RecordTimestamps) {
      if (rank == 0 && lane == 0) {
        task_times[static_cast<size_t>(task) * TIMING_FIELDS + 3] =
            read_globaltimer();
      }
    }
  }
  base::cluster_sync();
  if constexpr (RecordTimestamps) {
    if (rank == 0 && warp == 0 && elected) {
      task_times[static_cast<size_t>(task) * TIMING_FIELDS + 4] =
          read_globaltimer();
    }
  }

  const auto act_desc = base::make_smem_desc(smem_act);
  const auto wgt_desc = base::make_smem_desc(smem_wgt);

  if (warp == 0 && elected) {
    const uint16_t self_mask = static_cast<uint16_t>(1u << rank);
    if (is_fp8) {
      constexpr int tx_bytes = CLUSTER_SIZE *
          (C::FP8_ACT_DATA_BYTES + C::FP8_WGT_STAGE_BYTES);
      #pragma unroll 1
      for (int k_tile = 0; k_tile < C::FP8_K_TILES; ++k_tile) {
        const uint32_t st = static_cast<uint32_t>(k_tile) % C::FP8_STAGES;
        const uint32_t phase =
            (static_cast<uint32_t>(k_tile) / C::FP8_STAGES) & 1;
        s.empty[st].wait(phase ^ 1);
        if (rank == 0) {
          s.full[st].arrive_and_expect_tx(tx_bytes);
        }
        uint8_t* act_stage = smem_act + st * C::FP8_ACT_STAGE_BYTES;
        uint8_t* wgt_stage = smem_wgt + st * C::FP8_WGT_STAGE_BYTES;
        #pragma unroll
        for (int sub = 0; sub < C::FP8_BLOCK_K / FP8_TMA_K; ++sub) {
          base::tma_load_2sm(
              &desc_x8, &s.full[st],
              act_stage + sub *
                  (C::FP8_ACT_DATA_BYTES /
                      (C::FP8_BLOCK_K / FP8_TMA_K)), self_mask,
              k_tile * C::FP8_BLOCK_K + sub * FP8_TMA_K, local_x_row);
          base::tma_load_2sm(
              &desc_w8, &s.full[st],
              wgt_stage + sub *
                  (C::FP8_WGT_STAGE_BYTES /
                      (C::FP8_BLOCK_K / FP8_TMA_K)), self_mask,
              k_tile * C::FP8_BLOCK_K + sub * FP8_TMA_K, local_w_row);
        }
      }
    } else {
      constexpr int tx_bytes = CLUSTER_SIZE *
          (C::BF16_ACT_STAGE_BYTES + C::BF16_WGT_STAGE_BYTES);
      #pragma unroll 1
      for (int k_tile = 0; k_tile < C::BF16_K_TILES; ++k_tile) {
        const uint32_t st = static_cast<uint32_t>(k_tile) % C::BF16_STAGES;
        const uint32_t phase =
            (static_cast<uint32_t>(k_tile) / C::BF16_STAGES) & 1;
        s.empty[st].wait(phase ^ 1);
        if (rank == 0) {
          s.full[st].arrive_and_expect_tx(tx_bytes);
        }
        uint8_t* act_stage = smem_act + st * C::BF16_ACT_STAGE_BYTES;
        uint8_t* wgt_stage = smem_wgt + st * C::BF16_WGT_STAGE_BYTES;
        #pragma unroll
        for (int sub = 0; sub < C::BF16_BLOCK_K / BF16_TMA_K; ++sub) {
          base::tma_load_2sm(
              &desc_x16, &s.full[st],
              act_stage + sub *
                  (C::BF16_ACT_STAGE_BYTES /
                      (C::BF16_BLOCK_K / BF16_TMA_K)), self_mask,
              k_tile * C::BF16_BLOCK_K + sub * BF16_TMA_K, local_x_row);
          base::tma_load_2sm(
              &desc_w16, &s.full[st],
              wgt_stage + sub *
                  (C::BF16_WGT_STAGE_BYTES /
                      (C::BF16_BLOCK_K / BF16_TMA_K)), self_mask,
              k_tile * C::BF16_BLOCK_K + sub * BF16_TMA_K, local_w_row);
        }
      }
    }
    if constexpr (RecordTimestamps) {
      if (rank == 0) {
        task_times[static_cast<size_t>(task) * TIMING_FIELDS + 7] =
            read_globaltimer();
      }
    }
  } else if (warp == 5) {
    // Warp 5 is idle during the mainloop. Run the CSA post-gate + Sinkhorn
    // tail here and rejoin through the existing epilogue named barrier.
    if (hc.mix != nullptr) {
      base::hc_tail_run(hc);
    }
  } else if (warp == 1 && rank == 0 && elected) {
    constexpr uint16_t CTA_MASK = 0x3;
    base::tmem_fence_after_sync();
    if (is_fp8) {
        constexpr int act_bytes = C::FP8_ACT_DATA_BYTES;
        constexpr int wgt_bytes = C::FP8_WGT_STAGE_BYTES;
        const auto idesc = fp8_idesc<BatchN>();
        #pragma unroll 1
        for (int k_tile = 0; k_tile < C::FP8_K_TILES; ++k_tile) {
          const uint32_t st = static_cast<uint32_t>(k_tile) % C::FP8_STAGES;
          const uint32_t phase =
              (static_cast<uint32_t>(k_tile) / C::FP8_STAGES) & 1;
          s.full[st].wait(phase);
          if constexpr (RecordTimestamps) {
            if (k_tile == 0) {
              task_times[static_cast<size_t>(task) * TIMING_FIELDS + 5] =
                  read_globaltimer();
            } else if (k_tile == C::FP8_K_TILES - 1) {
              task_times[static_cast<size_t>(task) * TIMING_FIELDS + 8] =
                  read_globaltimer();
            }
          }
          const uint32_t w_base =
              wgt_desc.lo + st * (C::FP8_WGT_STAGE_BYTES / 16);
          const uint32_t x_base =
              act_desc.lo + st * (C::FP8_ACT_STAGE_BYTES / 16);
          #pragma unroll
          for (int half = 0; half < C::FP8_BLOCK_K / FP8_TMA_K; ++half) {
            const int scale_tile =
                k_tile * (C::FP8_BLOCK_K / FP8_TMA_K) + half;
            if ((scale_tile & 3) == 0) {
              const int g = scale_tile / 4;
              auto sfd = fp8d::make_sf_desc();
              fp8d::replace_sf_desc_addr(
                  sfd, sf_weight + g * SF_ROWS * 4);
              fp8d::utccp_4x32_2cta(
                  C::SF_WGT_COL, fp8d::sf_desc_bits(sfd));
              fp8d::replace_sf_desc_addr(
                  sfd, sf_act + g * SF_ROWS * 4);
              fp8d::utccp_4x32_2cta(
                  C::SF_ACT_COL, fp8d::sf_desc_bits(sfd));
            }
            const uint64_t rdesc = fp8d::make_runtime_idesc_with_sf_id(
                idesc, scale_tile & 3, scale_tile & 3);
            const uint32_t w_half =
                w_base + half *
                    (wgt_bytes / (C::FP8_BLOCK_K / FP8_TMA_K) / 16);
            const uint32_t x_half =
                x_base + half *
                    (act_bytes / (C::FP8_BLOCK_K / FP8_TMA_K) / 16);
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              const uint64_t w =
                  (static_cast<uint64_t>(wgt_desc.hi) << 32) |
                  (w_half + kk * 2);
              const uint64_t x =
                  (static_cast<uint64_t>(act_desc.hi) << 32) |
                  (x_half + kk * 2);
              fp8d::mma_2sm_block_scale(
                  0, w, x, rdesc,
                  (k_tile != 0 || half != 0 || kk != 0) ? 1u : 0u,
                  C::SF_WGT_COL, C::SF_ACT_COL);
            }
          }
          base::commit_2sm(&s.empty[st], CTA_MASK);
          if constexpr (RecordTimestamps) {
            if (k_tile == 0) {
              task_times[static_cast<size_t>(task) * TIMING_FIELDS + 6] =
                  read_globaltimer();
            }
          }
          if (k_tile == C::FP8_K_TILES - 1) {
            base::commit_2sm(&s.tmem_full, CTA_MASK);
            if constexpr (RecordTimestamps) {
              task_times[static_cast<size_t>(task) * TIMING_FIELDS + 9] =
                  read_globaltimer();
            }
          }
        }
    } else {
      constexpr int act_bytes = C::BF16_ACT_STAGE_BYTES;
      constexpr int wgt_bytes = C::BF16_WGT_STAGE_BYTES;
      const uint64_t idesc = bf16_idesc<BatchN>();
      #pragma unroll 1
      for (int k_tile = 0; k_tile < C::BF16_K_TILES; ++k_tile) {
        const uint32_t st = static_cast<uint32_t>(k_tile) % C::BF16_STAGES;
        const uint32_t phase =
            (static_cast<uint32_t>(k_tile) / C::BF16_STAGES) & 1;
        s.full[st].wait(phase);
        if constexpr (RecordTimestamps) {
          if (k_tile == 0) {
            task_times[static_cast<size_t>(task) * TIMING_FIELDS + 5] =
                read_globaltimer();
          } else if (k_tile == C::BF16_K_TILES - 1) {
            task_times[static_cast<size_t>(task) * TIMING_FIELDS + 8] =
                read_globaltimer();
          }
        }
        const uint32_t w_base =
            wgt_desc.lo + st * (C::BF16_WGT_STAGE_BYTES / 16);
        const uint32_t x_base =
            act_desc.lo + st * (C::BF16_ACT_STAGE_BYTES / 16);
        #pragma unroll
        for (int kk = 0; kk < C::BF16_BLOCK_K / 16; ++kk) {
          const int atom = kk / (BF16_TMA_K / 16);
          const int k_in_atom = kk % (BF16_TMA_K / 16);
          const uint64_t w = (static_cast<uint64_t>(wgt_desc.hi) << 32) |
              (w_base + atom *
                  (wgt_bytes / (C::BF16_BLOCK_K / BF16_TMA_K) / 16) +
                  k_in_atom * 2);
          const uint64_t x = (static_cast<uint64_t>(act_desc.hi) << 32) |
              (x_base + atom *
                  (act_bytes / (C::BF16_BLOCK_K / BF16_TMA_K) / 16) +
                  k_in_atom * 2);
          mma_2sm_bf16(
              0, w, x, idesc, (k_tile != 0 || kk != 0) ? 1u : 0u);
        }
        base::commit_2sm(&s.empty[st], CTA_MASK);
        if constexpr (RecordTimestamps) {
          if (k_tile == 0) {
            task_times[static_cast<size_t>(task) * TIMING_FIELDS + 6] =
                read_globaltimer();
          }
        }
        if (k_tile == C::BF16_K_TILES - 1) {
          base::commit_2sm(&s.tmem_full, CTA_MASK);
          if constexpr (RecordTimestamps) {
            task_times[static_cast<size_t>(task) * TIMING_FIELDS + 9] =
                read_globaltimer();
          }
        }
      }
    }
  }

  if (warp == 0 && elected) {
    s.tmem_full.wait(0);
  }
  if (warp == 0 || warp >= 4) {
    cutlass::arch::NamedBarrier::sync(160, 0);
  }
  if (warp >= 4) {
    base::tmem_fence_after_sync();
  }
  if constexpr (RecordTimestamps) {
    if (rank == 0 && warp == 0 && elected) {
      task_times[static_cast<size_t>(task) * TIMING_FIELDS + 10] =
          read_globaltimer();
    }
  }

  if (is_fp8) {
    constexpr int batch_stores = BatchN / STORE_M;
    const bool emit_win = emit.main_state != nullptr && feature_base >= 1536;
    #pragma unroll
    for (int st = 0; st < batch_stores; ++st) {
      if (warp >= 4) {
        const int epi_warp = warp - 4;
        const int out_col = feature_base + rank * FP8_CTA_N +
                            epi_warp * STORE_N + lane;
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
          base::tmem_load_8x(st * STORE_M + sub * 8,
                            v0, v1, v2, v3, v4, v5, v6, v7);
          base::tmem_load_fence();
          const uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
          #pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int out_row = batch_base + st * STORE_M + sub * 8 + r;
            if (out_row < problem_m) {
              const float value = __uint_as_float(vals[r]);
              if (emit_win) {
                emit.win_y2[static_cast<size_t>(out_row) * 512 +
                            out_col - 1536] =
                    __bfloat162float(__float2bfloat16_rn(value));
              } else {
                out[static_cast<size_t>(out_row) * N + out_col] =
                    __float2bfloat16_rn(value);
              }
            }
          }
        }
      }
    }
  } else {
    // 2SM M128 is a 2x2 quadrant layout: CTA rank selects the 64-feature
    // half, while epilogue warp pairs 0/1 and 2/3 select the two BatchN/2
    // halves. Physical warps 4..7 form the dedicated epilogue warpgroup.
    // The bf16 TAIL tile (feature_base 4608) has 64 valid rows: rank1's
    // columns land at >= N and are clipped.
    if (warp >= 4 && emit.main_state != nullptr && feature_base >= 4608) {
      const int epi_warp = warp - 4;
      const int out_col = feature_base + rank * BF16_CTA_N +
                          (epi_warp & 1) * 32 + lane;
      if (out_col < N) {
        #pragma unroll
        for (int c = 0; c < BatchN / CLUSTER_SIZE; c += 8) {
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
          base::tmem_load_8x(c, v0, v1, v2, v3, v4, v5, v6, v7);
          base::tmem_load_fence();
          const uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
          #pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int out_row = batch_base + (epi_warp >> 1) *
                (BatchN / CLUSTER_SIZE) + c + r;
            if (out_row < problem_m) {
              emit.w64[static_cast<size_t>(out_row) * 64 + out_col - 4608] =
                  __bfloat162float(
                      __float2bfloat16_rn(__uint_as_float(vals[r])));
            }
          }
        }
      }
    } else if (warp >= 4 && emit.main_state != nullptr) {
      // swapAB's natural TMEM drain assigns one feature column to each lane.
      // Transpose 32x32 chunks in shared memory so each lane owns one output
      // row and can publish aligned 32B vectors to the sparse RTP state rings.
      // The pipeline storage is dead after tmem_full, and each epilogue warp
      // owns a disjoint 4KB region.
      float* tile = reinterpret_cast<float*>(s.storage) +
                    (warp - 4) * 32 * 32;
      const int row_half_base = ((warp - 4) >> 1) * (BatchN / CLUSTER_SIZE);
      const int feature_chunk = feature_base + rank * BF16_CTA_N +
                                ((warp - 4) & 1) * 32;
      const bool is_main = feature_base < 4096;
      const int family_base = is_main ? 2048 : 4096;
      const int state_width = is_main ? 2048 : 512;
      const int half_width = state_width / 2;
      const bool add_ape = feature_base >= (is_main ? 3072 : 4352);
      float* const state = is_main ? emit.main_state : emit.idx_state;
      const float* const ape = is_main ? emit.main_ape : emit.idx_ape;
      const int feature_offset = feature_chunk - family_base;
      #pragma unroll
      for (int wave = 0; wave < BatchN / CLUSTER_SIZE; wave += 32) {
        #pragma unroll
        for (int c = 0; c < 32; c += 8) {
          if (wave + c < BatchN / CLUSTER_SIZE) {
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
            base::tmem_load_8x(wave + c, v0, v1, v2, v3,
                              v4, v5, v6, v7);
            base::tmem_load_fence();
            const uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
              const int tile_row = c + r;
              const int swizzled_col =
                  (((lane >> 3) ^ (tile_row & 3)) << 3) + (lane & 7);
              tile[tile_row * 32 + swizzled_col] =
                  __uint_as_float(vals[r]);
            }
          }
        }
        __syncwarp();

        const int row_in_half = wave + lane;
        const int out_row = batch_base + row_half_base + row_in_half;
        if (row_in_half < BatchN / CLUSTER_SIZE &&
            out_row < problem_m) {
          const int state_row = is_main ? emit.main_state_row[out_row]
                                        : emit.idx_state_row[out_row];
          if (state_row >= 0) {
            float* const dst_row = state +
                static_cast<size_t>(state_row) * state_width + feature_offset;
            if (add_ape) {
              const int phase =
                  emit.ape_phase[out_row] & (base::APE_RATIO - 1);
              const float* const ape_row = ape +
                  static_cast<size_t>(phase) * half_width +
                  feature_offset - half_width;
              #pragma unroll
              for (int q = 0; q < 4; ++q) {
                const int physical_q = q ^ (lane & 3);
                float4 a = *reinterpret_cast<const float4*>(
                    tile + lane * 32 + physical_q * 8);
                float4 b = *reinterpret_cast<const float4*>(
                    tile + lane * 32 + physical_q * 8 + 4);
                const float4 p0 = *reinterpret_cast<const float4*>(
                    ape_row + q * 8);
                const float4 p1 = *reinterpret_cast<const float4*>(
                    ape_row + q * 8 + 4);
                a.x += p0.x; a.y += p0.y;
                a.z += p0.z; a.w += p0.w;
                b.x += p1.x; b.y += p1.y;
                b.z += p1.z; b.w += p1.w;
                reinterpret_cast<float4*>(dst_row + q * 8)[0] = a;
                reinterpret_cast<float4*>(dst_row + q * 8)[1] = b;
              }
            } else {
              #pragma unroll
              for (int q = 0; q < 4; ++q) {
                const int physical_q = q ^ (lane & 3);
                const float4 a = *reinterpret_cast<const float4*>(
                    tile + lane * 32 + physical_q * 8);
                const float4 b = *reinterpret_cast<const float4*>(
                    tile + lane * 32 + physical_q * 8 + 4);
                reinterpret_cast<float4*>(dst_row + q * 8)[0] = a;
                reinterpret_cast<float4*>(dst_row + q * 8)[1] = b;
              }
            }
          }
        }
        __syncwarp();
      }
    } else if (warp >= 4) {
      const int epi_warp = warp - 4;
      const int feature_lane =
          rank * BF16_CTA_N + (epi_warp & 1) * 32 + lane;
      const int out_col = feature_base + feature_lane;
      if (out_col < N) {
        #pragma unroll
        for (int c = 0; c < BatchN / CLUSTER_SIZE; c += 8) {
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
          base::tmem_load_8x(c, v0, v1, v2, v3, v4, v5, v6, v7);
          base::tmem_load_fence();
          const uint32_t vals[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
          #pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int out_row = batch_base + (epi_warp >> 1) *
                (BatchN / CLUSTER_SIZE) + c + r;
            if (out_row < problem_m) {
              out[static_cast<size_t>(out_row) * N + out_col] =
                  __float2bfloat16_rn(__uint_as_float(vals[r]));
            }
          }
        }
      }
    }
  }
  if (warp >= 4) {
    base::tmem_fence_before_sync();
  }
  if (warp == 2 || warp >= 4) {
    cutlass::arch::NamedBarrier::sync(160, 1);
  }
  if constexpr (RecordTimestamps) {
    if (rank == 0 && warp == 4 && elected) {
      task_times[static_cast<size_t>(task) * TIMING_FIELDS + 11] =
          read_globaltimer();
    }
  }
  if (warp == 2) {
    base::tmem_dealloc_2sm(0, C::NUM_TMEM_COLS);
    if constexpr (RecordTimestamps) {
      if (rank == 0 && lane == 0) {
        task_times[static_cast<size_t>(task) * TIMING_FIELDS + 12] =
            read_globaltimer();
      }
    }
  }
}

template <int BatchN, bool RecordTimestamps = false>
inline cudaError_t configure_kernel() {
  return cudaFuncSetAttribute(
      swapab_kernel<BatchN, RecordTimestamps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(SharedStorage<BatchN>));
}

template <int BatchN, bool RecordTimestamps = false>
inline cudaError_t launch(
    const CUtensorMap& desc_x16, const CUtensorMap& desc_w16,
    const CUtensorMap& desc_x8, const CUtensorMap& desc_w8,
    const uint8_t* x_sf, const uint8_t* w_sf, __nv_bfloat16* out,
    const base::HcTailArgs& hc, const base::FrontEmitArgs& emit,
    int m, uint64_t* task_times, cudaStream_t stream) {
  const int batch_tiles = (m + BatchN - 1) / BatchN;
  void* args[] = {
      const_cast<CUtensorMap*>(&desc_x16),
      const_cast<CUtensorMap*>(&desc_w16),
      const_cast<CUtensorMap*>(&desc_x8),
      const_cast<CUtensorMap*>(&desc_w8),
      &x_sf, &w_sf, &out, const_cast<base::HcTailArgs*>(&hc),
      const_cast<base::FrontEmitArgs*>(&emit),
      &m, &task_times};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(
      batch_tiles * TASKS_PER_BATCH * CLUSTER_SIZE, 1, 1);
  config.blockDim = dim3(TPB, 1, 1);
  config.dynamicSmemBytes = sizeof(SharedStorage<BatchN>);
  config.stream = stream;
  cudaLaunchAttribute cluster_attr{};
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = CLUSTER_SIZE;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;
  return cudaLaunchKernelExC(
      &config,
      reinterpret_cast<void*>(swapab_kernel<BatchN, RecordTimestamps>), args);
}

}  // namespace front_mixed_csa_swapab
