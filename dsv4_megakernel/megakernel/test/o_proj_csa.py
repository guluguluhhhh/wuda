"""Official DeepSeek-V4 O projection for the B300 decode pipeline.

VERBATIM port of mega_csa python/mega_csa/o_proj.py (Flash_DeepSeek_V4_Pro).
Only the two package-internal imports were inlined so this drops into the
megakernel test tree standalone: the DeepGEMM runtime scopes and the handful
of DEEPSEEK_V4_PRO_CSA constants this module reads.

The ownership boundary matches vLLM's DeepSeek-V4 implementation:
raw FlashMLA BF16 output -> inverse RoPE + FP8 quant -> DeepGEMM wo_a ->
FP8 quant -> DeepGEMM wo_b -> mHC post.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Iterator

import torch
import triton
import triton.language as tl


# ---- inlined from mega_csa/_deepgemm_runtime.py ------------------------
# The pinned DeepGEMM runtime stores PDL enablement in one process-wide
# DeviceRuntime singleton. Keep configuration and all affected host enqueues in
# one critical section so concurrent pipelines cannot observe each other's PDL
# choice. The lock ends after enqueue; it never waits for GPU completion.
_DEEPGEMM_RUNTIME_LOCK = threading.RLock()
_BOUND_DEVICE_INDEX: int | None = None


@contextmanager
def deepgemm_device_scope(device_index: int) -> Iterator[None]:
    """Serialize DeepGEMM use and bind its process singleton to one device."""

    global _BOUND_DEVICE_INDEX
    with _DEEPGEMM_RUNTIME_LOCK:
        if _BOUND_DEVICE_INDEX is None:
            _BOUND_DEVICE_INDEX = device_index
        elif _BOUND_DEVICE_INDEX != device_index:
            raise RuntimeError(
                "the pinned DeepGEMM runtime is process-global and already "
                f"bound to CUDA device {_BOUND_DEVICE_INDEX}; cannot use "
                f"DeepGEMM paths on CUDA device {device_index}"
            )
        yield


@contextmanager
def deepgemm_pdl_enqueue_scope(
    deep_gemm: Any, enable_pdl: bool, device_index: int
) -> Iterator[None]:
    """Temporarily set DeepGEMM PDL while serializing affected enqueues."""

    with deepgemm_device_scope(device_index):
        previous_pdl = deep_gemm.get_pdl()
        try:
            deep_gemm.set_pdl(enable_pdl)
            yield
        finally:
            deep_gemm.set_pdl(previous_pdl)


# ---- inlined from mega_csa/model_contract.py (DEEPSEEK_V4_PRO_CSA) ----
HEADS = 128              # num_attention_heads
HEAD_DIM = 512           # head_dim
ROPE_DIM = 64            # qk_rope_head_dim
NOPE_DIM = HEAD_DIM - ROPE_DIM
N_GROUPS = 16            # o_groups
O_LORA_RANK = 1024       # o_lora_rank
O_INTERMEDIATE_DIM = N_GROUPS * O_LORA_RANK
HIDDEN_DIM = 7168        # hidden_size
QUANT_GROUP_SIZE = 128
FP8_MAX = 448.0


@dataclass(frozen=True)
class OProjWeights:
    """Prequantized DeepGEMM weights; scale tensors are already TMA layout."""

    wo_a: torch.Tensor       # FP8 [16,1024,heads_per_group*512]
    wo_a_scale: torch.Tensor # I32 [16,1024,heads_per_group], MN-major
    wo_b: torch.Tensor       # FP8 [7168,16384]
    wo_b_scale: torch.Tensor # I32 [7168,32], MN-major


@dataclass(frozen=True)
class OProjWorkspace:
    o_fp8: torch.Tensor       # FP8 [M,16,4096]
    o_scale: torch.Tensor     # I32 [M,16,8], MN-major within each group
    z: torch.Tensor           # BF16 [M,16,1024]
    z_fp8: torch.Tensor       # FP8 [M,16384]
    z_scale: torch.Tensor     # I32 [M,32], MN-major
    projected: torch.Tensor   # BF16 [M,7168]
    mhc_output: torch.Tensor  # BF16 [M,4,7168]


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _storage_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the complete backing-storage range without synchronizing CUDA."""

    storage = tensor.untyped_storage()
    begin = storage.data_ptr()
    return begin, begin + storage.nbytes()


def _validate_workspace_aliases(
    writes: tuple[tuple[str, torch.Tensor], ...],
    reads: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Reject aliases before any O-projection stage mutates data.

    Scale workspaces contain padding outside their logical tensor views.
    Comparing complete backing storages is deliberately conservative and also
    covers those kernel-owned padding writes.
    """

    write_ranges = tuple(
        (name, _storage_byte_range(tensor)) for name, tensor in writes
    )
    read_ranges = tuple(
        (name, _storage_byte_range(tensor)) for name, tensor in reads
    )
    for index, (write_name, write_range) in enumerate(write_ranges):
        for other_name, other_range in write_ranges[index + 1 :]:
            _require(
                not (
                    write_range[0] < other_range[1]
                    and other_range[0] < write_range[1]
                ),
                f"{write_name} aliases writable {other_name}",
            )
        for read_name, read_range in read_ranges:
            _require(
                not (
                    write_range[0] < read_range[1]
                    and read_range[0] < write_range[1]
                ),
                f"{write_name} aliases live input {read_name}",
            )


@triton.jit(do_not_specialize=["num_tokens"])
def _inv_rope_quant_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    HEAD_DIM_T: tl.constexpr,
    NOPE_DIM_T: tl.constexpr,
    ROPE_DIM_T: tl.constexpr,
    QUANT_GROUP_SIZE_T: tl.constexpr,
    FP8_MAX_T: tl.constexpr,
    USE_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    global_head = tl.program_id(1).to(tl.int64)
    group = global_head // heads_per_group
    head_in_group = global_head % heads_per_group
    if USE_PDL:
        # This is a middle PDL stage: release the dependent grid so its
        # initialization can overlap, then wait before reading MLA output.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    if token >= num_tokens:
        tl.store(
            scale_ptr + group * scale_stride_group + token
            + head_in_group * scale_stride_k,
            tl.zeros((), dtype=tl.int32),
        )
        return

    offsets = tl.arange(0, HEAD_DIM_T)
    base = o_ptr + token * o_stride_token + global_head * o_stride_head
    x = tl.load(base + offsets).to(tl.float32)

    is_rope = offsets >= NOPE_DIM_T
    rope_local = offsets - NOPE_DIM_T
    partner = tl.load(base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
    position = tl.load(positions_ptr + token)
    cache = cos_sin_ptr + position * cache_stride_pos
    pair = tl.maximum(rope_local >> 1, 0)
    cos = tl.load(cache + pair, mask=is_rope, other=1.0)
    sin = tl.load(cache + ROPE_DIM_T // 2 + pair, mask=is_rope, other=0.0)
    even = x * cos + partner * sin
    odd = x * cos - partner * sin
    rotated = tl.where((rope_local & 1) == 0, even, odd)
    x = tl.where(is_rope, rotated, x)
    # Official apply_rotary_emb writes back into BF16 `o` before wo_a. Keep
    # that model-dtype boundary in registers before DeepGEMM activation cast.
    x = x.to(tl.bfloat16).to(tl.float32)

    absolute = tl.reshape(tl.abs(x), (4, QUANT_GROUP_SIZE_T))
    # Match official DeepGEMM per-token activation quantization exactly.
    amax = tl.maximum(tl.max(absolute, axis=1), 1e-4)
    scale = tl.math.exp2(tl.ceil(tl.log2(amax / FP8_MAX_T)))
    expanded = tl.reshape(
        tl.broadcast_to(tl.reshape(scale, (4, 1)), (4, QUANT_GROUP_SIZE_T)),
        (HEAD_DIM_T,),
    )
    quantized = tl.clamp(x / expanded, -FP8_MAX_T, FP8_MAX_T).to(tl.float8e4nv)
    fp8_base = (
        fp8_ptr + group * fp8_stride_group + token * fp8_stride_token
        + head_in_group * HEAD_DIM_T
    )
    tl.store(fp8_base + offsets, quantized)

    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 0xFF
    shifts = tl.arange(0, 4) * 8
    packed = tl.sum(exponent << shifts)
    tl.store(
        scale_ptr + group * scale_stride_group + token
        + head_in_group * scale_stride_k,
        packed,
    )


@triton.jit(do_not_specialize=["num_tokens"])
def _quant_o_lora_kernel(
    x_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    x_stride_token,
    fp8_stride_token,
    scale_stride_k,
    QUANT_GROUP_SIZE_T: tl.constexpr,
    FP8_MAX_T: tl.constexpr,
    USE_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    packed_block = tl.program_id(1).to(tl.int64)
    if USE_PDL:
        # Match the official vLLM middle-stage PDL protocol.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    if token >= num_tokens:
        tl.store(scale_ptr + token + packed_block * scale_stride_k, 0)
        return

    offsets = packed_block * 512 + tl.arange(0, 512)
    x = tl.load(x_ptr + token * x_stride_token + offsets).to(tl.float32)
    absolute = tl.reshape(tl.abs(x), (4, QUANT_GROUP_SIZE_T))
    amax = tl.maximum(tl.max(absolute, axis=1), 1e-4)
    scale = tl.math.exp2(tl.ceil(tl.log2(amax / FP8_MAX_T)))
    expanded = tl.reshape(
        tl.broadcast_to(tl.reshape(scale, (4, 1)), (4, QUANT_GROUP_SIZE_T)),
        (512,),
    )
    quantized = tl.clamp(x / expanded, -FP8_MAX_T, FP8_MAX_T).to(tl.float8e4nv)
    tl.store(fp8_ptr + token * fp8_stride_token + offsets, quantized)
    exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 0xFF
    packed = tl.sum(exponent << (tl.arange(0, 4) * 8))
    tl.store(scale_ptr + token + packed_block * scale_stride_k, packed)


def _check_weights(weights: OProjWeights, heads_per_group: int) -> None:
    d = heads_per_group * HEAD_DIM
    _require(weights.wo_a.shape == (N_GROUPS, O_LORA_RANK, d) and
             weights.wo_a.dtype == torch.float8_e4m3fn and
             weights.wo_a.is_cuda and weights.wo_a.is_contiguous(),
             f"wo_a must be FP8 [{N_GROUPS},{O_LORA_RANK},{d}]")
    _require(weights.wo_a_scale.shape == (N_GROUPS, O_LORA_RANK, d // 512) and
             weights.wo_a_scale.dtype == torch.int32 and weights.wo_a_scale.is_cuda and
             weights.wo_a_scale.stride() ==
             (O_LORA_RANK * (d // 512), 1, O_LORA_RANK),
             "wo_a_scale has the wrong shape, dtype, or TMA layout")
    _require(weights.wo_b.shape == (HIDDEN_DIM, O_INTERMEDIATE_DIM) and
             weights.wo_b.dtype == torch.float8_e4m3fn and
             weights.wo_b.is_cuda and weights.wo_b.is_contiguous(),
             f"wo_b must be FP8 [{HIDDEN_DIM},{O_INTERMEDIATE_DIM}]")
    _require(weights.wo_b_scale.shape ==
             (HIDDEN_DIM, O_INTERMEDIATE_DIM // 512) and
             weights.wo_b_scale.dtype == torch.int32 and weights.wo_b_scale.is_cuda and
             weights.wo_b_scale.stride() == (1, HIDDEN_DIM),
             "wo_b_scale has the wrong shape, dtype, or TMA layout")


@torch.inference_mode()
def prepare_o_proj_workspace(
    m: int,
    device: torch.device | str,
    *,
    heads: int = HEADS,
) -> OProjWorkspace:
    _require(1 <= m <= 128 and heads == HEADS and heads % N_GROUPS == 0,
             f"O projection requires M in [1,128] and heads={HEADS}")
    heads_per_group = heads // N_GROUPS
    d = heads_per_group * HEAD_DIM
    aligned_m = _align(m, 4)

    o_fp8_base = torch.empty(
        (N_GROUPS, m, d), device=device, dtype=torch.float8_e4m3fn
    )
    o_scale_base = torch.empty(
        N_GROUPS * heads_per_group * aligned_m,
        device=device,
        dtype=torch.int32,
    )
    o_scale_base = o_scale_base.as_strided(
        (N_GROUPS, m, heads_per_group),
        (heads_per_group * aligned_m, 1, aligned_m),
    )
    z = torch.empty(
        (m, N_GROUPS, O_LORA_RANK), device=device, dtype=torch.bfloat16
    )
    z_fp8 = torch.empty(
        (m, O_INTERMEDIATE_DIM), device=device, dtype=torch.float8_e4m3fn
    )
    z_scale_blocks = O_INTERMEDIATE_DIM // 512
    z_scale_base = torch.empty(
        aligned_m * z_scale_blocks, device=device, dtype=torch.int32
    )
    z_scale = z_scale_base.as_strided(
        (m, z_scale_blocks), (1, aligned_m)
    )
    return OProjWorkspace(
        o_fp8=o_fp8_base.transpose(0, 1),
        o_scale=o_scale_base.transpose(0, 1),
        z=z,
        z_fp8=z_fp8,
        z_scale=z_scale,
        projected=torch.empty((m, HIDDEN_DIM), device=device, dtype=torch.bfloat16),
        mhc_output=torch.empty(
            (m, 4, HIDDEN_DIM), device=device, dtype=torch.bfloat16
        ),
    )


@torch.inference_mode()
def run_o_proj_mhc_post(
    mla_out: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    weights: OProjWeights,
    workspace: OProjWorkspace,
    mhc_post_module: Any,
    *,
    use_pdl: bool = True,
    run_mhc_post: bool = True,
) -> torch.Tensor:
    """Run allocation-free O projection and mHC post on the current stream.

    run_mhc_post=False stops after wo_b and returns workspace.projected: the
    caller then owns the mHC-post launch (the e2e bench does this to time the
    two halves as separate operators). mhc_post's PDL is its own launch
    attribute, so splitting the call does not change the pipeline protocol.
    """

    import deep_gemm

    if mla_out.dim() == 4:
        _require(mla_out.shape[1] == 1, "4D mla_out must have singleton scope dim")
        mla_out = mla_out[:, 0]
    _require(mla_out.dim() == 3, "mla_out must be [M,128,512] or [M,1,128,512]")
    m, heads, head_dim = mla_out.shape
    _require(mla_out.dtype == torch.bfloat16 and mla_out.is_cuda and
             mla_out.is_contiguous(), "mla_out must be contiguous CUDA BF16")
    _require(1 <= m <= 128 and heads == HEADS and head_dim == HEAD_DIM,
             f"mla_out must be [M,{HEADS},{HEAD_DIM}] with M in [1,128]")
    _require(positions.shape == (m,) and positions.dtype == torch.int64 and
             positions.is_cuda and positions.is_contiguous(),
             "positions must be contiguous CUDA I64 [M]")
    _require(cos_sin_cache.ndim == 2 and cos_sin_cache.shape[1] == ROPE_DIM and
             cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_cuda and
             cos_sin_cache.is_contiguous(),
             f"cos_sin_cache must be contiguous CUDA FP32 [max_pos,{ROPE_DIM}]")
    _require(residual.shape == (m, 4, HIDDEN_DIM) and
             residual.dtype == torch.bfloat16 and residual.is_cuda and
             residual.is_contiguous(),
             f"residual must be contiguous CUDA BF16 [M,4,{HIDDEN_DIM}]")
    _require(post.shape == (m, 4) and post.dtype == torch.float32 and
             post.is_cuda and post.is_contiguous(),
             "post must be contiguous CUDA FP32 [M,4]")
    _require(comb.shape == (m, 4, 4) and comb.dtype == torch.float32 and
             comb.is_cuda and comb.is_contiguous(),
             "comb must be contiguous CUDA FP32 [M,4,4]")
    device = mla_out.device
    _require(all(t.device == device for t in (
        positions, cos_sin_cache, residual, post, comb, weights.wo_a,
        weights.wo_a_scale, weights.wo_b, weights.wo_b_scale,
        workspace.o_fp8, workspace.o_scale, workspace.z, workspace.z_fp8,
        workspace.z_scale, workspace.projected, workspace.mhc_output,
    )), "all O-projection tensors must be on the same CUDA device")
    heads_per_group = heads // N_GROUPS
    _check_weights(weights, heads_per_group)
    aligned_m = _align(m, 4)
    d = heads_per_group * HEAD_DIM
    _require(workspace.o_fp8.shape == (m, N_GROUPS, d) and
             workspace.o_fp8.dtype == torch.float8_e4m3fn and
             workspace.o_fp8.stride() == (d, m * d, 1),
             "workspace.o_fp8 has the wrong shape, dtype, or grouped layout")
    _require(workspace.o_scale.shape == (m, N_GROUPS, heads_per_group) and
             workspace.o_scale.dtype == torch.int32 and
             workspace.o_scale.stride() ==
             (1, heads_per_group * aligned_m, aligned_m),
             "workspace.o_scale has the wrong shape, dtype, or MN-major layout")
    _require(workspace.z.shape == (m, N_GROUPS, O_LORA_RANK) and
             workspace.z.dtype == torch.bfloat16 and workspace.z.is_contiguous(),
             "workspace.z must be contiguous BF16 [M,16,1024]")
    _require(workspace.z_fp8.shape == (m, O_INTERMEDIATE_DIM) and
             workspace.z_fp8.dtype == torch.float8_e4m3fn and
             workspace.z_fp8.is_contiguous(),
             "workspace.z_fp8 must be contiguous FP8 [M,16384]")
    _require(workspace.z_scale.shape == (m, O_INTERMEDIATE_DIM // 512) and
             workspace.z_scale.dtype == torch.int32 and
             workspace.z_scale.stride() == (1, aligned_m),
             "workspace.z_scale has the wrong shape, dtype, or MN-major layout")
    _require(workspace.projected.shape == (m, HIDDEN_DIM) and
             workspace.projected.dtype == torch.bfloat16 and
             workspace.projected.is_contiguous(),
             "workspace.projected must be contiguous BF16 [M,7168]")
    _require(workspace.mhc_output.shape == (m, 4, HIDDEN_DIM) and
             workspace.mhc_output.dtype == torch.bfloat16 and
             workspace.mhc_output.is_contiguous(),
             "workspace.mhc_output must be contiguous BF16 [M,4,7168]")
    _validate_workspace_aliases(
        (
            ("workspace.o_fp8", workspace.o_fp8),
            ("workspace.o_scale", workspace.o_scale),
            ("workspace.z", workspace.z),
            ("workspace.z_fp8", workspace.z_fp8),
            ("workspace.z_scale", workspace.z_scale),
            ("workspace.projected", workspace.projected),
            ("workspace.mhc_output", workspace.mhc_output),
        ),
        (
            ("mla_out", mla_out),
            ("positions", positions),
            ("cos_sin_cache", cos_sin_cache),
            ("residual", residual),
            ("post", post),
            ("comb", comb),
            ("weights.wo_a", weights.wo_a),
            ("weights.wo_a_scale", weights.wo_a_scale),
            ("weights.wo_b", weights.wo_b),
            ("weights.wo_b_scale", weights.wo_b_scale),
        ),
    )

    # The pinned DeepGEMM stores PDL enablement in one process-wide runtime.
    # Serialize the setting and complete enqueue chain, bind launches to the
    # input device, then restore the caller's setting without waiting on GPU.
    assert device.index is not None
    with torch.cuda.device(device):
        with deepgemm_pdl_enqueue_scope(deep_gemm, use_pdl, device.index):
            _inv_rope_quant_kernel[(aligned_m, heads)](
                mla_out,
                positions,
                cos_sin_cache,
                workspace.o_fp8,
                workspace.o_scale,
                m,
                heads_per_group=heads_per_group,
                o_stride_token=mla_out.stride(0),
                o_stride_head=mla_out.stride(1),
                cache_stride_pos=cos_sin_cache.stride(0),
                fp8_stride_group=workspace.o_fp8.stride(1),
                fp8_stride_token=workspace.o_fp8.stride(0),
                scale_stride_group=workspace.o_scale.stride(1),
                scale_stride_k=workspace.o_scale.stride(2),
                HEAD_DIM_T=HEAD_DIM,
                NOPE_DIM_T=NOPE_DIM,
                ROPE_DIM_T=ROPE_DIM,
                QUANT_GROUP_SIZE_T=QUANT_GROUP_SIZE,
                FP8_MAX_T=FP8_MAX,
                USE_PDL=use_pdl,
                launch_pdl=use_pdl,
                num_warps=1,
                num_stages=1,
            )
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (workspace.o_fp8, workspace.o_scale),
                (weights.wo_a, weights.wo_a_scale),
                workspace.z,
                recipe=(1, 1, 128),
            )

            _quant_o_lora_kernel[(aligned_m, O_INTERMEDIATE_DIM // 512)](
                workspace.z,
                workspace.z_fp8,
                workspace.z_scale,
                m,
                x_stride_token=workspace.z.stride(0),
                fp8_stride_token=workspace.z_fp8.stride(0),
                scale_stride_k=workspace.z_scale.stride(1),
                QUANT_GROUP_SIZE_T=QUANT_GROUP_SIZE,
                FP8_MAX_T=FP8_MAX,
                USE_PDL=use_pdl,
                launch_pdl=use_pdl,
                num_warps=4,
                num_stages=1,
            )
            deep_gemm.fp8_gemm_nt(
                (workspace.z_fp8, workspace.z_scale),
                (weights.wo_b, weights.wo_b_scale),
                workspace.projected,
                recipe=(1, 1, 128),
            )
            if run_mhc_post:
                mhc_post_module.mhc_post_out(
                    workspace.projected,
                    residual,
                    post,
                    comb,
                    workspace.mhc_output,
                    use_pdl,
                )
    return workspace.mhc_output if run_mhc_post else workspace.projected


@torch.inference_mode()
def quantize_o_proj_weights(
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
) -> OProjWeights:
    """Reference setup helper; never call in the decode hot region."""

    import deep_gemm
    from deep_gemm.utils import per_block_cast_to_fp8

    expected_wo_a_shape = (
        N_GROUPS, O_LORA_RANK, (HEADS // N_GROUPS) * HEAD_DIM
    )
    _require(wo_a.shape == expected_wo_a_shape and
             wo_a.dtype == torch.bfloat16 and wo_a.is_cuda and wo_a.is_contiguous(),
             f"wo_a must be contiguous CUDA BF16 {expected_wo_a_shape}")
    _require(wo_b.shape == (HIDDEN_DIM, O_INTERMEDIATE_DIM) and
             wo_b.dtype == torch.bfloat16 and wo_b.is_cuda and wo_b.is_contiguous(),
             f"wo_b must be contiguous CUDA BF16 [{HIDDEN_DIM},{O_INTERMEDIATE_DIM}]")
    _require(wo_a.device == wo_b.device, "wo_a and wo_b must be on the same device")
    device = wo_a.device
    assert device.index is not None
    with torch.cuda.device(device):
        with deepgemm_device_scope(device.index):
            wo_a_fp8 = torch.empty_like(wo_a, dtype=torch.float8_e4m3fn)
            wo_a_scale = torch.empty(
                (N_GROUPS, O_LORA_RANK // 128, wo_a.shape[2] // 128),
                device=device,
                dtype=torch.float32,
            )
            for group in range(N_GROUPS):
                wo_a_fp8[group], wo_a_scale[group] = per_block_cast_to_fp8(
                    wo_a[group], use_ue8m0=True
                )
            wo_a_scale = deep_gemm.transform_sf_into_required_layout(
                wo_a_scale,
                O_LORA_RANK,
                wo_a.shape[2],
                (1, 128, 128),
                num_groups=N_GROUPS,
                is_sfa=False,
            )
            wo_b_fp8, wo_b_scale = per_block_cast_to_fp8(
                wo_b, use_ue8m0=True
            )
            wo_b_scale = deep_gemm.transform_sf_into_required_layout(
                wo_b_scale,
                HIDDEN_DIM,
                O_INTERMEDIATE_DIM,
                (1, 128, 128),
                is_sfa=False,
            )
    return OProjWeights(
        wo_a_fp8, wo_a_scale, wo_b_fp8, wo_b_scale
    )
