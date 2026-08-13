"""Official DeepSeek-V4 O projection for the B300 decode pipeline.

Port of mega_csa python/mega_csa/o_proj.py (Flash_DeepSeek_V4_Pro). The local
version supports both the original full-rank geometry and the exact TP2 shard:
64 heads map to 8 complete O groups, and the TP2 ProjB publishes BF16 partials
to symmetric memory. Package-internal imports are inlined so this remains
standalone in the test tree.

The ownership boundary matches vLLM's DeepSeek-V4 implementation:
raw FlashMLA BF16 output -> inverse RoPE + FP8 quant -> DeepGEMM wo_a ->
FP8 quant -> DeepGEMM wo_b -> mHC post.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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
TP2_HEADS = HEADS // 2
HEAD_DIM = 512           # head_dim
ROPE_DIM = 64            # qk_rope_head_dim
N_GROUPS = 16            # o_groups
TP2_GROUPS = N_GROUPS // 2
O_LORA_RANK = 1024       # o_lora_rank
HIDDEN_DIM = 7168        # hidden_size
QUANT_GROUP_SIZE = 128
FP8_MAX = 448.0
VALID_GEOMETRIES = ((HEADS, N_GROUPS), (TP2_HEADS, TP2_GROUPS))


@triton.jit
def _ceil_ue8m0_scale(amax, fp8_max: tl.constexpr):
    bits = (amax / fp8_max).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
    exponent = tl.maximum(1, tl.minimum(254, exponent))
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    return scale, exponent


@dataclass(frozen=True)
class OProjWeights:
    """Prequantized DeepGEMM weights; scale tensors are already TMA layout."""

    wo_a: torch.Tensor       # FP8 [groups,1024,heads_per_group*512]
    wo_a_scale: torch.Tensor # I32 [groups,1024,heads_per_group], MN-major
    wo_b: torch.Tensor       # FP8 [7168,groups*1024]
    wo_b_scale: torch.Tensor # I32 [7168,groups*2], MN-major


@dataclass(frozen=True)
class OProjWorkspace:
    o_fp8: torch.Tensor       # FP8 [M,groups,4096]
    o_scale: torch.Tensor     # I32 [M,groups,8], MN-major within each group
    z: torch.Tensor           # BF16 [M,groups,1024]
    z_fp8: torch.Tensor       # FP8 [M,groups*1024]
    z_scale: torch.Tensor     # I32 [M,groups*2], MN-major
    projected: torch.Tensor   # BF16 full output or local TP reference [M,7168]
    mhc_output: torch.Tensor  # BF16 [M,4,7168]
    quant_module: Any
    tp2_comm: Any | None = None


class TP2Comm:
    """Graph-stable symmetric ProjB output and completion state for TP2."""

    def __init__(self, module: Any, group: Any, device: torch.device, rank: int):
        import torch.distributed._symmetric_memory as symm_mem

        _require(rank in (0, 1), "TP2 rank must be 0 or 1")
        symm_mem.set_signal_pad_size(4096)
        self.module = module
        self.partials = symm_mem.empty(
            (2, 128, HIDDEN_DIM), dtype=torch.bfloat16, device=device
        )
        self.handle = symm_mem.rendezvous(self.partials, group=group)
        self.grid_done = torch.zeros(1, device=device, dtype=torch.int64)
        self.generation = torch.zeros(1, device=device, dtype=torch.int32)
        self.local_second_output = torch.empty(
            (128, HIDDEN_DIM), device=device, dtype=torch.bfloat16
        )
        self.select_grid_done = torch.zeros(
            1, device=device, dtype=torch.int64
        )
        self.select_generation = torch.zeros(
            1, device=device, dtype=torch.int32
        )
        self.benchmark_mode = torch.ones(
            1, device=device, dtype=torch.int32
        )
        self.benchmark_generation = torch.zeros(
            1, device=device, dtype=torch.int32
        )
        self.rank = rank
        group.barrier()
        torch.cuda.synchronize(device)

    @property
    def pointers(self) -> list[int]:
        return [int(value) for value in self.handle.buffer_ptrs]

    @property
    def signal_pad_pointers(self) -> list[int]:
        return [int(value) for value in self.handle.signal_pad_ptrs]


def load_tp2_comm_module() -> Any:
    """Build the production symmetric ProjB + reduce/mHC-post extension."""

    import deep_gemm
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    dg_include = Path(deep_gemm.__file__).resolve().parent / "include"
    _require((dg_include / "deep_gemm/scheduler/gemm.cuh").is_file(),
             f"DeepGEMM headers are missing from {dg_include}")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    host_cxx = Path("/opt/rh/gcc-toolset-12/root/usr/bin/g++")
    host_compiler_flag = ([f"-ccbin={host_cxx}"]
                          if host_cxx.is_file() else [])
    return load(
        name="wuda_o_proj_b_tp2_comm",
        sources=[str(root / "kernels/o_proj_b_tp2_symm.cu")],
        extra_include_paths=[str(root / "include"), str(dg_include)],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-lineinfo",
            "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DCUTE_ARCH_TCGEN05_MMA_ENABLED=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
        ] + host_compiler_flag,
        extra_ldflags=["-lcuda"],
        verbose=False,
    )


def load_mla_o_quant_module() -> Any:
    """Build the single-launch MLA inverse-RoPE and FP8 producer."""

    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    host_cxx = Path("/opt/rh/gcc-toolset-12/root/usr/bin/g++")
    host_compiler_flag = ([f"-ccbin={host_cxx}"]
                          if host_cxx.is_file() else [])
    return load(
        name="wuda_mla_o_inv_rope_quant",
        sources=[str(root / "kernels/mla_o_inv_rope_quant.cu")],
        extra_include_paths=[str(root / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "-lineinfo",
            f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
        ] + host_compiler_flag,
        verbose=False,
    )


def load_mhc_post_module() -> Any:
    """Build the single-rank mHC post reference from the megakernel source."""

    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return load(
        name="e2e_mhc_post",
        sources=[str(root / "kernels/mhc_post.cu")],
        extra_include_paths=[str(root / "include")],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            f"-gencode=arch=compute_{sm}a,code=sm_{sm}a",
        ],
        verbose=False,
    )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _geometry(heads: int, groups: int) -> tuple[int, int]:
    _require((heads, groups) in VALID_GEOMETRIES,
             f"O projection geometry must be one of {VALID_GEOMETRIES}")
    heads_per_group = heads // groups
    _require(heads_per_group == HEADS // N_GROUPS,
             "TP must shard complete O groups")
    return heads_per_group, groups * O_LORA_RANK


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
    TRIGGER_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    packed_block = tl.program_id(1).to(tl.int64)
    if TRIGGER_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if token >= num_tokens:
        tl.store(scale_ptr + token + packed_block * scale_stride_k, 0)
        return

    offsets = packed_block * 512 + tl.arange(0, 512)
    x = tl.load(x_ptr + token * x_stride_token + offsets).to(tl.float32)
    absolute = tl.reshape(tl.abs(x), (4, QUANT_GROUP_SIZE_T))
    amax = tl.maximum(tl.max(absolute, axis=1), 1e-4)
    scale, exponent = _ceil_ue8m0_scale(amax, FP8_MAX_T)
    expanded = tl.reshape(
        tl.broadcast_to(tl.reshape(scale, (4, 1)), (4, QUANT_GROUP_SIZE_T)),
        (512,),
    )
    quantized = tl.clamp(x / expanded, -FP8_MAX_T, FP8_MAX_T).to(tl.float8e4nv)
    tl.store(fp8_ptr + token * fp8_stride_token + offsets, quantized)
    packed = tl.sum(exponent << (tl.arange(0, 4) * 8))
    tl.store(scale_ptr + token + packed_block * scale_stride_k, packed)


def _check_weights(
    weights: OProjWeights, num_groups: int, heads_per_group: int
) -> None:
    d = heads_per_group * HEAD_DIM
    intermediate_dim = num_groups * O_LORA_RANK
    _require(weights.wo_a.shape == (num_groups, O_LORA_RANK, d) and
             weights.wo_a.dtype == torch.float8_e4m3fn and
             weights.wo_a.is_cuda and weights.wo_a.is_contiguous(),
             f"wo_a must be FP8 [{num_groups},{O_LORA_RANK},{d}]")
    _require(weights.wo_a_scale.shape ==
             (num_groups, O_LORA_RANK, d // 512) and
             weights.wo_a_scale.dtype == torch.int32 and weights.wo_a_scale.is_cuda and
             weights.wo_a_scale.stride() ==
             (O_LORA_RANK * (d // 512), 1, O_LORA_RANK),
             "wo_a_scale has the wrong shape, dtype, or TMA layout")
    _require(weights.wo_b.shape == (HIDDEN_DIM, intermediate_dim) and
             weights.wo_b.dtype == torch.float8_e4m3fn and
             weights.wo_b.is_cuda and weights.wo_b.is_contiguous(),
             f"wo_b must be FP8 [{HIDDEN_DIM},{intermediate_dim}]")
    _require(weights.wo_b_scale.shape ==
             (HIDDEN_DIM, intermediate_dim // 512) and
             weights.wo_b_scale.dtype == torch.int32 and weights.wo_b_scale.is_cuda and
             weights.wo_b_scale.stride() == (1, HIDDEN_DIM),
             "wo_b_scale has the wrong shape, dtype, or TMA layout")


@torch.inference_mode()
def prepare_o_proj_workspace(
    m: int,
    device: torch.device | str,
    *,
    heads: int = HEADS,
    groups: int = N_GROUPS,
    quant_module: Any,
    tp2_comm: TP2Comm | None = None,
) -> OProjWorkspace:
    _require(1 <= m <= 128, "O projection requires M in [1,128]")
    _require(quant_module is not None, "MLA O-projection quant module is required")
    heads_per_group, intermediate_dim = _geometry(heads, groups)
    _require((tp2_comm is None) == (groups != TP2_GROUPS),
             "TP2 geometry requires a symmetric communication workspace")
    d = heads_per_group * HEAD_DIM
    aligned_m = _align(m, 4)

    o_fp8_base = torch.empty(
        (groups, m, d), device=device, dtype=torch.float8_e4m3fn
    )
    o_scale_base = torch.empty(
        groups * heads_per_group * aligned_m,
        device=device,
        dtype=torch.int32,
    )
    o_scale_base = o_scale_base.as_strided(
        (groups, m, heads_per_group),
        (heads_per_group * aligned_m, 1, aligned_m),
    )
    z = torch.empty(
        (m, groups, O_LORA_RANK), device=device, dtype=torch.bfloat16
    )
    z_fp8 = torch.empty(
        (m, intermediate_dim), device=device, dtype=torch.float8_e4m3fn
    )
    z_scale_blocks = intermediate_dim // 512
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
        projected=torch.empty(
            (m, HIDDEN_DIM), device=device,
            dtype=torch.bfloat16,
        ),
        mhc_output=torch.empty(
            (m, 4, HIDDEN_DIM), device=device, dtype=torch.bfloat16
        ),
        quant_module=quant_module,
        tp2_comm=tp2_comm,
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
    force_pdl: bool = False,
    run_mhc_post: bool = True,
    tp2_benchmark_select: bool = False,
) -> torch.Tensor:
    """Run allocation-free O projection and mHC post on the current stream.

    Full-rank run_mhc_post=False stops after wo_b and returns
    workspace.projected so a benchmark can time mHC post separately. TP2 uses
    the symmetric ProjB and fused reduce+mHC post as one communication stage.
    tp2_benchmark_select is the paired-measurement path: one captured graph
    selects the second ProjB destination through a device flag.
    """

    import deep_gemm

    if mla_out.dim() == 4:
        _require(mla_out.shape[1] == 1, "4D mla_out must have singleton scope dim")
        mla_out = mla_out[:, 0]
    _require(mla_out.dim() == 3, "mla_out must be [M,H,512] or [M,1,H,512]")
    m, heads, head_dim = mla_out.shape
    _require(mla_out.dtype == torch.bfloat16 and mla_out.is_cuda and
             mla_out.is_contiguous(), "mla_out must be contiguous CUDA BF16")
    _require(1 <= m <= 128 and head_dim == HEAD_DIM,
             f"mla_out must have M in [1,128] and head_dim={HEAD_DIM}")
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
    num_groups = weights.wo_a.size(0)
    heads_per_group, intermediate_dim = _geometry(heads, num_groups)
    _check_weights(weights, num_groups, heads_per_group)
    aligned_m = _align(m, 4)
    d = heads_per_group * HEAD_DIM
    _require(workspace.o_fp8.shape == (m, num_groups, d) and
             workspace.o_fp8.dtype == torch.float8_e4m3fn and
             workspace.o_fp8.stride() == (d, m * d, 1),
             "workspace.o_fp8 has the wrong shape, dtype, or grouped layout")
    _require(workspace.o_scale.shape == (m, num_groups, heads_per_group) and
             workspace.o_scale.dtype == torch.int32 and
             workspace.o_scale.stride() ==
             (1, heads_per_group * aligned_m, aligned_m),
             "workspace.o_scale has the wrong shape, dtype, or MN-major layout")
    _require(workspace.z.shape == (m, num_groups, O_LORA_RANK) and
             workspace.z.dtype == torch.bfloat16 and workspace.z.is_contiguous(),
             f"workspace.z must be contiguous BF16 [M,{num_groups},1024]")
    _require(workspace.z_fp8.shape == (m, intermediate_dim) and
             workspace.z_fp8.dtype == torch.float8_e4m3fn and
             workspace.z_fp8.is_contiguous(),
             f"workspace.z_fp8 must be contiguous FP8 [M,{intermediate_dim}]")
    _require(workspace.z_scale.shape == (m, intermediate_dim // 512) and
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
    tp2_comm = workspace.tp2_comm
    _require((tp2_comm is None) == (num_groups != TP2_GROUPS),
             "TP2 geometry requires a symmetric communication workspace")
    _require(not tp2_benchmark_select or num_groups == TP2_GROUPS,
             "tp2_benchmark_select requires TP2 geometry")
    _require(run_mhc_post or num_groups == N_GROUPS,
             "the symmetric TP2 producer and fused post must run together")
    _require(not (run_mhc_post and tp2_comm is None)
             or mhc_post_module is not None,
             "the ordinary mHC post module is required")
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

    # This producer/consumer pair did not show stable overlap in E2E traces.
    # Enable it only for the force-all ablation mode.
    pdl_quant_to_wo_b = use_pdl and force_pdl

    # Serialize the process-wide DeepGEMM PDL setting and all affected enqueues.
    assert device.index is not None
    with torch.cuda.device(device):
        with deepgemm_pdl_enqueue_scope(
            deep_gemm, False, device.index
        ):
            workspace.quant_module.inv_rope_quant(
                mla_out,
                positions,
                cos_sin_cache,
                workspace.o_fp8,
                workspace.o_scale,
            )
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (workspace.o_fp8, workspace.o_scale),
                (weights.wo_a, weights.wo_a_scale),
                workspace.z,
                recipe=(1, 1, 128),
            )

        with deepgemm_pdl_enqueue_scope(
            deep_gemm, pdl_quant_to_wo_b, device.index
        ):
            _quant_o_lora_kernel[(aligned_m, intermediate_dim // 512)](
                workspace.z,
                workspace.z_fp8,
                workspace.z_scale,
                m,
                x_stride_token=workspace.z.stride(0),
                fp8_stride_token=workspace.z_fp8.stride(0),
                scale_stride_k=workspace.z_scale.stride(1),
                QUANT_GROUP_SIZE_T=QUANT_GROUP_SIZE,
                FP8_MAX_T=FP8_MAX,
                TRIGGER_PDL=pdl_quant_to_wo_b,
                launch_pdl=False,
                num_warps=4,
                num_stages=1,
            )
            if tp2_comm is not None:
                if tp2_benchmark_select:
                    tp2_comm.module.o_proj_b_select(
                        workspace.z_fp8,
                        workspace.z_scale,
                        weights.wo_b,
                        weights.wo_b_scale,
                        tp2_comm.partials,
                        tp2_comm.local_second_output,
                        tp2_comm.select_grid_done,
                        tp2_comm.select_generation,
                        tp2_comm.benchmark_mode,
                        tp2_comm.pointers,
                        tp2_comm.signal_pad_pointers,
                        tp2_comm.rank,
                    )
                    tp2_comm.module.mhc_post_select(
                        tp2_comm.partials,
                        tp2_comm.local_second_output,
                        tp2_comm.benchmark_mode,
                        residual,
                        post,
                        comb,
                        workspace.mhc_output,
                        tp2_comm.rank,
                    )
                else:
                    tp2_comm.module.o_proj_b(
                        workspace.z_fp8,
                        workspace.z_scale,
                        weights.wo_b,
                        weights.wo_b_scale,
                        tp2_comm.partials,
                        tp2_comm.grid_done,
                        tp2_comm.generation,
                        tp2_comm.pointers,
                        tp2_comm.signal_pad_pointers,
                        tp2_comm.rank,
                    )
                    tp2_comm.module.mhc_post(
                        tp2_comm.partials,
                        residual,
                        post,
                        comb,
                        workspace.mhc_output,
                    )
            else:
                deep_gemm.fp8_gemm_nt(
                    (workspace.z_fp8, workspace.z_scale),
                    (weights.wo_b, weights.wo_b_scale),
                    workspace.projected,
                    recipe=(1, 1, 128),
                )
            if run_mhc_post and tp2_comm is None:
                mhc_post_module.mhc_post_out(
                    workspace.projected,
                    residual,
                    post,
                    comb,
                    workspace.mhc_output,
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

    _require(wo_a.dim() == 3 and wo_a.shape[1] == O_LORA_RANK
             and wo_a.shape[2] % HEAD_DIM == 0,
             "wo_a must be [groups,1024,heads_per_group*512]")
    num_groups = wo_a.shape[0]
    heads = num_groups * (wo_a.shape[2] // HEAD_DIM)
    _, intermediate_dim = _geometry(heads, num_groups)
    expected_wo_a_shape = (num_groups, O_LORA_RANK, wo_a.shape[2])
    _require(wo_a.shape == expected_wo_a_shape and
             wo_a.dtype == torch.bfloat16 and wo_a.is_cuda and wo_a.is_contiguous(),
             f"wo_a must be contiguous CUDA BF16 {expected_wo_a_shape}")
    _require(wo_b.shape == (HIDDEN_DIM, intermediate_dim) and
             wo_b.dtype == torch.bfloat16 and wo_b.is_cuda and wo_b.is_contiguous(),
             f"wo_b must be contiguous CUDA BF16 [{HIDDEN_DIM},{intermediate_dim}]")
    _require(wo_a.device == wo_b.device, "wo_a and wo_b must be on the same device")
    device = wo_a.device
    assert device.index is not None
    with torch.cuda.device(device):
        with deepgemm_device_scope(device.index):
            wo_a_fp8 = torch.empty_like(wo_a, dtype=torch.float8_e4m3fn)
            wo_a_scale = torch.empty(
                (num_groups, O_LORA_RANK // 128, wo_a.shape[2] // 128),
                device=device,
                dtype=torch.float32,
            )
            for group in range(num_groups):
                wo_a_fp8[group], wo_a_scale[group] = per_block_cast_to_fp8(
                    wo_a[group], use_ue8m0=True
                )
            wo_a_scale = deep_gemm.transform_sf_into_required_layout(
                wo_a_scale,
                O_LORA_RANK,
                wo_a.shape[2],
                (1, 128, 128),
                num_groups=num_groups,
                is_sfa=False,
            )
            wo_b_fp8, wo_b_scale = per_block_cast_to_fp8(
                wo_b, use_ue8m0=True
            )
            wo_b_scale = deep_gemm.transform_sf_into_required_layout(
                wo_b_scale,
                HIDDEN_DIM,
                intermediate_dim,
                (1, 128, 128),
                is_sfa=False,
            )
    return OProjWeights(
        wo_a_fp8, wo_a_scale, wo_b_fp8, wo_b_scale
    )
