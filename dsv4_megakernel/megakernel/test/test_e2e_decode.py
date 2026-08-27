"""
test_e2e_decode.py -- END-TO-END DSV4 decode chain over the design-image KV
cache management (test/kv_cache_manager.py):

  hc_fused(nopc + fused attn_norm) -> front_mixed_swapAB(+hc tail) ->
  qnorm_quant(PDL) + wq_b(ALL fusions: idxpost + indexer compressor + winkv)
  -> query RMSNorm+RoPE (PDL side branch) + Wuda fused FP8 paged MQA + MAIN
  compressor -> topk_v2(512, page transform) -> official flash_mla(SWA pool +
  compressed pool, native 512-d query, per-head attn_sink) ->
  o_proj(inv-RoPE + fp8 -> wo_a -> fp8 -> wo_b) -> mhc_post -> [B,4,7168].

The tail follows the official mega_csa post-attention boundary. o_proj_csa.py
adds TP2 support around the two Triton quant kernels and DeepGEMM wo_a. TP2
ProjB publishes both BF16 partials through symmetric memory, then fuses their
pairwise reduction into mHC post. residual is this layer's input, while
post/comb come from the front hc tail.

Multi-step decode simulation (the cache/state semantics ONLY show up across
steps): B requests advance pos together; mid-run one request is freed and its
slot reused (KV=0 / score=-inf re-init gate). P1 discipline: kernels unchanged,
KVCacheManager supplies slot indirection + paged-pool scatter shims.

Stage gates per step:
  A front y (fp8/bf16 segment calc_diff)          D topk vs torch.topk+transform
  B wq_b x_fp8 quant chain (byte match)           E flashMLA vs torch attention
  C mqa logits vs torch ref over DEQUANT pools        over the DEQUANT pools
  H checks symmetric TP2 o_proj+reduce+mhc_post against an NCCL reference
Global gate: the whole simulation runs TWICE from the same seed -> all caches,
states and final outputs bitwise identical (run-to-run determinism).

flash_mla is optional (import-guarded): without it stages A-D still run, and
the o_proj/mhc_post tail (which consumes the MLA output) is skipped with it.
The default is the RTP-compatible FP8 Indexer; --indexer-fp4 is the explicit
legacy A/B path. The default geometry is the full-rank e2e chain. --tpdp uses
main-Q TP2 with replicated index-Q, a symmetric TP-head gather into request-DP2
MQA/TopK/FlashMLA, and the global-B TP2 O-proj shard (64 heads, 8 groups).
"""
import argparse
import json, os, sys, math, warnings, torch
import torch.distributed as dist
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_cache_manager import (KVCacheManager, PAGE, WIN, RATIO,
                              D_NOPE, D_ROPE, TILE, NTILES)

import test_hc_fused_tc as t_hc
import test_front_mixed_gemm as t_fm
import test_front_mixed_gemm_csa_swapab as t_fm_swap
import test_wq_b_fp8_gemm as t_wq
import o_proj_csa

def _import_flash_mla():
    """Import the official FlashMLA checkout, never the private fused-query
    fork or an older installed wheel."""
    import importlib
    here = os.path.dirname(os.path.abspath(__file__))
    default_src = os.path.abspath(os.path.join(
        here, "..", "..", "..", "..", "FlashMLA"))
    src = os.path.abspath(os.environ.get("FLASH_MLA_DIR", default_src))
    if not os.path.isdir(os.path.join(src, "flash_mla")):
        raise ModuleNotFoundError(f"official FlashMLA checkout not found at {src}")
    for key in [k for k in sys.modules if k.startswith("flash_mla")]:
        del sys.modules[key]
    sys.path.insert(0, src)
    m = importlib.import_module("flash_mla")
    if not os.path.realpath(m.__file__).startswith(os.path.realpath(src) + os.sep):
        raise ImportError(f"expected FlashMLA from {src}, got {m.__file__}")
    print(f"[flash_mla] official source tree: {src}")
    return m


try:
    flash_mla = _import_flash_mla()
    HAS_FLASH_MLA = True
    import inspect
    FMLA_ATTN_SINK = "attn_sink" in inspect.signature(
        flash_mla.flash_mla_with_kvcache).parameters
    if not FMLA_ATTN_SINK:
        raise ImportError("official FlashMLA build lacks attn_sink support")
except Exception as e:
    print(f"[warn] flash_mla unavailable ({e}) -- stage E skipped")
    HAS_FLASH_MLA = False

DEV = "cuda"
HC, DIM = 4, 7168
N_FRONT = 4672
TOPK = 1024                    # DSV4 Pro index_topk (origin/config.json;
                               # matches the vLLM baseline's hard gate)
SWA_TOPK = 128
Q_DIM = 512
Q_HEADS = 128                 # full post-attention path; unchanged by --tpdp
WQ_HEADS = Q_HEADS            # local WQ_B main heads (64 in --tpdp mode)
IDX_HEADS, IDX_D = 64, 128
MAX_POS = 4096
EPS = 1e-6
O_GROUPS, O_LORA = 16, 1024        # o_proj: 16 groups x rank 1024
O_INTER = O_GROUPS * O_LORA        # 16384
TPDP_MODE = False
MLA_DP_MODE = False
TP_RANK = 0
TP_WORLD_SIZE = 1
TP2_COMM_MODULE = None
MLA_O_QUANT_MODULE = None
TP_SYNC_GROUP = None
QUERY_READY_OFFSET = 32
INDEXER_FP8 = True
PDL_MODE = "on"
Q_RMS_MAX_BLOCKS = 1024
Q_RMS_ABLATION = False
FUSE_QUERY_RMS = True

def configure_geometry(tpdp=False):
    global WQ_HEADS, TPDP_MODE, MLA_DP_MODE
    TPDP_MODE = tpdp
    MLA_DP_MODE = tpdp
    t_wq.configure_geometry(tpdp)
    WQ_HEADS = t_wq.N_TOTAL // Q_DIM


def configure_pdl(mode):
    global PDL_MODE
    if mode not in ("on", "all", "none"):
        raise ValueError(f"unsupported PDL mode: {mode}")
    PDL_MODE = mode


def pdl_enabled():
    return PDL_MODE != "none"


def pdl_forced():
    return PDL_MODE == "all"


def fused_query_rms():
    return FUSE_QUERY_RMS


def inplace_query_rms():
    # Full rank has a one-to-one [B,128,512] input/output mapping. TPDP expands
    # 64 local WQ heads to 128 DP attention heads and therefore stays out-of-place.
    return fused_query_rms() and not TPDP_MODE


def post_attn_enabled():
    return HAS_FLASH_MLA


def mhc_post_enabled():
    return HAS_FLASH_MLA


def attention_heads():
    return Q_HEADS if MLA_DP_MODE else WQ_HEADS


def oproj_heads():
    return WQ_HEADS


def local_o_groups():
    return O_GROUPS // 2 if TPDP_MODE else O_GROUPS


def local_batch(B):
    if not TPDP_MODE:
        return B
    split = (B + 1) // 2
    return split if TP_RANK == 0 else B - split


def owned_slice(B):
    if not TPDP_MODE:
        return slice(0, B)
    split = (B + 1) // 2
    return slice(0, split) if TP_RANK == 0 else slice(split, B)


def mla_batch(B):
    return local_batch(B) if MLA_DP_MODE else B


def init_tpdp_distributed():
    """Bind one CUDA device per TP rank and create the NCCL world group."""
    global TP_RANK, TP_WORLD_SIZE, TP_SYNC_GROUP
    if not TPDP_MODE:
        return
    TP_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    if TP_WORLD_SIZE != 2:
        raise RuntimeError(
            "--tpdp requires exactly two ranks; launch with torchrun "
            "--standalone --nproc-per-node=2"
        )
    TP_RANK = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    TP_SYNC_GROUP = dist.new_group(ranks=[0, 1], backend="gloo")


def tp_host_barrier():
    """Align TP ranks without adding a collective to the measured GPU path."""
    if TPDP_MODE:
        dist.barrier(group=TP_SYNC_GROUP)


def tp_device_barrier(tp2_comm):
    """Queue an excluded GPU rendezvous before a cross-rank timed replay."""
    if TPDP_MODE:
        tp2_comm.module.benchmark_barrier(
            tp2_comm.benchmark_generation,
            tp2_comm.signal_pad_pointers,
            tp2_comm.rank,
        )


def tp2_reference_reduce(local_partial):
    """Match NCCL's BF16 pairwise sum for the correctness-only reference."""
    assert TPDP_MODE and dist.is_initialized()
    reduced = local_partial.to(torch.bfloat16)
    dist.all_reduce(reduced)
    return reduced


def load_query_rms_rope():
    """Build the allocation-free post-Q_B RMSNorm+RoPE side branch."""
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return load(
        name="e2e_query_rms_rope",
        sources=[os.path.join(proj, "kernels", "rmsnorm_prod.cu")],
        extra_include_paths=[os.path.join(proj, "include")],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math",
                           f"-gencode=arch=compute_{sm}a,code=sm_{sm}a"],
        verbose=False,
    )


_OPROJ_WS = {}
_QUERY_TP2_WS = {}


class QueryTP2Comm:
    """Graph-stable TP-head gather workspace consumed directly by FlashMLA."""

    def __init__(self, B):
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.set_signal_pad_size(4096)
        max_local = (B + 1) // 2
        self.buffer = symm_mem.empty(
            (max_local, 1, Q_HEADS, Q_DIM), dtype=torch.bfloat16,
            device=torch.device("cuda", torch.cuda.current_device()))
        self.handle = symm_mem.rendezvous(
            self.buffer, group=dist.group.WORLD)
        self.generation = torch.zeros(
            1, device=DEV, dtype=torch.int32)
        dist.barrier()
        torch.cuda.synchronize()

    @property
    def pointers(self):
        return [int(value) for value in self.handle.buffer_ptrs]

    @property
    def signal_pad_pointers(self):
        return [int(value) for value in self.handle.signal_pad_ptrs]

    def output(self, B):
        return self.buffer[:local_batch(B)]


def query_tp2_ws(B):
    if B not in _QUERY_TP2_WS:
        _QUERY_TP2_WS[B] = QueryTP2Comm(B)
    return _QUERY_TP2_WS[B]


def oproj_ws(B):
    """Cached O-projection workspace (the run path is allocation-free, and a
    stable workspace is what lets CUDA-graph replay keep its addresses)."""
    key = (B, oproj_heads(), local_o_groups())
    if key not in _OPROJ_WS:
        tp2_comm = None
        if TPDP_MODE:
            assert TP2_COMM_MODULE is not None and dist.is_initialized()
            tp2_comm = o_proj_csa.TP2Comm(
                TP2_COMM_MODULE, dist.group.WORLD,
                torch.device("cuda", torch.cuda.current_device()), TP_RANK, B)
        _OPROJ_WS[key] = o_proj_csa.prepare_o_proj_workspace(
            B, DEV, heads=oproj_heads(), groups=local_o_groups(),
            quant_module=MLA_O_QUANT_MODULE,
            wo_a_module=TP2_COMM_MODULE,
            tp2_comm=tp2_comm)
    return _OPROJ_WS[key]


# ==================== weights (one-time, layer constants) ====================
def make_weights(seed=7):
    torch.manual_seed(seed)
    w = {}
    w["hc_fn"] = torch.randn(24, HC * DIM, device=DEV) * 0.01
    w["hc_base"] = torch.randn(24, device=DEV) * 0.1
    w["hc_scale"] = torch.rand(3, device=DEV) + 0.5
    # attn_norm gamma: BF16 = the CHECKPOINT dtype (model.py L188); the
    # kernel widens to fp32 in-flight (lossless, official compute chain).
    w["attn_norm"] = (torch.rand(DIM, device=DEV) + 0.5).bfloat16()
    fw = torch.randn(N_FRONT, DIM, device=DEV, dtype=torch.bfloat16) * 0.02
    w["front_fp8"], w["front_sf"] = t_fm.quant_weight_fp8(fw[:2048])
    w["front_bf16"] = fw[2048:].contiguous()
    w["q_norm"] = torch.rand(1536, device=DEV) + 0.5
    wq_idx = torch.randn(t_wq.N_IDX, 1536, device=DEV) * 0.03
    if TPDP_MODE:
        # Different main-Q shards make the TP-head gather correctness-visible;
        # the indexer projection and all following weights remain replicated.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed + 2000 + TP_RANK)
            wq_main = torch.randn(t_wq.N_TOTAL, 1536, device=DEV) * 0.03
    else:
        wq_main = torch.randn(t_wq.N_TOTAL, 1536, device=DEV) * 0.03
    wq = torch.cat((wq_idx, wq_main), dim=0)
    w["wq_fp8"] = wq.to(torch.float8_e4m3fn)
    w["wq_sf"] = t_wq.make_weight_sf_ones(DEV)
    w["idx_ape"] = torch.randn(4, 256, device=DEV)
    # main-compressor ape [RATIO, 2*512] (model.py Compressor L294/L332:
    # score_state = wgate(x) + ape[pos%ratio] -- folded AT PUBLISH, exactly
    # like the idx chain does in-kernel)
    w["main_ape"] = torch.randn(4, 1024, device=DEV)
    w["idx_norm"] = torch.rand(128, device=DEV) + 0.5
    w["win_norm"] = torch.rand(512, device=DEV) + 0.5
    w["comp_norm"] = torch.rand(512, device=DEV) + 0.5
    ang = torch.rand(MAX_POS, 32, device=DEV) * 6.28
    w["cos"], w["sin"] = ang.cos().contiguous(), ang.sin().contiguous()
    if not post_attn_enabled():
        return w

    # attn_sink: per-head LEARNED logit of a virtual key whose value is 0
    # (model.py L456 nn.Parameter[n_local_heads] fp32; convert.py shards it on
    # dim 0). It enters the softmax DENOMINATOR only (kernel.py L346), so the
    # real weights sum to <1 and a head can attend to nothing -- which matters
    # for top-k sparse attention, where otherwise every head must spend its
    # full mass on whatever the indexer picked.
    w["attn_sink"] = torch.randn(attention_heads(), device=DEV,
                                  dtype=torch.float32)
    # o_proj (official DSV4 two-stage attention output): full uses 16 groups;
    # TP2 uses 8 complete groups and a row-parallel wo_b half. The bf16
    # originals stay resident for the stage-H torch reference; the fp8 +
    # TMA-layout scales are built ONCE here (never in the hot region).
    # o_proj_csa imports deep_gemm PLAINLY, so the shared resolver must have
    # put it on sys.path first; its two extra APIs are checked up front so a
    # too-old build fails here instead of mid-chain.
    dg = get_dg()
    for api in ("fp8_einsum", "fp8_gemm_nt", "transform_sf_into_required_layout"):
        assert hasattr(dg, api), f"deep_gemm build lacks {api} (o_proj needs it)"
    groups = local_o_groups()
    if TPDP_MODE:
        # Distinct TP shards make the E2E gate prove that peer data is used.
        torch.manual_seed(seed + 1000 + TP_RANK)
    w["wo_a"] = torch.randn(groups, O_LORA,
                            (oproj_heads() // groups) * Q_DIM,
                            device=DEV, dtype=torch.bfloat16) * 0.01
    w["wo_b"] = torch.randn(DIM, groups * O_LORA, device=DEV,
                            dtype=torch.bfloat16) * 0.01
    w["o_proj"] = o_proj_csa.quantize_o_proj_weights(w["wo_a"], w["wo_b"])
    # ONE contiguous cos||sin table [max_pos,64]: what the inverse-RoPE
    # kernel indexes by position (same angles the forward rope used).
    w["cos_sin"] = torch.cat((w["cos"], w["sin"]), dim=-1).contiguous()
    return w


# ==================== torch references ====================
def deq_fp4(codes_u8, table=None):
    """packed e2m1 u8 [.., n/2] -> fp32 [.., n] (test_wq_b's table)."""
    lo, hi = (codes_u8 & 0x0F).to(torch.int8), (codes_u8 >> 4).to(torch.int8)
    inter = torch.stack([lo, hi], dim=-1).flatten(-2)
    return t_wq._dequantize_from_fp4_e2m1(inter)


def deq_idx_pool(mgr, slots, pos):
    """DEQUANT indexer pool rows -> per-request [ncmp,128] fp32 list."""
    out = []
    for i, s in enumerate(slots):
        ncmp = (int(pos[i]) + 1) // RATIO
        epb = mgr.entries_per_block["idx"]
        rows = []
        for ct in range(ncmp):
            page = mgr.reqs[s]["idx"][ct // epb]
            off = ct % epb
            if mgr.indexer_fp8:
                q8 = mgr.idx_pool[page, :epb * IDX_D].view(epb, IDX_D)[off]
                scale = mgr.idx_pool[page, epb * IDX_D + off * 4:
                                     epb * IDX_D + off * 4 + 4] \
                    .view(torch.float32)[0]
                v = q8.view(torch.float8_e4m3fn).float() * scale
            else:
                fp4 = mgr.idx_pool[page, : epb * 64].view(epb, 64)[off]
                sf = mgr.idx_pool[page, epb * 64 + off * 4:
                                  epb * 64 + off * 4 + 4]
                scale = torch.pow(2.0, sf[:4].float() - 127.0)
                v = deq_fp4(fp4).view(4, 32) * scale.view(4, 1)
            rows.append(v.view(128))
        out.append(torch.stack(rows) if rows else
                   torch.zeros(0, 128, device=DEV))
    return out


def deq_model1_row(pool, page, off, entries_per_block):
    epb = entries_per_block
    body = pool[page, : epb * (D_NOPE + 2 * D_ROPE)] \
        .view(epb, D_NOPE + 2 * D_ROPE)[off]
    sf = pool[page, epb * (D_NOPE + 2 * D_ROPE):
              epb * (D_NOPE + 2 * D_ROPE) + epb * 8].view(epb, 8)[off]
    nope = body[:D_NOPE].view(torch.float8_e4m3fn).float().view(NTILES, TILE)
    scale = sf[:NTILES].view(torch.float8_e8m0fnu).float()
    rope = body[D_NOPE:].view(torch.bfloat16).float()
    return torch.cat([(nope * scale.view(NTILES, 1)).view(D_NOPE), rope])


def ref_mqa_logits(iq, iq_aux, weights, kv_rows_list):
    """relu(<q_h, k_t>)·w_h summed over the 64 indexer heads; q dequant via
    test_wq_b's dequant_kernel_iq (iq_sf = 4 packed block-32 ue8m0 / head)."""
    B = iq.size(0)
    if iq.dtype == torch.float8_e4m3fn:
        q = iq.float()
        weights = iq_aux
    else:
        q = t_wq.dequant_kernel_iq(iq, iq_aux)[0].view(B, IDX_HEADS, IDX_D)
    outs = []
    for b in range(B):
        kv = kv_rows_list[b]                                   # [n,128]
        if kv.numel() == 0:
            outs.append(torch.zeros(0, device=DEV)); continue
        lg = torch.relu(q[b] @ kv.t())                         # [64,n]
        outs.append((lg * weights[b].unsqueeze(-1)).sum(0))
    return outs


def ref_flash_attn(qn, swa_rows, cmp_rows, scale, sink=None):
    """softmax over [swa ++ cmp] rows; v == k row (head_dim_v = 512).

    sink [128] fp32 = the official attn_sink. Since its value vector is 0, the
    exact reference is a softmax over the n+1 logits [scores || sink] with the
    sink COLUMN DROPPED from the weighted sum -- algebraically identical to
    kernel.py L346's 'add exp(sink - max) to the denominator', but numerically
    stabler (torch.softmax owns the max subtraction) and well-defined at n=0.
    FlashMLA states it a third way (scale the sinkless output by
    exp(lse)/(exp(lse)+exp(sink))); all three agree exactly."""
    k = torch.cat([swa_rows, cmp_rows])                        # [n,512]
    lg = (qn @ k.t()).float() * scale                          # [128,n]
    if sink is None:
        return torch.softmax(lg, dim=-1) @ k.float()           # [128,512]
    lg = torch.cat([lg, sink.float().view(-1, 1)], dim=-1)     # [128,n+1]
    return torch.softmax(lg, dim=-1)[:, :-1] @ k.float()       # [128,512]


def ref_o_proj_mhc(
    mla3, pos64, cos_sin, residual, post, comb, wo_a, wo_b,
    run_mhc_post=True,
):
    """fp32 torch chain for the official post-attention boundary (mega_csa
    unit-test reference): INVERSE RoPE on the last 64 dims (note the sign
    flip vs the forward rope) -> bf16 model boundary -> grouped wo_a ->
    wo_b -> mHC mix. The kernel path is fp8 twice over, so the gate is the
    official calc_diff < 2e-2."""
    B = mla3.size(0)
    heads = mla3.size(1)
    groups = wo_a.size(0)
    r = mla3.float().clone()
    rope = r[..., Q_DIM - 64:].view(B, heads, 32, 2)
    even, odd = rope.unbind(-1)
    f = cos_sin.index_select(0, pos64)
    cos, sin = f[:, :32].unsqueeze(1), f[:, 32:].unsqueeze(1)
    r[..., Q_DIM - 64:] = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin), dim=-1).flatten(-2)
    rot = r.to(torch.bfloat16).float()
    z = torch.einsum("mhr,hdr->mhd",
                     rot.view(B, groups, -1), wo_a.float())
    proj = z.flatten(1) @ wo_b.float().t()
    if not run_mhc_post:
        return proj
    return ref_mhc_post(proj, residual, post, comb)


def ref_mhc_post(attention_out, residual, post, comb):
    return (torch.einsum("mij,mih->mjh", comb, residual.float())
            + post.unsqueeze(-1) * attention_out.float().unsqueeze(1)) \
        .to(torch.bfloat16)


# ==================== one decode step ====================
def get_dg():
    """deep_gemm (hard dep for the hybrid mhc): shared resolver in
    bench_utils (env DEEP_GEMM_DIR > installed wheel > sibling checkout).
    The chain-leading mHC GEMM has no PDL producer; O-proj configures its two
    measured PDL edges independently."""
    from bench_utils import get_deep_gemm
    enabled = pdl_forced()
    dg = get_deep_gemm(pdl=enabled)
    if hasattr(dg, "set_pdl"):
        dg.set_pdl(enabled)
    assert hasattr(dg, "tf32_hc_prenorm_gemm"), \
        "deep_gemm build lacks tf32_hc_prenorm_gemm"
    return dg


def hc_n_splits(m):
    """vLLM compute_num_split(block_k=64, k=28672, grid=cdiv(m,64)) for
    deep_gemm.tf32_hc_prenorm_gemm."""
    n_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(min(n_sms // max((m + 63) // 64, 1), (HC * DIM // 64) // 4), 1)


def run_mhc_hybrid(dg, hcm, hidden, w, collapsed, pre, po, cb,
                   ws=None, sq=None, mix_out=None, xq_out=None,
                   xsf_out=None):
    """HYBRID mhc: deep_gemm tf32 split-K GEMM + OUR fused epilogue
    (identical workspace layout: mul [S,M,24] + sqrsum [S,M]).
    mix_out [M,24] exports the rms-folded mix row (front tail consumes IT
    for post/comb); xq_out/xsf_out emit front's fp8 activation + 1x128
    ue8m0 sf straight from the epilogue registers (fused act quant -- no
    separate quant kernel; vLLM pays one)."""
    B = hidden.size(0)
    S = hc_n_splits(B)
    if ws is None:
        ws = torch.empty(S, B, 24, device=DEV, dtype=torch.float32)
        sq = torch.empty(S, B, device=DEV, dtype=torch.float32)
    dg.tf32_hc_prenorm_gemm(hidden.view(B, HC * DIM), w["hc_fn"], ws, sq, S)
    hcm.hc_reduce_fuse_out(hidden, ws, sq, w["hc_base"], w["hc_scale"],
                           EPS, EPS, collapsed, pre, po, cb,
                           with_post_comb=False, attn_norm_w=w["attn_norm"],
                           attn_norm_eps=EPS, mix_out=mix_out,
                           xq_out=xq_out, xsf_out=xsf_out,
                           pdl=pdl_forced())


def decode_step(mods, w, mgr, slots, hidden, logits_buf, stats, dbg=None):
    hcm, fmm, wqm, qrm, mqm, mq8m, tkm, mpm = mods
    B = len(slots)
    st = mgr.step_begin(slots)
    pos, q_pos = st["pos"], st["q_pos"]

    # -- 1. MHC hybrid (deep_gemm tf32 GEMM + our nopc+norm epilogue);
    #       mix_out exports the rms-folded mix row for the front tail ------
    collapsed = torch.empty(B, DIM, device=DEV, dtype=torch.bfloat16)
    pre = torch.empty(B, HC, device=DEV, dtype=torch.float32)
    po = torch.empty(B, HC, device=DEV, dtype=torch.float32)
    cb = torch.empty(B, HC, HC, device=DEV, dtype=torch.float32)
    mix = torch.empty(B, 24, device=DEV, dtype=torch.float32)
    x_fp8 = torch.empty(B, DIM, device=DEV, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty(B, DIM // 128, device=DEV, dtype=torch.uint8)
    run_mhc_hybrid(get_dg(), hcm, hidden, w, collapsed, pre, po, cb,
                   mix_out=mix, xq_out=x_fp8.view(torch.uint8), xsf_out=x_sf)

    # -- 2. front_mixed (+hc tail: post/comb from the REAL mhc mix; fp8 x
    #       comes fused from the mhc epilogue). [FRONT-EMIT]: the epilogue
    #       scatters MAIN and IDX state (fp32 + ape on score halves) into
    #       the pools and emits fp32 win_y2 / w64 side buffers -- no glue
    #       op, no y writes for cols [1536,4672) --------------------------
    hc = {"hc_mix": mix, "hc_base": w["hc_base"], "hc_scale": w["hc_scale"],
          "hc_post": torch.empty(B, HC, device=DEV, dtype=torch.float32),
          "hc_comb": torch.empty(B, HC * HC, device=DEV,
                                 dtype=torch.float32)}
    win_y2 = torch.empty(B, 512, device=DEV, dtype=torch.float32)
    w64 = torch.empty(B, 64, device=DEV, dtype=torch.float32)
    y = fmm.front_mixed_gemm_csa_swapab(
        collapsed, x_fp8, x_sf, w["front_bf16"], w["front_fp8"],
        w["front_sf"], **hc, enable_tail=True,
        main_state=mgr.main_state, main_ape=w["main_ape"],
        main_state_row=st["main_state_row"], ape_phase=(pos % 4).int(),
        idx_state=mgr.idx_state, idx_state_row=st["idx_state_row"],
        idx_ape=w["idx_ape"], win_y2=win_y2, w64=w64,
        pdl=pdl_enabled())
    ref8 = (t_fm.dequant_fp8(x_fp8, x_sf, True)
            @ t_fm.dequant_fp8(w["front_fp8"], w["front_sf"], False).t())
    d8 = max(t_fm.calc_diff(y[:, :1536].float(), ref8[:, :1536]),
             t_fm.calc_diff(win_y2, ref8[:, 1536:]))
    stats["A_front_d8"] = max(stats.get("A_front_d8", 0.0), d8)

    # -- 3. wq_b ALL fusions; the KERNEL reads the folded idx state ring,
    #       the indexer fused pages AND the SWA MODEL1 pages directly --------
    xq = torch.empty(B, 1536, device=DEV, dtype=torch.float8_e4m3fn)
    xq_sf = torch.empty(B, 12, device=DEV, dtype=torch.uint8)
    weights64 = w64
    B_mla = mla_batch(B)
    owned = owned_slice(B)
    pos_mla = pos[owned] if MLA_DP_MODE else pos
    query_comm = query_tp2_ws(B) if TPDP_MODE and fused_query_rms() else None
    q_ready = (query_comm.output(B) if query_comm is not None else
               None if inplace_query_rms() else
               torch.empty(B_mla, 1, attention_heads(), Q_DIM,
                           device=DEV, dtype=torch.bfloat16))

    # Prepare the MQA metadata before entering the PDL submission sequence.
    ncmp = mgr.n_compressed(slots, pos)
    logits_buf.fill_(float("-inf"))
    B_local = local_batch(B)
    idx_bt = mgr.block_table("idx", slots)[owned]
    if INDEXER_FP8:
        ctx2 = ncmp[owned].view(-1, 1).contiguous()
        schedule = get_dg().get_paged_mqa_logits_metadata(
            ctx2, mgr.entries_per_block["idx"], get_dg().get_num_sms())
    mqa_state_row = (st["main_state_row"] if INDEXER_FP8 else
                     st["main_state_row"].to(torch.int32))
    mqa_cmp_dst = (st["cmp_dst"] if INDEXER_FP8 else
                   st["cmp_dst"].to(torch.int32))

    # Submit the PDL pair without intervening CUDA work. The FP8 MQA grid waits
    # for Q_B, then its independent tail warpgroup produces q_ready alongside
    # attention and the MAIN compressor.
    rets = wqm.wq_b_proj_gemm_merged(
        xq, t_wq.as_ue8m0(xq_sf), w["wq_fp8"], w["wq_sf"], q_pos,
        w["cos"], w["sin"], head_ssq=None, enable_ssq=False, mock_post=False,
        cmp_pos=pos,
        idx_norm=w["idx_norm"],
        cos_tab=w["cos"], sin_tab=w["sin"],
        idx_state=mgr.idx_state, idx_state_row=st["idx_state_row"],
        state_ring_entries=mgr.state_ring_entries,
        win_y2=win_y2, win_norm=w["win_norm"],
        q_y=y[:, :1536], q_norm_w=w["q_norm"],
        idx_cache=mgr.idx_pool, idx_dst=st["idx_dst"],
        idx_entries_per_block=mgr.entries_per_block["idx"],
        idx_block_stride_bytes=mgr.block_stride_bytes["idx"],
        swa_cache=mgr.swa_pool, swa_dst=st["swa_dst"],
        swa_entries_per_block=mgr.entries_per_block["swa"],
        swa_block_stride_bytes=mgr.block_stride_bytes["swa"],
        indexer_fp8=INDEXER_FP8,
        iq_weights=weights64 if INDEXER_FP8 else None,
        pdl=pdl_enabled())
    yq, iq_fp4, iq_sf = rets[0], rets[1], rets[2]
    if inplace_query_rms():
        q_ready = yq.view(B_mla, 1, attention_heads(), Q_DIM)
    if fused_query_rms():
        common = dict(
            kv_entries_per_block=mgr.entries_per_block["idx"],
            kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
            cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=w["cos"],
            sin_tab=w["sin"], comp_state=mgr.main_state,
            comp_state_row=mqa_state_row, cmp_cache=mgr.cmp_pool,
            cmp_dst=mqa_cmp_dst,
            comp_state_ring_entries=mgr.state_ring_entries,
            cmp_entries_per_block=mgr.entries_per_block["cmp"],
            cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"],
            query_x=yq, query_positions=pos if query_comm else pos_mla,
            query_cos=w["cos"],
            query_sin=w["sin"], query_out=q_ready,
            query_input_heads=WQ_HEADS, query_eps=EPS,
            query_symmetric_ptrs=(query_comm.pointers if query_comm else []),
            query_tp_rank=TP_RANK,
            query_batch_total=B if query_comm else 0,
            pdl=pdl_enabled())
        if INDEXER_FP8:
            mq8m.mqa_logits_fp8_decode_out(
                iq_fp4[owned], mgr.idx_pool, iq_sf[owned], ctx2, idx_bt,
                schedule, logits_buf[owned], **common)
        else:
            mqm.mqa_logits_fp4_decode_out(
                iq_fp4[owned], iq_sf[owned], mgr.idx_pool, weights64[owned],
                ncmp[owned], idx_bt, logits_buf[owned], **common)
        logits = logits_buf[owned]
    else:
        qrm.rmsnorm_rope_out(yq, pos_mla, w["cos"], w["sin"], q_ready,
                             WQ_HEADS, EPS, pdl_enabled(), Q_RMS_MAX_BLOCKS)
        if INDEXER_FP8:
            mq8m.mqa_logits_fp8_decode_out(
                iq_fp4[owned], mgr.idx_pool, iq_sf[owned], ctx2, idx_bt,
                schedule, logits_buf[owned],
                kv_entries_per_block=mgr.entries_per_block["idx"],
                kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
                cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=w["cos"],
                sin_tab=w["sin"], comp_state=mgr.main_state,
                comp_state_row=mqa_state_row, cmp_cache=mgr.cmp_pool,
                cmp_dst=mqa_cmp_dst,
                comp_state_ring_entries=mgr.state_ring_entries,
                cmp_entries_per_block=mgr.entries_per_block["cmp"],
                cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"],
                pdl=pdl_enabled())
            logits = logits_buf[owned]
        else:
            mqm.mqa_logits_fp4_decode_out(
                iq_fp4[owned], iq_sf[owned], mgr.idx_pool, weights64[owned],
                ncmp[owned], idx_bt, logits_buf[owned],
                cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=w["cos"],
                sin_tab=w["sin"], comp_state=mgr.main_state,
                comp_state_row=mqa_state_row, cmp_cache=mgr.cmp_pool,
                cmp_dst=mqa_cmp_dst,
                kv_entries_per_block=mgr.entries_per_block["idx"],
                kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
                comp_state_ring_entries=mgr.state_ring_entries,
                cmp_entries_per_block=mgr.entries_per_block["cmp"],
                cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"])
            logits = logits_buf[owned]

    qr, _ = t_wq.ref_qnorm_quant(y[:, :1536], w["q_norm"])
    stats["B_xq_match"] = min(stats.get("B_xq_match", 1.0),
                              (xq.view(torch.uint8) == qr.view(torch.uint8))
                              .float().mean().item())

    # -- 4. MAIN state is published by FRONT; MQA also writes compressed KV.
    # Assert every live score is finite (no NaN/+inf in logits).
    torch.cuda.synchronize()
    finite = True
    for b in range(B_local):
        global_b = (owned.start or 0) + b
        n = int(ncmp[global_b])
        if n:
            lg = logits[b, :n]
            finite &= bool((torch.isnan(lg) | torch.isinf(lg)).sum().item() == 0)
    stats["G_finite"] = stats.get("G_finite", True) and finite

    kv_rows = deq_idx_pool(mgr, slots, pos)
    refs = ref_mqa_logits(iq_fp4, iq_sf, weights64, kv_rows)
    for b in range(B_local):
        global_b = (owned.start or 0) + b
        n = int(ncmp[global_b])
        if n:
            d = (logits[b, :n].float() - refs[global_b]).abs().max().item()
            rel = d / (refs[global_b].abs().max().item() + 1e-6)
            stats["C_mqa_rel"] = max(stats.get("C_mqa_rel", 0.0), rel)

    # -- 6. topk_v2 (512, page transform -> compressed physical) ----------
    # scores: COLUMN SLICE of the wide buffer (row stride 256 % 4 == 0 keeps
    # the 16B-vector-load contract; a .contiguous() copy would break it for
    # tiny ncmp). -inf tail beyond seq_len is never read.
    cmp_bt = mgr.block_table("cmp", slots)[owned]
    L = max(int(ncmp[owned].max()), 1)
    if query_comm is None:
        page_idx = tkm.topk_v2(logits[:, :L], ncmp[owned], cmp_bt,
                               TOPK, PAGE, None)
    else:
        page_idx = torch.empty(
            B_local, TOPK, device=DEV, dtype=torch.int32)
        topk_meta = torch.zeros(
            B_local + 1, 2, device=DEV, dtype=torch.int32)
        tkm.topk_v2_transform(
            logits[:, :L], ncmp[owned], cmp_bt, page_idx, PAGE,
            topk_meta, None, query_comm.generation,
            query_comm.signal_pad_pointers, TP_RANK, QUERY_READY_OFFSET)
    for b in range(B_local):
        global_b = (owned.start or 0) + b
        n = int(ncmp[global_b])
        k = min(n, TOPK)
        if k == 0:
            ok = bool((page_idx[b] == -1).all())
        else:
            ref_raw = torch.topk(refs[global_b], k).indices if n > TOPK else \
                torch.arange(n, device=DEV)
            ref_phys = cmp_bt[b][ref_raw // PAGE] * PAGE + ref_raw % PAGE
            ok = (torch.sort(page_idx[b][:k])[0]
                  == torch.sort(ref_phys.int())[0]).all().item()
        stats["D_topk_ok"] = stats.get("D_topk_ok", True) and bool(ok)

    if TPDP_MODE and not MLA_DP_MODE:
        cmp_bt_full = mgr.block_table("cmp", slots)
        logical = torch.arange(TOPK, device=DEV, dtype=torch.int64)
        logical = logical % (cmp_bt_full.size(1) * PAGE)
        page_idx_full = (cmp_bt_full[:, logical // PAGE] * PAGE
                         + (logical % PAGE).int()).contiguous()
        page_idx_full[owned].copy_(page_idx)
    else:
        page_idx_full = page_idx

    rc = w["cos"][pos_mla].view(B_mla, 1, 32).contiguous()
    rs = w["sin"][pos_mla].view(B_mla, 1, 32).contiguous()
    if inplace_query_rms():
        # yq has been normalized in-place. Reconstruct the raw BF16 GEMM output
        # from the already-published Q_B inputs for the correctness reference.
        q_raw = (t_wq.dequant_act(xq, xq_sf)
                 @ t_wq.dequant_weight(
                     w["wq_fp8"][t_wq.N_IDX:],
                     w["wq_sf"].view(torch.uint8)[t_wq.N_IDX // 128:]).t())
        q_raw = q_raw.bfloat16().view(B, 1, WQ_HEADS, Q_DIM)[:B_mla]
    else:
        q_raw = yq.view(B, 1, WQ_HEADS, Q_DIM)
    if TPDP_MODE:
        gathered = [torch.empty_like(q_raw) for _ in range(TP_WORLD_SIZE)]
        dist.all_gather(gathered, q_raw)
        q_raw = torch.cat(gathered, dim=2)[owned]
    else:
        q_raw = q_raw[:B_mla]
    q_ref = q_raw.float()
    q_ref = q_ref * torch.rsqrt(q_ref.square().mean(-1, keepdim=True) + EPS)
    even, odd = q_ref[..., 448::2].clone(), q_ref[..., 449::2].clone()
    rc_heads, rs_heads = rc.unsqueeze(-2), rs.unsqueeze(-2)
    q_ref[..., 448::2] = even * rc_heads - odd * rs_heads
    q_ref[..., 449::2] = even * rs_heads + odd * rc_heads
    stats["Q_rms_rope_diff"] = max(
        stats.get("Q_rms_rope_diff", 0.0),
        t_fm.calc_diff(q_ready.float(), q_ref.bfloat16().float()))

    # -- 7. official flashMLA over the native post-RMSNorm+RoPE query ------
    if post_attn_enabled():
        slots_mla = slots[owned] if MLA_DP_MODE else slots
        swa_idx, swa_len = mgr.swa_indices(slots, pos, SWA_TOPK)
        if MLA_DP_MODE:
            swa_idx, swa_len = swa_idx[owned], swa_len[owned]
        cmp_len = torch.minimum(ncmp[owned] if MLA_DP_MODE else ncmp,
                                torch.tensor(TOPK, device=DEV)).int()
        # Fresh sched_meta EVERY step: it is only reusable while shapes AND
        # cache_seqlens/topk_length values stay identical (interface doc);
        # our lens advance each step.
        sched, _ = flash_mla.get_mla_metadata()
        sink = w["attn_sink"]
        res = flash_mla.flash_mla_with_kvcache(
            q=q_ready, k_cache=mgr.model1_cache_view("swa"),
            block_table=None, cache_seqlens=None, head_dim_v=Q_DIM,
            tile_scheduler_metadata=sched, num_splits=None,
            softmax_scale=Q_DIM ** -0.5, causal=False, is_fp8_kvcache=True,
            indices=swa_idx, topk_length=swa_len,
            extra_k_cache=mgr.model1_cache_view("cmp"),
            extra_indices_in_kvcache=page_idx_full.view(B_mla, 1, TOPK),
            extra_topk_length=cmp_len,
            attn_sink=sink)
        out = res[0] if isinstance(res, tuple) else res
        # torch reference over the DEQUANT pools
        for b in range(B_mla):
            p = int(pos_mla[b])
            qn = q_ref[b, 0]
            table = mgr.reqs[slots_mla[b]]["swa"]
            n_sw = int(swa_len[b])
            sw = torch.stack([deq_model1_row(mgr.swa_pool,
                                             table[(t % WIN) // mgr.entries_per_block["swa"]],
                                             (t % WIN) % mgr.entries_per_block["swa"],
                                             mgr.entries_per_block["swa"])
                              for t in range(p + 1 - n_sw, p + 1)]) \
                if n_sw else torch.zeros(0, Q_DIM, device=DEV)
            cm = torch.stack([deq_model1_row(mgr.cmp_pool, int(pi) // PAGE,
                                             int(pi) % PAGE,
                                             mgr.entries_per_block["cmp"])
                              for pi in page_idx_full[b][:int(cmp_len[b])]]) \
                if int(cmp_len[b]) else torch.zeros(0, Q_DIM, device=DEV)
            ref = ref_flash_attn(qn.bfloat16().float(), sw, cm, Q_DIM ** -0.5,
                                 sink)
            if torch.isnan(out[b]).any():   # NaN source diag (F regression)
                print(f"  [NaN-diag] b={b} pos={p} "
                      f"out={int(torch.isnan(out[b]).sum())} "
                      f"q={int(torch.isnan(qn).sum())} "
                      f"sw={int(torch.isnan(sw).sum())}/{int(swa_len[b])} "
                      f"cm={int(torch.isnan(cm).sum())}/{int(cmp_len[b])} "
                      f"ref={int(torch.isnan(ref).sum())}")
            cos = torch.nn.functional.cosine_similarity(
                out[b, 0].float().flatten(), ref.flatten(), dim=0).item()
            stats["E_mla_cos"] = min(stats.get("E_mla_cos", 1.0), cos)
        if dbg is not None:   # F-bisect: last-step producer snapshots
            # DeepGEMM intentionally leaves columns >= context_len unspecified.
            # Normalize only the debug snapshot; TopK already gates reads with
            # ncmp and the production path pays no cleanup kernel.
            dbg_logits = logits.clone()
            dbg_logits.masked_fill_(
                torch.arange(dbg_logits.size(1), device=DEV).view(1, -1)
                >= ncmp[owned].view(-1, 1), float("-inf"))
            dbg.update(y_q=y[:, :1536].clone(), win_y2=win_y2.clone(),
                       yq=yq.clone(), q_ready=q_ready.clone(),
                       logits=dbg_logits, page_idx=page_idx_full.clone(),
                       iq=rets[1].clone())

        # -- 8. O projection -> TP2 symmetric publish/reduce -> mHC post. ----
        mla3_dp = (out[:, 0] if out.dim() == 4 else out).contiguous()
        pos64 = pos.to(torch.int64).contiguous()
        post_t, comb_t = hc["hc_post"], hc["hc_comb"].view(B, HC, HC)
        projected = o_proj_csa.run_o_proj_mhc_post(
            mla3_dp, pos64, w["cos_sin"], hidden, post_t, comb_t,
            w["o_proj"], oproj_ws(B), mpm, use_pdl=pdl_enabled(),
            force_pdl=pdl_forced(), run_mhc_post=True,
            tpdp_mla_scatter=MLA_DP_MODE)
        if MLA_DP_MODE:
            max_local = (B + 1) // 2
            padded = torch.zeros(
                max_local, Q_HEADS, Q_DIM, device=DEV,
                dtype=torch.bfloat16)
            padded[:mla3_dp.size(0)].copy_(mla3_dp)
            gathered = [torch.empty_like(padded) for _ in range(2)]
            dist.all_gather(gathered, padded, group=TP_SYNC_GROUP)
            split = (B + 1) // 2
            mla_full = torch.cat(
                (gathered[0][:split], gathered[1][:B - split]), dim=0)
            head_begin = TP_RANK * WQ_HEADS
            mla3_ref = mla_full[:, head_begin:head_begin + WQ_HEADS].contiguous()
        else:
            mla3_ref = mla3_dp
        ref_partial = ref_o_proj_mhc(
            mla3_ref, pos64, w["cos_sin"], hidden, post_t, comb_t,
            w["wo_a"], w["wo_b"], run_mhc_post=False)
        if TPDP_MODE:
            ref_partial = tp2_reference_reduce(ref_partial)
        ref_final = ref_mhc_post(ref_partial, hidden, post_t, comb_t)
        stats["H_oproj_diff"] = max(
            stats.get("H_oproj_diff", 0.0),
            t_fm.calc_diff(projected.float(), ref_final.float()))
        return projected.reshape(B, -1)
    # No flash_mla: the VALID y segment only -- under FRONT-EMIT the cols
    # [1536,4672) are never written (side buffers replace them), so the
    # full-y snapshot would compare uninitialized memory.
    return y[:, :1536].contiguous()


# ==================== per-operator latency ====================
# (the old Triton "glue" op is GONE: front's epilogue now emits the main/idx
# state rows, win_y2 and w64 directly -- see [FRONT-EMIT])


def probe_kernel_names(fn):
    """All kernel names fn launches (flush/memset/aten prep excluded); every
    benched stage is a pure named-kernel call (quant/glue are single Triton/
    CUDA kernels), so the strict filter fits all of them."""
    from torch.profiler import ProfilerActivity, profile
    tp_host_barrier()
    fn(); torch.cuda.synchronize()
    try:
        ctx = profile(activities=[ProfilerActivity.CUDA], acc_events=True)
    except TypeError:
        ctx = profile(activities=[ProfilerActivity.CUDA])
    with ctx as prof:
        tp_host_barrier()
        fn(); torch.cuda.synchronize()
    names = []
    for e in prof.events():
        n = e.name
        if (any(s in n for s in ("Memset", "memset", "fill", "Memcpy",
                                 "zero_", "elementwise", "reduce_kernel"))
                and "mhc_post_reduce_kernel" not in n):
            continue
        d = getattr(e, "device_time", None)
        if d is None:
            d = getattr(e, "cuda_time", 0.0)
        if d and d > 0 and n[:80] not in names:
            names.append(n[:80])
    assert names, "probe found no kernels"
    return tuple(names)


def benchmark(mods, w, ncmp=2048, batches=None,
              pure_dp_baseline_only=False, results_json=None):
    """COLD-L2 measurement, ONE full CSA layer per call: an 8GB memset (the
    bench_utils.bench_kineto flusher, i.e. what every single-operator test in
    this tree already uses) runs before EVERY chain call, so a layer starts
    with none of its own weights in L2. That is the real decode situation --
    consecutive layers hold DIFFERENT weights, so layer N never finds layer
    N-1's 113MB wq_b / 78MB front / 130MB o_proj streams cached, and a hot
    no-flush number would flatter every weight-bound stage.
      per-stage : mean Kineto device time over 30 cold eager-chain iterations,
                  bucketed by stage (wq_b bucket = qnorm + merged GEMM;
                  duration sums double-count PDL and intra-stage overlaps)
      perfetto  : first GPU kernel start through last GPU kernel end in the
                  saved Perfetto trace (vLLM serving form; THE end-to-end
                  number). The per-layer CUDA-event graph envelope and CPU
                  graph-launch wall are excluded on purpose:
                  a whole model is ONE graph, so its launch, synchronization,
                  and graph-envelope cost is paid once for all 61 layers,
                  not once per layer.
    The flush is ONE per layer (never per stage) and is EXCLUDED from every
    reported number: Kineto buckets name only the chain's own kernels, and the
    Perfetto span starts at the first chain kernel after the flush.
    Context ops (mqa/topk/mla) at a FABRICATED long context: page tables
    and lens real, cache bytes arbitrary -- latency-neutral. The chain runs
    through O-proj and mHC post. TPDP uses the symmetric dual-write ProjB and
    fused BF16 reduce+mHC post. The O-proj entry point performs validation, which
    does not affect graph replay or Kineto device time."""
    import builtins
    print = (builtins.print if not TPDP_MODE or TP_RANK == 0
             else lambda *args, **kwargs: None)
    hcm, fmm, wqm, qrm, mqm, mq8m, tkm, mpm = mods
    trace_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "docs", "timeline"))
    os.makedirs(trace_dir, exist_ok=True)
    trace_mode = (f"tpdp_rank{TP_RANK}" if TPDP_MODE else "full")
    if not INDEXER_FP8:
        trace_mode += "_idx-fp4"
    if not fused_query_rms():
        trace_mode += "_qrms-separate"
    if PDL_MODE != "on":
        trace_mode += f"_pdl-{PDL_MODE}"
    print("\n" + "=" * 76)
    title = ("Pure-DP B/2 communication-study baseline"
             if pure_dp_baseline_only else "Per-operator latency")
    print(f"{title} (us) -- compress-row step, ncmp={ncmp} "
          f"compressed tokens ({ncmp * RATIO} ctx)")
    print("=" * 76)
    S = 4 * ncmp + 8
    ang = torch.rand(S, 32, device=DEV) * 6.28
    cosl, sinl = ang.cos().contiguous(), ang.sin().contiguous()
    cols = (["graph+rms", "graph-no-rms", "delta"] if Q_RMS_ABLATION else
            ["mhc", "front", "wq_b"] +
            ([] if fused_query_rms() else ["q_rms"]) + ["mqa", "topk"] +
            (["mla", "o_proj"] if post_attn_enabled() else []) +
            (["mhc_post"] if mhc_post_enabled() and not TPDP_MODE else []) +
            ["stages", "perfetto"])
    # Match DeepGEMM bench_kineto exactly: an excessive 8 GB stream operation
    # evicts L2 and keeps the device out of the low-clock idle state.
    flush_bytes = int(8e9)
    graph_perfetto_samples = 30
    graph_clock_sleep_cycles = int(2e7)
    print("  L2 policy: memset eviction "
          f"({flush_bytes / 1e9:.0f} GB, excluded from every number); "
          "all reported values are device us")
    if pure_dp_baseline_only:
        print("  --bench-batches denotes global B; this process runs full-model "
              "pure DP at local B/2 with no distributed initialization, peer "
              "access, communication flag, or communication wait")
        print("  perfetto follows DeepGEMM bench_kineto: one warmup phase and "
              f"one active phase of {graph_perfetto_samples} cold graph replays; "
              "the active-sample mean is primary")
    if pure_dp_baseline_only:
        pass
    elif Q_RMS_ABLATION:
        print("  one CUDA graph and identical PDL/grid; only the RMSNorm+RoPE "
              "body is toggled by a device flag")
        if inplace_query_rms():
            print("  full-rank production layout is in-place: FlashMLA consumes "
                  "normalized query when on and raw Q_B query when off")
        else:
            print("  TPDP is out-of-place: the prior normalized output is retained "
                  "when the body is off")
        print("  paired samples alternate order; delta = graph+rms - graph-no-rms")
    else:
        print("  wq_b = qnorm producer + merged GEMM device-time sum; standalone "
              "fused(us) reports only the merged GEMM")
        if fused_query_rms():
            print("  query RMSNorm+RoPE runs in the MQA tail warpgroup and is "
                  "included in mqa; use perfetto for end-to-end latency")
        else:
            print("  q_rms and mqa are overlapping kernel durations; stages sums both. "
                  "Use perfetto for end-to-end latency")
        print("  operator columns are 30 cold eager-chain Kineto means; stages sums "
              "kernel durations and includes overlap")
        print("  perfetto follows DeepGEMM bench_kineto: one warmup phase and "
              f"one active phase of {graph_perfetto_samples} cold graph replays")
        if TPDP_MODE:
            print("  every active sample first takes the cross-rank maximum; "
                  "perfetto reports the mean of those paired samples")
        else:
            print("  perfetto reports the active-sample mean; the complete active "
                  "timeline is saved for inspection")
    if not fused_query_rms():
        print(f"  q_rms persistent grid cap = {Q_RMS_MAX_BLOCKS} CTAs")
    if TPDP_MODE:
        print("  --tpdp: MQA/TopK/MLA own disjoint half-batches; MLA H=128; O-proj "
              "remains global B x H=64 with the 8-group TP2 shard")
        print("  MQA gathers TP query heads through symmetric memory; post-attention "
              "includes the MLA handoff, symmetric ProjB publication, and fused "
              "BF16 reduce+mHC post")
    col_width = 13 if Q_RMS_ABLATION else 9
    if not pure_dp_baseline_only:
        print(f"  {'B':<5}" + "".join(f"{c:>{col_width}}" for c in cols))
        print("  " + "-" * (5 + col_width * len(cols)))
    bw_rows = []    # (B, [(stage, us)]) for the bandwidth table
    communication_study_rows = []
    # Keep the DeepGEMM-sized buffer stable across samples. Reusing the storage
    # has the same device memset behavior without allocator events in the loop.
    flush_buf = torch.empty((flush_bytes + 3) // 4,
                            dtype=torch.int, device=DEV)

    def flush_l2(drain=True):
        """Evict L2.
        drain=True (wall path): also drain before returning.
        drain=False (event path): leave the eviction IN FLIGHT and enqueue
          behind it. An event recorded next is still stream-ordered AFTER the
          flush, so the flush stays out of the window, but the device stays
          busy ~1ms while the host submits the replay -- otherwise the host's
          submit latency shows up as device idle inside the window (~10us for
          this graph)."""
        flush_buf.zero_()
        if drain:
            torch.cuda.synchronize()

    def graph_spans_from_trace(trace, expected_samples):
        """Return one first-to-last graph-kernel span per cudaGraphLaunch."""
        launches = [
            event for event in trace["traceEvents"]
            if event.get("ph") == "X"
            and event.get("cat") == "cuda_runtime"
            and event.get("name") == "cudaGraphLaunch"
        ]
        launches.sort(key=lambda event: event["ts"])
        if len(launches) != expected_samples:
            raise RuntimeError(
                f"Perfetto trace contains {len(launches)} cudaGraphLaunch "
                f"events, expected {expected_samples}")

        kernels_by_correlation = {}
        for event in trace["traceEvents"]:
            if event.get("ph") != "X" or event.get("cat") != "kernel":
                continue
            correlation = event.get("args", {}).get("correlation")
            kernels_by_correlation.setdefault(correlation, []).append(event)

        spans = []
        for launch in launches:
            correlation = launch.get("args", {}).get("correlation")
            kernels = kernels_by_correlation.get(correlation, ())
            if not kernels:
                raise RuntimeError(
                    "Perfetto trace has no GPU kernels correlated with "
                    f"cudaGraphLaunch {correlation}")
            first = min(event["ts"] for event in kernels)
            last = max(event["ts"] + event["dur"] for event in kernels)
            spans.append(float(last - first))
        return spans

    def measure_graph_perfetto(graph, trace_label, batch,
                               samples=graph_perfetto_samples):
        """DeepGEMM-style cold graph sampling with exact graph envelopes.

        Conditioning work is stream-ordered before cudaGraphLaunch and then
        excluded by correlation ID. In TPDP, corresponding rank samples are
        reduced before statistics are computed.
        """
        import statistics
        from torch.profiler import profile as _trace_profile
        from torch.profiler import ProfilerActivity

        trace_path = os.path.join(
            trace_dir, f"e2e_{trace_label}_B{batch}.json")

        # DeepGEMM performs one untimed call before opening Kineto.
        tp_host_barrier()
        torch.cuda._sleep(graph_clock_sleep_cycles)
        if TPDP_MODE:
            tp_device_barrier(ows.tp2_comm)
        graph.replay()
        torch.cuda.synchronize()
        tp_host_barrier()

        schedule = torch.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=1)
        profiler_args = {
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            "schedule": schedule,
            "acc_events": True,
        }
        try:
            trace_prof = _trace_profile(**profiler_args)
        except TypeError:
            profiler_args.pop("acc_events")
            trace_prof = _trace_profile(**profiler_args)

        with trace_prof:
            for _ in range(2):
                tp_host_barrier()
                for _ in range(samples):
                    flush_l2(drain=False)
                    # DeepGEMM uses this before distributed barriers to avoid
                    # clock and host-launch imbalance. Apply it on both paths
                    # so pure-DP and TPDP enter their graphs identically.
                    torch.cuda._sleep(graph_clock_sleep_cycles)
                    if TPDP_MODE:
                        tp_device_barrier(ows.tp2_comm)
                    graph.replay()
                torch.cuda.synchronize()
                tp_host_barrier()
                trace_prof.step()

        trace_prof.export_chrome_trace(trace_path)
        with open(trace_path, "r", encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
        local_spans = graph_spans_from_trace(trace, samples)

        paired_spans = local_spans
        rank_samples = None
        if TPDP_MODE:
            local_tensor = torch.tensor(
                local_spans, device=DEV, dtype=torch.float64)
            gathered = [torch.empty_like(local_tensor) for _ in range(2)]
            dist.all_gather(gathered, local_tensor)
            rank_samples = [values.cpu().tolist() for values in gathered]
            paired_spans = torch.stack(gathered).amax(dim=0).cpu().tolist()

        return {
            "mean_us": float(statistics.mean(paired_spans)),
            "median_us": float(statistics.median(paired_spans)),
            "min_us": float(min(paired_spans)),
            "max_us": float(max(paired_spans)),
            "samples_us": [float(span) for span in paired_spans],
            "rank_samples_us": rank_samples,
            "sample_aggregation": (
                "per-replay cross-rank max, then statistics"
                if TPDP_MODE else "per-replay statistics"),
            "trace_path": trace_path,
        }

    def measure_eager_stage_chain(stage_functions, chain_fn, reps=30):
        """Bucket eager-chain CUDA kernel time by the probed stage names."""
        name2stage = {}
        for stage_name, stage_fn in stage_functions:
            for kernel_name in probe_kernel_names(stage_fn):
                name2stage.setdefault(kernel_name, stage_name)
        from torch.profiler import profile as profiler
        from torch.profiler import ProfilerActivity
        for _ in range(3):
            flush_l2()
            tp_host_barrier()
            chain_fn()
        torch.cuda.synchronize()
        try:
            profile_ctx = profiler(
                activities=[ProfilerActivity.CUDA], acc_events=True)
        except TypeError:
            profile_ctx = profiler(activities=[ProfilerActivity.CUDA])
        with profile_ctx as profile_result:
            for _ in range(reps):
                flush_l2()
                tp_host_barrier()
                chain_fn()
                torch.cuda.synchronize()
        accumulated = {name: 0.0 for name, _ in stage_functions}
        for event in profile_result.events():
            stage_name = name2stage.get(event.name[:80])
            if stage_name is None:
                continue
            duration = getattr(event, "device_time", None)
            if duration is None:
                duration = getattr(event, "cuda_time", 0.0)
            accumulated[stage_name] += duration
        return [accumulated[name] / reps for name, _ in stage_functions]

    if batches is None:
        batches = (2, 16, 32, 48, 64, 80, 96, 112, 128)
    if pure_dp_baseline_only:
        if any(global_b <= 0 or global_b % 2 for global_b in batches):
            raise ValueError(
                "pure-DP baseline global batches must be positive and even")
        batch_cases = tuple((global_b, global_b // 2)
                            for global_b in batches)
    else:
        batch_cases = tuple((batch, batch) for batch in batches)

    for reported_B, B in batch_cases:
        mgr = KVCacheManager(capacity=B + 2, pages_per_pool=(ncmp // PAGE) * B
                             + 4 * B, max_pages_per_req=ncmp // PAGE,
                             indexer_fp8=INDEXER_FP8)
        slots = [mgr.alloc_request() for _ in range(B)]
        for s in slots:                       # fabricate history: next step is
            mgr.reqs[s]["pos"] = 4 * ncmp - 2   # pos 4*ncmp-1 (compress row)
            # _page_of back-fills every page up to tok//PAGE -- ONE call per
            # pool (the old per-token loop was 4M python calls at 64k ctx)
            mgr._page_of("idx", s, ncmp - 1)
            mgr._page_of("cmp", s, ncmp - 1)
        st = mgr.step_begin(slots)
        pos, q_pos = st["pos"], st["q_pos"]
        hidden = torch.randn(B, HC, DIM, device=DEV, dtype=torch.bfloat16) * .1

        # ---- CONNECTED dataflow chain (audit fix): every stage reads THIS
        # iteration's producer outputs. Fixed-address buffers where kernels
        # take out=; wq_b's fresh returns flow through a holder dict -- at
        # graph capture those allocations land in the graph pool, so replay
        # keeps the same addresses and the flow stays connected.
        collapsed = torch.empty(B, DIM, device=DEV, dtype=torch.bfloat16)
        pre = torch.empty(B, HC, device=DEV, dtype=torch.float32)
        po, cb = pre.clone(), torch.empty(B, HC, HC, device=DEV)
        mix = torch.empty(B, 24, device=DEV, dtype=torch.float32)
        S = hc_n_splits(B)     # hybrid mhc: preallocated split-K workspace
        hws = torch.empty(S, B, 24, device=DEV, dtype=torch.float32)
        hsq = torch.empty(S, B, device=DEV, dtype=torch.float32)
        # fp8 x + sf: written by the mhc epilogue's FUSED act quant (rows
        # [:B]); pad rows stay zero from init (front reads Bp rows).
        Bp = max(B, 16)
        x8 = torch.zeros(Bp, DIM, device=DEV, dtype=torch.float8_e4m3fn)
        x8v = x8.view(torch.uint8)
        x8s = torch.zeros(Bp, DIM // 128, device=DEV, dtype=torch.uint8)
        run_hc = lambda: run_mhc_hybrid(get_dg(), hcm, hidden, w, collapsed,
                                        pre, po, cb, hws, hsq, mix_out=mix,
                                        xq_out=x8v[:B], xsf_out=x8s[:B])
        # front: pad16 contract for B<16; hc tail consumes the REAL mhc mix;
        # fp8 x comes fused from the mhc epilogue (no quant kernel).
        coll_p = collapsed if Bp == B else \
            torch.zeros(Bp, DIM, device=DEV, dtype=torch.bfloat16)
        mix_p = mix if Bp == B else \
            torch.zeros(Bp, 24, device=DEV, dtype=torch.float32)
        hcd = {"hc_mix": mix_p, "hc_base": w["hc_base"],
               "hc_scale": w["hc_scale"],
               "hc_post": torch.empty(Bp, HC, device=DEV),
               "hc_comb": torch.empty(Bp, HC * HC, device=DEV)}
        y_p = torch.empty(Bp, N_FRONT, device=DEV, dtype=torch.bfloat16)
        y = y_p[:B]
        hold = {}
        # [FRONT-EMIT] side buffers ([Bp]; consumers read [:B]). Pad rows
        # scatter into slot B row 0 -- an allocated-but-unused trash slot
        # (mgr capacity = B+2, bench uses slots [0,B)). IDX state goes to
        # the pool directly (same state_row as main).
        win_y2_p = torch.empty(Bp, 512, device=DEV, dtype=torch.float32)
        w64_p = torch.empty(Bp, 64, device=DEV, dtype=torch.float32)
        win_y2, weights64 = win_y2_p[:B], w64_p[:B]
        p0 = 4 * ncmp - 1                       # same pos for every request
        main_state_row = torch.full((Bp,), -1, dtype=torch.int64, device=DEV)
        main_state_row[:B] = mgr.state_rows(slots, pos)
        idx_state_row = main_state_row.clone()
        ape_phase = torch.full((Bp,), p0 % 4, dtype=torch.int32, device=DEV)

        def run_front():
            if Bp != B:
                coll_p[:B].copy_(collapsed)
                mix_p[:B].copy_(mix)
            fmm.front_mixed_gemm_csa_swapab(
                coll_p, x8, x8s, w["front_bf16"], w["front_fp8"],
                w["front_sf"], out=y_p, **hcd, enable_tail=True,
                main_state=mgr.main_state, main_ape=w["main_ape"],
                main_state_row=main_state_row, ape_phase=ape_phase,
                idx_state=mgr.idx_state, idx_state_row=idx_state_row,
                idx_ape=w["idx_ape"], win_y2=win_y2_p, w64=w64_p,
                pdl=pdl_enabled())
        run_hc(); run_front()

        xq = torch.empty(B, 1536, device=DEV, dtype=torch.float8_e4m3fn)
        xq_sf = t_wq.as_ue8m0(torch.empty(B, 12, device=DEV,
                                          dtype=torch.uint8))
        def run_wqb():
            hold["r"] = wqm.wq_b_proj_gemm_merged(
                xq, xq_sf, w["wq_fp8"], w["wq_sf"], q_pos, cosl, sinl,
                head_ssq=None, enable_ssq=False, mock_post=False, cmp_pos=pos,
                idx_norm=w["idx_norm"], cos_tab=cosl,
                sin_tab=sinl, idx_state=mgr.idx_state,
                idx_state_row=st["idx_state_row"],
                state_ring_entries=mgr.state_ring_entries,
                win_y2=win_y2, win_norm=w["win_norm"],
                q_y=y[:, :1536], q_norm_w=w["q_norm"],
                idx_cache=mgr.idx_pool, idx_dst=st["idx_dst"],
                idx_entries_per_block=mgr.entries_per_block["idx"],
                idx_block_stride_bytes=mgr.block_stride_bytes["idx"],
                swa_cache=mgr.swa_pool, swa_dst=st["swa_dst"],
                swa_entries_per_block=mgr.entries_per_block["swa"],
                swa_block_stride_bytes=mgr.block_stride_bytes["swa"],
                indexer_fp8=INDEXER_FP8,
                iq_weights=weights64 if INDEXER_FP8 else None,
                pdl=pdl_enabled())
        nc = mgr.n_compressed(slots, pos)
        B_local = local_batch(B)
        owned = owned_slice(B)
        B_mla = mla_batch(B)
        pos_mla = pos[owned] if MLA_DP_MODE else pos
        query_comm = (query_tp2_ws(B)
                      if TPDP_MODE and fused_query_rms() else None)
        q_ready = (query_comm.output(B) if query_comm is not None else
                   None if inplace_query_rms() else
                   torch.empty(B_mla, 1, attention_heads(), Q_DIM,
                               device=DEV, dtype=torch.bfloat16))
        q_rms_work_flag = (torch.ones((), device=DEV, dtype=torch.int32)
                           if Q_RMS_ABLATION else None)

        def query_ready():
            if inplace_query_rms():
                return hold["r"][0].view(
                    B_mla, 1, attention_heads(), Q_DIM)
            return q_ready

        def run_qrms():
            qrm.rmsnorm_rope_out(
                hold["r"][0], pos_mla, cosl, sinl, query_ready(), WQ_HEADS,
                EPS, pdl_enabled(), Q_RMS_MAX_BLOCKS)

        idx_bt = mgr.block_table("idx", slots)[owned]
        logits = torch.full((B, (ncmp + 255) // 256 * 256), float("-inf"),
                            device=DEV)
        fp8_schedule = None
        if INDEXER_FP8:
            fp8_schedule = get_dg().get_paged_mqa_logits_metadata(
                nc[owned].view(-1, 1).contiguous(),
                mgr.entries_per_block["idx"], get_dg().get_num_sms())
        mqa_state_row = (st["main_state_row"] if INDEXER_FP8 else
                         st["main_state_row"].to(torch.int32))
        mqa_cmp_dst = (st["cmp_dst"] if INDEXER_FP8 else
                       st["cmp_dst"].to(torch.int32))

        # PRODUCTION form: compact comp outputs omitted (cache direct write).
        def run_mqa_for(query_workspace, query_output):
            r = hold["r"]
            if fused_query_rms():
                query_x = (r[0]
                           if query_workspace is not None or not TPDP_MODE
                           else r[0][owned])
            if fused_query_rms() and INDEXER_FP8:
                mq8m.mqa_logits_fp8_decode_out(
                    r[1][owned], mgr.idx_pool, r[2][owned],
                    nc[owned].view(-1, 1), idx_bt, fp8_schedule,
                    logits[owned],
                    kv_entries_per_block=mgr.entries_per_block["idx"],
                    kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
                    cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=cosl,
                    sin_tab=sinl, comp_state=mgr.main_state,
                    comp_state_row=mqa_state_row,
                    cmp_cache=mgr.cmp_pool, cmp_dst=mqa_cmp_dst,
                    comp_state_ring_entries=mgr.state_ring_entries,
                    cmp_entries_per_block=mgr.entries_per_block["cmp"],
                    cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"],
                    query_x=query_x,
                    query_positions=(pos if query_workspace else pos_mla),
                    query_cos=cosl, query_sin=sinl, query_out=query_output,
                    query_input_heads=WQ_HEADS, query_eps=EPS,
                    query_work_flag=q_rms_work_flag,
                    query_symmetric_ptrs=(query_workspace.pointers
                                          if query_workspace else []),
                    query_tp_rank=(TP_RANK if query_workspace else -1),
                    query_batch_total=B if query_workspace else 0,
                    pdl=pdl_enabled())
                hold["logits"] = logits[owned]
            elif fused_query_rms():
                mqm.mqa_logits_fp4_decode_out(
                    r[1][owned], r[2][owned], mgr.idx_pool, weights64[owned],
                    nc[owned], idx_bt, logits[owned],
                    cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=cosl,
                    sin_tab=sinl, comp_state=mgr.main_state,
                    comp_state_row=mqa_state_row, cmp_cache=mgr.cmp_pool,
                    cmp_dst=mqa_cmp_dst,
                    kv_entries_per_block=mgr.entries_per_block["idx"],
                    kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
                    comp_state_ring_entries=mgr.state_ring_entries,
                    cmp_entries_per_block=mgr.entries_per_block["cmp"],
                    cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"],
                    query_x=query_x,
                    query_positions=(pos if query_workspace else pos_mla),
                    query_cos=cosl, query_sin=sinl, query_out=query_output,
                    query_input_heads=WQ_HEADS, query_eps=EPS,
                    query_work_flag=q_rms_work_flag,
                    query_symmetric_ptrs=(query_workspace.pointers
                                          if query_workspace else []),
                    query_tp_rank=(TP_RANK if query_workspace else -1),
                    query_batch_total=B if query_workspace else 0,
                    pdl=pdl_enabled())
                hold["logits"] = logits[owned]

            elif INDEXER_FP8:
                mq8m.mqa_logits_fp8_decode_out(
                    r[1][owned], mgr.idx_pool, r[2][owned],
                    nc[owned].view(-1, 1), idx_bt, fp8_schedule,
                    logits[owned],
                    kv_entries_per_block=mgr.entries_per_block["idx"],
                    kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
                    cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=cosl,
                    sin_tab=sinl, comp_state=mgr.main_state,
                    comp_state_row=mqa_state_row,
                    cmp_cache=mgr.cmp_pool, cmp_dst=mqa_cmp_dst,
                    comp_state_ring_entries=mgr.state_ring_entries,
                    cmp_entries_per_block=mgr.entries_per_block["cmp"],
                    cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"],
                    pdl=pdl_enabled())
                hold["logits"] = logits[owned]
            else:
                mqm.mqa_logits_fp4_decode_out(
                    r[1][owned], r[2][owned], mgr.idx_pool, weights64[owned],
                    nc[owned], idx_bt, logits[owned],
                    cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=cosl,
                    sin_tab=sinl, comp_state=mgr.main_state,
                    comp_state_row=mqa_state_row, cmp_cache=mgr.cmp_pool,
                    cmp_dst=mqa_cmp_dst,
                    kv_entries_per_block=mgr.entries_per_block["idx"],
                    kv_block_stride_bytes=mgr.block_stride_bytes["idx"],
                    comp_state_ring_entries=mgr.state_ring_entries,
                    cmp_entries_per_block=mgr.entries_per_block["cmp"],
                    cmp_block_stride_bytes=mgr.block_stride_bytes["cmp"])
                hold["logits"] = logits[owned]
        def run_mqa():
            return run_mqa_for(query_comm, query_ready())
        # FP8: MQA waits directly for Q_B and its tail produces q_ready.
        # FP4 retains the separate RMS/RoPE relay.
        run_wqb()
        if not fused_query_rms():
            run_qrms()
        run_mqa()
        cmp_bt_full = mgr.block_table("cmp", slots)
        cmp_bt = cmp_bt_full[owned]
        logical = torch.arange(TOPK, dtype=torch.int64, device=DEV)
        page_idx_full = (cmp_bt_full[:, logical // PAGE] * PAGE
                         + (logical % PAGE).int()).contiguous()
        page_idx = page_idx_full[owned]
        meta = torch.zeros(B_local + 1, 2, dtype=torch.int32, device=DEV)
        def run_topk_for(query_workspace):
            tkm.topk_v2_transform(
                hold["logits"][:, :ncmp], nc[owned], cmp_bt, page_idx,
                PAGE, meta, None,
                query_workspace.generation if query_workspace else None,
                (query_workspace.signal_pad_pointers
                 if query_workspace else []),
                TP_RANK if query_workspace else -1,
                QUERY_READY_OFFSET if query_workspace else 0)

        def run_topk():
            return run_topk_for(query_comm)
        run_topk()

        stage_fns = [("mhc", run_hc), ("front", run_front),
                     ("wq_b", run_wqb)]
        if not fused_query_rms():
            stage_fns.append(("q_rms", run_qrms))
        stage_fns.extend((("mqa", run_mqa), ("topk", run_topk)))
        if post_attn_enabled():
            swa_idx, swa_len = mgr.swa_indices(slots, pos, SWA_TOPK)
            if MLA_DP_MODE:
                swa_idx, swa_len = swa_idx[owned], swa_len[owned]
                ext_idx = page_idx.view(B_mla, 1, TOPK)
                cmp_len = torch.minimum(
                    nc[owned], torch.tensor(TOPK, device=DEV)).int()
            else:
                ext_idx = page_idx_full.view(B, 1, TOPK)
                cmp_len = torch.minimum(
                    nc, torch.tensor(TOPK, device=DEV)).int()
            swa_v, cmp_v = mgr.model1_cache_view("swa"), mgr.model1_cache_view("cmp")
            sched, _ = flash_mla.get_mla_metadata()
            mla_sink_kw = ({} if not FMLA_ATTN_SINK
                           else dict(attn_sink=w["attn_sink"]))

            def run_mla_for(query):
                res = flash_mla.flash_mla_with_kvcache(
                    q=query, k_cache=swa_v,
                    block_table=None, cache_seqlens=None,
                    head_dim_v=Q_DIM, tile_scheduler_metadata=sched,
                    num_splits=None, softmax_scale=Q_DIM ** -0.5,
                    causal=False, is_fp8_kvcache=True, indices=swa_idx,
                    topk_length=swa_len, extra_k_cache=cmp_v,
                    extra_indices_in_kvcache=ext_idx,
                    extra_topk_length=cmp_len, **mla_sink_kw)
                hold["mla"] = res[0] if isinstance(res, tuple) else res
                return res

            def run_mla():
                return run_mla_for(query_ready())
            run_mla()
            stage_fns.append(("mla", run_mla))

            cos_sin_l = torch.cat((cosl, sinl), dim=-1).contiguous()
            pos64 = pos.to(torch.int64).contiguous()
            ows = oproj_ws(B)
            post_b = hcd["hc_post"][:B]
            comb_b = hcd["hc_comb"][:B].view(B, HC, HC)

            def run_oproj():
                mla_o = hold["mla"]
                mla_tp = (mla_o[:, 0] if mla_o.dim() == 4
                          else mla_o).contiguous()
                return o_proj_csa.run_o_proj_mhc_post(
                    mla_tp, pos64, cos_sin_l, hidden, post_b, comb_b,
                    w["o_proj"], ows, mpm, use_pdl=pdl_enabled(),
                    force_pdl=pdl_forced(), run_mhc_post=TPDP_MODE,
                    tpdp_mla_scatter=MLA_DP_MODE)

            def run_mhcpost():
                mpm.mhc_post_out(ows.projected, hidden, post_b, comb_b,
                                 ows.mhc_output)
            run_oproj()
            stage_fns.append(("o_proj", run_oproj))
            if mhc_post_enabled() and not TPDP_MODE:
                run_mhcpost()
                stage_fns.append(("mhc_post", run_mhcpost))

        chain_fns = [f for _, f in stage_fns]

        def chain():
            for f in chain_fns:
                f()

        def capture_tp2_graph(fn):
            torch.cuda.synchronize()
            tp_host_barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            tp_host_barrier()
            return graph

        if pure_dp_baseline_only:
            assert not TPDP_MODE and post_attn_enabled()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                chain(); chain()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = capture_tp2_graph(chain)
            timing = measure_graph_perfetto(
                graph, f"pure_dp_ctx{ncmp * RATIO}", reported_B)
            row = {
                "global_batch": reported_B,
                "local_batch": B,
                "timing": timing,
            }
            communication_study_rows.append(row)
            print(f"  global B={reported_B:<3} local B={B:<3} "
                  f"pure-DP(perfetto mean)={timing['mean_us']:.2f} us "
                  f"[median {timing['median_us']:.2f}, "
                  f"range {timing['min_us']:.2f}..{timing['max_us']:.2f}]")
            del graph, mgr
            torch.cuda.empty_cache()
            continue

        # ---- per-layer COLD measurement ---------------------------------
        def cold_graph_rms_ablation(f, flag, warmup=6, iters=40, reps=9):
            """Paired cold-L2 samples toggling only fused query RMS/RoPE work."""
            for i in range(warmup):
                flag.fill_(i & 1)
                flush_l2()
                f()
            torch.cuda.synchronize()

            rep_work, rep_relay, rep_delta = [], [], []
            for rep in range(reps):
                samples = {0: [], 1: []}
                events = []
                for i in range(iters):
                    order = (1, 0) if ((rep + i) & 1) == 0 else (0, 1)
                    for mode in order:
                        flag.fill_(mode)
                        flush_l2(drain=False)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        f()
                        end.record()
                        events.append((mode, start, end))
                torch.cuda.synchronize()
                for mode, start, end in events:
                    samples[mode].append(start.elapsed_time(end) * 1e3)
                work = sum(samples[1]) / len(samples[1])
                relay = sum(samples[0]) / len(samples[0])
                rep_work.append(work)
                rep_relay.append(relay)
                rep_delta.append(work - relay)

            import statistics
            return (statistics.median(rep_work), statistics.median(rep_relay),
                    statistics.median(rep_delta), min(rep_delta), max(rep_delta))

        # In strict ablation mode the graph is captured below, then the exact
        # same executable is replayed with a device flag. Skip per-stage
        # profiling because only the end-to-end graph span answers this test.
        if Q_RMS_ABLATION:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                q_rms_work_flag.fill_(1)
                chain(); chain()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                chain()
            work, relay, delta, delta_lo, delta_hi = cold_graph_rms_ablation(
                g.replay, q_rms_work_flag)
            print(f"  {B:<5}{work:>12.1f}{relay:>12.1f}{delta:>12.2f}"
                  f"   paired range [{delta_lo:+.2f}, {delta_hi:+.2f}]")
            bw_rows.append((B, [("graph+rms", work), ("graph-no-rms", relay),
                                ("delta", delta)]))
            del g, mgr
            torch.cuda.empty_cache()
            continue

        # Per-stage: bucket Kineto device time over 30 cold eager-chain iters.
        # Each bucket contains only that stage's kernel durations; summing
        # buckets still double-counts PDL and intra-stage overlap.
        ts = measure_eager_stage_chain(stage_fns, chain)
        # (mhc chain-vs-solo diagnostic removed after the verdict: the
        # chain-solo gemm delta matched the hidden+hc_fn L2-refetch
        # bandwidth at every B -- real input refetch, not a timing bug.)

        # (front emit-vs-legacy and wq_b PDL-accounting solo ablations
        # removed after their verdicts landed: emit costs +0.1..1.1us vs
        # legacy; wq_b wall-vs-kineto gap = PDL pair double-count.)

        # ---- End-to-end: exact Perfetto GPU-kernel span -------------------
        t_perfetto = float("nan")
        perfetto_timing = None
        g = None
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):     # allocator warmup off-capture
                chain(); chain()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            tp_host_barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                chain()
        except Exception as err:
            print(f"  (graph capture failed at B={B}: {err})")

        # DeepGEMM-style warmup/active phases. Conditioning kernels are outside
        # the reported graph correlation, so the value remains the exact first
        # to last kernel envelope of one layer graph.
        if g is not None:
            try:
                perfetto_timing = measure_graph_perfetto(
                    g, f"{trace_mode}_ctx{ncmp * RATIO}", B)
                t_perfetto = perfetto_timing["mean_us"]
            except Exception as err:
                print(f"  (graph trace failed at B={B}: {err})")

        if TPDP_MODE:
            rank_times = torch.tensor(ts, device=DEV, dtype=torch.float64)
            dist.all_reduce(rank_times, op=dist.ReduceOp.MAX)
            ts = rank_times.tolist()

        print(f"  {B:<5}" + "".join(f"{t:>9.1f}" for t in ts)
              + f"{sum(ts):>9.1f}{t_perfetto:>9.1f}")
        if perfetto_timing is not None:
            print("       perfetto samples: "
                  f"median={perfetto_timing['median_us']:.2f} us, "
                  f"range={perfetto_timing['min_us']:.2f}.."
                  f"{perfetto_timing['max_us']:.2f} us")
        bw_rows.append((B, [(sname, t) for (sname, _), t
                            in zip(stage_fns, ts)]))
        del mgr
        torch.cuda.empty_cache()

    if pure_dp_baseline_only:
        payload = {
            "schema": "tpdp-communication-study-v1",
            "kind": "pure-dp-b-over-2",
            "context_length": ncmp * RATIO,
            "compressed_context_length": ncmp,
            "l2_flush_mode": "write",
            "l2_flush_bytes": flush_bytes,
            "study_reps": graph_perfetto_samples,
            "graph_perfetto_samples": graph_perfetto_samples,
            "device": torch.cuda.get_device_name(),
            "rows": communication_study_rows,
        }
        if results_json is not None and (not TPDP_MODE or TP_RANK == 0):
            result_dir = os.path.dirname(os.path.abspath(results_json))
            os.makedirs(result_dir, exist_ok=True)
            with open(results_json, "w", encoding="utf-8") as result_file:
                json.dump(payload, result_file, ensure_ascii=False, indent=2)
            print(f"  wrote structured results: {results_json}")
        return

    if Q_RMS_ABLATION:
        return

    # ---- Bandwidth table: per-operator traffic / stage time ------------
    # total = ALL bytes the op moves; internal = intermediate activations
    # (inter-op buffers). END-TO-END-effective = weights + KV-cache pools +
    # boundary activations (hidden) -- the tokens/s-relevant bytes.
    def op_bytes(B):
        MB = 1e6
        B_local = local_batch(B)
        B_mla = mla_batch(B)
        W_FRONT = (2048 * DIM + 2624 * DIM * 2) / MB          # fp8 + bf16
        W_WQB = t_wq.N_MERGED * 1536 / MB
        eff_state_w = B * (1024 + 1024 + 256 + 256) * 4 / MB  # front fresh rows
        idx_bytes = 132 if INDEXER_FP8 else 68
        t = {}   # stage -> (total_MB, internal_MB)
        hid = B * HC * DIM * 2 / MB
        mhc_int = B * (DIM * 2 + DIM + 24 * 4 + 4 * 8 + 16 * 4) / MB
        t["mhc"] = (hid + 1.5 + mhc_int, mhc_int)
        fr_int = (B * (DIM * 2 + DIM)            # collapsed + x8 (re)read
                  + B * 1536 * 2 + B * 512 * 4 + B * 64 * 4) / MB
        t["front"] = (W_FRONT + fr_int + eff_state_w, fr_int)
        wq_int = (B * 1536 * 3                   # xq write + qnorm y read
                  + B * t_wq.N_TOTAL * 2         # y bf16 write (q for mla)
                  + B * 64 * idx_bytes + B * 512 * 4) / MB
        wq_eff = W_WQB + (B * 8 * 512 * 4 + B * (584 + idx_bytes)) / MB
        t["wq_b"] = (wq_eff + wq_int, wq_int)
        qr_int = 2 * B_mla * attention_heads() * Q_DIM * 2 / MB
        t["q_rms"] = (qr_int, qr_int)
        # Attention/logits are request-DP; the fused main-compressor remains
        # replicated over global B so every rank publishes a complete cache.
        mqa_eff = (B_local * ncmp * idx_bytes + B * 8 * 2048 * 4 + B * 584) / MB
        mqa_int = (B_local * ncmp * 4 + B_local * 64 * idx_bytes
                   + B_local * 64 * 4) / MB
        t["mqa"] = (mqa_eff + mqa_int, mqa_int)
        tk_int = (B_local * ncmp * 4 + B_local * TOPK * 4) / MB
        t["topk"] = (tk_int, tk_int)
        H_mla = attention_heads()
        mla_int = 2 * B_mla * H_mla * Q_DIM * 2 / MB  # q read + out
        mla_eff = B_mla * (TOPK + WIN) * 584 / MB
        t["mla"] = (mla_eff + mla_int, mla_int)
        # o_proj: the wo_a/wo_b fp8 weights are the effective traffic; the
        # mla_out re-read and every staged intermediate (o_fp8, z, z_fp8,
        # projected) is internal. mhc_post: residual read + output write are
        # the boundary, its projected read is internal.
        groups = local_o_groups()
        intermediate = groups * O_LORA
        H_op = oproj_heads()
        projected_bytes = 2
        W_OA = groups * O_LORA * (H_op // groups) * Q_DIM / MB
        W_OB = DIM * intermediate / MB
        op_int = (B * H_op * Q_DIM * 2        # mla_out read
                  + 2 * B * H_op * Q_DIM      # o_fp8 write + read
                  + 2 * B * intermediate * 2  # z bf16 write + read
                  + 2 * B * intermediate      # z_fp8 write + read
                  + B * DIM * projected_bytes) / MB
        t["o_proj"] = (W_OA + W_OB + op_int, op_int)
        mp_int = (B * DIM * 2 + B * HC * 4 + B * HC * HC * 4) / MB
        mp_eff = 2 * B * HC * DIM * 2 / MB  # residual read + out write
        t["mhc_post"] = (mp_eff + mp_int, mp_int)
        if TPDP_MODE:
            op_total, op_internal = t["o_proj"]
            post_total, post_internal = t.pop("mhc_post")
            t["o_proj"] = (op_total + post_total,
                           op_internal + post_internal)
        return t

    print("\n" + "=" * 76)
    print(f"Per-operator bandwidth (TB/s) -- stage traffic / stage time; "
          f"(nn%)=internal share")
    print("  effective = weights + KV pools + boundary activations "
          "(intermediates excluded)")
    print("  cold L2: the weight traffic really does come from HBM, so these "
          "are achieved HBM rates")
    print("=" * 76)
    hdr = [s for s, _ in bw_rows[0][1]]
    print(f"  {'B':<5}" + "".join(f"{c + ' ':>13}" for c in hdr)
          + f"{'EFF':>8}")
    for B, stages_ts in bw_rows:
        ob = op_bytes(B)
        cells, eff_b, tot_t = [], 0.0, 0.0
        for sname, us in stages_ts:
            total, internal = ob[sname]
            bw = total / us if us > 0 else 0.0   # MB/us == TB/s
            cells.append(f"{bw:6.2f}({internal / total * 100:3.0f}%)")
            eff_b += total - internal
            tot_t += us
        print(f"  {B:<5}" + "".join(f"{c:>13}" for c in cells)
              + f"{eff_b / tot_t:>8.2f}")


# ==================== simulation ====================
def run_sim(mods, w, B=16, steps=8, seed=42, reuse_at=5):
    torch.manual_seed(seed)
    mgr = KVCacheManager(capacity=B + 4, pages_per_pool=256,
                         max_pages_per_req=8, indexer_fp8=INDEXER_FP8)
    slots = [mgr.alloc_request() for _ in range(B)]
    logits_buf = torch.full((B, 256), float("-inf"), device=DEV)
    stats, finals = {}, None
    dbg = {}
    for t in range(steps):
        if t == reuse_at:                      # slot lifecycle gate
            mgr.free_request(slots[0])
            slots[0] = mgr.alloc_request()     # KV=0 / score=-inf re-init
        hidden = torch.randn(B, HC, DIM, device=DEV,
                             dtype=torch.bfloat16) * 0.1
        if t == 3:      # ALL-ZERO row on a COMPRESS step (pos=3): the fp4
            hidden[1].zero_()   # se-no-clamp path must stay consumer-safe
        finals = decode_step(mods, w, mgr, slots, hidden, logits_buf, stats,
                             dbg=dbg)
    snap = (finals.clone(), mgr.swa_pool.clone(), mgr.cmp_pool.clone(),
            mgr.idx_pool.clone(), mgr.main_state.clone(), mgr.idx_state.clone()) + tuple(
        dbg[k] for k in ("y_q", "win_y2", "yq", "q_ready", "logits",
                         "page_idx", "iq") if k in dbg)
    return stats, snap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tpdp", action="store_true",
        help="WQB TP2, MQA/TopK/MLA DP2, and O-proj TP2; default is full rank")
    parser.add_argument(
        "--indexer-fp4", action="store_true",
        help="use the legacy Wuda FP4 Indexer path (default: RTP-compatible FP8)")
    parser.add_argument(
        "--pdl-mode", choices=("on", "all", "none"), default="on",
        help="on enables the fixed production PDL set; all forces every "
             "E2E-controlled PDL edge; none disables those controlled edges "
             "(FlashMLA internal PDL is unchanged)")
    parser.add_argument(
        "--skip-bench", action="store_true",
        help="run correctness and determinism gates without the latency table")
    parser.add_argument(
        "--bench-only", action="store_true",
        help="skip simulation gates and run the requested latency batches")
    parser.add_argument(
        "--bench-batches", default=None,
        help="comma-separated batch sizes for latency runs (default: full table)")
    parser.add_argument(
        "--context-length", type=int, default=65536,
        help="uncompressed context length for latency runs (default: 65536)")
    parser.add_argument(
        "--q-rms-blocks", type=int, default=1024,
        help="persistent CTA cap for query RMSNorm+RoPE (default: 1024)")
    parser.add_argument(
        "--q-rms-ablation", action="store_true",
        help="paired graph-only ablation of the RMSNorm+RoPE compute body")
    parser.add_argument(
        "--pure-dp-baseline-only", action="store_true",
        help="run the strict no-distributed pure-DP baseline at B/2; "
             "--bench-batches denotes global B (requires --bench-only)")
    parser.add_argument(
        "--separate-query-rms", action="store_true",
        help="use the original standalone query RMSNorm+RoPE kernel")
    parser.add_argument(
        "--results-json", default=None,
        help="write communication-study samples and derived effects as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    INDEXER_FP8 = not args.indexer_fp4
    configure_geometry(args.tpdp)
    init_tpdp_distributed()
    configure_pdl(args.pdl_mode)
    Q_RMS_MAX_BLOCKS = args.q_rms_blocks
    Q_RMS_ABLATION = args.q_rms_ablation
    FUSE_QUERY_RMS = not args.separate_query_rms
    if Q_RMS_ABLATION and not fused_query_rms():
        parser.error("--q-rms-ablation requires the default fused query path")
    if args.pure_dp_baseline_only and (args.tpdp or not args.bench_only):
        parser.error(
            "--pure-dp-baseline-only requires --bench-only without --tpdp")
    context_granularity = RATIO * PAGE
    if (args.context_length <= 0
            or args.context_length % context_granularity != 0):
        parser.error(
            f"--context-length must be a positive multiple of "
            f"{context_granularity}")
    bench_ncmp = args.context_length // RATIO
    bench_batches = (None if args.bench_batches is None else
                     tuple(int(x) for x in args.bench_batches.split(",")))
    print(f"Device: {torch.cuda.get_device_name()}"
          + (f" (TP rank {TP_RANK})" if args.tpdp else ""))
    geometry = ("TPDP local rank: WQB TP2, MQA/TopK/MLA DP2, O-proj TP2"
                if args.tpdp else "full-rank e2e")
    print(f"Geometry: {geometry} main={t_wq.N_TOTAL}, index={t_wq.N_IDX}, "
          f"merged={t_wq.N_MERGED}, "
          f"indexer={'fp8-rtp' if INDEXER_FP8 else 'fp4'}, PDL={PDL_MODE}")
    mqa_test = __import__("test_mqa_logits_fp4")
    get_dg()  # Resolve the checkout before the extension imports headers.
    if args.tpdp:
        # Compile once, then let the peer load the same cached extension.
        if TP_RANK == 0:
            MLA_O_QUANT_MODULE = o_proj_csa.load_mla_o_quant_module()
            TP2_COMM_MODULE = o_proj_csa.load_tp2_comm_module()
        dist.barrier()
        if TP_RANK == 1:
            MLA_O_QUANT_MODULE = o_proj_csa.load_mla_o_quant_module()
            TP2_COMM_MODULE = o_proj_csa.load_tp2_comm_module()
        dist.barrier()
    else:
        MLA_O_QUANT_MODULE = o_proj_csa.load_mla_o_quant_module()
        # The strict pure-DP comparison uses the TP2 (8-group) WoA geometry
        # without symmetric memory, so it still needs the Wuda extension.
        # Ordinary full-rank runs keep the 16-group DeepGEMM reference path
        # and avoid compiling an unused TP2 module.
        if args.pure_dp_baseline_only:
            TP2_COMM_MODULE = o_proj_csa.load_tp2_comm_module()
    mods = (t_hc.load_cuda_module(),
            t_fm_swap.load_module('front_mixed_gemm_csa_swapab',
                                  'front_mixed_gemm_csa_swapab.cu'),
            t_wq.load_module(tpdp=args.tpdp),
            None if fused_query_rms() else load_query_rms_rope(),
            mqa_test.load_cuda_module() if not INDEXER_FP8 else None,
            mqa_test.load_cuda_module_fp8() if INDEXER_FP8 else None,
            __import__("test_topk_v2").load_cuda_module(),
            o_proj_csa.load_mhc_post_module()
            if mhc_post_enabled() else None)
    assert (mods[2].n_main, mods[2].n_index, mods[2].n_merged) == \
        (t_wq.N_TOTAL, t_wq.N_IDX, t_wq.N_MERGED), \
        "Python/kernel WQB geometry mismatch"
    w = make_weights()

    if args.bench_only:
        benchmark(mods, w, ncmp=bench_ncmp, batches=bench_batches,
                  pure_dp_baseline_only=args.pure_dp_baseline_only,
                  results_json=args.results_json)
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)

    sim_name = "TPDP two-rank E2E decode" if args.tpdp else "E2E decode"
    sim_batch = 32 if args.tpdp else 16
    print(f"\n{sim_name} simulation (global B={sim_batch}, 8 steps, "
          "slot reuse at step 5):")
    stats, snap1 = run_sim(mods, w, B=sim_batch)
    _, snap2 = run_sim(mods, w, B=sim_batch)    # determinism gate
    det = True
    for name, a, b in zip(("finals", "swa_pool", "cmp_pool", "idx_pool",
                           "main_state", "idx_state", "y_q", "win_y2",
                           "yq", "q_ready", "logits", "page_idx", "iq"),
                          snap1, snap2):
        # TRUE bitwise compare (byte view): identical NaN bit patterns are
        # EQUAL (torch.equal treats NaN != NaN -> false alarm).
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            det = False
            av, bv = a.float(), b.float()
            neq = (av != bv) & ~(torch.isnan(av) & torch.isnan(bv))
            n = int(neq.sum())
            if n:
                first = neq.nonzero()[0].tolist()
                print(f"  [F-diag] {name}: {n}/{a.numel()} differ, "
                      f"first@{first} "
                      f"run1={av[tuple(first)].item():.6g} "
                      f"run2={bv[tuple(first)].item():.6g}")
            else:
                print(f"  [F-diag] {name}: same values, different NaN bits")
    n_nan = int(torch.isnan(snap1[0].float()).sum())
    if n_nan:   # NaN in finals is a REGRESSION even when deterministic
        det = False
        print(f"  [F-diag] finals carries {n_nan} NaN "
              f"(deterministic, but must be finite)")

    gates = [
        ("A front fp8 seg calc_diff", stats.get("A_front_d8", 1), "< 1e-5",
         stats.get("A_front_d8", 1) < 1e-5),
        ("B wq_b x_fp8 byte match", stats.get("B_xq_match", 0), "> 0.999",
         stats.get("B_xq_match", 0) > 0.999),
        ("Q query rms+rope calc_diff", stats.get("Q_rms_rope_diff", 1),
         "< 1e-5", stats.get("Q_rms_rope_diff", 1) < 1e-5),
        ("C mqa logits max rel err", stats.get("C_mqa_rel", 1), "< 5e-2",
         stats.get("C_mqa_rel", 1) < 5e-2),
        ("D topk set match", stats.get("D_topk_ok", False), "== True",
         stats.get("D_topk_ok", False)),
        ("G zero-row logits finite", stats.get("G_finite", False), "== True",
         stats.get("G_finite", False)),
        ("F run-to-run bitwise/tol", det, "== True", det),
    ]
    if post_attn_enabled():
        gates.insert(4, ("E flashMLA cos vs torch ref",
                         stats.get("E_mla_cos", 0), "> 0.98",
                         stats.get("E_mla_cos", 0) > 0.98))
        # o_proj is fp8-quantized twice (activation + both weight stages), so
        # the official mega_csa unit-test threshold applies here too.
        gates.insert(5, ("H o_proj+mhc_post calc_diff",
                         stats.get("H_oproj_diff", 1), "< 2e-2",
                         stats.get("H_oproj_diff", 1) < 2e-2))
    ok = True
    for name, val, cond, passed in gates:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<28} = {val} ({cond})")

    if ok and not args.skip_bench:  # never bench on broken correctness
        benchmark(mods, w, ncmp=bench_ncmp, batches=bench_batches)

    print("=" * 60)
    result_name = "TPDP TWO-RANK E2E" if args.tpdp else "E2E"
    print(result_name + (" ALL PASS" if ok else " SOME FAILED"))
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0 if ok else 1)
