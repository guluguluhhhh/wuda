"""
test_e2e_decode.py -- END-TO-END DSV4 decode chain over the design-image KV
cache management (test/kv_cache_manager.py):

  hc_fused(nopc + fused attn_norm) -> front_mixed(+hc tail) ->
  qnorm_quant(PDL) + wq_b(ALL fusions: idxpost + head_ssq + indexer compressor
  + winkv) -> mqa_logits(paged, + MAIN compressor tail) -> topk_v2(512, page
  transform) -> flash_mla(SWA pool + compressed pool, fused query_rms_rope,
  per-head attn_sink) ->
  o_proj(inv-RoPE + fp8 -> wo_a -> fp8 -> wo_b) -> mhc_post -> [B,4,7168].

The tail follows the official mega_csa post-attention boundary. o_proj_csa.py
adds TP2 support around the two Triton quant kernels and DeepGEMM wo_a/wo_b;
mhc_post.cu closes the full-rank residual mix. residual is this layer's input,
while post/comb come from the front hc tail.

Multi-step decode simulation (the cache/state semantics ONLY show up across
steps): B requests advance pos together; mid-run one request is freed and its
slot reused (KV=0 / score=-inf re-init gate). P1 discipline: kernels unchanged,
KVCacheManager supplies slot indirection + paged-pool scatter shims.

Stage gates per step:
  A front y (fp8/bf16 segment calc_diff)          D topk vs torch.topk+transform
  B wq_b x_fp8 quant chain (byte match)           E flashMLA vs torch attention
  C mqa logits vs torch ref over DEQUANT pools        over the DEQUANT pools
  H full rank checks o_proj+mhc_post; TPDP checks the FP32 O-proj partial
Global gate: the whole simulation runs TWICE from the same seed -> all caches,
states and final outputs bitwise identical (run-to-run determinism).

flash_mla is optional (import-guarded): without it stages A-D still run, and
the o_proj/mhc_post tail (which consumes the MLA output) is skipped with it.
The default geometry is the full-rank e2e chain. --tpdp benchmarks the mixed
local rank: main-Q TP2 with replicated index-Q, request-DP2 MQA/TopK/FlashMLA,
and the original global-B TP2 O-proj shard (64 heads, 8 groups). Layout handoffs
remain outside this single-process test.
"""
import argparse
import os, sys, math, warnings, torch
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_cache_manager import (KVCacheManager, PAGE, WIN, RATIO,
                              D_NOPE, D_ROPE, TILE, NTILES)

import test_hc_fused_tc as t_hc
import test_front_mixed_gemm as t_fm
import test_wq_b_fp8_gemm as t_wq
import o_proj_csa

def _import_flash_mla():
    """Import flash_mla from the SIBLING source package (identical layout on
    the dev box and B300): dsv4_megakernel/rmsnorm_epilogue_rope_flashmla_v4_
    b300_20260726/FlashMLA. The installed wheel may predate this interface
    (no fused-q kwargs), so the SOURCE tree wins; its compiled backend
    flash_mla.cuda resolves in order:
      1. a .so built inside the source package (pip install -e / build_ext)
      2. artifacts/cuda.cpython-*.so packaged next to it (ABI must match)
      3. fall back to the installed wheel (old interface; stage E then uses
         whatever it offers)."""
    import glob, importlib, importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.abspath(os.path.join(
        here, "..", "..", "rmsnorm_epilogue_rope_flashmla_v4_b300_20260726"))
    src = os.path.join(pkg, "FlashMLA")
    if os.path.isdir(src):
        if not glob.glob(os.path.join(src, "flash_mla", "cuda*.so")):
            # ABI probe: try any packaged backend; a cpython-version mismatch
            # (e.g. artifacts built for 3.10 under a 3.11 env) just skips.
            for cand in glob.glob(os.path.join(pkg, "artifacts", "cuda*.so")):
                try:
                    spec = importlib.util.spec_from_file_location(
                        "flash_mla.cuda", cand)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    sys.modules["flash_mla.cuda"] = mod
                    print(f"[flash_mla] packaged backend: {cand}")
                    break
                except Exception as e:
                    print(f"[flash_mla] artifact {os.path.basename(cand)} "
                          f"unusable ({e})")
        sys.path.insert(0, src)
        try:
            m = importlib.import_module("flash_mla")
            print(f"[flash_mla] source tree: {src}")
            return m
        except Exception as e:   # ABI mismatch etc. -> installed wheel
            print(f"[flash_mla] source import failed ({e}); trying the wheel")
            sys.path.remove(src)
            for k in [k for k in sys.modules if k.startswith("flash_mla")]:
                del sys.modules[k]
    m = importlib.import_module("flash_mla")
    print(f"[flash_mla] installed wheel: {m.__file__}")
    return m


try:
    flash_mla = _import_flash_mla()
    HAS_FLASH_MLA = True
    import inspect
    # Old wheels lack the FUSED query-prologue kwargs; fall back to the
    # standalone query_rms_rope KERNEL (same CUDA op, separate launch).
    FMLA_FUSED_Q = "q_rms_sum_sq" in inspect.signature(
        flash_mla.flash_mla_with_kvcache).parameters
    # OFFICIAL per-head attn_sink (model.py L456 is an nn.Parameter, so it is a
    # CHECKPOINT weight, not an option). Old wheels lack the kwarg; the torch
    # ref then drops it too so stage E stays self-consistent -- but such a run
    # no longer matches official inference.
    FMLA_ATTN_SINK = "attn_sink" in inspect.signature(
        flash_mla.flash_mla_with_kvcache).parameters
    if not FMLA_ATTN_SINK:
        print("[warn] flash_mla has no attn_sink kwarg -- stage E runs the "
              "SINKLESS chain (kernel and ref both), which is NOT official "
              "model.py semantics")
    if not FMLA_FUSED_Q and not hasattr(flash_mla, "query_rms_rope"):
        print("[warn] flash_mla has neither fused-q nor query_rms_rope -- "
              "stage E skipped (build the source package on this box)")
        HAS_FLASH_MLA = False
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


def configure_geometry(tpdp=False):
    global WQ_HEADS, TPDP_MODE, MLA_DP_MODE
    TPDP_MODE = tpdp
    MLA_DP_MODE = tpdp
    t_wq.configure_geometry(tpdp)
    WQ_HEADS = t_wq.N_TOTAL // Q_DIM


def post_attn_enabled():
    return HAS_FLASH_MLA


def mhc_post_enabled():
    """TP O-proj needs the missing reduction before mHC post."""
    return HAS_FLASH_MLA and not TPDP_MODE


def attention_heads():
    return Q_HEADS if MLA_DP_MODE else WQ_HEADS


def oproj_heads():
    return WQ_HEADS


def local_o_groups():
    return O_GROUPS // 2 if TPDP_MODE else O_GROUPS


def local_batch(B):
    return (B + 1) // 2 if TPDP_MODE else B


def mla_batch(B):
    return local_batch(B) if MLA_DP_MODE else B


def mla_dp_output_to_tp(mla3, B):
    """Fabricate this TP rank's global-B head shard after DP MLA.

    The real path redistributes peer request rows. Communication is explicitly
    out of scope here, so peer rows reuse deterministic local results.
    """
    if not MLA_DP_MODE:
        return mla3
    B_local = local_batch(B)
    peer_rows = B - B_local
    local_heads = mla3[:, :WQ_HEADS]
    if not peer_rows:
        return local_heads.contiguous()
    return torch.cat((local_heads, local_heads[:peer_rows]), dim=0).contiguous()


def load_mhc_post():
    """mHC post kernel, ported from mega_csa: its kernel math and PDL protocol
    are byte-for-byte upstream, merged into one .cu with the torch binding
    (this tree has a single consumer, so upstream's cross-TU split is moot)."""
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    return load(
        name="e2e_mhc_post",
        sources=[os.path.join(proj, "kernels", "mhc_post.cu")],
        extra_include_paths=[os.path.join(proj, "include")],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math",
                           f"-gencode=arch=compute_{sm}a,code=sm_{sm}a"],
        verbose=False,
    )


_OPROJ_WS = {}


def oproj_ws(B):
    """Cached O-projection workspace (the run path is allocation-free, and a
    stable workspace is what lets CUDA-graph replay keep its addresses)."""
    key = (B, oproj_heads(), local_o_groups())
    if key not in _OPROJ_WS:
        _OPROJ_WS[key] = o_proj_csa.prepare_o_proj_workspace(
            B, DEV, heads=oproj_heads(), groups=local_o_groups())
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
    wq = torch.randn(t_wq.N_MERGED, 1536, device=DEV) * 0.03
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
        rows = []
        for ct in range(ncmp):
            page = mgr.reqs[s]["idx"][ct // PAGE]
            off = ct % PAGE
            fp4 = mgr.idx_pool[page, : PAGE * 64].view(PAGE, 64)[off]
            sf = mgr.idx_pool[page, PAGE * 64 + off * 4:
                              PAGE * 64 + off * 4 + 4]      # 4 x block-32 e8m0
            scale = torch.pow(2.0, sf[:4].float() - 127.0)
            v = deq_fp4(fp4).view(4, 32) * scale.view(4, 1)
            rows.append(v.view(128))
        out.append(torch.stack(rows) if rows else
                   torch.zeros(0, 128, device=DEV))
    return out


def deq_model1_row(pool, page, off):
    body = pool[page, : PAGE * (D_NOPE + 2 * D_ROPE)] \
        .view(PAGE, D_NOPE + 2 * D_ROPE)[off]
    sf = pool[page, PAGE * (D_NOPE + 2 * D_ROPE):
              PAGE * (D_NOPE + 2 * D_ROPE) + PAGE * 8].view(PAGE, 8)[off]
    nope = body[:D_NOPE].view(torch.float8_e4m3fn).float().view(NTILES, TILE)
    scale = sf[:NTILES].view(torch.float8_e8m0fnu).float()
    rope = body[D_NOPE:].view(torch.bfloat16).float()
    return torch.cat([(nope * scale.view(NTILES, 1)).view(D_NOPE), rope])


def ref_mqa_logits(iq_fp4, iq_sf, weights, kv_rows_list):
    """relu(<q_h, k_t>)·w_h summed over the 64 indexer heads; q dequant via
    test_wq_b's dequant_kernel_iq (iq_sf = 4 packed block-32 ue8m0 / head)."""
    B = iq_fp4.size(0)
    q = t_wq.dequant_kernel_iq(iq_fp4, iq_sf)[0].view(B, IDX_HEADS, IDX_D)
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
    return (torch.einsum("mij,mih->mjh", comb, residual.float())
            + post.unsqueeze(-1) * proj.unsqueeze(1)).to(torch.bfloat16)


# ==================== one decode step ====================
def get_dg():
    """deep_gemm (hard dep for the hybrid mhc): shared resolver in
    bench_utils (env DEEP_GEMM_DIR > installed wheel > sibling checkout),
    PDL enabled once."""
    from bench_utils import get_deep_gemm
    dg = get_deep_gemm()
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
                           xq_out=xq_out, xsf_out=xsf_out)


def decode_step(mods, w, mgr, slots, hidden, logits_buf, stats, dbg=None):
    hcm, fmm, wqm, mqm, tkm, mpm = mods
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
    y = fmm.front_mixed_gemm(collapsed, x_fp8, x_sf, w["front_bf16"],
                             w["front_fp8"], w["front_sf"], **hc,
                             main_kv=mgr.main_kv, main_sc=mgr.main_sc,
                             main_ape=w["main_ape"],
                             state_row=mgr.main_state_rows(slots, pos),
                             ape_phase=(pos % 4).int(),
                             idx_kv=mgr.idx_kv, idx_sc=mgr.idx_sc,
                             idx_ape=w["idx_ape"],
                             win_y2=win_y2, w64=w64)
    ref8 = (t_fm.dequant_fp8(x_fp8, x_sf, True)
            @ t_fm.dequant_fp8(w["front_fp8"], w["front_sf"], False).t())
    d8 = max(t_fm.calc_diff(y[:, :1536].float(), ref8[:, :1536]),
             t_fm.calc_diff(win_y2, ref8[:, 1536:]))
    stats["A_front_d8"] = max(stats.get("A_front_d8", 0.0), d8)

    # -- 3. wq_b ALL fusions; the KERNEL writes idx state pool (slot_map),
    #       the indexer fused pages AND the SWA MODEL1 pages directly --------
    xq = torch.empty(B, 1536, device=DEV, dtype=torch.float8_e4m3fn)
    xq_sf = torch.empty(B, 12, device=DEV, dtype=torch.uint8)
    ssq = torch.zeros(B, WQ_HEADS, device=DEV, dtype=torch.float32)
    rets = wqm.wq_b_proj_gemm_merged(
        xq, t_wq.as_ue8m0(xq_sf), w["wq_fp8"], w["wq_sf"], q_pos,
        w["cos"], w["sin"], head_ssq=ssq, mock_post=False,
        cmp_pos=pos,
        idx_norm=w["idx_norm"],
        cos_tab=w["cos"], sin_tab=w["sin"],
        idx_kv=mgr.idx_kv, idx_sc=mgr.idx_sc,
        win_y2=win_y2, win_norm=w["win_norm"],
        q_y=y[:, :1536], q_norm_w=w["q_norm"],
        slot_map=st["slot_map"],
        idx_cache=mgr.idx_pool, idx_dst=st["idx_dst"],
        swa_cache=mgr.swa_pool, swa_dst=st["swa_dst"])
    yq, iq_fp4, iq_sf = rets[0], rets[1], rets[2]
    qr, sr = t_wq.ref_qnorm_quant(y[:, :1536], w["q_norm"])
    stats["B_xq_match"] = min(stats.get("B_xq_match", 1.0),
                              (xq.view(torch.uint8) == qr.view(torch.uint8))
                              .float().mean().item())

    # -- 4. MAIN state: published by the FRONT-EMIT epilogue above (fp32
    #       accum + ape[pos%4] on the score half; model.py L332 contract) --

    # -- 5. mqa_logits (paged idx pool) + MAIN compressor tail: the KERNEL
    #       reads the state POOL via slot_map and writes the Main-compressed
    #       MODEL1 pages directly ------------------------------------------
    ncmp = mgr.n_compressed(slots, pos)
    weights64 = w64
    logits_buf.fill_(float("-inf"))
    # One process models one DP rank: attention and TopK own only these rows.
    # The fused main-compressor still receives the unsliced global-B tensors so
    # each rank updates its complete replicated MODEL1 cache.
    B_local = local_batch(B)
    owned = slice(0, B_local)
    # PRODUCTION form: compact comp_q8/s8/rope OMITTED (cache mode writes the
    # MODEL1 pages directly; compact would double-write ~600B/row).
    mqm.mqa_logits_fp4_decode_out(
        iq_fp4[owned], iq_sf[owned], mgr.idx_pool, weights64[owned], ncmp[owned],
        mgr.block_table("idx", slots)[owned], logits_buf[owned],
        cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=w["cos"],
        sin_tab=w["sin"], comp_kv=mgr.main_kv, comp_sc=mgr.main_sc,
        slot_map=st["slot_map"], cmp_cache=mgr.cmp_pool,
        cmp_dst=st["cmp_dst"])
    # zero-row robustness: the fp4 chain's se has NO clamp (amax=0 -> sf byte
    # 0, codes 0); assert the consumer stays finite (no NaN/+inf in logits).
    torch.cuda.synchronize()
    lg = logits_buf[owned, : max(int(ncmp[owned].max()), 1)]
    stats["G_finite"] = stats.get("G_finite", True) and \
        bool((torch.isnan(lg) | (lg == float("inf"))).sum().item() == 0)

    kv_rows = deq_idx_pool(mgr, slots, pos)
    refs = ref_mqa_logits(iq_fp4, iq_sf, weights64, kv_rows)
    for b in range(B_local):
        n = int(ncmp[b])
        if n:
            d = (logits_buf[b, :n].float() - refs[b]).abs().max().item()
            rel = d / (refs[b].abs().max().item() + 1e-6)
            stats["C_mqa_rel"] = max(stats.get("C_mqa_rel", 0.0), rel)

    # -- 6. topk_v2 (512, page transform -> compressed physical) ----------
    # scores: COLUMN SLICE of the wide buffer (row stride 256 % 4 == 0 keeps
    # the 16B-vector-load contract; a .contiguous() copy would break it for
    # tiny ncmp). -inf tail beyond seq_len is never read.
    cmp_bt = mgr.block_table("cmp", slots)[owned]
    L = max(int(ncmp[owned].max()), 1)
    page_idx = tkm.topk_v2(logits_buf[owned, :L], ncmp[owned], cmp_bt,
                           TOPK, PAGE, None)
    for b in range(B_local):
        n = int(ncmp[b])
        k = min(n, TOPK)
        if k == 0:
            ok = bool((page_idx[b] == -1).all())
        else:
            ref_raw = torch.topk(refs[b], k).indices if n > TOPK else \
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

    # -- 7. flashMLA (SWA pool + compressed pool, fused q rms+rope) -------
    if post_attn_enabled():
        B_mla = mla_batch(B)
        if MLA_DP_MODE:
            # Simulate the peer TP head shard without running a collective.
            q_shard = yq.view(B, 1, WQ_HEADS, Q_DIM)[owned]
            ssq_shard = ssq.view(B, 1, WQ_HEADS)[owned]
            q_raw = torch.cat((q_shard, q_shard), dim=2).contiguous()
            sum_sq = torch.cat((ssq_shard, ssq_shard), dim=2).contiguous()
        else:
            q_raw = yq.view(B, 1, WQ_HEADS, Q_DIM)
            sum_sq = ssq.view(B, 1, WQ_HEADS)
        pos_mla = pos[owned] if MLA_DP_MODE else pos
        slots_mla = slots[:B_mla] if MLA_DP_MODE else slots
        rc = w["cos"][pos_mla].view(B_mla, 1, 32).contiguous()
        rs = w["sin"][pos_mla].view(B_mla, 1, 32).contiguous()
        swa_idx, swa_len = mgr.swa_indices(slots, pos, SWA_TOPK)
        if MLA_DP_MODE:
            swa_idx, swa_len = swa_idx[owned], swa_len[owned]
        cmp_len = torch.minimum(ncmp[owned] if MLA_DP_MODE else ncmp,
                                torch.tensor(TOPK, device=DEV)).int()
        # Fresh sched_meta EVERY step: it is only reusable while shapes AND
        # cache_seqlens/topk_length values stay identical (interface doc);
        # our lens advance each step.
        sched, _ = flash_mla.get_mla_metadata()
        if FMLA_FUSED_Q:
            q_in, fused_q = q_raw, dict(q_rms_sum_sq=sum_sq,
                                        q_rope_cos=rc, q_rope_sin=rs)
        else:   # old wheel: standalone query_rms_rope kernel, plain q after
            q_in, fused_q = flash_mla.query_rms_rope(q_raw, sum_sq, rc, rs,
                                                     EPS), {}
        # attn_sink is orthogonal to the fused-q prologue: it only rescales the
        # softmax denominator, so it applies to both wheel generations.
        sink = w["attn_sink"] if FMLA_ATTN_SINK else None
        sink_kw = {} if sink is None else dict(attn_sink=sink)
        res = flash_mla.flash_mla_with_kvcache(
            q=q_in, k_cache=mgr.model1_cache_view("swa"),
            block_table=None, cache_seqlens=None, head_dim_v=Q_DIM,
            tile_scheduler_metadata=sched, num_splits=None,
            softmax_scale=Q_DIM ** -0.5, causal=False, is_fp8_kvcache=True,
            indices=swa_idx, topk_length=swa_len,
            extra_k_cache=mgr.model1_cache_view("cmp"),
            extra_indices_in_kvcache=page_idx_full.view(B_mla, 1, TOPK),
            extra_topk_length=cmp_len,
            **fused_q, **sink_kw)
        out = res[0] if isinstance(res, tuple) else res
        # torch reference over the DEQUANT pools
        for b in range(B_mla):
            p = int(pos_mla[b])
            qn = (q_raw[b, 0].float()
                  * torch.rsqrt(sum_sq[b, 0].view(-1, 1) / Q_DIM + EPS))
            e, o = qn[:, 448::2].clone(), qn[:, 449::2].clone()
            c, s = rc[b, 0], rs[b, 0]
            qn[:, 448::2] = e * c - o * s
            qn[:, 449::2] = e * s + o * c
            table = mgr.reqs[slots_mla[b]]["swa"]
            n_sw = int(swa_len[b])
            sw = torch.stack([deq_model1_row(mgr.swa_pool,
                                             table[(t % WIN) // PAGE],
                                             (t % WIN) % PAGE)
                              for t in range(p + 1 - n_sw, p + 1)]) \
                if n_sw else torch.zeros(0, Q_DIM, device=DEV)
            cm = torch.stack([deq_model1_row(mgr.cmp_pool, int(pi) // PAGE,
                                             int(pi) % PAGE)
                              for pi in page_idx_full[b][:int(cmp_len[b])]]) \
                if int(cmp_len[b]) else torch.zeros(0, Q_DIM, device=DEV)
            ref = ref_flash_attn(qn.bfloat16().float(), sw, cm, Q_DIM ** -0.5,
                                 sink)
            if torch.isnan(out[b]).any():   # NaN source diag (F regression)
                print(f"  [NaN-diag] b={b} pos={p} "
                      f"out={int(torch.isnan(out[b]).sum())} "
                      f"q={int(torch.isnan(qn).sum())} "
                      f"ssq0={int((sum_sq[b, 0] == 0).sum())} "
                      f"sw={int(torch.isnan(sw).sum())}/{int(swa_len[b])} "
                      f"cm={int(torch.isnan(cm).sum())}/{int(cmp_len[b])} "
                      f"ref={int(torch.isnan(ref).sum())}")
            cos = torch.nn.functional.cosine_similarity(
                out[b, 0].float().flatten(), ref.flatten(), dim=0).item()
            stats["E_mla_cos"] = min(stats.get("E_mla_cos", 1.0), cos)
        if dbg is not None:   # F-bisect: last-step producer snapshots
            dbg.update(y_q=y[:, :1536].clone(), win_y2=win_y2.clone(),
                       yq=yq.clone(), ssq=ssq.clone(),
                       logits=logits_buf.clone(), page_idx=page_idx_full.clone(),
                       iq=rets[1].clone())

        # -- 8. TP2 O projection stops at this rank's FP32 partial. ----------
        mla3_dp = (out[:, 0] if out.dim() == 4 else out).contiguous()
        mla3 = mla_dp_output_to_tp(mla3_dp, B)
        pos64 = pos.to(torch.int64).contiguous()
        post_t, comb_t = hc["hc_post"], hc["hc_comb"].view(B, HC, HC)
        run_post = not TPDP_MODE
        projected = o_proj_csa.run_o_proj_mhc_post(
            mla3, pos64, w["cos_sin"], hidden, post_t, comb_t,
            w["o_proj"], oproj_ws(B), mpm, use_pdl=True,
            run_mhc_post=run_post)
        ref_final = ref_o_proj_mhc(mla3, pos64, w["cos_sin"], hidden,
                                   post_t, comb_t, w["wo_a"], w["wo_b"],
                                   run_mhc_post=run_post)
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
    fn(); torch.cuda.synchronize()
    try:
        ctx = profile(activities=[ProfilerActivity.CUDA], acc_events=True)
    except TypeError:
        ctx = profile(activities=[ProfilerActivity.CUDA])
    with ctx as prof:
        fn(); torch.cuda.synchronize()
    names = []
    for e in prof.events():
        n = e.name
        if any(s in n for s in ("Memset", "memset", "fill", "Memcpy",
                                "zero_", "elementwise", "reduce_kernel")):
            continue
        d = getattr(e, "device_time", None)
        if d is None:
            d = getattr(e, "cuda_time", 0.0)
        if d and d > 0 and n[:80] not in names:
            names.append(n[:80])
    assert names, "probe found no kernels"
    return tuple(names)


def benchmark(mods, w, ncmp=2048):
    """COLD-L2 measurement, ONE full CSA layer per call: an 8GB memset (the
    bench_utils.bench_kineto flusher, i.e. what every single-operator test in
    this tree already uses) runs before EVERY chain call, so a layer starts
    with none of its own weights in L2. That is the real decode situation --
    consecutive layers hold DIFFERENT weights, so layer N never finds layer
    N-1's 113MB wq_b / 78MB front / 130MB o_proj streams cached, and a hot
    no-flush number would flatter every weight-bound stage.
      per-stage : kineto device time over R cold chain iters, kernels
                  bucketed by stage (wq_b bucket = qnorm + merged GEMM;
                  device-sum slightly double-counts their PDL overlap)
      graph     : DEVICE span/step of CUDA-graph replay, via CUDA events
                  (vLLM serving form; THE end-to-end number). Events and
                  not a wall on purpose: per-call host launch+sync adds
                  ~17us at B=128 that real serving never pays PER LAYER,
                  because a whole model is ONE graph -- one launch and one
                  sync for all 61 layers, not one each. Cross-checked
                  against the perfetto trace's replay span at B=128.
    The flush is ONE per layer (never per stage) and is EXCLUDED from every
    reported number: the kineto buckets name only the chain's own kernels,
    and the graph timing window is opened stream-ordered behind the flush.
    Context ops (mqa/topk/mla) at a FABRICATED long context: page tables
    and lens real, cache bytes arbitrary -- latency-neutral. The chain runs
    through O-proj; full rank also times mHC post, while TPDP stops before its
    missing partial reduction. The ported O-proj entry point performs host-side
    validation, which does not affect graph replay or Kineto device time."""
    hcm, fmm, wqm, mqm, tkm, mpm = mods
    print("\n" + "=" * 76)
    print(f"Per-operator latency (us) -- compress-row step, ncmp={ncmp} "
          f"compressed tokens ({ncmp * RATIO} ctx)")
    print("=" * 76)
    S = 4 * ncmp + 8
    ang = torch.rand(S, 32, device=DEV) * 6.28
    cosl, sinl = ang.cos().contiguous(), ang.sin().contiguous()
    cols = ["mhc", "front", "wq_b", "mqa", "topk"] + \
        (["mla", "o_proj"] if post_attn_enabled() else []) + \
        (["mhc_post"] if mhc_post_enabled() else []) + \
        ["stages", "graph"]
    print("  cold L2 (8GB memset per layer call, excluded from every number);"
          " stages+graph = device us")
    print("  wq_b = qnorm producer + merged GEMM device-time sum; standalone "
          "fused(us) reports only the merged GEMM")
    if TPDP_MODE:
        print("  --tpdp: MQA/TopK/MLA run ceil(B/2) rows; MLA H=128; O-proj "
              "remains global B x H=64 with the 8-group TP2 shard")
        print("  Both TP/DP layout handoffs are fabricated outside timing; "
              "communication is not measured")
    print(f"  {'B':<5}" + "".join(f"{c:>9}" for c in cols))
    print("  " + "-" * (5 + 9 * len(cols)))
    bw_rows = []    # (B, [(stage, us)]) for the bandwidth table
    # L2 flusher, allocated ONCE: keeps the per-B measured regions
    # allocation-free and keeps an 8GB buffer out of any CUDA-graph pool.
    flush_buf = torch.empty(int(8e9 // 4), dtype=torch.int, device=DEV)

    def flush_l2(drain=True):
        """Evict L2.
        drain=True (wall path): also DRAIN. The drain is not optional there:
          the chain's first op (deep_gemm's mhc GEMM) launches with PDL, and
          without it that GEMM would overlap the memset tail.
        drain=False (event path): leave the memset IN FLIGHT and enqueue
          behind it. An event recorded next is still stream-ordered AFTER the
          flush, so the flush stays out of the window, but the device stays
          busy ~1ms while the host submits the replay -- otherwise the host's
          submit latency shows up as device IDLE inside the window (~10us for
          this 16-node graph). L2 is still cold for the data: a PSS consumer
          may run its prologue (barrier init / descriptor prefetch) early,
          but cudaGridDependencySynchronize blocks its loads until the
          non-signalling memset has actually completed."""
        flush_buf.zero_()
        if drain:
            torch.cuda.synchronize()

    for B in (2, 16, 32, 48, 64, 80, 96, 112, 128):
        mgr = KVCacheManager(capacity=B + 2, pages_per_pool=(ncmp // PAGE) * B
                             + 4 * B, max_pages_per_req=ncmp // PAGE)
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
        phys_row = ((4 * ((p0 >> 2) & 1)) + 4 + (p0 & 3)) & 7
        state_row = torch.full((Bp,), B * 8, dtype=torch.int32, device=DEV)
        state_row[:B] = torch.tensor([s * 8 + phys_row for s in slots],
                                     dtype=torch.int32, device=DEV)
        ape_phase = torch.full((Bp,), p0 % 4, dtype=torch.int32, device=DEV)

        def run_front():
            if Bp != B:
                coll_p[:B].copy_(collapsed)
                mix_p[:B].copy_(mix)
            fmm.front_mixed_gemm(coll_p, x8, x8s, w["front_bf16"],
                                 w["front_fp8"], w["front_sf"], out=y_p,
                                 **hcd, main_kv=mgr.main_kv,
                                 main_sc=mgr.main_sc,
                                 main_ape=w["main_ape"],
                                 state_row=state_row, ape_phase=ape_phase,
                                 idx_kv=mgr.idx_kv, idx_sc=mgr.idx_sc,
                                 idx_ape=w["idx_ape"],
                                 win_y2=win_y2_p, w64=w64_p)
        run_hc(); run_front()

        xq = torch.empty(B, 1536, device=DEV, dtype=torch.float8_e4m3fn)
        xq_sf = t_wq.as_ue8m0(torch.empty(B, 12, device=DEV,
                                          dtype=torch.uint8))
        ssq = torch.zeros(B, WQ_HEADS, device=DEV, dtype=torch.float32)

        def run_wqb():
            hold["r"] = wqm.wq_b_proj_gemm_merged(
                xq, xq_sf, w["wq_fp8"], w["wq_sf"], q_pos, cosl, sinl,
                head_ssq=ssq, mock_post=False, cmp_pos=pos,
                idx_norm=w["idx_norm"], cos_tab=cosl,
                sin_tab=sinl, idx_kv=mgr.idx_kv, idx_sc=mgr.idx_sc,
                win_y2=win_y2, win_norm=w["win_norm"],
                q_y=y[:, :1536], q_norm_w=w["q_norm"],
                slot_map=st["slot_map"],
                idx_cache=mgr.idx_pool, idx_dst=st["idx_dst"],
                swa_cache=mgr.swa_pool, swa_dst=st["swa_dst"])
        run_wqb()
        nc = mgr.n_compressed(slots, pos)
        B_local = local_batch(B)
        owned = slice(0, B_local)
        idx_bt = mgr.block_table("idx", slots)[owned]
        logits = torch.full((B, (ncmp + 255) // 256 * 256), float("-inf"),
                            device=DEV)

        # PRODUCTION form: compact comp outputs omitted (cache direct write).
        def run_mqa():
            r = hold["r"]
            mqm.mqa_logits_fp4_decode_out(
                r[1][owned], r[2][owned], mgr.idx_pool, weights64[owned],
                nc[owned], idx_bt, logits[owned],
                cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=cosl,
                sin_tab=sinl, comp_kv=mgr.main_kv, comp_sc=mgr.main_sc,
                slot_map=st["slot_map"], cmp_cache=mgr.cmp_pool,
                cmp_dst=st["cmp_dst"])
        run_mqa()
        cmp_bt_full = mgr.block_table("cmp", slots)
        cmp_bt = cmp_bt_full[owned]
        logical = torch.arange(TOPK, dtype=torch.int64, device=DEV)
        page_idx_full = (cmp_bt_full[:, logical // PAGE] * PAGE
                         + (logical % PAGE).int()).contiguous()
        page_idx = page_idx_full[owned]
        meta = torch.zeros(B_local + 1, 2, dtype=torch.int32, device=DEV)
        run_topk = lambda: tkm.topk_v2_transform(
            logits[owned, :ncmp], nc[owned], cmp_bt, page_idx, PAGE, meta, None)
        run_topk()

        stage_fns = [("mhc", run_hc), ("front", run_front),
                     ("wq_b", run_wqb),
                     ("mqa", run_mqa), ("topk", run_topk)]
        if post_attn_enabled():
            B_mla = mla_batch(B)
            pos_mla = pos[owned] if MLA_DP_MODE else pos
            rc = cosl[pos_mla].view(B_mla, 1, 32).contiguous()
            rs = sinl[pos_mla].view(B_mla, 1, 32).contiguous()
            swa_idx, swa_len = mgr.swa_indices(slots, pos, SWA_TOPK)
            if MLA_DP_MODE:
                swa_idx, swa_len = swa_idx[owned], swa_len[owned]
                q_shard = hold["r"][0].view(B, 1, WQ_HEADS, Q_DIM)[owned]
                ssq_shard = ssq.view(B, 1, WQ_HEADS)[owned]
                # Receive buffers after the omitted TP->DP all-to-all. Peer
                # values do not affect kernel timing, so duplicate local data.
                q_external = torch.cat((q_shard, q_shard), dim=2).contiguous()
                ssq_external = torch.cat(
                    (ssq_shard, ssq_shard), dim=2).contiguous()
                ext_idx = page_idx.view(B_mla, 1, TOPK)
                cmp_len = torch.minimum(
                    nc[owned], torch.tensor(TOPK, device=DEV)).int()
            else:
                q_external = ssq_external = None
                ext_idx = page_idx_full.view(B, 1, TOPK)
                cmp_len = torch.minimum(
                    nc, torch.tensor(TOPK, device=DEV)).int()
            swa_v, cmp_v = mgr.model1_cache_view("swa"), mgr.model1_cache_view("cmp")
            sched, _ = flash_mla.get_mla_metadata()
            mla_sink_kw = ({} if not FMLA_ATTN_SINK
                           else dict(attn_sink=w["attn_sink"]))

            def run_mla():
                q_raw = (q_external if MLA_DP_MODE else
                         hold["r"][0].view(B, 1, WQ_HEADS, Q_DIM))
                sum_sq = (ssq_external if MLA_DP_MODE else
                          ssq.view(B, 1, WQ_HEADS))
                if FMLA_FUSED_Q:
                    qi, fq = q_raw, dict(q_rms_sum_sq=sum_sq,
                                         q_rope_cos=rc, q_rope_sin=rs)
                else:
                    qi, fq = flash_mla.query_rms_rope(q_raw, sum_sq, rc, rs,
                                                      EPS), {}
                res = flash_mla.flash_mla_with_kvcache(
                    q=qi, k_cache=swa_v, block_table=None, cache_seqlens=None,
                    head_dim_v=Q_DIM, tile_scheduler_metadata=sched,
                    num_splits=None, softmax_scale=Q_DIM ** -0.5,
                    causal=False, is_fp8_kvcache=True, indices=swa_idx,
                    topk_length=swa_len, extra_k_cache=cmp_v,
                    extra_indices_in_kvcache=ext_idx,
                    extra_topk_length=cmp_len, **fq, **mla_sink_kw)
                hold["mla"] = res[0] if isinstance(res, tuple) else res
                return res
            run_mla()
            stage_fns.append(("mla", run_mla))

            # O-proj remains TP2. In MLA-DP mode, fabricate its post-handoff
            # global-B/64-head input once, outside every timed region.
            cos_sin_l = torch.cat((cosl, sinl), dim=-1).contiguous()
            pos64 = pos.to(torch.int64).contiguous()
            ows = oproj_ws(B)
            post_b = hcd["hc_post"][:B]
            comb_b = hcd["hc_comb"][:B].view(B, HC, HC)
            mla_o = hold["mla"]
            mla3 = (mla_o[:, 0] if mla_o.dim() == 4 else mla_o).contiguous()
            mla_tp_external = mla_dp_output_to_tp(mla3, B)

            def run_oproj():
                if MLA_DP_MODE:
                    mla_tp = mla_tp_external
                else:
                    mla_o = hold["mla"]
                    mla_tp = (mla_o[:, 0] if mla_o.dim() == 4
                              else mla_o).contiguous()
                return o_proj_csa.run_o_proj_mhc_post(
                    mla_tp, pos64, cos_sin_l, hidden, post_b, comb_b,
                    w["o_proj"], ows, mpm, use_pdl=True, run_mhc_post=False)

            def run_mhcpost():
                mpm.mhc_post_out(ows.projected, hidden, post_b, comb_b,
                                 ows.mhc_output, True)
            run_oproj()
            stage_fns.append(("o_proj", run_oproj))
            if mhc_post_enabled():
                run_mhcpost()
                stage_fns.append(("mhc_post", run_mhcpost))

        chain_fns = [f for _, f in stage_fns]

        def chain():
            for f in chain_fns:
                f()

        # ---- per-layer COLD measurement ---------------------------------
        def cold_graph_step(f, warmup=5, iters=20, reps=3):
            """CUDA-event latency with a cold L2 on every graph replay."""
            for _ in range(warmup):
                flush_l2()
                f()
            torch.cuda.synchronize()
            evs = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
                   for _ in range(iters)]
            best = float("inf")
            for _ in range(reps):
                for i in range(iters):
                    flush_l2(drain=False)
                    evs[i][0].record()
                    f()
                    evs[i][1].record()
                torch.cuda.synchronize()
                tot = sum(a.elapsed_time(b) for a, b in evs) / 1e3
                best = min(best, tot / iters * 1e6)
            return best

        # per-stage: bucket kineto device time of a COLD R-iter chain loop
        # (strict probe everywhere: quant/glue are single named Triton
        # kernels now; mla's q bf16 cast stays unbucketed -> walls only)
        name2stage = {}
        for sname, f in stage_fns:
            for n in probe_kernel_names(f):
                name2stage.setdefault(n, sname)
        from torch.profiler import profile as _prof, ProfilerActivity
        R = 30      # 30-iter stage means (n=10 was too jittery)
        for _ in range(3):
            flush_l2()
            chain()
        torch.cuda.synchronize()
        try:
            pctx = _prof(activities=[ProfilerActivity.CUDA], acc_events=True)
        except TypeError:
            pctx = _prof(activities=[ProfilerActivity.CUDA])
        with pctx as prof:
            for _ in range(R):
                # cold L2 for every layer iteration. The memset is never
                # bucketed: name2stage holds only the chain's own kernel
                # names, and probe_kernel_names filters memset/zero_ anyway.
                flush_l2()
                chain()
                # STEPPED stage semantics (aligned with the vLLM baseline's
                # per-decode-step windows): drain between iterations so every
                # step's first op starts on a quiet GPU -- the vLLM engine
                # has host scheduling gaps there. Without this, the previous
                # iteration's MLA tail folds into the mHC GEMM's PDL wait. The
                # graph envelope remains gapless.
                torch.cuda.synchronize()
        acc = {sname: 0.0 for sname, _ in stage_fns}
        for e in prof.events():
            sname = name2stage.get(e.name[:80])
            if sname is None:
                continue
            d = getattr(e, "device_time", None)
            if d is None:
                d = getattr(e, "cuda_time", 0.0)
            acc[sname] += d
        ts = [acc[sname] / R for sname, _ in stage_fns]
        # (mhc chain-vs-solo diagnostic removed after the verdict: the
        # chain-solo gemm delta matched the hidden+hc_fn L2-refetch
        # bandwidth at every B -- real input refetch, not a timing bug.)

        # (front emit-vs-legacy and wq_b PDL-accounting solo ablations
        # removed after their verdicts landed: emit costs +0.1..1.1us vs
        # legacy; wq_b wall-vs-kineto gap = PDL pair double-count.)

        # ---- CUDA-graph envelope: the whole chain in ONE replay ----------
        t_graph = float("nan")
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):     # allocator warmup off-capture
                chain(); chain()
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                chain()
            t_graph = cold_graph_step(g.replay)
        except Exception as err:
            print(f"  (graph capture failed at B={B}: {err})")

        # ---- perfetto timeline: one cold-L2 graph replay only. ------------
        if B in (16, 128) and not math.isnan(t_graph):
            flush_l2()
            with _prof(activities=[ProfilerActivity.CPU,
                                   ProfilerActivity.CUDA]) as prof:
                g.replay()
                torch.cuda.synchronize()
            tp = f"/tmp/e2e_trace_{os.getpid()}_B{B}.json"
            prof.export_chrome_trace(tp)
            del prof

        print(f"  {B:<5}" + "".join(f"{t:>9.1f}" for t in ts)
              + f"{sum(ts):>9.1f}{t_graph:>9.1f}")
        bw_rows.append((B, [(sname, t) for (sname, _), t
                            in zip(stage_fns, ts)]))
        del mgr
        torch.cuda.empty_cache()

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
        t = {}   # stage -> (total_MB, internal_MB)
        hid = B * HC * DIM * 2 / MB
        mhc_int = B * (DIM * 2 + DIM + 24 * 4 + 4 * 8 + 16 * 4) / MB
        t["mhc"] = (hid + 1.5 + mhc_int, mhc_int)
        fr_int = (B * (DIM * 2 + DIM)            # collapsed + x8 (re)read
                  + B * 1536 * 2 + B * 512 * 4 + B * 64 * 4) / MB
        t["front"] = (W_FRONT + fr_int + eff_state_w, fr_int)
        wq_int = (B * 1536 * 3                   # xq write + qnorm y read
                  + B * t_wq.N_TOTAL * 2         # y bf16 write (q for mla)
                  + B * 64 * 68 + B * 512 * 4) / MB
        wq_eff = W_WQB + (B * 8 * 512 * 4 + B * (584 + 68)) / MB
        t["wq_b"] = (wq_eff + wq_int, wq_int)
        # Attention/logits are request-DP; the fused main-compressor remains
        # replicated over global B so every rank publishes a complete cache.
        mqa_eff = (B_local * ncmp * 68 + B * 8 * 2048 * 4 + B * 584) / MB
        mqa_int = (B_local * ncmp * 4 + B_local * 64 * 68
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
        projected_bytes = 4 if TPDP_MODE else 2
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
                         max_pages_per_req=8)
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
            mgr.idx_pool.clone(), mgr.main_kv.clone(), mgr.idx_kv.clone(),
            mgr.main_sc.clone()) + tuple(
        dbg[k] for k in ("y_q", "win_y2", "yq", "ssq", "logits",
                         "page_idx", "iq") if k in dbg)
    return stats, snap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tpdp", action="store_true",
        help="WQB TP2, MQA/TopK/MLA DP2, and O-proj TP2; default is full rank")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    configure_geometry(args.tpdp)
    print(f"Device: {torch.cuda.get_device_name()}")
    geometry = ("TPDP local rank: WQB TP2, MQA/TopK/MLA DP2, O-proj TP2"
                if args.tpdp else "full-rank e2e")
    print(f"Geometry: {geometry} main={t_wq.N_TOTAL}, index={t_wq.N_IDX}, "
          f"merged={t_wq.N_MERGED}")
    mods = (t_hc.load_cuda_module(), t_fm.load_module(),
            t_wq.load_module(tpdp=args.tpdp),
            __import__("test_mqa_logits_fp4").load_cuda_module(),
            __import__("test_topk_v2").load_cuda_module(),
            load_mhc_post() if mhc_post_enabled() else None)
    assert (mods[2].n_main, mods[2].n_index, mods[2].n_merged) == \
        (t_wq.N_TOTAL, t_wq.N_IDX, t_wq.N_MERGED), \
        "Python/kernel WQB geometry mismatch"
    w = make_weights()

    sim_name = "TPDP MLA-DP local-rank" if args.tpdp else "E2E decode"
    print(f"\n{sim_name} simulation (global B=16, 8 steps, slot reuse at step 5):")
    stats, snap1 = run_sim(mods, w)
    _, snap2 = run_sim(mods, w)                # determinism gate
    det = True
    # head_ssq is fp32-RED accumulated (arrival order varies run-to-run by
    # design; the deterministic rewrite measured too slow and was rejected).
    # Its downstream (finals) may swing ONE quant step on a boundary hit.
    # Those two get physics-based tolerances; everything else stays BITWISE
    # (not downstream of any atomic -> any diff there is a real bug).
    def _lax_ok(name, av, bv):
        if name == "ssq":   # ulp-level jitter on every element
            rel = ((av - bv).abs() /
                   av.abs().clamp_min(1e-6)).max().item()
            return rel < 1e-5, f"elem rel {rel:.2e} (<1e-5)"
        # finals: one ssq ulp can flip a SINGLE fp8 code in o_proj's quant,
        # and wo_b [7168,16384] + mhc_post's post*proj then spread that code
        # over the WHOLE request (every HC scope) -- so the differing-ELEMENT
        # fraction is bimodal (0 or ~1/B) and can never sit just under a
        # small bound (measured on B300: 0 or 2.56e-2, nothing between).
        # Gate the contaminated-REQUEST count instead: boundary hits touch a
        # minority, a real race touches all of them.
        B = av.size(0)
        nreq = int((av != bv).flatten(1).any(-1).sum())
        rel = ((av - bv).abs().max() /
               av.abs().max().clamp_min(1e-6)).item()
        return (nreq * 4 <= B and rel < 2e-2), \
            f"{nreq}/{B} req differ (<=25%), max/global {rel:.2e} (<2e-2)"

    for name, a, b in zip(("finals", "swa_pool", "cmp_pool", "idx_pool",
                           "main_kv", "idx_kv", "main_sc", "y_q", "win_y2",
                           "yq", "ssq", "logits", "page_idx", "iq"),
                          snap1, snap2):
        if name in ("finals", "ssq"):
            ok, msg = _lax_ok(name, a.float(), b.float())
            if not ok:
                det = False
                print(f"  [F-diag] {name}: TOLERANCE EXCEEDED: {msg}")
            continue
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
        h_name = ("H o_proj partial calc_diff"
                  if TPDP_MODE
                  else "H o_proj+mhc_post calc_diff")
        gates.insert(5, (h_name,
                         stats.get("H_oproj_diff", 1), "< 2e-2",
                         stats.get("H_oproj_diff", 1) < 2e-2))
    ok = True
    for name, val, cond, passed in gates:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<28} = {val} ({cond})")

    if ok:                       # never bench on broken correctness
        benchmark(mods, w, ncmp=16384)    # 64k ctx (65536 tokens)

    print("=" * 60)
    result_name = "TPDP MLA-DP LOCAL OPS" if args.tpdp else "E2E"
    print(result_name + (" ALL PASS" if ok else " SOME FAILED"))
    sys.exit(0 if ok else 1)
