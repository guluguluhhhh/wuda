"""
test_e2e_decode.py -- END-TO-END DSV4 decode chain over the design-image KV
cache management (test/kv_cache_manager.py):

  hc_fused(nopc + fused attn_norm) -> front_mixed(+hc tail) ->
  qnorm_quant(PDL) + wq_b(ALL fusions: idxpost + head_ssq + indexer compressor
  + winkv) -> mqa_logits(paged, + MAIN compressor tail) -> topk_v2(512, page
  transform) -> flash_mla(SWA pool + compressed pool, fused query_rms_rope).

Multi-step decode simulation (the cache/state semantics ONLY show up across
steps): B requests advance pos together; mid-run one request is freed and its
slot reused (KV=0 / score=-inf re-init gate). P1 discipline: kernels unchanged,
KVCacheManager supplies slot indirection + paged-pool scatter shims.

Stage gates per step:
  A front y (fp8/bf16 segment calc_diff)          D topk vs torch.topk+transform
  B wq_b x_fp8 quant chain (byte match)           E flashMLA vs torch attention
  C mqa logits vs torch ref over DEQUANT pools        over the DEQUANT pools
Global gate: the whole simulation runs TWICE from the same seed -> all caches,
states and final outputs bitwise identical (run-to-run determinism).

flash_mla is optional (import-guarded): without it stages A-D still run.
"""
import os, sys, math, time, warnings, torch
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_cache_manager import (KVCacheManager, PAGE, WIN, RATIO,
                              D_NOPE, D_ROPE, TILE, NTILES)

import test_hc_fused_tc as t_hc
import test_front_mixed_gemm as t_fm
import test_wq_b_fp8_gemm as t_wq

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
TOPK = 512                     # DSV4 decode index_topk (topk_v2.cu contract)
SWA_TOPK = 128
Q_HEADS, Q_DIM = 128, 512
IDX_HEADS, IDX_D = 64, 128
MAX_POS = 4096
EPS = 1e-6


# ==================== weights (one-time, layer constants) ====================
def make_weights(seed=7):
    torch.manual_seed(seed)
    w = {}
    w["hc_fn"] = torch.randn(24, HC * DIM, device=DEV) * 0.01
    w["hc_base"] = torch.randn(24, device=DEV) * 0.1
    w["hc_scale"] = torch.rand(3, device=DEV) + 0.5
    w["attn_norm"] = torch.rand(DIM, device=DEV) + 0.5
    fw = torch.randn(N_FRONT, DIM, device=DEV, dtype=torch.bfloat16) * 0.02
    w["front_fp8"], w["front_sf"] = t_fm.quant_weight_fp8(fw[:2048])
    w["front_bf16"] = fw[2048:].contiguous()
    w["q_norm"] = torch.rand(1536, device=DEV) + 0.5
    wq = torch.randn(t_wq.N_MERGED, 1536, device=DEV) * 0.03
    w["wq_fp8"] = wq.to(torch.float8_e4m3fn)
    w["wq_sf"] = t_wq.make_weight_sf_ones(DEV)
    w["idx_ape"] = torch.randn(4, 256, device=DEV)
    w["idx_norm"] = torch.rand(128, device=DEV) + 0.5
    w["win_norm"] = torch.rand(512, device=DEV) + 0.5
    w["comp_norm"] = torch.rand(512, device=DEV) + 0.5
    ang = torch.rand(MAX_POS, 32, device=DEV) * 6.28
    w["cos"], w["sin"] = ang.cos().contiguous(), ang.sin().contiguous()
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


def ref_flash_attn(qn, swa_rows, cmp_rows, scale):
    """softmax over [swa ++ cmp] rows; v == k row (head_dim_v = 512)."""
    k = torch.cat([swa_rows, cmp_rows])                        # [n,512]
    lg = (qn @ k.t()) * scale                                  # [128,n]
    p = torch.softmax(lg.float(), dim=-1)
    return p @ k.float()                                       # [128,512]


# ==================== one decode step ====================
def decode_step(mods, w, mgr, slots, hidden, logits_buf, stats):
    hcm, fmm, wqm, mqm, tkm = mods
    B = len(slots)
    st = mgr.step_begin(slots)
    pos, q_pos = st["pos"], st["q_pos"]

    # -- 1. MHC (nopc + fused attn_norm) -> normed collapsed --------------
    collapsed = torch.empty(B, DIM, device=DEV, dtype=torch.bfloat16)
    pre = torch.empty(B, HC, device=DEV, dtype=torch.float32)
    po = torch.empty(B, HC, device=DEV, dtype=torch.float32)
    cb = torch.empty(B, HC, HC, device=DEV, dtype=torch.float32)
    hcm.hc_fused_forward_out(hidden, w["hc_fn"], w["hc_base"], w["hc_scale"],
                             EPS, EPS, collapsed, pre, po, cb,
                             with_post_comb=False, attn_norm_w=w["attn_norm"],
                             attn_norm_eps=EPS)

    # -- 2. front_mixed (+hc tail; hc_mix random side-channel, P1 note) ---
    x_fp8, x_sf = t_fm.quant_act_fp8(collapsed)
    hc = t_fm.make_hc(B, seed=int(pos[0]) + 11)
    y = fmm.front_mixed_gemm(collapsed, x_fp8, x_sf, w["front_bf16"],
                             w["front_fp8"], w["front_sf"], **hc)
    d8 = t_fm.calc_diff(y[:, :2048].float(),
                        t_fm.dequant_fp8(x_fp8, x_sf, True)
                        @ t_fm.dequant_fp8(w["front_fp8"], w["front_sf"], False).t())
    stats["A_front_d8"] = max(stats.get("A_front_d8", 0.0), d8)

    # -- 3. wq_b ALL fusions; the KERNEL writes idx state pool (slot_map),
    #       the indexer fused pages AND the SWA MODEL1 pages directly --------
    xq = torch.empty(B, 1536, device=DEV, dtype=torch.float8_e4m3fn)
    xq_sf = torch.empty(B, 12, device=DEV, dtype=torch.uint8)
    ssq = torch.zeros(B, Q_HEADS, device=DEV, dtype=torch.float32)
    rets = wqm.wq_b_proj_gemm_merged(
        xq, t_wq.as_ue8m0(xq_sf), w["wq_fp8"], w["wq_sf"], q_pos,
        w["cos"], w["sin"], head_ssq=ssq, mock_post=False,
        cmp_pos=pos, idx_y4=y[:, 4096:4608].float().contiguous(),
        idx_ape=w["idx_ape"], idx_norm=w["idx_norm"],
        cos_tab=w["cos"], sin_tab=w["sin"],
        idx_kv=mgr.idx_kv, idx_sc=mgr.idx_sc,
        win_y2=y[:, 1536:2048].float().contiguous(), win_norm=w["win_norm"],
        q_y=y[:, :1536], q_norm_w=w["q_norm"],
        slot_map=st["slot_map"],
        idx_cache=mgr.idx_pool, idx_dst=st["idx_dst"],
        swa_cache=mgr.swa_pool, swa_dst=st["swa_dst"])
    yq, iq_fp4, iq_sf = rets[0], rets[1], rets[2]
    qr, sr = t_wq.ref_qnorm_quant(y[:, :1536], w["q_norm"])
    stats["B_xq_match"] = min(stats.get("B_xq_match", 1.0),
                              (xq.view(torch.uint8) == qr.view(torch.uint8))
                              .float().mean().item())

    # -- 4. MAIN state fresh row (pending producer op; plain relayout) ----
    mgr.write_main_state(slots, pos, y[:, 2048:4096].float())

    # -- 5. mqa_logits (paged idx pool) + MAIN compressor tail: the KERNEL
    #       reads the state POOL via slot_map and writes the Main-compressed
    #       MODEL1 pages directly ------------------------------------------
    ncmp = mgr.n_compressed(slots, pos)
    weights64 = y[:, 4608:4672].float().contiguous()
    logits_buf.fill_(float("-inf"))
    # PRODUCTION form: compact comp_q8/s8/rope OMITTED (cache mode writes the
    # MODEL1 pages directly; compact would double-write ~600B/row).
    mqm.mqa_logits_fp4_decode_out(
        iq_fp4, iq_sf, mgr.idx_pool, weights64, ncmp,
        mgr.block_table("idx", slots), logits_buf,
        cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=w["cos"],
        sin_tab=w["sin"], comp_kv=mgr.main_kv, comp_sc=mgr.main_sc,
        slot_map=st["slot_map"], cmp_cache=mgr.cmp_pool,
        cmp_dst=st["cmp_dst"])
    # zero-row robustness: the fp4 chain's se has NO clamp (amax=0 -> sf byte
    # 0, codes 0); assert the consumer stays finite (no NaN/+inf in logits).
    torch.cuda.synchronize()
    lg = logits_buf[:, : max(int(ncmp.max()), 1)]
    stats["G_finite"] = stats.get("G_finite", True) and \
        bool((torch.isnan(lg) | (lg == float("inf"))).sum().item() == 0)

    kv_rows = deq_idx_pool(mgr, slots, pos)
    refs = ref_mqa_logits(iq_fp4, iq_sf, weights64, kv_rows)
    for b in range(B):
        n = int(ncmp[b])
        if n:
            d = (logits_buf[b, :n].float() - refs[b]).abs().max().item()
            rel = d / (refs[b].abs().max().item() + 1e-6)
            stats["C_mqa_rel"] = max(stats.get("C_mqa_rel", 0.0), rel)

    # -- 6. topk_v2 (512, page transform -> compressed physical) ----------
    # scores: COLUMN SLICE of the wide buffer (row stride 256 % 4 == 0 keeps
    # the 16B-vector-load contract; a .contiguous() copy would break it for
    # tiny ncmp). -inf tail beyond seq_len is never read.
    cmp_bt = mgr.block_table("cmp", slots)
    L = max(int(ncmp.max()), 1)
    page_idx = tkm.topk_v2(logits_buf[:, :L], ncmp, cmp_bt, TOPK, PAGE, None)
    for b in range(B):
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

    # -- 7. flashMLA (SWA pool + compressed pool, fused q rms+rope) -------
    if HAS_FLASH_MLA:
        q_raw = yq.view(B, 1, Q_HEADS, Q_DIM).bfloat16()
        sum_sq = ssq.view(B, 1, Q_HEADS)
        rc = w["cos"][pos].view(B, 1, 32).contiguous()
        rs = w["sin"][pos].view(B, 1, 32).contiguous()
        swa_idx, swa_len = mgr.swa_indices(slots, pos, SWA_TOPK)
        cmp_len = torch.minimum(ncmp, torch.tensor(TOPK, device=DEV)).int()
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
        res = flash_mla.flash_mla_with_kvcache(
            q=q_in, k_cache=mgr.model1_cache_view("swa"),
            block_table=None, cache_seqlens=None, head_dim_v=Q_DIM,
            tile_scheduler_metadata=sched, num_splits=None,
            softmax_scale=Q_DIM ** -0.5, causal=False, is_fp8_kvcache=True,
            indices=swa_idx, topk_length=swa_len,
            extra_k_cache=mgr.model1_cache_view("cmp"),
            extra_indices_in_kvcache=page_idx.view(B, 1, TOPK),
            extra_topk_length=cmp_len,
            **fused_q)
        out = res[0] if isinstance(res, tuple) else res
        # torch reference over the DEQUANT pools
        for b in range(B):
            p = int(pos[b])
            qn = (q_raw[b, 0].float()
                  * torch.rsqrt(sum_sq[b, 0].view(-1, 1) / Q_DIM + EPS))
            e, o = qn[:, 448::2].clone(), qn[:, 449::2].clone()
            c, s = rc[b, 0], rs[b, 0]
            qn[:, 448::2] = e * c - o * s
            qn[:, 449::2] = e * s + o * c
            table = mgr.reqs[slots[b]]["swa"]
            n_sw = int(swa_len[b])
            sw = torch.stack([deq_model1_row(mgr.swa_pool,
                                             table[(t % WIN) // PAGE],
                                             (t % WIN) % PAGE)
                              for t in range(p + 1 - n_sw, p + 1)]) \
                if n_sw else torch.zeros(0, Q_DIM, device=DEV)
            cm = torch.stack([deq_model1_row(mgr.cmp_pool, int(pi) // PAGE,
                                             int(pi) % PAGE)
                              for pi in page_idx[b][:int(cmp_len[b])]]) \
                if int(cmp_len[b]) else torch.zeros(0, Q_DIM, device=DEV)
            ref = ref_flash_attn(qn.bfloat16().float(), sw, cm, Q_DIM ** -0.5)
            cos = torch.nn.functional.cosine_similarity(
                out[b, 0].float().flatten(), ref.flatten(), dim=0).item()
            stats["E_mla_cos"] = min(stats.get("E_mla_cos", 1.0), cos)
    return y


# ==================== per-operator latency ====================
def probe_kernel_names(fn):
    """All kernel names fn launches (flush/memset/aten prep excluded); the
    benched fns below are PURE kernel calls -- inputs pre-materialized."""
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


def time_op(fn):
    from bench_utils import bench_kineto
    t = bench_kineto(fn, probe_kernel_names(fn), suppress_kineto_output=True,
                     with_multiple_kernels=True)
    return 1e6 * (sum(t) if isinstance(t, tuple) else t)


def time_wall(fn, warmup=3, iters=20):
    """CUDA-event WALL median with an L2 flush per iter. For PDL pairs
    (quant producer + GEMM): kineto per-kernel sums DOUBLE-COUNT the overlap
    and the consumer's GDS wait; the wall span is the honest number."""
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=DEV)   # 1GB >> L2
    s = [torch.cuda.Event(True) for _ in range(iters)]
    e = [torch.cuda.Event(True) for _ in range(iters)]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        flush.zero_()
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    ts = sorted(s[i].elapsed_time(e[i]) for i in range(iters))
    return 1e3 * ts[len(ts) // 2]


def benchmark(mods, w, ncmp=2048):
    """vLLM-style HOT measurement (the comparison target): weights/caches
    warm, decode steps BACK-TO-BACK, no L2 flush.
      per-stage : kineto device time over R hot chain iters, kernels
                  bucketed by stage (wq_b bucket = qnorm + merged GEMM;
                  device-sum slightly double-counts their PDL overlap)
      eager     : wall/step of the hot eager loop (launch overhead visible)
      graph     : wall/step of CUDA-graph replay (vLLM serving form; THE
                  end-to-end number)
    Context ops (mqa/topk/mla) at a FABRICATED long context: page tables
    and lens real, cache bytes arbitrary -- latency-neutral. Chain ends at
    flashMLA (no o-proj etc. yet, matching the comparison scope)."""
    hcm, fmm, wqm, mqm, tkm = mods
    print("\n" + "=" * 76)
    print(f"Per-operator latency (us) -- compress-row step, ncmp={ncmp} "
          f"compressed tokens ({ncmp * RATIO} ctx)")
    print("=" * 76)
    S = 4 * ncmp + 8
    ang = torch.rand(S, 32, device=DEV) * 6.28
    cosl, sinl = ang.cos().contiguous(), ang.sin().contiguous()
    cols = ["mhc", "front", "wq_b", "mqa", "topk"] + \
        (["mla"] if HAS_FLASH_MLA else []) + ["stages", "eager", "graph"]
    print("  hot, no-flush; stages = kineto device us; eager/graph = wall/step")
    print(f"  {'B':<5}" + "".join(f"{c:>9}" for c in cols))
    print("  " + "-" * (5 + 9 * len(cols)))
    notes = []      # ablation/trace lines buffered AFTER the main table
    for B in (1, 16, 32, 48, 64, 80, 96, 112, 128):
        mgr = KVCacheManager(capacity=B + 2, pages_per_pool=(ncmp // PAGE) * B
                             + 4 * B, max_pages_per_req=ncmp // PAGE)
        slots = [mgr.alloc_request() for _ in range(B)]
        for s in slots:                       # fabricate history: next step is
            mgr.reqs[s]["pos"] = 4 * ncmp - 2   # pos 4*ncmp-1 (compress row)
            for ct in range(ncmp):
                mgr._page_of("idx", s, ct); mgr._page_of("cmp", s, ct)
        st = mgr.step_begin(slots)
        pos, q_pos = st["pos"], st["q_pos"]
        hidden = torch.randn(B, HC, DIM, device=DEV, dtype=torch.bfloat16) * .1

        # ---- materialize every op's inputs by running the chain once ----
        collapsed = torch.empty(B, DIM, device=DEV, dtype=torch.bfloat16)
        pre = torch.empty(B, HC, device=DEV, dtype=torch.float32)
        po, cb = pre.clone(), torch.empty(B, HC, HC, device=DEV)
        run_hc = lambda: hcm.hc_fused_forward_out(
            hidden, w["hc_fn"], w["hc_base"], w["hc_scale"], EPS, EPS,
            collapsed, pre, po, cb, with_post_comb=False,
            attn_norm_w=w["attn_norm"], attn_norm_eps=EPS)
        run_hc()
        # front: PRODUCTION pad16 contract for B<16 (test_front discipline);
        # downstream consumes y[:B].
        Bp = max(B, 16)
        coll_p = collapsed
        if Bp != B:
            coll_p = torch.zeros(Bp, DIM, device=DEV, dtype=torch.bfloat16)
            coll_p[:B] = collapsed
        x_fp8, x_sf = t_fm.quant_act_fp8(coll_p)
        hc_t = t_fm.make_hc(Bp, seed=3)
        y_p = torch.empty(Bp, N_FRONT, device=DEV, dtype=torch.bfloat16)
        run_front = lambda: fmm.front_mixed_gemm(
            coll_p, x_fp8, x_sf, w["front_bf16"], w["front_fp8"],
            w["front_sf"], out=y_p, **hc_t)
        run_front()
        y = y_p[:B]
        idx_y4 = y[:, 4096:4608].float().contiguous()
        win_y2 = y[:, 1536:2048].float().contiguous()
        weights64 = y[:, 4608:4672].float().contiguous()
        xq = torch.empty(B, 1536, device=DEV, dtype=torch.float8_e4m3fn)
        xq_sf = t_wq.as_ue8m0(torch.empty(B, 12, device=DEV,
                                          dtype=torch.uint8))
        ssq = torch.zeros(B, Q_HEADS, device=DEV, dtype=torch.float32)
        run_wqb = lambda: wqm.wq_b_proj_gemm_merged(
            xq, xq_sf, w["wq_fp8"], w["wq_sf"], q_pos, cosl, sinl,
            head_ssq=ssq, mock_post=False, cmp_pos=pos, idx_y4=idx_y4,
            idx_ape=w["idx_ape"], idx_norm=w["idx_norm"], cos_tab=cosl,
            sin_tab=sinl, idx_kv=mgr.idx_kv, idx_sc=mgr.idx_sc,
            win_y2=win_y2, win_norm=w["win_norm"],
            q_y=y[:, :1536], q_norm_w=w["q_norm"], slot_map=st["slot_map"],
            idx_cache=mgr.idx_pool, idx_dst=st["idx_dst"],
            swa_cache=mgr.swa_pool, swa_dst=st["swa_dst"])
        rets = run_wqb()
        yq, iq_fp4, iq_sf = rets[0], rets[1], rets[2]
        nc = mgr.n_compressed(slots, pos)
        idx_bt = mgr.block_table("idx", slots)
        logits = torch.full((B, (ncmp + 255) // 256 * 256), float("-inf"),
                            device=DEV)
        # PRODUCTION form: compact comp outputs omitted (cache direct write).
        run_mqa = lambda: mqm.mqa_logits_fp4_decode_out(
            iq_fp4, iq_sf, mgr.idx_pool, weights64, nc, idx_bt, logits,
            cmp_pos=pos, comp_norm=w["comp_norm"], cos_tab=cosl,
            sin_tab=sinl, comp_kv=mgr.main_kv, comp_sc=mgr.main_sc,
            slot_map=st["slot_map"], cmp_cache=mgr.cmp_pool,
            cmp_dst=st["cmp_dst"])
        run_mqa()
        cmp_bt = mgr.block_table("cmp", slots)
        page_idx = torch.empty(B, TOPK, dtype=torch.int32, device=DEV)
        meta = torch.zeros(B + 1, 2, dtype=torch.int32, device=DEV)
        run_topk = lambda: tkm.topk_v2_transform(
            logits[:, :ncmp], nc, cmp_bt, page_idx, PAGE, meta, None)
        run_topk()

        stage_fns = [("mhc", run_hc), ("front", run_front),
                     ("wq_b", run_wqb), ("mqa", run_mqa), ("topk", run_topk)]
        if HAS_FLASH_MLA:
            q_raw = yq.view(B, 1, Q_HEADS, Q_DIM).bfloat16()
            sum_sq = ssq.view(B, 1, Q_HEADS)
            rc = cosl[pos].view(B, 1, 32).contiguous()
            rs = sinl[pos].view(B, 1, 32).contiguous()
            swa_idx, swa_len = mgr.swa_indices(slots, pos, SWA_TOPK)
            cmp_len = torch.minimum(nc, torch.tensor(TOPK, device=DEV)).int()
            ext_idx = page_idx.view(B, 1, TOPK)
            swa_v, cmp_v = mgr.model1_cache_view("swa"), mgr.model1_cache_view("cmp")
            sched, _ = flash_mla.get_mla_metadata()

            def run_mla():
                if FMLA_FUSED_Q:
                    qi, fq = q_raw, dict(q_rms_sum_sq=sum_sq,
                                         q_rope_cos=rc, q_rope_sin=rs)
                else:
                    qi, fq = flash_mla.query_rms_rope(q_raw, sum_sq, rc, rs,
                                                      EPS), {}
                return flash_mla.flash_mla_with_kvcache(
                    q=qi, k_cache=swa_v, block_table=None, cache_seqlens=None,
                    head_dim_v=Q_DIM, tile_scheduler_metadata=sched,
                    num_splits=None, softmax_scale=Q_DIM ** -0.5,
                    causal=False, is_fp8_kvcache=True, indices=swa_idx,
                    topk_length=swa_len, extra_k_cache=cmp_v,
                    extra_indices_in_kvcache=ext_idx,
                    extra_topk_length=cmp_len, **fq)
            run_mla()
            stage_fns.append(("mla", run_mla))

        chain_fns = [f for _, f in stage_fns]

        def chain():
            for f in chain_fns:
                f()

        # ---- vLLM-style hot measurement --------------------------------
        def hot_wall(f, warmup=5, iters=50):   # back-to-back wall/step
            for _ in range(warmup):
                f()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                f()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1e6

        # per-stage: bucket kineto device time of a HOT R-iter chain loop
        name2stage = {}
        for sname, f in stage_fns:
            for n in probe_kernel_names(f):
                name2stage.setdefault(n, sname)
        from torch.profiler import profile as _prof, ProfilerActivity
        R = 10
        for _ in range(3):
            chain()
        torch.cuda.synchronize()
        try:
            pctx = _prof(activities=[ProfilerActivity.CUDA], acc_events=True)
        except TypeError:
            pctx = _prof(activities=[ProfilerActivity.CUDA])
        with pctx as prof:
            for _ in range(R):
                chain()
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

        t_eager = hot_wall(chain)

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
            t_graph = hot_wall(g.replay)
        except Exception as err:
            print(f"  (graph capture failed at B={B}: {err})")

        # ---- perfetto timeline: 3x HOT eager chain + 1x graph replay.
        # Drop the json onto ui.perfetto.dev / chrome://tracing.
        if B in (16, 128):
            with _prof(activities=[ProfilerActivity.CPU,
                                   ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    chain()
                if not math.isnan(t_graph):
                    g.replay()
                torch.cuda.synchronize()
            tp = f"/tmp/e2e_trace_B{B}.json"
            prof.export_chrome_trace(tp)
            notes.append(f"  B={B:<4} perfetto trace -> {tp} (3x hot eager + "
                         f"{'1x graph' if not math.isnan(t_graph) else 'no graph'})")
            del prof

        print(f"  {B:<5}" + "".join(f"{t:>9.1f}" for t in ts)
              + f"{sum(ts):>9.1f}{t_eager:>9.1f}{t_graph:>9.1f}")
        del mgr
        torch.cuda.empty_cache()
    if notes:
        print()
        for ln in notes:
            print(ln)


# ==================== simulation ====================
def run_sim(mods, w, B=16, steps=8, seed=42, reuse_at=5):
    torch.manual_seed(seed)
    mgr = KVCacheManager(capacity=B + 4, pages_per_pool=256,
                         max_pages_per_req=8)
    slots = [mgr.alloc_request() for _ in range(B)]
    logits_buf = torch.full((B, 256), float("-inf"), device=DEV)
    stats, finals = {}, None
    for t in range(steps):
        if t == reuse_at:                      # slot lifecycle gate
            mgr.free_request(slots[0])
            slots[0] = mgr.alloc_request()     # KV=0 / score=-inf re-init
        hidden = torch.randn(B, HC, DIM, device=DEV,
                             dtype=torch.bfloat16) * 0.1
        if t == 3:      # ALL-ZERO row on a COMPRESS step (pos=3): the fp4
            hidden[1].zero_()   # se-no-clamp path must stay consumer-safe
        finals = decode_step(mods, w, mgr, slots, hidden, logits_buf, stats)
    snap = (finals.clone(), mgr.swa_pool.clone(), mgr.cmp_pool.clone(),
            mgr.idx_pool.clone(), mgr.main_kv.clone(), mgr.idx_kv.clone())
    return stats, snap


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available"); sys.exit(0)
    print(f"Device: {torch.cuda.get_device_name()}")
    mods = (t_hc.load_cuda_module(), t_fm.load_module(), t_wq.load_module(),
            __import__("test_mqa_logits_fp4").load_cuda_module(),
            __import__("test_topk_v2").load_cuda_module())
    w = make_weights()

    print("\nE2E decode simulation (B=16, 8 steps, slot reuse at step 5):")
    stats, snap1 = run_sim(mods, w)
    _, snap2 = run_sim(mods, w)                # determinism gate
    det = all(torch.equal(a, b) for a, b in zip(snap1, snap2))

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
        ("F run-to-run bitwise", det, "== True", det),
    ]
    if HAS_FLASH_MLA:
        gates.insert(4, ("E flashMLA cos vs torch ref",
                         stats.get("E_mla_cos", 0), "> 0.98",
                         stats.get("E_mla_cos", 0) > 0.98))
    ok = True
    for name, val, cond, passed in gates:
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<28} = {val} ({cond})")

    if ok:                       # never bench on broken correctness
        benchmark(mods, w)

    print("=" * 60)
    print("E2E " + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)
