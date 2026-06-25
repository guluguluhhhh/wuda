"""
DeepSeek-V4 Pro Decode Attention 纯 PyTorch 实现。
只处理 start_pos>0 的 decode 路径（兼容 seqlen>=1 的 chunked prefill）。

- HCA (compress_ratio=128): SWA + 网格压缩, 无 Indexer
- CSA (compress_ratio=4):   SWA + Indexer 稀疏 top-1024 压缩

对齐官方 Attention.forward decode 路径。
零外部依赖（仅 torch），所有 helper 均本地实现。
"""

import torch
import torch.nn.functional as F


# ===================== helpers =====================

def _precompute_freqs_cis(dim, end, base=10000.0):
    """预计算复数 freqs_cis, shape [end, dim//2]。
    对齐官方 precompute_freqs_cis（decode 场景无需 YaRN 插值）。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(x, freqs_cis, inverse=False):
    """对齐官方 apply_rotary_emb。x: [..., rd], freqs_cis: [..., rd//2] complex。"""
    dtype = x.dtype
    y = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    y = torch.view_as_real(y * freqs_cis).flatten(-2)
    return y.to(dtype)


def _act_quant(x, block_size=64):
    """对齐官方 act_quant(inplace=True): bf16 → FP32 scale → cast float8_e4m3fn → bf16。"""
    fp8_max = 448.0; sh = x.shape
    xf = x.float().reshape(-1, block_size)
    am = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = am / fp8_max
    q = (xf / scale).clamp(-fp8_max, fp8_max)
    # 真正走 FP8 round-trip, 不是连续裁剪
    xf = q.to(torch.float8_e4m3fn).to(torch.float32) * scale
    return xf.to(x.dtype).reshape(sh)


def _window_topk(win, B, S, start_pos):
    """[B, S, win_size], -1 for invalid (causal+ring)。
    对齐官方 get_window_topk_idxs 的三分支逻辑。"""
    if start_pos >= win - 1:
        rows = []
        for i in range(S):
            pp = (start_pos + i) % win
            r = torch.cat([torch.arange(pp + 1, win), torch.arange(0, pp + 1)])
            rows.append(F.pad(r, (0, win - len(r)), value=-1)[:win])
        return torch.stack(rows).unsqueeze(0).expand(B, -1, -1)
    elif start_pos > 0:
        rows = []
        for i in range(S):
            n_valid = start_pos + i + 1
            rows.append(F.pad(torch.arange(n_valid), (0, win - n_valid), value=-1))
        return torch.stack(rows).unsqueeze(0).expand(B, -1, -1)
    else:
        base = torch.arange(S).unsqueeze(1)
        matrix = (base - win + 1).clamp(0) + torch.arange(min(S, win))
        matrix = torch.where(matrix > base, -1, matrix)
        return matrix.unsqueeze(0).expand(B, -1, -1)


def _hca_compress_topk(ratio, B, S, start_pos, offset):
    """HCA grid-based: 取 start_pos+S 之前的所有压缩 slot 作为 topk，-1 标记因果无效位置。"""
    n_slots = (start_pos + S) // ratio
    rows = []
    for i in range(S):
        n_valid = (start_pos + i + 1) // ratio
        idxs = torch.arange(n_valid) + offset
        rows.append(F.pad(idxs, (0, n_slots - n_valid), value=-1))
    return torch.stack(rows).unsqueeze(0).expand(B, -1, -1)


def _comp_step(x_tok, pos, ratio, overlap, coff, Hd, rd,
               wkv_w, wgate_w, ape, norm_w, kv_s, sc_s,
               kv_main, kv_start, freqs_cis, eps):
    """处理一个 token 的 Compressor 增量。pos = start_pos + tok_idx。
    原地修改: kv_s[:B], sc_s[:B], kv_main[:B]。"""
    dtype = x_tok.dtype
    B = x_tok.size(0)

    c_kv = F.linear(x_tok.float(), wkv_w.float())
    c_sc = F.linear(x_tok.float(), wgate_w.float())
    c_sc = c_sc + ape[pos % ratio].float()

    do_compress = (pos + 1) % ratio == 0

    if overlap:
        kv_s[:B, ratio + pos % ratio] = c_kv
        sc_s[:B, ratio + pos % ratio] = c_sc
        if do_compress:
            kv2 = torch.cat([kv_s[:B, :ratio, :Hd], kv_s[:B, ratio:, Hd:]], dim=1)
            sc2 = torch.cat([sc_s[:B, :ratio, :Hd], sc_s[:B, ratio:, Hd:]], dim=1)
            c = (kv2 * sc2.softmax(dim=1)).sum(dim=1)
            kv_s[:B, :ratio] = kv_s[:B, ratio:]
            sc_s[:B, :ratio] = sc_s[:B, ratio:]
    else:
        kv_s[:B, pos % ratio] = c_kv
        sc_s[:B, pos % ratio] = c_sc
        if do_compress:
            c = (kv_s[:B] * sc_s[:B].softmax(dim=1)).sum(dim=1)

    if do_compress:
        c = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps)
        c = (norm_w * c).to(dtype)

        # RoPE
        idx = pos + 1 - ratio
        fc = freqs_cis[idx:idx + 1].to(device=c.device)
        c[..., -rd:] = _apply_rotary_emb(
            c[..., -rd:].unsqueeze(1), fc).squeeze(1)

        c[..., :-rd] = _act_quant(c[..., :-rd], 64)
        kv_main[:B, kv_start + pos // ratio] = c


# ===================== HCA decode =====================

def deepseek_v4_hca_decode(
    x, start_pos,
    kv_cache,

    compressor_wkv_weight=None, compressor_wgate_weight=None,
    compressor_ape=None, compressor_norm_weight=None,
    compressor_freqs_cis=None,
    comp_kv_state=None, comp_score_state=None,
    max_seq_len=4096,

    rms_norm_eps=1e-6, num_heads=128, head_dim=512, rope_head_dim=64,
    q_lora_rank=1536, o_groups=16, o_lora_rank=1024, window_size=128,

    q_a_proj_weight=None, q_a_norm_weight=None, q_b_proj_weight=None,
    kv_proj_weight=None, kv_norm_weight=None,
    sinks=None, o_a_proj_weight=None, o_b_proj_weight=None,
    freqs_cis=None,
):
    """HCA decode (compress_ratio=128)。

    原地修改（in-place mutation）:
        kv_cache[:B, (start_pos+i)%win]  ← KV 写入窗口环形缓冲
        comp_kv_state[:B]                ← Compressor 增量更新（_comp_step 内）
        comp_score_state[:B]             ← Compressor 增量更新

    freqs_cis: tensor [max_seq_len, rope_head_dim//2] complex, 由 _precompute_freqs_cis 生成。
    compressor_freqs_cis: 同上，可选（默认同 freqs_cis），使用 compress_rope_theta 预计算。
    """
    dtype = x.dtype
    B, S, D = x.shape
    H, Hd, win, ratio, rd = num_heads, head_dim, window_size, 128, rope_head_dim

    fc = freqs_cis[start_pos:start_pos + S]

    # ---- q ----
    qr = F.linear(x, q_a_proj_weight)
    qrf = qr.float()
    qr = (qrf * torch.rsqrt(qrf.square().mean(-1, keepdim=True) + rms_norm_eps)).to(dtype)
    qr = qr * q_a_norm_weight
    q = F.linear(qr, q_b_proj_weight).view(B, S, H, Hd).transpose(1, 2)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + rms_norm_eps)
    q[..., -rd:] = _apply_rotary_emb(q[..., -rd:], fc)

    # ---- kv ----
    kv = F.linear(x, kv_proj_weight)
    kvf = kv.float()
    kv = (kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + rms_norm_eps)).to(dtype)
    kv = kv * kv_norm_weight
    kv[..., -rd:] = _apply_rotary_emb(kv[..., -rd:], fc)
    nd = Hd - rd * 2
    if nd > 0:
        kv = torch.cat([_act_quant(kv[..., :nd], 64), kv[..., nd:]], dim=-1)

    # ---- kv_cache window ring buffer ----
    for i in range(S):
        kv_cache[:B, (start_pos + i) % win] = kv[:, i]

    # ---- Compressor ----
    if comp_kv_state is None:
        comp_kv_state = torch.zeros(B, ratio, Hd, device=x.device, dtype=torch.float32)
    if comp_score_state is None:
        comp_score_state = torch.full((B, ratio, Hd), float('-inf'), device=x.device, dtype=torch.float32)
    cfc = compressor_freqs_cis if compressor_freqs_cis is not None else freqs_cis
    for i in range(S):
        _comp_step(
            x[:, i], start_pos + i, ratio, False, 1, Hd, rd,
            compressor_wkv_weight, compressor_wgate_weight,
            compressor_ape, compressor_norm_weight,
            comp_kv_state, comp_score_state, kv_cache, win,
            cfc, rms_norm_eps)

    # ---- topk_idxs ----
    wt = _window_topk(win, B, S, start_pos).to(x.device)
    ct = _hca_compress_topk(ratio, B, S, start_pos, win).to(x.device)
    topk = torch.cat([wt, ct], dim=-1).int()

    # ---- sparse attention ----
    n_topk = topk.shape[-1]
    safe_idxs = topk.clamp(min=0)
    batch_idxs = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, S, n_topk)
    kg = kv_cache[batch_idxs, safe_idxs]
    kg_heads = kg.unsqueeze(1).expand(B, H, S, n_topk, Hd)

    scores   = torch.einsum('bhsd,bhskd->bhsk', q.float(), kg_heads.float())
    scores   = scores * (Hd ** -0.5)
    scores   = scores.masked_fill((topk == -1).unsqueeze(1), float('-inf'))
    smax     = scores.max(dim=-1, keepdim=True).values
    exp_s    = torch.exp(scores - smax)
    exp_sink = torch.exp(sinks.float().view(1, H, 1, 1) - smax)
    attn_w   = (exp_s / (exp_s.sum(dim=-1, keepdim=True) + exp_sink)).to(dtype)
    ao = torch.einsum('bhsk,bhskd->bhsd', attn_w, kg_heads)

    # ---- de-rope + output proj ----
    ao[..., -rd:] = _apply_rotary_emb(ao[..., -rd:], fc, inverse=True)
    groups = ao.transpose(1, 2).reshape(B, S, o_groups, -1)
    pg = (H * Hd) // o_groups
    oa_out = o_a_proj_weight.shape[0] // o_groups
    w_oa = o_a_proj_weight.view(o_groups, oa_out, pg).transpose(1, 2)
    gx = groups.reshape(B * S, o_groups, pg).transpose(0, 1)
    oa = torch.bmm(gx, w_oa).transpose(0, 1).reshape(B, S, -1)
    output = F.linear(oa, o_b_proj_weight)
    return output


# ===================== CSA decode =====================

def deepseek_v4_csa_decode(
    x, start_pos,
    kv_cache,

    indexer_wq_b_weight=None, indexer_weights_proj_weight=None,
    index_n_heads=64, index_head_dim=128, index_topk=1024,
    idx_compressor_wkv_weight=None, idx_compressor_wgate_weight=None,
    idx_compressor_ape=None, idx_compressor_norm_weight=None,
    idx_compressor_freqs_cis=None,
    idx_kv_state=None, idx_score_state=None,
    idx_kv_cache=None,

    compressor_wkv_weight=None, compressor_wgate_weight=None,
    compressor_ape=None, compressor_norm_weight=None,
    compressor_freqs_cis=None,
    comp_kv_state=None, comp_score_state=None,
    max_seq_len=4096,

    rms_norm_eps=1e-6, num_heads=128, head_dim=512, rope_head_dim=64,
    q_lora_rank=1536, o_groups=16, o_lora_rank=1024, window_size=128,

    q_a_proj_weight=None, q_a_norm_weight=None, q_b_proj_weight=None,
    kv_proj_weight=None, kv_norm_weight=None,
    sinks=None, o_a_proj_weight=None, o_b_proj_weight=None,
    freqs_cis=None,
):
    """CSA decode (compress_ratio=4)。

    原地修改（in-place mutation）:
        kv_cache[:B, (start_pos+i)%win]  ← KV 写入窗口环形缓冲
        comp_kv_state[:B]                ← 主 Compressor 增量更新
        comp_score_state[:B]             ← 主 Compressor 增量更新
        idx_kv_cache[:B]                 ← Indexer Compressor 写入压缩 KV
        idx_kv_state[:B]                 ← Indexer Compressor 增量更新
        idx_score_state[:B]              ← Indexer Compressor 增量更新

    freqs_cis: tensor [max_seq_len, rope_head_dim//2] complex, 由 _precompute_freqs_cis 生成。
    compressor_freqs_cis / idx_compressor_freqs_cis: 同上，使用 compress_rope_theta 预计算。
    """
    dtype = x.dtype
    B, S, D = x.shape
    H, Hd, win, ratio, rd = num_heads, head_dim, window_size, 4, rope_head_dim
    iH, iHd = index_n_heads, index_head_dim

    fc = freqs_cis[start_pos:start_pos + S]

    # ---- q ----
    qr = F.linear(x, q_a_proj_weight)
    qrf = qr.float()
    qr = (qrf * torch.rsqrt(qrf.square().mean(-1, keepdim=True) + rms_norm_eps)).to(dtype)
    qr = qr * q_a_norm_weight
    q = F.linear(qr, q_b_proj_weight).view(B, S, H, Hd).transpose(1, 2)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + rms_norm_eps)
    q[..., -rd:] = _apply_rotary_emb(q[..., -rd:], fc)

    # ---- kv ----
    kv = F.linear(x, kv_proj_weight)
    kvf = kv.float()
    kv = (kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + rms_norm_eps)).to(dtype)
    kv = kv * kv_norm_weight
    kv[..., -rd:] = _apply_rotary_emb(kv[..., -rd:], fc)
    nd = Hd - rd * 2
    if nd > 0:
        kv = torch.cat([_act_quant(kv[..., :nd], 64), kv[..., nd:]], dim=-1)

    # ---- kv_cache window ----
    for i in range(S):
        kv_cache[:B, (start_pos + i) % win] = kv[:, i]

    # ---- main Compressor ----
    if comp_kv_state is None:
        comp_kv_state = torch.zeros(B, 2 * ratio, 2 * Hd, device=x.device, dtype=torch.float32)
    if comp_score_state is None:
        comp_score_state = torch.full((B, 2 * ratio, 2 * Hd), float('-inf'), device=x.device, dtype=torch.float32)
    cfc = compressor_freqs_cis if compressor_freqs_cis is not None else freqs_cis
    for i in range(S):
        _comp_step(
            x[:, i], start_pos + i, ratio, True, 2, Hd, rd,
            compressor_wkv_weight, compressor_wgate_weight,
            compressor_ape, compressor_norm_weight,
            comp_kv_state, comp_score_state, kv_cache, win,
            cfc, rms_norm_eps)

    # ---- Indexer Compressor ----
    if idx_kv_state is None:
        idx_kv_state = torch.zeros(B, 2 * ratio, 2 * iHd, device=x.device, dtype=torch.float32)
    if idx_score_state is None:
        idx_score_state = torch.full((B, 2 * ratio, 2 * iHd), float('-inf'), device=x.device, dtype=torch.float32)
    if idx_kv_cache is None:
        idx_kv_cache = torch.zeros(B, max_seq_len // ratio, iHd, device=x.device, dtype=dtype)
    icfc = idx_compressor_freqs_cis if idx_compressor_freqs_cis is not None else cfc
    for i in range(S):
        _comp_step(
            x[:, i], start_pos + i, ratio, True, 2, iHd, rd,
            idx_compressor_wkv_weight, idx_compressor_wgate_weight,
            idx_compressor_ape, idx_compressor_norm_weight,
            idx_kv_state, idx_score_state, idx_kv_cache, 0,
            icfc, rms_norm_eps)

    # ---- Indexer topk ----
    iq = F.linear(qr, indexer_wq_b_weight).view(B, S, iH, iHd)
    iq[..., -rd:] = _apply_rotary_emb(iq[..., -rd:], fc)

    ep = start_pos + S
    n_comp_slots = ep // ratio
    wgt = F.linear(x, indexer_weights_proj_weight)
    wgt = wgt * ((iHd ** -0.5) * (iH ** -0.5))
    iscore = torch.einsum("bshd,bnd->bshn", iq, idx_kv_cache[:B, :n_comp_slots])
    iscore = (iscore.relu_() * wgt.unsqueeze(-1)).sum(dim=2)

    for i in range(S):
        n_valid = (start_pos + i + 1) // ratio
        mask = torch.arange(n_comp_slots, device=x.device) >= n_valid
        iscore[:, i] += torch.where(mask, float('-inf'), 0.)

    tk = min(index_topk, n_comp_slots)
    cidx = iscore.topk(tk, dim=-1)[1] + win

    for i in range(S):
        n_valid = (start_pos + i + 1) // ratio
        bad = torch.arange(tk, device=x.device) >= n_valid
        cidx[:, i] = torch.where(bad, -1, cidx[:, i])

    # ---- topk_idxs ----
    wt = _window_topk(win, B, S, start_pos).to(x.device)
    topk = torch.cat([wt, cidx], dim=-1).int()

    # ---- sparse attention ----
    n_topk = topk.shape[-1]
    safe_idxs = topk.clamp(min=0)
    batch_idxs = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, S, n_topk)
    kg = kv_cache[batch_idxs, safe_idxs]
    kg_heads = kg.unsqueeze(1).expand(B, H, S, n_topk, Hd)

    scores   = torch.einsum('bhsd,bhskd->bhsk', q.float(), kg_heads.float())
    scores   = scores * (Hd ** -0.5)
    scores   = scores.masked_fill((topk == -1).unsqueeze(1), float('-inf'))
    smax     = scores.max(dim=-1, keepdim=True).values
    exp_s    = torch.exp(scores - smax)
    exp_sink = torch.exp(sinks.float().view(1, H, 1, 1) - smax)
    attn_w   = (exp_s / (exp_s.sum(dim=-1, keepdim=True) + exp_sink)).to(dtype)
    ao = torch.einsum('bhsk,bhskd->bhsd', attn_w, kg_heads)

    # ---- de-rope + output proj ----
    ao[..., -rd:] = _apply_rotary_emb(ao[..., -rd:], fc, inverse=True)
    groups = ao.transpose(1, 2).reshape(B, S, o_groups, -1)
    pg = (H * Hd) // o_groups
    oa_out = o_a_proj_weight.shape[0] // o_groups
    w_oa = o_a_proj_weight.view(o_groups, oa_out, pg).transpose(1, 2)
    gx = groups.reshape(B * S, o_groups, pg).transpose(0, 1)
    oa = torch.bmm(gx, w_oa).transpose(0, 1).reshape(B, S, -1)
    output = F.linear(oa, o_b_proj_weight)
    return output

