"""
B300_decode_onefunc.py — 完全自包含，不依赖 kernel.py / model.py 的 import。

tilelang kernel 定义（act_quant_kernel, fp4_quant_kernel, sparse_attn_kernel,
hc_split_sinkhorn_kernel）直接从 kernel.py 复制到本文件。

apply_rotary_emb / rotate_activation 从 model.py 源码复制为嵌套函数，
定义在两个大函数内部。

HCA (compress_ratio=128): SWA + 网格压缩, 无 Indexer
CSA (compress_ratio=4):   SWA + Indexer 稀疏 topk
"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from typing import Optional

# ===================== tilelang setup (copied from kernel.py) =====================

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


# ===================== helper functions (copied from kernel.py) =====================

def fast_log2_ceil(x):
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


# ===================== act_quant kernel + wrapper (copied from kernel.py) =====================

@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32,
    round_scale=False, inplace=False
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    # Internal computation in FP32; scale_dtype controls output storage format.
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP8, T.clamp(
                                x_local[i, j] / s_local[i], fp8_min, fp8_max
                            ))) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP)."""
    N = x.size(-1)
    assert N % block_size == 0
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    kernel = act_quant_kernel(
        N, block_size, scale_dtype=tl_dtype,
        round_scale=scale_fmt is not None, inplace=inplace,
    )
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


# ===================== fp4_quant kernel + wrapper (copied from kernel.py) =====================

@tilelang.jit(pass_configs=pass_configs)
def fp4_quant_kernel(
    N, block_size=32, in_dtype=BF16, scale_dtype=FE8M0, inplace=False
):
    """Block-wise FP4 quantization. Power-of-2 scale via bit ops. inplace=True does fused quant+dequant."""
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32
    group_size = block_size
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else FP4

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP4, T.clamp(
                                x_local[i, j] / s_local[i], -fp4_max, fp4_max
                            ))) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], -fp4_max, fp4_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_quant_kernel_


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP4 quantization. inplace=True does fused quant+dequant back to BF16."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
    kernel = fp4_quant_kernel(N, block_size, inplace=inplace)
    kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


# ===================== sparse_attn kernel + wrapper (copied from kernel.py) =====================

@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    """Sparse multi-head attention via index gathering + online softmax (FlashAttention-style).
    For each (batch, seq_pos), gathers top-k KV positions by index, computes attention
    with numerically stable running max/sum, and includes a learnable attn_sink bias."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] *= scale
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, o[by, bx, :, :])

    return sparse_attn_kernel_


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    b, s, h, d = q.size()
    # Pad heads to 16 for kernel efficiency (stripped after)
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
    o = torch.empty_like(q)
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o


# ===================== hc_split_sinkhorn kernel + wrapper (copied from kernel.py) =====================

@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base,
           pre.view(-1, hc_mult), post.view(-1, hc_mult), comb.view(-1, hc_mult, hc_mult))
    return pre, post, comb


# ===================== HCA decode (ratio=128) =====================

@torch.no_grad()
def deepseek_v4_hca_decode_onefunc(
    x, start_pos,
    kv_cache,
    freqs_cis,

    compressor_wkv_weight=None, compressor_wgate_weight=None,
    compressor_ape=None, compressor_norm_weight=None,
    comp_kv_state=None, comp_score_state=None,
    max_seq_len=4096,

    rms_norm_eps=1e-6, num_heads=128, head_dim=512, rope_head_dim=64,
    q_lora_rank=1536, o_groups=16, o_lora_rank=1024, window_size=128,

    q_a_proj_weight=None, q_a_norm_weight=None, q_b_proj_weight=None,
    kv_proj_weight=None, kv_norm_weight=None,
    sinks=None, o_a_proj_weight=None, o_b_proj_weight=None,

    # Hyper-Connections
    hc_mult=None,
    hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None,
    hc_sinkhorn_iters=20, hc_eps=1e-6,
    attn_norm_weight=None,
):
    """HCA decode (compress_ratio=128) — 完全自包含，不 import kernel.py / model.py。

    逐行对齐官方 model.py:
      Block.hc_pre / Block.hc_post / RMSNorm / Attention.forward (decode path)
      / Compressor.forward (decode path, overlap=False)
      / get_window_topk_idxs / get_compress_topk_idxs
    apply_rotary_emb 从 model.py 源码复制为嵌套函数。
    """
    # --- apply_rotary_emb (copied from model.py) ---
    def apply_rotary_emb(x, freqs_cis, inverse=False):
        y = x
        x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        if inverse:
            freqs_cis = freqs_cis.conj()
        if x.ndim == 3:
            freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
        else:
            freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        x = torch.view_as_real(x * freqs_cis).flatten(-2)
        y.copy_(x)
        return y

    dtype = x.dtype
    use_hc = hc_mult is not None and hc_mult > 1

    # ==================== HC pre (inline Block.hc_pre) ====================
    if use_hc:
        assert x.dim() == 4 and x.shape[2] == hc_mult, f"HC mode requires x [B,S,hc_mult,D], got {x.shape}"
        residual = x
        # --- Block.hc_pre ---
        hc_shape = x.size()
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
        mixes = F.linear(x, hc_attn_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_attn_scale, hc_attn_base, hc_mult, hc_sinkhorn_iters, hc_eps)
        x = torch.sum(pre.unsqueeze(-1) * x.view(hc_shape), dim=2).to(dtype)

        # --- attn_norm (RMSNorm) ---
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        x = (attn_norm_weight * (xf * torch.rsqrt(var + rms_norm_eps))).to(dtype)

    B, S, D = x.shape
    H, Hd, win, ratio, rd = num_heads, head_dim, window_size, 128, rope_head_dim
    scale_fmt = None
    scale_dtype = torch.float32

    fc = freqs_cis[start_pos:start_pos + S]

    # ==================== q (inline Attention.forward) ====================
    # qr = q = self.q_norm(self.wq_a(x))
    qr = F.linear(x, q_a_proj_weight)
    qrf = qr.float()
    qrf = qrf * torch.rsqrt(qrf.square().mean(-1, keepdim=True) + rms_norm_eps)
    qr = (q_a_norm_weight * qrf).to(dtype)
    # q = self.wq_b(q).unflatten(-1, (n_local_heads, head_dim))
    q = F.linear(qr, q_b_proj_weight).view(B, S, H, Hd)
    # q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + rms_norm_eps)
    # apply_rotary_emb(q[..., -rd:], freqs_cis)
    apply_rotary_emb(q[..., -rd:], fc)

    # ==================== kv (inline Attention.forward) ====================
    # kv = self.wkv(x)
    kv = F.linear(x, kv_proj_weight)
    # kv = self.kv_norm(kv)
    kvf = kv.float()
    kvf = kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + rms_norm_eps)
    kv = (kv_norm_weight * kvf).to(dtype)
    # apply_rotary_emb(kv[..., -rd:], freqs_cis)
    apply_rotary_emb(kv[..., -rd:], fc)
    # act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)

    # ==================== kv_cache window ring buffer ====================
    # self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    for i in range(S):
        kv_cache[:B, (start_pos + i) % win] = kv[:, i]

    # ==================== Compressor (inline Compressor.forward, decode path) ====================
    # HCA: ratio=128, overlap=False, coff=1, rotate=False, head_dim=Hd
    if comp_kv_state is None:
        comp_kv_state = torch.zeros(B, ratio, Hd, device=x.device, dtype=torch.float32)
    if comp_score_state is None:
        comp_score_state = torch.full((B, ratio, Hd), float('-inf'), device=x.device, dtype=torch.float32)

    for i in range(S):
        pos = start_pos + i
        # compression need fp32
        x_tok_f = x[:, i].float()
        # kv = self.wkv(x)   [B, 1, coff*Hd] → [B, Hd]
        c_kv = F.linear(x_tok_f, compressor_wkv_weight.float())
        # score = self.wgate(x)
        c_sc = F.linear(x_tok_f, compressor_wgate_weight.float())
        # score += self.ape[start_pos % ratio]
        c_sc = c_sc + compressor_ape[pos % ratio].float()

        should_compress = (pos + 1) % ratio == 0

        # overlap=False:
        # self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
        comp_kv_state[:B, pos % ratio] = c_kv
        # self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
        comp_score_state[:B, pos % ratio] = c_sc
        if should_compress:
            # kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
            c = (comp_kv_state[:B] * comp_score_state[:B].softmax(dim=1)).sum(dim=1)

        if not should_compress:
            continue

        # kv = self.norm(kv.to(dtype))
        cf = c.float()
        cf = cf * torch.rsqrt(cf.square().mean(-1, keepdim=True) + rms_norm_eps)
        c = (compressor_norm_weight * cf).to(dtype)

        # freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        idx = pos + 1 - ratio
        c_fc = freqs_cis[idx:idx + 1]
        # apply_rotary_emb(kv[..., -rd:], freqs_cis)
        apply_rotary_emb(c[..., -rd:].unsqueeze(1), c_fc)

        # rotate=False → act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        act_quant(c[..., :-rd], 64, scale_fmt, scale_dtype, True)

        # self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        kv_cache[:B, win + pos // ratio] = c

    # ==================== topk_idxs ====================
    # topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    if start_pos >= win - 1:
        wt_rows = []
        for i in range(S):
            pp = (start_pos + i) % win
            r = torch.cat([torch.arange(pp + 1, win), torch.arange(0, pp + 1)])
            wt_rows.append(r)
        wt = torch.stack(wt_rows).unsqueeze(0).expand(B, -1, -1)
    elif start_pos > 0:
        wt_rows = []
        for i in range(S):
            n_valid = start_pos + i + 1
            wt_rows.append(F.pad(torch.arange(n_valid), (0, win - n_valid), value=-1))
        wt = torch.stack(wt_rows).unsqueeze(0).expand(B, -1, -1)
    else:
        base = torch.arange(S).unsqueeze(1)
        wt = (base - win + 1).clamp(0) + torch.arange(min(S, win))
        wt = torch.where(wt > base, -1, wt)
        wt = wt.unsqueeze(0).expand(B, -1, -1)
    wt = wt.to(x.device)

    # compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset=win)
    n_slots = (start_pos + S) // ratio
    ct_rows = []
    for i in range(S):
        n_valid = (start_pos + i + 1) // ratio
        idxs = torch.arange(n_valid) + win
        ct_rows.append(F.pad(idxs, (0, n_slots - n_valid), value=-1))
    ct = torch.stack(ct_rows).unsqueeze(0).expand(B, -1, -1).to(x.device)

    # topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
    topk = torch.cat([wt, ct], dim=-1).int()

    # ==================== sparse attention (tilelang kernel) ====================
    # o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
    softmax_scale = Hd ** -0.5
    o = sparse_attn(q, kv_cache[:B], sinks, topk, softmax_scale)

    # ==================== de-rope + output proj (inline Attention.forward) ====================
    # apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    apply_rotary_emb(o[..., -rd:], fc, True)
    # o = o.view(bsz, seqlen, n_local_groups, -1)
    o = o.view(B, S, o_groups, -1)
    # wo_a = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
    wo_a = o_a_proj_weight.view(o_groups, o_lora_rank, -1)
    # o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    # x = self.wo_b(o.flatten(2))
    output = F.linear(o.flatten(2), o_b_proj_weight)

    # ==================== HC post (inline Block.hc_post) ====================
    if use_hc:
        attn_out = output
        # y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        y = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        output = y.type_as(attn_out)

    return output


# ===================== CSA decode (ratio=4) =====================

@torch.no_grad()
def deepseek_v4_csa_decode_onefunc(
    x, start_pos,
    kv_cache,
    freqs_cis,

    indexer_wq_b_weight=None, indexer_weights_proj_weight=None,
    index_n_heads=64, index_head_dim=128, index_topk=1024,
    idx_compressor_wkv_weight=None, idx_compressor_wgate_weight=None,
    idx_compressor_ape=None, idx_compressor_norm_weight=None,
    idx_kv_state=None, idx_score_state=None,
    idx_kv_cache=None,

    compressor_wkv_weight=None, compressor_wgate_weight=None,
    compressor_ape=None, compressor_norm_weight=None,
    comp_kv_state=None, comp_score_state=None,
    max_seq_len=4096,

    rms_norm_eps=1e-6, num_heads=128, head_dim=512, rope_head_dim=64,
    q_lora_rank=1536, o_groups=16, o_lora_rank=1024, window_size=128,

    q_a_proj_weight=None, q_a_norm_weight=None, q_b_proj_weight=None,
    kv_proj_weight=None, kv_norm_weight=None,
    sinks=None, o_a_proj_weight=None, o_b_proj_weight=None,

    # Hyper-Connections
    hc_mult=None,
    hc_attn_fn=None, hc_attn_scale=None, hc_attn_base=None,
    hc_sinkhorn_iters=20, hc_eps=1e-6,
    attn_norm_weight=None,
):
    """CSA decode (compress_ratio=4) — 完全自包含，不 import kernel.py / model.py。

    逐行对齐官方 model.py:
      Block.hc_pre / Block.hc_post / RMSNorm / Attention.forward (decode path)
      / Compressor.forward (decode path, overlap=True)
      / Indexer.forward (decode path)
      / get_window_topk_idxs
    apply_rotary_emb / rotate_activation 从 model.py 源码复制为嵌套函数。
    """
    # --- apply_rotary_emb (copied from model.py) ---
    def apply_rotary_emb(x, freqs_cis, inverse=False):
        y = x
        x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        if inverse:
            freqs_cis = freqs_cis.conj()
        if x.ndim == 3:
            freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
        else:
            freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        x = torch.view_as_real(x * freqs_cis).flatten(-2)
        y.copy_(x)
        return y

    # --- rotate_activation (copied from model.py) ---
    def rotate_activation(x):
        assert x.dtype == torch.bfloat16
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=x.size(-1) ** -0.5)

    dtype = x.dtype
    use_hc = hc_mult is not None and hc_mult > 1

    # ==================== HC pre (inline Block.hc_pre) ====================
    if use_hc:
        assert x.dim() == 4 and x.shape[2] == hc_mult, f"HC mode requires x [B,S,hc_mult,D], got {x.shape}"
        residual = x
        # --- Block.hc_pre ---
        hc_shape = x.size()
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
        mixes = F.linear(x, hc_attn_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_attn_scale, hc_attn_base, hc_mult, hc_sinkhorn_iters, hc_eps)
        x = torch.sum(pre.unsqueeze(-1) * x.view(hc_shape), dim=2).to(dtype)

        # --- attn_norm (RMSNorm) ---
        xf = x.float()
        var = xf.square().mean(-1, keepdim=True)
        x = (attn_norm_weight * (xf * torch.rsqrt(var + rms_norm_eps))).to(dtype)

    B, S, D = x.shape
    H, Hd, win, ratio, rd = num_heads, head_dim, window_size, 4, rope_head_dim
    iH, iHd = index_n_heads, index_head_dim
    scale_fmt = None
    scale_dtype = torch.float32

    fc = freqs_cis[start_pos:start_pos + S]

    # ==================== q (inline Attention.forward) ====================
    # qr = q = self.q_norm(self.wq_a(x))
    qr = F.linear(x, q_a_proj_weight)
    qrf = qr.float()
    qrf = qrf * torch.rsqrt(qrf.square().mean(-1, keepdim=True) + rms_norm_eps)
    qr = (q_a_norm_weight * qrf).to(dtype)
    # q = self.wq_b(q).unflatten(-1, (n_local_heads, head_dim))
    q = F.linear(qr, q_b_proj_weight).view(B, S, H, Hd)
    # q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + rms_norm_eps)
    # apply_rotary_emb(q[..., -rd:], freqs_cis)
    apply_rotary_emb(q[..., -rd:], fc)

    # ==================== kv (inline Attention.forward) ====================
    # kv = self.wkv(x)
    kv = F.linear(x, kv_proj_weight)
    # kv = self.kv_norm(kv)
    kvf = kv.float()
    kvf = kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + rms_norm_eps)
    kv = (kv_norm_weight * kvf).to(dtype)
    # apply_rotary_emb(kv[..., -rd:], freqs_cis)
    apply_rotary_emb(kv[..., -rd:], fc)
    # act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
    act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)

    # ==================== kv_cache window ring buffer ====================
    for i in range(S):
        kv_cache[:B, (start_pos + i) % win] = kv[:, i]

    # ==================== Main Compressor (inline Compressor.forward, decode path) ====================
    # CSA main: ratio=4, overlap=True, coff=2, rotate=False, head_dim=Hd
    if comp_kv_state is None:
        comp_kv_state = torch.zeros(B, 2 * ratio, 2 * Hd, device=x.device, dtype=torch.float32)
    if comp_score_state is None:
        comp_score_state = torch.full((B, 2 * ratio, 2 * Hd), float('-inf'), device=x.device, dtype=torch.float32)

    for i in range(S):
        pos = start_pos + i
        # compression need fp32
        x_tok_f = x[:, i].float()
        # kv = self.wkv(x)   [B, 1, coff*Hd] = [B, 1, 2*Hd]
        c_kv = F.linear(x_tok_f, compressor_wkv_weight.float())
        # score = self.wgate(x)
        c_sc = F.linear(x_tok_f, compressor_wgate_weight.float())
        # score += self.ape[start_pos % ratio]
        c_sc = c_sc + compressor_ape[pos % ratio].float()

        should_compress = (pos + 1) % ratio == 0

        # overlap=True:
        # self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
        comp_kv_state[:B, ratio + pos % ratio] = c_kv
        # self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
        comp_score_state[:B, ratio + pos % ratio] = c_sc
        if should_compress:
            # kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
            kv2 = torch.cat([comp_kv_state[:B, :ratio, :Hd], comp_kv_state[:B, ratio:, Hd:]], dim=1)
            # score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
            sc2 = torch.cat([comp_score_state[:B, :ratio, :Hd], comp_score_state[:B, ratio:, Hd:]], dim=1)
            # kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
            c = (kv2 * sc2.softmax(dim=1)).sum(dim=1)
            # self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
            comp_kv_state[:B, :ratio] = comp_kv_state[:B, ratio:]
            # self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            comp_score_state[:B, :ratio] = comp_score_state[:B, ratio:]

        if not should_compress:
            continue

        # kv = self.norm(kv.to(dtype))
        cf = c.float()
        cf = cf * torch.rsqrt(cf.square().mean(-1, keepdim=True) + rms_norm_eps)
        c = (compressor_norm_weight * cf).to(dtype)

        # freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        idx = pos + 1 - ratio
        c_fc = freqs_cis[idx:idx + 1]
        # apply_rotary_emb(kv[..., -rd:], freqs_cis)
        apply_rotary_emb(c[..., -rd:].unsqueeze(1), c_fc)

        # rotate=False → act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        act_quant(c[..., :-rd], 64, scale_fmt, scale_dtype, True)

        # self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        kv_cache[:B, win + pos // ratio] = c

    # ==================== Indexer Compressor (inline Compressor.forward, decode path) ====================
    # Indexer compressor: ratio=4, overlap=True, coff=2, rotate=True, head_dim=iHd
    if idx_kv_state is None:
        idx_kv_state = torch.zeros(B, 2 * ratio, 2 * iHd, device=x.device, dtype=torch.float32)
    if idx_score_state is None:
        idx_score_state = torch.full((B, 2 * ratio, 2 * iHd), float('-inf'), device=x.device, dtype=torch.float32)
    if idx_kv_cache is None:
        idx_kv_cache = torch.zeros(B, max_seq_len // ratio, iHd, device=x.device, dtype=dtype)

    for i in range(S):
        pos = start_pos + i
        # compression need fp32
        x_tok_f = x[:, i].float()
        # kv = self.wkv(x)   [B, 1, coff*iHd] = [B, 1, 2*iHd]
        c_kv = F.linear(x_tok_f, idx_compressor_wkv_weight.float())
        # score = self.wgate(x)
        c_sc = F.linear(x_tok_f, idx_compressor_wgate_weight.float())
        # score += self.ape[start_pos % ratio]
        c_sc = c_sc + idx_compressor_ape[pos % ratio].float()

        should_compress = (pos + 1) % ratio == 0

        # overlap=True:
        idx_kv_state[:B, ratio + pos % ratio] = c_kv
        idx_score_state[:B, ratio + pos % ratio] = c_sc
        if should_compress:
            kv2 = torch.cat([idx_kv_state[:B, :ratio, :iHd], idx_kv_state[:B, ratio:, iHd:]], dim=1)
            sc2 = torch.cat([idx_score_state[:B, :ratio, :iHd], idx_score_state[:B, ratio:, iHd:]], dim=1)
            c = (kv2 * sc2.softmax(dim=1)).sum(dim=1)
            idx_kv_state[:B, :ratio] = idx_kv_state[:B, ratio:]
            idx_score_state[:B, :ratio] = idx_score_state[:B, ratio:]

        if not should_compress:
            continue

        # kv = self.norm(kv.to(dtype))
        cf = c.float()
        cf = cf * torch.rsqrt(cf.square().mean(-1, keepdim=True) + rms_norm_eps)
        c = (idx_compressor_norm_weight * cf).to(dtype)

        # freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        idx = pos + 1 - ratio
        c_fc = freqs_cis[idx:idx + 1]
        # apply_rotary_emb(kv[..., -rd:], freqs_cis)
        apply_rotary_emb(c[..., -rd:].unsqueeze(1), c_fc)

        # rotate=True → kv = rotate_activation(kv); fp4_act_quant(kv, fp4_block_size, True)
        c = rotate_activation(c)
        fp4_act_quant(c, 32, True)

        # self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        idx_kv_cache[:B, pos // ratio] = c

    # ==================== Indexer (inline Indexer.forward, decode path) ====================
    # q = self.wq_b(qr)
    iq = F.linear(qr, indexer_wq_b_weight).view(B, S, iH, iHd)
    # apply_rotary_emb(q[..., -rd:], freqs_cis)
    apply_rotary_emb(iq[..., -rd:], fc)
    # q = rotate_activation(q)
    iq = rotate_activation(iq)
    # fp4_act_quant(q, fp4_block_size, True)
    fp4_act_quant(iq, 32, True)

    # weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    ep = start_pos + S
    n_comp_slots = ep // ratio
    wgt = F.linear(x, indexer_weights_proj_weight)
    wgt = wgt * ((iHd ** -0.5) * (iH ** -0.5))

    # index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
    iscore = torch.einsum("bshd,btd->bsht", iq, idx_kv_cache[:B, :n_comp_slots])
    # index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    iscore = (iscore.relu_() * wgt.unsqueeze(-1)).sum(dim=2)

    # Mask invalid slots (for S > 1; no-op for S=1, start_pos > 0)
    for i in range(S):
        n_valid = (start_pos + i + 1) // ratio
        mask = torch.arange(n_comp_slots, device=x.device) >= n_valid
        iscore[:, i] += torch.where(mask, float('-inf'), 0.)

    # topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
    tk = min(index_topk, n_comp_slots)
    cidx = iscore.topk(tk, dim=-1)[1]
    # topk_idxs += offset  (offset = win for start_pos > 0)
    cidx = cidx + win

    # Mask invalid topk entries (for S > 1; no-op for S=1, start_pos > 0)
    for i in range(S):
        n_valid = (start_pos + i + 1) // ratio
        bad = torch.arange(tk, device=x.device) >= n_valid
        cidx[:, i] = torch.where(bad, -1, cidx[:, i])

    # ==================== topk_idxs ====================
    # topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
    if start_pos >= win - 1:
        wt_rows = []
        for i in range(S):
            pp = (start_pos + i) % win
            r = torch.cat([torch.arange(pp + 1, win), torch.arange(0, pp + 1)])
            wt_rows.append(r)
        wt = torch.stack(wt_rows).unsqueeze(0).expand(B, -1, -1)
    elif start_pos > 0:
        wt_rows = []
        for i in range(S):
            n_valid = start_pos + i + 1
            wt_rows.append(F.pad(torch.arange(n_valid), (0, win - n_valid), value=-1))
        wt = torch.stack(wt_rows).unsqueeze(0).expand(B, -1, -1)
    else:
        base = torch.arange(S).unsqueeze(1)
        wt = (base - win + 1).clamp(0) + torch.arange(min(S, win))
        wt = torch.where(wt > base, -1, wt)
        wt = wt.unsqueeze(0).expand(B, -1, -1)
    wt = wt.to(x.device)

    # topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
    topk = torch.cat([wt, cidx], dim=-1).int()

    # ==================== sparse attention (tilelang kernel) ====================
    # o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
    softmax_scale = Hd ** -0.5
    o = sparse_attn(q, kv_cache[:B], sinks, topk, softmax_scale)

    # ==================== de-rope + output proj (inline Attention.forward) ====================
    # apply_rotary_emb(o[..., -rd:], freqs_cis, True)
    apply_rotary_emb(o[..., -rd:], fc, True)
    # o = o.view(bsz, seqlen, n_local_groups, -1)
    o = o.view(B, S, o_groups, -1)
    # wo_a = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
    wo_a = o_a_proj_weight.view(o_groups, o_lora_rank, -1)
    # o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    # x = self.wo_b(o.flatten(2))
    output = F.linear(o.flatten(2), o_b_proj_weight)

    # ==================== HC post (inline Block.hc_post) ====================
    if use_hc:
        attn_out = output
        # y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        y = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        output = y.type_as(attn_out)

    return output
