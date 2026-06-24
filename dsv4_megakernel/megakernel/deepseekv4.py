import torch
import torch.nn.functional as F

def deepseek_v4_decoder_layer_forward_single_function(
    hidden_states,                    # [B, S, HC, D]
    attention_mask=None,              # [B, 1, S, K] or None

    # ===== config =====
    hc_mult=None,
    hc_sinkhorn_iters=3,
    hc_eps=1e-6,
    rms_norm_eps=1e-6,
    num_heads=None,
    head_dim=None,
    hidden_size=None,
    q_lora_rank=None,
    o_groups=None,
    o_lora_rank=None,
    swiglu_limit=7.0,

    # ===== attn hc weights =====
    attn_hc_fn=None,                  # [(2+HC)*HC, HC*D]
    attn_hc_base=None,                # [(2+HC)*HC]
    attn_hc_scale=None,               # [3]

    # ===== layernorm before attn =====
    input_layernorm_weight=None,      # [D]

    # ===== attention weights =====
    q_a_proj_weight=None,             # [q_lora_rank, D]
    q_a_norm_weight=None,             # [q_lora_rank]
    q_b_proj_weight=None,             # [num_heads*head_dim, q_lora_rank]

    kv_proj_weight=None,              # [head_dim, D]
    kv_norm_weight=None,              # [head_dim]

    sinks=None,                       # [num_heads]

    o_a_proj_weight=None,             # [o_groups*o_lora_rank, num_heads*head_dim//o_groups]
    o_b_proj_weight=None,             # [D, o_groups*o_lora_rank]

    # ===== rope cos/sin =====
    cos=None,                         # [B, S, rope_dim/2] or [B, S, head_dim/2] for full rope
    sin=None,                         # [B, S, rope_dim/2] or [B, S, head_dim/2]

    # ===== ffn hc weights =====
    ffn_hc_fn=None,                   # [(2+HC)*HC, HC*D]
    ffn_hc_base=None,                 # [(2+HC)*HC]
    ffn_hc_scale=None,                # [3]

    # ===== layernorm before mlp =====
    post_attention_layernorm_weight=None,   # [D]

    # ===== mlp weights (simple SwiGLU version, not MoE) =====
    mlp_gate_proj_weight=None,        # [intermediate, D]
    mlp_up_proj_weight=None,          # [intermediate, D]
    mlp_down_proj_weight=None,        # [D, intermediate]
):
    dtype = hidden_states.dtype
    B, S, HC, D = hidden_states.shape

    # =========================
    # 1. attn_hc(hidden_states)
    # =========================
    flat = hidden_states.reshape(B, S, HC * D).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_norm_eps) # RMSNorm With out weight

    # Gemv, split k
    mix = F.linear(flat, attn_hc_fn.float())  # [B, S, (2+HC)*HC]
    pre_w, post_w, comb_w = torch.split(mix, [HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = torch.split(attn_hc_base, [HC, HC, HC * HC], dim=0)
    pre_scale, post_scale, comb_scale = attn_hc_scale[0], attn_hc_scale[1], attn_hc_scale[2]
 
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(B, S, HC, HC) * comb_scale + comb_b.view(HC, HC)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    # cuda core functions
    for _ in range(hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(dtype)   # [B, S, D]

    # =========================
    # 2. input_layernorm(collapsed)
    # =========================
    x = collapsed.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    x = (x.to(dtype) * input_layernorm_weight)   # [B, S, D]

    # =========================
    # 3. q_a_proj -> q_a_norm
    # =========================
    # batched gemm: [batch, seqlen, dim] * [dim, q_lora_rank]
    q_residual = F.linear(x, q_a_proj_weight)   # [B, S, q_lora_rank]
    q_residual_f = q_residual.float()
    q_residual_f = q_residual_f * torch.rsqrt(q_residual_f.square().mean(-1, keepdim=True) + rms_norm_eps)
    q_residual = q_residual_f.to(dtype) * q_a_norm_weight

    # =========================
    # 4. q_b_proj -> reshape -> q_b_norm
    # =========================
    q = F.linear(q_residual, q_b_proj_weight)   # [B, S, H*Hd]
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)   # [B, H, S, Hd]
    qf = q.float()
    q = (qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + rms_norm_eps)).to(dtype)

    # =========================
    # 5. kv_proj -> kv_norm
    # =========================
    kv = F.linear(x, kv_proj_weight)   # [B, S, Hd]
    kvf = kv.float()
    kv = (kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + rms_norm_eps)).to(dtype) * kv_norm_weight
    kv = kv.view(B, S, 1, head_dim).transpose(1, 2)   # [B, 1, S, Hd]
    # =========================
    # 6. apply_rotary_pos_emb(q, kv)
    # 这里按文件里的 interleaved rope 逻辑手写
    # 假设 cos/sin 的最后一维是 rope_dim/2
    # rope 作用在最后 2 * cos.shape[-1] 维上
    # =========================
    rope_dim = cos.shape[-1] * 2
    cos_full = cos.repeat_interleave(2, dim=-1).unsqueeze(1)   # [B,1,S,rope_dim]
    sin_full = sin.repeat_interleave(2, dim=-1).unsqueeze(1)

    q_nope = q[..., :-rope_dim]
    q_rope = q[..., -rope_dim:]
    q_rope_half1 = q_rope[..., ::2]
    q_rope_half2 = q_rope[..., 1::2]
    q_rot = torch.stack((-q_rope_half2, q_rope_half1), dim=-1).reshape_as(q_rope)
    q_rope = ((q_rope.float() * cos_full) + (q_rot.float() * sin_full)).to(dtype)
    q = torch.cat([q_nope, q_rope], dim=-1)

    kv_nope = kv[..., :-rope_dim]
    kv_rope = kv[..., -rope_dim:]
    kv_rope_half1 = kv_rope[..., ::2]
    kv_rope_half2 = kv_rope[..., 1::2]
    kv_rot = torch.stack((-kv_rope_half2, kv_rope_half1), dim=-1).reshape_as(kv_rope)
    kv_rope = ((kv_rope.float() * cos_full) + (kv_rot.float() * sin_full)).to(dtype)
    kv = torch.cat([kv_nope, kv_rope], dim=-1)

    # =========================
    # 7. shared-KV attention
    # kv: [B,1,S,Hd] -> broadcast to all heads
    # =========================
    k = kv.expand(B, num_heads, S, head_dim)
    v = kv.expand(B, num_heads, S, head_dim)

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * (head_dim ** -0.5)   # [B,H,S,S]
    scores = scores + sinks.view(1, num_heads, 1, 1).float()

    if attention_mask is not None:
        scores = scores + attention_mask.float()

    attn_weights = torch.softmax(scores, dim=-1).to(dtype)
    attn_output = torch.matmul(attn_weights, v)   # [B,H,S,Hd]

    # =========================
    # 8. 对 attn_output 应用共轭旋转: apply_rotary_pos_emb(..., cos, -sin)
    # 文件里是 attn_output.transpose(1,2) 后做，再转回去
    # =========================
    y = attn_output.transpose(1, 2)   # [B,S,H,Hd]
    cos_full2 = cos.repeat_interleave(2, dim=-1).unsqueeze(2)   # [B,S,1,rope_dim]
    sin_full2 = (-sin).repeat_interleave(2, dim=-1).unsqueeze(2)

    y_nope = y[..., :-rope_dim]
    y_rope = y[..., -rope_dim:]
    y_half1 = y_rope[..., ::2]
    y_half2 = y_rope[..., 1::2]
    y_rot = torch.stack((-y_half2, y_half1), dim=-1).reshape_as(y_rope)
    y_rope = ((y_rope.float() * cos_full2) + (y_rot.float() * sin_full2)).to(dtype)
    y = torch.cat([y_nope, y_rope], dim=-1)
    attn_output = y.transpose(1, 2)   # [B,H,S,Hd]

    # =========================
    # 9. grouped output projection
    # grouped = attn_output.reshape(B,S,o_groups,-1)
    # o_a_proj 是 grouped linear
    # =========================
    grouped = attn_output.transpose(1, 2).reshape(B, S, o_groups, (num_heads * head_dim) // o_groups)

    hidden_dim_per_group = (num_heads * head_dim) // o_groups
    oa_out = o_a_proj_weight.shape[0] // o_groups

    w = o_a_proj_weight.view(o_groups, oa_out, hidden_dim_per_group).transpose(1, 2)   # [G, in, out]
    gx = grouped.reshape(B * S, o_groups, hidden_dim_per_group).transpose(0, 1)         # [G, BS, in]
    gy = torch.bmm(gx, w).transpose(0, 1)                                                # [BS, G, out]
    grouped = gy.reshape(B, S, o_groups, oa_out).flatten(2)                             # [B,S,G*out]

    attn_output_proj = F.linear(grouped, o_b_proj_weight)   # [B,S,D]

    # =========================
    # 10. attention residual merge
    # hidden_states = post * attn_output + comb^T * hidden_states
    # =========================
    hidden_states = (
        post.to(dtype).unsqueeze(-1) * attn_output_proj.unsqueeze(-2)
        + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states)
    )   # [B,S,HC,D]

    # =========================
    # 11. ffn_hc(hidden_states)
    # =========================
    flat = hidden_states.reshape(B, S, HC * D).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_norm_eps)

    mix = F.linear(flat, ffn_hc_fn.float())
    pre_w, post_w, comb_w = torch.split(mix, [HC, HC, HC * HC], dim=-1)
    pre_b, post_b, comb_b = torch.split(ffn_hc_base, [HC, HC, HC * HC], dim=0)
    pre_scale, post_scale, comb_scale = ffn_hc_scale[0], ffn_hc_scale[1], ffn_hc_scale[2]

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(B, S, HC, HC) * comb_scale + comb_b.view(HC, HC)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(dtype)   # [B,S,D]

    # =========================
    # 12. post_attention_layernorm
    # =========================
    x = collapsed.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    x = (x.to(dtype) * post_attention_layernorm_weight)

    # =========================
    # 13. simple SwiGLU MLP
    # 注意：这里为了简化，没有展开源码里的 SparseMoeBlock
    # =========================
    gate = F.linear(x, mlp_gate_proj_weight).clamp(max=swiglu_limit)
    up = F.linear(x, mlp_up_proj_weight).clamp(min=-swiglu_limit, max=swiglu_limit)
    mlp_hidden = F.silu(gate) * up
    mlp_output = F.linear(mlp_hidden, mlp_down_proj_weight)

    # =========================
    # 14. ffn residual merge
    # =========================
    hidden_states = (
        post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2)
        + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states)
    )   # [B,S,HC,D]

    return hidden_states
