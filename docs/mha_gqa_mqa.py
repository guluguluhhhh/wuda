"""Compact, interview-friendly implementations of MHA, GQA, and MQA.

The three variants share the same attention math. Only the number of KV heads
changes:

    MHA: num_kv_heads == num_query_heads
    GQA: 1 < num_kv_heads < num_query_heads
    MQA: num_kv_heads == 1

Input/output shape: [batch, sequence, d_model].
The KV cache keeps only the original KV heads, so GQA/MQA retain their memory
savings during decode.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


KVCache = Tuple[Tensor, Tensor]  # Each tensor: [B, H_kv, T_cached, D_head]


def repeat_kv(x: Tensor, num_query_heads: int) -> Tensor:
    """Share each KV head across its group of query heads."""
    batch, num_kv_heads, seq_len, head_dim = x.shape
    if num_kv_heads == num_query_heads:
        return x

    repeats = num_query_heads // num_kv_heads
    return (
        x[:, :, None, :, :]
        .expand(batch, num_kv_heads, repeats, seq_len, head_dim)
        .reshape(batch, num_query_heads, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """GQA core; setting H_kv=H_q gives MHA and H_kv=1 gives MQA."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads

        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_query_heads * self.head_dim, d_model, bias=bias)

    def _split_heads(self, x: Tensor, num_heads: int) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
        *,
        causal: bool = True,
        return_cache: bool = False,
    ) -> Tensor | Tuple[Tensor, KVCache]:
        """Run self-attention over new tokens, optionally appending past KV."""
        batch, query_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), self.num_query_heads)
        k_new = self._split_heads(self.k_proj(x), self.num_kv_heads)
        v_new = self._split_heads(self.v_proj(x), self.num_kv_heads)

        if kv_cache is None:
            k, v = k_new, v_new
            past_len = 0
        else:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k_new), dim=2)
            v = torch.cat((past_v, v_new), dim=2)
            past_len = past_k.size(2)

        # RoPE, when needed, is applied to q and k_new before k_new is cached.
        k_for_attn = repeat_kv(k, self.num_query_heads)
        v_for_attn = repeat_kv(v, self.num_query_heads)

        scores = torch.matmul(q, k_for_attn.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal:
            key_len = k.size(2)
            query_positions = past_len + torch.arange(query_len, device=x.device)
            key_positions = torch.arange(key_len, device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(~allowed[None, None, :, :], float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v_for_attn)
        context = context.transpose(1, 2).contiguous().view(batch, query_len, self.d_model)
        output = self.out_proj(context)

        if return_cache:
            return output, (k, v)
        return output


class MultiHeadAttention(GroupedQueryAttention):
    def __init__(self, d_model: int, num_heads: int, bias: bool = False) -> None:
        super().__init__(d_model, num_heads, num_heads, bias)


class MultiQueryAttention(GroupedQueryAttention):
    def __init__(self, d_model: int, num_query_heads: int, bias: bool = False) -> None:
        super().__init__(d_model, num_query_heads, 1, bias)


def _demo() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64)

    models = {
        "MHA": MultiHeadAttention(64, num_heads=8),
        "GQA": GroupedQueryAttention(64, num_query_heads=8, num_kv_heads=2),
        "MQA": MultiQueryAttention(64, num_query_heads=8),
    }

    for name, model in models.items():
        full = model(x)

        # Prefill four tokens, then decode one token from the compact KV cache.
        _, cache = model(x[:, :4], return_cache=True)
        decoded = model(x[:, 4:], kv_cache=cache)

        torch.testing.assert_close(decoded, full[:, 4:], rtol=1e-5, atol=1e-6)
        print(
            f"{name}: output={tuple(full.shape)}, "
            f"cached_k={tuple(cache[0].shape)}, "
            f"parameters={sum(p.numel() for p in model.parameters()):,}"
        )


if __name__ == "__main__":
    _demo()
