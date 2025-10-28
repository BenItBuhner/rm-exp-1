from typing import Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
import einops

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty(out_features, in_features), std=1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(num_embeddings, embedding_dim), std=init_std))
        self.dtype = dtype

    def forward(self, input_ids: torch.Tensor):
        return F.embedding(input_ids, self.weight.to(self.dtype))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Parameter(emb.cos(), requires_grad=False)
        self.sin_cached = nn.Parameter(emb.sin(), requires_grad=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, causal: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal

        kv_heads = max(1, num_heads // 2)
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * kv_heads) * head_dim, bias=False)
        self.out_proj = CastedLinear(num_heads * head_dim, hidden_size, bias=False)
        self.kv_heads = kv_heads

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor):
        b, s, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(b, s, self.num_heads + 2 * self.kv_heads, self.head_dim)
        q, k, v = qkv.split([self.num_heads, self.kv_heads, self.kv_heads], dim=2)
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = [einops.rearrange(x, "b s h d -> b h s d") for x in (q, k, v)]
        if k.shape[1] != self.num_heads:
            if self.num_heads % k.shape[1] != 0:
                k = k.expand(-1, self.num_heads, -1, -1)
                v = v.expand(-1, self.num_heads, -1, -1)
            else:
                repeat = self.num_heads // k.shape[1]
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = einops.rearrange(attn, "b h s d -> b s (h d)")
        return self.out_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)


def rms_norm(x: torch.Tensor, eps: float):
    dtype = x.dtype
    x = x.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return normed.to(dtype)
import math
