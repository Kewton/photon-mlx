"""LLaMA-style transformer blocks in MLX (shared by encoder and decoder)."""
from __future__ import annotations


import mlx.core as mx
import mlx.nn as nn


def precompute_rope(
    dim: int, max_len: int, theta: float = 1_000_000.0,
) -> tuple[mx.array, mx.array]:
    """Return (cos, sin) each of shape (max_len, dim // 2)."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(max_len).astype(mx.float32)
    angles = t[:, None] * freqs[None, :]  # (max_len, dim//2)
    return mx.cos(angles), mx.sin(angles)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: (B, H, T, D). cos/sin: (max_len, D//2). Apply rotary PE."""
    T = x.shape[2]
    cos_t = cos[:T].reshape(1, 1, T, -1)   # (1, 1, T, D//2)
    sin_t = sin[:T].reshape(1, 1, T, -1)
    # Split into pairs: (B, H, T, D) → (B, H, T, D//2, 2)
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = xr[..., 0], xr[..., 1]       # each (B, H, T, D//2)
    o1 = x1 * cos_t - x2 * sin_t
    o2 = x1 * sin_t + x2 * cos_t
    return mx.stack([o1, o2], axis=-1).reshape(x.shape)


class Attention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU FFN (LLaMA-style)."""
    def __init__(self, dim: int, ffn_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
        ffn_dim: int, norm_eps: float = 1e-5, bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, n_heads, n_kv_heads, head_dim, bias)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, ffn_dim, bias)

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def causal_mask(T: int) -> mx.array:
    """Upper-triangular -inf mask of shape (T, T)."""
    mask = mx.full((T, T), float("-inf"))
    mask = mx.triu(mask, k=1)
    return mask
