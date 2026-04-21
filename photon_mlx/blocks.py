"""LLaMA-style transformer blocks in MLX (shared by encoder and decoder)."""

from __future__ import annotations


import mlx.core as mx
import mlx.nn as nn


# Phase 1 KV cache type alias (DR1-002 / OCP).
# Phase 2 may replace this with a Protocol (.k / .v) without changing
# Attention's signature.
KVCache = tuple[mx.array, mx.array]


def _effective_positions_and_theta(
    dim: int,
    max_len: int,
    theta: float,
    scaling: str,
    scale_factor: float,
) -> tuple[mx.array, float]:
    """Return ``(positions, effective_theta)`` for the requested RoPE scaling.

    - ``scaling="none"``: identity (positions = arange(max_len), theta unchanged).
    - ``scaling="ntk"``: NTK-aware RoPE — positions unchanged, theta rescaled
      by ``scale_factor ** (dim / (dim - 2))``.

    Any other ``scaling`` value is treated as ``"none"``; the caller
    (``ModelConfig.__post_init__``) is responsible for validating the
    allowed set, so this helper stays permissive.
    """
    positions = mx.arange(max_len)
    if scaling == "ntk":
        eff_theta = theta * (scale_factor ** (dim / (dim - 2)))
        return positions, float(eff_theta)
    return positions, theta


def precompute_rope(
    dim: int,
    max_len: int,
    theta: float = 1_000_000.0,
    *,
    scaling: str = "none",
    scale_factor: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Return ``(cos, sin)`` each of shape ``(max_len, dim // 2)``.

    Args:
        dim: head dimension (``head_dim``).
        max_len: number of positions to precompute.
        theta: base RoPE frequency (e.g. ``1e6`` for LLaMA-style).
        scaling: RoPE position-scaling method. ``"none"`` (default) uses
            vanilla RoPE; ``"ntk"`` rescales ``theta`` per NTK-aware
            interpolation (Su et al. follow-up).
        scale_factor: multiplicative scale for ``"ntk"`` (must be >= 1.0).
            ``scaling="ntk"`` with ``scale_factor=1.0`` is mathematically
            equivalent to ``scaling="none"`` (the rescaling term
            ``scale_factor ** (dim / (dim - 2))`` collapses to 1.0).
    """
    t, eff_theta = _effective_positions_and_theta(
        dim, max_len, theta, scaling, scale_factor
    )
    freqs = 1.0 / (eff_theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = t.astype(mx.float32)
    angles = t[:, None] * freqs[None, :]  # (max_len, dim//2)
    return mx.cos(angles), mx.sin(angles)


def _apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    offset: int = 0,
) -> mx.array:
    """x: (B, H, T, D). cos/sin: (max_len, D//2). Apply rotary PE.

    Args:
        x: (B, H, T, D) input tensor to rotate.
        cos: (max_len, D//2) precomputed cosine table.
        sin: (max_len, D//2) precomputed sine table.
        offset: starting position index within ``cos/sin`` to read from.
            Default 0 preserves pre-Issue-54 behavior. Used by the KV-cache
            increment path to apply RoPE at the correct absolute position
            for freshly computed K/V (DR1-006).
    """
    T = x.shape[2]
    cos_t = cos[offset : offset + T].reshape(1, 1, T, -1)  # (1, 1, T, D//2)
    sin_t = sin[offset : offset + T].reshape(1, 1, T, -1)
    # Split into pairs: (B, H, T, D) → (B, H, T, D//2, 2)
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = xr[..., 0], xr[..., 1]  # each (B, H, T, D//2)
    o1 = x1 * cos_t - x2 * sin_t
    o2 = x1 * sin_t + x2 * cos_t
    return mx.stack([o1, o2], axis=-1).reshape(x.shape)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        kv_cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, KVCache]:
        """Attention with optional KV cache reuse.

        Args:
            x: (B, T, D) — T can be 1 for increment path.
            cos, sin: precomputed RoPE tables (shape ``(max_len, head_dim // 2)``).
            mask: additive attention bias ``(T_q, T_k)`` or None.
            kv_cache: optional ``(K_prev, V_prev)`` from a previous call.
                Shapes: ``(B, n_kv_heads, T_prev, head_dim)`` (pre-repeat).
                When provided, the freshly computed K/V for ``x`` are
                concatenated with the cached K/V and attention runs over
                the full (prev + new) key/value sequence.
            position_offset: RoPE absolute position for the first position in
                ``x``. Callers are responsible for computing this (typically
                ``T_prev`` when ``kv_cache`` is provided).

        Returns:
            ``(output, (K_full, V_full))`` where ``K_full``/``V_full`` are the
            concatenated K/V (cached prefix + new). The caller decides whether
            to *append* or *replace* the last slot (SRP / DR1-001).
        """
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k_new = (
            self.k_proj(x)
            .reshape(B, T, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v_new = (
            self.v_proj(x)
            .reshape(B, T, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        # Apply RoPE. Query and new K/V share the same absolute position range
        # starting at ``position_offset``.
        q = _apply_rope(q, cos, sin, offset=position_offset)
        k_new = _apply_rope(k_new, cos, sin, offset=position_offset)

        # Merge with cached K/V (cached entries already have RoPE baked in
        # from earlier calls — DR1-001 SRP: concat only, no replace logic).
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k_full = mx.concatenate([k_prev, k_new], axis=-2)
            v_full = mx.concatenate([v_prev, v_new], axis=-2)
        else:
            k_full = k_new
            v_full = v_new

        # Apply GQA head repeat after cache concat so the cached K/V stays
        # in the compact ``n_kv_heads`` shape (DR1-002 memory budget).
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_attn = mx.repeat(k_full, rep, axis=1)
            v_attn = mx.repeat(v_full, rep, axis=1)
        else:
            k_attn = k_full
            v_attn = v_full

        attn = (q @ k_attn.swapaxes(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v_attn).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(out), (k_full, v_full)


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
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        norm_eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, n_heads, n_kv_heads, head_dim, bias)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, ffn_dim, bias)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None = None,
        kv_cache: KVCache | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, KVCache]:
        """Pre-norm Transformer block with KV cache passthrough.

        Returns ``(output, updated_kv_cache)``. Callers that do not need
        the cache (prefill / encoders / local decoder) may discard the
        second tuple element with ``x, _ = block(...)``.
        """
        attn_out, new_cache = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


def causal_mask(T: int) -> mx.array:
    """Upper-triangular -inf mask of shape (T, T)."""
    mask = mx.full((T, T), float("-inf"))
    mask = mx.triu(mask, k=1)
    return mask
