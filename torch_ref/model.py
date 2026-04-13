"""
Minimal LLaMA-style decoder-only LM in PyTorch.

Used as the correctness reference for PHOTON's encoder/decoder blocks,
mask patterns, and forward pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PhotonConfig


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_rope(dim: int, max_len: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, T, D)  rope: (T, D//2)
    T = q.size(2)
    rope = rope[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(xc * rope).flatten(-2)
        return out.type_as(x)

    return rotate(q), rotate(k)


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
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, rope)

        # GQA expansion
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask[:T, :T]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward (LLaMA-style)."""

    def __init__(self, dim: int, ffn_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, n_heads, n_kv_heads, head_dim, bias)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, ffn_dim, bias)

    def forward(
        self, x: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MinimalLM(nn.Module):
    """Decoder-only LM matching LLaMA/PHOTON block structure."""

    def __init__(self, cfg: PhotonConfig) -> None:
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.tokenizer.vocab_size, m.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=m.hidden_size,
                n_heads=m.num_attention_heads,
                n_kv_heads=m.num_key_value_heads,
                head_dim=m.head_dim,
                ffn_dim=m.intermediate_size,
                norm_eps=m.norm_eps,
                bias=m.bias,
            )
            for _ in range(sum(cfg.hierarchy.encoder_layers_per_level))
        ])
        self.norm = RMSNorm(m.hidden_size, eps=m.norm_eps)
        self.lm_head = nn.Linear(m.hidden_size, cfg.tokenizer.vocab_size, bias=False)

        # Precompute causal mask and RoPE
        self.register_buffer(
            "causal_mask",
            torch.full((m.max_position_embeddings, m.max_position_embeddings),
                        float("-inf")).triu(1),
            persistent=False,
        )
        self.register_buffer(
            "rope",
            _precompute_rope(m.head_dim, m.max_position_embeddings, m.rope_theta),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        labels: torch.Tensor | None = None, # (B, T) for teacher forcing
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (logits, loss|None)."""
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope, self.causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def greedy_decode(
        self,
        input_ids: torch.Tensor,   # (B, T_prompt)
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedy autoregressive decode. Returns (B, T_prompt + generated)."""
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
