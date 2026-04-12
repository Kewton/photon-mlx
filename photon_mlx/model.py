"""
PHOTON hierarchical decoder model in MLX.

Architecture (2-level, chunk_sizes=[4,4]):

  Bottom-up (encoding):
    tokens → embed → project → chunker[0] → encoder[0] → chunker[1] → encoder[1]

  Top-down (decoding):
    encoder[1] output → decoder[1] (top, no prefix)
      → converter[1] → decoder[0] (with prefix + encoder[0] output)
      → converter[0] → local_decoder (with prefix + token projections)
      → lm_head → logits
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .blocks import TransformerBlock, causal_mask, precompute_rope

# ── sys.path for shared config ──────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_ref.config import PhotonConfig  # noqa: E402


# ================================================================
# Sub-modules
# ================================================================

class ConcatChunker(nn.Module):
    """Level-1 chunker: concatenate C embeddings and project."""
    def __init__(self, chunk_size: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Linear(chunk_size * in_dim, out_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        C = self.chunk_size
        return self.proj(x.reshape(B, T // C, C * D))


class LinearChunker(nn.Module):
    """Upper-level chunker: linear projection of concatenated reps."""
    def __init__(self, chunk_size: int, dim: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.proj = nn.Linear(chunk_size * dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape
        C = self.chunk_size
        return self.proj(x.reshape(B, N // C, C * D))


class Converter(nn.Module):
    """Expand each higher-level rep into chunk_size sets of prefix_length vectors."""
    def __init__(self, dim: int, chunk_size: int, prefix_length: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.prefix_length = prefix_length
        self.pos_embed = mx.zeros((chunk_size, dim))  # learnable
        self.proj = nn.Linear(dim, prefix_length * dim, bias=False)
        self.norm = nn.RMSNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape
        C, P = self.chunk_size, self.prefix_length
        # Repeat and add positional differentiation
        x = mx.broadcast_to(x[:, :, None, :], (B, N, C, D))
        x = x + self.pos_embed[None, None, :, :]
        x = x.reshape(B, N * C, D)
        out = self.proj(x).reshape(B, N * C, P, D)
        return self.norm(out)


# ================================================================
# Full model
# ================================================================

def _make_block(cfg: PhotonConfig) -> TransformerBlock:
    m = cfg.model
    return TransformerBlock(
        dim=m.hidden_size,
        n_heads=m.num_attention_heads,
        n_kv_heads=m.num_key_value_heads,
        head_dim=m.head_dim,
        ffn_dim=m.intermediate_size,
        norm_eps=m.norm_eps,
        bias=m.bias,
    )


class PhotonModel(nn.Module):
    def __init__(self, cfg: PhotonConfig) -> None:
        super().__init__()
        self.cfg = cfg
        m, h, t = cfg.model, cfg.hierarchy, cfg.tokenizer
        L = h.levels
        CS = h.chunk_sizes
        PL = h.converter_prefix_lengths

        # ── Embedding ───────────────────────────────────────────
        self.token_embed = nn.Embedding(t.vocab_size, m.base_embed_dim)
        self.token_proj = nn.Linear(m.base_embed_dim, m.hidden_size, bias=False)

        # ── Bottom-up ───────────────────────────────────────────
        self.chunkers: list[Any] = []
        self.encoders: list[list[TransformerBlock]] = []
        for lv in range(L):
            if lv == 0:
                self.chunkers.append(
                    ConcatChunker(CS[0], m.hidden_size, m.hidden_size))
            else:
                self.chunkers.append(LinearChunker(CS[lv], m.hidden_size))
            self.encoders.append(
                [_make_block(cfg) for _ in range(h.encoder_layers_per_level[lv])])

        # ── Top-down ────────────────────────────────────────────
        self.decoders: list[list[TransformerBlock]] = []
        self.converters: list[Converter] = []
        for lv in range(L):
            self.decoders.append(
                [_make_block(cfg) for _ in range(h.decoder_layers_per_level[lv])])
            self.converters.append(Converter(m.hidden_size, CS[lv], PL[lv]))

        # ── Local decoder (token level) ─────────────────────────
        self.local_decoder = [
            _make_block(cfg) for _ in range(h.decoder_layers_per_level[0])]

        # ── Output ──────────────────────────────────────────────
        self.output_norm = nn.RMSNorm(m.hidden_size, eps=m.norm_eps)
        self.lm_head = nn.Linear(m.hidden_size, t.vocab_size, bias=False)

        # ── Pre-computed RoPE ───────────────────────────────────
        max_local = max(c * (p + 1) for c, p in zip(CS, PL))
        self._rope_cos, self._rope_sin = precompute_rope(
            m.head_dim, m.max_position_embeddings, m.rope_theta)
        self._local_cos, self._local_sin = precompute_rope(
            m.head_dim, max_local, m.rope_theta)

    # ────────────────────────────────────────────────────────────
    # Decode one hierarchical level
    # ────────────────────────────────────────────────────────────
    def _decode_level(
        self,
        h_above: mx.array,        # (B, N_high, D) — decoder output from above
        enc_out: mx.array,         # (B, N_low, D)  — encoder output at this level
        converter: Converter,
        blocks: list[TransformerBlock],
        chunk_size: int,
    ) -> mx.array:
        """Decode one level: prefix from converter + chunked encoder output."""
        B, N_high, D = h_above.shape
        P = converter.prefix_length
        C = chunk_size
        N_low = N_high * C

        prefix = converter(h_above)                    # (B, N_low, P, D)
        enc_chunked = enc_out.reshape(B, N_high, C, D) # (B, N_high, C, D)
        prefix_chunked = prefix.reshape(B, N_high, C, P, D)

        # Per-chunk: [P prefix vecs, 1 encoder vec] = P+1 positions
        # But actually each super-chunk has C encoder positions.
        # So per super-chunk: [P prefix, C encoder positions] = P+C positions
        # Wait — prefix is per-lower-level-position, not per-chunk.
        # prefix shape after converter: (B, N_low, P, D)
        # We want per-CHUNK processing where each chunk is one super-chunk
        # of C lower-level positions.

        # Reshape prefix to per-super-chunk: (B, N_high, C*P, D)
        prefix_flat = prefix.reshape(B, N_high, C * P, D)

        # Concatenate: (B, N_high, C*P + C, D)
        combined = mx.concatenate([prefix_flat, enc_chunked], axis=2)
        L = C * P + C

        # Batch all super-chunks: (B * N_high, L, D)
        combined = combined.reshape(B * N_high, L, D)
        mask = causal_mask(L)

        for block in blocks:
            combined = block(combined, self._local_cos, self._local_sin, mask)

        # Extract encoder positions (last C): (B * N_high, C, D)
        decoded = combined[:, C * P:, :]
        return decoded.reshape(B, N_low, D)

    # ────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────
    def __call__(
        self,
        input_ids: mx.array,
        labels: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        h = self.cfg.hierarchy
        L = h.levels
        CS = h.chunk_sizes

        B, T = input_ids.shape

        # 1. Embed + project
        tok = self.token_proj(self.token_embed(input_ids))  # (B, T, D)

        # 2. Bottom-up
        enc_outputs = [tok]   # index 0 = token projections
        x = tok
        for lv in range(L):
            x = self.chunkers[lv](x)
            seq_len = x.shape[1]
            mask = causal_mask(seq_len)
            for block in self.encoders[lv]:
                x = block(x, self._rope_cos, self._rope_sin, mask)
            enc_outputs.append(x)

        # 3. Top-down
        # 3a. Top-level decoder (no prefix)
        h_dec = enc_outputs[L]          # (B, N_top, D)
        seq_len = h_dec.shape[1]
        mask = causal_mask(seq_len)
        for block in self.decoders[L - 1]:
            h_dec = block(h_dec, self._rope_cos, self._rope_sin, mask)

        # 3b. Descend through hierarchy
        for lv in reversed(range(1, L)):
            h_dec = self._decode_level(
                h_above=h_dec,
                enc_out=enc_outputs[lv],
                converter=self.converters[lv],
                blocks=self.decoders[lv - 1],
                chunk_size=CS[lv],
            )

        # 3c. Local decoder (token level)
        h_dec = self._decode_level(
            h_above=h_dec,
            enc_out=enc_outputs[0],
            converter=self.converters[0],
            blocks=self.local_decoder,
            chunk_size=CS[0],
        )

        # 4. LM head
        logits = self.lm_head(self.output_norm(h_dec))   # (B, T, V)

        # 5. Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = mx.mean(
                nn.losses.cross_entropy(
                    shift_logits.reshape(-1, logits.shape[-1]),
                    shift_labels.reshape(-1),
                )
            )

        return logits, loss

    def count_parameters(self) -> int:
        from mlx.utils import tree_flatten
        return sum(v.size for _, v in tree_flatten(self.parameters()))
