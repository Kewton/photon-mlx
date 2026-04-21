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

import warnings
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from math import prod

from .blocks import TransformerBlock, causal_mask, precompute_rope
from .optimize import pad_input_ids, pad_to_multiple

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
                self.chunkers.append(ConcatChunker(CS[0], m.hidden_size, m.hidden_size))
            else:
                self.chunkers.append(LinearChunker(CS[lv], m.hidden_size))
            self.encoders.append(
                [_make_block(cfg) for _ in range(h.encoder_layers_per_level[lv])]
            )

        # ── Top-down ────────────────────────────────────────────
        self.decoders: list[list[TransformerBlock]] = []
        self.converters: list[Converter] = []
        for lv in range(L):
            self.decoders.append(
                [_make_block(cfg) for _ in range(h.decoder_layers_per_level[lv])]
            )
            self.converters.append(Converter(m.hidden_size, CS[lv], PL[lv]))

        # ── Local decoder (token level) ─────────────────────────
        self.local_decoder = [
            _make_block(cfg) for _ in range(h.decoder_layers_per_level[0])
        ]

        # ── Output ──────────────────────────────────────────────
        self.output_norm = nn.RMSNorm(m.hidden_size, eps=m.norm_eps)
        self.lm_head = nn.Linear(m.hidden_size, t.vocab_size, bias=False)

        # ── Pre-computed RoPE ───────────────────────────────────
        max_local = max(c * (p + 1) for c, p in zip(CS, PL))
        self._rope_cos, self._rope_sin = precompute_rope(
            m.head_dim, m.max_position_embeddings, m.rope_theta
        )
        self._local_cos, self._local_sin = precompute_rope(
            m.head_dim, max_local, m.rope_theta
        )

    # ────────────────────────────────────────────────────────────
    # Decode one hierarchical level
    # ────────────────────────────────────────────────────────────
    def _decode_level(
        self,
        h_above: mx.array,  # (B, N_high, D) — decoder output from above
        enc_out: mx.array,  # (B, N_low, D)  — encoder output at this level
        converter: Converter,
        blocks: list[TransformerBlock],
        chunk_size: int,
    ) -> mx.array:
        """Decode one level: prefix from converter + chunked encoder output."""
        B, N_high, D = h_above.shape
        P = converter.prefix_length
        C = chunk_size
        N_low = N_high * C

        prefix = converter(h_above)  # (B, N_low, P, D)
        enc_chunked = enc_out.reshape(B, N_high, C, D)  # (B, N_high, C, D)
        _prefix_chunked = prefix.reshape(B, N_high, C, P, D)

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
            combined, _ = block(combined, self._local_cos, self._local_sin, mask)

        # Extract encoder positions (last C): (B * N_high, C, D)
        decoded = combined[:, C * P :, :]
        return decoded.reshape(B, N_low, D)

    # ────────────────────────────────────────────────────────────
    # Shared helpers (DR1-003 / DRY)
    # ────────────────────────────────────────────────────────────
    def _encode_bottom_up(self, input_ids: mx.array) -> list[mx.array]:
        """Bottom-up encoding: tokens → enc_outputs[0..L].

        Contract (DR3-001):
            input_ids must be chunk-aligned, i.e.
            ``T % prod(chunk_sizes) == 0``. Callers needing padding should
            pre-pad via ``pad_to_multiple``. Raises ``ValueError`` otherwise.

        Args:
            input_ids: (B, T) chunk-aligned token ids.

        Returns:
            enc_outputs: list of length L+1. Index 0 holds token projections
            (shape (B, T, D)); index k (1 <= k <= L) holds encoder output at
            level k-1 after chunker[k-1] + encoder[k-1].
        """
        h = self.cfg.hierarchy
        L = h.levels
        CS = h.chunk_sizes

        total_chunk = prod(CS)
        T = input_ids.shape[1]
        if T == 0 or T % total_chunk != 0:
            raise ValueError(
                f"_encode_bottom_up requires chunk-aligned length "
                f"(T % {total_chunk} == 0), got T={T}"
            )

        # 1. Embed + project
        tok = self.token_proj(self.token_embed(input_ids))  # (B, T, D)

        # 2. Bottom-up
        enc_outputs: list[mx.array] = [tok]  # index 0 = token projections
        x = tok
        for lv in range(L):
            x = self.chunkers[lv](x)
            seq_len = x.shape[1]
            mask = causal_mask(seq_len)
            for block in self.encoders[lv]:
                x, _ = block(x, self._rope_cos, self._rope_sin, mask)
            enc_outputs.append(x)

        return enc_outputs

    def _decode_from_enc_outputs(
        self,
        enc_outputs: list[mx.array],
        top_kv_cache: dict | None = None,
    ) -> tuple[mx.array, dict]:
        """Top-down decoding over enc_outputs → logits.

        Runs the full top-down stack in one helper: top-level decoder
        (``decoders[L-1]``) + lower decoders (``decoders[0]``) +
        ``local_decoder`` + output_norm + lm_head.

        Phase 1 constraint (DR1-003):
            KV cache is applied **only** to the top-level decoder
            (``decoders[L-1]``). ``decoders[0]`` and ``local_decoder`` always
            receive ``kv_cache=None`` and are fully recomputed. This is
            intentional — adding a per-level cache selector is Phase 2 scope.

        position_offset (DR1-006):
            Derived internally from the cached K/V length when a cache is
            provided (0 otherwise). Callers do not pass it.

        cache indexing (DR1-004 / DIP):
            Layer-indexed access into ``top_kv_cache`` is confined to this
            helper. Callers treat ``top_kv_cache`` as opaque.

        Append/replace (DR1-001):
            - ``T_top_new == T_top_cached + 1``  → append new slot.
            - ``T_top_new == T_top_cached``       → replace the last slot
              (in-progress super-chunk).
            Any other case raises ``ValueError``.

        Fail-closed guards (DR4-002):
            Validates cache length/shape and token length against
            ``max_position_embeddings`` before any allocation. Raises
            ``ValueError`` / ``RuntimeError`` on mismatch / overrun.

        Cache structure:
            The returned ``top_kv_cache`` is an opaque dict with keys:
              - ``"layers"``: list of per-layer (K, V) tuples, each
                ``(B, n_kv_heads, T_top, head_dim)``.
              - ``"top_out"``: the final top-level hidden state sequence
                ``(B, T_top, D)`` needed to feed the lower decoders on the
                next call.
            Callers treat this as opaque.

        Args:
            enc_outputs: list from ``_encode_bottom_up`` (length L+1).
            top_kv_cache: opaque cache dict from a prior call, or None for
                prefill.

        Returns:
            ``(logits, top_kv_cache_out)``.
        """
        h = self.cfg.hierarchy
        L = h.levels
        CS = h.chunk_sizes

        top_layers = self.decoders[L - 1]
        n_top_layers = len(top_layers)

        # ── Fail-closed validations (DR4-002 / DR4-003) ─────────────
        # Top-level token length check (DR4-002): the actual-token-length
        # guard lives in generate() / _generate_with_cache where actual_len
        # is known; here we additionally enforce T_top_new (post-chunking
        # top-level length) against max_position_embeddings so that the
        # RoPE table slice and KV allocation cannot overrun.
        top_seq = enc_outputs[L]
        B_top, T_top_new, _D = top_seq.shape

        max_pos = self.cfg.model.max_position_embeddings
        if T_top_new > max_pos:
            raise ValueError(
                f"Top-level sequence length {T_top_new} exceeds "
                f"max_position_embeddings={max_pos} (DR4-002)"
            )

        # Memory guard (DR4-002): Phase 1 soft-warn threshold is 50 MB
        # (design policy §9 quotes a 50 MB memory budget) and the hard
        # fail-closed ceiling stays at 500 MB. Crossing 50 MB emits a
        # RuntimeWarning so operators notice the Phase 1 budget was
        # exceeded without blocking; crossing 500 MB raises RuntimeError.
        # 2 (K+V) * layers * T_top_new * n_kv_heads * head_dim * 4 bytes (f32).
        m = self.cfg.model
        projected_kv_bytes = (
            2 * n_top_layers * T_top_new * m.num_key_value_heads * m.head_dim * 4
        )
        _PHASE1_SOFT_LIMIT = 50 * 1024 * 1024
        _HARD_LIMIT = 500 * 1024 * 1024
        if projected_kv_bytes >= _HARD_LIMIT:
            raise RuntimeError(
                f"Projected top-level KV cache allocation "
                f"{projected_kv_bytes / (1024 * 1024):.1f} MB exceeds "
                f"500 MB hard ceiling (DR4-002)"
            )
        if projected_kv_bytes >= _PHASE1_SOFT_LIMIT:
            warnings.warn(
                f"Projected top-level KV cache allocation "
                f"{projected_kv_bytes / (1024 * 1024):.1f} MB exceeds "
                f"Phase 1 soft limit 50 MB (design policy §9); "
                f"consider Phase 2 extension.",
                RuntimeWarning,
                stacklevel=2,
            )

        T_top_cached = 0
        cached_layers: list[tuple[mx.array, mx.array]] | None = None
        cached_top_out: mx.array | None = None
        if top_kv_cache is not None:
            if not isinstance(top_kv_cache, dict):
                raise ValueError(
                    f"top_kv_cache must be a dict (opaque cache), "
                    f"got {type(top_kv_cache).__name__}"
                )
            if "layers" not in top_kv_cache or "top_out" not in top_kv_cache:
                raise ValueError(
                    "top_kv_cache missing required keys {'layers', 'top_out'}"
                )
            cached_layers = top_kv_cache["layers"]
            cached_top_out = top_kv_cache["top_out"]
            if len(cached_layers) != n_top_layers:
                raise ValueError(
                    f"top_kv_cache['layers'] length mismatch: "
                    f"got {len(cached_layers)}, expected {n_top_layers}"
                )

            # Validate ALL layers (DR4-003): shape/dtype/T integrity must
            # hold for every layer, not just layer 0. A single corrupt
            # layer would otherwise surface as a cryptic concat error
            # later in the attention matmul.
            T_top_cached_seen: int | None = None
            expected_dtype = None
            for layer_idx, entry in enumerate(cached_layers):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] must be a "
                        f"2-tuple (K, V), got {type(entry).__name__}"
                    )
                k_l, v_l = entry
                if not isinstance(k_l, mx.array) or not isinstance(v_l, mx.array):
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] entries must "
                        f"be mx.array, got K={type(k_l).__name__}, "
                        f"V={type(v_l).__name__}"
                    )
                if k_l.shape != v_l.shape:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] K/V shape "
                        f"mismatch: {k_l.shape} vs {v_l.shape}"
                    )
                if k_l.ndim != 4:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] expected 4D "
                        f"(B, n_kv_heads, T, head_dim), got ndim={k_l.ndim}"
                    )
                expected_shape = (
                    B_top,
                    m.num_key_value_heads,
                    int(k_l.shape[-2]),
                    m.head_dim,
                )
                if tuple(k_l.shape) != expected_shape:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] K shape "
                        f"{k_l.shape} incompatible with expected "
                        f"{expected_shape}"
                    )
                if k_l.dtype != v_l.dtype:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] K/V dtype "
                        f"mismatch: {k_l.dtype} vs {v_l.dtype}"
                    )
                if expected_dtype is None:
                    expected_dtype = k_l.dtype
                elif k_l.dtype != expected_dtype:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] dtype "
                        f"{k_l.dtype} differs from layer 0 dtype "
                        f"{expected_dtype}"
                    )
                T_this = int(k_l.shape[-2])
                if T_top_cached_seen is None:
                    T_top_cached_seen = T_this
                elif T_this != T_top_cached_seen:
                    raise ValueError(
                        f"top_kv_cache['layers'][{layer_idx}] T={T_this} "
                        f"differs from layer 0 T={T_top_cached_seen}"
                    )
            T_top_cached = T_top_cached_seen or 0
            if cached_top_out.shape[1] != T_top_cached:
                raise ValueError(
                    f"top_kv_cache['top_out'] T={cached_top_out.shape[1]} "
                    f"mismatched with K/V T={T_top_cached}"
                )

        # ── Decide prefill / append / replace (SRP / DR1-001) ───────
        if cached_layers is None:
            mode = "prefill"
        elif T_top_new == T_top_cached + 1:
            mode = "append"
        elif T_top_new == T_top_cached and T_top_new >= 1:
            mode = "replace"
        else:
            raise ValueError(
                f"Cache length mismatch: cached T_top={T_top_cached}, "
                f"new enc_outputs[L].shape[1]={T_top_new}; expected "
                f"+1 (append) or equal (replace)"
            )

        # ── Top-level decoder with KV cache (layer-indexed) ─────────
        updated_layers: list[tuple[mx.array, mx.array]] = []

        if mode == "prefill":
            # Full top-level forward, build cache from scratch.
            h_dec = top_seq
            mask = causal_mask(T_top_new)
            for block in top_layers:
                h_dec, layer_cache = block(
                    h_dec,
                    self._rope_cos,
                    self._rope_sin,
                    mask,
                    kv_cache=None,
                    position_offset=0,
                )
                updated_layers.append(layer_cache)
            top_out_full = h_dec

        else:
            # "append" or "replace": feed only the new tail position into
            # each layer, using cached K/V as context. The full top-level
            # output sequence is reconstructed by concatenating the cached
            # prefix with the recomputed new tail.
            # DR4-003: explicit state integrity checks (not assert).
            if cached_layers is None:
                raise ValueError(
                    "cached_layers is None in append/replace path "
                    "(DR4-003 state integrity)"
                )
            if cached_top_out is None:
                raise ValueError(
                    "cached_top_out is None in append/replace path "
                    "(DR4-003 state integrity)"
                )

            new_tail = top_seq[:, -1:, :]  # (B, 1, D)

            if mode == "append":
                # Prefix cache is the full cached K/V (T_top_cached).
                # position_offset = T_top_cached (first new position).
                for layer_idx, block in enumerate(top_layers):
                    prev = cached_layers[layer_idx]
                    new_tail, layer_cache = block(
                        new_tail,
                        self._rope_cos,
                        self._rope_sin,
                        mask=None,
                        kv_cache=prev,
                        position_offset=T_top_cached,
                    )
                    updated_layers.append(layer_cache)
                # Final top hidden state sequence: cached prefix + new tail.
                top_out_full = mx.concatenate([cached_top_out, new_tail], axis=1)

            else:  # mode == "replace"
                # Drop the last slot from each cached layer; recompute it.
                for layer_idx, block in enumerate(top_layers):
                    k_prev, v_prev = cached_layers[layer_idx]
                    k_prefix = k_prev[..., :-1, :]
                    v_prefix = v_prev[..., :-1, :]
                    prefix_cache = (
                        (k_prefix, v_prefix) if k_prefix.shape[-2] > 0 else None
                    )
                    new_tail, layer_cache = block(
                        new_tail,
                        self._rope_cos,
                        self._rope_sin,
                        mask=None,
                        kv_cache=prefix_cache,
                        position_offset=T_top_cached - 1,
                    )
                    updated_layers.append(layer_cache)
                # Final top hidden state: cached prefix (minus last slot)
                # + new tail.
                top_out_full = mx.concatenate(
                    [cached_top_out[:, :-1, :], new_tail], axis=1
                )

            h_dec = top_out_full

        # ── Lower decoders + local decoder (always cache=None, DR1-003) ─
        for lv in reversed(range(1, L)):
            h_dec = self._decode_level(
                h_above=h_dec,
                enc_out=enc_outputs[lv],
                converter=self.converters[lv],
                blocks=self.decoders[lv - 1],
                chunk_size=CS[lv],
            )

        h_dec = self._decode_level(
            h_above=h_dec,
            enc_out=enc_outputs[0],
            converter=self.converters[0],
            blocks=self.local_decoder,
            chunk_size=CS[0],
        )

        logits = self.lm_head(self.output_norm(h_dec))  # (B, T, V)
        updated_cache: dict = {
            "layers": updated_layers,
            "top_out": top_out_full,
        }
        return logits, updated_cache

    # ────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────
    def __call__(
        self,
        input_ids: mx.array,
        labels: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        enc_outputs = self._encode_bottom_up(input_ids)
        logits, _ = self._decode_from_enc_outputs(enc_outputs, top_kv_cache=None)

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

    # ────────────────────────────────────────────────────────────
    # Greedy decode
    # ────────────────────────────────────────────────────────────
    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 64,
        return_logits: bool = False,
        # DR1-005 follow-up: kept public for equivalence oracle; Phase 2 will remove.
        use_kv_cache: bool = True,
    ) -> tuple[mx.array, mx.array | None]:
        """Greedy autoregressive decode with chunk-aligned padding.

        Args:
            input_ids: (B, T_prompt) pre-tokenized prompt.
            max_new_tokens: number of new tokens to generate.
            return_logits: if True, return per-step next-token logits.
            use_kv_cache: enable top-level KV cache (Phase 1 default path).
                ``False`` takes the full-recompute fallback path; retained as
                an internal testing helper to certify the cached path via
                ``TestKVCacheEquivalence`` and scheduled for removal once
                Phase 2 lands (see DR1-005 / Issue #54). Not part of the
                documented public API — the parameter is effectively
                private (``# noqa: DR1-005 follow-up``).

        Returns:
            (generated_ids, step_logits | None)
            generated_ids: (B, T_prompt + max_new_tokens)
            step_logits:   (B, max_new_tokens, V) if return_logits else None
        """
        # Token-level fail-closed guard (DR4-002 / CB-004): applied before the
        # use_kv_cache branch so both cached and nocache paths enforce the same
        # precondition. Keeps guard symmetry even if _generate_nocache is kept
        # as internal oracle.
        prompt_len = input_ids.shape[1]
        final_actual_len = prompt_len + max_new_tokens
        max_pos = self.cfg.model.max_position_embeddings
        if final_actual_len > max_pos:
            raise ValueError(
                f"final actual_len={final_actual_len} (prompt_len={prompt_len} + "
                f"max_new_tokens={max_new_tokens}) exceeds "
                f"max_position_embeddings={max_pos} (DR4-002)"
            )

        if not use_kv_cache:
            return self._generate_nocache(
                input_ids,
                max_new_tokens=max_new_tokens,
                return_logits=return_logits,
            )
        return self._generate_with_cache(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_logits=return_logits,
        )

    def _generate_nocache(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 64,
        return_logits: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Full-recompute greedy decode. Retained as a reference oracle for
        ``TestKVCacheEquivalence`` (DR1-005). TODO(#54 follow-up): delete once
        Phase 2 cache extension lands and the cached path is trusted.
        """
        total_chunk = prod(self.cfg.hierarchy.chunk_sizes)
        prompt_len = input_ids.shape[1]
        padded_len = pad_to_multiple(prompt_len + max_new_tokens, total_chunk)
        ids = pad_input_ids(input_ids, padded_len)
        actual_len = prompt_len

        step_logits_list: list[mx.array] | None = [] if return_logits else None

        for _step in range(max_new_tokens):
            logits, _loss = self.__call__(ids, labels=None)
            next_logits = logits[:, actual_len - 1, :]
            next_token = mx.argmax(next_logits, axis=-1)

            if step_logits_list is not None:
                step_logits_list.append(next_logits)

            ids[:, actual_len] = next_token
            actual_len += 1
            mx.eval(ids)

        step_logits = mx.stack(step_logits_list, axis=1) if step_logits_list else None
        return ids[:, : prompt_len + max_new_tokens], step_logits

    def _generate_with_cache(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 64,
        return_logits: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Greedy decode with Phase-1 top-level KV cache (DR1-003)."""
        total_chunk = prod(self.cfg.hierarchy.chunk_sizes)
        prompt_len = input_ids.shape[1]
        # Fail-closed token-length guard (DR4-002): enforce the actual
        # generated-token horizon against max_position_embeddings before
        # any encoder replay so long prompts cannot slip past the
        # top-level check in _decode_from_enc_outputs.
        max_pos = self.cfg.model.max_position_embeddings
        final_actual_len = prompt_len + max_new_tokens
        if final_actual_len > max_pos:
            raise ValueError(
                f"actual_len {final_actual_len} (prompt_len={prompt_len} + "
                f"max_new_tokens={max_new_tokens}) exceeds "
                f"max_position_embeddings={max_pos} (DR4-002)"
            )
        padded_len = pad_to_multiple(prompt_len + max_new_tokens, total_chunk)
        ids = pad_input_ids(input_ids, padded_len)
        actual_len = prompt_len

        step_logits_list: list[mx.array] | None = [] if return_logits else None
        top_kv_cache: dict | None = None

        for _step in range(max_new_tokens):
            # Chunk-aligned replay length (DR3-001).
            padded_actual = pad_to_multiple(actual_len, total_chunk)
            enc_outputs = self._encode_bottom_up(ids[:, :padded_actual])
            logits, top_kv_cache = self._decode_from_enc_outputs(
                enc_outputs, top_kv_cache=top_kv_cache
            )

            next_logits = logits[:, actual_len - 1, :]
            next_token = mx.argmax(next_logits, axis=-1)

            if step_logits_list is not None:
                step_logits_list.append(next_logits)

            ids[:, actual_len] = next_token
            actual_len += 1
            mx.eval(ids)

        step_logits = mx.stack(step_logits_list, axis=1) if step_logits_list else None
        return ids[:, : prompt_len + max_new_tokens], step_logits

    def count_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.parameters()))
