"""
PHOTON session inference pipeline.

Wraps the PhotonModel for multi-turn session usage:
- Hierarchical prefill from evidence pack
- Session state update with drift tracking
- Answer-time local refresh
- Grounded generation via conditioning
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .model import PhotonModel
from .session import DriftMetrics, HierarchicalState, PhotonSessionState

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_ref.config import PhotonConfig  # noqa: E402


class PhotonInference:
    """
    Stateful inference engine for PHOTON-RAG.

    Manages per-session hierarchical state and drift metrics.
    """

    def __init__(self, model: PhotonModel, cfg: PhotonConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._sessions: dict[str, PhotonSessionState] = {}

    def get_session(
        self,
        session_id: str,
        repo_id: str,
        repo_commit: str,
    ) -> PhotonSessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = PhotonSessionState(
                session_id,
                repo_id,
                repo_commit,
            )
        return self._sessions[session_id]

    def hierarchical_prefill(
        self,
        input_ids: mx.array,
    ) -> tuple[mx.array, HierarchicalState]:
        """
        Run bottom-up encoding + top-down decoding, return (logits, state).

        The state captures encoder outputs at each hierarchy level
        for reuse in follow-up turns.
        """
        h = self.cfg.hierarchy
        L = h.levels
        model = self.model

        B, T = input_ids.shape

        # Embed + project
        tok = model.token_proj(model.token_embed(input_ids))

        # Bottom-up
        enc_outputs = [tok]
        x = tok
        for lv in range(L):
            x = model.chunkers[lv](x)
            from .blocks import causal_mask

            mask = causal_mask(x.shape[1])
            for block in model.encoders[lv]:
                x = block(x, model._rope_cos, model._rope_sin, mask)
            enc_outputs.append(x)

        # Top-down
        h_dec = enc_outputs[L]
        mask = causal_mask(h_dec.shape[1])
        for block in model.decoders[L - 1]:
            h_dec = block(h_dec, model._rope_cos, model._rope_sin, mask)

        for lv in reversed(range(1, L)):
            h_dec = model._decode_level(
                h_dec,
                enc_outputs[lv],
                model.converters[lv],
                model.decoders[lv - 1],
                h.chunk_sizes[lv],
            )

        h_dec = model._decode_level(
            h_dec,
            enc_outputs[0],
            model.converters[0],
            model.local_decoder,
            h.chunk_sizes[0],
        )

        logits = model.lm_head(model.output_norm(h_dec))

        state = HierarchicalState(
            level_states=[mx.array(e) for e in enc_outputs[1:]],
            token_proj=tok,
        )

        return logits, state

    def session_forward(
        self,
        input_ids: mx.array,
        session_id: str,
        repo_id: str,
        repo_commit: str,
    ) -> tuple[mx.array, DriftMetrics]:
        """
        Run a session-aware forward pass:
        1. Hierarchical prefill
        2. Update session state
        3. Compute drift metrics

        Returns (logits, drift_metrics).
        """
        session = self.get_session(session_id, repo_id, repo_commit)

        logits, h_state = self.hierarchical_prefill(input_ids)
        mx.eval(logits)

        drift = session.update(h_state, logits)

        return logits, drift

    def prune_evidence(
        self,
        chunk_texts: list[str],
        chunk_ids: list[str],
        session_id: str,
        max_chunks: int = 8,
    ) -> list[int]:
        """Return indices of the most relevant chunks based on PHOTON coarse state.

        For turn 1 (no session state), returns all indices (no pruning).
        For turn 2+, tokenizes each chunk, computes PHOTON encoding,
        and scores against the session's coarse state via cosine similarity.
        Returns top max_chunks indices sorted by relevance.
        """
        session = self._sessions.get(session_id)
        all_indices = list(range(len(chunk_texts)))

        # No session or no prior state → no pruning (turn 1)
        if (
            session is None
            or session.current_state is None
            or not session.current_state.level_states
        ):
            return all_indices

        # Already within budget → no pruning needed
        if len(chunk_texts) <= max_chunks:
            return all_indices

        # Get the coarse (top-level) state from the session
        coarse_state = session.current_state.level_states[-1]
        # Mean-pool to a single vector for comparison
        coarse_vec = mx.mean(
            coarse_state.astype(mx.float32),
            axis=tuple(range(coarse_state.ndim - 1)),
        )

        # Score each chunk by cosine similarity to the session coarse state
        scores: list[tuple[int, float]] = []
        for idx, text in enumerate(chunk_texts):
            if not text.strip():
                scores.append((idx, -1.0))
                continue

            # Tokenize chunk text using stub tokenizer byte encoding
            token_ids = [
                b % self.cfg.tokenizer.vocab_size for b in text.encode("utf-8")
            ]
            if not token_ids:
                scores.append((idx, -1.0))
                continue

            # Pad to chunk-aligned length
            from math import prod

            padding_multiple = prod(self.cfg.hierarchy.chunk_sizes)
            remainder = len(token_ids) % padding_multiple
            if remainder != 0:
                token_ids = token_ids + [0] * (padding_multiple - remainder)

            # Cap length to avoid OOM on large chunks
            max_len = 2048
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
                # Re-align after truncation
                remainder = len(token_ids) % padding_multiple
                if remainder != 0:
                    token_ids = token_ids[: len(token_ids) - remainder]

            if not token_ids:
                scores.append((idx, -1.0))
                continue

            input_ids = mx.array(token_ids, dtype=mx.int32).reshape(1, -1)
            _, h_state = self.hierarchical_prefill(input_ids)

            # Get the chunk's top-level representation
            chunk_top = h_state.level_states[-1]
            chunk_vec = mx.mean(
                chunk_top.astype(mx.float32),
                axis=tuple(range(chunk_top.ndim - 1)),
            )

            # Cosine similarity (higher = more relevant)
            dot = mx.sum(coarse_vec * chunk_vec)
            norm_a = mx.sqrt(mx.sum(coarse_vec * coarse_vec))
            norm_b = mx.sqrt(mx.sum(chunk_vec * chunk_vec))
            sim = dot / (norm_a * norm_b + 1e-8)
            mx.eval(sim)
            scores.append((idx, float(sim.item())))

        # Sort by similarity descending, take top max_chunks
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([s[0] for s in scores[:max_chunks]])
        return selected

    def get_drift_history(self, session_id: str) -> list[dict]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [d.as_dict() for d in session.drift_history]
