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
        self, session_id: str, repo_id: str, repo_commit: str,
    ) -> PhotonSessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = PhotonSessionState(
                session_id, repo_id, repo_commit,
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
                h_dec, enc_outputs[lv],
                model.converters[lv], model.decoders[lv - 1],
                h.chunk_sizes[lv],
            )

        h_dec = model._decode_level(
            h_dec, enc_outputs[0],
            model.converters[0], model.local_decoder,
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

    def get_drift_history(self, session_id: str) -> list[dict]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [d.as_dict() for d in session.drift_history]
