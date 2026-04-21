"""
PHOTON session inference pipeline.

Wraps the PhotonModel for multi-turn session usage:
- Hierarchical prefill from evidence pack
- Session state update with drift tracking
- Answer-time local refresh
- Grounded generation via conditioning
"""

from __future__ import annotations

import logging
import sys
from math import prod
from pathlib import Path
from typing import Any

import mlx.core as mx

from .model import PhotonModel
from .session import DriftMetrics, HierarchicalState, PhotonSessionState

sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_ref.config import PhotonConfig  # noqa: E402

_logger = logging.getLogger(__name__)


class _TokenizerEncodeFailure(RuntimeError):
    """Raised when ``self.tokenizer.encode`` fails inside prune scoring.

    Issue #58 CB-002 requires ``prune_evidence`` to fail closed (return every
    candidate index) when tokenisation is unreliable, so that the caller keeps
    all evidence instead of silently ranking on a partial input. This sentinel
    propagates from the scoring core back up to :meth:`prune_evidence`.
    """


# ────────────────────────────────────────────────────────────────────
# Module-level constants and helpers (Issue #61: batched prune_evidence)
# ────────────────────────────────────────────────────────────────────

PAD_TOKEN_ID: int = 0
"""Single source of truth for the pad token id used by chunk-aligned padding
and the right-padding that pads variable-length chunk batches up to the
maximum sequence length within the micro-batch.

Note: Issue #58 may eventually replace the byte-level stub tokenizer; the
``valid_top_steps`` book-keeping in :func:`PhotonInference._score_prune_candidates`
intentionally avoids depending on this token id semantically (validity is
derived from chunk-aligned token-id lengths, not from token id equality).
"""

MICRO_BATCH_SIZE: int = 64
"""Default micro-batch size for the batched evidence-pruning forward pass.

Used as the fallback when ``micro_batch_size=None`` is passed to
:meth:`PhotonInference.prune_evidence`. The current production setting is the
maximum number of chunks (``max_chunks*expansion=64``), so the default matches
production and effectively no sub-batching occurs. The argument is kept as a
test-injection seam (``micro_batch_size=2``-style tests) and as a future
escape valve for OOM.
"""


def _batch_cosine_similarity(
    query: mx.array,
    keys: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """Compute cosine similarity between a single query and many key vectors.

    Args:
        query: shape ``(D,)`` — the reference vector.
        keys:  shape ``(N, D)`` — N candidate vectors.
        eps:   small constant added to the denominator to avoid division by
               zero / NaN when one of the vectors has zero norm.

    Returns:
        shape ``(N,)`` — cosine similarity scores in ``[-1, 1]`` (modulo eps).

    The implementation matches the per-chunk formula used by the original
    sequential ``prune_evidence`` (``dot / (norm_a * norm_b + eps)``) so that
    raw scores remain bit-comparable up to MLX accumulation order.
    """
    query_norm = mx.sqrt(mx.sum(query * query))
    key_norms = mx.sqrt(mx.sum(keys * keys, axis=1))
    dots = mx.sum(keys * query[None, :], axis=1)
    return dots / (query_norm * key_norms + eps)


class PhotonInference:
    """
    Stateful inference engine for PHOTON-RAG.

    Manages per-session hierarchical state and drift metrics.
    """

    def __init__(
        self,
        model: PhotonModel,
        cfg: PhotonConfig,
        tokenizer: Any,
    ) -> None:
        self.model = model
        self.cfg = cfg
        # Shared tokenizer instance used by both the pipeline (question+evidence
        # prefill) and ``prune_evidence`` (chunk scoring).  Required so both
        # paths live in the same semantic space (Issue #58).
        self.tokenizer = tokenizer
        self._sessions: dict[str, PhotonSessionState] = {}
        # Pre-compute the chunk-aligned padding multiple once per inference
        # engine. cfg.hierarchy.chunk_sizes is treated as immutable across the
        # instance lifetime (DR3-001 / Risk R7).
        self._chunk_alignment: int = prod(cfg.hierarchy.chunk_sizes)

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

        Uses ``PhotonModel._encode_bottom_up`` / ``_decode_from_enc_outputs``
        shared helpers (DR1-003) to avoid duplicating the top-down stack.
        """
        model = self.model

        # Bottom-up (shared helper — chunk-aligned input required)
        enc_outputs = model._encode_bottom_up(input_ids)

        # Top-down (shared helper; prefill path, cache disabled)
        logits, _ = model._decode_from_enc_outputs(enc_outputs, top_kv_cache=None)

        state = HierarchicalState(
            level_states=[mx.array(e) for e in enc_outputs[1:]],
            token_proj=enc_outputs[0],
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

    # ────────────────────────────────────────────────────────────
    # Evidence pruning (Issue #37 + Issue #61 batched)
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_micro_batch_size(micro_batch_size: int | None) -> None:
        """Validation contract shared by ``prune_evidence`` and the scoring
        core (CB-001). ``bool`` is rejected before ``int`` because it is a
        Python ``int`` subclass.
        """
        if micro_batch_size is None:
            return
        if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int):
            raise ValueError(
                "micro_batch_size must be a positive int or None, "
                f"got type {type(micro_batch_size).__name__}"
            )
        if micro_batch_size < 1:
            raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")

    def _tokenize_chunk(self, text: str) -> list[int]:
        """Tokenize ``text`` into chunk-aligned token ids using the real tokenizer.

        Steps:
        1. Encode ``text`` via ``self.tokenizer.encode`` so the chunk and
           question paths share a single semantic space (Issue #58).
        2. Right-pad to the next multiple of ``self._chunk_alignment`` using
           :data:`PAD_TOKEN_ID`.
        3. Cap length at ``cfg.model.max_position_embeddings`` and re-align
           after truncation if necessary.

        Returns an empty list if ``text`` is empty or yields no tokens.
        Raises :class:`_TokenizerEncodeFailure` (Issue #58 CB-002) if the real
        tokenizer raises — this propagates to :meth:`prune_evidence`, which
        fails closed by returning every candidate index instead of silently
        ranking on a partial input.
        """
        if not text:
            return []
        try:
            token_ids = list(self.tokenizer.encode(text))
        except Exception as exc:
            # Security logging (CB-002 codex-fix): log the closed-enum
            # exception class name only; the raw exception body may carry
            # prompt fragments / tokenizer internals and must never appear
            # in warning logs.
            _logger.warning(
                "tokenizer.encode failed; disabling pruning (fail-closed, "
                "Issue #58 CB-002, reason=%s)",
                type(exc).__name__,
            )
            raise _TokenizerEncodeFailure(str(exc)) from exc
        if not token_ids:
            return []

        alignment = self._chunk_alignment
        remainder = len(token_ids) % alignment
        if remainder != 0:
            token_ids = token_ids + [PAD_TOKEN_ID] * (alignment - remainder)

        max_len = self.cfg.model.max_position_embeddings
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            remainder = len(token_ids) % alignment
            if remainder != 0:
                token_ids = token_ids[: len(token_ids) - remainder]

        return token_ids

    def _encode_chunks_to_vecs(
        self,
        chunk_texts: list[str],
        micro_batch_size: int | None = None,
    ) -> tuple[list[int], mx.array | None]:
        """Tokenise → pad → hierarchical_prefill (micro-batched) → masked-mean.

        Returns ``(valid_indices, chunk_vecs)`` where ``chunk_vecs`` is either
        ``None`` (no chunk produced any tokens) or an ``mx.array`` of shape
        ``(len(valid_indices), D)``.  Steps ①〜④ of the design-policy data
        flow are encapsulated here so Pass 1 scoring and Turn 2+ scoring share
        identical chunk-vectorisation logic (DR1-001).

        The helper is pure with respect to ``self._sessions`` — it reads the
        tokenizer and model only. ``_TokenizerEncodeFailure`` raised by
        :meth:`_tokenize_chunk` propagates out so the caller can fail-closed.
        """
        # ① Tokenize all chunks once, collect valid_indices and valid_top_steps.
        valid_indices: list[int] = []
        all_token_ids: list[list[int]] = []
        for idx, text in enumerate(chunk_texts):
            if not text or not text.strip():
                continue
            token_ids = self._tokenize_chunk(text)
            if not token_ids:
                continue
            valid_indices.append(idx)
            all_token_ids.append(token_ids)

        if not valid_indices:
            return valid_indices, None

        alignment = self._chunk_alignment
        valid_top_steps = [len(ids) // alignment for ids in all_token_ids]

        # ② Right-pad to the maximum within the micro-batch group. We pad the
        # whole valid set up to the global max once and slice per micro-batch
        # below. The padding length is rounded up to ``alignment`` so the
        # hierarchy reshape never sees a partial chunk.
        max_batch_len = max(len(ids) for ids in all_token_ids)
        rem = max_batch_len % alignment
        if rem != 0:
            max_batch_len += alignment - rem
        padded = [
            ids + [PAD_TOKEN_ID] * (max_batch_len - len(ids)) for ids in all_token_ids
        ]
        batch_input = mx.array(padded, dtype=mx.int32)

        # ②' Resolve micro-batch size (DR1-006 fallback responsibility).
        effective_micro = (
            micro_batch_size if micro_batch_size is not None else MICRO_BATCH_SIZE
        )

        # ③+④ Forward each micro-batch through hierarchical_prefill, then
        # apply masked-mean (path B per pre-check decision) to obtain one
        # vector per chunk. Concatenate on the GPU; never tolist() between
        # micro-batches.
        valid_top_arr = mx.array(valid_top_steps, dtype=mx.int32)
        n_valid = batch_input.shape[0]
        chunk_vec_pieces: list[mx.array] = []
        for start in range(0, n_valid, effective_micro):
            end = min(start + effective_micro, n_valid)
            sub_input = batch_input[start:end]
            sub_steps = valid_top_arr[start:end]

            _, h_state = self.hierarchical_prefill(sub_input)
            chunk_tops = h_state.level_states[-1].astype(mx.float32)
            # chunk_tops shape: (sub_B, T_top, D)
            T_top = chunk_tops.shape[1]
            pos = mx.arange(T_top, dtype=mx.int32)[None, :]
            mask = (pos < sub_steps[:, None]).astype(mx.float32)
            mask_3d = mask[..., None]
            masked_sum = mx.sum(chunk_tops * mask_3d, axis=1)
            valid_count = mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1.0)
            chunk_vec_pieces.append(masked_sum / valid_count)

        chunk_vecs = mx.concatenate(chunk_vec_pieces, axis=0)  # (N_valid, D)
        return valid_indices, chunk_vecs

    def _score_prune_candidates(
        self,
        chunk_texts: list[str],
        session_id: str,
        micro_batch_size: int | None = None,
    ) -> list[tuple[int, float]]:
        """Scoring core (steps ①〜⑤ of the design-policy data flow).

        Returns a list of ``(index, raw_score)`` tuples, one per input chunk
        (length == ``len(chunk_texts)``). For chunks that should not be
        scored (empty text, no valid tokens after alignment, or no session
        state available), the raw score is ``-1.0``.

        ``chunk_ids`` is intentionally NOT a parameter (DR3-002): chunk_ids
        are not used by scoring; they are a presentation/selection concern
        handled by the caller.
        """
        # CB-001: enforce the same contract as the public wrapper so direct
        # callers (tests and helpers) get an identical ValueError on bool / 0 /
        # negative / non-int.
        self._validate_micro_batch_size(micro_batch_size)

        n = len(chunk_texts)
        scores: list[tuple[int, float]] = [(i, -1.0) for i in range(n)]
        if n == 0:
            return scores

        session = self._sessions.get(session_id)
        if (
            session is None
            or session.current_state is None
            or not session.current_state.level_states
        ):
            # No session state → caller falls back to the trivial path. We
            # still return -1.0 placeholders so that the caller never has to
            # special-case None.
            return scores

        # Coarse session vector (mean-pool along all leading dims).
        coarse_state = session.current_state.level_states[-1].astype(mx.float32)
        coarse_vec = mx.mean(
            coarse_state,
            axis=tuple(range(coarse_state.ndim - 1)),
        )

        # ①〜④ Chunk vectorisation (shared with Pass 1 scoring via helper).
        valid_indices, chunk_vecs = self._encode_chunks_to_vecs(
            chunk_texts, micro_batch_size=micro_batch_size
        )
        if chunk_vecs is None:
            return scores

        # ⑤ Cosine similarity, single GPU→CPU sync.
        sims = _batch_cosine_similarity(coarse_vec, chunk_vecs)
        mx.eval(sims)
        sims_list = sims.tolist()

        for k, idx in enumerate(valid_indices):
            scores[idx] = (idx, float(sims_list[k]))

        return scores

    def _score_prune_candidates_from_question(
        self,
        chunk_texts: list[str],
        question: str,
        micro_batch_size: int | None = None,
    ) -> list[tuple[int, float]]:
        """Pass 1 scoring (Turn 1): score chunks against a transient
        question-derived coarse vector.

        Builds a one-off coarse_vec from ``question`` via
        :meth:`hierarchical_prefill` (no ``session_forward``, so
        ``self._sessions`` is never mutated — DR1-001) and cosine-scores each
        chunk against it using the shared :meth:`_encode_chunks_to_vecs`
        helper.

        Raises :class:`_TokenizerEncodeFailure` if ``question`` or any chunk
        fails to tokenize (the caller fails closed by returning all indices).
        """
        self._validate_micro_batch_size(micro_batch_size)

        n = len(chunk_texts)
        scores: list[tuple[int, float]] = [(i, -1.0) for i in range(n)]
        if n == 0:
            return scores

        # ① Question → one-off coarse vector (no session mutation).
        question_tokens = self._tokenize_chunk(question)
        if not question_tokens:
            return scores

        q_input = mx.array([question_tokens], dtype=mx.int32)
        _, q_state = self.hierarchical_prefill(q_input)
        q_top = q_state.level_states[-1].astype(mx.float32)
        question_coarse_vec = mx.mean(q_top, axis=tuple(range(q_top.ndim - 1)))

        # ②〜④ Chunk vectorisation via shared helper.
        valid_indices, chunk_vecs = self._encode_chunks_to_vecs(
            chunk_texts, micro_batch_size=micro_batch_size
        )
        if chunk_vecs is None:
            return scores

        # ⑤ Cosine similarity, single GPU→CPU sync.
        sims = _batch_cosine_similarity(question_coarse_vec, chunk_vecs)
        mx.eval(sims)
        sims_list = sims.tolist()

        for k, idx in enumerate(valid_indices):
            scores[idx] = (idx, float(sims_list[k]))

        return scores

    def prune_evidence(
        self,
        chunk_texts: list[str],
        chunk_ids: list[str],
        session_id: str,
        max_chunks: int = 8,
        micro_batch_size: int | None = None,
        *,
        question: str | None = None,
    ) -> list[int]:
        """Return indices of the most relevant chunks based on PHOTON coarse state.

        Dispatcher (Issue #56, DR1-004):
        - Turn 2+ (session state exists): score against the session coarse_vec
          via :meth:`_score_prune_candidates` (pre-existing behaviour).
        - Turn 1 + ``question`` provided: score against a transient
          question-derived coarse_vec via
          :meth:`_score_prune_candidates_from_question` (Pass 1, session is
          NOT mutated — DR1-001).
        - Turn 1 + ``question is None`` (or blank): return all indices
          (pre-Issue-#56 behaviour, DR1-006 backward compatibility).

        Args:
            chunk_texts: candidate chunk content (one string per candidate).
            chunk_ids:   candidate chunk identifiers (kept for API stability;
                         not used for scoring — see DR3-002).
            session_id:  session key used to look up the coarse state.
            max_chunks:  number of indices to return (top-K).
            micro_batch_size: optional override for the GPU forward batch
                              size. ``None`` (default) uses
                              :data:`MICRO_BATCH_SIZE`. Validation rules
                              (DR1-007): must be ``int >= 1``; ``bool``
                              (``True``/``False``) is rejected because it is
                              technically a Python ``int`` subclass.
            question:    keyword-only. When provided on Turn 1 (no session
                         state), enables Pass 1 scoring against a transient
                         question-derived coarse vector (Issue #56). Default
                         ``None`` preserves pre-Issue-#56 behaviour.

        Returns:
            list of indices into ``chunk_texts`` in ascending order.
        """
        # Validate micro_batch_size (DR1-007 + CB-001). The same contract is
        # enforced inside _score_prune_candidates so direct callers see the
        # identical ValueError surface.
        self._validate_micro_batch_size(micro_batch_size)

        all_indices = list(range(len(chunk_texts)))

        # Structural early returns (§4.4 / DR1-008).
        if len(chunk_texts) == 0:
            return []
        if len(chunk_texts) <= max_chunks:
            return all_indices

        session = self._sessions.get(session_id)
        has_state = (
            session is not None
            and session.current_state is not None
            and bool(session.current_state.level_states)
        )

        # Fail closed on tokenizer errors (Issue #58 CB-002): if encode raises
        # we cannot rank chunks reliably, so hand every chunk back to the
        # caller instead of returning an arbitrary prefix.
        try:
            if has_state:
                raw_scores = self._score_prune_candidates(
                    chunk_texts,
                    session_id,
                    micro_batch_size=micro_batch_size,
                )
            elif question is not None and question.strip():
                raw_scores = self._score_prune_candidates_from_question(
                    chunk_texts,
                    question,
                    micro_batch_size=micro_batch_size,
                )
            else:
                # Turn 1 without a question → preserve pre-Issue-#56 behaviour.
                return all_indices
        except _TokenizerEncodeFailure:
            return all_indices

        # Selection: sort by score desc, take top max_chunks, return indices
        # in ascending order.
        ranked = sorted(raw_scores, key=lambda x: x[1], reverse=True)
        selected = sorted(idx for idx, _ in ranked[:max_chunks])
        return selected

    # ────────────────────────────────────────────────────────────
    # PHOTON single-path generation (Issue #62 Phase 1)
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_max_new_tokens(max_new_tokens: int) -> None:
        """Validation contract for ``generate_answer.max_new_tokens``.

        Mirrors ``_validate_micro_batch_size``: reject ``bool`` before ``int``
        (Python ``bool`` is an ``int`` subclass) and reject non-positive
        values. Stage 4 DR4-003 / DR-62-005 fail-fast guard.
        """
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise ValueError(
                "max_new_tokens must be a positive int, "
                f"got type {type(max_new_tokens).__name__}"
            )
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    def generate_answer(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int,
    ) -> str:
        """Generate an answer string using the PHOTON model (Issue #62 Phase 1).

        Contract (design §8.1, DR-62-002..005):

        - Tokenizes ``prompt_text`` via ``self.tokenizer`` (shared with
          ``prune_evidence`` / ``session_forward``).
        - Calls ``self.model.generate(input_ids, max_new_tokens=...)``.
        - Decodes the newly generated tokens back to a string via
          ``self.tokenizer.decode`` (prompt tokens stripped).

        Fail-fast behaviour (API is fail-fast, pipeline is fail-closed):

        - ``_TokenizerEncodeFailure`` / ``RuntimeError`` raised by the real
          tokenizer propagate to the caller unchanged.
        - ``ValueError`` from :meth:`PhotonModel.generate`'s DR4-002 length
          guard propagates unchanged.
        - ``max_new_tokens`` must be a positive ``int`` (``bool`` rejected).

        Session handling (Phase 1): stateless. Each call re-prefills the
        prompt from scratch; no KV cache is reused from ``session_forward``.
        Phase 2 may add session parameters additively (YAGNI per DR1-002).
        """
        self._validate_max_new_tokens(max_new_tokens)

        # Encode — propagate failures as _TokenizerEncodeFailure so the
        # pipeline layer can fall back to Qwen with a stable fallback_reason
        # (§7.2 closed enum).
        try:
            token_ids = list(self.tokenizer.encode(prompt_text))
        except Exception as exc:
            raise _TokenizerEncodeFailure(str(exc)) from exc

        if not token_ids:
            # An empty prompt cannot be decoded meaningfully.  Surface this
            # as a ValueError so the pipeline records it under the closed
            # enum ``ValueError`` (§7.2) and falls back to Qwen.
            raise ValueError("prompt_text produced zero tokens after encoding")

        # Preflight context-window guard (CB-001 codex-fix): reject prompts
        # that cannot fit with ``max_new_tokens`` *before* allocating the
        # ``mx.array`` input_ids buffer. ``PhotonModel.generate`` already
        # has the same guard at ``model.py`` (DR4-002) but allocating a
        # large array only to raise afterwards is wasteful and widens a
        # DoS window when oversize evidence packs hit a small PHOTON
        # window. Belt-and-suspenders: the deeper guard remains in place.
        max_pos = self.cfg.model.max_position_embeddings
        if len(token_ids) + max_new_tokens > max_pos:
            raise ValueError(
                f"prompt_len={len(token_ids)} + max_new_tokens={max_new_tokens} "
                f"exceeds max_position_embeddings={max_pos}"
            )

        input_ids = mx.array([token_ids], dtype=mx.int32)
        prompt_len = input_ids.shape[1]

        # Call the real PhotonModel.generate — its DR4-002 length guard and
        # MLX-level errors (RuntimeError, etc.) propagate unchanged.
        generated, _step_logits = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )
        mx.eval(generated)

        # Strip the prompt prefix; decode only the newly-generated tokens.
        new_token_ids = generated[0, prompt_len:].tolist()
        return self.tokenizer.decode(new_token_ids)

    def get_drift_history(self, session_id: str) -> list[dict]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [d.as_dict() for d in session.drift_history]
