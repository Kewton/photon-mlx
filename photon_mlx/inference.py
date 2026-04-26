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
from typing import Any, NamedTuple

import mlx.core as mx

from .model import PhotonModel
from .session import (
    _UNSET,
    DriftMetrics,
    HierarchicalState,
    PhotonSessionState,
    WorkingMemoryConfig,
    _UnsetType,
    weighted_hierarchical_score,
)

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


class HierarchicalVecs(NamedTuple):
    """Issue #63: per-level chunk vectors produced by the shared tokenize +
    prefill helper.

    Fields are ordered to match ``drift_level_weights`` (token, mid, top)
    so the weighted combination in :meth:`PhotonInference._score_prune_candidates`
    can stack them directly along the last axis (DR1-003).

    All three fields have shape ``(N, D)`` — one mean-pooled vector per
    chunk at the corresponding hierarchy level.
    """

    token: mx.array
    mid: mx.array
    top: mx.array


def _masked_mean_along_time(
    tensor: mx.array,
    valid_steps: mx.array,
) -> mx.array:
    """Masked mean pool along the time axis of a (B, T, D) tensor.

    ``valid_steps`` is an ``(B,)`` ``int32`` array of per-row valid-step
    counts. Rows with zero valid steps are divided by ``1.0`` (fallback)
    to avoid division-by-zero; callers must either skip those rows or
    accept a zero vector. The helper is extracted from the shared
    tokenize+prefill pipeline so each granularity (token / mid / top)
    computes the mean the same way (Issue #63 / DR2-003).
    """
    T = tensor.shape[1]
    pos = mx.arange(T, dtype=mx.int32)[None, :]
    mask = (pos < valid_steps[:, None]).astype(mx.float32)
    mask_3d = mask[..., None]
    masked_sum = mx.sum(tensor * mask_3d, axis=1)
    valid_count = mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1.0)
    return masked_sum / valid_count


def _check_weight_initialization(model: PhotonModel, threshold: float) -> None:
    """Issue #140 / S7-001: emit a WARNING when ``model.token_embed.weight``
    looks like a fresh random init (σ above ``threshold``).

    Behaviour (design judgement #2 / DR1-004):
    * Silent skip when the model lacks ``token_embed`` (or ``weight``), when
      ``weight`` is not an ``mx.array`` (e.g. ``MagicMock`` test doubles), or
      when any attribute access raises — the start-up sanity check must NOT
      break ``__init__`` for non-standard models.
    * The WARNING records only ``σ`` and ``threshold`` (Issue #58 CB-002 /
      Issue #64 CB-003 — never log tensor values or sample elements).
    """
    try:
        embed_attr = getattr(model, "token_embed", None)
        if embed_attr is None:
            return
        weight = getattr(embed_attr, "weight", None)
        if not isinstance(weight, mx.array):
            return
        norm_std = float(mx.std(weight).item())
        if norm_std > threshold:
            _logger.warning(
                "PHOTON embedding has high variance (σ=%.4f, threshold=%.4f) — "
                "possibly random-init. Check model.checkpoint_path and load result "
                "(Issue #135 / S7-001).",
                norm_std,
                threshold,
            )
    except Exception as exc:
        _logger.debug("skip embedding init check (reason=%s)", type(exc).__name__)
        return


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
        *,
        drift_level_weights: tuple[float, ...] | list[float] | None = None,
        working_memory_cfg: WorkingMemoryConfig | None | _UnsetType = _UNSET,
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
        # Issue #63 / DR1-005: keyword-only to preserve the existing 3-arg
        # constructor contract. The weights are stored as a plain tuple so
        # :class:`PhotonSessionState` (and hierarchical scoring below) can
        # use them without coupling to :class:`SafeRecGenConfig`.
        if drift_level_weights is None:
            self._drift_level_weights: tuple[float, ...] = (0.2, 0.3, 0.5)
        else:
            self._drift_level_weights = tuple(float(w) for w in drift_level_weights)
        # Issue #64 / Codex CB-001: cross-turn working memory policy is
        # propagated to every PhotonSessionState created via get_session().
        # ``_UNSET`` (argument omitted) keeps the session default (working
        # memory enabled with defaults) for legacy callers. Explicit ``None``
        # is the fail-closed signal from ``_build_photon_deps()`` and MUST
        # produce a disabled session. A ``WorkingMemoryConfig`` instance is
        # used as-is.
        self._working_memory_cfg: WorkingMemoryConfig | None | _UnsetType = (
            working_memory_cfg
        )

        # Issue #140 / S7-001: start-up sanity check for the embedding norm.
        # Runs last so all other state is initialised even if the check raises
        # (it never should — silent skip — but keeping it last is defensive).
        _check_weight_initialization(model, cfg.model.embedding_random_init_threshold)

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
                drift_level_weights=self._drift_level_weights,
                working_memory_cfg=self._working_memory_cfg,
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
        *,
        question: str | None = None,
    ) -> tuple[mx.array, DriftMetrics]:
        """
        Run a session-aware forward pass:
        1. Hierarchical prefill
        2. Update session state (optionally recording ``question`` into
           :attr:`PhotonSessionState.turn_history` when Issue #64 working
           memory is enabled).
        3. Compute drift metrics

        Returns (logits, drift_metrics).
        """
        session = self.get_session(session_id, repo_id, repo_commit)

        logits, h_state = self.hierarchical_prefill(input_ids)
        mx.eval(logits)

        drift = session.update(h_state, logits, question_text=question)

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
            # Security logging (Issue #58 CB-002 + Issue #64 Codex CB-003):
            # log the closed-enum exception class name only. tokenizer.encode()
            # receives question text (Pass 1) or repo chunk text (Turn 2+); a
            # tokenizer that echoes the input in its exception message would
            # leak those fragments to log sinks if we surfaced ``exc`` or
            # ``str(exc)`` (design §7).
            _logger.warning(
                "tokenizer.encode failed; disabling pruning (fail-closed, "
                "Issue #58 CB-002, Codex CB-003, reason=%s)",
                type(exc).__name__,
            )
            raise _TokenizerEncodeFailure(type(exc).__name__) from exc
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

    def _tokenize_and_prefill_chunks(
        self,
        chunk_texts: list[str],
        micro_batch_size: int | None = None,
    ) -> tuple[list[int], HierarchicalVecs | None]:
        """Shared tokenize → pad → micro-batched prefill → masked-mean helper.

        Issue #63 / DR1-002 / DR2-002: the single source of truth for
        chunk vectorisation. It produces per-level masked-mean vectors
        (``token``, ``mid``, ``top``) so both top-only consumers and the
        hierarchical scoring path stay bit-for-bit compatible.

        Returns ``(valid_indices, HierarchicalVecs | None)``:
        * ``valid_indices`` — indices into ``chunk_texts`` of chunks that
          produced at least one token.
        * ``HierarchicalVecs`` — three ``(len(valid_indices), D)`` tensors,
          or ``None`` if no chunk was usable.

        Fails closed (DR4-002): if the prefill returns a state with
        ``len(level_states) < 1``, the whole call returns ``(indices, None)``
        so the caller reverts to the no-prune sentinel path. Missing
        ``token_proj`` is NOT considered partial — it only means the
        ``token`` vecs are degraded to a replica of ``top`` (level-1 builds).
        """
        # ① Tokenize all chunks once, collect valid_indices and step tables
        # for each of the three granularities (DR2-003).
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
        chunk_sizes = self.cfg.hierarchy.chunk_sizes
        # Granularity of each level after the bottom-up encoder:
        # * token: T
        # * mid:   T / chunk_sizes[0]
        # * top:   T / prod(chunk_sizes)
        valid_token_steps = [len(ids) for ids in all_token_ids]
        valid_mid_steps = [len(ids) // chunk_sizes[0] for ids in all_token_ids]
        valid_top_steps = [len(ids) // alignment for ids in all_token_ids]

        # ② Right-pad to the maximum within the micro-batch group.
        max_batch_len = max(len(ids) for ids in all_token_ids)
        rem = max_batch_len % alignment
        if rem != 0:
            max_batch_len += alignment - rem
        padded = [
            ids + [PAD_TOKEN_ID] * (max_batch_len - len(ids)) for ids in all_token_ids
        ]
        batch_input = mx.array(padded, dtype=mx.int32)

        # ②' Resolve micro-batch size (DR1-006).
        effective_micro = (
            micro_batch_size if micro_batch_size is not None else MICRO_BATCH_SIZE
        )

        valid_token_arr = mx.array(valid_token_steps, dtype=mx.int32)
        valid_mid_arr = mx.array(valid_mid_steps, dtype=mx.int32)
        valid_top_arr = mx.array(valid_top_steps, dtype=mx.int32)
        n_valid = batch_input.shape[0]
        token_pieces: list[mx.array] = []
        mid_pieces: list[mx.array] = []
        top_pieces: list[mx.array] = []

        for start in range(0, n_valid, effective_micro):
            end = min(start + effective_micro, n_valid)
            sub_input = batch_input[start:end]
            sub_top_steps = valid_top_arr[start:end]
            sub_mid_steps = valid_mid_arr[start:end]
            sub_token_steps = valid_token_arr[start:end]

            _, h_state = self.hierarchical_prefill(sub_input)

            # Fail-closed guard (DR4-002): if no level_states were produced,
            # we cannot reliably score any chunk. Returning ``None`` ensures
            # the caller drops to the no-prune sentinel.
            if not h_state.level_states:
                return valid_indices, None

            # ③ top (level_states[-1]) masked-mean
            chunk_tops = h_state.level_states[-1].astype(mx.float32)
            top_pieces.append(_masked_mean_along_time(chunk_tops, sub_top_steps))

            # ④ mid (level_states[0]) masked-mean; if only one level exists,
            # fall back to top so ``mid`` stays shape-compatible with ``top``.
            if len(h_state.level_states) >= 2:
                chunk_mids = h_state.level_states[0].astype(mx.float32)
                mid_pieces.append(_masked_mean_along_time(chunk_mids, sub_mid_steps))
            else:
                mid_pieces.append(_masked_mean_along_time(chunk_tops, sub_top_steps))

            # ⑤ token (token_proj) masked-mean; if token_proj is absent,
            # fall back to top so the three vecs can still be stacked.
            if h_state.token_proj is not None:
                chunk_tokens = h_state.token_proj.astype(mx.float32)
                token_pieces.append(
                    _masked_mean_along_time(chunk_tokens, sub_token_steps)
                )
            else:
                token_pieces.append(_masked_mean_along_time(chunk_tops, sub_top_steps))

        token_vecs = mx.concatenate(token_pieces, axis=0)
        mid_vecs = mx.concatenate(mid_pieces, axis=0)
        top_vecs = mx.concatenate(top_pieces, axis=0)
        return valid_indices, HierarchicalVecs(
            token=token_vecs, mid=mid_vecs, top=top_vecs
        )

    def _encode_chunks_to_vecs(
        self,
        chunk_texts: list[str],
        micro_batch_size: int | None = None,
    ) -> tuple[list[int], mx.array | None]:
        """Thin wrapper preserving pre-Issue-#63 behaviour: returns top-only
        chunk vectors. Delegates to :meth:`_tokenize_and_prefill_chunks` so
        both top-only and hierarchical scoring paths share a single
        tokenize/prefill pipeline (DR1-002, bit-for-bit compatible).
        """
        valid_indices, vecs = self._tokenize_and_prefill_chunks(
            chunk_texts, micro_batch_size=micro_batch_size
        )
        if vecs is None:
            return valid_indices, None
        return valid_indices, vecs.top

    def _encode_chunks_to_vecs_hierarchical(
        self,
        chunk_texts: list[str],
        micro_batch_size: int | None = None,
    ) -> tuple[list[int], HierarchicalVecs | None]:
        """Thin wrapper returning per-level chunk vectors (Issue #63)."""
        return self._tokenize_and_prefill_chunks(
            chunk_texts, micro_batch_size=micro_batch_size
        )

    def _build_query_hierarchical_vecs(
        self,
        state: HierarchicalState,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Build per-level (token, mid, top) mean-pooled query vectors.

        Issue #63: used by both ``_score_prune_candidates`` (session state)
        and ``_score_prune_candidates_from_question`` (one-off prefill).
        Missing levels fall back to the top-level vector so the stacked
        cosines always have three finite entries (design §5 decision #4).
        """
        top_state = state.level_states[-1].astype(mx.float32)
        q_top = mx.mean(top_state, axis=tuple(range(top_state.ndim - 1)))
        if len(state.level_states) >= 2:
            mid_state = state.level_states[0].astype(mx.float32)
            q_mid = mx.mean(mid_state, axis=tuple(range(mid_state.ndim - 1)))
        else:
            q_mid = q_top
        if state.token_proj is not None:
            tok_state = state.token_proj.astype(mx.float32)
            q_token = mx.mean(tok_state, axis=tuple(range(tok_state.ndim - 1)))
        else:
            q_token = q_top
        return q_token, q_mid, q_top

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

        Issue #63: raw_score is a weighted combination of per-level cosine
        similarities (token / mid / top) via
        :func:`weighted_hierarchical_score` — the weighting reuses
        ``self._drift_level_weights`` so drift and scoring stay consistent.

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

        # Issue #63 + #64: build all three query vectors (token/mid/top).
        # When working memory is active (get_session_coarse_state returns an
        # aggregate), override q_top with the cross-turn coarse aggregate so
        # scoring reflects the whole session; token/mid remain per-turn. When
        # working memory is disabled or the aggregate is unavailable, fall
        # back to the pure #63 hierarchical path (DR1-004).
        q_token, q_mid, q_top = self._build_query_hierarchical_vecs(
            session.current_state
        )
        coarse_vec = session.get_session_coarse_state()
        if coarse_vec is not None:
            q_top = coarse_vec

        # ①〜④ Chunk vectorisation (shared helper, hierarchical).
        valid_indices, vecs = self._encode_chunks_to_vecs_hierarchical(
            chunk_texts, micro_batch_size=micro_batch_size
        )
        if vecs is None:
            return scores

        # ⑤ Per-level cosine similarity, then weighted combination.
        sim_token = _batch_cosine_similarity(q_token, vecs.token)
        sim_mid = _batch_cosine_similarity(q_mid, vecs.mid)
        sim_top = _batch_cosine_similarity(q_top, vecs.top)
        sim_stack = mx.stack([sim_token, sim_mid, sim_top], axis=-1)
        sims = weighted_hierarchical_score(sim_stack, self._drift_level_weights)
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
        question-derived hierarchical vector.

        Builds a one-off 3-level coarse vector from ``question`` via
        :meth:`hierarchical_prefill` (no ``session_forward``, so
        ``self._sessions`` is never mutated — DR1-001) and combines per-level
        cosine similarities via :func:`weighted_hierarchical_score`.

        Raises :class:`_TokenizerEncodeFailure` if ``question`` or any chunk
        fails to tokenize (the caller fails closed by returning all indices).
        """
        self._validate_micro_batch_size(micro_batch_size)

        n = len(chunk_texts)
        scores: list[tuple[int, float]] = [(i, -1.0) for i in range(n)]
        if n == 0:
            return scores

        # ① Question → one-off 3-level coarse vectors (no session mutation).
        question_tokens = self._tokenize_chunk(question)
        if not question_tokens:
            return scores

        q_input = mx.array([question_tokens], dtype=mx.int32)
        _, q_state = self.hierarchical_prefill(q_input)
        # Issue #63: one-off 3-level query vectors for Pass 1 hierarchical
        # scoring. Turn 1 has no session history, so there is no coarse
        # aggregate to blend in (DR1-004).
        q_token, q_mid, q_top = self._build_query_hierarchical_vecs(q_state)

        # ②〜④ Chunk vectorisation via shared hierarchical helper.
        valid_indices, vecs = self._encode_chunks_to_vecs_hierarchical(
            chunk_texts, micro_batch_size=micro_batch_size
        )
        if vecs is None:
            return scores

        # ⑤ Per-level cosine similarity, then weighted combination.
        sim_token = _batch_cosine_similarity(q_token, vecs.token)
        sim_mid = _batch_cosine_similarity(q_mid, vecs.mid)
        sim_top = _batch_cosine_similarity(q_top, vecs.top)
        sim_stack = mx.stack([sim_token, sim_mid, sim_top], axis=-1)
        sims = weighted_hierarchical_score(sim_stack, self._drift_level_weights)
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

        # Fail closed on tokenizer errors (Issue #58 CB-002) and on
        # unexpected scoring-path failures (Issue #63 / CB-003): if any
        # step of the scoring pipeline raises (tokenizer, hierarchical
        # prefill, masked-mean, weighted scoring, MLX runtime / shape
        # assertion), we cannot rank chunks reliably → hand every chunk
        # back to the caller instead of propagating the exception.
        # The catch list is intentionally NOT bare ``Exception``: it
        # enumerates the exception classes that legitimate failure modes
        # raise (tokenizer failure, ValueError from shape / dtype /
        # validation, RuntimeError from MLX, AssertionError from contract
        # checks). Programming errors outside this set still surface.
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
        except (ValueError, RuntimeError, AssertionError) as exc:
            _logger.warning(
                "prune_evidence scoring failed (%s: %s); "
                "failing closed and returning all candidate indices "
                "(Issue #63 / CB-003)",
                type(exc).__name__,
                exc,
            )
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
