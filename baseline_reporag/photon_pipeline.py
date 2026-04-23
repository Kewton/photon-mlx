"""PHOTON-RAG pipeline integration (Issue #3).

Provides:
- build_pipeline(cfg) — factory that routes to RepoRAGPipeline or PhotonRAGPipeline
- PhotonRAGPipeline — PHOTON-enhanced RAG with drift tracking and fallback
- tokenize_evidence_pack() — encode evidence text for PHOTON prefill
- compute_confidence() — extract confidence from PHOTON logits
"""

from __future__ import annotations

import logging
import uuid
from math import prod
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from .citation import resolve_citations
from .config import Config
from .generation.evidence_pack import build_evidence_pack
from .generation.prompt import (
    _EVIDENCE_HEADER,
    build_messages,
    flatten_messages_for_plain_lm,
)
from .memory.session import SessionState
from .pipeline import QueryResult, RepoRAGPipeline, apply_citation_postprocess
from .profiler import TurnProfiler
from .retrieval.graph_expansion import expand_with_graph
from .retrieval.hybrid import apply_file_type_boost, hybrid_search
from .retrieval.query_expansion import expand_query

if TYPE_CHECKING:
    # Issue #103 / DR2-008: ``TurnState`` is a PHOTON type; importing it at
    # runtime would force MLX/PHOTON load on baseline-only paths. The file
    # uses ``from __future__ import annotations`` so the cache type
    # ``dict[str, TurnState]`` resolves lazily.
    from photon_mlx.session import TurnState

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Two-pass search configuration (Issue #56)
# ---------------------------------------------------------------------------


def _resolve_two_pass_search_cfg(
    retrieval_cfg: Any,
    fused_top_k: int,
    evidence_max_chunks: int,
) -> tuple[bool, int, int]:
    """Resolve and validate ``retrieval.two_pass_search`` settings.

    Returns ``(enabled, pass1_top_k, pass2_top_k)``. ``enabled`` defaults to
    ``False`` when the section is missing so existing configs continue to work.

    Validation rules (design §4.5 / DR1-008):
    - ``pass1_top_k >= pass2_top_k >= 1`` — violation raises ``ValueError``
    - ``pass1_top_k < fused_top_k`` — warn and clamp up to ``fused_top_k``
      (avoids silently dropping candidates supplied by retrieval)

    Validation is performed even when ``enabled=False`` so mis-configurations
    surface early (Stage 3 S3-002).
    """
    section = (
        retrieval_cfg.get("two_pass_search", {}) if retrieval_cfg is not None else {}
    )
    if section is None:
        section = {}
    # Support both ``Config`` wrappers and plain dicts.
    getter = section.get

    enabled_raw = getter("enabled", False)
    enabled = bool(enabled_raw)
    pass1_top_k = getter("pass1_top_k", fused_top_k)
    pass2_top_k = getter("pass2_top_k", evidence_max_chunks)

    if not isinstance(pass1_top_k, int) or isinstance(pass1_top_k, bool):
        raise ValueError(
            "retrieval.two_pass_search.pass1_top_k must be an int, "
            f"got {type(pass1_top_k).__name__}"
        )
    if not isinstance(pass2_top_k, int) or isinstance(pass2_top_k, bool):
        raise ValueError(
            "retrieval.two_pass_search.pass2_top_k must be an int, "
            f"got {type(pass2_top_k).__name__}"
        )
    if pass2_top_k < 1:
        raise ValueError(
            f"retrieval.two_pass_search.pass2_top_k must be >= 1, got {pass2_top_k}"
        )
    if pass1_top_k < pass2_top_k:
        raise ValueError(
            "retrieval.two_pass_search.pass1_top_k must be >= pass2_top_k, "
            f"got pass1_top_k={pass1_top_k}, pass2_top_k={pass2_top_k}"
        )
    if pass1_top_k < fused_top_k:
        _logger.warning(
            "retrieval.two_pass_search.pass1_top_k (%d) < retrieval.fused_top_k "
            "(%d); clamping pass1_top_k up to fused_top_k to preserve retrieval "
            "candidates.",
            pass1_top_k,
            fused_top_k,
        )
        pass1_top_k = fused_top_k
    return enabled, pass1_top_k, pass2_top_k


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def tokenize_evidence_pack(
    text: str,
    tokenizer: Any,
    cfg: Any,
    max_tokens: int | None = None,
) -> mx.array:
    """Tokenize evidence text with chunk-aligned padding.

    Args:
        text: raw evidence text.
        tokenizer: tokenizer with ``encode()`` and ``pad_token_id``.
        cfg: a :class:`torch_ref.config.PhotonConfig` instance.  The baseline
            ``Config`` (from ``configs/baseline.yaml``) does **not** define
            ``model.max_position_embeddings`` and must not be passed here.
        max_tokens: hard cap on token count.  When ``None`` (default), the
            cap is taken from ``cfg.model.max_position_embeddings``.  Must be
            positive; a :class:`ValueError` is raised otherwise (DR1-001).

    Returns:
        mx.array of token ids, length is a multiple of prod(chunk_sizes).
    """
    if max_tokens is None:
        max_tokens = cfg.model.max_position_embeddings

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    ids = tokenizer.encode(text)
    if not ids:
        return mx.array([], dtype=mx.int32)

    if len(ids) > max_tokens:
        ids = ids[:max_tokens]

    padding_multiple = prod(cfg.hierarchy.chunk_sizes)
    remainder = len(ids) % padding_multiple
    if remainder != 0:
        pad_count = padding_multiple - remainder
        ids = ids + [tokenizer.pad_token_id] * pad_count

    return mx.array(ids, dtype=mx.int32)


def compute_confidence(logits: mx.array) -> float:
    """Compute mean max-softmax confidence from logits.

    Args:
        logits: (B, seq_len, vocab_size) tensor.

    Returns:
        float in [0, 1].
    """
    probs = mx.softmax(logits, axis=-1)
    max_probs = mx.max(probs, axis=-1)
    return float(mx.mean(max_probs).item())


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _build_baseline_deps(cfg: Config) -> dict[str, Any]:
    """Construct real baseline pipeline dependencies from config.

    The canonical implementation lives in
    :func:`baseline_reporag.pipeline_factory._build_baseline_deps_no_mlx`
    so the factory module can stay MLX-free at import time. This wrapper
    is preserved as a module attribute so existing tests that patch
    ``baseline_reporag.photon_pipeline._build_baseline_deps`` keep working
    (Issue #62 Phase 1 refactor R-1: single source of truth, no
    lockstep-drift risk).
    """
    from .pipeline_factory import _build_baseline_deps_no_mlx

    return _build_baseline_deps_no_mlx(cfg)


def _resolve_working_memory_cfg(raw: Any) -> Any:
    """Normalise ``session_memory.working_memory`` into a ``WorkingMemoryConfig``.

    Accepts ``None`` (feature disabled), a dict (YAML form), or an already
    constructed :class:`photon_mlx.session.WorkingMemoryConfig`. Anything
    else triggers a warning (type name only; raw values are never surfaced,
    design §7) and fails closed to ``None`` so the query path continues.

    Returns either a ``WorkingMemoryConfig`` instance or ``None``.
    """
    from photon_mlx.session import WorkingMemoryConfig

    if raw is None:
        return None
    if isinstance(raw, WorkingMemoryConfig):
        return raw
    # Support the baseline Config wrapper (has .to_dict()) and plain dicts.
    raw_dict: dict[str, Any]
    if isinstance(raw, Config):
        raw_dict = raw.to_dict()
    elif isinstance(raw, dict):
        raw_dict = dict(raw)
    else:
        _logger.warning(
            "session_memory.working_memory has unsupported type %s; "
            "disabling working memory for this session",
            type(raw).__name__,
        )
        return None
    try:
        return WorkingMemoryConfig(**raw_dict)
    except (TypeError, ValueError) as exc:
        # Intentionally omit the raw dict (may contain attacker-controlled
        # values from YAML). Only the exception class name is logged.
        _logger.warning(
            "WorkingMemoryConfig rejected session_memory.working_memory "
            "(%s); disabling working memory for this session",
            type(exc).__name__,
        )
        return None


def _extract_working_memory_cfg(cfg: Config) -> Any:
    """Pull ``session_memory.working_memory`` out of a baseline ``Config``.

    Uses ``getattr`` / ``get`` so missing sections surface as ``None``
    rather than raising (design §3-3 fail-closed rules).
    """
    session_memory = getattr(cfg, "session_memory", None)
    if session_memory is None:
        return None
    raw = None
    if hasattr(session_memory, "get"):
        raw = session_memory.get("working_memory", None)
    else:
        raw = getattr(session_memory, "working_memory", None)
    return _resolve_working_memory_cfg(raw)


def _build_photon_deps(cfg: Config) -> dict[str, Any]:
    """Construct PHOTON-specific dependencies from config."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from torch_ref.config import PhotonConfig

    from photon_mlx.inference import PhotonInference
    from photon_mlx.model import PhotonModel
    from photon_mlx.safe_recgen import SafeRecGenConfig, SafeRecGenController

    from torch_ref.config import (
        HierarchyConfig,
        ModelConfig,
        TokenizerConfig,
    )

    # Issue #55: wire long-context RoPE fields from baseline cfg so
    # `photon_long_context.yaml` reaches PhotonModel unchanged.  When the
    # baseline cfg lacks these keys (e.g. legacy 2048 profiles), we fall
    # back to ModelConfig defaults via ``rope_scaling_from``.
    scaling, factor = ModelConfig.rope_scaling_from(cfg.model)
    model_cfg = ModelConfig(
        architecture=cfg.model.get("architecture", "photon_decoder"),
        base_embed_dim=cfg.model.base_embed_dim,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        num_attention_heads=cfg.model.get("num_heads", 4),
        num_key_value_heads=cfg.model.get("num_heads", 4),
        head_dim=getattr(cfg.model, "head_dim", 64),
        max_position_embeddings=getattr(cfg.model, "max_position_embeddings", 2048),
        rope_theta=getattr(cfg.model, "rope_theta", 1_000_000.0),
        rope_scaling=scaling,
        rope_scale_factor=factor,
    )
    hierarchy_cfg = HierarchyConfig(
        levels=cfg.hierarchy.levels,
        chunk_sizes=cfg.hierarchy.chunk_sizes,
        encoder_layers_per_level=cfg.hierarchy.encoder_layers_per_level,
        decoder_layers_per_level=cfg.hierarchy.decoder_layers_per_level,
    )
    tok_cfg = TokenizerConfig(
        vocab_size=cfg.model.get("vocab_size", 1000),
    )
    photon_cfg = PhotonConfig(
        model=model_cfg,
        hierarchy=hierarchy_cfg,
        tokenizer=tok_cfg,
    )

    # Build the tokenizer before PhotonInference so both paths (question+evidence
    # prefill in PhotonRAGPipeline and chunk scoring in prune_evidence) share
    # the same stub instance (Issue #58).
    tokenizer = _get_stub_tokenizer(photon_cfg.tokenizer.vocab_size)
    model = PhotonModel(photon_cfg)
    # Issue #64 / Codex CB-001: extract working memory policy once, pass it
    # into PhotonInference alongside the Issue #63 drift_level_weights below.
    working_memory_cfg = _extract_working_memory_cfg(cfg)

    safe_recgen_enabled = getattr(cfg.get("inference"), "safe_recgen_enabled", True)
    if safe_recgen_enabled:
        sr_cfg_data = cfg.get("safe_recgen")
        if sr_cfg_data is not None:
            triggers = sr_cfg_data.get("triggers")
            thresholds = sr_cfg_data.get("thresholds")
            # Issue #63 / DR1-010: alias resolution happens here, not inside
            # SafeRecGenConfig. The legacy YAML key
            # ``thresholds.latent_cosine_drift`` maps onto the new
            # ``latent_cosine_drift_top_threshold``; when both are present,
            # the new explicit ``latent_cosine_drift_top`` wins.
            legacy_top_threshold = (
                getattr(thresholds, "latent_cosine_drift", 0.18) if thresholds else 0.18
            )
            top_threshold = (
                getattr(thresholds, "latent_cosine_drift_top", legacy_top_threshold)
                if thresholds
                else legacy_top_threshold
            )
            # DR2-005: fall back to defaults for missing new keys.
            mid_threshold = (
                getattr(thresholds, "latent_cosine_drift_mid", 0.40)
                if thresholds
                else 0.40
            )
            token_threshold = (
                getattr(thresholds, "latent_cosine_drift_token", 0.30)
                if thresholds
                else 0.30
            )
            drift_level_weights = sr_cfg_data.get("drift_level_weights")
            if drift_level_weights is None:
                drift_level_weights = (0.2, 0.3, 0.5)
            sr_config = SafeRecGenConfig(
                enabled=True,
                trigger_exact_quote=getattr(triggers, "exact_quote", True)
                if triggers
                else True,
                trigger_diff_or_patch=getattr(triggers, "diff_or_patch", True)
                if triggers
                else True,
                trigger_high_risk_query=getattr(triggers, "high_risk_query", True)
                if triggers
                else True,
                trigger_topic_shift=getattr(triggers, "topic_shift", True)
                if triggers
                else True,
                trigger_latent_drift=getattr(triggers, "latent_drift", True)
                if triggers
                else True,
                trigger_low_confidence=getattr(triggers, "low_confidence", True)
                if triggers
                else True,
                # Legacy top-only threshold (kept in sync with the new field
                # for backward-compat log/schema consumers).
                latent_cosine_drift_threshold=top_threshold,
                topic_shift_score_threshold=getattr(
                    thresholds, "topic_shift_score", 0.65
                )
                if thresholds
                else 0.65,
                confidence_floor=getattr(thresholds, "confidence_floor", 0.40)
                if thresholds
                else 0.40,
                logit_kl_threshold=getattr(thresholds, "logit_kl", 0.75)
                if thresholds
                else 0.75,
                # Issue #63 new fields.
                latent_cosine_drift_top_threshold=top_threshold,
                latent_cosine_drift_mid_threshold=mid_threshold,
                latent_cosine_drift_token_threshold=token_threshold,
                drift_level_weights=drift_level_weights,
            )
        else:
            sr_config = SafeRecGenConfig(enabled=True)
        safe_recgen = SafeRecGenController(sr_config)
    else:
        sr_config = None
        safe_recgen = None

    # Issue #63 / DR1-005: pass drift_level_weights (not the whole
    # SafeRecGenConfig) into PhotonInference so the inference layer only
    # depends on what it actually needs (ISP).
    drift_weights_for_inference = (
        sr_config.drift_level_weights if sr_config is not None else None
    )
    photon_inference = PhotonInference(
        model,
        photon_cfg,
        tokenizer,
        drift_level_weights=drift_weights_for_inference,
        working_memory_cfg=working_memory_cfg,
    )

    return {
        "photon_inference": photon_inference,
        "safe_recgen": safe_recgen,
        "photon_cfg": photon_cfg,
        "tokenizer": tokenizer,
    }


class _StubTokenizer:
    """Minimal tokenizer for PHOTON prefill (encode/decode via utf-8 byte ids)."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


def _get_stub_tokenizer(vocab_size: int) -> _StubTokenizer:
    return _StubTokenizer(vocab_size)


def _clear_photon_session_state(photon_inference: Any, session_id: str) -> None:
    """Drop PHOTON coarse/prev state and cached logits for ``session_id``.

    Centralised fail-closed reset used in three places (design §8):
    - ``tokenize_evidence_pack`` failure in the pipeline (CB-001).
    - ``reprefill_hierarchy`` Safe RecGen action.
    - ``fallback_to_baseline_path`` Safe RecGen action.

    ``prev_logits`` must be cleared alongside ``current_state`` /
    ``prev_state`` because ``PhotonSessionState.update()`` derives
    ``token_agreement`` / ``logit_kl`` from ``prev_logits`` independently
    of the hierarchy; leaving it set would leak stale drift into the next
    turn (Codex CB-004).
    """
    photon_session = photon_inference._sessions.get(session_id)
    if photon_session is None:
        return
    # Issue #64: delegate to PhotonSessionState.reset_working_memory() so
    # ``turn_history`` is cleared atomically alongside the stale latents
    # while ``drift_history`` / ``turn_count`` are preserved for telemetry.
    photon_session.reset_working_memory()


def build_pipeline(cfg: Config) -> RepoRAGPipeline | PhotonRAGPipeline:
    """Factory: create the right pipeline based on cfg.model.provider.

    CB-004 (codex-fix): the canonical factory lives in
    ``baseline_reporag.pipeline_factory`` so baseline-only entry points can
    route via a module that does not import MLX at load time.  This
    function is a thin backward-compat re-export; prefer importing from
    ``baseline_reporag.pipeline_factory`` directly.
    """
    from .pipeline_factory import build_pipeline as _factory_build_pipeline

    return _factory_build_pipeline(cfg)


# ---------------------------------------------------------------------------
# PhotonRAGPipeline
# ---------------------------------------------------------------------------


class PhotonRAGPipeline:
    """PHOTON-enhanced RepoRAG pipeline with drift tracking and fallback."""

    def __init__(
        self,
        cfg: Config,
        baseline_deps: dict[str, Any],
        photon_deps: dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.baseline = RepoRAGPipeline(
            config=cfg,
            store=baseline_deps["store"],
            lexical=baseline_deps["lexical"],
            embedding=baseline_deps["embedding"],
            graph=baseline_deps["graph"],
            sessions=baseline_deps["sessions"],
            generator=baseline_deps["generator"],
            logger=baseline_deps["logger"],
            reranker=baseline_deps["reranker"],
        )
        self.photon_inference = photon_deps["photon_inference"]
        self.safe_recgen = photon_deps["safe_recgen"]
        self.photon_cfg = photon_deps["photon_cfg"]
        self.tokenizer = photon_deps["tokenizer"]
        # Issue #103: 1-session-1-entry sidecar cache for past-turn pinning.
        # write/read/pop must always go through ``query()`` or
        # :meth:`_clear_photon_session_artifacts` so the lifecycle invariant
        # documented in design §3 (write at end of Turn N, pop at start of
        # Turn N+1) is preserved.
        self._relevant_past_turn_cache: dict[str, TurnState] = {}

    def _clear_photon_session_artifacts(self, session_id: str) -> None:
        """Centralised reset for PHOTON state + Issue #103 sidecar cache.

        Replaces direct ``_clear_photon_session_state`` call sites
        (``tokenize_evidence_pack`` fail-closed, Safe RecGen
        ``reprefill_hierarchy`` / ``fallback_to_baseline_path``) so cache
        cleanup is *always* paired with PHOTON state reset.

        DR1-003 / DR1-007: ``artifacts ⊃ state + cache``. Any future
        session-delete / session-reset API (SessionManager, FastAPI,
        CLI) MUST funnel through this single entry point before mutating
        ``PhotonInference._sessions``. The pop-then-clear order matters
        only insofar as both happen — the cache pop is idempotent
        (``dict.pop(..., None)``) so missing entries are not an error.
        """
        self._relevant_past_turn_cache.pop(session_id, None)
        _clear_photon_session_state(self.photon_inference, session_id)

    @staticmethod
    def _extract_pinned_chunk_ids(
        session: SessionState | None,
        matched: TurnState,
        max_pinned: int,
    ) -> list[str] | None:
        """Look up cited chunks for the matched PHOTON ``turn_id``.

        DR2-004: PHOTON ``turn_count`` and Baseline ``len(turns)`` can drift
        across fail-closed paths (tokenize failure / Safe RecGen reset
        clears ``turn_history`` while preserving ``turn_count``; baseline
        always appends). We therefore guard with
        ``session.turns[idx].turn_id == matched.turn_id`` before trusting
        the index. On drift we fall back to a linear scan; if that still
        fails to locate the matched turn we fail closed (return ``None``)
        rather than risk pinning the wrong chunks.

        DR2-009: dedup is delegated to
        :func:`baseline_reporag.generation.evidence_pack._merge_pinned_sets`
        (set union). This helper performs only the slice; double-counting
        is impossible by construction.

        DR3-001 / DR4-001: the linear-search fallback is O(N) over
        ``session.turns`` but only fires after fail-closed drift; the
        whole helper is wrapped in ``prof.phase("past_turn_pinning")`` by
        the caller. Production telemetry intentionally does not surface
        ``matched_turn_id`` / ``scanned_turns`` — long-session diagnosis
        is restricted to self-hosted benchmarks and unit tests.
        """
        if session is None or not session.turns:
            return None
        idx = matched.turn_id - 1
        if 0 <= idx < len(session.turns):
            candidate = session.turns[idx]
        else:
            candidate = None
        if candidate is None or candidate.turn_id != matched.turn_id:
            # turn_id alignment is broken (PHOTON / Baseline drift after
            # fail-closed). Linear search; failing that, fail closed.
            for t in session.turns:
                if t.turn_id == matched.turn_id:
                    candidate = t
                    break
            else:
                return None
        cited = candidate.cited_chunk_ids
        return list(cited[:max_pinned]) if cited else None

    # ---------------------------------------------------------------
    # Issue #62 Phase 1: opt-in PHOTON single-path generation
    # ---------------------------------------------------------------

    @staticmethod
    def _resolve_photon_max_new_tokens(
        followup_tokens: int | None,
        inference_cfg: Any,
        cfg: Config,
    ) -> int:
        """Resolve the Phase 1 ``max_new_tokens`` contract (DR-62-005 / DR1-004).

        Precedence:
        1. ``followup_tokens`` when non-None (multi-turn cap).
        2. ``inference.answer_max_new_tokens`` when set.
        3. ``generation.max_new_tokens`` when a top-level generation section
           exists (non-photon configs).
        4. Hard default ``512`` (matches Qwen first-turn behaviour).

        Strict type enforcement (DR4-003): rejects ``bool`` and non-``int``,
        rejects values < 1.
        """
        if followup_tokens is not None:
            raw_value: Any = followup_tokens
        else:
            raw_value = getattr(inference_cfg, "answer_max_new_tokens", None)
            if raw_value is None:
                generation_cfg = cfg.get("generation")
                if generation_cfg is not None:
                    raw_value = getattr(generation_cfg, "max_new_tokens", 512)
                else:
                    raw_value = 512

        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(
                "PHOTON max_new_tokens must be a positive int, "
                f"got {type(raw_value).__name__}"
            )
        if raw_value < 1:
            raise ValueError(f"PHOTON max_new_tokens must be >= 1, got {raw_value}")
        return raw_value

    def _run_photon_generation(
        self,
        *,
        messages: list[dict],
        bl: RepoRAGPipeline,
        cfg: Config,
        inference_cfg: Any,
        followup_tokens: int | None,
        fallback_policy: str,
    ) -> tuple[str, str, str | None]:
        """Execute the PHOTON generation branch with fail-closed semantics.

        Returns ``(answer, generator_used, generator_fallback_reason)``.

        Contract (design §8.2 + §9):

        - ``_TokenizerEncodeFailure`` / ``ValueError`` / ``RuntimeError`` →
          fall back to Qwen unless ``fallback_policy == "abort"`` in which
          case a ``RuntimeError`` is raised with a sanitized message.
        - Empty PHOTON output → fall back with ``generator_fallback_reason
          == "empty_output"``.
        - Security logging: warning message uses ``type(exc).__name__``
          only; the raw exception body is never logged (Stage 4 DR4-002).
        """
        from photon_mlx.inference import _TokenizerEncodeFailure

        prompt_text = flatten_messages_for_plain_lm(messages)  # DR-62-003
        photon_max_new = self._resolve_photon_max_new_tokens(
            followup_tokens, inference_cfg, cfg
        )

        try:
            photon_answer = self.photon_inference.generate_answer(
                prompt_text,
                max_new_tokens=photon_max_new,
            )
        except (_TokenizerEncodeFailure, ValueError, RuntimeError) as exc:
            reason = type(exc).__name__
            if fallback_policy == "abort":
                # Sanitized error — do NOT include exc body in the message.
                raise RuntimeError(
                    "PHOTON generation failed and fallback policy=abort"
                ) from None
            # Stage 4 DR4-002: log the closed-enum reason only; the raw
            # exception body must not appear in the warning.
            _logger.warning(
                "PHOTON generation failed; falling back to Qwen (reason=%s)",
                reason,
            )
            qwen_answer = bl.generator.generate(
                messages, max_new_tokens=followup_tokens
            )
            return qwen_answer, "qwen", reason

        # DR1-001: empty / whitespace-only output is fail-closed.
        if not photon_answer or not photon_answer.strip():
            _logger.warning(
                "PHOTON returned empty answer; falling back to Qwen (reason=%s)",
                "empty_output",
            )
            if fallback_policy == "abort":
                raise RuntimeError("PHOTON generation failed and fallback policy=abort")
            qwen_answer = bl.generator.generate(
                messages, max_new_tokens=followup_tokens
            )
            return qwen_answer, "qwen", "empty_output"

        return photon_answer, "photon", None

    def query(
        self,
        question: str,
        session_id: str = "",
        repo_id: str = "",
    ) -> QueryResult:
        """Run PHOTON-enhanced query with drift tracking and evidence pruning.

        On follow-up turns (turn 2+), PHOTON coarse state is used to prune
        the evidence pack from max_chunks down to pruned_max_chunks, halving
        LLM prefill time while retaining the most session-relevant chunks.
        """
        cfg = self.cfg
        bl = self.baseline  # access baseline components without calling query()
        prof = TurnProfiler()
        prof.start()

        session_id = session_id or str(uuid.uuid4())
        repo_id = repo_id or cfg.repo.repo_id
        session = bl.sessions.get_or_create(
            session_id,
            repo_id,
            cfg.repo.repo_commit,
        )

        photon_session_id = session_id or "default"

        # --- Query expansion ---
        qe_cfg = cfg.retrieval.query_expansion
        if qe_cfg.get("enabled", False):
            _queries = expand_query(question)
            expansion_terms: str | None = _queries[1] if len(_queries) > 1 else None
        else:
            expansion_terms = None

        is_follow_up = len(session.turns) > 0

        # --- Two-pass search configuration (Issue #56) ---
        two_pass_enabled, pass1_top_k, pass2_top_k = _resolve_two_pass_search_cfg(
            cfg.retrieval,
            fused_top_k=cfg.retrieval.fused_top_k,
            evidence_max_chunks=cfg.evidence_pack.max_chunks,
        )
        effective_fused_top_k = (
            max(cfg.retrieval.fused_top_k, pass1_top_k)
            if two_pass_enabled and not is_follow_up
            else cfg.retrieval.fused_top_k
        )

        # --- Retrieval ---
        with prof.phase("retrieval"):
            raw = hybrid_search(
                query=question,
                lexical_index=bl.lexical,
                embedding_index=bl.embedding,
                lexical_top_k=cfg.retrieval.lexical_top_k,
                embedding_top_k=cfg.retrieval.embedding_top_k,
                fused_top_k=effective_fused_top_k,
                lexical_weight=cfg.retrieval.weights.lexical,
                embedding_weight=cfg.retrieval.weights.embedding,
                expanded_queries=[expansion_terms] if expansion_terms else [],
            )

        # --- Reranking ---
        # On follow-up turns, PHOTON pruning handles chunk selection.  On turn
        # 1 (or when no reranker is configured), reranking runs as usual.  The
        # current-turn Safe RecGen fallback decision is now computed *after*
        # the evidence pack is built (see design §5.3 / Issue #58), so it no
        # longer gates reranking or pruning within this turn.
        with prof.phase("reranking"):
            if bl.reranker is not None and not is_follow_up:
                raw = bl.reranker.rerank(
                    query=question,
                    results=raw,
                    store=bl.store,
                    top_k=cfg.retrieval.rerank_top_k,
                    rerank_query=expansion_terms,
                )

        # --- File-type boost ---
        file_type_boost = cfg.retrieval.get("file_type_boost", 0.0)
        if file_type_boost:
            raw = apply_file_type_boost(raw, boost=file_type_boost)

        # --- Graph expansion ---
        with prof.phase("graph_expansion"):
            expanded_ids = expand_with_graph(
                results=raw,
                store=bl.store,
                graph=bl.graph,
                repo_id=repo_id,
                repo_commit=cfg.repo.repo_commit,
                max_hops=cfg.retrieval.graph_expansion.max_hops,
                max_nodes=cfg.retrieval.graph_expansion.max_nodes,
                neighborhood_before=cfg.retrieval.neighborhood_expansion.before,
                neighborhood_after=cfg.retrieval.neighborhood_expansion.after,
            )

        # --- Evidence pruning (PHOTON-guided) and Pass 1 scoring (Issue #56) ---
        # Uses the *previous* turn's coarse state on Turn 2+ (1-pass constraint,
        # design §4); Turn 1 optionally scores with a question-derived transient
        # coarse_vec when two_pass_search.enabled=true (Issue #56, DR1-003).
        inference_cfg = cfg.get("inference")
        pruning_enabled = (
            getattr(inference_cfg, "evidence_pruning_enabled", False)
            if inference_cfg is not None
            else False
        )
        pruned_max_chunks = (
            getattr(inference_cfg, "pruned_max_chunks", 8)
            if inference_cfg is not None
            else 8
        )
        effective_max_chunks = cfg.evidence_pack.max_chunks
        do_pass1 = two_pass_enabled and not is_follow_up
        do_pass2plus = pruning_enabled and is_follow_up
        if do_pass1 or do_pass2plus:
            chunks_for_scoring = bl.store.get_many(expanded_ids)
            chunk_texts = [c.content for c in chunks_for_scoring]
            chunk_ids_for_scoring = [c.chunk_id for c in chunks_for_scoring]

            scoring_max_chunks = pass2_top_k if do_pass1 else pruned_max_chunks
            with prof.phase("pass1_scoring" if do_pass1 else "evidence_pruning"):
                selected_indices = self.photon_inference.prune_evidence(
                    chunk_texts=chunk_texts,
                    chunk_ids=chunk_ids_for_scoring,
                    session_id=photon_session_id,
                    max_chunks=scoring_max_chunks,
                    question=question if do_pass1 else None,
                )
            expanded_ids = [chunk_ids_for_scoring[i] for i in selected_indices]
            effective_max_chunks = scoring_max_chunks

        # --- Issue #103: read cached past-turn pin (before evidence pack) ---
        # DR2-001: use the module-level helper (PhotonRAGPipeline has no
        # ``_resolve_working_memory_cfg`` method). DR2-002: the helper
        # returns ``None`` when the YAML lacks ``working_memory:`` or the
        # block is malformed — guard before accessing fields.
        working_memory_cfg = _extract_working_memory_cfg(cfg)
        pinning_enabled = (
            working_memory_cfg is not None
            and working_memory_cfg.past_turn_pinning_enabled
        )
        additional_pinned_ids: list[str] | None = None
        if pinning_enabled and is_follow_up:
            cached_turn = self._relevant_past_turn_cache.pop(photon_session_id, None)
            if cached_turn is not None:
                additional_pinned_ids = self._extract_pinned_chunk_ids(
                    session,
                    cached_turn,
                    working_memory_cfg.max_pinned_chunks,
                )

        # --- Evidence pack ---
        with prof.phase("evidence_pack"):
            pack = build_evidence_pack(
                chunk_ids=expanded_ids,
                store=bl.store,
                session=session,
                max_chunks=effective_max_chunks,
                max_tokens=cfg.evidence_pack.max_tokens,
                additional_pinned_ids=additional_pinned_ids,
            )

        # --- PHOTON prefill on question + evidence (new coarse state) ---
        # Issue #58: the coarse state is now built from the concatenation of
        # the question and the evidence text so drift, Safe RecGen, and the
        # next turn's prune_evidence operate in a richer semantic space.
        # Fail-closed: if tokenization fails we clear the PHOTON session
        # state and fall through to the baseline generation path rather than
        # silently reusing a stale coarse state on the next turn (design §8
        # + CB-001).
        evidence_text_for_photon = pack.format_for_prompt()
        photon_input_text = question + "\n\n" + evidence_text_for_photon
        drift = None
        drift_dict = None
        confidence = 1.0
        tokenization_failed = False
        try:
            evidence_tokens = tokenize_evidence_pack(
                photon_input_text,
                self.tokenizer,
                self.photon_cfg,
            )
        except Exception as exc:
            # Security logging (Issue #58 CB-001 + Issue #64 Codex CB-002):
            # log only the closed-enum exception class name. The tokenizer
            # was handed ``question + evidence_pack``; a pathological or
            # mis-configured tokenizer could echo that payload back in its
            # exception message, so surfacing ``str(exc)`` / ``%s % exc``
            # would leak question/evidence fragments to log sinks (design §7
            # bars raw question_text and attacker-controlled values from
            # fail-closed telemetry).
            _logger.warning(
                "tokenize_evidence_pack failed; clearing PHOTON session "
                "state and falling back to baseline path for this turn "
                "(fail-closed, CB-001, Codex CB-002, reason=%s)",
                type(exc).__name__,
            )
            tokenization_failed = True
            evidence_tokens = mx.array([], dtype=mx.int32)
            # Explicit fail-closed: drop any prior coarse/prev state so the
            # next turn cannot reuse a stale hierarchy.  No raw input text,
            # token ids, or latents are retained (design §8). Issue #103
            # routes through ``_clear_photon_session_artifacts`` so the
            # past-turn pin sidecar cache is also dropped.
            self._clear_photon_session_artifacts(photon_session_id)

        if evidence_tokens.size > 0:
            input_ids = evidence_tokens.reshape(1, -1)
            logits, drift = self.photon_inference.session_forward(
                input_ids,
                session_id=photon_session_id,
                repo_id=repo_id or "unknown",
                repo_commit="HEAD",
                question=question,
            )
            confidence = compute_confidence(logits)
            drift_dict = drift.as_dict() if drift else None

        # --- Safe RecGen evaluation (uses new coarse state) ---
        fallback_dict = None
        if self.safe_recgen is not None and drift is not None:
            decision = self.safe_recgen.evaluate(
                question, drift=drift, confidence=confidence
            )
            fallback_dict = decision.as_dict()
        fallback_actions = (
            set(fallback_dict.get("actions", [])) if fallback_dict else set()
        )

        # A fallback that invalidates the PHOTON hierarchy must clear the
        # session state (including prev_logits) so subsequent turns do not
        # reuse a coarse state or drift reference from a stale topic
        # (design §8 fail-closed; Codex CB-004). Issue #103 routes through
        # ``_clear_photon_session_artifacts`` so the past-turn pin sidecar
        # cache is dropped in lockstep with PHOTON state.
        if fallback_actions & {"reprefill_hierarchy", "fallback_to_baseline_path"}:
            self._clear_photon_session_artifacts(photon_session_id)

        # --- Issue #103: write past-turn pin cache for next turn ---
        # 3 branches (design §4-3):
        #   OFF                       → skip entirely (no profiler phase).
        #   drift is None             → pop only (DR2-011 stale-cache safety).
        #   drift is not None         → try find_relevant_past_turn.
        # DR4-001: production observability is limited to the
        # ``past_turn_pinning`` phase duration and the failure exception
        # class name — turn_id, similarity, and scanned_turns are NOT
        # logged so attacker-controlled YAML or pathological session state
        # cannot leak into log sinks.
        if pinning_enabled:
            if drift is None:
                # tokenize fail-closed / Safe RecGen reset / session_forward
                # not run: drop any stale cache entry from the prior turn so
                # the next turn cannot consume a misaligned pin.
                self._relevant_past_turn_cache.pop(photon_session_id, None)
            else:
                with prof.phase("past_turn_pinning"):
                    photon_session = self.photon_inference._sessions.get(
                        photon_session_id
                    )
                    match: TurnState | None
                    if photon_session is not None:
                        try:
                            match = photon_session.find_relevant_past_turn(
                                photon_session.current_state
                            )
                        except (AttributeError, RuntimeError, ValueError) as exc:
                            # DR1-002 + Codex CB-001/CB-002: closed exception
                            # set (no Pokémon catch). Only the type name is
                            # surfaced, never raw message content.
                            _logger.warning(
                                "find_relevant_past_turn failed; skipping "
                                "pin cache (fail-closed, reason=%s)",
                                type(exc).__name__,
                            )
                            match = None
                    else:
                        # PHOTON session was never initialised for this
                        # session_id — keep the cache empty.
                        match = None

                    if match is not None:
                        self._relevant_past_turn_cache[photon_session_id] = match
                    else:
                        self._relevant_past_turn_cache.pop(photon_session_id, None)

        # --- Generation (Issue #62 Phase 1: opt-in PHOTON single-path) ---
        # DR-62-001 / DR4-003: strict bool validation for the opt-in flag.
        raw_photon_gen_enabled = (
            getattr(inference_cfg, "photon_generation_enabled", False)
            if inference_cfg is not None
            else False
        )
        if not isinstance(raw_photon_gen_enabled, bool):
            raise ValueError(
                "inference.photon_generation_enabled must be bool, "
                f"got {type(raw_photon_gen_enabled).__name__}"
            )
        photon_gen_enabled = raw_photon_gen_enabled

        # DR4-004: closed-enum validation for the deployment policy knob.
        fallback_policy = (
            getattr(inference_cfg, "generation_fallback_policy", "qwen")
            if inference_cfg is not None
            else "qwen"
        )
        if fallback_policy not in {"qwen", "abort"}:
            raise ValueError(
                "inference.generation_fallback_policy must be 'qwen' or 'abort', "
                f"got {fallback_policy!r}"
            )

        generator_used = "qwen"
        generator_fallback_reason: str | None = None

        with prof.phase("generation"):
            evidence_text = pack.format_for_prompt()
            is_first_turn = len(session.turns) == 0
            if is_first_turn:
                evidence_text = f"{_EVIDENCE_HEADER}\n\n{evidence_text}"
            messages = build_messages(
                question=question,
                evidence_text=evidence_text,
                history_text=session.history_text(max_turns=4),
                include_few_shot=is_first_turn,
            )
            followup_tokens = 512 if not is_first_turn else None

            if photon_gen_enabled:
                (
                    answer,
                    generator_used,
                    generator_fallback_reason,
                ) = self._run_photon_generation(
                    messages=messages,
                    bl=bl,
                    cfg=cfg,
                    inference_cfg=inference_cfg,
                    followup_tokens=followup_tokens,
                    fallback_policy=fallback_policy,
                )
            else:
                answer = bl.generator.generate(messages, max_new_tokens=followup_tokens)

        # --- Citation ---
        with prof.phase("citation"):
            citation = resolve_citations(answer, pack)
            answering_cfg = getattr(cfg, "answering", None)
            if answering_cfg is not None:
                postprocess_enabled = answering_cfg.get(
                    "citation_postprocess_enabled", True
                )
            else:
                postprocess_enabled = True
            if not isinstance(postprocess_enabled, bool):
                raise RuntimeError(
                    "answering.citation_postprocess_enabled must be bool, "
                    f"got {type(postprocess_enabled)}"
                )
            answer, citation, citation_postprocessed = apply_citation_postprocess(
                answer, pack, citation, enabled=postprocess_enabled
            )

        latency, memory = prof.finish()

        # --- Session update ---
        session_cited_ids = [] if citation_postprocessed else citation.cited_chunk_ids
        turn = session.add_turn(question, answer, session_cited_ids)
        bl.sessions.save(session)

        # --- Log ---
        bl.logger.log_turn(
            {
                "session_id": session_id,
                "turn_id": turn.turn_id,
                "repo_id": repo_id,
                "repo_commit": cfg.repo.repo_commit,
                "model_id": cfg.model.model_id,
                "question": question,
                "answer": answer,
                "retrieval_chunk_ids": [r.chunk_id for r in raw],
                "evidence_pack_ids": [c.chunk_id for c in pack.chunks],
                "cited_chunk_ids": citation.cited_chunk_ids,
                "wrong_citation_indices": citation.wrong_citation_indices,
                "no_citation": citation.no_citation,
                "citation_postprocessed": citation_postprocessed,
                "latency": latency.as_dict(),
                "memory": memory.as_dict(),
                "fallback_flag": bool(
                    fallback_dict and fallback_dict.get("should_fallback")
                ),
                "fallback_reason": (
                    fallback_dict.get("reasons") if fallback_dict else None
                ),
                "evidence_pruning_applied": (pruning_enabled and is_follow_up),
                "photon_tokenization_failed": tokenization_failed,
                # Issue #62 Phase 1: generation-level observability.
                # ``generator_used`` ∈ {"photon", "qwen"} and
                # ``generator_fallback_reason`` is a closed enum (§7.2):
                # None | "_TokenizerEncodeFailure" | "ValueError"
                #      | "RuntimeError" | "empty_output".
                "generator_used": generator_used,
                "generator_fallback_reason": generator_fallback_reason,
            }
        )

        result = QueryResult(
            answer=answer,
            session_id=session_id,
            turn_id=turn.turn_id,
            cited_chunk_ids=citation.cited_chunk_ids,
            wrong_citation_indices=citation.wrong_citation_indices,
            no_citation=citation.no_citation,
            latency=latency,
            memory=memory,
            citation_postprocessed=citation_postprocessed,
            # Issue #62 Phase 1 (CB-003 codex-fix): expose the generator
            # that produced ``answer`` on the structured result so
            # comparison tools can distinguish a real PHOTON answer from
            # a Qwen fallback without having to parse the log stream.
            generator_used=generator_used,
            generator_fallback_reason=generator_fallback_reason,
        )

        # Attach PHOTON metadata
        result.drift_metrics = drift_dict
        result.confidence = confidence
        result.fallback_decision = fallback_dict

        return result
