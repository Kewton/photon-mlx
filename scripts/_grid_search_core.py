"""Pure-function core of the retrieval grid-search tool (Issue #88).

This module is deliberately MLX-free so it can be imported from unit
tests on a CI runner that has no MLX installed. It provides:

- ``ConfigParams`` / ``ConfigResult`` dataclasses
- ``generate_phase1_grid`` / ``generate_phase2_grid`` (grid construction)
- ``is_invalid_combo`` (4-condition filter per design §5)
- ``aggregate_metrics`` (pinned against ``ci_eval_check.check_static``)
- ``validate_override`` (fail-fast key typo guard)
- ``atomic_write_json`` / ``write_markdown_report`` (IO helpers)

The CLI-facing ``scripts/retrieval_grid_search.py`` is the only place
that imports MLX-backed pipeline code.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from baseline_reporag.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigParams:
    """Sweep-target parameters. Immutable + hashable."""

    lexical_top_k: int
    embedding_top_k: int
    fused_top_k: int
    rerank_top_k: int
    weights_lexical: float
    weights_embedding: float

    def to_override_dict(self) -> dict[str, Any]:
        return {
            "lexical_top_k": self.lexical_top_k,
            "embedding_top_k": self.embedding_top_k,
            "fused_top_k": self.fused_top_k,
            "rerank_top_k": self.rerank_top_k,
            "weights": {
                "lexical": self.weights_lexical,
                "embedding": self.weights_embedding,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConfigParams:
        weights = d.get("weights", {})
        return cls(
            lexical_top_k=int(d["lexical_top_k"]),
            embedding_top_k=int(d["embedding_top_k"]),
            fused_top_k=int(d["fused_top_k"]),
            rerank_top_k=int(d["rerank_top_k"]),
            weights_lexical=float(
                weights.get("lexical", d.get("weights_lexical", 0.0))
            ),
            weights_embedding=float(
                weights.get("embedding", d.get("weights_embedding", 0.0))
            ),
        )


@dataclass
class ConfigResult:
    """Per-config aggregated result. Serialized to JSON with _ms suffix."""

    config_idx: int
    params: ConfigParams
    raw_no_citation_rate: float
    true_nc_rate: float
    wrong_citation_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    n_questions: int
    n_no_citation: int
    duration_seconds: float
    started_at: str
    completed_at: str
    memory_peak_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "config_idx": self.config_idx,
            "params": self.params.to_override_dict(),
            "raw_no_citation_rate": self.raw_no_citation_rate,
            "true_nc_rate": self.true_nc_rate,
            "wrong_citation_count": self.wrong_citation_count,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "n_questions": self.n_questions,
            "n_no_citation": self.n_no_citation,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.memory_peak_mb is not None:
            out["memory_peak_mb"] = self.memory_peak_mb
        if self.extra:
            out["extra"] = self.extra
        return out


# ---------------------------------------------------------------------------
# Invalid combo filter (design §5)
# ---------------------------------------------------------------------------


_WEIGHTS_SUM_MAX = 0.90
_WEIGHTS_SUM_EPS = 1e-9


def is_invalid_combo(p: ConfigParams) -> bool:
    """Return True when ``p`` violates at least one of the 4 sanity rules.

    See design §5 for the full rationale of each condition.
    """
    if p.rerank_top_k > p.fused_top_k:
        return True
    if p.lexical_top_k <= 0 or p.embedding_top_k <= 0 or p.fused_top_k <= 0:
        return True
    if (p.weights_lexical + p.weights_embedding) > _WEIGHTS_SUM_MAX + _WEIGHTS_SUM_EPS:
        return True
    if p.lexical_top_k <= 15 and p.embedding_top_k <= 15 and p.fused_top_k >= 20:
        return True
    return False


# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------


_PHASE1_LEX_EMB: tuple[tuple[int, int], ...] = ((15, 15), (20, 20), (25, 25))
_PHASE1_FUSED_RERANK: tuple[tuple[int, int], ...] = ((10, 8), (16, 12), (20, 16))
_PHASE1_WEIGHTS: tuple[tuple[float, float], ...] = (
    (0.55, 0.35),
    (0.45, 0.45),
    (0.35, 0.55),
)


def generate_phase1_grid() -> list[ConfigParams]:
    """Return the 24 valid Phase 1 configs (27 candidates minus 3 invalid)."""
    valid: list[ConfigParams] = []
    for lex, emb in _PHASE1_LEX_EMB:
        for fused, rerank in _PHASE1_FUSED_RERANK:
            for w_lex, w_emb in _PHASE1_WEIGHTS:
                params = ConfigParams(
                    lexical_top_k=lex,
                    embedding_top_k=emb,
                    fused_top_k=fused,
                    rerank_top_k=rerank,
                    weights_lexical=w_lex,
                    weights_embedding=w_emb,
                )
                if is_invalid_combo(params):
                    continue
                valid.append(params)
    return valid


def _key(params: ConfigParams) -> str:
    return json.dumps(params.to_override_dict(), sort_keys=True)


def generate_phase2_grid(
    seeds: list[ConfigParams],
    *,
    topk_delta: int = 5,
    topk_step: int = 5,
    weight_delta: float = 0.05,
    weight_step: float = 0.05,
) -> list[ConfigParams]:
    """Generate Phase 2 neighborhood around each seed.

    For every seed we walk ``top_k ± topk_delta`` in ``topk_step`` increments
    on each of the four ``*_top_k`` axes, and ``weights ± weight_delta`` in
    ``weight_step`` increments on the two weight axes. Invalid combos are
    filtered out and duplicates (including the seed itself) are dropped.
    """
    if not seeds:
        return []

    seed_keys = {_key(s) for s in seeds}
    seen: set[str] = set(seed_keys)
    result: list[ConfigParams] = []

    topk_offsets = _range_int(-topk_delta, topk_delta, topk_step)
    weight_offsets = _range_float(-weight_delta, weight_delta, weight_step)

    for seed in seeds:
        for d_lex in topk_offsets:
            for d_emb in topk_offsets:
                for d_fused in topk_offsets:
                    for d_rerank in topk_offsets:
                        for d_wl in weight_offsets:
                            for d_we in weight_offsets:
                                cand = ConfigParams(
                                    lexical_top_k=seed.lexical_top_k + d_lex,
                                    embedding_top_k=seed.embedding_top_k + d_emb,
                                    fused_top_k=seed.fused_top_k + d_fused,
                                    rerank_top_k=seed.rerank_top_k + d_rerank,
                                    weights_lexical=round(
                                        seed.weights_lexical + d_wl, 4
                                    ),
                                    weights_embedding=round(
                                        seed.weights_embedding + d_we, 4
                                    ),
                                )
                                if is_invalid_combo(cand):
                                    continue
                                k = _key(cand)
                                if k in seen:
                                    continue
                                seen.add(k)
                                result.append(cand)
    return result


def _range_int(lo: int, hi: int, step: int) -> list[int]:
    if step <= 0:
        return [0]
    out: list[int] = []
    n = lo
    while n <= hi + 0:
        out.append(n)
        n += step
    return out


def _range_float(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        return [0.0]
    out: list[float] = []
    # Use integer math to avoid floating-point drift.
    n_steps = int(round((hi - lo) / step))
    for i in range(n_steps + 1):
        out.append(round(lo + i * step, 6))
    return out


# ---------------------------------------------------------------------------
# aggregate_metrics (pinned against ci_eval_check.check_static)
# ---------------------------------------------------------------------------


def aggregate_metrics(
    records: list[dict[str, Any]],
    unanswerable_ids: set[str],
) -> dict[str, Any]:
    """Compute the same metrics as ``ci_eval_check.check_static``.

    Returned keys match that function's dict (no ``_ms`` suffix on latency).
    ``true_nc_rate`` is only present when ``unanswerable_ids`` is non-empty,
    matching the reference implementation.
    """
    total = len(records)
    if total == 0:
        return {"total": 0, "error": "no records found"}

    no_cite = sum(1 for r in records if r.get("no_citation"))
    wrong_cite = sum(1 for r in records if r.get("wrong_citation_indices"))
    latencies = [lat for r in records if (lat := _extract_latency(r)) is not None]

    answerable_records = [r for r in records if _eval_id(r) not in unanswerable_ids]
    answerable_total = len(answerable_records)
    answerable_no_cite = sum(1 for r in answerable_records if r.get("no_citation"))
    correct_abstains = sum(
        1 for r in records if _eval_id(r) in unanswerable_ids and r.get("no_citation")
    )

    result: dict[str, Any] = {
        "total": total,
        "no_citation_rate": no_cite / total,
        "wrong_citation_count": wrong_cite,
        "latency_p50": statistics.median(latencies) if latencies else 0,
        "latency_p95": _percentile(latencies, 0.95) if latencies else 0,
        "n_questions": total,
        "n_no_citation": no_cite,
    }

    if unanswerable_ids:
        result["unanswerable_count"] = len(unanswerable_ids)
        result["correct_abstains"] = correct_abstains
        result["answerable_total"] = answerable_total
        result["answerable_no_cite"] = answerable_no_cite
        result["true_nc_rate"] = (
            answerable_no_cite / answerable_total if answerable_total else 0
        )

    return result


def _extract_latency(record: dict[str, Any]) -> float | None:
    if "latency" in record and isinstance(record["latency"], dict):
        return record["latency"].get("total_ms")
    return record.get("latency_ms")


def _eval_id(record: dict[str, Any]) -> str:
    qid = record.get("eval_id", "") or record.get("id", "")
    if not qid:
        sid = record.get("session_id", "")
        qid = sid.replace("eval-", "", 1) if sid.startswith("eval-") else sid
    return qid


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # nearest-rank method: ceil(q * N) - 1, clipped to [0, N-1].
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


# ---------------------------------------------------------------------------
# validate_override (O8: fail-fast before 16h run)
# ---------------------------------------------------------------------------


def validate_override(base_cfg: Config, params: ConfigParams) -> None:
    """Raise ValueError if ``params`` contains a key absent from ``base_cfg.retrieval``.

    Catches typos (e.g. ``lexcial_top_k``) before they silently collapse to
    the base config value during a 16h sweep.
    """
    retrieval_cfg = getattr(base_cfg, "retrieval", None)
    if retrieval_cfg is None:
        raise ValueError("base_cfg has no .retrieval section")

    base_keys = set(_cfg_to_dict(retrieval_cfg).keys())
    override = params.to_override_dict()
    for key, value in override.items():
        if key not in base_keys:
            raise ValueError(
                f"Unknown retrieval key: {key!r} (not in base_cfg.retrieval)"
            )
        if key == "weights" and isinstance(value, dict):
            weights_base = _cfg_to_dict(getattr(retrieval_cfg, "weights", {}))
            for wk in value:
                if wk not in weights_base:
                    raise ValueError(f"Unknown retrieval key: weights.{wk}")


def _cfg_to_dict(cfg_or_dict: Any) -> dict[str, Any]:
    if hasattr(cfg_or_dict, "to_dict"):
        return cfg_or_dict.to_dict()
    if isinstance(cfg_or_dict, dict):
        return cfg_or_dict
    return dict(vars(cfg_or_dict))


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically replace ``path`` with a JSON dump of ``payload``.

    The tmp file is created in the same directory (same filesystem) so
    ``os.replace`` is atomic on POSIX. Permissions are restricted to the
    owner and ``fsync`` is called before the rename so a power loss after
    rename still yields a durable, valid file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=target.stem + ".",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            os.chmod(tmp.name, 0o600)
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        tmp_path.replace(target)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.warning("Failed to clean up tmp file %s", tmp_path)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(
    state: dict[str, Any],
    best: ConfigResult,
    out_path: str | Path,
) -> None:
    """Render a Markdown summary of all configs with best highlighted."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Retrieval Grid Search Report")
    lines.append("")
    lines.append(f"- Base config: `{state.get('base_config_path', '')}`")
    lines.append(f"- Phase: `{state.get('phase', '')}`")
    lines.append(
        f"- Max questions per config: {state.get('max_questions_per_config', '')}"
    )
    lines.append(f"- Started at: {state.get('started_at', '')}")
    lines.append(f"- Completed at: {state.get('completed_at', '')}")
    lines.append(f"- Total configs: {len(state.get('configs', []))}")
    lines.append("")

    lines.append("## Best Config")
    lines.append("")
    lines.append(
        f"- **config_idx**: {best.config_idx}  "
        f"(lexical_top_k={best.params.lexical_top_k}, "
        f"embedding_top_k={best.params.embedding_top_k}, "
        f"fused_top_k={best.params.fused_top_k}, "
        f"rerank_top_k={best.params.rerank_top_k}, "
        f"weights_lexical={best.params.weights_lexical}, "
        f"weights_embedding={best.params.weights_embedding})"
    )
    lines.append(
        f"- **raw no_citation_rate**: {best.raw_no_citation_rate:.2%} "
        f"({best.n_no_citation}/{best.n_questions})"
    )
    lines.append(f"- **true_nc_rate**: {best.true_nc_rate:.2%}")
    lines.append(f"- **wrong_citation_count**: {best.wrong_citation_count}")
    lines.append(f"- **latency_p50_ms**: {best.latency_p50_ms:.0f}")
    lines.append(f"- **latency_p95_ms**: {best.latency_p95_ms:.0f}")
    lines.append(f"- **duration_seconds**: {best.duration_seconds:.1f}")
    lines.append("")

    lines.append("## All Configs")
    lines.append("")
    lines.append(
        "| idx | lex | emb | fused | rerank | w_lex | w_emb | "
        "raw NC | true NC | p50 ms | p95 ms | wrong |"
    )
    lines.append(
        "|-----|-----|-----|-------|--------|-------|-------|"
        "--------|---------|--------|--------|-------|"
    )
    for entry in state.get("configs", []):
        params = entry.get("params", {})
        weights = params.get("weights", {})
        lines.append(
            "| {idx} | {lex} | {emb} | {fused} | {rerank} | "
            "{wl:.2f} | {we:.2f} | {raw:.2%} | {true_nc} | "
            "{p50:.0f} | {p95:.0f} | {wrong} |".format(
                idx=entry.get("config_idx", ""),
                lex=params.get("lexical_top_k", ""),
                emb=params.get("embedding_top_k", ""),
                fused=params.get("fused_top_k", ""),
                rerank=params.get("rerank_top_k", ""),
                wl=float(weights.get("lexical", 0.0)),
                we=float(weights.get("embedding", 0.0)),
                raw=entry.get("raw_no_citation_rate", 0.0),
                true_nc=_format_pct_or_dash(entry.get("true_nc_rate")),
                p50=entry.get("latency_p50_ms", 0.0),
                p95=entry.get("latency_p95_ms", 0.0),
                wrong=entry.get("wrong_citation_count", 0),
            )
        )

    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")


def _format_pct_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "-"
