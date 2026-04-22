"""Unit tests for scripts/_grid_search_core (Issue #88).

Covers the pure-function grid search core module per design §7.1:
ConfigParams / ConfigResult, grid generation, invalid-combo filtering,
metric aggregation pinned against ci_eval_check.check_static, and
override validation.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._grid_search_core import (  # noqa: E402
    ConfigParams,
    ConfigResult,
    aggregate_metrics,
    atomic_write_json,
    generate_phase1_grid,
    generate_phase2_grid,
    is_invalid_combo,
    validate_override,
    write_markdown_report,
)


# ---------------------------------------------------------------------------
# ConfigParams / ConfigResult
# ---------------------------------------------------------------------------


def _sample_params(**overrides: object) -> ConfigParams:
    base = dict(
        lexical_top_k=20,
        embedding_top_k=20,
        fused_top_k=16,
        rerank_top_k=12,
        weights_lexical=0.45,
        weights_embedding=0.45,
    )
    base.update(overrides)
    return ConfigParams(**base)  # type: ignore[arg-type]


def test_config_params_to_override_dict() -> None:
    params = _sample_params()
    override = params.to_override_dict()

    assert override["lexical_top_k"] == 20
    assert override["embedding_top_k"] == 20
    assert override["fused_top_k"] == 16
    assert override["rerank_top_k"] == 12
    assert override["weights"] == {"lexical": 0.45, "embedding": 0.45}


def test_config_params_roundtrip() -> None:
    original = _sample_params(weights_lexical=0.55, weights_embedding=0.35)
    restored = ConfigParams.from_dict(original.to_override_dict())
    assert restored == original


def test_config_result_to_json_uses_ms_suffix() -> None:
    params = _sample_params()
    result = ConfigResult(
        config_idx=3,
        params=params,
        raw_no_citation_rate=0.175,
        true_nc_rate=0.16,
        wrong_citation_count=1,
        latency_p50_ms=19500.0,
        latency_p95_ms=28000.0,
        n_questions=40,
        n_no_citation=7,
        duration_seconds=780.5,
        started_at="2026-04-22T10:01:00+09:00",
        completed_at="2026-04-22T10:14:00+09:00",
    )
    j = result.to_json()
    assert j["config_idx"] == 3
    assert j["latency_p50_ms"] == 19500.0
    assert j["latency_p95_ms"] == 28000.0
    assert j["params"] == params.to_override_dict()


# ---------------------------------------------------------------------------
# is_invalid_combo
# ---------------------------------------------------------------------------


def test_is_invalid_combo_rerank_exceeds_fused() -> None:
    p = _sample_params(fused_top_k=16, rerank_top_k=20)
    assert is_invalid_combo(p) is True


def test_is_invalid_combo_non_positive_topk() -> None:
    p = _sample_params(lexical_top_k=0)
    assert is_invalid_combo(p) is True


def test_is_invalid_combo_weights_sum_overflow() -> None:
    p = _sample_params(weights_lexical=0.55, weights_embedding=0.55)
    assert is_invalid_combo(p) is True


def test_is_invalid_combo_known_cases() -> None:
    assert (
        is_invalid_combo(
            _sample_params(
                lexical_top_k=15,
                embedding_top_k=15,
                fused_top_k=20,
                rerank_top_k=16,
            )
        )
        is True
    )
    assert is_invalid_combo(_sample_params()) is False


def test_is_invalid_combo_weights_sum_exactly_0_90_is_valid() -> None:
    p = _sample_params(weights_lexical=0.55, weights_embedding=0.35)
    assert is_invalid_combo(p) is False


# ---------------------------------------------------------------------------
# generate_phase1_grid
# ---------------------------------------------------------------------------


def test_generate_phase1_grid_count() -> None:
    grid = generate_phase1_grid()
    assert len(grid) == 24


def test_phase1_grid_no_invalid_combos() -> None:
    for params in generate_phase1_grid():
        assert not is_invalid_combo(params), params


def test_phase1_grid_unique_configs() -> None:
    grid = generate_phase1_grid()
    seen = {json.dumps(p.to_override_dict(), sort_keys=True) for p in grid}
    assert len(seen) == len(grid)


# ---------------------------------------------------------------------------
# generate_phase2_grid
# ---------------------------------------------------------------------------


def test_phase2_neighborhood_dedup() -> None:
    seeds = [
        _sample_params(),
        _sample_params(),
    ]
    grid = generate_phase2_grid(seeds)
    seen = {json.dumps(p.to_override_dict(), sort_keys=True) for p in grid}
    assert len(seen) == len(grid)


def test_phase2_no_invalid_combos() -> None:
    seeds = [_sample_params()]
    for p in generate_phase2_grid(seeds):
        assert not is_invalid_combo(p), p


def test_phase2_excludes_seed_itself() -> None:
    seed = _sample_params()
    grid = generate_phase2_grid([seed])
    seed_key = json.dumps(seed.to_override_dict(), sort_keys=True)
    keys = {json.dumps(p.to_override_dict(), sort_keys=True) for p in grid}
    assert seed_key not in keys


# ---------------------------------------------------------------------------
# aggregate_metrics (pin against scripts.ci_eval_check.check_static)
# ---------------------------------------------------------------------------


def _sample_records() -> list[dict]:
    return [
        {
            "eval_id": "SE-001",
            "no_citation": False,
            "wrong_citation_indices": [],
            "latency_ms": 10_000.0,
        },
        {
            "eval_id": "SE-002",
            "no_citation": True,
            "wrong_citation_indices": [],
            "latency_ms": 20_000.0,
        },
        {
            "eval_id": "SE-003",
            "no_citation": False,
            "wrong_citation_indices": [2],
            "latency_ms": 30_000.0,
        },
        {
            "eval_id": "SE-UA-001",
            "no_citation": True,
            "wrong_citation_indices": [],
            "latency_ms": 5_000.0,
        },
    ]


def test_aggregate_metrics_matches_ci_eval_check(tmp_path: Path) -> None:
    from scripts.ci_eval_check import check_static

    records = _sample_records()
    unanswerable_ids = {"SE-UA-001"}

    metrics = aggregate_metrics(records, unanswerable_ids)

    log_path = tmp_path / "fake.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    ref = check_static(str(log_path), unanswerable_ids=unanswerable_ids)

    assert metrics["no_citation_rate"] == pytest.approx(ref["no_citation_rate"])
    assert metrics["wrong_citation_count"] == ref["wrong_citation_count"]
    assert metrics["latency_p50"] == pytest.approx(ref["latency_p50"])
    assert metrics["true_nc_rate"] == pytest.approx(ref["true_nc_rate"])


def test_aggregate_metrics_p50_calculation() -> None:
    records = _sample_records()
    metrics = aggregate_metrics(records, unanswerable_ids=set())
    expected_p50 = statistics.median([r["latency_ms"] for r in records])
    assert metrics["latency_p50"] == pytest.approx(expected_p50)


def test_aggregate_metrics_no_true_nc_when_empty_unanswerable() -> None:
    records = _sample_records()
    metrics = aggregate_metrics(records, unanswerable_ids=set())
    assert "true_nc_rate" not in metrics


def test_aggregate_metrics_counts_wrong_citation_records() -> None:
    records = _sample_records()
    metrics = aggregate_metrics(records, unanswerable_ids=set())
    # ci_eval_check counts records with non-empty wrong_citation_indices.
    assert metrics["wrong_citation_count"] == 1


def test_aggregate_metrics_latency_p95() -> None:
    records = _sample_records()
    metrics = aggregate_metrics(records, unanswerable_ids=set())
    # p95 of 4 points = index min(len-1, ceil(.95 * len) - 1) = 3
    assert metrics["latency_p95"] == pytest.approx(30_000.0)


# ---------------------------------------------------------------------------
# validate_override
# ---------------------------------------------------------------------------


def _load_base_cfg():  # type: ignore[no-untyped-def]
    from baseline_reporag.config import load_config

    return load_config(REPO_ROOT / "configs" / "baseline.yaml")


def test_validate_override_rejects_unknown_key() -> None:
    base_cfg = _load_base_cfg()
    bad = _sample_params()
    # Patch override dict to include a typo'd key by wrapping in a subclass
    # that returns a custom dict. validate_override must inspect the dict.

    class _BadParams(ConfigParams):
        def to_override_dict(self) -> dict:  # type: ignore[override]
            d = super().to_override_dict()
            d["lexcial_top_k"] = 20  # intentional typo
            return d

    bad_typed = _BadParams(
        lexical_top_k=bad.lexical_top_k,
        embedding_top_k=bad.embedding_top_k,
        fused_top_k=bad.fused_top_k,
        rerank_top_k=bad.rerank_top_k,
        weights_lexical=bad.weights_lexical,
        weights_embedding=bad.weights_embedding,
    )
    with pytest.raises(ValueError, match="Unknown retrieval key"):
        validate_override(base_cfg, bad_typed)


def test_validate_override_passes_valid_params() -> None:
    base_cfg = _load_base_cfg()
    params = _sample_params()
    validate_override(base_cfg, params)  # must not raise


# ---------------------------------------------------------------------------
# merge_override preserves sibling keys (D2-a / D3 pin)
# ---------------------------------------------------------------------------


def test_merge_override_preserves_reranker() -> None:
    base_cfg = _load_base_cfg()
    params = _sample_params()
    cfg2 = base_cfg.merge_override({"retrieval": params.to_override_dict()})
    assert cfg2.retrieval.reranker.enabled is True


def test_merge_override_preserves_graph_weight() -> None:
    base_cfg = _load_base_cfg()
    params = _sample_params()
    cfg2 = base_cfg.merge_override({"retrieval": params.to_override_dict()})
    assert cfg2.retrieval.weights.graph == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# atomic_write_json
# ---------------------------------------------------------------------------


def test_atomic_write_json_writes_content(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    payload = {"configs": [{"config_idx": 0}], "phase": "phase1"}
    atomic_write_json(target, payload)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_atomic_write_json_replaces_existing(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"stale": true}', encoding="utf-8")
    payload = {"fresh": True}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_atomic_write_json_no_tmp_leftover(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": True})
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_atomic_write_json_chmod_owner_only(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": True})
    mode = target.stat().st_mode & 0o777
    # Owner-only permissions: no group / world access.
    assert mode & 0o077 == 0


# ---------------------------------------------------------------------------
# write_markdown_report
# ---------------------------------------------------------------------------


def test_write_markdown_report_renders_best_config(tmp_path: Path) -> None:
    params = _sample_params()
    best = ConfigResult(
        config_idx=0,
        params=params,
        raw_no_citation_rate=0.125,
        true_nc_rate=0.10,
        wrong_citation_count=0,
        latency_p50_ms=19000.0,
        latency_p95_ms=27000.0,
        n_questions=40,
        n_no_citation=5,
        duration_seconds=780.0,
        started_at="2026-04-22T10:00:00+09:00",
        completed_at="2026-04-22T10:13:00+09:00",
    )
    state = {
        "phase": "both",
        "base_config_path": "configs/baseline.yaml",
        "max_questions_per_config": 40,
        "started_at": "2026-04-22T10:00:00+09:00",
        "completed_at": "2026-04-22T10:13:00+09:00",
        "configs": [best.to_json()],
    }
    out = tmp_path / "report.md"
    write_markdown_report(state, best, out)

    text = out.read_text(encoding="utf-8")
    assert "# Retrieval Grid Search" in text
    assert "12.50%" in text  # raw NC as %
    assert "lexical_top_k=20" in text or "lexical_top_k: 20" in text


# ---------------------------------------------------------------------------
# resume skip (state helper used by run_phase)
# ---------------------------------------------------------------------------


def test_atomic_write_json_survives_partial_write(tmp_path: Path) -> None:
    # Write once, verify, then overwrite and verify again. The tmp+replace
    # pattern guarantees the reader either sees the old or the new file,
    # never a truncated one. We can't simulate a real crash inside a unit
    # test, but we can assert the contract's key property: after a
    # successful call the file is always valid JSON.
    target = tmp_path / "state.json"
    atomic_write_json(target, {"n": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"n": 1}
    atomic_write_json(target, {"n": 2, "nested": {"x": "y"}})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "n": 2,
        "nested": {"x": "y"},
    }
    # No leftover temp files in the directory.
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


# ---------------------------------------------------------------------------
# Misc: atomic_write_json dir must exist
# ---------------------------------------------------------------------------


def test_atomic_write_json_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "nested" / "state.json"
    atomic_write_json(target, {"ok": True})
    assert target.exists()


# ---------------------------------------------------------------------------
# Sanity: is_invalid_combo boundary fused_top_k == 15
# ---------------------------------------------------------------------------


def test_is_invalid_combo_fused_19_below_threshold() -> None:
    # (15,15,19,?) is NOT filtered by condition (4); only fused>=20 triggers.
    p = _sample_params(
        lexical_top_k=15,
        embedding_top_k=15,
        fused_top_k=19,
        rerank_top_k=12,
    )
    assert is_invalid_combo(p) is False


# ---------------------------------------------------------------------------
# Ensure tempfile clean-up survives a filesystem where dir is read-only.
# (Guard: we only check non-hostile case; hostile case tested via exception.)
# ---------------------------------------------------------------------------


def test_atomic_write_json_fsync_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def _fake_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fake_fsync)
    atomic_write_json(tmp_path / "x.json", {"ok": True})
    assert calls, "os.fsync must be invoked before rename"


# ---------------------------------------------------------------------------
# Misc: atomic_write_json with pathlib.Path and str both accepted.
# ---------------------------------------------------------------------------


def test_atomic_write_json_accepts_str_path(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    atomic_write_json(str(target), {"ok": True})  # type: ignore[arg-type]
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


# ---------------------------------------------------------------------------
# Regression: ConfigParams is hashable (used as dict key / set).
# ---------------------------------------------------------------------------


def test_config_params_is_hashable() -> None:
    s = {_sample_params(), _sample_params(lexical_top_k=25)}
    assert len(s) == 2


# ---------------------------------------------------------------------------
# Smoke: Phase 2 without any seeds returns empty grid.
# ---------------------------------------------------------------------------


def test_phase2_empty_seeds_returns_empty() -> None:
    assert generate_phase2_grid([]) == []


# ---------------------------------------------------------------------------
# Smoke: tempfile is placed in target dir, not /tmp.
# ---------------------------------------------------------------------------


def test_atomic_write_json_tmp_in_same_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_paths: list[str] = []

    real_ntf = tempfile.NamedTemporaryFile

    def _fake_ntf(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        f = real_ntf(*args, **kwargs)
        tmp_paths.append(f.name)
        return f

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _fake_ntf)

    target = tmp_path / "x.json"
    atomic_write_json(target, {"ok": True})
    assert tmp_paths, "atomic_write_json must use NamedTemporaryFile"
    assert Path(tmp_paths[0]).parent == target.parent
