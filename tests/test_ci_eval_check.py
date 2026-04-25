"""Regression tests for ci_eval_check (Issue #124).

Ensures repo_id-scoped glob does not pick up logs from other corpora
(e.g. institutional_documents) in the weekly fastapi_fastapi check path.

NOTE: The literal "fastapi_fastapi" must stay in sync with
.github/workflows/weekly_eval.yml glob and configs/baseline.yaml repo_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci_eval_check import _resolve_latest  # noqa: E402


def test_resolve_latest_excludes_other_repo_id_logs(tmp_path: Path) -> None:
    """fastapi_fastapi-scoped glob must NOT pick up institutional logs."""
    logs = tmp_path / "logs"
    logs.mkdir()
    # File content is irrelevant: _resolve_latest matches by name only.
    # institutional log lands under logs/ directly (run log behavior of
    # scripts/run_baseline_eval.py:111).
    (logs / "baseline_eval_institutional_documents_20260425_010000.jsonl").write_text(
        ""
    )
    # fastapi_fastapi log written 1s earlier — sorted latest would have
    # picked the institutional one under the unscoped glob.
    (logs / "baseline_eval_fastapi_fastapi_20260425_005959.jsonl").write_text("")

    pattern = str(logs / "baseline_eval_fastapi_fastapi_*.jsonl")
    resolved = _resolve_latest(pattern)

    assert resolved is not None
    assert "fastapi_fastapi" in resolved
    assert "institutional_documents" not in resolved


def test_resolve_latest_returns_lexicographic_max(tmp_path: Path) -> None:
    """Multiple fastapi_fastapi logs -> lexicographically latest wins."""
    logs = tmp_path / "logs"
    logs.mkdir()
    # Empty files OK — _resolve_latest only inspects names.
    (logs / "baseline_eval_fastapi_fastapi_20260101_000000.jsonl").write_text("")
    (logs / "baseline_eval_fastapi_fastapi_20260425_120000.jsonl").write_text("")
    (logs / "baseline_eval_fastapi_fastapi_20260301_060000.jsonl").write_text("")

    pattern = str(logs / "baseline_eval_fastapi_fastapi_*.jsonl")
    resolved = _resolve_latest(pattern)

    assert resolved is not None
    assert resolved.endswith("20260425_120000.jsonl")


def test_resolve_latest_returns_none_when_no_match(tmp_path: Path) -> None:
    """Empty glob result -> None (caller treats as failure)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    pattern = str(logs / "baseline_eval_fastapi_fastapi_*.jsonl")
    assert _resolve_latest(pattern) is None
