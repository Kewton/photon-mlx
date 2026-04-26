"""Tests for the Issue #135 / Phase 2 corpus generation scripts.

Two scripts are under test:

- ``scripts/_corpus_core.py``: pure helpers (eval-overlap detection, ratio
  measurement, train/val split). No LLM calls — testable directly.
- ``scripts/generate_institutional_training_corpus.py``: thin CLI that
  composes ``_corpus_core`` with an injected ``LLMClient``. We test
  ``build_sessions`` / ``verify_corpus`` against a fake client so we
  never hit a real model.

These tests deliberately avoid running the script's ``main()`` end-to-end
to keep the LLM call gate at the user's discretion (Issue #135 says
the actual generation run is blocked until #137 is done).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# scripts/_corpus_core.py: pure helpers
# ---------------------------------------------------------------------------


class TestValidateEvalOverlap:
    """DR4-005: training corpus must not overlap with the eval set."""

    def test_zero_overlap_returns_zero(self, tmp_path: Path) -> None:
        from scripts._corpus_core import validate_eval_overlap

        eval_path = tmp_path / "eval.jsonl"
        eval_path.write_text(
            json.dumps({"session_id": "eval_0001"})
            + "\n"
            + json.dumps({"session_id": "eval_0002"})
            + "\n",
            encoding="utf-8",
        )
        n = validate_eval_overlap({"train_0001", "train_0002"}, eval_path)
        assert n == 0

    def test_overlap_is_counted(self, tmp_path: Path) -> None:
        from scripts._corpus_core import validate_eval_overlap

        eval_path = tmp_path / "eval.jsonl"
        eval_path.write_text(
            json.dumps({"session_id": "shared_0001"})
            + "\n"
            + json.dumps({"session_id": "eval_0001"})
            + "\n",
            encoding="utf-8",
        )
        n = validate_eval_overlap({"shared_0001", "train_0002"}, eval_path)
        assert n == 1

    def test_missing_eval_file_raises(self, tmp_path: Path) -> None:
        from scripts._corpus_core import validate_eval_overlap

        with __import__("pytest").raises(FileNotFoundError):
            validate_eval_overlap({"x"}, tmp_path / "missing.jsonl")


class TestSplitTrainVal:
    """DR1-005: held-out fraction matches the requested ratio (deterministic)."""

    def test_split_size_matches_ratio(self) -> None:
        from scripts._corpus_core import split_train_val

        items = list(range(100))
        train, val = split_train_val(items, val_ratio=0.1, seed=42)
        assert len(train) == 90
        assert len(val) == 10
        # No overlap.
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(items)

    def test_split_is_deterministic(self) -> None:
        from scripts._corpus_core import split_train_val

        items = list(range(100))
        a = split_train_val(items, val_ratio=0.2, seed=999)
        b = split_train_val(items, val_ratio=0.2, seed=999)
        assert a == b

    def test_invalid_val_ratio_raises(self) -> None:
        from scripts._corpus_core import split_train_val

        for bad in (-0.1, 0.0, 0.5, 1.0):
            try:
                split_train_val([1, 2, 3], val_ratio=bad, seed=0)
            except ValueError:
                continue
            raise AssertionError(f"val_ratio={bad} must raise ValueError")


# ---------------------------------------------------------------------------
# scripts/generate_institutional_training_corpus.py: composed pipeline
# ---------------------------------------------------------------------------


class _FakeLLMClient:
    """Minimal LLMClient stand-in. Returns canned turn texts so build_sessions
    can be exercised without hitting a real model."""

    def __init__(self, turn_texts: list[str]) -> None:
        self.turn_texts = turn_texts
        self.calls: list[dict] = []

    def generate_turns(self, *, source_md: str, scenario: str, n_turns: int, lang: str):
        self.calls.append(
            {
                "source_md": source_md,
                "scenario": scenario,
                "n_turns": n_turns,
                "lang": lang,
            }
        )
        return list(self.turn_texts[:n_turns])


def _make_md(tmp_path: Path, name: str, content: str = "本文") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class TestBuildSessions:
    def test_yields_one_session_per_request(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import build_sessions

        # Two source markdown files in the corpus dir.
        _make_md(tmp_path, "doc_a.md")
        _make_md(tmp_path, "doc_b.md")

        client = _FakeLLMClient(["A?", "B!", "C?"])
        sessions = list(
            build_sessions(
                corpus_dir=tmp_path,
                n=3,
                scenarios={"cross_reference": 1.0},
                llm_client=client,
                seed=42,
                lang="ja",
                n_turns=3,
            )
        )

        assert len(sessions) == 3
        assert all(s.scenario == "cross_reference" for s in sessions)
        assert all(s.lang == "ja" for s in sessions)
        assert all(s.n_turns == 3 for s in sessions)
        # All session IDs unique (DR4-005 prerequisite).
        assert len({s.session_id for s in sessions}) == 3

    def test_scenario_distribution_matches_weights(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import build_sessions

        for i in range(5):
            _make_md(tmp_path, f"doc_{i}.md")

        client = _FakeLLMClient(["q1", "q2"])
        sessions = list(
            build_sessions(
                corpus_dir=tmp_path,
                n=100,
                scenarios={"cross_reference": 0.5, "drill_down": 0.5},
                llm_client=client,
                seed=42,
                lang="ja",
                n_turns=2,
            )
        )

        cross = sum(1 for s in sessions if s.scenario == "cross_reference")
        drill = sum(1 for s in sessions if s.scenario == "drill_down")
        # Allow ±10pp slack for round-off / int truncation in 100-sample mix.
        assert 40 <= cross <= 60
        assert 40 <= drill <= 60
        assert cross + drill == 100

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import build_sessions

        client = _FakeLLMClient(["a", "b"])
        try:
            list(
                build_sessions(
                    corpus_dir=tmp_path,
                    n=3,
                    scenarios={"cross_reference": 1.0},
                    llm_client=client,
                    seed=42,
                    lang="ja",
                    n_turns=2,
                )
            )
        except ValueError as e:
            assert "no markdown" in str(e).lower() or "empty" in str(e).lower()
            return
        raise AssertionError("empty corpus dir must raise ValueError")


class TestVerifyCorpus:
    """verify_corpus is the LLM-free metrics+leak gate."""

    def _ja_session(self, sid: str, n_turns: int = 3):
        from scripts.generate_institutional_training_corpus import Session

        return Session(
            session_id=sid,
            scenario="cross_reference",
            lang="ja",
            n_turns=n_turns,
            turns=["これは日本語の質問です。"] * n_turns,
            source_md="x.md",
        )

    def _en_session(self, sid: str, n_turns: int = 3):
        from scripts.generate_institutional_training_corpus import Session

        return Session(
            session_id=sid,
            scenario="cross_reference",
            lang="en",
            n_turns=n_turns,
            turns=["This is an English question."] * n_turns,
            source_md="y.md",
        )

    def test_clean_corpus_zero_overlap(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import verify_corpus

        eval_path = tmp_path / "eval.jsonl"
        eval_path.write_text(
            json.dumps({"session_id": "eval_0001"}) + "\n",
            encoding="utf-8",
        )
        sessions = [self._ja_session(f"train_{i}") for i in range(10)]
        report = verify_corpus(sessions, eval_path)
        assert report.eval_overlap == 0
        assert report.n_sessions_succeeded == 10

    def test_overlap_detected(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import verify_corpus

        eval_path = tmp_path / "eval.jsonl"
        eval_path.write_text(
            json.dumps({"session_id": "leaked"}) + "\n",
            encoding="utf-8",
        )
        sessions = [self._ja_session("leaked"), self._ja_session("ok")]
        report = verify_corpus(sessions, eval_path)
        assert report.eval_overlap == 1

    def test_jp_sequence_ratio_measured(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import verify_corpus

        eval_path = tmp_path / "eval.jsonl"
        eval_path.write_text("", encoding="utf-8")

        sessions = [self._ja_session(f"j_{i}") for i in range(7)] + [
            self._en_session(f"e_{i}") for i in range(3)
        ]
        report = verify_corpus(sessions, eval_path)
        assert abs(report.jp_sequence_ratio - 0.7) < 1e-9
        assert report.n_sessions_succeeded == 10
