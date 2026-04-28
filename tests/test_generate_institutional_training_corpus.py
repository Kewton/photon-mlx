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
    """Production-shape ``LLMClient`` stub.

    Mirrors the signature of
    ``baseline_reporag.eval.institutional.llm_client.LLMClient``: a single
    ``generate(prompt) -> str`` returning a JSON object string, plus
    ``name`` / ``model`` attributes. Returns a fixed 6-turn JSON payload
    so ``build_sessions`` exercises the real ``multi_turn.generate_session``
    parser path without a live model.
    """

    name = "fake"
    model = "fake-model-v0"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        # Mirrors the schema enforced by multi_turn._parse_session: top-level
        # ``turns`` array of length 6, each turn carrying question /
        # reference_answer / etc. Distinct reference_chunk_ids per
        # assert_distinct_citations.
        return json.dumps(
            {
                "turns": [
                    {
                        "question": f"Q{i + 1}",
                        "reference_answer": f"A{i + 1}",
                        "expected_citation_patterns": [f"第{i + 1}条"],
                        "reference_chunk_ids": [f"c{i + 1}"],
                        "grading_notes": "n",
                        "tags": ["t"],
                    }
                    for i in range(6)
                ]
            },
            ensure_ascii=False,
        )


def _make_doc_dir(root: Path, name: str, body: str | None = None) -> Path:
    """Create a DocIndex-shaped directory: ``<root>/<name>/document.md``.

    ``build_doc_index`` walks immediate subdirectories of ``root`` looking
    for ``document.md``; this helper produces that exact layout. The body
    must contain a Japanese article marker so the doc is flagged with
    ``has_articles`` (``generate_multi_turn_set`` filters on this).
    """
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    text = body if body is not None else "第1条 適用範囲。\n第2条 例外但書。\n罰則。\n"
    (d / "document.md").write_text(text, encoding="utf-8")
    return d


class TestBuildSessions:
    def test_yields_one_session_per_request(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import build_sessions

        _make_doc_dir(tmp_path, "doc_a")
        _make_doc_dir(tmp_path, "doc_b")

        client = _FakeLLMClient()
        sessions = list(
            build_sessions(
                corpus_dir=tmp_path,
                n=3,
                scenarios={"cross_reference": 1.0},
                llm_client=client,
                seed=42,
            )
        )

        assert len(sessions) == 3
        assert all(s.scenario == "cross_reference" for s in sessions)
        assert all(s.lang == "ja" for s in sessions)
        # Each session must carry exactly 6 turns (institutional eval design).
        assert all(s.n_turns == 6 for s in sessions)
        # All session IDs unique (DR4-005 prerequisite).
        assert len({s.session_id for s in sessions}) == 3

    def test_scenario_distribution_matches_weights(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import build_sessions

        for i in range(5):
            _make_doc_dir(tmp_path, f"doc_{i}")

        client = _FakeLLMClient()
        sessions = list(
            build_sessions(
                corpus_dir=tmp_path,
                n=100,
                scenarios={"cross_reference": 0.5, "drill_down": 0.5},
                llm_client=client,
                seed=42,
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

        client = _FakeLLMClient()
        try:
            list(
                build_sessions(
                    corpus_dir=tmp_path,
                    n=3,
                    scenarios={"cross_reference": 1.0},
                    llm_client=client,
                    seed=42,
                )
            )
        except ValueError as e:
            assert "no" in str(e).lower() or "empty" in str(e).lower()
            return
        raise AssertionError("empty corpus dir must raise ValueError")

    def test_calls_real_protocol_generate(self, tmp_path: Path) -> None:
        """Production smoke: build_sessions must use the LLM client's
        ``generate(prompt)`` method, not ``generate_turns(...)``. Issue #135
        Day 3 caught the API mismatch — this regression test pins the fix."""
        from scripts.generate_institutional_training_corpus import build_sessions

        _make_doc_dir(tmp_path, "doc_a")
        client = _FakeLLMClient()
        list(
            build_sessions(
                corpus_dir=tmp_path,
                n=1,
                scenarios={"cross_reference": 1.0},
                llm_client=client,
                seed=42,
            )
        )
        assert len(client.calls) >= 1, "build_sessions must invoke client.generate()"
        # The prompt must contain the institutional system marker so we know
        # the right prompt builder was used (not a stub).
        assert any("制度文書" in p or "法令" in p for p in client.calls)


class TestTokenizeSessions:
    """Output schema must be ``{"tokens": [int, ...]}`` per session, matching
    the existing ``data/processed/train_multi.jsonl`` format so the trainer's
    ``iterate_mixed_batches`` consumes both EN and JP corpora identically."""

    class _FakeTokenizer:
        """Minimal HF tokenizer-shaped stub: ``encode(text) -> list[int]``."""

        def encode(self, text, add_special_tokens=False):  # noqa: D401
            # Map characters to deterministic small ids so tests can verify
            # without asserting on the exact tokenization scheme.
            return [hash(c) % 1024 for c in text]

    def test_session_tokenizes_to_int_list(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            Session,
            tokenize_sessions,
        )

        sessions = [
            Session(
                session_id="train_0001",
                scenario="cross_reference",
                lang="ja",
                n_turns=2,
                turns=["第1条 質問。", "答え。"],
                source_md="doc_a",
            )
        ]
        tokenized = list(tokenize_sessions(sessions, tokenizer=self._FakeTokenizer()))
        assert len(tokenized) == 1
        rec = tokenized[0]
        assert set(rec.keys()) == {"tokens"}
        assert isinstance(rec["tokens"], list)
        assert all(isinstance(t, int) for t in rec["tokens"])
        assert len(rec["tokens"]) > 0


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


# ---------------------------------------------------------------------------
# DR4-001: CLI argument hardening for generate_institutional_training_corpus
# ---------------------------------------------------------------------------


class TestCLIArgValidation:
    """``parse_validated_args`` rejects unsafe inputs before any LLM call.

    The contract is documented in design policy §6.6 (CLI / I/O security):
    --corpus-dir / --output must resolve under approved roots, --sessions
    must fit a sane bound, and --val-ratio / --seed must be inside their
    well-defined ranges. The validator never reads the LLM client, so we
    can pin its behaviour without GPU or network.
    """

    def _approved_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "approved_corpus"
        d.mkdir()
        # An approved corpus_dir needs at least one .md file so build_sessions
        # can later proceed; the validator itself doesn't open them, but the
        # existence check is part of resolve(strict=True).
        (d / "doc.md").write_text("body", encoding="utf-8")
        return d

    def test_corpus_dir_must_exist(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            parse_validated_args,
        )

        bogus = tmp_path / "nope"
        try:
            parse_validated_args(
                [
                    "--corpus-dir",
                    str(bogus),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--eval-set",
                    str(tmp_path / "eval.jsonl"),
                    "--sessions",
                    "100",
                ],
                approved_corpus_roots=[tmp_path],
                approved_output_roots=[tmp_path],
            )
        except (FileNotFoundError, ValueError):
            return
        raise AssertionError("missing --corpus-dir must raise")

    def test_corpus_dir_outside_approved_root_rejected(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            parse_validated_args,
        )

        approved = tmp_path / "approved"
        approved.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        try:
            parse_validated_args(
                [
                    "--corpus-dir",
                    str(outside),
                    "--output",
                    str(approved / "out.jsonl"),
                    "--eval-set",
                    str(tmp_path / "eval.jsonl"),
                    "--sessions",
                    "100",
                ],
                approved_corpus_roots=[approved],
                approved_output_roots=[approved],
            )
        except ValueError as e:
            assert "approved" in str(e).lower() or "root" in str(e).lower()
            return
        raise AssertionError("--corpus-dir outside approved roots must raise")

    def test_sessions_upper_bound(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            parse_validated_args,
        )

        approved = self._approved_dir(tmp_path)
        try:
            parse_validated_args(
                [
                    "--corpus-dir",
                    str(approved),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--eval-set",
                    str(tmp_path / "eval.jsonl"),
                    "--sessions",
                    "999999",
                ],
                approved_corpus_roots=[tmp_path],
                approved_output_roots=[tmp_path],
            )
        except ValueError as e:
            assert "sessions" in str(e).lower()
            return
        raise AssertionError("--sessions above 5000 must raise")

    def test_sessions_must_be_positive(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            parse_validated_args,
        )

        approved = self._approved_dir(tmp_path)
        try:
            parse_validated_args(
                [
                    "--corpus-dir",
                    str(approved),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--eval-set",
                    str(tmp_path / "eval.jsonl"),
                    "--sessions",
                    "0",
                ],
                approved_corpus_roots=[tmp_path],
                approved_output_roots=[tmp_path],
            )
        except ValueError:
            return
        raise AssertionError("--sessions=0 must raise")

    def test_val_ratio_bounds(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            parse_validated_args,
        )

        approved = self._approved_dir(tmp_path)
        for bad in ("0.0", "0.5", "0.9", "-0.1"):
            try:
                parse_validated_args(
                    [
                        "--corpus-dir",
                        str(approved),
                        "--output",
                        str(tmp_path / "out.jsonl"),
                        "--eval-set",
                        str(tmp_path / "eval.jsonl"),
                        "--sessions",
                        "100",
                        "--val-ratio",
                        bad,
                    ],
                    approved_corpus_roots=[tmp_path],
                    approved_output_roots=[tmp_path],
                )
            except ValueError:
                continue
            raise AssertionError(f"--val-ratio={bad} must raise")

    def test_valid_args_pass(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import (
            parse_validated_args,
        )

        approved = self._approved_dir(tmp_path)
        ns = parse_validated_args(
            [
                "--corpus-dir",
                str(approved),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--eval-set",
                str(tmp_path / "eval.jsonl"),
                "--sessions",
                "1000",
                "--val-ratio",
                "0.05",
                "--seed",
                "42",
            ],
            approved_corpus_roots=[tmp_path],
            approved_output_roots=[tmp_path],
        )
        # The validator returns the parsed args with paths already resolved.
        assert ns.sessions == 1000
        assert ns.val_ratio == 0.05
        assert ns.seed == 42
        assert ns.corpus_dir == approved.resolve()


class TestAtomicWrite:
    """Output JSONL writes go through ``write_atomic`` so a crash mid-write
    cannot leave a partial file in ``data/training/``."""

    def test_atomic_write_creates_file_with_restricted_permissions(
        self, tmp_path: Path
    ) -> None:
        from scripts.generate_institutional_training_corpus import write_atomic

        out = tmp_path / "out.jsonl"
        write_atomic(out, "line1\nline2\n")
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "line1\nline2\n"
        # 0o600 = owner read/write only (DR4-001).
        assert (out.stat().st_mode & 0o777) == 0o600

    def test_atomic_write_does_not_leave_tmp_on_success(self, tmp_path: Path) -> None:
        from scripts.generate_institutional_training_corpus import write_atomic

        out = tmp_path / "out.jsonl"
        write_atomic(out, "hello")
        # No .tmp file lingers next to the final output.
        assert list(tmp_path.glob("out.jsonl.tmp*")) == []
