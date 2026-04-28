"""Tests for ``scripts/tokenize_institutional_corpus.py`` (Issue #135 / Day 4).

The Day 3 review revealed that the existing EN training corpus
(``mulmoclaude/train_multi.jsonl``) is **raw tokenized source text**
trained for 600 steps, not synthesised Q&A pairs. To match that
paradigm on the JP side, we tokenize the institutional documents
directly without invoking an LLM. The script must:

- Walk ``corpus_dir`` and read each ``<doc>/document.md`` raw.
- Exclude any doc whose name appears as a ``source_document_id`` in
  the eval JSONL — Issue #135 Day 4 requires train/eval doc-level
  separation so the retrained model is never measured against
  documents it was trained on.
- Tokenize via an injected HF-shaped tokenizer (real run uses Qwen
  via ``transformers.AutoTokenizer``; tests use a fake).
- Chunk lines longer than ``max_tokens_per_line`` (default 8192 =
  context_length × 4, matching DR4-002) so the trainer's loader does
  not reject any line.
- Output ``{"tokens": [int, ...]}`` JSONL — schema-compatible with
  ``data/processed/train_multi.jsonl`` (the EN side of the mix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _FakeTokenizer:
    """One token id per character — deterministic and easy to inspect."""

    def encode(self, text, add_special_tokens=False):  # noqa: D401, ARG002
        return [ord(c) % 1024 for c in text]


def _make_doc_dir(root: Path, name: str, body: str | None = None) -> Path:
    """Mirror the ``build_doc_index`` ``<root>/<name>/document.md`` shape."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    text = body if body is not None else "第1条 適用範囲。罰則は別途。"
    (d / "document.md").write_text(text, encoding="utf-8")
    return d


def _write_eval_jsonl(path: Path, source_ids: list[str]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {"session_id": f"INST-MT-{i + 1:03d}", "source_document_id": sid}
            )
            for i, sid in enumerate(source_ids)
        )
        + "\n",
        encoding="utf-8",
    )


class TestLoadEvalDocIds:
    def test_returns_unique_source_document_ids(self, tmp_path: Path) -> None:
        from scripts.tokenize_institutional_corpus import load_eval_doc_ids

        eval_path = tmp_path / "eval.jsonl"
        _write_eval_jsonl(eval_path, ["doc_a", "doc_b", "doc_a"])  # dup ok
        ids = load_eval_doc_ids(eval_path)
        assert ids == {"doc_a", "doc_b"}

    def test_missing_eval_path_raises(self, tmp_path: Path) -> None:
        from scripts.tokenize_institutional_corpus import load_eval_doc_ids

        try:
            load_eval_doc_ids(tmp_path / "nope.jsonl")
        except FileNotFoundError:
            return
        raise AssertionError("missing eval set must raise")


class TestTokenizeDocuments:
    def test_excludes_eval_source_docs(self, tmp_path: Path) -> None:
        """Doc-level train/eval separation: any doc whose name appears in
        ``excluded_doc_ids`` must be skipped (Issue #135 Day 4)."""
        from scripts.tokenize_institutional_corpus import tokenize_documents

        _make_doc_dir(tmp_path, "doc_in_eval")
        _make_doc_dir(tmp_path, "doc_train_only")

        records = list(
            tokenize_documents(
                corpus_dir=tmp_path,
                tokenizer=_FakeTokenizer(),
                excluded_doc_ids={"doc_in_eval"},
            )
        )
        # Only the train-only doc survives.
        assert len(records) == 1

    def test_yields_tokens_only(self, tmp_path: Path) -> None:
        from scripts.tokenize_institutional_corpus import tokenize_documents

        _make_doc_dir(tmp_path, "doc_a", body="hello")
        records = list(
            tokenize_documents(
                corpus_dir=tmp_path,
                tokenizer=_FakeTokenizer(),
                excluded_doc_ids=set(),
            )
        )
        assert len(records) == 1
        rec = records[0]
        assert set(rec.keys()) == {"tokens"}
        assert all(isinstance(t, int) for t in rec["tokens"])
        # 5 chars in "hello" → 5 token ids.
        assert len(rec["tokens"]) == 5

    def test_chunks_long_documents(self, tmp_path: Path) -> None:
        """A single doc longer than ``max_tokens_per_line`` must be split into
        multiple JSONL lines so the trainer's DR4-002 length cap is not hit."""
        from scripts.tokenize_institutional_corpus import tokenize_documents

        # 25 chars → 25 tokens; chunk size 10 → 3 chunks (10 + 10 + 5).
        _make_doc_dir(tmp_path, "long_doc", body="a" * 25)
        records = list(
            tokenize_documents(
                corpus_dir=tmp_path,
                tokenizer=_FakeTokenizer(),
                excluded_doc_ids=set(),
                max_tokens_per_line=10,
            )
        )
        assert len(records) == 3
        assert sum(len(r["tokens"]) for r in records) == 25

    def test_skips_empty_documents(self, tmp_path: Path) -> None:
        from scripts.tokenize_institutional_corpus import tokenize_documents

        _make_doc_dir(tmp_path, "empty_doc", body="")
        _make_doc_dir(tmp_path, "ok_doc", body="text")
        records = list(
            tokenize_documents(
                corpus_dir=tmp_path,
                tokenizer=_FakeTokenizer(),
                excluded_doc_ids=set(),
            )
        )
        # Empty doc skipped, only ok_doc emitted.
        assert len(records) == 1


class TestSplitTrainVal:
    """Reuses ``scripts/_corpus_core.split_train_val`` but documents the
    expected behaviour for the raw-tokenize path: deterministic seeded
    split into (train, val) lists with no overlap."""

    def test_split_preserves_all_records(self) -> None:
        from scripts._corpus_core import split_train_val

        items = [{"tokens": [i]} for i in range(20)]
        train, val = split_train_val(items, val_ratio=0.1, seed=42)
        assert len(train) + len(val) == 20
        assert {tuple(r["tokens"]) for r in train + val} == {(i,) for i in range(20)}
