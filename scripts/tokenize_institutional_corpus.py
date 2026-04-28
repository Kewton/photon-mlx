"""Tokenize raw institutional documents into a training JSONL (Issue #135 / Day 4).

The Day 3 review confirmed the EN side (``mulmoclaude/train_multi.jsonl``)
is raw tokenized source text — *continued pretraining*, not synthesised
Q&A. To match that paradigm on the JP side we tokenize each
institutional document directly with the Qwen tokenizer; no LLM call
required, so the script runs in minutes rather than days.

Train / eval separation (Day 4 user requirement): any document whose
name appears as a ``source_document_id`` in the eval JSONL is excluded
so the retrained model is never measured against documents it was
trained on.

Output schema matches ``data/processed/train_multi.jsonl``:
``{"tokens": [int, ...]}`` per line. Lines longer than
``max_tokens_per_line`` (default ``context_length × 4 = 8192`` per
DR4-002) are chunked into multiple lines so
``photon_mlx.data.iterate_mixed_batches`` accepts them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._corpus_core import split_train_val  # noqa: E402

# DR4-002: per-line token length cap (must match the validator in
# ``photon_mlx.data._load_validated_jsonl``: context_length × 4).
DEFAULT_MAX_TOKENS_PER_LINE = 8192

# DR4-001: production allow-lists (mirrored from
# generate_institutional_training_corpus.py for consistency). Tests
# pass their own approved_* lists to bypass without disabling logic.
DEFAULT_APPROVED_CORPUS_ROOTS = (
    Path("/Users/maenokota/share/work/github_kewton/myWebData/markdowndb"),
)
DEFAULT_APPROVED_OUTPUT_ROOTS = (Path("./data/training"),)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: load_eval_doc_ids (no LLM, pure I/O)
# ---------------------------------------------------------------------------


def load_eval_doc_ids(eval_path: Path) -> set[str]:
    """Return the set of ``source_document_id`` values in the eval JSONL.

    Day 4 train/eval separation guard. Caller passes the result to
    ``tokenize_documents(excluded_doc_ids=...)`` so any doc that
    appears in eval is dropped from the training corpus.
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"eval set not found: {eval_path}")
    ids: set[str] = set()
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        sid = obj.get("source_document_id")
        if sid:
            ids.add(sid)
    return ids


# ---------------------------------------------------------------------------
# Layer 2: tokenize_documents (LLM-free, pure transform)
# ---------------------------------------------------------------------------


def tokenize_documents(
    *,
    corpus_dir: Path,
    tokenizer: Any,
    excluded_doc_ids: set[str],
    max_tokens_per_line: int = DEFAULT_MAX_TOKENS_PER_LINE,
) -> Iterator[dict[str, list[int]]]:
    """Walk ``corpus_dir`` and yield ``{"tokens": [...]}`` per chunk.

    For each ``<doc_dir>/document.md`` whose ``doc_dir.name`` is *not*
    in ``excluded_doc_ids``: read the markdown raw, tokenize via
    ``tokenizer.encode``, and split into chunks of at most
    ``max_tokens_per_line`` ids. Empty docs are skipped.
    """
    from baseline_reporag.eval.institutional.corpus import build_doc_index

    docs = build_doc_index(Path(corpus_dir))
    for doc in docs:
        if doc.doc_id in excluded_doc_ids:
            continue
        try:
            text = doc.path.read_text(encoding="utf-8")
        except OSError:
            _logger.warning("could not read %s — skipping", doc.path)
            continue
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        for start in range(0, len(tokens), max_tokens_per_line):
            chunk = tokens[start : start + max_tokens_per_line]
            if chunk:
                yield {"tokens": [int(t) for t in chunk]}


# ---------------------------------------------------------------------------
# Layer 3: CLI entry (composition)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize raw institutional documents into JSONL "
        "(Issue #135 / continued pretraining path)."
    )
    p.add_argument("--corpus-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--eval-set",
        type=Path,
        default=Path("data/eval_sets/institutional_multi_turn_eval.jsonl"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument(
        "--tokenizer-id",
        default="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        help="HF tokenizer id; must match the trainer's tokenizer config "
        "(institutional_docs_photon_retrain.yaml: vocab_size=152064).",
    )
    p.add_argument(
        "--max-tokens-per-line",
        type=int,
        default=DEFAULT_MAX_TOKENS_PER_LINE,
        help="Per-line token cap (DR4-002, default context_length × 4 = 8192).",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Cap document count for smoke runs. 0 means no cap.",
    )
    return p.parse_args(argv)


def _resolve_under_root(
    raw: Path,
    *,
    must_exist: bool,
    approved_roots: Iterable[Path],
    label: str,
) -> Path:
    roots = [Path(r).resolve() for r in approved_roots]
    if must_exist:
        try:
            resolved = raw.resolve(strict=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{label} does not exist: {raw}") from e
    else:
        parent = raw.parent.resolve(strict=False)
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        resolved = parent / raw.name
    for root in roots:
        try:
            (resolved if must_exist else resolved.parent).relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"{label} is outside approved roots {[str(r) for r in roots]}: {resolved} "
        "(DR4-001)"
    )


def parse_validated_args(
    argv: list[str] | None,
    *,
    approved_corpus_roots: Iterable[Path] | None = None,
    approved_output_roots: Iterable[Path] | None = None,
) -> argparse.Namespace:
    """Parse + DR4-001 hardening: resolve(strict=True) under approved roots,
    bound val-ratio, sanitise paths."""
    ns = _parse_args(argv)
    corpus_roots = (
        list(approved_corpus_roots)
        if approved_corpus_roots is not None
        else list(DEFAULT_APPROVED_CORPUS_ROOTS)
    )
    output_roots = (
        list(approved_output_roots)
        if approved_output_roots is not None
        else list(DEFAULT_APPROVED_OUTPUT_ROOTS)
    )
    ns.corpus_dir = _resolve_under_root(
        ns.corpus_dir,
        must_exist=True,
        approved_roots=corpus_roots,
        label="--corpus-dir",
    )
    ns.output = _resolve_under_root(
        ns.output,
        must_exist=False,
        approved_roots=output_roots,
        label="--output",
    )
    if not (0.0 < ns.val_ratio < 0.5):
        raise ValueError(
            f"--val-ratio must be in (0.0, 0.5), got {ns.val_ratio} (DR1-005)"
        )
    if ns.max_docs < 0:
        raise ValueError(f"--max-docs must be >= 0, got {ns.max_docs}")
    return ns


def write_atomic(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Tmp + os.replace with 0o600 perms (DR4-001 / matches the QA-path
    helper in generate_institutional_training_corpus.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(tmp), flags, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - I/O heavy
    args = parse_validated_args(argv)

    # Lazy import: keep this script importable for unit tests on machines
    # without transformers installed.
    from transformers import AutoTokenizer

    excluded = load_eval_doc_ids(args.eval_set)
    _logger.info("excluding %d eval source docs from training", len(excluded))

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id, trust_remote_code=False
    )

    records = []
    for rec in tokenize_documents(
        corpus_dir=args.corpus_dir,
        tokenizer=tokenizer,
        excluded_doc_ids=excluded,
        max_tokens_per_line=args.max_tokens_per_line,
    ):
        records.append(rec)
        if args.max_docs and len(records) >= args.max_docs:
            break

    if not records:
        _logger.error(
            "no documents tokenised; aborting (corpus_dir=%s)", args.corpus_dir
        )
        return 1

    train, val = split_train_val(records, val_ratio=args.val_ratio, seed=args.seed)

    train_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
    val_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val)
    write_atomic(args.output, train_text)
    val_path = args.output.with_name(args.output.stem + "_val.jsonl")
    write_atomic(val_path, val_text)

    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    write_atomic(
        metadata_path,
        json.dumps(
            {
                "n_records_total": len(records),
                "train_size": len(train),
                "val_size": len(val),
                "total_tokens_train": sum(len(r["tokens"]) for r in train),
                "total_tokens_val": sum(len(r["tokens"]) for r in val),
                "n_excluded_eval_docs": len(excluded),
                "tokenizer_id": args.tokenizer_id,
                "max_tokens_per_line": args.max_tokens_per_line,
                "seed": args.seed,
                "val_ratio": args.val_ratio,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
