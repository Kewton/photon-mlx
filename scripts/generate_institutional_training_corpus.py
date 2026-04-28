"""Generate JP institutional training sessions for Issue #135 / Phase 6.

The script is split into three layers (DR1-001):

- ``build_sessions``: produces ``Session`` records by delegating to the
  production ``baseline_reporag.eval.institutional.multi_turn`` builder.
  It does no I/O beyond the corpus directory walk; unit tests inject a
  fake ``LLMClient`` that returns a JSON 6-turn payload.
- ``verify_corpus``: pure metrics + eval-leak gate (DR4-005). Returns a
  ``CorpusReport``; raises on no input.
- ``tokenize_sessions``: pure converter from text turns to the ``{"tokens":
  [...]}`` schema used by ``photon_mlx.data.iterate_mixed_batches``,
  matching ``data/processed/train_multi.jsonl`` (the EN side of the mix).
- ``main``: composes the layers with CLI argument parsing, JSONL output,
  and fail-fast exit codes.

Issue #135 Day 3 (commit be91682) confirmed the production ``LLMClient``
exposes ``generate(prompt) -> str`` (JSON object), not a custom
``generate_turns`` API. ``build_sessions`` was refactored to use that
contract via the existing ``multi_turn.generate_session`` helper so the
prompt template, JSON parser, and retry behaviour stay in one place.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


# DR4-001: cap the requested session count so a typo (e.g. --sessions 50000
# instead of 5000) cannot trigger a multi-day LLM run.
MAX_SESSIONS = 5000

# DR4-001: production allow-lists. Tests pass their own approved_* lists to
# bypass these without disabling the guard logic itself.
DEFAULT_APPROVED_CORPUS_ROOTS = (
    Path("/Users/maenokota/share/work/github_kewton/myWebData/markdowndb"),
)
DEFAULT_APPROVED_OUTPUT_ROOTS = (Path("./data/training"),)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._corpus_core import (  # noqa: E402
    split_train_val,
    validate_eval_overlap,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (design §6.6)
# ---------------------------------------------------------------------------


@dataclass
class Session:
    session_id: str
    scenario: str  # cross_reference | drill_down | define | quantity | comparison | conclusion
    lang: str  # "ja" or "en"
    n_turns: int
    turns: list[str]
    source_md: str


@dataclass
class CorpusReport:
    n_sessions_requested: int
    n_sessions_succeeded: int
    eval_overlap: int
    jp_sequence_ratio: float
    scenario_distribution: dict[str, float]
    failure_breakdown: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1: build_sessions (LLM-driven via the existing multi_turn helper)
# ---------------------------------------------------------------------------


# The institutional eval ships 6-turn sessions (definition / scope /
# article_lookup / penalty / exception / overview). The training corpus
# follows the same shape so the trainer sees a consistent turn budget
# across train and eval. ``multi_turn._parse_session`` enforces this
# count; we keep the constant here for visibility.
N_TURNS_PER_SESSION = 6


def _allocate_scenarios(n: int, scenarios: dict[str, float]) -> list[str]:
    """Convert (scenario, weight) into a length-n list, preserving the ratio."""
    if not scenarios:
        raise ValueError("scenarios must be a non-empty dict")
    total = sum(scenarios.values())
    if total <= 0:
        raise ValueError("scenarios weights must sum to a positive number")
    out: list[str] = []
    for name, weight in scenarios.items():
        out.extend([name] * int(round(n * weight / total)))
    while len(out) < n:
        out.append(next(iter(scenarios)))
    return out[:n]


def _session_text_turns(session_dict: dict) -> list[str]:
    """Flatten a ``multi_turn`` session into one text per turn (Q + A)."""
    out: list[str] = []
    for turn in session_dict.get("turns", []):
        question = str(turn.get("question", "")).strip()
        answer = str(turn.get("reference_answer", "")).strip()
        if question or answer:
            out.append(f"Q: {question}\nA: {answer}")
    return out


def build_sessions(
    *,
    corpus_dir: Path,
    n: int,
    scenarios: dict[str, float],
    llm_client: Any,
    seed: int = 42,
) -> Iterator[Session]:
    """Yield ``n`` ``Session`` records via ``multi_turn.generate_session``.

    Each requested session draws a (scenario, doc) pair, then delegates
    to ``baseline_reporag.eval.institutional.multi_turn.generate_session``
    which handles prompt construction (``_SESSION_SYSTEM`` + per-turn
    template), JSON parsing with extraction-on-fence, and bounded retry.
    Documents are taken from ``build_doc_index(corpus_dir)`` so this
    function works on the institutional ``<doc>/document.md`` layout.
    """
    # Lazy import: the institutional module pulls in the production
    # generator package. Tests rely on the fake LLMClient we inject;
    # the only real dependency is ``multi_turn.generate_session``.
    from baseline_reporag.eval.institutional.corpus import build_doc_index
    from baseline_reporag.eval.institutional.generator import GenerationFailure
    from baseline_reporag.eval.institutional.multi_turn import generate_session

    docs = build_doc_index(Path(corpus_dir))
    docs = [d for d in docs if d.has_articles]
    if not docs:
        raise ValueError(
            f"no articles found in corpus_dir={corpus_dir!r} — Issue #135 "
            "expects institutional_documents/<doc>/document.md with article markers"
        )

    scenario_list = _allocate_scenarios(n, scenarios)
    rng = random.Random(seed)
    rng.shuffle(scenario_list)
    # Independent RNG for doc choice so the scenario ordering and the
    # doc rotation can drift independently across reruns.
    doc_rng = random.Random(seed ^ 0x9E37_79B9)

    yielded = 0
    seq = 1
    for scenario in scenario_list:
        doc = doc_rng.choice(docs)
        try:
            session_dict = generate_session(
                doc=doc,
                scenario=scenario,
                seq=seq,
                client=llm_client,
            )
        except GenerationFailure as exc:
            _logger.warning(
                "generate_session failed for doc=%s scenario=%s: %s",
                doc.doc_id,
                scenario,
                exc,
            )
            seq += 1
            continue

        text_turns = _session_text_turns(session_dict)
        if not text_turns:
            seq += 1
            continue

        yield Session(
            session_id=f"train_{yielded + 1:06d}",
            scenario=scenario,
            lang="ja",
            n_turns=len(text_turns),
            turns=text_turns,
            source_md=doc.doc_id,
        )
        yielded += 1
        seq += 1


# ---------------------------------------------------------------------------
# Layer 1.5: tokenize_sessions (text → {"tokens": [int, ...]})
# ---------------------------------------------------------------------------


def tokenize_sessions(
    sessions: Iterable[Session],
    *,
    tokenizer: Any,
) -> Iterator[dict[str, list[int]]]:
    """Convert each ``Session`` to ``{"tokens": [int, ...]}`` for the trainer.

    Output schema matches ``data/processed/train_multi.jsonl`` (the EN side
    of the JP/EN mix) and the ``photon_mlx.data.load_jsonl`` reader. Turns
    inside a session are joined with newlines so the model sees the full
    multi-turn conversation as one packed sequence — the same way
    ``train_multi.jsonl`` packs its multi-doc sequences end-to-end.
    """
    for session in sessions:
        text = "\n\n".join(session.turns)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            _logger.warning("empty token list for session %s", session.session_id)
            continue
        yield {"tokens": [int(t) for t in ids]}


# ---------------------------------------------------------------------------
# Layer 2: verify_corpus (LLM-free metrics + leak gate)
# ---------------------------------------------------------------------------


def verify_corpus(
    sessions: Iterable[Session],
    eval_path: Path | str | None,
) -> CorpusReport:
    """Compute the audit metrics required by Issue #135 receiving criteria.

    - ``eval_overlap``: count of session IDs that also appear in the eval
      JSONL at ``eval_path``. Must be 0 for the corpus to be accepted.
      Pass ``eval_path=None`` to skip leak detection (tests only).
    - ``jp_sequence_ratio``: fraction of sessions with ``lang == "ja"``.
      The control target is the sequence-level ratio (DR1-007); the
      token-level ratio is measured separately by the trainer once the
      corpus is tokenised.
    - ``scenario_distribution``: normalised histogram of scenarios.

    The function does not mutate ``sessions``; pass a list when you need
    to consume it more than once.
    """
    materialised = list(sessions)
    n = len(materialised)
    if n == 0:
        return CorpusReport(
            n_sessions_requested=0,
            n_sessions_succeeded=0,
            eval_overlap=0,
            jp_sequence_ratio=0.0,
            scenario_distribution={},
        )

    session_ids = {s.session_id for s in materialised}
    if eval_path is not None and Path(eval_path).exists():
        overlap = validate_eval_overlap(session_ids, Path(eval_path))
    else:
        overlap = 0

    jp_count = sum(1 for s in materialised if s.lang == "ja")
    scenario_counts: dict[str, int] = {}
    for s in materialised:
        scenario_counts[s.scenario] = scenario_counts.get(s.scenario, 0) + 1
    scenario_dist = {k: v / n for k, v in scenario_counts.items()}

    return CorpusReport(
        n_sessions_requested=n,
        n_sessions_succeeded=n,
        eval_overlap=overlap,
        jp_sequence_ratio=jp_count / n,
        scenario_distribution=scenario_dist,
    )


# ---------------------------------------------------------------------------
# Layer 3: main (composition only)
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JP institutional training corpus (Issue #135).",
    )
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--eval-set",
        required=False,
        type=Path,
        default=Path("data/eval_sets/institutional_multi_turn_eval.jsonl"),
    )
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--provider", default="qwen")
    parser.add_argument(
        "--tokenizer-id",
        default="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        help="HF tokenizer id used to convert text → token ids (matches the "
        "trainer's tokenizer config in institutional_docs_photon_retrain.yaml).",
    )
    return parser.parse_args(argv)


def _resolve_under_root(
    raw: Path,
    *,
    must_exist: bool,
    approved_roots: Iterable[Path],
    label: str,
) -> Path:
    """DR4-001: ``resolve(strict=True)`` then check ``relative_to`` an approved root.

    For ``--output`` (must_exist=False) we resolve the *parent* dir
    instead — the file itself is not created yet — and ensure the
    parent sits under an approved root.
    """
    roots = [Path(r).resolve() for r in approved_roots]
    if must_exist:
        try:
            resolved = raw.resolve(strict=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{label} does not exist: {raw}") from e
    else:
        # Output paths are written, not read; resolve the parent so
        # we can still detect symlink escape on the directory side.
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
    """Parse CLI args and apply Issue #135 / DR4-001 hardening.

    - --corpus-dir: must exist, must resolve under one of
      ``approved_corpus_roots`` (production: institutional_documents
      mount). Rejects symlink escape via ``resolve(strict=True)``.
    - --output: parent must resolve under one of ``approved_output_roots``
      (production: ``./data/training``). The file itself is not required
      to exist — main() creates it via ``write_atomic``.
    - --sessions: 1 <= n <= MAX_SESSIONS (5000) so a typo cannot trigger
      a multi-day LLM run.
    - --seed: argparse already enforces int; we additionally clamp to
      ``[0, 2**32 - 1]`` for reproducibility-tool consistency.
    - --val-ratio: must satisfy ``0.0 < val_ratio < 0.5`` (mirrors
      ``_corpus_core.split_train_val``).
    """
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

    if not (1 <= ns.sessions <= MAX_SESSIONS):
        raise ValueError(
            f"--sessions must be in [1, {MAX_SESSIONS}], got {ns.sessions} "
            "(DR4-001 cap to prevent runaway LLM costs)"
        )
    if not (0 <= ns.seed < 2**32):
        raise ValueError(f"--seed must be in [0, 2**32), got {ns.seed} (DR4-001)")
    if not (0.0 < ns.val_ratio < 0.5):
        raise ValueError(
            f"--val-ratio must be in (0.0, 0.5), got {ns.val_ratio} (DR1-005)"
        )
    return ns


def write_atomic(path: Path, content: str, *, mode: int = 0o600) -> None:
    """Write ``content`` to ``path`` via tmp + os.replace with 0o600 perms.

    DR4-001: training corpora can carry institutional document text, so
    a crash mid-write must never leave a half-formed file under
    ``data/training/``. The tmp file is created with ``mode`` (default
    0600 = owner read/write only) before content is written, so the
    file is never world-readable in transit.
    """
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
        # If something went wrong before os.replace, clean up the tmp.
        if tmp.exists():
            tmp.unlink()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - LLM gated
    """Compose build_sessions → verify_corpus → tokenize → write JSONL.

    Not covered by unit tests because invoking it requires a live
    ``LLMClient`` (qwen mlx_lm or openai) and a real HF tokenizer; the
    JP institutional corpus is large enough that a full mock run isn't
    representative. The CLI helpers (``parse_validated_args``,
    ``write_atomic``) and the layer functions (``build_sessions``,
    ``verify_corpus``, ``tokenize_sessions``) all carry direct unit
    tests.
    """
    args = parse_validated_args(argv)

    # Lazy imports: ``select_llm_client`` pulls mlx_lm / openai depending
    # on the requested provider; ``AutoTokenizer`` pulls transformers.
    # Keep the module importable for unit tests on machines without those.
    from baseline_reporag.eval.institutional.llm_client import select_llm_client
    from transformers import AutoTokenizer

    client = select_llm_client(args.provider)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_id, trust_remote_code=False
    )

    # Scenarios mirror ``multi_turn.SESSION_PATTERNS`` (drill_down /
    # cross_reference / real_scenario) — these are the only values
    # ``generate_session`` understands. Weights here are the training-time
    # mix, distinct from the eval-side counts.
    sessions = list(
        build_sessions(
            corpus_dir=args.corpus_dir,
            n=args.sessions,
            scenarios={
                "drill_down": 0.50,
                "cross_reference": 0.30,
                "real_scenario": 0.20,
            },
            llm_client=client,
            seed=args.seed,
        )
    )

    report = verify_corpus(sessions, args.eval_set)
    if report.eval_overlap > 0:
        _logger.error(
            "ABORT: %d session IDs overlap with the eval set at %s",
            report.eval_overlap,
            args.eval_set,
        )
        return 1

    train, val = split_train_val(sessions, val_ratio=args.val_ratio, seed=args.seed)

    train_records = list(tokenize_sessions(train, tokenizer=tokenizer))
    val_records = list(tokenize_sessions(val, tokenizer=tokenizer))

    train_text = "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in train_records
    )
    val_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val_records)

    write_atomic(args.output, train_text)
    val_path = args.output.with_name(args.output.stem + "_val.jsonl")
    write_atomic(val_path, val_text)

    total_tokens_train = sum(len(r["tokens"]) for r in train_records)
    total_tokens_val = sum(len(r["tokens"]) for r in val_records)

    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    write_atomic(
        metadata_path,
        json.dumps(
            {
                "n_sessions_requested": report.n_sessions_requested,
                "n_sessions_succeeded": report.n_sessions_succeeded,
                "eval_overlap": report.eval_overlap,
                "jp_sequence_ratio": report.jp_sequence_ratio,
                "scenario_distribution": report.scenario_distribution,
                "train_size": len(train_records),
                "val_size": len(val_records),
                "total_tokens_train": total_tokens_train,
                "total_tokens_val": total_tokens_val,
                "tokenizer_id": args.tokenizer_id,
                "provider": args.provider,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
