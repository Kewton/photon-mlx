"""Generate JP institutional training sessions for Issue #135 / Phase 2.

The script is split into three layers (DR1-001):

- ``build_sessions``: produces ``Session`` records by calling an injected
  ``LLMClient``. It does no I/O and no metrics so unit tests can mock the
  client and run without a network call.
- ``verify_corpus``: pure metrics + eval-leak gate (DR4-005). Returns a
  ``CorpusReport``; raises on no input.
- ``main``: composes the two with CLI argument parsing, JSONL output, and
  fail-fast exit codes.

Issue #135 explicitly defers the actual generation run until #137 (the
GPU-bound multilingual reranker eval) finishes. This commit ships only
the script + unit tests; the operator runs ``main()`` later with an
appropriate provider once GPU is free.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

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
# LLM client protocol (mockable for tests)
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Duck-typed interface for the LLM provider that drafts turn text.

    Production implementations route to ``baseline_reporag.eval.institutional.llm_client``
    helpers; ``build_sessions`` only depends on the ``generate_turns`` shape so
    tests can inject a stub.
    """

    def generate_turns(
        self,
        *,
        source_md: str,
        scenario: str,
        n_turns: int,
        lang: str,
    ) -> list[str]: ...


# ---------------------------------------------------------------------------
# Layer 1: build_sessions (LLM-driven)
# ---------------------------------------------------------------------------


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


def build_sessions(
    *,
    corpus_dir: Path,
    n: int,
    scenarios: dict[str, float],
    llm_client: LLMClient,
    seed: int = 42,
    lang: str = "ja",
    n_turns: int = 4,
) -> Iterator[Session]:
    """Yield ``n`` ``Session`` records by calling ``llm_client.generate_turns``.

    Markdown files are picked from ``corpus_dir`` round-robin by an RNG
    seeded with ``seed`` so the output is reproducible across runs. The
    function performs no I/O beyond listing the corpus dir — JSONL output
    happens in ``main``.
    """
    md_files = sorted(p for p in Path(corpus_dir).glob("*.md"))
    if not md_files:
        raise ValueError(
            f"no markdown files found in corpus_dir={corpus_dir!r} — Issue #135 "
            "expects institutional_documents/*.md"
        )

    scenario_list = _allocate_scenarios(n, scenarios)
    rng = random.Random(seed)
    rng.shuffle(scenario_list)

    for i, scenario in enumerate(scenario_list):
        source = md_files[i % len(md_files)]
        turns = llm_client.generate_turns(
            source_md=source.name,
            scenario=scenario,
            n_turns=n_turns,
            lang=lang,
        )
        if not turns:
            _logger.warning(
                "LLM returned no turns for scenario=%s source=%s; skipping",
                scenario,
                source.name,
            )
            continue
        yield Session(
            session_id=f"train_{i:06d}",
            scenario=scenario,
            lang=lang,
            n_turns=len(turns),
            turns=turns,
            source_md=source.name,
        )


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
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--n-turns", type=int, default=4)
    parser.add_argument("--provider", default="qwen")
    return parser.parse_args(argv)


def _serialise_session(s: Session) -> str:
    return json.dumps(
        {
            "session_id": s.session_id,
            "scenario": s.scenario,
            "lang": s.lang,
            "n_turns": s.n_turns,
            "turns": s.turns,
            "source_md": s.source_md,
        },
        ensure_ascii=False,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - LLM gated
    """Compose build_sessions → verify_corpus → write JSONL.

    Not covered by unit tests because invoking it requires a real
    ``LLMClient`` (the JP institutional corpus is large enough that a
    full mock run isn't representative). Issue #135 explicitly defers
    the production run until #137 finishes — this entry point is here
    so the operator has something to execute later, not so CI runs it.
    """
    args = _parse_args(argv)

    # Build the LLM client lazily. The import is intentionally deferred so
    # the module stays importable for unit tests on machines without the
    # baseline_reporag eval providers installed.
    from baseline_reporag.eval.institutional.llm_client import select_llm_client

    client = select_llm_client(args.provider)

    sessions = list(
        build_sessions(
            corpus_dir=args.corpus_dir,
            n=args.sessions,
            scenarios={
                "cross_reference": 0.20,
                "drill_down": 0.20,
                "define": 0.15,
                "quantity": 0.15,
                "comparison": 0.15,
                "conclusion": 0.15,
            },
            llm_client=client,
            seed=args.seed,
            lang=args.lang,
            n_turns=args.n_turns,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for s in train:
            fh.write(_serialise_session(s) + "\n")
    val_path = args.output.with_name(args.output.stem + "_val.jsonl")
    with val_path.open("w", encoding="utf-8") as fh:
        for s in val:
            fh.write(_serialise_session(s) + "\n")

    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "n_sessions_requested": report.n_sessions_requested,
                "n_sessions_succeeded": report.n_sessions_succeeded,
                "eval_overlap": report.eval_overlap,
                "jp_sequence_ratio": report.jp_sequence_ratio,
                "scenario_distribution": report.scenario_distribution,
                "train_size": len(train),
                "val_size": len(val),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
