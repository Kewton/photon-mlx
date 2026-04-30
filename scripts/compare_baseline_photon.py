"""compare_baseline_photon.py — 1 質問で baseline と PHOTON を並べて比較する。

A-1 (self-use) Phase 2: 試行錯誤中に "この質問で PHOTON が baseline より速いか?
citation を返すか?" を即座に確認するためのスクリプト。
bench/run_all.py は重量級 (datasets 経由) なので、軽量な 1 question 比較を提供する。

Usage:
    python scripts/compare_baseline_photon.py \\
        --repo-id fastapi_fastapi \\
        --question "認証処理の入口は？"

    # 別の config を指定:
    python scripts/compare_baseline_photon.py \\
        --baseline-config configs/baseline.yaml \\
        --photon-config configs/photon_small.yaml \\
        --repo-id fastapi_fastapi \\
        --question "..."

    # JSON 出力 (CI / 後続処理用):
    python scripts/compare_baseline_photon.py --question "..." --repo-id ... --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-export from baseline_reporag.comparison for backward compatibility.
# VariantResult and build_pipeline are imported into this module's namespace
# so that tests using importlib + monkeypatch.setattr(module, ...) continue to work.
from baseline_reporag.comparison import VariantResult, _build_and_run  # noqa: F401
from baseline_reporag.pipeline_factory import build_pipeline, override_repo_for_pipeline  # noqa: F401


def run_variant(
    variant_id: str,
    config_path: str,
    question: str,
    repo_id: str,
    session_id: str,
) -> VariantResult:
    """1 variant 分の pipeline を立てて 1 question を実行。

    Uses module-level build_pipeline so tests can monkeypatch it via
    monkeypatch.setattr(module, "build_pipeline", ...).
    """
    from baseline_reporag.config import load_config
    from baseline_reporag.comparison import run_variant_with_pipeline

    cfg = load_config(config_path)
    resolved_repo_id = repo_id or cfg.repo.repo_id
    override_repo_for_pipeline(cfg, resolved_repo_id)
    pipeline = build_pipeline(
        cfg
    )  # uses module-level build_pipeline (monkeypatch target)
    return run_variant_with_pipeline(
        pipeline, question, session_id, resolved_repo_id, variant_id, config_path
    )


def print_text_report(question: str, results: list[VariantResult]) -> None:
    """側並びの text レポートを出力。"""
    print("=" * 78)
    print(f"Question: {question}")
    print("=" * 78)

    for r in results:
        print(f"\n[{r.variant_id}]  config={r.config_path}")
        print(
            f"  latency: {r.latency_total_ms:.0f} ms"
            f"  (retrieval {r.latency_retrieval_ms:.0f}"
            f" | gen {r.latency_generation_ms:.0f})"
        )
        print(f"  memory:  {r.memory_peak_mb:.1f} MB")
        print(f"  cited:   {r.cited_chunk_ids}")
        if r.no_citation:
            print("  [WARNING] No citations")
        print(f"\n  Answer:\n{_indent(r.answer, prefix='  ')}")

    print("\n" + "=" * 78)
    print("Summary (latency):")
    for r in results:
        print(f"  {r.variant_id:>10s}: {r.latency_total_ms:>8.0f} ms")
    if len(results) >= 2:
        delta = results[1].latency_total_ms - results[0].latency_total_ms
        pct = (
            (delta / results[0].latency_total_ms * 100)
            if results[0].latency_total_ms
            else 0.0
        )
        print(
            f"  delta:      {delta:>+8.0f} ms"
            f"  ({pct:>+5.1f}% vs {results[0].variant_id})"
        )
    print("=" * 78)


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def to_json_payload(question: str, results: list[VariantResult]) -> dict[str, Any]:
    return {
        "question": question,
        "variants": [asdict(r) for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and PHOTON pipelines on a single question.",
    )
    parser.add_argument("--question", required=True, help="The question to ask.")
    parser.add_argument(
        "--repo-id",
        default="",
        help="Repo id to query against (falls back to cfg.repo.repo_id).",
    )
    parser.add_argument(
        "--baseline-config",
        default="configs/baseline.yaml",
        help="Path to the baseline config (default: configs/baseline.yaml).",
    )
    parser.add_argument(
        "--photon-config",
        default="configs/photon_small.yaml",
        help="Path to the PHOTON config (default: configs/photon_small.yaml).",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help=(
            "Session id to use. Defaults to a fresh per-variant id so memory "
            "state does not leak between baseline and PHOTON runs."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text report.",
    )
    args = parser.parse_args()

    baseline_session = args.session_id or "compare-baseline"
    photon_session = args.session_id or "compare-photon"

    baseline_result = run_variant(
        variant_id="baseline",
        config_path=args.baseline_config,
        question=args.question,
        repo_id=args.repo_id,
        session_id=baseline_session,
    )
    photon_result = run_variant(
        variant_id="photon",
        config_path=args.photon_config,
        question=args.question,
        repo_id=args.repo_id,
        session_id=photon_session,
    )

    results = [baseline_result, photon_result]

    if args.json:
        print(
            json.dumps(
                to_json_payload(args.question, results), ensure_ascii=False, indent=2
            )
        )
    else:
        print_text_report(args.question, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
