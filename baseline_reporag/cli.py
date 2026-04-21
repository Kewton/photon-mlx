"""
Baseline RepoRAG – CLI.

Usage (single question):
    python -m baseline_reporag.cli \
        --repo-id fastapi_fastapi \
        --question "認証処理の入口はどこですか？"

Usage (interactive):
    python -m baseline_reporag.cli --repo-id fastapi_fastapi
"""

from __future__ import annotations

import argparse

from .config import load_config

# CB-004 (codex-fix): import from the lightweight factory so baseline-only
# environments without MLX do not fail at ``python -m baseline_reporag.cli``
# load time.  The factory lazy-imports the PHOTON pipeline (and thus MLX)
# only when ``cfg.model.provider == "photon"``.
from .pipeline_factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RepoRAG CLI")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg.repo.repo_id

    # Route via ``build_pipeline`` so ``model.provider`` (baseline vs
    # photon) is honoured end-to-end — Issue #62 Phase 1 Stage 3 DR3-001:
    # without this switch, ``inference.photon_generation_enabled=true``
    # would have no effect when invoked from the CLI.
    pipeline = build_pipeline(cfg)

    def run_query(question: str) -> None:
        result = pipeline.query(
            question=question,
            session_id=args.session_id,
            repo_id=repo_id,
        )
        print(
            f"\n[Turn {result.turn_id}]  {result.latency.total_ms:.0f} ms"
            f"  (retrieval {result.latency.retrieval_ms:.0f}"
            f" | gen {result.latency.generation_ms:.0f})"
            f"  mem {result.memory.peak_mb:.1f} MB\n"
        )
        print(result.answer)
        if result.no_citation:
            print("\n[WARNING] No citations in this answer.")
        if result.wrong_citation_indices:
            print(
                f"[WARNING] Unknown citation indices: {result.wrong_citation_indices}"
            )
        print(f"\nCited: {result.cited_chunk_ids}")
        print(f"Session: {result.session_id}")

    if args.question:
        run_query(args.question)
    else:
        print(f"session: {args.session_id or '(auto)'}")
        print("Type your question (empty line to quit):\n")
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_query(q)


if __name__ == "__main__":
    main()
