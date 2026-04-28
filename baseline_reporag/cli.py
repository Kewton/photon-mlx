"""
Baseline RepoRAG – CLI.

Usage (single question):
    python -m baseline_reporag.cli \
        --repo-id fastapi_fastapi \
        --question "認証処理の入口はどこですか？"

Usage (interactive):
    python -m baseline_reporag.cli --repo-id fastapi_fastapi

Usage (PHOTON pipeline shortcut):
    python -m baseline_reporag.cli --use-photon \
        --repo-id fastapi_fastapi \
        --question "..."
"""

from __future__ import annotations

import argparse

from .config import load_config

# CB-004 (codex-fix): import from the lightweight factory so baseline-only
# environments without MLX do not fail at ``python -m baseline_reporag.cli``
# load time.  The factory lazy-imports the PHOTON pipeline (and thus MLX)
# only when ``cfg.model.provider == "photon"``.
from .pipeline_factory import build_pipeline, override_repo_for_pipeline

# A-1 Phase 2: ``--use-photon`` のショートカットが指す PHOTON config。
# baseline.yaml は PHOTON 専用フィールド (checkpoint_path, base_embed_dim 等)
# を持たないため provider のみ override しても動かない。専用 config を使う。
_PHOTON_DEFAULT_CONFIG = "configs/photon_small.yaml"
_BASELINE_DEFAULT_CONFIG = "configs/baseline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline RepoRAG CLI")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            f"Config path (default: {_BASELINE_DEFAULT_CONFIG}; "
            f"with --use-photon: {_PHOTON_DEFAULT_CONFIG})"
        ),
    )
    parser.add_argument(
        "--use-photon",
        action="store_true",
        help=(
            "Shortcut to use the PHOTON pipeline. Equivalent to "
            f"--config {_PHOTON_DEFAULT_CONFIG}. Cannot be combined with --config."
        ),
    )
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    if args.use_photon and args.config is not None:
        parser.error("--use-photon cannot be combined with --config")
    config_path = args.config or (
        _PHOTON_DEFAULT_CONFIG if args.use_photon else _BASELINE_DEFAULT_CONFIG
    )

    cfg = load_config(config_path)
    repo_id = args.repo_id or cfg.repo.repo_id

    # ``--repo-id`` が config の ``repo.repo_id`` と異なる場合、build_pipeline
    # は ``data/indexes/{cfg.repo.repo_id}`` を読むため別 repo の index が
    # ロードされてしまう。``override_repo_for_pipeline`` で cfg を整合させて
    # から build_pipeline を呼び出す。
    override_repo_for_pipeline(cfg, repo_id)

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
