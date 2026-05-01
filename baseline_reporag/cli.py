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
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

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
_SCRIPT_COMMANDS = {
    "ingest": ("scripts.ingest_repo", "Ingest a repository or markdown corpus"),
    "index": ("scripts.build_indexes", "Build lexical and embedding indexes"),
    "symbol-graph": ("scripts.build_symbol_graph", "Build the optional symbol graph"),
    "heading-graph": (
        "scripts.build_heading_graph",
        "Build the optional heading graph",
    ),
}


def _add_query_args(parser: argparse.ArgumentParser) -> None:
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


def _run_query_cli(args: argparse.Namespace) -> None:
    """Run the existing single-question / interactive CLI flow."""

    if args.use_photon and args.config is not None:
        raise ValueError("--use-photon cannot be combined with --config")
    config_path = args.config or (
        _PHOTON_DEFAULT_CONFIG if args.use_photon else _BASELINE_DEFAULT_CONFIG
    )
    config_path = _resolve_config_path(config_path)

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


def _run_server(args: argparse.Namespace) -> None:
    from .server import main as server_main

    server_argv = [
        "--config",
        _resolve_config_path(args.config),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    server_main(server_argv)


def _dispatch_script(module_name: str, forwarded_args: Sequence[str]) -> None:
    """Dispatch installable wrapper commands to existing script modules."""

    import importlib

    old_argv = sys.argv[:]
    sys.argv = [module_name.rsplit(".", 1)[-1], *forwarded_args]
    try:
        module = importlib.import_module(module_name)
        module.main()
    finally:
        sys.argv = old_argv


def _resolve_config_path(config_path: str) -> str:
    """Resolve packaged default configs when running after wheel install."""

    path = Path(config_path)
    if path.exists() or path.is_absolute() or path.parent.name != "configs":
        return config_path
    try:
        packaged = resources.files("configs").joinpath(path.name)
    except ModuleNotFoundError:
        return config_path
    if packaged.is_file():
        return str(packaged)
    return config_path


def _build_command_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photon-rag",
        description="PHOTON RepoRAG command line interface",
    )
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    _add_query_args(ask_parser)
    ask_parser.set_defaults(handler=_run_query_cli)

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--config", default=_BASELINE_DEFAULT_CONFIG)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(handler=_run_server)

    for command, (_, help_text) in _SCRIPT_COMMANDS.items():
        subparsers.add_parser(command, help=help_text)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for both ``python -m baseline_reporag.cli`` and ``photon-rag``.

    Backwards compatibility: when no subcommand is present, arguments are
    interpreted as the historical query CLI.
    """

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in _SCRIPT_COMMANDS:
        module_name, _ = _SCRIPT_COMMANDS[raw_argv[0]]
        _dispatch_script(module_name, raw_argv[1:])
        return

    if raw_argv and raw_argv[0] in {"ask", "serve"}:
        parser = _build_command_parser()
        args = parser.parse_args(raw_argv)
        try:
            args.handler(args)
        except ValueError as exc:
            parser.error(str(exc))
        return

    if raw_argv and raw_argv[0] in {"-h", "--help"}:
        _build_command_parser().print_help()
        return

    legacy_parser = argparse.ArgumentParser(description="Baseline RepoRAG CLI")
    _add_query_args(legacy_parser)
    args = legacy_parser.parse_args(raw_argv)
    try:
        _run_query_cli(args)
    except ValueError as exc:
        legacy_parser.error(str(exc))


if __name__ == "__main__":
    main()
