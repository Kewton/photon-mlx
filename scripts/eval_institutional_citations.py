"""Post-hoc citation evaluator CLI for institutional eval set (Issue #110)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from baseline_reporag.eval.institutional.citation_eval import (  # noqa: E402
    ChunkLookup,
    DictChunkLookup,
    grade_eval_set,
)


class _ChunkStoreLookup:
    """Adapter wrapping ``baseline_reporag.ingestion.store.ChunkStore``."""

    def __init__(self, db_path: Path) -> None:
        from baseline_reporag.ingestion.store import ChunkStore  # lazy

        self._store = ChunkStore(db_path)

    def get_chunk_text(self, chunk_id: str) -> str:
        chunk = self._store_get(chunk_id)
        return chunk.content if chunk is not None else ""

    def get_doc_id(self, chunk_id: str) -> str:
        chunk = self._store_get(chunk_id)
        if chunk is None:
            return ""
        parent = Path(chunk.rel_path).parent.name
        return parent

    def _store_get(self, chunk_id: str):
        getter = getattr(self._store, "get", None)
        if callable(getter):
            return getter(chunk_id)
        return None


def _build_lookup(args: argparse.Namespace) -> ChunkLookup:
    if args.in_memory:
        return DictChunkLookup(data={})
    if not args.chunk_store:
        print("error: --chunk-store is required unless --in-memory", file=sys.stderr)
        sys.exit(2)
    return _ChunkStoreLookup(Path(args.chunk_store))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score institutional citations")
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--run-log", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--chunk-store", default=None)
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.run_log and not args.predictions:
        print("error: one of --run-log / --predictions is required", file=sys.stderr)
        return 2
    lookup = _build_lookup(args)

    report = grade_eval_set(
        eval_set_path=Path(args.eval_set),
        lookup=lookup,
        run_log_path=Path(args.run_log) if args.run_log else None,
        predictions_path=Path(args.predictions) if args.predictions else None,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d")
        output_path = Path("reports") / f"institutional_citation_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"report": str(output_path), "total": report["total"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
