from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.document_collection import (  # noqa: E402
    collect_documents,
    default_fetcher,
    parse_url_list,
    write_collection_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect HTML/PDF/text documents into an ingest-ready markdown corpus"
    )
    parser.add_argument("--urls", required=True, help="Markdown/plain-text URL list")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--output-root",
        default="data/processed/document_corpora",
        help="Generated corpus root. The default is gitignored.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument(
        "--user-agent",
        default="photon-rag-document-collector/0.1",
        help="HTTP User-Agent sent to source servers.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional JSON manifest path. Defaults to <output>/<corpus>/manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the URL list and print sources without downloading.",
    )
    args = parser.parse_args(argv)

    sources = parse_url_list(args.urls)
    if args.dry_run:
        print(json.dumps([source.__dict__ for source in sources], indent=2))
        return 0

    fetcher = partial(
        default_fetcher,
        timeout_seconds=args.timeout_seconds,
        user_agent=args.user_agent,
    )
    collected = collect_documents(
        sources,
        output_root=args.output_root,
        corpus_id=args.corpus_id,
        fetcher=fetcher,
        delay_seconds=args.delay_seconds,
    )
    manifest = (
        Path(args.manifest)
        if args.manifest
        else Path(args.output_root) / args.corpus_id / "manifest.json"
    )
    write_collection_manifest(collected, manifest)
    print(f"Collected {len(collected)} documents -> {Path(args.output_root) / args.corpus_id}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
