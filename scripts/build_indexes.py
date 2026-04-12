"""
build_indexes.py

Build lexical and embedding indexes for an ingested repository.

Usage:
    python scripts/build_indexes.py --repo-id target_repo
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lexical and embedding indexes.")
    parser.add_argument("--repo-id", required=True, help="Repository identifier")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO: implement index building")


if __name__ == "__main__":
    main()
