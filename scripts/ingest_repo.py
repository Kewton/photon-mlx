"""
ingest_repo.py

Ingest a target repository into the chunk store and metadata store.

Usage:
    python scripts/ingest_repo.py \
        --repo /path/to/target-repo \
        --repo-id target_repo \
        --commit HEAD
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a repository into the chunk store.")
    parser.add_argument("--repo", required=True, help="Absolute path to the target repository")
    parser.add_argument("--repo-id", required=True, help="Identifier for this repository")
    parser.add_argument("--commit", default="HEAD", help="Commit SHA to fix the snapshot")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO: implement repo ingestion")


if __name__ == "__main__":
    main()
