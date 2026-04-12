"""
build_symbol_graph.py

Build the symbol and dependency graph for an ingested repository.

Usage:
    python scripts/build_symbol_graph.py --repo-id target_repo
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the symbol and dependency graph.")
    parser.add_argument("--repo-id", required=True, help="Repository identifier")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO: implement symbol graph building")


if __name__ == "__main__":
    main()
