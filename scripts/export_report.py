"""
export_report.py

Export a benchmark report for a completed run.

Usage:
    python scripts/export_report.py --run-id latest
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a benchmark report.")
    parser.add_argument("--run-id", required=True, help="Run identifier (or 'latest')")
    parser.add_argument("--config", default="configs/eval.yaml", help="Config file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError("TODO: implement report export")


if __name__ == "__main__":
    main()
