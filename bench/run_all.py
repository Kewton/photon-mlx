"""
run_all.py  –  Run all benchmark variants defined in eval.yaml.

Usage:
    python bench/run_all.py --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Variant runner (stub – wire to each system's query interface when ready)
# ---------------------------------------------------------------------------

def run_variant(variant_cfg: dict, eval_cfg: dict) -> list[dict]:
    """
    Run a single benchmark variant against all enabled eval sets.
    Returns a list of prediction records.
    """
    raise NotImplementedError(
        f"TODO: implement runner for variant '{variant_cfg['id']}'"
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def save_run_predictions(
    run_id: str,
    variant_id: str,
    predictions: list[dict],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_id}_{variant_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all benchmark variants")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = args.run_id or (
        f"bench_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    output_dir = Path(cfg["run"]["output_dir"]) / run_id
    print(f"run_id:     {run_id}")
    print(f"output_dir: {output_dir}\n")

    for variant in cfg.get("variants", []):
        print(f"  variant: {variant['id']} ...")
        predictions = run_variant(variant, cfg)
        path = save_run_predictions(run_id, variant["id"], predictions, output_dir)
        print(f"    saved {len(predictions)} predictions -> {path}")

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
