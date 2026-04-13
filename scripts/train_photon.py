"""
train_photon.py  –  Train a PHOTON model.

Usage:
    python scripts/train_photon.py --config configs/photon_tiny.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_ref.config import load_photon_config
from photon_mlx.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PHOTON model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_photon_config(args.config)

    import yaml

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    t = raw.get("training", {})
    train(
        cfg=cfg,
        train_corpus=t.get("train_corpus", "data/processed/train_tiny.jsonl"),
        val_corpus=t.get("val_corpus", "data/processed/val_tiny.jsonl"),
        checkpoint_dir=raw.get("paths", {}).get("checkpoint_root", "checkpoints"),
        log_dir=raw.get("paths", {}).get("log_root", "logs"),
        resume_from=args.resume or None,
    )


if __name__ == "__main__":
    main()
