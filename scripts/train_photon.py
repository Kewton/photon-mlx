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

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PHOTON model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_photon_config(args.config)

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = raw.get("paths", {})
    train(
        cfg=cfg,
        checkpoint_dir=paths.get("checkpoint_root", "checkpoints"),
        log_dir=paths.get("log_root", "logs"),
        resume_from=args.resume or None,
    )


if __name__ == "__main__":
    main()
