"""
train_photon.py  –  Train a PHOTON model.

Usage (manual CLI – uses YAML paths: section as default):
    python scripts/train_photon.py --config configs/photon_tiny.yaml

Usage (Streamlit app / run-namespaced run):
    python -u -m scripts.train_photon \\
        --config configs/photon_fastapi.yaml \\
        --checkpoint-dir checkpoints/fastapi/train_20260420_101500 \\
        --log-dir logs/train_20260420_101500

Environment variable fallbacks (used only when --checkpoint-dir / --log-dir
are omitted):
    PHOTON_CHECKPOINT_DIR
    PHOTON_LOG_DIR

If neither CLI nor env vars are supplied, the script falls back to the YAML
`paths.checkpoint_root` / `paths.log_root` values (original behavior).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_ref.config import load_photon_config  # noqa: E402
from photon_mlx.trainer import train  # noqa: E402

import yaml  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = (PROJECT_ROOT / "checkpoints").resolve()
LOG_ROOT = (PROJECT_ROOT / "logs").resolve()


def _normalize_under(root: Path, raw: str | os.PathLike) -> Path:
    """Normalize ``raw`` and ensure it resolves under ``root``.

    Rejects ``..`` escapes and absolute paths that land outside ``root``.
    Relative paths are resolved against the project root.
    """
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve(strict=False)
    else:
        p = p.resolve(strict=False)
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path {p} is outside the allowed root {root}") from exc
    return p


def _resolve_dir(
    *,
    cli_value: str | None,
    env_var: str,
    yaml_fallback: str,
    allowed_root: Path,
) -> Path:
    """Resolve a directory from CLI > env > YAML, under ``allowed_root``."""
    raw = cli_value or os.environ.get(env_var) or yaml_fallback
    return _normalize_under(allowed_root, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PHOTON model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint root (falls back to PHOTON_CHECKPOINT_DIR "
        "env var then YAML paths.checkpoint_root).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Override log root (falls back to PHOTON_LOG_DIR env var then "
        "YAML paths.log_root).",
    )
    parser.add_argument(
        "--approved-roots",
        nargs="+",
        default=None,
        help="Issue #135: explicit allow-list of directories the mixed-corpus "
        "loader (iterate_mixed_batches) may read from. Required when "
        "training.train_corpora_mix uses absolute paths outside "
        "data/training/ + data/processed/ (DR4-002 default). Multiple paths "
        "may be passed; relative paths resolve against cwd.",
    )
    args = parser.parse_args()

    cfg = load_photon_config(args.config)

    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    paths = raw.get("paths", {}) or {}
    checkpoint_dir = _resolve_dir(
        cli_value=args.checkpoint_dir,
        env_var="PHOTON_CHECKPOINT_DIR",
        yaml_fallback=paths.get("checkpoint_root", "checkpoints"),
        allowed_root=CHECKPOINT_ROOT,
    )
    log_dir = _resolve_dir(
        cli_value=args.log_dir,
        env_var="PHOTON_LOG_DIR",
        yaml_fallback=paths.get("log_root", "logs"),
        allowed_root=LOG_ROOT,
    )

    approved_roots = (
        [Path(r) for r in args.approved_roots] if args.approved_roots else None
    )

    train(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        resume_from=args.resume or None,
        approved_roots=approved_roots,
    )


if __name__ == "__main__":
    main()
