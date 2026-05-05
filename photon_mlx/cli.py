"""Lightweight PHOTON-only console entrypoints."""

from __future__ import annotations


def train_main() -> None:
    from scripts.train_photon import main

    main()


def generate_main() -> None:
    from scripts.generate_photon import main

    main()
