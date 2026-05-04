from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path


def test_photon_mlx_public_api_import_does_not_import_rag() -> None:
    for name in list(sys.modules):
        if name == "baseline_reporag" or name.startswith("baseline_reporag."):
            sys.modules.pop(name, None)
        if name == "photon_mlx" or name.startswith("photon_mlx."):
            sys.modules.pop(name, None)
        if name == "mlx.core":
            sys.modules.pop(name, None)

    photon_mlx = importlib.import_module("photon_mlx")

    assert "PhotonModel" in photon_mlx.__all__
    assert "PhotonInference" in photon_mlx.__all__
    assert photon_mlx.load_photon_config is not None
    assert "baseline_reporag" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_photon_only_console_scripts_are_declared() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["photon-train"] == "photon_mlx.cli:train_main"
    assert scripts["photon-generate"] == "photon_mlx.cli:generate_main"


def test_photon_only_console_script_targets_are_importable() -> None:
    train_module_name, train_attr = "photon_mlx.cli", "train_main"
    generate_module_name, generate_attr = "photon_mlx.cli", "generate_main"

    assert hasattr(importlib.import_module(train_module_name), train_attr)
    assert hasattr(importlib.import_module(generate_module_name), generate_attr)
