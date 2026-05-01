"""Issue #148 Phase A0 — checkpoint loading + model_id allowlist tests.

These tests cover the new checkpoint-loading code path added to
``_build_photon_deps`` (design policy §3 / §4 / §6) and the ``model_id``
HF repo-id allowlist validator (DR4-001 / DR4-002).

The ``_build_photon_deps`` body is heavy (PhotonModel construction +
HuggingFace tokenizer load + PhotonInference wiring), so each test patches
``PhotonModel`` / ``_load_hf_tokenizer`` / ``_validate_tokenizer_id`` and
exercises the load + validation surface area only.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from baseline_reporag.config import load_config


def _mlx_metal_available() -> bool:
    probe = "import mlx.core as mx; mx.array([1]); print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _mlx_metal_available(),
    reason="PHOTON checkpoint tests require an accessible Metal device",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal photon yaml fragment that satisfies all the unrelated _build_photon_deps
# branches (vocab_size, hierarchy, safe_recgen). Each test injects a different
# model section / tokenizer block on top.
_BASE_YAML = (
    "model:\n"
    "  provider: photon\n"
    "  architecture: photon_decoder\n"
    "  base_embed_dim: 64\n"
    "  hidden_size: 128\n"
    "  intermediate_size: 256\n"
    "  num_heads: 4\n"
    "  vocab_size: 1000\n"
    "{model_extra}"
    "tokenizer:\n"
    '  tokenizer_id: "fake-org/fake-tokenizer"\n'
    "  vocab_size: 1000\n"
    "hierarchy:\n"
    "  levels: 2\n"
    "  chunk_sizes: [4, 4]\n"
    "  encoder_layers_per_level: [2, 2]\n"
    "  decoder_layers_per_level: [2, 2]\n"
    "inference:\n"
    "  hierarchical_prefill: true\n"
    "  safe_recgen_enabled: false\n"
)


def _write_photon_cfg(tmp_path: Path, *, checkpoint_path: str | None = None) -> Path:
    """Write a minimal photon yaml to ``tmp_path/photon.yaml``."""
    extra = ""
    if checkpoint_path is not None:
        # Quote so backslashes / colons in tmp paths survive yaml load.
        extra = f'  checkpoint_path: "{checkpoint_path}"\n'
    cfg_file = tmp_path / "photon.yaml"
    cfg_file.write_text(_BASE_YAML.format(model_extra=extra))
    return cfg_file


def _make_valid_ckpt(root: Path, name: str = "ckpt") -> Path:
    """Materialise a directory shaped like a photon_mlx checkpoint."""
    ckpt = root / name
    ckpt.mkdir()
    (ckpt / "weights.npz").write_bytes(b"\x00")
    (ckpt / "state.json").write_text("{}", encoding="utf-8")
    return ckpt


@pytest.fixture(autouse=True)
def _patch_heavy_deps(monkeypatch):
    """Replace the heavy PHOTON deps so the load/validate path runs alone.

    ``PhotonModel`` and ``PhotonInference`` are imported locally inside
    ``_build_photon_deps`` (lazy import boundary), so they must be patched at
    their source module (``photon_mlx.model`` / ``photon_mlx.inference``) —
    not at ``baseline_reporag.photon_pipeline``.

    ``_load_photon_checkpoint`` is a module-level wrapper in
    ``baseline_reporag.photon_pipeline`` so it can be monkeypatched at the
    module boundary without re-exporting the heavy trainer import.
    """

    # Stub HF tokenizer loader (autouse fixture in test_photon_pipeline.py
    # patches the same target, but this test file is independent).
    def _fake_loader(tokenizer_id, expected_vocab_size):
        fake = MagicMock()
        fake.vocab_size = expected_vocab_size
        fake.pad_token_id = 0
        fake.encode.return_value = [1, 2, 3]
        return fake

    monkeypatch.setattr(
        "baseline_reporag.photon_pipeline._load_hf_tokenizer", _fake_loader
    )
    # Bypass tokenizer_id validation so missing ckpt errors surface first
    # (design §10.1 — without this, _validate_tokenizer_id raises before the
    # checkpoint code path runs).
    monkeypatch.setattr(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        lambda v: v,
    )
    # PhotonModel is imported lazily inside _build_photon_deps; patch at
    # the source module so the local ``from photon_mlx.model import
    # PhotonModel`` picks up the mock.
    monkeypatch.setattr(
        "photon_mlx.model.PhotonModel",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "photon_mlx.inference.PhotonInference",
        MagicMock(return_value=MagicMock()),
    )
    # SafeRecGenController is also lazy — patch at source.
    monkeypatch.setattr(
        "photon_mlx.safe_recgen.SafeRecGenController",
        MagicMock(return_value=MagicMock()),
    )


# ---------------------------------------------------------------------------
# Phase A0 — checkpoint loading code path
# ---------------------------------------------------------------------------


class TestBuildPhotonDepsCheckpointLoad:
    """Phase A0: ``_build_photon_deps`` must load PhotonModel weights from
    ``cfg.model.checkpoint_path`` when set (design §3 DR-1)."""

    def test_build_photon_deps_loads_checkpoint_when_path_set(
        self, tmp_path, monkeypatch
    ):
        """``load_checkpoint(model, ckpt_dir)`` is invoked when checkpoint_path is set."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(ckpt))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            _build_photon_deps(cfg)

        mock_load.assert_called_once()
        # Positional args: (model, ckpt_dir). The directory must point at the
        # checkpoint root we just materialised.
        args = mock_load.call_args[0]
        assert len(args) == 2
        assert Path(args[1]).resolve() == ckpt.resolve()

    def test_build_photon_deps_raises_on_load_failure_by_default(
        self, tmp_path, monkeypatch
    ):
        """Default policy is fail-fast: load failure becomes RuntimeError."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
        # Ensure the fail-soft escape hatch is OFF.
        monkeypatch.delenv("PHOTON_ALLOW_RANDOM_INIT", raising=False)

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(ckpt))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint",
            side_effect=FileNotFoundError("not found"),
        ):
            with pytest.raises(RuntimeError, match="checkpoint load failed"):
                _build_photon_deps(cfg)

    def test_build_photon_deps_falls_back_when_PHOTON_ALLOW_RANDOM_INIT(
        self, tmp_path, monkeypatch, caplog
    ):
        """``PHOTON_ALLOW_RANDOM_INIT=1`` switches load failure to WARNING + continue."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
        monkeypatch.setenv("PHOTON_ALLOW_RANDOM_INIT", "1")

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(ckpt))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint",
            side_effect=FileNotFoundError("not found"),
        ):
            with caplog.at_level(
                logging.WARNING, logger="baseline_reporag.photon_pipeline"
            ):
                # Should NOT raise — fail-soft path.
                _build_photon_deps(cfg)

        joined = " ".join(record.message for record in caplog.records)
        assert "checkpoint load failed" in joined
        assert "PHOTON_ALLOW_RANDOM_INIT" in joined

    def test_checkpoint_path_root_containment_validation(self, tmp_path, monkeypatch):
        """Checkpoints outside the allowed root are rejected (DR4-001)."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        # Materialise a valid-shape ckpt under the disallowed location.
        outside_ckpt = _make_valid_ckpt(outside, "evil_ckpt")
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(allowed))

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(outside_ckpt))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            with pytest.raises(RuntimeError, match="approved checkpoint roots"):
                _build_photon_deps(cfg)
        mock_load.assert_not_called()

    def test_checkpoint_path_symlink_escape_rejected(self, tmp_path, monkeypatch):
        """A symlink that resolves outside the allowed root is rejected."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        real_ckpt = _make_valid_ckpt(outside, "real_ckpt")
        # Create a symlink under ``allowed/`` that points to the real ckpt
        # outside ``allowed``. The validator must follow the symlink and
        # reject the resolved real path.
        link = allowed / "linked_ckpt"
        link.symlink_to(real_ckpt)

        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(allowed))
        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(link))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            with pytest.raises(RuntimeError, match="approved checkpoint roots"):
                _build_photon_deps(cfg)
        mock_load.assert_not_called()

    def test_build_photon_deps_rejects_invalid_checkpoint_shape(
        self, tmp_path, monkeypatch
    ):
        """Missing weights.npz / state.json is rejected before load_checkpoint."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        broken = tmp_path / "broken_ckpt"
        broken.mkdir()
        # Intentionally create only one of the two required files.
        (broken / "weights.npz").write_bytes(b"\x00")
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(broken))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            with pytest.raises(RuntimeError, match="weights.npz"):
                _build_photon_deps(cfg)
        mock_load.assert_not_called()

    def test_build_photon_deps_warns_when_no_checkpoint_path(self, tmp_path, caplog):
        """When ``checkpoint_path`` is unset, a WARNING is emitted but no raise."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=None)
        cfg = load_config(str(cfg_file))

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            _build_photon_deps(cfg)

        joined = " ".join(record.message for record in caplog.records)
        assert "checkpoint_path" in joined or "random-init" in joined


# ---------------------------------------------------------------------------
# DR4-002 — model_id HF repo-id allowlist
# ---------------------------------------------------------------------------


class TestModelIdRepoIdAllowlist:
    """``model.model_id`` must conform to the HF repo-id allowlist
    ``<org>/<name>`` with ASCII ``[A-Za-z0-9._-]`` only (DR4-002)."""

    def test_model_id_repo_id_allowlist_rejects_url(self, tmp_path, monkeypatch):
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            _BASE_YAML.format(
                model_extra=(
                    f'  checkpoint_path: "{ckpt}"\n'
                    '  model_id: "https://hf.co/org/repo"\n'
                )
            )
        )
        cfg = load_config(str(cfg_file))

        with patch("baseline_reporag.photon_pipeline._load_photon_checkpoint"):
            with pytest.raises(ValueError, match="model_id"):
                _build_photon_deps(cfg)

    def test_model_id_repo_id_allowlist_rejects_local_path(self, tmp_path, monkeypatch):
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            _BASE_YAML.format(
                model_extra=(
                    f'  checkpoint_path: "{ckpt}"\n  model_id: "/abs/path/to/model"\n'
                )
            )
        )
        cfg = load_config(str(cfg_file))

        with patch("baseline_reporag.photon_pipeline._load_photon_checkpoint"):
            with pytest.raises(ValueError, match="model_id"):
                _build_photon_deps(cfg)

    def test_model_id_repo_id_allowlist_rejects_traversal(self, tmp_path, monkeypatch):
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            _BASE_YAML.format(
                model_extra=(
                    f'  checkpoint_path: "{ckpt}"\n  model_id: "../org/name"\n'
                )
            )
        )
        cfg = load_config(str(cfg_file))

        with patch("baseline_reporag.photon_pipeline._load_photon_checkpoint"):
            with pytest.raises(ValueError, match="model_id"):
                _build_photon_deps(cfg)

    def test_model_id_repo_id_allowlist_accepts_valid_slug(self, tmp_path, monkeypatch):
        """A canonical ``mlx-community/...`` slug passes validation."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path)
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(
            _BASE_YAML.format(
                model_extra=(
                    f'  checkpoint_path: "{ckpt}"\n'
                    '  model_id: "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"\n'
                )
            )
        )
        cfg = load_config(str(cfg_file))

        with patch("baseline_reporag.photon_pipeline._load_photon_checkpoint"):
            # Should NOT raise.
            _build_photon_deps(cfg)

    def test_model_id_validation_error_does_not_leak_raw_value(self):
        """CB-004: unsafe model_id must not appear in the ValueError message."""
        from baseline_reporag.photon_pipeline import _validate_repo_id

        unsafe_values = [
            "https://hf.co/org/repo",
            "/abs/path/to/model",
            "../org/name",
            "noSlash",
            "org/name/extra",
            "~user/model",
            "org\x00name",
        ]
        for bad in unsafe_values:
            with pytest.raises(ValueError) as exc_info:
                _validate_repo_id(bad, "model_id")
            # The raw input value must NOT appear in the error message.
            assert bad not in str(exc_info.value), (
                f"raw value {bad!r} leaked into ValueError message: {exc_info.value}"
            )
            # The key name must appear so operators know which field failed.
            assert "model_id" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CB-001 — relative checkpoint_path resolved against PHOTON_CHECKPOINT_ROOT
# ---------------------------------------------------------------------------


class TestCheckpointPathRelativeResolution:
    """CB-001: relative checkpoint_path must be resolved relative to
    PHOTON_CHECKPOINT_ROOT, not cwd."""

    def test_checkpoint_path_relative_resolved_against_root(
        self, tmp_path, monkeypatch
    ):
        """A bare name like 'mulmoclaude_step600' resolves under the root."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt = _make_valid_ckpt(tmp_path, "mulmoclaude_step600")
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

        # Pass the relative name only (no directory prefix).
        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path="mulmoclaude_step600")
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            _build_photon_deps(cfg)

        mock_load.assert_called_once()
        args = mock_load.call_args[0]
        assert Path(args[1]).resolve() == ckpt.resolve()

    def test_checkpoint_path_relative_outside_root_rejected(
        self, tmp_path, monkeypatch
    ):
        """A relative path that resolves outside the root must be rejected.

        root = tmp_path/allowed
        outside = tmp_path/outside
        relative "../outside/evil" from root resolves to tmp_path/outside/evil,
        which is outside the allowed root — must be rejected.
        """
        from baseline_reporag.photon_pipeline import _resolve_checkpoint_path

        root = tmp_path / "allowed"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        _make_valid_ckpt(outside, "evil")

        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(root))

        # "../outside/evil" from root resolves to tmp_path/outside/evil,
        # which is outside root (tmp_path/allowed) — must be rejected.
        with pytest.raises(RuntimeError, match="approved checkpoint roots"):
            _resolve_checkpoint_path("../outside/evil")


# ---------------------------------------------------------------------------
# CB-002 — directory entries for weights.npz / state.json are rejected
# ---------------------------------------------------------------------------


class TestCheckpointShapeIsFile:
    """CB-002: directory-shaped weights.npz / state.json must be rejected."""

    def test_checkpoint_path_rejects_when_required_entry_is_directory(
        self, tmp_path, monkeypatch
    ):
        """weights.npz that is a directory (not a file) must be rejected."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        broken = tmp_path / "broken_ckpt"
        broken.mkdir()
        # Create weights.npz as a **directory** instead of a file.
        (broken / "weights.npz").mkdir()
        # state.json is a real file so only weights.npz triggers the rejection.
        (broken / "state.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(broken))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            with pytest.raises(RuntimeError, match="weights.npz"):
                _build_photon_deps(cfg)
        mock_load.assert_not_called()

    def test_checkpoint_path_rejects_when_state_json_is_directory(
        self, tmp_path, monkeypatch
    ):
        """state.json that is a directory (not a file) must be rejected."""
        from baseline_reporag.photon_pipeline import _build_photon_deps

        broken = tmp_path / "broken_state_ckpt"
        broken.mkdir()
        (broken / "weights.npz").write_bytes(b"\x00")
        # Create state.json as a **directory**.
        (broken / "state.json").mkdir()
        monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

        cfg_file = _write_photon_cfg(tmp_path, checkpoint_path=str(broken))
        cfg = load_config(str(cfg_file))

        with patch(
            "baseline_reporag.photon_pipeline._load_photon_checkpoint"
        ) as mock_load:
            with pytest.raises(RuntimeError, match="state.json"):
                _build_photon_deps(cfg)
        mock_load.assert_not_called()
