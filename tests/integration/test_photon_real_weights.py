"""Real-weight integration test for the PHOTON checkpoint load path (Issue #145).

This module exercises the full production checkpoint load chain
``baseline_reporag.photon_pipeline._build_photon_deps`` →
``_load_photon_checkpoint`` → ``photon_mlx.trainer.load_checkpoint`` against a
self-trained 1-step minimal checkpoint.  Together with two negative-path
tests covering ``PHOTON_ALLOW_RANDOM_INIT`` bypass, we pin the structural
guard added in Issue #135 / S7-001 (random-init weights silently used in
production).

Design references:

- ``workspace/design/issue-145-real-weight-test-design-policy.md`` §4–§5
- ``photon_mlx/tests/test_training.py::test_load_model_forward_consistency``
  (forward-logits equality pattern reused here)
- ``baseline_reporag/tests/test_photon_pipeline_checkpoint_load.py`` (boundary
  test for the ``_build_photon_deps`` checkpoint code path; uses MagicMock for
  PhotonModel.  This integration test is the "real PhotonModel" complement.)

The tests are intentionally serial / single-process — ``PHOTON_CHECKPOINT_ROOT``
and ``PHOTON_ALLOW_RANDOM_INIT`` are process-wide env vars, so xdist/forked
parallelism would race (DR4-005).  Mark ``integration`` so the weekly_eval
runner can select these tests explicitly if needed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import numpy as np
import pytest
from mlx.utils import tree_flatten

# torch_ref is a sibling package; the production code paths under test
# (``_build_photon_deps`` / ``photon_mlx.trainer``) use the same
# ``sys.path.insert`` trick to import it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline_reporag.config import load_config  # noqa: E402
from baseline_reporag.photon_pipeline import _build_photon_deps  # noqa: E402
from photon_mlx.loss import photon_loss  # noqa: E402
from photon_mlx.model import PhotonModel  # noqa: E402
from photon_mlx.trainer import TrainState, save_checkpoint  # noqa: E402
from torch_ref.config import (  # noqa: E402
    HierarchyConfig,
    ModelConfig,
    PhotonConfig,
    TokenizerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MINIMAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "photon_test_minimal.yaml"
)


def _photon_cfg_from_yaml(yaml_cfg: Any) -> PhotonConfig:
    """Build the dataclass ``PhotonConfig`` PhotonModel expects.

    The YAML loader yields a dot-access ``baseline_reporag.config.Config`` but
    ``PhotonModel`` (and ``save_checkpoint``) operate on the dataclass
    ``torch_ref.config.PhotonConfig`` directly.  We mirror the field
    extraction performed inside ``_build_photon_deps`` so the trained model
    we save and the model loaded back via the production path use the same
    underlying architecture.
    """
    model_cfg = ModelConfig(
        architecture=yaml_cfg.model.get("architecture", "photon_decoder"),
        base_embed_dim=yaml_cfg.model.base_embed_dim,
        hidden_size=yaml_cfg.model.hidden_size,
        intermediate_size=yaml_cfg.model.intermediate_size,
        num_attention_heads=yaml_cfg.model.get("num_heads", 4),
        num_key_value_heads=yaml_cfg.model.get("num_heads", 4),
        head_dim=yaml_cfg.model.head_dim,
        max_position_embeddings=yaml_cfg.model.max_position_embeddings,
    )
    hierarchy_cfg = HierarchyConfig(
        levels=yaml_cfg.hierarchy.levels,
        chunk_sizes=list(yaml_cfg.hierarchy.chunk_sizes),
        encoder_layers_per_level=list(yaml_cfg.hierarchy.encoder_layers_per_level),
        decoder_layers_per_level=list(yaml_cfg.hierarchy.decoder_layers_per_level),
    )
    tok_cfg = TokenizerConfig(vocab_size=yaml_cfg.tokenizer.vocab_size)
    return PhotonConfig(model=model_cfg, hierarchy=hierarchy_cfg, tokenizer=tok_cfg)


def _flat_l2_norm(model: PhotonModel) -> float:
    """Sum of L2 norms across every parameter (numpy-side, lazy-eval safe).

    Uses ``mlx.utils.tree_flatten`` to walk the nested ``parameters()`` tree
    (DR1-005) — naïve ``.values()`` traversal hits sub-dicts that
    ``np.asarray`` cannot consume.  Each parameter is forced to materialise
    via ``mx.eval`` before conversion to dodge MLX's lazy-evaluation traps.
    """
    total = 0.0
    for _name, param in tree_flatten(model.parameters()):
        mx.eval(param)
        total += float(np.linalg.norm(np.asarray(param).ravel()))
    return total


def _run_one_training_step(model: PhotonModel, vocab_size: int) -> TrainState:
    """Forward + backward + optimizer step on a tiny synthetic batch.

    Mirrors ``photon_mlx/tests/test_training.py::test_tiny_overfit`` (DR1-006)
    so the weights end up materially different from the random init.  We do
    NOT need the loss to converge; just one step is enough to make the
    trained-weight L2 norm distinguishable from a fresh random init.
    """
    batch = mx.random.randint(0, vocab_size, (2, 16))

    def loss_fn(m: PhotonModel, b: mx.array) -> mx.array:
        logits, _ = m(b, labels=b)
        total, _ = photon_loss(logits, b, 0.0)
        return total

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = mlx.optimizers.Adam(learning_rate=1e-3)
    _loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    return TrainState(step=1)


def _make_corrupt_checkpoint(tmp_path: Path, name: str = "corrupt_ckpt") -> Path:
    """Materialise a directory shaped like a photon_mlx checkpoint with bogus weights.

    The directory contains a valid ``state.json`` (matches ``CheckpointState``
    schema, DR1-004) and a ``weights.npz`` filled with raw zero bytes which is
    not a valid NPZ.  ``integrity.json`` is intentionally omitted (DR2-004):
    ``photon_mlx.checkpoint._verify_integrity(strict=False)`` logs a WARNING
    and returns when the file is absent, so the load path proceeds to
    ``mx.load(weights.npz)`` which then fails — exactly the corruption
    scenario the bypass guard is designed to handle.

    DR4-004: the fixture writes only static raw bytes; no pickle / object
    dtype / executable payload reaches the loader.
    """
    ckpt_dir = tmp_path / name
    ckpt_dir.mkdir()
    state_payload = {
        "step": 1,
        "best_val_loss": float("inf"),
        "best_step": 0,
        "patience_counter": 0,
        "train_losses": [],
        "val_losses": [],
    }
    (ckpt_dir / "state.json").write_text(json.dumps(state_payload), encoding="utf-8")
    (ckpt_dir / "weights.npz").write_bytes(b"\x00" * 100)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_photon_pipeline_with_self_trained_minimal_ckpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_hf_tokenizer: Any,
) -> None:
    """Production load path actually populates PhotonModel weights from disk.

    Detector (b) — weight closeness verification (DR1-002 / DR2-007):
    save the trained model's L2 norm before persistence, then assert the
    L2 norm of the model that came back through ``_build_photon_deps``
    matches it within 1e-5.  A silent load skip would leave the post-load
    model at a fresh random init, which has a measurably different norm.
    """
    mx.random.seed(42)

    yaml_cfg = load_config(str(_MINIMAL_CONFIG_PATH))
    photon_cfg = _photon_cfg_from_yaml(yaml_cfg)

    # Step 1-2: build trained model + run one optimizer step.
    trained_model = PhotonModel(photon_cfg)
    state = _run_one_training_step(trained_model, photon_cfg.tokenizer.vocab_size)

    # Step 3: capture the trained reference *before* save so we have a
    # known-good baseline for the post-load comparison.
    norm_trained = _flat_l2_norm(trained_model)
    save_checkpoint(trained_model, state, tmp_path / "test_ckpt")

    # Step 4-5: feed the freshly-saved checkpoint back through the production
    # build path.  ``checkpoint_path`` is a tmp-root-relative basename
    # (DR4-002) so ``_resolve_checkpoint_path``'s root containment check
    # treats ``tmp_path`` as the allowed root.
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    yaml_cfg.model.checkpoint_path = "test_ckpt"

    deps = _build_photon_deps(yaml_cfg)
    assert deps is not None
    assert "photon_inference" in deps

    # DR1-001: ``_build_photon_deps`` returns ``photon_inference`` (the
    # PhotonInference instance), not ``photon_model``.  Reach the model
    # through ``deps['photon_inference'].model`` (PhotonInference.__init__
    # stores it on line 189).
    model_after = deps["photon_inference"].model
    norm_after = _flat_l2_norm(model_after)

    # Detector (b): the weights at this point must be close to the trained
    # reference.  A silent load skip would leave them at a random init whose
    # L2 norm differs by orders of magnitude.
    assert abs(norm_after - norm_trained) < 1e-5, (
        f"PHOTON checkpoint did not load: trained norm={norm_trained!r}, "
        f"loaded norm={norm_after!r}"
    )

    # Smoke: one-token generate to confirm the wired-up PhotonInference is
    # callable (does not assert on output content — fake tokenizer's
    # ``decode`` returns an empty string).  Failures here would indicate a
    # surface-level integration bug rather than a load issue.
    out = deps["photon_inference"].generate_answer("hi", max_new_tokens=1)
    assert isinstance(out, str)


@pytest.mark.integration
def test_photon_load_failure_without_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_hf_tokenizer: Any,
) -> None:
    """Default behaviour: a corrupt checkpoint surfaces as ``RuntimeError``.

    Detector (c) negative path 1 — without ``PHOTON_ALLOW_RANDOM_INIT`` the
    load path must fail loudly so production cannot silently proceed with
    random weights.
    """
    _make_corrupt_checkpoint(tmp_path)
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

    yaml_cfg = load_config(str(_MINIMAL_CONFIG_PATH))
    yaml_cfg.model.checkpoint_path = "corrupt_ckpt"

    with pytest.raises(RuntimeError):
        _build_photon_deps(yaml_cfg)


@pytest.mark.integration
def test_photon_load_failure_with_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    patched_hf_tokenizer: Any,
) -> None:
    """``PHOTON_ALLOW_RANDOM_INIT=1`` bypasses load failure with a WARNING.

    Detector (c) negative path 2: ``_build_photon_deps`` must return a
    populated deps dict (so test fixtures can still exercise the rest of
    the wiring) and emit a WARNING that operators can grep for.
    """
    _make_corrupt_checkpoint(tmp_path)
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    monkeypatch.setenv("PHOTON_ALLOW_RANDOM_INIT", "1")

    yaml_cfg = load_config(str(_MINIMAL_CONFIG_PATH))
    yaml_cfg.model.checkpoint_path = "corrupt_ckpt"

    with caplog.at_level(logging.WARNING):
        deps = _build_photon_deps(yaml_cfg)

    assert deps is not None
    assert "photon_inference" in deps

    matching = [
        r
        for r in caplog.records
        if r.levelname == "WARNING"
        and (
            "random-init" in r.getMessage().lower()
            or "PHOTON_ALLOW_RANDOM_INIT" in r.getMessage()
        )
    ]
    assert matching, (
        "expected a WARNING mentioning 'random-init' or 'PHOTON_ALLOW_RANDOM_INIT' "
        f"in caplog; got: {[r.getMessage() for r in caplog.records]}"
    )
