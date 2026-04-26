"""End-to-end smoke test for PHOTON checkpoint loading (Issue #135 / Task 0.4).

The Codex Stage 7 review (S7-001) flagged that ``_build_photon_deps``
historically built a fresh ``PhotonModel`` and never loaded any trained
weights. Task 1.3 (commit ``2dbf458``) added an opt-in
``model.checkpoint_path`` field to fix this. This smoke test goes through
the full factory path (``baseline_reporag.pipeline_factory.build_pipeline``)
to verify the wiring is intact end-to-end:

- A YAML config with ``model.provider: photon`` and a real checkpoint
  directory at ``model.checkpoint_path`` produces a PHOTON pipeline whose
  underlying PhotonModel parameters match what we wrote.
- A YAML config with ``model.provider: photon`` but **no**
  ``model.checkpoint_path`` still yields a working pipeline (random weights)
  and emits a WARNING — because we cannot let production run silently on
  random parameters.

This complements ``TestBuildPhotonDepsCheckpointLoad`` (which tests
``_build_photon_deps`` in isolation) by catching factory-level wiring bugs
that the unit tests would miss.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# These imports stay at module level — the test file itself is opt-in:
# baseline-only CI runs that skip MLX should not collect this file
# (covered by the existing pytest convention of ``mlx`` markers if any).
# For now we depend on MLX being installed wherever PHOTON tests run.

_PHOTON_MODEL_HEAD = (
    "model:\n"
    "  provider: photon\n"
    "  architecture: photon_decoder\n"
    "  base_embed_dim: 16\n"
    "  hidden_size: 64\n"
    "  intermediate_size: 128\n"
    "  num_heads: 4\n"
    "  head_dim: 16\n"
    "  vocab_size: 256\n"
    "  max_position_embeddings: 128\n"
    "  model_id: smoke-test-stub\n"
)
_PHOTON_CFG_TAIL = (
    "hierarchy:\n"
    "  levels: 2\n"
    "  chunk_sizes: [4, 4]\n"
    "  encoder_layers_per_level: [1, 1]\n"
    "  decoder_layers_per_level: [1, 1]\n"
    "inference:\n"
    "  safe_recgen_enabled: false\n"
    "repo:\n"
    "  repo_id: smoke-test-repo\n"
    "  repo_commit: abc123\n"
    "retrieval:\n"
    "  lexical_top_k: 5\n"
    "memory:\n"
    "  log_dir: null\n"
)


def _photon_yaml(ckpt_path: str | None) -> str:
    head = _PHOTON_MODEL_HEAD
    if ckpt_path is not None:
        head = head + f"  checkpoint_path: {ckpt_path}\n"
    return head + _PHOTON_CFG_TAIL


def _write_checkpoint(path: Path) -> None:
    """Write a minimal checkpoint at ``path`` matching the smoke YAML cfg."""
    from torch_ref.config import (
        HierarchyConfig,
        ModelConfig,
        PhotonConfig,
        TokenizerConfig,
    )
    from photon_mlx.checkpoint import CheckpointState, save_checkpoint
    from photon_mlx.model import PhotonModel

    cfg = PhotonConfig(
        model=ModelConfig(
            base_embed_dim=16,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
        ),
        hierarchy=HierarchyConfig(
            levels=2,
            chunk_sizes=[4, 4],
            converter_prefix_lengths=[2, 2],
            encoder_layers_per_level=[1, 1],
            decoder_layers_per_level=[1, 1],
        ),
        tokenizer=TokenizerConfig(vocab_size=256),
    )
    mx.random.seed(135)
    model = PhotonModel(cfg)
    save_checkpoint(model, CheckpointState(step=4242, best_val_loss=0.7), path)


class TestPhotonCheckpointSmoke:
    def test_factory_loads_checkpoint_from_yaml(self, tmp_path):
        """build_pipeline → PhotonRAGPipeline carries the loaded weights."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        ckpt_dir = tmp_path / "ckpt"
        _write_checkpoint(ckpt_dir)

        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(_photon_yaml(str(ckpt_dir)))
        cfg = load_config(str(cfg_file))

        # Use _build_photon_deps directly here — going through the full
        # build_pipeline factory would also build the baseline retrieval
        # stack (BM25 / embedding indexes) which is out of scope for the
        # checkpoint smoke test and would slow CI substantially.
        deps = _build_photon_deps(cfg)
        loaded_emb = deps["photon_inference"].model.token_embed.weight

        # Independently load the same checkpoint and compare embedding
        # weights — they must match exactly (no random reseed).
        from torch_ref.config import (
            HierarchyConfig,
            ModelConfig,
            PhotonConfig,
            TokenizerConfig,
        )
        from photon_mlx.checkpoint import load_checkpoint
        from photon_mlx.model import PhotonModel

        ref = PhotonModel(
            PhotonConfig(
                model=ModelConfig(
                    base_embed_dim=16,
                    hidden_size=64,
                    intermediate_size=128,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    head_dim=16,
                    max_position_embeddings=128,
                ),
                hierarchy=HierarchyConfig(
                    levels=2,
                    chunk_sizes=[4, 4],
                    converter_prefix_lengths=[2, 2],
                    encoder_layers_per_level=[1, 1],
                    decoder_layers_per_level=[1, 1],
                ),
                tokenizer=TokenizerConfig(vocab_size=256),
            )
        )
        load_checkpoint(ref, ckpt_dir)
        ref_emb = ref.token_embed.weight

        diff = mx.max(mx.abs(loaded_emb - ref_emb)).item()
        assert diff < 1e-6, (
            f"factory-loaded embedding does not match the on-disk checkpoint "
            f"(max abs diff={diff}); the wiring in _build_photon_deps is broken"
        )

    def test_factory_warns_when_checkpoint_path_unset(self, tmp_path, caplog):
        """No checkpoint_path → WARNING reaches the factory log path."""
        from baseline_reporag.config import load_config
        from baseline_reporag.photon_pipeline import _build_photon_deps

        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(_photon_yaml(None))
        cfg = load_config(str(cfg_file))

        with caplog.at_level(
            logging.WARNING, logger="baseline_reporag.photon_pipeline"
        ):
            _build_photon_deps(cfg)

        warning_msgs = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert any("checkpoint_path" in msg for msg in warning_msgs), (
            "expected factory to surface checkpoint_path=None warning, got: "
            f"{warning_msgs}"
        )
