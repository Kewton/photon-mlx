"""Regression test for the photon_mlx lazy-import boundary (Issue #135 / DR1-002).

``baseline_reporag.pipeline_factory`` must keep its lazy-MLX promise even
on the photon branch: importing ``photon_pipeline`` and calling
``_build_photon_deps`` must not pull ``photon_mlx.trainer`` (which top-
level imports ``mlx.optimizers`` and ``photon_mlx.loss``). The boundary
was established by Issue #135 Phase 1 (commit ea2fa57) which physically
split checkpoint I/O out of the trainer module.

This test lives in its own file so it does not interfere with the
develop-side fixtures in ``test_photon_pipeline.py`` (autouse stubs for
``_load_hf_tokenizer``); the subprocess approach also avoids the
sys.modules manipulation that would taint other tests in the same
pytest session.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


_PHOTON_CFG_BASE = (
    "model:\n"
    "  provider: photon\n"
    "  architecture: photon_decoder\n"
    "  base_embed_dim: 16\n"
    "  hidden_size: 64\n"
    "  intermediate_size: 128\n"
    "  num_heads: 4\n"
    "  head_dim: 16\n"
    '  model_id: "fake-org/fake-model"\n'
    "  max_position_embeddings: 128\n"
    "hierarchy:\n"
    "  levels: 2\n"
    "  chunk_sizes: [4, 4]\n"
    "  encoder_layers_per_level: [1, 1]\n"
    "  decoder_layers_per_level: [1, 1]\n"
    "tokenizer:\n"
    '  tokenizer_id: "fake-org/fake-tokenizer"\n'
    "  vocab_size: 256\n"
    "inference:\n"
    "  safe_recgen_enabled: false\n"
)


class TestBuildPhotonDepsLazyImport:
    def test_build_photon_deps_does_not_import_trainer(self, tmp_path):
        """DR1-002 / DR3-001: _build_photon_deps must not pull photon_mlx.trainer.

        Runs in a subprocess so other tests in the same pytest session that
        legitimately import ``photon_mlx.trainer`` (end-to-end training
        tests) are not affected by sys.modules manipulation.
        """
        cfg_file = tmp_path / "photon.yaml"
        cfg_file.write_text(_PHOTON_CFG_BASE)

        repo_root = Path(__file__).resolve().parent.parent.parent
        script = textwrap.dedent(
            f"""
            import sys
            from unittest.mock import MagicMock, patch

            sys.path.insert(0, {str(repo_root)!r})

            fake_tokenizer = MagicMock()
            fake_tokenizer.vocab_size = 256
            fake_tokenizer.pad_token_id = 0

            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_tokenizer,
            ):
                from baseline_reporag.config import load_config
                from baseline_reporag.photon_pipeline import _build_photon_deps

                cfg = load_config({str(cfg_file)!r})
                try:
                    _build_photon_deps(cfg)
                except ImportError as exc:
                    if "Metal device" not in str(exc):
                        raise

            forbidden = [
                m for m in sys.modules
                if m == "photon_mlx.trainer"
                or m == "photon_mlx.loss"
                or m.startswith("mlx.optimizers")
            ]
            if forbidden:
                print("BOUNDARY_VIOLATION:" + ",".join(sorted(forbidden)))
                sys.exit(1)
            print("BOUNDARY_OK")
            """
        )

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=60,
        )

        assert "BOUNDARY_OK" in proc.stdout, (
            "_build_photon_deps boundary check failed.\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
