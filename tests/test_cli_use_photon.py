"""Tests for the ``--use-photon`` flag added in A-1 Phase 2.

The flag is a shortcut: ``--use-photon`` is equivalent to
``--config configs/photon_small.yaml`` (and is mutually exclusive with
``--config``). The tests below verify:

- ``--use-photon`` selects the PHOTON config path.
- omitting both flags selects the baseline config path.
- combining ``--use-photon`` with ``--config`` exits with an error.

We mock ``load_config`` and ``build_pipeline`` so the tests run on
baseline-only environments (no MLX, no real configs needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from baseline_reporag import cli as cli_module


@pytest.fixture
def mock_pipeline_and_config(monkeypatch):
    """load_config + build_pipeline をモック化し、選ばれた config_path を捕捉する。"""
    captured: dict[str, object] = {}

    fake_cfg = MagicMock()
    fake_cfg.repo.repo_id = "fake_repo"

    def _fake_load_config(path: str):
        captured["config_path"] = path
        return fake_cfg

    fake_pipeline = MagicMock()
    fake_result = MagicMock()
    fake_result.turn_id = 1
    fake_result.latency.total_ms = 1.0
    fake_result.latency.retrieval_ms = 0.5
    fake_result.latency.generation_ms = 0.5
    fake_result.memory.peak_mb = 1.0
    fake_result.answer = "fake-answer"
    fake_result.no_citation = False
    fake_result.wrong_citation_indices = []
    fake_result.cited_chunk_ids = []
    fake_result.session_id = "test-session"
    fake_pipeline.query.return_value = fake_result

    monkeypatch.setattr(cli_module, "load_config", _fake_load_config)
    monkeypatch.setattr(cli_module, "build_pipeline", lambda _cfg: fake_pipeline)

    return captured


def test_use_photon_picks_photon_config(mock_pipeline_and_config, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--use-photon", "--question", "Q?", "--repo-id", "fake_repo"],
    )

    cli_module.main()

    assert mock_pipeline_and_config["config_path"] == "configs/photon_small.yaml"


def test_default_picks_baseline_config(mock_pipeline_and_config, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["cli", "--question", "Q?", "--repo-id", "fake_repo"],
    )

    cli_module.main()

    assert mock_pipeline_and_config["config_path"] == "configs/baseline.yaml"


def test_explicit_config_wins_over_default(
    mock_pipeline_and_config, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cli",
            "--config",
            "configs/institutional_docs.yaml",
            "--question",
            "Q?",
            "--repo-id",
            "fake_repo",
        ],
    )

    cli_module.main()

    assert mock_pipeline_and_config["config_path"] == "configs/institutional_docs.yaml"


def test_use_photon_with_explicit_config_errors(monkeypatch, capsys) -> None:
    """--use-photon と --config を同時指定すると argparse の error 経由で SystemExit。"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "cli",
            "--use-photon",
            "--config",
            "configs/baseline.yaml",
            "--question",
            "Q?",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli_module.main()

    assert excinfo.value.code == 2  # argparse error exit code
    err = capsys.readouterr().err
    assert "--use-photon cannot be combined with --config" in err


def test_subcommand_ask_preserves_query_flow(
    mock_pipeline_and_config, monkeypatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["photon-rag", "ask", "--question", "Q?", "--repo-id", "fake_repo"],
    )

    cli_module.main()

    assert mock_pipeline_and_config["config_path"] == "configs/baseline.yaml"


def test_ingest_subcommand_dispatches_existing_script(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_dispatch(module_name, forwarded_args):
        captured["module_name"] = module_name
        captured["forwarded_args"] = list(forwarded_args)

    monkeypatch.setattr(cli_module, "_dispatch_script", fake_dispatch)

    cli_module.main(["ingest", "--repo", "/tmp/repo", "--repo-id", "tmp_repo"])

    assert captured == {
        "module_name": "scripts.ingest_repo",
        "forwarded_args": ["--repo", "/tmp/repo", "--repo-id", "tmp_repo"],
    }
