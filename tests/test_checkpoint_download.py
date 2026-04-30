from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from baseline_reporag.checkpoints import maybe_download_checkpoint


def test_maybe_download_checkpoint_noops_without_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

    maybe_download_checkpoint("missing_ckpt")

    assert not (tmp_path / "missing_ckpt").exists()


def test_maybe_download_checkpoint_uses_hf_snapshot_download(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        ckpt = tmp_path / "job" / "step_003000"
        ckpt.mkdir(parents=True)
        (ckpt / "weights.npz").write_bytes(b"weights")
        (ckpt / "state.json").write_text("{}", encoding="utf-8")
        (ckpt / "integrity.json").write_text("{}", encoding="utf-8")
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    maybe_download_checkpoint(
        "job/step_003000",
        repo_id="Kewton/photon-institutional-retrain-20260428",
        revision="main",
    )

    assert calls == [
        {
            "repo_id": "Kewton/photon-institutional-retrain-20260428",
            "revision": "main",
            "allow_patterns": [
                "job/step_003000/weights.npz",
                "job/step_003000/state.json",
                "job/step_003000/integrity.json",
            ],
            "local_dir": str(tmp_path),
        }
    ]
    assert (tmp_path / "job" / "step_003000" / "weights.npz").is_file()


def test_maybe_download_checkpoint_reports_download_failure(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))

    def fake_snapshot_download(**_kwargs):
        raise OSError("network down")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    with pytest.raises(RuntimeError, match="checkpoint download failed"):
        maybe_download_checkpoint(
            "job/step_003000",
            repo_id="Kewton/photon-institutional-retrain-20260428",
        )
