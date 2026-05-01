"""CB-004 / CB2-001 (codex-fix): ``baseline_reporag.pipeline_factory``
must not eagerly import any MLX module.

The first iteration of this test only asserted that
``baseline_reporag.photon_pipeline`` was absent from ``sys.modules``
after importing the factory, which was vacuous — the real leak path is
``baseline_reporag.pipeline`` → ``baseline_reporag.generation.generator``
→ ``mlx_lm`` (→ ``mlx.core``).  A factory that eagerly imported
``RepoRAGPipeline`` would still pass the old assertion even though MLX
was fully loaded.

This rewrite asserts the real invariant: after importing the factory,
*none* of the ``mlx*`` modules (``mlx_lm``, ``mlx``, ``mlx.core``) are
present in ``sys.modules``.  Only when ``build_pipeline(cfg)`` is
called with ``provider="photon"`` may MLX appear.

The checks run in a fresh Python subprocess so prior test imports in
this process cannot pollute ``sys.modules``.  The subprocess succeeds
on a dev box with MLX installed — we're verifying that *import time*
does not pull MLX in, not that MLX is uninstallable.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def _mlx_metal_available() -> bool:
    result = _run_subprocess("import mlx.core as mx; mx.array([1]); print('OK')")
    return result.returncode == 0


def test_import_pipeline_factory_does_not_load_mlx() -> None:
    """Importing ``baseline_reporag.pipeline_factory`` must NOT load any
    ``mlx*`` module.

    This is the real CB-004 invariant — the previous test was a proxy
    that a truly-leaking factory could still satisfy.
    """
    script = textwrap.dedent(
        """
        import sys

        # Sanity: MLX must not be preloaded by the subprocess.
        for mod in ("mlx_lm", "mlx", "mlx.core"):
            assert mod not in sys.modules, f"{mod} unexpectedly preloaded"

        import baseline_reporag.pipeline_factory  # noqa: F401

        # Core invariant: the lightweight factory must NOT pull in MLX
        # at import time — neither directly (via photon_pipeline) nor
        # transitively (via pipeline -> generation.generator -> mlx_lm).
        leaked = [m for m in ("mlx_lm", "mlx", "mlx.core") if m in sys.modules]
        if leaked:
            raise AssertionError(
                f"pipeline_factory import leaked MLX modules: {leaked}"
            )

        # And the heavy pipeline modules themselves must stay unloaded.
        for heavy in ("baseline_reporag.photon_pipeline",):
            assert heavy not in sys.modules, (
                f"pipeline_factory import leaked {heavy}"
            )

        # The factory surface must still expose build_pipeline.
        assert hasattr(baseline_reporag.pipeline_factory, "build_pipeline")
        print("OK")
        """
    )
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_build_pipeline_photon_branch_imports_photon_pipeline_lazily() -> None:
    """Calling ``build_pipeline`` with ``provider="photon"`` imports PHOTON lazily.

    This is the counterpart invariant to
    ``test_import_pipeline_factory_does_not_load_mlx``: we want
    laziness, not avoidance. The PHOTON branch may import the
    ``photon_pipeline`` module, but ``mlx.core`` itself should stay
    behind the module's lazy proxy until a real MLX operation is needed.
    We patch ``photon_pipeline`` builders to no-ops so the test doesn't
    need real indexes.
    """
    if not _mlx_metal_available():
        pytest.skip("MLX Metal device is not available on this runner")

    script = textwrap.dedent(
        """
        import sys

        for mod in ("mlx_lm", "mlx", "mlx.core"):
            assert mod not in sys.modules, f"{mod} unexpectedly preloaded"

        import baseline_reporag.pipeline_factory as pf

        # Confirm MLX stays out *before* we invoke the factory.
        pre_leak = [m for m in ("mlx_lm", "mlx", "mlx.core") if m in sys.modules]
        assert not pre_leak, f"MLX leaked pre-invocation: {pre_leak}"

        # Patch the PHOTON-branch helpers before triggering the photon
        # import, so we don't need real indexes on disk.
        class _Cfg:
            class model:
                provider = "photon"

        import baseline_reporag.photon_pipeline as pp

        def _fake_baseline_deps(cfg):
            return {
                "store": object(), "lexical": object(), "embedding": object(),
                "graph": object(), "sessions": object(), "generator": object(),
                "logger": object(), "reranker": None,
            }

        def _fake_photon_deps(cfg):
            return {
                "photon_inference": object(), "safe_recgen": object(),
                "photon_cfg": object(), "tokenizer": object(),
            }

        pp._build_baseline_deps = _fake_baseline_deps
        pp._build_photon_deps = _fake_photon_deps

        # Capture a real PhotonRAGPipeline without running its __init__
        # side effects — we only need to confirm MLX was loaded.
        class _FakePhotonPipeline:
            def __init__(self, cfg, baseline_deps, photon_deps):
                self.cfg = cfg
        pp.PhotonRAGPipeline = _FakePhotonPipeline

        pf.build_pipeline(_Cfg())

        # The branch imports photon_pipeline, but the module-level MLX proxy
        # keeps mlx.core unloaded until an actual MLX operation is requested.
        assert "baseline_reporag.photon_pipeline" in sys.modules
        assert "mlx.core" not in sys.modules, (
            "photon branch eagerly loaded mlx.core"
        )
        pp.mx.array([1])
        assert "mlx.core" in sys.modules
        print("OK")
        """
    )
    result = _run_subprocess(script)
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
