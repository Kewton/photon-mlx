"""Unit tests for ``app/components/`` (Issue #82 Wave 2).

These tests pin the security invariants documented in the design
policy §6.4–§6.5 and §8:

* ``T-C-streamlit-absent``: every module in ``app/components/`` is
  streamlit-free, even when imported into a fresh Python process.
* ``T-C7``: ``_safe_id`` on project-name inputs rejects traversal,
  metacharacters, and the empty string while accepting the allowlist.
* ``T-C8``: YAML loading rejects ``!!python/object``-style injection
  and the allowlist ``_assert_safe_yaml`` accepts normal trees.
* ``T-C9``: ``sanitize_job_id`` / ``make_eval_paths`` constrain eval
  artifacts to ``reports/eval_runs/`` and ``logs/eval/``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS_DIR = PROJECT_ROOT / "app" / "components"


def _load_component(mod_name: str, path: Path):
    """Load ``app/components/<mod_name>.py`` under a private module name.

    Using a direct file import avoids the namespace-package collision
    with the ``app`` directory of neighbouring projects on ``sys.path``
    (our test environment has another ``app`` under ``MySwiftAgent``).
    """

    full_name = f"_photon_components_{mod_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


eval_panel = _load_component("eval_panel", COMPONENTS_DIR / "eval_panel.py")
wizard = _load_component("wizard", COMPONENTS_DIR / "wizard.py")


# ---------------------------------------------------------------
# T-C-streamlit-absent: components/ must not import streamlit at
# module scope. Uses a fresh subprocess so the main test runner's
# imports (which may pull in streamlit transitively via conftest)
# cannot pollute the check.
# ---------------------------------------------------------------


class TestComponentsStreamlitAbsent:
    """Guardrail: no module in ``app/components/`` may import streamlit."""

    def test_subprocess_import_does_not_pull_streamlit(self) -> None:
        # Import each component module by absolute file path to bypass any
        # ``app``-named namespace collisions on the subprocess sys.path.
        components_dir = str(COMPONENTS_DIR)
        script = textwrap.dedent(
            f"""
            import importlib.util
            import sys
            from pathlib import Path
            root = Path({components_dir!r})
            for name in ("__init__", "eval_panel", "wizard"):
                spec = importlib.util.spec_from_file_location(
                    f"_c_{{name}}", root / f"{{name}}.py"
                )
                assert spec and spec.loader
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
            leaked = sorted(m for m in sys.modules if m.startswith("streamlit"))
            assert not leaked, leaked
            """
        )
        # Run in a cold interpreter so we are not contaminated by fixtures.
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


# ---------------------------------------------------------------
# T-C7: _safe_id on project_name
# ---------------------------------------------------------------


# Lazy-import app/photon_app.py without pulling streamlit for every
# test in this module — only the _safe_id helper is needed.
def _load_safe_id():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "photon_app_for_components_test",
        PROJECT_ROOT / "app" / "photon_app.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._safe_id


_safe_id = _load_safe_id()


class TestSafeIdProjectName:
    """T-C7: project_name inputs rejected/accepted by _safe_id."""

    @pytest.mark.parametrize(
        "bad",
        [
            "../foo",
            "foo bar",
            "",
            "foo/bar",
            "foo\\bar",
            "foo.bar",
        ],
    )
    def test_rejects_bad_names(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _safe_id(bad, label="project_name")

    @pytest.mark.parametrize(
        "good",
        [
            "valid",
            "valid-name_01",
            "ABC123",
        ],
    )
    def test_accepts_allowlist_names(self, good: str) -> None:
        assert _safe_id(good, label="project_name") == good


# ---------------------------------------------------------------
# T-C8: YAML safety
# ---------------------------------------------------------------


class TestAssertSafeYaml:
    """T-C8: unsafe YAML is rejected by safe_load or _assert_safe_yaml."""

    def test_python_object_apply_is_blocked(self) -> None:
        """!!python/object/apply:os.system must not materialize a callable.

        ``yaml.safe_load`` rejects the tag with a ConstructorError. If a
        future PyYAML version silenced that error, the resulting tree
        would still fail the allowlist check in ``_assert_safe_yaml``.
        Either line of defense is acceptable for the security contract.
        """

        payload = "!!python/object/apply:os.system ['echo pwned']\n"
        constructor_err = yaml.constructor.ConstructorError
        try:
            loaded = yaml.safe_load(payload)
        except constructor_err:
            return  # First line of defense held — done.
        # Unlikely fall-through path: allowlist must catch it instead.
        with pytest.raises(ValueError):
            wizard._assert_safe_yaml(loaded)

    def test_tuple_value_rejected_by_allowlist(self) -> None:
        """Crafted dict with a tuple (non-allowed type) → ValueError."""

        with pytest.raises(ValueError):
            wizard._assert_safe_yaml({"foo": (1, 2, 3)})

    def test_normal_nested_tree_is_accepted(self) -> None:
        tree = {
            "safe_recgen": {"enabled": True, "thresholds": [0.1, 0.2, 0.3]},
            "session_memory": {
                "working_memory": {
                    "enabled": False,
                    "max_turns": 8,
                    "aggregation": "mean",
                }
            },
            "retrieval": {"two_pass_search": None},
        }
        # Must not raise.
        wizard._assert_safe_yaml(tree)

    def test_bytes_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            wizard._assert_safe_yaml({"foo": b"bytes-not-allowed"})


# ---------------------------------------------------------------
# T-C9: sanitize_job_id + make_eval_paths
# ---------------------------------------------------------------


class TestSanitizeJobId:
    def test_traversal_rejected(self) -> None:
        with pytest.raises(ValueError):
            eval_panel.sanitize_job_id("../../etc/passwd")

    def test_plain_hex_accepted(self) -> None:
        assert eval_panel.sanitize_job_id("validhexabc123") == "validhexabc123"

    def test_underscore_rejected(self) -> None:
        # The stricter _SAFE_JOB_ID_RE disallows underscore so a tampered
        # state file cannot smuggle in characters that would pass repo_id's
        # broader allowlist.
        with pytest.raises(ValueError):
            eval_panel.sanitize_job_id("valid_hex_abc123")

    def test_none_returns_uuid_hex(self) -> None:
        job_id = eval_panel.sanitize_job_id()
        assert len(job_id) == 32
        assert all(c in "0123456789abcdef" for c in job_id)
        # Two draws should differ with overwhelming probability.
        assert job_id != eval_panel.sanitize_job_id()

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            eval_panel.sanitize_job_id("")


class TestMakeEvalPaths:
    def test_paths_under_allowed_dirs(self, tmp_path: Path) -> None:
        result_json, log_file, marker_file = eval_panel.make_eval_paths(
            "abc123", tmp_path
        )
        root = tmp_path.resolve()
        assert result_json.is_relative_to(root / "reports" / "eval_runs")
        assert log_file.is_relative_to(root / "logs" / "eval")
        assert marker_file.is_relative_to(root / "reports" / "eval_runs")
        # Concrete suffixes:
        assert result_json.name == "abc123.json"
        assert log_file.name == "abc123.log"
        assert marker_file.name == "abc123.done"

    def test_escape_via_bad_job_id_raises(self, tmp_path: Path) -> None:
        # Defense-in-depth: make_eval_paths re-validates the job_id so a
        # caller that forgot sanitize_job_id still cannot escape.
        with pytest.raises(ValueError):
            eval_panel.make_eval_paths("../x", tmp_path)

    def test_uuid_roundtrip(self, tmp_path: Path) -> None:
        job_id = eval_panel.sanitize_job_id()
        result_json, log_file, marker_file = eval_panel.make_eval_paths(
            job_id, tmp_path
        )
        # All three paths must live under project_root.
        root = tmp_path.resolve()
        for p in (result_json, log_file, marker_file):
            assert p.is_relative_to(root)


if __name__ == "__main__":  # pragma: no cover - manual run only
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
