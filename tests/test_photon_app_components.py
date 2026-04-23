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
drift_panel = _load_component("drift_panel", COMPONENTS_DIR / "drift_panel.py")
turn_history_panel = _load_component(
    "turn_history_panel", COMPONENTS_DIR / "turn_history_panel.py"
)


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
            for name in (
                "__init__",
                "eval_panel",
                "wizard",
                "drift_panel",
                "turn_history_panel",
            ):
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


class TestBuildEvalJobCmd:
    """W4-T1: build_eval_job_cmd returns a shell=False argv list."""

    def test_static_argv_shape(self, tmp_path: Path) -> None:
        out_json = tmp_path / "reports" / "eval_runs" / "abc.json"
        marker = tmp_path / "reports" / "eval_runs" / "abc.done"
        argv = eval_panel.build_eval_job_cmd(
            eval_type="static",
            project_name="demo_proj",
            repo_id="demo_repo",
            config_path=str(tmp_path / "configs" / "photon_small.yaml"),
            output_json=out_json,
            marker_file=marker,
            python_exec="/usr/bin/python3",
        )
        assert argv[0] == "/usr/bin/python3"
        assert "-u" in argv
        assert "-m" in argv
        assert "scripts.run_baseline_eval" in argv
        assert "--config" in argv
        assert "--repo-id" in argv
        assert "demo_repo" in argv
        assert "--output" in argv
        assert str(out_json) in argv
        assert "--marker-file" in argv
        assert str(marker) in argv

    def test_multi_turn_uses_correct_module(self, tmp_path: Path) -> None:
        out_json = tmp_path / "reports" / "eval_runs" / "abc.json"
        marker = tmp_path / "reports" / "eval_runs" / "abc.done"
        argv = eval_panel.build_eval_job_cmd(
            eval_type="multi_turn",
            project_name="demo",
            repo_id="demo_repo",
            config_path=str(tmp_path / "c.yaml"),
            output_json=out_json,
            marker_file=marker,
        )
        assert "scripts.run_multi_turn_eval" in argv

    def test_unknown_eval_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            eval_panel.build_eval_job_cmd(
                eval_type="nope",
                project_name="demo",
                repo_id="demo_repo",
                config_path=str(tmp_path / "c.yaml"),
                output_json=tmp_path / "a.json",
                marker_file=tmp_path / "a.done",
            )

    def test_rejects_bad_project_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            eval_panel.build_eval_job_cmd(
                eval_type="static",
                project_name="foo bar",
                repo_id="demo_repo",
                config_path=str(tmp_path / "c.yaml"),
                output_json=tmp_path / "a.json",
                marker_file=tmp_path / "a.done",
            )

    def test_rejects_bad_repo_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            eval_panel.build_eval_job_cmd(
                eval_type="static",
                project_name="demo",
                repo_id="../repo",
                config_path=str(tmp_path / "c.yaml"),
                output_json=tmp_path / "a.json",
                marker_file=tmp_path / "a.done",
            )


class TestParseEvalProgress:
    """W4-T1: parse_eval_progress extracts the latest PROGRESS line."""

    def test_extracts_latest(self, tmp_path: Path) -> None:
        log = tmp_path / "eval.log"
        log.write_text(
            "starting run\n"
            "PROGRESS done=10 total=120 p50_ms=18000 nc=0.15\n"
            "more output\n"
            "PROGRESS done=25 total=120 p50_ms=19300 nc=0.183\n"
            "trailing log line\n"
        )
        result = eval_panel.parse_eval_progress(log)
        assert result["done_q"] == 25
        assert result["total_q"] == 120
        assert result["p50_latency_ms"] == 19300.0
        assert result["nc_rate"] == 0.183

    def test_empty_log_returns_empty_dict(self, tmp_path: Path) -> None:
        log = tmp_path / "eval.log"
        log.write_text("no progress here\n")
        assert eval_panel.parse_eval_progress(log) == {}

    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert eval_panel.parse_eval_progress(tmp_path / "nope.log") == {}


class TestTailLogBytes:
    """W4-T1: tail_log_bytes returns the trailing max_bytes of a log file."""

    def test_truncates_to_max_bytes(self, tmp_path: Path) -> None:
        log = tmp_path / "big.log"
        log.write_bytes(b"A" * 4096)
        tail = eval_panel.tail_log_bytes(log, 2048)
        assert len(tail) == 2048
        assert tail == "A" * 2048

    def test_short_file_returned_verbatim(self, tmp_path: Path) -> None:
        log = tmp_path / "small.log"
        log.write_text("oops\nfailed\n")
        assert eval_panel.tail_log_bytes(log, 2048) == "oops\nfailed\n"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert eval_panel.tail_log_bytes(tmp_path / "nope.log", 2048) == ""


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


# ---------------------------------------------------------------
# T-C1: classify_drift boundary conditions
# T-C2 / T-C3: format_drift_panel behaviour
# (Wave 3 — drift_panel.py)
# ---------------------------------------------------------------


class TestClassifyDrift:
    """T-C1: classify_drift maps (value, threshold) -> 'ok'|'warn'|'alert'."""

    def test_none_value_returns_ok(self) -> None:
        assert drift_panel.classify_drift(None, 0.5) == "ok"

    def test_none_threshold_returns_ok(self) -> None:
        assert drift_panel.classify_drift(0.3, None) == "ok"

    def test_below_warn_band_returns_ok(self) -> None:
        # 0.3 / 0.5 = 0.6 → below 80% band
        assert drift_panel.classify_drift(0.3, 0.5) == "ok"

    def test_in_warn_band_returns_warn(self) -> None:
        # 0.41 / 0.5 = 0.82 → above 80% band, below threshold
        assert drift_panel.classify_drift(0.41, 0.5) == "warn"

    def test_above_threshold_returns_alert(self) -> None:
        assert drift_panel.classify_drift(0.6, 0.5) == "alert"


class TestFormatDriftPanel:
    """T-C2 / T-C3: format_drift_panel derives rows from DriftMetrics dict."""

    def test_none_metrics_marks_unavailable(self) -> None:
        result = drift_panel.format_drift_panel(
            None,
            {
                "token_level": None,
                "mid_level": None,
                "top_level": 0.5,
                "topic_shift": 0.65,
            },
        )
        assert result["available"] is False
        assert "N/A" in result["reason"]
        assert result["rows"] == []
        assert result["safe_recgen_fired"] is False

    def test_empty_metrics_marks_unavailable(self) -> None:
        result = drift_panel.format_drift_panel(
            {},
            {
                "token_level": None,
                "mid_level": None,
                "top_level": 0.5,
                "topic_shift": 0.65,
            },
        )
        assert result["available"] is False

    def test_full_metrics_produce_ordered_rows(self) -> None:
        dm = {
            "latent_cosine_drift_token": 0.1,
            "latent_cosine_drift_mid": 0.3,
            "latent_cosine_drift_top": 0.52,
            "topic_shift_score": 0.66,
        }
        th = {
            "token_level": None,
            "mid_level": None,
            "top_level": 0.50,
            "topic_shift": 0.65,
        }
        result = drift_panel.format_drift_panel(dm, th)
        assert result["available"] is True
        assert len(result["rows"]) == 4
        assert [r["name"] for r in result["rows"]] == [
            "token_level",
            "mid_level",
            "top_level",
            "topic_shift",
        ]
        # Top-level level: 0.52 > 0.50 → alert
        assert result["rows"][2]["level"] == "alert"
        # Topic shift: 0.66 > 0.65 → alert
        assert result["rows"][3]["level"] == "alert"
        assert result["rows"][2]["badge"] == "⚠"
        # Token and mid have no threshold → ok regardless of value
        assert result["rows"][0]["level"] == "ok"
        assert result["rows"][1]["level"] == "ok"
        assert result["rows"][0]["value_str"] == "0.10"
        assert result["rows"][1]["value_str"] == "0.30"

    def test_missing_key_yields_em_dash(self) -> None:
        dm = {
            "latent_cosine_drift_token": 0.1,
            "latent_cosine_drift_mid": 0.3,
            "latent_cosine_drift_top": 0.52,
            # topic_shift_score deliberately omitted
        }
        th = {
            "token_level": None,
            "mid_level": None,
            "top_level": 0.50,
            "topic_shift": 0.65,
        }
        result = drift_panel.format_drift_panel(dm, th)
        assert result["available"] is True
        topic_row = result["rows"][3]
        assert topic_row["name"] == "topic_shift"
        assert topic_row["value"] is None
        assert topic_row["value_str"] == "—"
        assert topic_row["level"] == "ok"
        assert topic_row["badge"] == ""

    def test_safe_recgen_fired_propagates(self) -> None:
        dm = {
            "latent_cosine_drift_token": 0.0,
            "latent_cosine_drift_mid": 0.0,
            "latent_cosine_drift_top": 0.0,
            "topic_shift_score": 0.0,
            "safe_recgen_fired": True,
        }
        th = {
            "token_level": None,
            "mid_level": None,
            "top_level": None,
            "topic_shift": None,
        }
        result = drift_panel.format_drift_panel(dm, th)
        assert result["safe_recgen_fired"] is True


# ---------------------------------------------------------------
# T-C4: format_turn_history_panel
# (Wave 3 — turn_history_panel.py)
# ---------------------------------------------------------------


class TestFormatTurnHistoryPanel:
    """T-C4: turn_history_panel joins PhotonSessionState + SessionManager."""

    def test_working_memory_disabled_marks_unavailable(self) -> None:
        result = turn_history_panel.format_turn_history_panel(
            None,
            None,
            working_memory_enabled=False,
            max_turns=8,
        )
        assert result["available"] is False
        assert "working_memory disabled" in result["reason"]
        assert result["rows"] == []

    def test_baseline_rag_marks_unavailable(self) -> None:
        result = turn_history_panel.format_turn_history_panel(
            None,
            [],
            working_memory_enabled=True,
            max_turns=8,
        )
        assert result["available"] is False
        assert "baseline_rag" in result["reason"]

    def test_max_turns_truncates_to_last_n(self) -> None:
        from types import SimpleNamespace

        ph_list = [
            SimpleNamespace(turn_id=i, question_text=f"q{i}", timestamp=f"t{i}")
            for i in range(10)
        ]
        result = turn_history_panel.format_turn_history_panel(
            ph_list,
            [],
            working_memory_enabled=True,
            max_turns=3,
        )
        assert result["available"] is True
        assert len(result["rows"]) == 3
        assert [r.turn_id for r in result["rows"]] == [7, 8, 9]
        assert result["rows"][0].question_text == "q7"

    def test_join_by_turn_id_fills_cited_chunks(self) -> None:
        from types import SimpleNamespace

        ph_list = [
            SimpleNamespace(turn_id=1, question_text="q1", timestamp="t1"),
            SimpleNamespace(turn_id=2, question_text="q2", timestamp="t2"),
            SimpleNamespace(turn_id=3, question_text="q3", timestamp="t3"),
        ]
        sm_turns = [
            SimpleNamespace(turn_id=1, cited_chunk_ids=["C:1"]),
            # turn 2 deliberately missing
            SimpleNamespace(turn_id=3, cited_chunk_ids=["C:5"]),
        ]
        result = turn_history_panel.format_turn_history_panel(
            ph_list,
            sm_turns,
            working_memory_enabled=True,
            max_turns=8,
        )
        assert result["available"] is True
        assert len(result["rows"]) == 3
        assert result["rows"][0].cited_chunk_ids == ["C:1"]
        assert result["rows"][1].cited_chunk_ids == []
        assert result["rows"][2].cited_chunk_ids == ["C:5"]

    def test_empty_history_is_available_empty_rows(self) -> None:
        result = turn_history_panel.format_turn_history_panel(
            [],
            [],
            working_memory_enabled=True,
            max_turns=8,
        )
        assert result["available"] is True
        assert result["reason"] == ""
        assert result["rows"] == []


# ---------------------------------------------------------------
# T-C5: apply_best_practice (Wave 5 — wizard.py)
# ---------------------------------------------------------------


class TestApplyBestPractice:
    """T-C5: apply_best_practice merges 5 keys with profile-aware warnings."""

    def test_photon_small_no_changes(self) -> None:
        # photon_small already has all 5 best-practice values → no warnings
        yaml_text = (
            "inference:\n"
            "  photon_generation_enabled: false\n"
            "retrieval:\n"
            "  two_pass_search:\n"
            "    enabled: false\n"
            "session_memory:\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "safe_recgen:\n"
            "  enabled: true\n"
            "generation:\n"
            "  evidence_pruning_enabled: true\n"
        )
        _new_text, warnings = wizard.apply_best_practice(yaml_text, "photon_small")
        assert warnings == []

    def test_photon_long_context_creates_missing_two_pass_section(self) -> None:
        yaml_text = (
            "inference:\n"
            "  photon_generation_enabled: false\n"
            "session_memory:\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "safe_recgen:\n"
            "  enabled: true\n"
            "generation:\n"
            "  evidence_pruning_enabled: true\n"
        )
        new_text, warnings = wizard.apply_best_practice(
            yaml_text, "photon_long_context"
        )
        assert any("retrieval.two_pass_search.enabled" in w for w in warnings)
        loaded = yaml.safe_load(new_text)
        assert loaded["retrieval"]["two_pass_search"]["enabled"] is False

    def test_photon_tiny_recgen_warns_on_conflict(self) -> None:
        # photon_tiny_recgen has photon_generation_enabled: true (intentional)
        # and working_memory deliberately omitted.
        yaml_text = (
            "inference:\n"
            "  photon_generation_enabled: true\n"
            "retrieval:\n"
            "  two_pass_search:\n"
            "    enabled: false\n"
            "safe_recgen:\n"
            "  enabled: true\n"
            "generation:\n"
            "  evidence_pruning_enabled: true\n"
        )
        new_text, warnings = wizard.apply_best_practice(yaml_text, "photon_tiny_recgen")
        # Expect conflict warning for photon_generation_enabled (was True,
        # target False, profile is intentional-conflict) and an additive
        # warning for the missing working_memory.enabled path.
        assert any("photon_generation_enabled" in w for w in warnings)
        assert any("session_memory.working_memory.enabled" in w for w in warnings)
        loaded = yaml.safe_load(new_text)
        assert loaded["inference"]["photon_generation_enabled"] is False
        assert loaded["session_memory"]["working_memory"]["enabled"] is True

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(ValueError):
            wizard.apply_best_practice("- a\n- b\n", "photon_small")

    def test_round_trip_preserves_other_keys(self) -> None:
        yaml_text = (
            "version: 1\n"
            "project:\n"
            "  name: demo\n"
            "safe_recgen:\n"
            "  enabled: true\n"
            "generation:\n"
            "  evidence_pruning_enabled: true\n"
            "session_memory:\n"
            "  working_memory:\n"
            "    enabled: true\n"
            "inference:\n"
            "  photon_generation_enabled: false\n"
            "retrieval:\n"
            "  two_pass_search:\n"
            "    enabled: false\n"
        )
        new_text, _warnings = wizard.apply_best_practice(yaml_text, "photon_small")
        loaded = yaml.safe_load(new_text)
        assert loaded["version"] == 1
        assert loaded["project"]["name"] == "demo"


# ---------------------------------------------------------------
# T-C6: generate_yaml_from_wizard (Wave 5 — wizard.py)
# ---------------------------------------------------------------


class TestGenerateYamlFromWizard:
    """T-C6: generate_yaml_from_wizard applies toggles + validates fallback."""

    def test_invalid_fallback_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="fallback_policy"):
            wizard.generate_yaml_from_wizard(
                "photon_small",
                {"fallback_policy": "invalid"},
                base_yaml_text="model: {}\n",
            )

    def test_accepts_qwen_and_abort(self) -> None:
        for policy in ("qwen", "abort"):
            result = wizard.generate_yaml_from_wizard(
                "photon_small",
                {"fallback_policy": policy},
                base_yaml_text="model: {}\n",
            )
            loaded = yaml.safe_load(result)
            assert loaded["inference"]["generation_fallback_policy"] == policy

    def test_applies_toggles_to_base(self) -> None:
        base = "model:\n  architecture: photon_decoder\n"
        result = wizard.generate_yaml_from_wizard(
            "photon_small",
            {
                "recgen_enabled": True,
                "two_pass_search_enabled": True,
                "two_pass_pass1_top_k": 32,
            },
            base_yaml_text=base,
        )
        loaded = yaml.safe_load(result)
        assert loaded["inference"]["photon_generation_enabled"] is True
        assert loaded["retrieval"]["two_pass_search"]["enabled"] is True
        assert loaded["retrieval"]["two_pass_search"]["pass1_top_k"] == 32
        # base key preserved
        assert loaded["model"]["architecture"] == "photon_decoder"

    def test_working_memory_toggles_all_applied(self) -> None:
        result = wizard.generate_yaml_from_wizard(
            "photon_small",
            {
                "working_memory_enabled": True,
                "working_memory_max_turns": 12,
                "working_memory_aggregation": "attention",
                "working_memory_storage_mode": "top_level_only",
                "past_turn_pinning_enabled": True,
            },
            base_yaml_text="model: {}\n",
        )
        loaded = yaml.safe_load(result)
        wm = loaded["session_memory"]["working_memory"]
        assert wm["enabled"] is True
        assert wm["max_turns"] == 12
        assert wm["aggregation"] == "attention"
        assert wm["storage_mode"] == "top_level_only"
        assert wm["past_turn_pinning_enabled"] is True

    def test_ignores_unknown_toggles(self) -> None:
        # Unknown keys must not raise so the UI can feed a full form dict.
        result = wizard.generate_yaml_from_wizard(
            "photon_small",
            {"recgen_enabled": True, "ignored_extra": "whatever"},
            base_yaml_text="model: {}\n",
        )
        loaded = yaml.safe_load(result)
        assert loaded["inference"]["photon_generation_enabled"] is True
        assert "ignored_extra" not in loaded


if __name__ == "__main__":  # pragma: no cover - manual run only
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
