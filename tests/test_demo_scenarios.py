"""Tests for the demo scenarios module.

A-1 Phase 2 Step 3: smoke / structural tests for ``demo/scenarios.py`` and
``demo/run_demo.py``. Full end-to-end demo runs (LLM-bound, several minutes
per scenario) are deliberately out of scope — those should be exercised
manually via ``make demo SCENARIO=demo-01`` against an ingested repo.

What we verify here is:

- ``SCENARIOS`` is a non-empty list of well-formed ``DemoScenario`` items.
- ``demo-01`` through ``demo-05`` are present (catches accidental rename / drop).
- Every scenario has at least one turn with a non-empty question.
- The ``run_demo`` script's ``--list`` code path works and prints all ids.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_demo_scenarios():
    """Import demo.scenarios via importlib (demo/ is not a package on its own)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from demo.scenarios import SCENARIOS, DemoScenario, DemoTurn

        return SCENARIOS, DemoScenario, DemoTurn
    finally:
        if str(REPO_ROOT) in sys.path:
            sys.path.remove(str(REPO_ROOT))


class TestScenariosCatalog:
    def test_scenarios_is_non_empty_list(self) -> None:
        SCENARIOS, _DemoScenario, _DemoTurn = _import_demo_scenarios()
        assert isinstance(SCENARIOS, list)
        assert len(SCENARIOS) >= 5, "expect at least 5 demo scenarios"

    def test_required_scenario_ids_present(self) -> None:
        SCENARIOS, _DemoScenario, _DemoTurn = _import_demo_scenarios()
        ids = {s.id for s in SCENARIOS}
        for required in ("demo-01", "demo-02", "demo-03", "demo-04", "demo-05"):
            assert required in ids, f"missing scenario {required}"

    def test_every_scenario_has_at_least_one_non_empty_turn(self) -> None:
        SCENARIOS, _DemoScenario, _DemoTurn = _import_demo_scenarios()
        for scenario in SCENARIOS:
            assert scenario.turns, f"{scenario.id} has no turns"
            for turn in scenario.turns:
                assert turn.question.strip(), f"{scenario.id}: empty question detected"

    def test_scenario_has_title_and_axis(self) -> None:
        SCENARIOS, _DemoScenario, _DemoTurn = _import_demo_scenarios()
        for scenario in SCENARIOS:
            assert scenario.title.strip(), f"{scenario.id}: empty title"
            assert scenario.axis.strip(), f"{scenario.id}: empty axis"


class TestRunDemoListMode:
    """Run ``demo/run_demo.py --list`` as a subprocess and verify the output."""

    def test_list_mode_prints_all_scenario_ids(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "demo" / "run_demo.py"), "--list"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        assert result.returncode == 0, f"--list exited non-zero. stderr={result.stderr}"
        for scenario_id in ("demo-01", "demo-02", "demo-03", "demo-04", "demo-05"):
            assert scenario_id in result.stdout, f"--list output missing {scenario_id}"

    def test_no_args_prints_scenarios(self) -> None:
        """`run_demo.py` without arguments falls through to print_scenarios()."""
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "demo" / "run_demo.py")],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        assert result.returncode == 0
        assert "demo-01" in result.stdout

    def test_unknown_scenario_reports_available(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "demo" / "run_demo.py"),
                "--scenario",
                "demo-nonexistent",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        # Script prints message and returns normally (no SystemExit).
        assert result.returncode == 0
        assert "Unknown scenario" in result.stdout
