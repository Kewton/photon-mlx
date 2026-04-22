"""Pure-function tests for the graph-expansion benchmark core (Issue #91)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._graph_bench_core import (  # noqa: E402
    GraphBenchParams,
    generate_graph_bench_grid,
    generate_graph_bench_phase2,
    is_invalid_graph_combo,
)


def _default_params(**kwargs) -> GraphBenchParams:
    defaults = dict(
        max_hops=1,
        max_nodes=24,
        neighborhood_before=1,
        neighborhood_after=1,
        edge_weights_call=1.0,
        edge_weights_inherit=0.8,
        edge_weights_import=0.5,
        adaptive_neighborhood=False,
    )
    defaults.update(kwargs)
    return GraphBenchParams(**defaults)


def test_generate_grid_produces_unique_configs() -> None:
    grid = generate_graph_bench_grid()
    assert len(grid) > 0
    keys = {json.dumps(p.to_override_dict(), sort_keys=True) for p in grid}
    assert len(keys) == len(grid), "grid contains duplicate configs"


def test_is_invalid_graph_combo_rules() -> None:
    # Rule 1: max_nodes < max_hops * 8
    assert is_invalid_graph_combo(_default_params(max_hops=2, max_nodes=8)) is True
    assert is_invalid_graph_combo(_default_params(max_hops=2, max_nodes=24)) is False

    # Rule 2: all zero disables expansion entirely
    bad = _default_params(
        max_hops=0,
        neighborhood_before=0,
        neighborhood_after=0,
        max_nodes=24,
    )
    assert is_invalid_graph_combo(bad) is True

    # Normal combo: valid
    assert is_invalid_graph_combo(_default_params()) is False


def test_to_override_dict_shape() -> None:
    p = _default_params(
        max_hops=2,
        max_nodes=32,
        neighborhood_before=2,
        neighborhood_after=2,
        edge_weights_call=1.0,
        edge_weights_inherit=0.5,
        edge_weights_import=0.0,
        adaptive_neighborhood=True,
    )
    d = p.to_override_dict()
    assert "graph_expansion" in d
    assert "neighborhood_expansion" in d
    ge = d["graph_expansion"]
    assert ge["max_hops"] == 2
    assert ge["max_nodes"] == 32
    assert ge["edge_weights"]["call"] == 1.0
    assert ge["edge_weights"]["inherit"] == 0.5
    assert ge["edge_weights"]["import"] == 0.0
    ne = d["neighborhood_expansion"]
    assert ne["before"] == 2
    assert ne["after"] == 2
    assert ne["adaptive"] is True


def test_phase2_narrows_around_seeds() -> None:
    seed = _default_params(max_hops=1, max_nodes=24)
    phase2 = generate_graph_bench_phase2([seed])
    assert len(phase2) > 0
    # Every phase-2 config is close to the seed on at least the integer axes.
    for p in phase2:
        # Near-seed: allow offsets of at most +/- 1 hop and +/- 8 nodes and +/- 1 before/after
        assert abs(p.max_hops - seed.max_hops) <= 1
        assert abs(p.max_nodes - seed.max_nodes) <= 8
        assert abs(p.neighborhood_before - seed.neighborhood_before) <= 1
        assert abs(p.neighborhood_after - seed.neighborhood_after) <= 1
    # Phase 2 must not include the seed itself.
    seed_key = json.dumps(seed.to_override_dict(), sort_keys=True)
    assert seed_key not in {
        json.dumps(p.to_override_dict(), sort_keys=True) for p in phase2
    }


def test_params_is_hashable() -> None:
    p1 = _default_params()
    p2 = _default_params()
    s = {p1, p2}
    assert len(s) == 1


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
