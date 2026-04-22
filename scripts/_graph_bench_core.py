"""Pure-function core for the graph-expansion grid search (Issue #91).

MLX-free so unit tests can run on a minimal CI runner. Provides:

- ``GraphBenchParams`` frozen dataclass (hashable, ``to_override_dict``)
- ``is_invalid_graph_combo`` (design §4.3 sanity rules)
- ``generate_graph_bench_grid`` (phase 1 broad grid)
- ``generate_graph_bench_phase2`` (narrow neighborhood around seeds)
- ``atomic_write_json`` is re-exported from ``scripts._grid_search_core``
  so callers do not need to import both modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from scripts._grid_search_core import atomic_write_json  # noqa: F401  (re-export)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphBenchParams:
    """Immutable + hashable sweep target for graph/neighborhood axes."""

    max_hops: int
    max_nodes: int
    neighborhood_before: int
    neighborhood_after: int
    edge_weights_call: float
    edge_weights_inherit: float
    edge_weights_import: float
    adaptive_neighborhood: bool

    def to_override_dict(self) -> dict[str, Any]:
        return {
            "graph_expansion": {
                "max_hops": self.max_hops,
                "max_nodes": self.max_nodes,
                "edge_weights": {
                    "call": self.edge_weights_call,
                    "inherit": self.edge_weights_inherit,
                    "import": self.edge_weights_import,
                },
            },
            "neighborhood_expansion": {
                "before": self.neighborhood_before,
                "after": self.neighborhood_after,
                "adaptive": self.adaptive_neighborhood,
            },
        }


# ---------------------------------------------------------------------------
# Invalid combo filter
# ---------------------------------------------------------------------------


def is_invalid_graph_combo(p: GraphBenchParams) -> bool:
    """Return True when ``p`` violates a sanity rule (design §4.3)."""
    if p.max_nodes < p.max_hops * 8:
        return True
    if p.neighborhood_before == 0 and p.neighborhood_after == 0 and p.max_hops == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------


_PHASE1_MAX_HOPS: tuple[int, ...] = (1, 2)
_PHASE1_MAX_NODES: tuple[int, ...] = (24, 32, 48)
_PHASE1_NBH: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (3, 3))
_PHASE1_EDGE_WEIGHTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 1.0, 1.0),  # equal
    (1.0, 0.8, 0.5),  # weighted (default)
    (1.0, 0.0, 0.0),  # call-only
)
_PHASE1_ADAPTIVE: tuple[bool, ...] = (False, True)


def generate_graph_bench_grid() -> list[GraphBenchParams]:
    """Return the phase-1 grid after dropping invalid combos + duplicates."""
    seen: set[str] = set()
    out: list[GraphBenchParams] = []
    for max_hops in _PHASE1_MAX_HOPS:
        for max_nodes in _PHASE1_MAX_NODES:
            for before, after in _PHASE1_NBH:
                for w_call, w_inh, w_imp in _PHASE1_EDGE_WEIGHTS:
                    for adaptive in _PHASE1_ADAPTIVE:
                        params = GraphBenchParams(
                            max_hops=max_hops,
                            max_nodes=max_nodes,
                            neighborhood_before=before,
                            neighborhood_after=after,
                            edge_weights_call=w_call,
                            edge_weights_inherit=w_inh,
                            edge_weights_import=w_imp,
                            adaptive_neighborhood=adaptive,
                        )
                        if is_invalid_graph_combo(params):
                            continue
                        key = _key(params)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(params)
    return out


def _key(params: GraphBenchParams) -> str:
    return json.dumps(params.to_override_dict(), sort_keys=True)


def generate_graph_bench_phase2(
    seeds: list[GraphBenchParams],
    *,
    hops_delta: int = 1,
    nodes_step: int = 8,
    nodes_delta: int = 8,
    nbh_delta: int = 1,
) -> list[GraphBenchParams]:
    """Generate phase-2 near-seed neighborhoods, de-duped across seeds.

    The seeds themselves are excluded from the output; ``is_invalid_graph_combo``
    is applied to each candidate and adaptive / edge-weight axes are inherited
    from the seed unchanged (narrow phase 2 keeps the categorical knobs fixed).
    """
    if not seeds:
        return []

    seed_keys = {_key(s) for s in seeds}
    seen: set[str] = set(seed_keys)
    out: list[GraphBenchParams] = []

    hops_offsets = range(-hops_delta, hops_delta + 1)
    node_offsets = [d for d in range(-nodes_delta, nodes_delta + 1, nodes_step) if True]
    nbh_offsets = range(-nbh_delta, nbh_delta + 1)

    for seed in seeds:
        for d_hops in hops_offsets:
            for d_nodes in node_offsets:
                for d_before in nbh_offsets:
                    for d_after in nbh_offsets:
                        cand = GraphBenchParams(
                            max_hops=max(0, seed.max_hops + d_hops),
                            max_nodes=max(1, seed.max_nodes + d_nodes),
                            neighborhood_before=max(
                                0, seed.neighborhood_before + d_before
                            ),
                            neighborhood_after=max(
                                0, seed.neighborhood_after + d_after
                            ),
                            edge_weights_call=seed.edge_weights_call,
                            edge_weights_inherit=seed.edge_weights_inherit,
                            edge_weights_import=seed.edge_weights_import,
                            adaptive_neighborhood=seed.adaptive_neighborhood,
                        )
                        if is_invalid_graph_combo(cand):
                            continue
                        k = _key(cand)
                        if k in seen:
                            continue
                        seen.add(k)
                        out.append(cand)
    return out
