"""baseline_reporag.indexing — graph and index helpers.

Public API:
    load_active_graph(cfg, idx_dir) -> GraphLike | None
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import is_heading_graph_enabled, is_symbol_graph_enabled
from .graph_protocol import GraphLike

if TYPE_CHECKING:
    from ..config import Config


def load_active_graph(
    cfg: "Config | dict[str, Any]", idx_dir: Path
) -> GraphLike | None:
    """Load the active graph based on config priority (heading > symbol).

    LSP contract (DR1-004 / DR2-006):
    - heading_graph takes priority over symbol_graph when both are enabled.
    - Returns None when both are disabled.
    - Raises FileNotFoundError if enabled but the JSON file is absent
      (fail-fast: silent None fallback is intentionally NOT used).

    Args:
        cfg: Config instance or plain dict.
        idx_dir: Directory containing *_graph.json files.

    Returns:
        A HeadingGraph, SymbolGraph, or None.

    Raises:
        FileNotFoundError: If an enabled graph's JSON file does not exist.
    """
    if is_heading_graph_enabled(cfg):
        from .heading_graph import HeadingGraph

        return HeadingGraph.load(idx_dir / "heading_graph.json")
    if is_symbol_graph_enabled(cfg):
        from .symbol_graph import SymbolGraph

        return SymbolGraph.load(idx_dir / "symbol_graph.json")
    return None


__all__ = ["GraphLike", "load_active_graph"]
