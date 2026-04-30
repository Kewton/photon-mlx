"""Unit tests for HeadingGraph — 2-dict KISS design (DR1-003)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from baseline_reporag.indexing.heading_graph import HeadingGraph


def _make_chunk(chunk_id: str, section_header: str, language: str = "markdown"):
    c = MagicMock()
    c.chunk_id = chunk_id
    c.section_header = section_header
    c.language = language
    return c


def _make_store(*chunks):
    store = MagicMock()
    store.iter_repo.return_value = iter(chunks)
    return store


class TestHeadingGraphBuild:
    def test_sections_dict_built(self) -> None:
        """build populates _sections with path → chunk_ids mapping."""
        c1 = _make_chunk("id1", "Intro > Overview")
        c2 = _make_chunk("id2", "Intro > Overview")
        store = _make_store(c1, c2)
        hg = HeadingGraph()
        hg.build(store, "repo", "abc")
        assert hg._sections["Intro > Overview"] == ["id1", "id2"]

    def test_non_markdown_skipped(self) -> None:
        """Chunks with language != 'markdown' are skipped (DR2-008)."""
        c1 = _make_chunk("py_id", "Intro", language="python")
        c2 = _make_chunk("md_id", "Intro", language="markdown")
        store = _make_store(c1, c2)
        hg = HeadingGraph()
        hg.build(store, "repo", "abc")
        assert "py_id" not in hg._chunk_to_path
        assert "md_id" in hg._chunk_to_path

    def test_empty_section_header_skipped(self) -> None:
        """Chunks with empty section_header are skipped."""
        c = _make_chunk("no_header", "")
        store = _make_store(c)
        hg = HeadingGraph()
        hg.build(store, "repo", "abc")
        assert "no_header" not in hg._chunk_to_path

    def test_chunk_to_path_populated(self) -> None:
        """_chunk_to_path maps chunk_id to its section_header."""
        c = _make_chunk("id1", "Chapter 1 > Section A")
        store = _make_store(c)
        hg = HeadingGraph()
        hg.build(store, "repo", "abc")
        assert hg._chunk_to_path["id1"] == "Chapter 1 > Section A"


class TestHeadingGraphGetRelatedChunks:
    def _build_simple(self) -> HeadingGraph:
        """
        Structure:
          Chapter 1          -> [c_parent]
          Chapter 1 > Sec A  -> [c1, c2, c3]
          Chapter 2          -> [c_other]
        """
        chunks = [
            _make_chunk("c_parent", "Chapter 1"),
            _make_chunk("c1", "Chapter 1 > Sec A"),
            _make_chunk("c2", "Chapter 1 > Sec A"),
            _make_chunk("c3", "Chapter 1 > Sec A"),
            _make_chunk("c_other", "Chapter 2"),
        ]
        store = _make_store(*chunks)
        hg = HeadingGraph()
        hg.build(store, "repo", "abc")
        return hg

    def test_returns_parent_and_siblings(self) -> None:
        """get_related_chunks returns parent-level chunks + siblings (excl. self)."""
        hg = self._build_simple()
        related = hg.get_related_chunks("c1")
        # c_parent is in parent section "Chapter 1"
        assert "c_parent" in related
        # c2 and c3 are siblings at same path
        assert "c2" in related
        assert "c3" in related

    def test_self_excluded(self) -> None:
        """The queried chunk_id itself must never appear in the result (DR1-002 §2)."""
        hg = self._build_simple()
        related = hg.get_related_chunks("c1")
        assert "c1" not in related

    def test_unrelated_chunks_excluded(self) -> None:
        """Chunks from unrelated sections must not appear."""
        hg = self._build_simple()
        related = hg.get_related_chunks("c1")
        assert "c_other" not in related

    def test_unknown_chunk_id_returns_empty(self) -> None:
        """Unknown chunk_id must return [] without raising (DR1-002 §1)."""
        hg = self._build_simple()
        assert hg.get_related_chunks("nonexistent") == []

    def test_return_order_deterministic(self) -> None:
        """Same query must return the same order (DR1-002 §4)."""
        hg = self._build_simple()
        r1 = hg.get_related_chunks("c1")
        r2 = hg.get_related_chunks("c1")
        assert r1 == r2

    def test_top_level_section_no_parent(self) -> None:
        """Top-level section has no parent; only siblings returned."""
        chunks = [
            _make_chunk("a", "Chapter 1"),
            _make_chunk("b", "Chapter 1"),
        ]
        hg = HeadingGraph()
        hg.build(_make_store(*chunks), "repo", "abc")
        related = hg.get_related_chunks("a")
        assert related == ["b"]
        assert "a" not in related


class TestHeadingGraphSaveLoad:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """save → load preserves dict equality (atomic write: tmp + os.replace)."""
        c1 = _make_chunk("id1", "Chapter 1 > Sec A")
        c2 = _make_chunk("id2", "Chapter 1 > Sec A")
        store = _make_store(c1, c2)
        hg = HeadingGraph()
        hg.build(store, "repo", "abc")
        path = tmp_path / "heading_graph.json"
        hg.save(path)
        hg2 = HeadingGraph.load(path)
        assert dict(hg2._sections) == dict(hg._sections)
        assert hg2._chunk_to_path == hg._chunk_to_path

    def test_load_rejects_missing_key(self, tmp_path: Path) -> None:
        """load raises ValueError if required top-level keys are absent (DR4-001)."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"sections": {}}), encoding="utf-8")
        with pytest.raises(ValueError):
            HeadingGraph.load(path)

    def test_load_rejects_oversized_file(self, tmp_path: Path) -> None:
        """load raises ValueError if file exceeds 100 MB size limit (DR4-001)."""
        path = tmp_path / "big.json"
        path.write_text(
            json.dumps({"sections": {}, "chunk_to_path": {}}), encoding="utf-8"
        )
        original_stat = os.stat(path).st_size
        with pytest.raises(ValueError):
            # Monkey-patch: temporarily raise if file too large
            import baseline_reporag.indexing.heading_graph as hg_mod

            original_limit = hg_mod._MAX_FILE_BYTES
            hg_mod._MAX_FILE_BYTES = original_stat - 1
            try:
                HeadingGraph.load(path)
            finally:
                hg_mod._MAX_FILE_BYTES = original_limit

    def test_load_rejects_invalid_structure(self, tmp_path: Path) -> None:
        """load raises ValueError if sections value is not a list (DR4-001)."""
        path = tmp_path / "bad_struct.json"
        path.write_text(
            json.dumps({"sections": {"Ch1": "not_a_list"}, "chunk_to_path": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            HeadingGraph.load(path)

    def test_load_rejects_non_str_list_item(self, tmp_path: Path) -> None:
        """load raises ValueError if a chunk_id in a sections list is not a str (DR4-001)."""
        path = tmp_path / "bad_item.json"
        path.write_text(
            json.dumps({"sections": {"Ch1": [123]}, "chunk_to_path": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            HeadingGraph.load(path)
