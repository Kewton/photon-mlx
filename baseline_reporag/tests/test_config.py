from __future__ import annotations

import pytest

from baseline_reporag.config import Config, deep_merge, is_symbol_graph_enabled


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_override(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99}, "b": 3}

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_deep_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 99, "d": 2}}}

    def test_empty_override(self):
        base = {"a": 1}
        result = deep_merge(base, {})
        assert result == {"a": 1}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        deep_merge(base, override)
        assert base == {"a": {"x": 1}}


class TestConfigMergeOverride:
    def test_merge_session_memory_mode(self):
        base_data = {
            "model": {"provider": "mlx_lm"},
            "session_memory": {"mode": "flat_recent", "max_turns": 8},
        }
        override = {"session_memory": {"mode": "summary_pinned"}}
        cfg = Config(base_data)
        merged = cfg.merge_override(override)
        assert merged.session_memory.mode == "summary_pinned"
        assert merged.session_memory.max_turns == 8

    def test_merge_preserves_original(self):
        base_data = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        cfg = Config(base_data)
        cfg.merge_override(override)
        assert cfg.a.x == 1

    def test_merge_returns_new_config(self):
        base_data = {"a": 1}
        override = {"a": 2}
        cfg = Config(base_data)
        merged = cfg.merge_override(override)
        assert isinstance(merged, Config)
        assert merged is not cfg

    def test_merge_empty_override(self):
        base_data = {"a": 1, "b": 2}
        cfg = Config(base_data)
        merged = cfg.merge_override({})
        assert merged.a == 1
        assert merged.b == 2


class TestConfigToDict:
    def test_flat(self):
        cfg = Config({"a": 1, "b": "hello"})
        d = cfg.to_dict()
        assert d == {"a": 1, "b": "hello"}

    def test_nested(self):
        cfg = Config({"a": {"x": 1, "y": 2}})
        d = cfg.to_dict()
        assert d == {"a": {"x": 1, "y": 2}}

    def test_list_of_dicts(self):
        cfg = Config({"items": [{"id": 1}, {"id": 2}]})
        d = cfg.to_dict()
        assert d == {"items": [{"id": 1}, {"id": 2}]}


class TestIsSymbolGraphEnabled:
    """Issue #109: ``indexing.symbol_graph.enabled`` flag honoring."""

    def test_explicit_true(self):
        cfg = Config({"indexing": {"symbol_graph": {"enabled": True}}})
        assert is_symbol_graph_enabled(cfg) is True

    def test_explicit_false(self):
        cfg = Config({"indexing": {"symbol_graph": {"enabled": False}}})
        assert is_symbol_graph_enabled(cfg) is False

    def test_missing_symbol_graph_block_defaults_true(self):
        cfg = Config({"indexing": {"other_key": 1}})
        assert is_symbol_graph_enabled(cfg) is True

    def test_missing_indexing_block_defaults_true(self):
        cfg = Config({"repo": {"repo_id": "x"}})
        assert is_symbol_graph_enabled(cfg) is True

    def test_plain_dict_compatible(self):
        cfg = {"indexing": {"symbol_graph": {"enabled": False}}}
        assert is_symbol_graph_enabled(cfg) is False

    def test_plain_dict_default_true(self):
        assert is_symbol_graph_enabled({}) is True

    # CB-003: non-bool values must raise rather than silently pass through.
    def test_quoted_string_false_raises(self):
        cfg = Config({"indexing": {"symbol_graph": {"enabled": "false"}}})
        with pytest.raises((TypeError, ValueError)):
            is_symbol_graph_enabled(cfg)

    def test_quoted_string_true_raises(self):
        cfg = Config({"indexing": {"symbol_graph": {"enabled": "true"}}})
        with pytest.raises((TypeError, ValueError)):
            is_symbol_graph_enabled(cfg)

    def test_integer_value_raises(self):
        cfg = Config({"indexing": {"symbol_graph": {"enabled": 1}}})
        with pytest.raises((TypeError, ValueError)):
            is_symbol_graph_enabled(cfg)

    def test_list_value_raises(self):
        cfg = {"indexing": {"symbol_graph": {"enabled": [True]}}}
        with pytest.raises((TypeError, ValueError)):
            is_symbol_graph_enabled(cfg)
