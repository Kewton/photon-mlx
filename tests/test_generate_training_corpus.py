"""Tests for scripts/generate_training_corpus.py – Llama tokenizer migration."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestTokenizeText(unittest.TestCase):
    """tokenize_text should delegate to Tokenizer.encode with add_special_tokens=False."""

    def test_tokenize_text_normal(self) -> None:
        from scripts.generate_training_corpus import tokenize_text

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]

        result = tokenize_text("hello world", mock_tokenizer)

        self.assertEqual(result, [1, 2, 3, 4])
        mock_tokenizer.encode.assert_called_once_with(
            "hello world", add_special_tokens=False
        )

    def test_tokenize_text_empty(self) -> None:
        from scripts.generate_training_corpus import tokenize_text

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = []

        result = tokenize_text("", mock_tokenizer)

        self.assertEqual(result, [])
        mock_tokenizer.encode.assert_called_once_with("", add_special_tokens=False)

    def test_tokenize_text_no_special_tokens(self) -> None:
        from scripts.generate_training_corpus import tokenize_text

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [10, 20]

        tokenize_text("test", mock_tokenizer)

        # Verify add_special_tokens=False is always passed
        call_kwargs = mock_tokenizer.encode.call_args
        self.assertFalse(call_kwargs.kwargs["add_special_tokens"])


class TestResolveTokenizerId(unittest.TestCase):
    """resolve_tokenizer_id should respect CLI > config > default priority."""

    def test_cli_priority(self) -> None:
        from scripts.generate_training_corpus import resolve_tokenizer_id

        result = resolve_tokenizer_id("org/cli-model", None)
        self.assertEqual(result, "org/cli-model")

    def test_config_priority(self) -> None:
        from scripts.generate_training_corpus import resolve_tokenizer_id

        result = resolve_tokenizer_id(None, "org/config-model")
        self.assertEqual(result, "org/config-model")

    def test_default(self) -> None:
        from scripts.generate_training_corpus import resolve_tokenizer_id
        from torch_ref.config import TokenizerConfig

        result = resolve_tokenizer_id(None, None)
        self.assertEqual(result, TokenizerConfig().tokenizer_id)

    def test_cli_overrides_config(self) -> None:
        from scripts.generate_training_corpus import resolve_tokenizer_id

        result = resolve_tokenizer_id("org/cli-model", "org/config-model")
        self.assertEqual(result, "org/cli-model")


class TestValidateTokenizerId(unittest.TestCase):
    """validate_tokenizer_id should accept HF repo IDs and reject paths/URLs."""

    def test_valid_hf_repo_id(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        result = validate_tokenizer_id("meta-llama/Llama-2-7b-hf")
        self.assertEqual(result, "meta-llama/Llama-2-7b-hf")

    def test_rejects_absolute_path(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        with self.assertRaises(ValueError):
            validate_tokenizer_id("/tmp/x")

    def test_rejects_relative_path(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        with self.assertRaises(ValueError):
            validate_tokenizer_id("../x")

    def test_rejects_url(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        with self.assertRaises(ValueError):
            validate_tokenizer_id("https://example.com/model")

    def test_rejects_dot_dot_traversal(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        with self.assertRaises(ValueError):
            validate_tokenizer_id("foo/bar/../baz")

    def test_rejects_backslash(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        with self.assertRaises(ValueError):
            validate_tokenizer_id("foo\\bar")

    def test_rejects_tilde_path(self) -> None:
        from scripts.generate_training_corpus import validate_tokenizer_id

        with self.assertRaises(ValueError):
            validate_tokenizer_id("~/models/llama")


class TestMain(unittest.TestCase):
    """main() should use AutoTokenizer via tokenize_text instead of simple_tokenize."""

    @patch("scripts.generate_training_corpus.argparse.ArgumentParser.parse_args")
    @patch("scripts.generate_training_corpus.load_config")
    @patch("scripts.generate_training_corpus.ChunkStore")
    @patch("scripts.generate_training_corpus.AutoTokenizer")
    def test_main_uses_tokenize_text(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_chunk_store_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_parse_args: MagicMock,
    ) -> None:
        import types

        from scripts.generate_training_corpus import main

        # Setup args
        args = types.SimpleNamespace(
            repo_id="test_repo",
            config="configs/baseline.yaml",
            photon_config=None,
            tokenizer_id=None,
            output_dir="/tmp/test_output_corpus",
            val_ratio=0.1,
            seed=42,
            max_chunks=1,
        )
        mock_parse_args.return_value = args

        # Setup config
        cfg = MagicMock()
        cfg.repo.repo_commit = "abc1234567890"
        cfg.paths.data_root = "/tmp/test_data"
        mock_load_config.return_value = cfg

        # Setup chunk store
        mock_chunk = MagicMock()
        mock_chunk.content = "hello world"
        mock_chunk.chunk_id = "c1"
        mock_chunk.rel_path = "test.py"
        mock_store = MagicMock()
        mock_store.iter_repo.return_value = [mock_chunk]
        mock_chunk_store_cls.return_value = mock_store

        # Setup tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(20))  # 20 tokens >= 16
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        main()

        # Verify AutoTokenizer.from_pretrained was called with trust_remote_code=False
        mock_auto_tokenizer.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_tokenizer.from_pretrained.call_args
        self.assertFalse(call_kwargs.kwargs.get("trust_remote_code", True))

        # Verify tokenizer.encode was called (via tokenize_text)
        mock_tokenizer.encode.assert_called_with(
            "hello world", add_special_tokens=False
        )

    @patch("scripts.generate_training_corpus.argparse.ArgumentParser.parse_args")
    @patch("scripts.generate_training_corpus.load_config")
    @patch("scripts.generate_training_corpus.ChunkStore")
    @patch("scripts.generate_training_corpus.AutoTokenizer")
    def test_main_cli_tokenizer_id(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_chunk_store_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_parse_args: MagicMock,
    ) -> None:
        import types

        from scripts.generate_training_corpus import main

        args = types.SimpleNamespace(
            repo_id="test_repo",
            config="configs/baseline.yaml",
            photon_config=None,
            tokenizer_id="org/custom-model",
            output_dir="/tmp/test_output_corpus",
            val_ratio=0.1,
            seed=42,
            max_chunks=0,
        )
        mock_parse_args.return_value = args

        cfg = MagicMock()
        cfg.repo.repo_commit = "abc1234567890"
        cfg.paths.data_root = "/tmp/test_data"
        mock_load_config.return_value = cfg

        mock_store = MagicMock()
        mock_store.iter_repo.return_value = []
        mock_chunk_store_cls.return_value = mock_store

        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Empty corpus raises ValueError, but we still verify from_pretrained was called
        with self.assertRaises(ValueError):
            main()

        # Verify from_pretrained was called with the CLI-specified tokenizer id
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "org/custom-model", trust_remote_code=False
        )


if __name__ == "__main__":
    unittest.main()
