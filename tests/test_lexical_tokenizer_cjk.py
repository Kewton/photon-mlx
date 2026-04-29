"""Tests for ``baseline_reporag.indexing.lexical._tokenize`` CJK support.

Issue #174: the legacy tokenizer (``re.split(r"[^a-z0-9_]+", ...)``) stripped
all non-ASCII characters, so BM25 contributed 0 signal for Japanese / Chinese
/ Korean corpora. Retrieval fell back to embedding-only ranking, which then
mis-ranked semantically-similar-but-wrong chunks (e.g. 3号 boilerplate had
the literal word ``認定基準`` while the actual answer chunk used the
synonym ``認定の条件``).

The fix combines two streams:
- ASCII alphanumeric tokens (legacy code/English path, unchanged behaviour)
- CJK character bigrams (new) so partial substring matching restores BM25
  for Japanese-heavy corpora (institutional documents 制度文書 use case)
"""

from __future__ import annotations

from baseline_reporag.indexing.lexical import _tokenize


class TestAsciiPathUnchanged:
    """Legacy English / Python code tokenization must keep working."""

    def test_camel_case_split(self) -> None:
        assert _tokenize("fooBar") == ["foo", "bar"]

    def test_snake_case_kept_as_one(self) -> None:
        assert _tokenize("my_function_name") == ["my_function_name"]

    def test_lowercases(self) -> None:
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_splits(self) -> None:
        assert _tokenize("FastAPI uses Depends() for DI") == [
            "fast",
            "api",
            "uses",
            "depends",
            "for",
            "di",
        ]

    def test_short_token_filter(self) -> None:
        # 1-char ASCII tokens dropped (legacy len>=2 filter)
        assert _tokenize("a b cd") == ["cd"]

    def test_empty_input(self) -> None:
        assert _tokenize("") == []


class TestCjkBigrams:
    def test_short_kanji_query_yields_bigrams(self) -> None:
        # ``認定基準`` (4 kanji) → 3 overlapping bigrams
        assert _tokenize("認定基準") == ["認定", "定基", "基準"]

    def test_section_header_form_yields_bigrams(self) -> None:
        # Full-width brackets are NOT in the CJK ranges so they act as
        # boundaries — only the kanji/hiragana run inside is bigram-ized.
        assert _tokenize("【認定の条件】") == ["認定", "定の", "の条", "条件"]

    def test_single_cjk_character_kept(self) -> None:
        # 区 (1 kanji) — keep it as a single-char token rather than dropping
        assert _tokenize("区") == ["区"]

    def test_query_and_corpus_bigram_overlap_drives_bm25(self) -> None:
        """``認定基準`` query and ``【認定の条件】`` corpus header share
        the bigram ``認定`` — the very property BM25 needs to score above 0
        for cross-vocabulary matches."""
        query_tokens = set(_tokenize("認定基準"))
        corpus_tokens = set(_tokenize("【認定の条件】"))
        assert "認定" in query_tokens & corpus_tokens

    def test_hiragana_run(self) -> None:
        # Pure hiragana: ``あいうえお`` (5 chars) → 4 overlapping bigrams
        assert _tokenize("あいうえお") == ["あい", "いう", "うえ", "えお"]

    def test_katakana_run(self) -> None:
        # ``セーフティネット`` includes ``ー`` (Katakana-Hiragana Prolonged
        # Sound Mark, U+30FC) which is in the katakana block and stays in
        # the CJK run.
        out = _tokenize("セーフティネット")
        assert out == ["セー", "ーフ", "フテ", "ティ", "ィネ", "ネッ", "ット"]


class TestMixedAsciiCjk:
    def test_ascii_and_cjk_concatenated(self) -> None:
        out = _tokenize("PHOTON モデル")
        assert "photon" in out  # ASCII path lower-cased
        assert "モデ" in out  # CJK bigrams
        assert "デル" in out

    def test_cjk_runs_separated_by_ascii(self) -> None:
        # ``保証 1 号`` — ASCII digit/space splits the CJK run
        out = _tokenize("保証1号認定")
        # ``保証`` and ``号認定`` runs treated independently
        assert "保証" in out
        assert "号認" in out
        assert "認定" in out
        # ``1`` is single-char ASCII → filtered (len < 2)
        assert "1" not in out


class TestEdgeCases:
    def test_only_punctuation(self) -> None:
        assert _tokenize("【】！？") == []

    def test_mixed_with_ideographic_iteration_mark(self) -> None:
        # ``々`` (U+3005, ideographic iteration mark) is included in the
        # CJK run so words like ``人々`` tokenize correctly.
        out = _tokenize("人々の生活")
        assert "人々" in out
