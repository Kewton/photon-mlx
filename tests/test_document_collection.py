from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.document_collection import (  # noqa: E402
    DocumentSource,
    FetchedDocument,
    collect_documents,
    convert_to_markdown,
    html_to_markdown,
    infer_kind,
    normalize_text,
    parse_url_list,
    pdf_to_text,
)


def test_parse_url_list_accepts_supported_formats(tmp_path: Path) -> None:
    urls = tmp_path / "urls.md"
    urls.write_text(
        "\n".join(
            [
                "# comment",
                "https://example.test/plain.md",
                "[HTML title](https://example.test/page.html)",
                "PDF title | https://example.test/form.pdf",
            ]
        ),
        encoding="utf-8",
    )

    sources = parse_url_list(urls)

    assert sources == [
        DocumentSource(url="https://example.test/plain.md"),
        DocumentSource(url="https://example.test/page.html", title="HTML title"),
        DocumentSource(url="https://example.test/form.pdf", title="PDF title"),
    ]


def test_parse_url_list_rejects_unknown_line(tmp_path: Path) -> None:
    urls = tmp_path / "urls.md"
    urls.write_text("not a url\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported URL list line"):
        parse_url_list(urls)


def test_html_to_markdown_extracts_content_and_ignores_scripts() -> None:
    title, markdown = html_to_markdown(
        """
        <html>
          <head><title> Example Product </title><script>alert(1)</script></head>
          <body>
            <h1>Overview</h1>
            <p>Solves multi-turn RAG questions.</p>
            <ul><li>Ingest</li><li>Ask</li></ul>
          </body>
        </html>
        """
    )

    assert title == "Example Product"
    assert "# Overview" in markdown
    assert "Solves multi-turn RAG questions." in markdown
    assert "- Ingest" in markdown
    assert "alert" not in markdown


def test_normalize_text_removes_control_chars_and_compacts_blank_lines() -> None:
    text = "ＡＢＣ\x00\r\n\r\n\r\nfoo\t\tbar\n"

    assert normalize_text(text) == "ABC\n\nfoo bar\n"


def test_infer_kind_uses_content_type_and_url_suffix() -> None:
    assert infer_kind("https://example.test/a", "application/pdf") == "pdf"
    assert infer_kind("https://example.test/a.pdf", "") == "pdf"
    assert infer_kind("https://example.test/a.md", "") == "markdown"
    assert infer_kind("https://example.test/a.html", "") == "html"
    assert infer_kind("https://example.test/a.bin", "") == "text"


def test_collect_documents_writes_ingest_ready_layout(tmp_path: Path) -> None:
    sources = [
        DocumentSource(url="https://example.test/product.html", title="Product"),
        DocumentSource(url="https://example.test/faq.md"),
    ]

    def fake_fetcher(source: DocumentSource) -> FetchedDocument:
        if source.url.endswith(".html"):
            return FetchedDocument(
                url=source.url,
                content=b"<html><title>Product</title><h1>Product</h1><p>Body</p></html>",
                content_type="text/html",
            )
        return FetchedDocument(
            url=source.url,
            content="## FAQ\n回答です。".encode(),
            content_type="text/markdown",
        )

    collected = collect_documents(
        sources,
        output_root=tmp_path,
        corpus_id="sample_corpus",
        fetcher=fake_fetcher,
    )

    assert len(collected) == 2
    product_dir = tmp_path / "sample_corpus" / "product"
    faq_dir = tmp_path / "sample_corpus" / "faq"
    assert (product_dir / "document.md").read_text(encoding="utf-8").startswith(
        "# Product"
    )
    assert "回答です。" in (faq_dir / "document.md").read_text(encoding="utf-8")

    metadata = json.loads((product_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_url"] == "https://example.test/product.html"
    assert metadata["kind"] == "html"
    assert "maenokota" not in json.dumps(metadata)


def test_collect_documents_keeps_duplicate_titles_separate(tmp_path: Path) -> None:
    sources = [
        DocumentSource(url="https://example.test/a.html", title="Same"),
        DocumentSource(url="https://example.test/b.html", title="Same"),
    ]

    def fake_fetcher(source: DocumentSource) -> FetchedDocument:
        return FetchedDocument(
            url=source.url,
            content=b"<html><title>Same</title><p>Body</p></html>",
            content_type="text/html",
        )

    collected = collect_documents(
        sources,
        output_root=tmp_path,
        corpus_id="sample_corpus",
        fetcher=fake_fetcher,
    )

    assert [Path(item.output_dir).name for item in collected] == ["same", "same-02"]


def test_convert_pdf_requires_optional_dependency_when_missing(monkeypatch) -> None:
    real_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="requires optional dependency 'pypdf'"):
        pdf_to_text(b"%PDF-1.4")


def test_convert_to_markdown_for_text() -> None:
    title, markdown, kind = convert_to_markdown(
        DocumentSource(url="https://example.test/readme.txt", title="Readme"),
        FetchedDocument(
            url="https://example.test/readme.txt",
            content=b"Line 1\r\n\r\nLine 2",
            content_type="text/plain",
        ),
    )

    assert title == "Readme"
    assert markdown == "Line 1\n\nLine 2\n"
    assert kind == "text"
