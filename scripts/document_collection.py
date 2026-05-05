from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse


_MARKDOWN_LINK_RE = re.compile(r"^\s*[-*]?\s*\[([^\]]+)\]\((https?://[^)]+)\)\s*$")
_BARE_URL_RE = re.compile(r"^\s*[-*]?\s*(https?://\S+)\s*$")
_TITLE_URL_RE = re.compile(r"^\s*[-*]?\s*([^|]+?)\s*\|\s*(https?://\S+)\s*$")
_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass(frozen=True)
class DocumentSource:
    url: str
    title: str | None = None
    kind: str = "auto"


@dataclass(frozen=True)
class FetchedDocument:
    url: str
    content: bytes
    content_type: str = ""
    status_code: int = 200


@dataclass(frozen=True)
class CollectedDocument:
    source: DocumentSource
    output_dir: str
    document_path: str
    metadata_path: str
    kind: str
    title: str


def parse_url_list(path: str | Path) -> list[DocumentSource]:
    """Read a small Markdown/plain-text URL list.

    Supported line formats:
    - ``https://example.test/page.html``
    - ``[Human title](https://example.test/page.html)``
    - ``Human title | https://example.test/page.html``

    Blank lines and Markdown comments beginning with ``#`` are ignored.
    """
    sources: list[DocumentSource] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        markdown_match = _MARKDOWN_LINK_RE.match(line)
        if markdown_match:
            title, url = markdown_match.groups()
            sources.append(DocumentSource(url=url, title=title.strip()))
            continue
        title_url_match = _TITLE_URL_RE.match(line)
        if title_url_match:
            title, url = title_url_match.groups()
            sources.append(DocumentSource(url=url, title=title.strip()))
            continue
        bare_match = _BARE_URL_RE.match(line)
        if bare_match:
            sources.append(DocumentSource(url=bare_match.group(1)))
            continue
        raise ValueError(f"Unsupported URL list line: {raw_line!r}")
    return sources


def infer_kind(url: str, content_type: str = "", explicit_kind: str = "auto") -> str:
    if explicit_kind != "auto":
        if explicit_kind not in {"html", "pdf", "text", "markdown"}:
            raise ValueError(f"Unsupported document kind: {explicit_kind!r}")
        return explicit_kind
    lowered_type = content_type.lower()
    path = urlparse(url).path.lower()
    if "pdf" in lowered_type or path.endswith(".pdf"):
        return "pdf"
    if "markdown" in lowered_type or path.endswith((".md", ".markdown")):
        return "markdown"
    if "html" in lowered_type or path.endswith((".html", ".htm")):
        return "html"
    return "text"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        ch
        for ch in text
        if ch in {"\n", "\t"} or not unicodedata.category(ch).startswith("C")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compact: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                compact.append("")
            blank_seen = True
            continue
        compact.append(line)
        blank_seen = False
    return "\n".join(compact).strip() + "\n"


class _HtmlMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._heading_level: int | None = None
        self._in_title = False
        self._pending_list_item = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in {"p", "div", "section", "article", "main"}:
            self.parts.append("\n\n")
        elif tag in {"br", "hr"}:
            self.parts.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self.parts.append("\n\n" + ("#" * self._heading_level) + " ")
        elif tag == "li":
            self._pending_list_item = True
            self.parts.append("\n- ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag in {"p", "div", "section", "article", "main", "li"}:
            self.parts.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = None
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        text = data.strip()
        if not text:
            return
        if self._pending_list_item:
            self._pending_list_item = False
        self.parts.append(text + " ")


def html_to_markdown(html: str, *, fallback_title: str = "Untitled") -> tuple[str, str]:
    parser = _HtmlMarkdownParser()
    parser.feed(html)
    title = normalize_text(" ".join(parser.title_parts)).strip() or fallback_title
    body = normalize_text("".join(parser.parts))
    if not body.strip():
        body = title + "\n"
    return title, body


def pdf_to_text(content: bytes) -> str:
    """Extract text from a PDF using optional ``pypdf``.

    ``pypdf`` is intentionally optional so the base RAG package does not gain
    another heavy dependency just for corpus preparation. Install it in the
    caller environment when PDF extraction is required.
    """
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires optional dependency 'pypdf'. "
            "Install it in your corpus-preparation environment."
        ) from exc

    import io

    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize_text("\n\n".join(pages))


def slugify(value: str, fallback: str = "document") -> str:
    value = unicodedata.normalize("NFKC", value).strip().lower()
    value = _SAFE_SLUG_RE.sub("-", value).strip("-._")
    return value[:80] or fallback


def source_to_slug(source: DocumentSource, index: int) -> str:
    if source.title:
        return slugify(source.title, fallback=f"document-{index:03d}")
    parsed = urlparse(source.url)
    candidate = Path(parsed.path).stem or parsed.netloc or f"document-{index:03d}"
    return slugify(candidate, fallback=f"document-{index:03d}")


def default_fetcher(
    source: DocumentSource,
    *,
    timeout_seconds: float = 20.0,
    user_agent: str = "photon-rag-document-collector/0.1",
) -> FetchedDocument:
    import httpx

    response = httpx.get(
        source.url,
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={"User-Agent": user_agent},
    )
    response.raise_for_status()
    return FetchedDocument(
        url=str(response.url),
        content=response.content,
        content_type=response.headers.get("content-type", ""),
        status_code=response.status_code,
    )


def convert_to_markdown(
    source: DocumentSource,
    fetched: FetchedDocument,
) -> tuple[str, str, str]:
    kind = infer_kind(
        fetched.url or source.url,
        content_type=fetched.content_type,
        explicit_kind=source.kind,
    )
    fallback_title = source.title or source.url
    if kind == "html":
        title, markdown = html_to_markdown(
            fetched.content.decode("utf-8", errors="replace"),
            fallback_title=fallback_title,
        )
        return title, markdown, kind
    if kind == "pdf":
        title = source.title or Path(urlparse(source.url).path).stem or source.url
        return title, pdf_to_text(fetched.content), kind
    text = fetched.content.decode("utf-8", errors="replace")
    title = source.title or Path(urlparse(source.url).path).stem or source.url
    return title, normalize_text(text), kind


def collect_documents(
    sources: Iterable[DocumentSource],
    *,
    output_root: str | Path,
    corpus_id: str,
    fetcher: Callable[[DocumentSource], FetchedDocument],
    delay_seconds: float = 0.0,
) -> list[CollectedDocument]:
    """Fetch sources and write an ingest-ready markdown corpus.

    Output layout:
    ``<output_root>/<corpus_id>/<slug>/document.md`` plus ``metadata.json``.
    This mirrors the institutional corpus layout already used by evaluation
    helpers while keeping generated data outside the repository by default.
    """
    root = Path(output_root) / corpus_id
    root.mkdir(parents=True, exist_ok=True)
    collected: list[CollectedDocument] = []
    slug_counts: dict[str, int] = {}

    for index, source in enumerate(sources, start=1):
        if index > 1 and delay_seconds > 0:
            time.sleep(delay_seconds)
        fetched = fetcher(source)
        title, markdown, kind = convert_to_markdown(source, fetched)
        base_slug = source_to_slug(DocumentSource(url=source.url, title=title), index)
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        slug = (
            base_slug
            if slug_counts[base_slug] == 1
            else f"{base_slug}-{slug_counts[base_slug]:02d}"
        )
        doc_dir = root / slug
        doc_dir.mkdir(parents=True, exist_ok=True)
        document_path = doc_dir / "document.md"
        metadata_path = doc_dir / "metadata.json"
        document_path.write_text(markdown, encoding="utf-8")
        metadata = {
            "source_url": source.url,
            "final_url": fetched.url,
            "title": title,
            "kind": kind,
            "content_type": fetched.content_type,
            "status_code": fetched.status_code,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        collected.append(
            CollectedDocument(
                source=source,
                output_dir=str(doc_dir),
                document_path=str(document_path),
                metadata_path=str(metadata_path),
                kind=kind,
                title=title,
            )
        )
    return collected


def write_collection_manifest(
    collected: list[CollectedDocument],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in collected], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
