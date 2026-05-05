# Document Collection

This guide describes the generic document collection tool for preparing a small markdown corpus for PHOTON-RepoRAG.

The tool is intentionally separate from the RAG pipeline. It helps collect HTML / PDF / text sources and writes an ingest-ready markdown layout, but it does not include downloaded documents, real URL lists, checkpoints, or personal local paths.

## Scope

Included:

- Parse a small Markdown/plain-text URL list
- Download HTML / PDF / text documents
- Convert HTML to Markdown using a lightweight extractor
- Extract PDF text when optional `pypdf` is installed
- Normalize whitespace and control characters
- Write an ingest-ready corpus layout
- Write per-document metadata

Not included:

- Downloaded documents
- Large real URL lists
- `download/`, `markdowndb/`, `.venv/`, `.git/`, `workspace/`, `__pycache__/`
- Private or license-unclear documents
- Checkpoints or external model weights

## URL List Format

The URL list accepts these formats:

```markdown
https://example.com/docs/faq.md
[Product guide](https://example.com/docs/product.html)
Application form | https://example.com/docs/form.pdf
```

See `examples/document_urls.sample.md` for a syntax-only example. Do not commit large production URL lists.

## Collect Documents

```bash
python scripts/collect_documents.py \
  --urls examples/document_urls.sample.md \
  --corpus-id my_corpus \
  --output-root data/processed/document_corpora
```

The default output root is under `data/processed/`, which is gitignored.

Output layout:

```text
data/processed/document_corpora/my_corpus/
  manifest.json
  product-guide/
    document.md
    metadata.json
  application-form/
    document.md
    metadata.json
```

Each document directory contains:

- `document.md`: normalized Markdown/text for RAG ingest
- `metadata.json`: source URL, final URL, content type, status code, detected kind

## PDF Extraction

PDF extraction uses optional `pypdf`.

```bash
python -m pip install pypdf
```

`pypdf` is not a runtime dependency of PHOTON-RepoRAG. Install it only in the corpus-preparation environment when PDF extraction is needed.

## Rate Limits And Source Terms

The collector provides `--delay-seconds`, `--timeout-seconds`, and `--user-agent`. The operator is responsible for checking source terms, robots policy, and redistribution permissions before collecting or publishing derived corpora.

Example:

```bash
python scripts/collect_documents.py \
  --urls urls/my_sources.md \
  --corpus-id my_corpus \
  --delay-seconds 1.0 \
  --user-agent "my-org-rag-corpus-builder/0.1"
```

## Ingest The Generated Corpus

After collection, ingest the generated markdown corpus:

```bash
photon-rag ingest \
  --repo data/processed/document_corpora/my_corpus \
  --repo-id my_corpus \
  --commit HEAD \
  --config configs/institutional_docs.yaml

photon-rag index \
  --repo-id my_corpus \
  --config configs/institutional_docs.yaml

photon-rag heading-graph \
  --repo-id my_corpus \
  --config configs/institutional_docs.yaml
```

Then ask a baseline question:

```bash
photon-rag ask \
  --config configs/institutional_docs.yaml \
  --repo-id my_corpus \
  --question "この文書群の主な対象は何ですか？"
```

## Repository Hygiene

Do not commit generated corpora or downloaded source documents. Keep them under gitignored directories such as:

- `data/raw/`
- `data/processed/`
- `workspace/`

If a public example corpus is needed, add a tiny hand-written fixture under `examples/` instead of committing scraped real documents.
