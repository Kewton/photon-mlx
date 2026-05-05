# Minimal Baseline Corpus

This directory is a tiny markdown corpus for checking the baseline ingest / index / ask flow before connecting a private repository or a larger document set.

It intentionally does not require a PHOTON checkpoint. Use `configs/baseline.yaml` for a normal repository-style smoke check, or `configs/institutional_docs.yaml` if you want to validate markdown heading graph behavior.

## Baseline Smoke

From the repository root:

```bash
photon-rag ingest \
  --repo examples/minimal_corpus \
  --repo-id minimal_demo \
  --commit HEAD \
  --config configs/baseline.yaml

photon-rag index \
  --repo-id minimal_demo \
  --config configs/baseline.yaml

photon-rag ask \
  --repo-id minimal_demo \
  --config configs/baseline.yaml \
  --question "PHOTON-RepoRAG はどのような業務課題を解決しますか？"
```

`ask` uses the generation model configured in the selected YAML. If the model or embedding files are not already cached, the first run may download them according to the upstream model configuration and license.

## Markdown Corpus Smoke

```bash
photon-rag ingest \
  --repo examples/minimal_corpus \
  --repo-id minimal_demo_docs \
  --commit HEAD \
  --config configs/institutional_docs.yaml

photon-rag index \
  --repo-id minimal_demo_docs \
  --config configs/institutional_docs.yaml

photon-rag heading-graph \
  --repo-id minimal_demo_docs \
  --config configs/institutional_docs.yaml
```
