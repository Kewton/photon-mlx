# Troubleshooting Guide

## Model Download Failures

**Symptom**: Error during first run when downloading models from Hugging Face.

**Possible causes and solutions**:

| Cause | Solution |
|-------|----------|
| No network access | Ensure internet connectivity; models are downloaded from `huggingface.co` |
| Insufficient disk space | Free at least 10 GB; models are cached in `~/.cache/huggingface/` |
| Hugging Face rate limit | Wait and retry, or set `HF_TOKEN` environment variable for authenticated access |
| Proxy/firewall blocking | Configure `HTTP_PROXY` / `HTTPS_PROXY` environment variables |

To verify models are cached:

```bash
ls ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-14B-Instruct-4bit/
ls ~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/
ls ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
```

---

## Stale Indexes After Repository Update

**Symptom**: Answers reference old code or miss recent changes.

**Solution**: Re-run the full ingestion and indexing pipeline:

```bash
python scripts/ingest_repo.py --config configs/baseline.yaml
python scripts/build_indexes.py --config configs/baseline.yaml
python scripts/build_symbol_graph.py --config configs/baseline.yaml
```

Then restart the server.

---

## High No-Citation Rate

**Symptom**: Many answers lack `[C:N]` citations (target: < 17.5%).

**Checklist**:

1. **Reranker enabled?** Check `retrieval.reranker.enabled: true` in `configs/baseline.yaml`.
2. **Citation post-processing enabled?** Check `answering.citation_postprocess_enabled: true`.
3. **Retrieval noise?** Run evaluation and check retrieval precision:
   ```bash
   python -m scripts.run_baseline_eval --config configs/baseline.yaml --max-questions 120
   ```
4. **Evidence pack too small?** Try increasing `evidence_pack.max_chunks` (default: 16) or `evidence_pack.max_tokens` (default: 16000).
5. **Local refresh enabled?** Ensure `evidence_pack.local_refresh.enabled: true`.

---

## PHOTON Multi-Turn Not Supported

**Symptom**: Errors when attempting to use PHOTON hierarchical decoder for multi-turn conversations.

**Status**: Known limitation. PHOTON was evaluated at Gate 2 and received a No-Go judgment. Use `baseline_reporag` only for production workloads.

The PHOTON module (`photon_mlx/`) is retained for research purposes but is not production-ready.

---

## Memory Issues / Out of Memory

**Symptom**: Process killed or extremely slow due to memory pressure.

**Solutions**:

| Parameter | Location | Default | Action |
|-----------|----------|---------|--------|
| `evidence_pack.max_chunks` | `configs/baseline.yaml` | 16 | Reduce to 8-12 |
| `evidence_pack.max_tokens` | `configs/baseline.yaml` | 16000 | Reduce to 8000-12000 |
| `generation.max_new_tokens` | `configs/baseline.yaml` | 768 | Reduce to 512 |
| `retrieval.rerank_top_k` | `configs/baseline.yaml` | 12 | Reduce to 8 |
| `indexing.embedding.batch_size` | `configs/baseline.yaml` | 64 | Reduce to 32 |

Monitor memory usage:

```bash
# Check process memory
ps aux | grep baseline_reporag

# macOS Activity Monitor (CLI)
top -pid $(pgrep -f baseline_reporag)
```

---

## Server Not Responding

**Symptom**: Server starts but does not respond to requests.

**Checklist**:

1. **Port conflict?** Check if port 8080 is already in use:
   ```bash
   lsof -i :8080
   ```
2. **Timeout?** Default request timeout is 180 seconds. Complex queries on large repos may exceed this. Increase `serving.request_timeout_seconds` if needed.
3. **Logs?** Check `logs/` directory for error details.
4. **Model loaded?** First request triggers model loading, which can take 30-60 seconds. Wait for the server to log that model loading is complete.

---

## Evaluation Script Errors

**Symptom**: `run_baseline_eval` fails or produces unexpected results.

**Checklist**:

1. **Eval set exists?** Ensure evaluation data is present under `data/eval_sets/`.
2. **Indexes built?** Run the full indexing pipeline before evaluation.
3. **Config matches?** Ensure the `--config` flag points to the correct YAML file.
