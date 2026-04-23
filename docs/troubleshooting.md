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

---

## 長コンテキストで RAM 不足 (Issue #55)

**Symptom**: `configs/photon_long_context.yaml` + 長 prompt（16k+ トークン）で OOM または極端に遅い。

**Checklist**:

1. **`use_kv_cache=False` を試す**: 実測では 16,384 prompt 時、KV cache 無効の方が **RAM 約 7 GB 節約** かつ **11% 高速**。top-level KV cache の encoder_replay / top_level_increment / local_tail_decode の累積コストが、nocache prefill を上回るため。`PhotonInference.generate(..., use_kv_cache=False)` もしくは設定で `photon.use_kv_cache: false`（実装に応じて）。

2. **`training.context_length` を段階的に下げる**: 65,536 → 32,768 → 16,384 の順で試す。RoPE テーブル自体は 65,536 固定でも、実際に流す prompt 長を下げれば attention 側のメモリが二次的に減る。

3. **YAML タイポを疑う**: `rope_scale` (typo) は silently 無視されるため warning ログで検出。ログに `unknown config key ignored: rope_scale` が出ていないか確認。正しくは `rope_scaling: ntk`。

4. **`rope_scaling: none` と `rope_scale_factor: 32.0` を併記していないか**: この組み合わせは factor が silently 無視される（WARNING ログが出る）。`rope_scaling: ntk` に修正する。

5. **`torch_ref` 経路で長コンテキスト**: `torch_ref` は 128 位置までしか扱えない。長コンテキストは必ず MLX 経路（`PhotonModel`）で使うこと。`torch_ref/_precompute_rope` で `scaling != "none"` を渡すと `NotImplementedError` が明示的に raise される（silent fallback なし）。

参考: `reports/issue-55-long-context.md`

---

## Streamlit アプリ: drift_metrics が `N/A` のまま表示される (Issue #82)

**Symptom**: PHOTON プロジェクトのチャット画面で drift metrics パネルが常に `N/A (baseline_rag or first turn)` を表示し、4 指標が取れない。

**Checklist**:

1. **`cfg.model.provider` が `"photon"` か**: `build_pipeline(cfg)` は `cfg.model.provider == "photon"` の場合のみ `PhotonRAGPipeline` を返す。プロジェクトの `photon_config_path` が指す YAML を確認し、`model.provider: "photon"` が設定されているか (`configs/photon_small.yaml:155` 等が参考)。
2. **MLX がインストールされているか**: baseline-only マシンでは `ModuleNotFoundError: mlx.core` が `build_pipeline` 内で発生し、UI は `photon_unavailable_{project_name}` フラグを立てて送信をブロックする。チャット画面上部の赤色エラーバナーを確認。
3. **初回ターン**: drift metrics は 2 ターン目以降から値が入る仕様。最初の質問では `N/A (first turn)` は正常。
4. **`use_photon=False` の baseline プロジェクト**: これは仕様通り `N/A (baseline_rag)`。PHOTON を試したい場合は新規プロジェクトを `use_photon=True` + PHOTON config で作成。

---

## Streamlit アプリ: eval ジョブが進まない / 消えない (Issue #82)

**Symptom**: `[Run Static Eval]` を押したがステータスが `running` のまま止まる、または Streamlit 再起動後に孤児ジョブが残る。

**Checklist**:

1. **state ファイルで PID 確認**: `.cache/photon_app_state.json` を開き、該当 `eval_jobs[<job_id>]` の `pid` を確認。
2. **プロセス存在確認**: `ps -p <pid>`。プロセスがいなければマーカーが作られなかったまま死んだ（OOM や SIGKILL が典型）。この場合次回 `_sync_eval_job` が走ったタイミングで `status='failed'` に遷移する（即時反映したい場合は Streamlit を一度再起動）。
3. **Wall-clock timeout**: 経過 3600 秒を超えると自動的に `status='failed'` + `error_message='wall-clock timeout'` に遷移する（Apple Silicon で 120Q の Static eval でも通常 40 分で終わる想定）。
4. **手動 kill**: `kill -15 <pid>`（SIGTERM）で停止。`kill -9 <pid>`（SIGKILL）は log/marker が不完全になるため最終手段に。
5. **ログ確認**: `logs/eval/<job_id>.log` の末尾を確認。tokenizer エラーや MLX 初期化エラーが典型。
6. **マニュアル cleanup（retention）**: `AppState.eval_jobs` は自動削除されないため、不要になったエントリは `.cache/photon_app_state.json` を直接編集して削除するか、Streamlit を停止 → JSON 編集 → 再起動。成果物（`reports/eval_runs/*.json`、`logs/eval/*.log`、`reports/eval_runs/*.done`）は安全に削除可能（gitignore 済み）。
7. **Concurrent 起動**: `MAX_CONCURRENT_EVAL=1` のため、既に running の eval がある場合は Start ボタンが disabled になる（仕様）。

参考: 設計方針書 `workspace/design/issue-82-app-photon-features-design-policy.md` §6.4 / §7.2
