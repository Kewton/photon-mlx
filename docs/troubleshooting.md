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
ls ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/
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

## PHOTON Checkpoint Load Failure (Issue #148)

**Symptom**: `RuntimeError: checkpoint load failed (...)` when starting with `model.provider: photon`.

**Causes and solutions**:

| Cause | Solution |
|-------|----------|
| `cfg.model.checkpoint_path` not set | Add `checkpoint_path: "<name>"` under `model:` in the PHOTON yaml. For Phase A evaluation, place a valid checkpoint before starting — do not use `PHOTON_ALLOW_RANDOM_INIT=1` as a substitute (see note below). |
| Checkpoint directory missing `weights.npz` or `state.json` | Ensure the checkpoint directory contains both files (produced by `photon_mlx.trainer.save_checkpoint`) |
| Checkpoint path outside `PHOTON_CHECKPOINT_ROOT` | Move the checkpoint under the allowed root, or set `PHOTON_CHECKPOINT_ROOT` to the parent directory: `export PHOTON_CHECKPOINT_ROOT=/data/photon_checkpoints` |
| Symlink escaping the allowed root | Remove the symlink and copy the checkpoint directly under `PHOTON_CHECKPOINT_ROOT` |
| Corrupted `weights.npz` | Re-run training or restore the checkpoint from backup |
| Auto-download source missing | Set `PHOTON_CHECKPOINT_REPO_ID=<org>/<repo>` or `model.checkpoint_repo_id` in the PHOTON yaml. Without a source, missing checkpoints intentionally fail fast. |
| Hugging Face auth / network failure | Set `HF_TOKEN` for private repos, verify network access to `huggingface.co`, then retry. The downloader writes under `PHOTON_CHECKPOINT_ROOT`. |

**重要 (CB-003 / 設計方針書 §3 DR-1)**: `PHOTON_ALLOW_RANDOM_INIT=1` は **unit/CI の negative-path テスト専用** です。Phase A 評価や本番環境では使用しないでください。チェックポイントが手元にない場合は、checkpoint を配置するまで評価を開始しないでください (`PHOTON_ALLOW_RANDOM_INIT=1` で代替することは S7-001 random-init eval の再発を招きます)。

**unit/CI negative-path test 専用** (評価・本番では使用禁止):

```bash
# 下記は unit/CI の negative-path テストでのみ使用すること。
# Phase A 評価・本番環境では checkpoint を正しく配置して実行すること。
export PHOTON_ALLOW_RANDOM_INIT=1
python -m baseline_reporag.cli --config configs/institutional_docs_photon.yaml ...
```

This logs a WARNING and continues with random-init weights. The random-init model produces garbage answers and **must not** be used for Phase A evaluation or production inference.

**Diagnosing path containment errors**:

```bash
# Verify PHOTON_CHECKPOINT_ROOT covers the checkpoint
python -c "
import os
from pathlib import Path
root = Path(os.environ.get('PHOTON_CHECKPOINT_ROOT', 'checkpoints')).resolve()
ckpt = Path('path/to/ckpt').resolve()
print('root:', root)
print('ckpt:', ckpt)
print('OK:', ckpt.is_relative_to(root))
"
```

---

## PHOTON Multi-Turn Quality Looks Worse Than Baseline

**Symptom**: PHOTON プロジェクトは起動するが、follow-up 質問で baseline より回答が薄い、必要な evidence が落ちる、または古い話題の citation が混ざる。

**Status**: PHOTON multi-turn は比較・評価対象としてサポートされています。現在の実装では、PHOTON を回答生成モデルそのものとしてだけでなく、関連過去質問、evidence selection、citation eligibility の判断レイヤーとして使います。

**Checklist**:

1. **baseline config と PHOTON config が分かれているか**: 比較モードでは、baseline 側に `configs/baseline.yaml` または `configs/institutional_docs.yaml`、PHOTON 側に `model.provider: photon` の YAML を設定します。
2. **checkpoint が有効か**: `PHOTON_CHECKPOINT_ROOT` と `model.checkpoint_path` が正しいことを確認します。random-init は unit/CI negative-path test 専用です。
3. **pruning 枠が小さすぎないか**: Streamlit のプロジェクト登録/編集で `retrieval/reranker 上位保護N件` と `PHOTON score 選別M件` を確認します。既定値はそれぞれ 4 です。
4. **関連過去質問の取得件数が不足していないか**: `関連過去質問 最大件数` と `関連過去 evidence 件数` を確認します。省略質問が多い業務 Q&A では、過去質問からの evidence 補完が効きます。
5. **Retrieval debug を確認する**: `PHOTON score`, `PHOTON current`, `PHOTON session`, `source`, `Used`, `Citation` を見て、現在質問に効いている evidence と過去文脈由来の evidence を分けて確認します。
6. **citation budget の影響を見る**: 回答中の citation が入れ替わる場合があります。ログの `citation_budget_reranked`, `citation_budget_removed_indices`, `citation_eligibility_scores` を確認します。

---

## Memory Issues / Out of Memory

**Symptom**: Process killed or extremely slow due to memory pressure.

**Solutions**:

| Parameter | Location | Default | Action |
|-----------|----------|---------|--------|
| `evidence_pack.max_chunks` | `configs/baseline.yaml` | 16 | Reduce to 8-12 |
| `evidence_pack.max_tokens` | `configs/baseline.yaml` | 16000 | Reduce to 8000-12000 |
| `generation.max_new_tokens` | `configs/baseline.yaml` | 2048 | Reduce to 1024 or 512 |
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
5. **`tokenizer.tokenizer_id` 未設定 → `ValueError` (Issue #139)**: PHOTON pipeline 構築時に `cfg.tokenizer.tokenizer_id` が未設定だと `_build_photon_deps` が `cfg.tokenizer.tokenizer_id is required for provider=='photon'` を raise する (Issue #139 で旧 stub fallback を撤去)。yaml の `tokenizer:` ブロックに `tokenizer_id`/`vocab_size` が両方設定されていることを確認 (`configs/photon_small.yaml:147-149` 等が参考)。
6. **tokenizer load 失敗 → `ValueError("failed to load tokenizer ...")` (Issue #139)**: `transformers.AutoTokenizer.from_pretrained` が HF Hub 障害 / gated model / 未 cache / network 不通 / `tokenizer_id` 誤り等で失敗すると `_build_photon_deps` が sanitized message を含む `ValueError` を raise する。確認項目:
   - `huggingface-cli login` 状態 (gated model 利用時)
   - `hf cache scan` で対象 tokenizer が cache されているか
   - network 疎通 (`curl -I https://huggingface.co`)
   - yaml の `tokenizer.tokenizer_id` の値が allowlist (`<org>/<name>` 形式、`[A-Za-z0-9._-]` のみ) を満たしているか
   - `trust_remote_code=False` 固定のため、custom Python loader を要求する tokenizer は対象外
   - **機密情報の取り扱い**: HF token / PAT / secret env var は `yaml` / Issue / Slack / log に **平文で貼らない**。認証は `huggingface-cli login` または CI runner secret で行い、private model id を public な log / PR description に書く際は redaction を検討。raw exception text の貼り付けも避ける (sanitized message のみ転載する)。

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

---

## Symbol Graph が生成されない / ロードされない (Issue #109)

**Symptom**: `scripts/build_symbol_graph.py` を実行しても `data/indexes/<repo_id>/symbol_graph.json` が生成されない、あるいは `pipeline_factory` 経由で pipeline を組み立てても `graph` が `None` になる。

**Cause**: Issue #109 で `indexing.symbol_graph.enabled` フラグが honor されるようになった。制度文書など非 Python リポジトリ向けに **`false` を明示した config** を使っている場合、`scripts/build_symbol_graph.py` は skip ログを出して早期 return し、`pipeline_factory` は `SymbolGraph.load` を呼ばず `graph=None` で pipeline を組み立てる。これは意図した挙動である。

**Checklist**:

1. **意図的な skip か?**: 使用中の YAML の `indexing.symbol_graph.enabled` を確認。`false` なら symbol graph は使われない（Python 以外の repo で推奨）。
2. **Python repo でも disable になっていないか**: `true` に戻し、`python scripts/build_symbol_graph.py --repo-id <id>` を再実行。
3. **`symbol_graph.json` 欠落 + enabled=true**: 現行設計では fail-fast（`FileNotFoundError`）。`python scripts/build_symbol_graph.py --repo-id <id>` で再生成すること。

---

## Breaking changes / Migration: Markdown chunker の index 再構築 (Issue #109)

**Symptom**: Issue #109 の markdown chunker が有効になった後、既存 index から markdown chunk を query すると `section_header` が空のままだったり、BM25/embedding のスコアが以前と大きく異なる。

**Cause**: Issue #109 以前は `.md` ファイルを単純な sliding-window で chunk していたため、`section_header=""` が DB に保存されている。新しい chunker は見出し（H1-H3）や条文（`第N条`）境界を尊重するため、chunk 境界・`section_header` の内容・chunk_id が **すべて変わる**。

**Migration 手順**:

```bash
# 1. 既存の index を完全削除（SQLite + embedding + BM25 + symbol_graph）
rm -rf data/indexes/<repo_id>/

# 2. ingest からやり直し
python scripts/ingest_repo.py --repo <path-to-repo> --repo-id <repo_id> --config configs/<your>.yaml

# 3. BM25 + embedding index 再構築
python scripts/build_indexes.py --repo-id <repo_id> --config configs/<your>.yaml

# 4. symbol graph（Python repo のみ。非 Python repo は enabled=false で skip される）
python scripts/build_symbol_graph.py --repo-id <repo_id> --config configs/<your>.yaml
```

**注意**: SQLite chunk ID は `{repo_id}::{rel_path}::{start_line}-{end_line}` で、markdown の境界が変わる以上、旧 chunk_id と新 chunk_id は一致しない。セッション履歴（`logs/sessions/`）に残る旧 chunk_id は次ターンの retrieval 対象にはならないが、index を削除する前に参照していた citation は意味を失う点に留意。
