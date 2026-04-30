# Deployment Guide - Baseline RepoRAG

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| **OS** | macOS with Apple Silicon (M1/M2/M3) | M3 Ultra |
| **RAM** | 32 GB | 64 GB |
| **Python** | 3.12+ | 3.12+ |
| **Storage** | ~10 GB (model + indexes) | 20 GB+ |

> The Qwen3.5-9B-MLX-4bit model (採用済 / 2026-04-28) loads approximately **5-6 GB** into memory.
> See `reports/qwen_model_matrix_20260428_400cmp_report.md` for the 400-sample comparison
> that drove the switch (Qwen 2.5 → Qwen 3.5 no-think): static p50 -38.6%, multi-turn p50 -46.2%
> on baseline; PHOTON+Qwen3.5 multi-turn p50 7,438 ms (-43.6% vs PHOTON+Qwen2.5).
> Sentence-transformers embedding model and cross-encoder reranker require additional memory:
> - **Global default profile** (`configs/baseline.yaml`, `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`): ~1 GB
> - **Institutional profile** (`configs/institutional_docs.yaml`, `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` post-#137): **~5-6 GB** (bge-m3 ~2.3 GB + bge-reranker-v2-m3 ~2.3 GB resident).

---

## Setup Steps

### 1. Clone the repository

```bash
git clone https://github.com/Kewton/photon-mlx.git
cd photon-mlx
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ingest the target repository

```bash
python scripts/ingest_repo.py --config configs/baseline.yaml
```

This extracts and chunks source files into the SQLite store under `data/processed/`.

### 4. Build indexes

```bash
python scripts/build_indexes.py --config configs/baseline.yaml
```

Builds BM25 lexical index and sentence-transformer embedding index under `data/indexes/`.

### 5. Build the symbol graph

```bash
python scripts/build_symbol_graph.py --config configs/baseline.yaml
```

Constructs import/call/inheritance edges for graph-expanded retrieval.

> **Note (Issue #109)**: Symbol graph is **Python-centric**. Non-Python
> corpora (e.g. 制度文書 markdown) can set
> `indexing.symbol_graph.enabled: false` in the YAML to skip both
> `build_symbol_graph.py` and the runtime `SymbolGraph.load()`. In that
> mode `expand_with_graph` still returns file-neighbors, only the
> graph-neighbor branch is skipped.

### 5.5. Build the heading graph (optional, Issue #180)

```bash
python -m scripts.build_heading_graph --config configs/institutional_docs.yaml --repo-id institutional_documents
```

Constructs a markdown heading hierarchy graph (parent/sibling chunk expansion) for institutional document retrieval.
Skipped automatically when `indexing.heading_graph.enabled: false` (default).

> **Note**: heading_graph ships `enabled: false` in all production configs.
> Enable in `institutional_docs*.yaml` separately after validating AC 5 multi-turn NC 0.00%.

### 6. Start the server

```bash
python -m baseline_reporag.server --config configs/baseline.yaml
```

The server listens on `127.0.0.1:8080` by default (configurable in `configs/baseline.yaml` under `serving`).

### 7. Or use the CLI

```bash
python -m baseline_reporag.cli --config configs/baseline.yaml --question "How does FastAPI handle dependency injection?"
```

---

## Configuration

The main configuration file is `configs/baseline.yaml`.

### Key parameters

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model.model_id` | LLM model | `mlx-community/Qwen3.5-9B-MLX-4bit` | Auto-downloaded on first run. no-think モード使用 (Generator 内で `enable_thinking=False`) |
| `retrieval.reranker.model_id` | Reranker (global default) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Auto-downloaded on first run |
| `retrieval.reranker.model_id` (institutional) | Reranker for institutional profile | `BAAI/bge-reranker-v2-m3` | Used by `configs/institutional_docs.yaml` (#137) |
| `indexing.embedding.model_id` (institutional) | Embedding for institutional profile | `BAAI/bge-m3` | Used by `configs/institutional_docs.yaml` (#137), `max_input_chars=8192` |
| `retrieval.weights` | Fusion weights | lexical: 0.45, embedding: 0.45, graph: 0.10 | Hybrid retrieval blending |
| `retrieval.rerank_top_k` | Rerank cutoff | 12 | Number of chunks after reranking |
| `evidence_pack.max_chunks` | Max evidence chunks | 16 | Reduce to save memory |
| `evidence_pack.max_tokens` | Max evidence tokens | 16000 | Reduce to save memory |
| `generation.max_new_tokens` | Generation length | 768 | Max tokens per answer |
| `generation.temperature` | Temperature | 0.2 | Lower = more deterministic |
| `serving.host` | Server host | `127.0.0.1` | Bind address |
| `serving.port` | Server port | `8080` | HTTP port |
| `serving.request_timeout_seconds` | Request timeout | 180 | Seconds before timeout |

### Model downloads

Both the LLM and reranker models are downloaded automatically from Hugging Face on first use.
They are cached in `~/.cache/huggingface/`. Ensure network access is available for the first run.

---

## PHOTON Checkpoint Distribution

PHOTON pipeline (`provider: "photon"` in YAML) は学習済 checkpoint の物理配置と環境変数の整合が必要。
本セクションは Issue #135 採用 checkpoint (`photon_institutional_retrain_20260428/step_003000`、val_loss 0.4777) の配備運用を記録する。

### 採用 checkpoint の配置 (2026-04-28 整理済)

```
$REPO/checkpoints/photon_institutional_retrain_20260428/
├── best/         (1.4 GB) ← step_003000 と同内容、念のため保持
└── step_003000/  (1.4 GB) ← ★採用、configs/institutional_docs_photon.yaml が参照

合計: ~2.8 GB
```

**配置先の決定根拠** (worktree cleanup 耐性):
- 学習は `feature/issue-135-photon-retrain` worktree 内 (`./checkpoints/`) で実施。
- そのまま参照すると worktree cleanup で消失するリスクがあるため、**develop worktree の `./checkpoints/` 配下にコピー**して長期保管する運用に変更。
- 中間 step (step_001000, _001500, _002000, _002500, final) は disk space 節約のため削除済 (~7 GB 解放)。再現性は `logs/train_photon_institutional_retrain_20260428/train_log.jsonl` で担保。

### 環境変数の設定

```bash
# 本マシン (M3 Ultra Mac Studio) では develop worktree の絶対パスを使用
export PHOTON_CHECKPOINT_ROOT=/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints
```

config 側 (`configs/institutional_docs_photon.yaml`) は相対パスで参照:
```yaml
model:
  checkpoint_path: "photon_institutional_retrain_20260428/step_003000"
```

`baseline_reporag/photon_pipeline.py::_resolve_checkpoint_path` (Issue #148 Phase A0) が
`$PHOTON_CHECKPOINT_ROOT/$checkpoint_path` を `Path.resolve(strict=True)` で解決し、root containment と
weights.npz/state.json の存在を fail-loud で検証する。

### Random-init 安全装置

`PHOTON_CHECKPOINT_ROOT` 未設定 or 不正パスで起動した場合、`_load_photon_checkpoint` は **RuntimeError で fail-fast**。
unit/CI の negative-path テスト用 escape として `PHOTON_ALLOW_RANDOM_INIT=1` のみ許可するが、
production/eval では設定禁止 (S7-001 random-init eval の再発防止)。

### 旧 checkpoint (mulmoclaude 600-step) の温存

| 用途 | 場所 | 内容 |
|------|------|------|
| **再学習 resume 起点として保持** | `$REPO/checkpoints/step_000600/` | mulmoclaude 600-step (val_loss 1.6238) |
| 過去 step (step_000100 〜 step_001600 等) | 同上 | 学習途上のスナップショット、研究用 |

これらは Issue #135 の resume_from を再現する場合のみ必要なので、disk space 圧迫時は最小限 (step_000600 のみ) に削減してよい。

### Phase 3 (将来) — External Storage への移行

Production deploy 時には設計方針書 §11 リスク表「派生学習物 (corpus / checkpoint) は public 配布対象外」に準じ、
private S3 / GCS / HuggingFace Private Hub にアップして、各環境は `download_checkpoint.sh` 等で取得する運用に切り替える予定。

```bash
# 例: huggingface_hub 経由 (Phase 3 で実装予定)
huggingface-cli download <org>/photon-institutional-retrain-20260428 \
  --local-dir ./checkpoints/photon_institutional_retrain_20260428 \
  --include "step_003000/*"
```

詳細は Phase 3 設計時に別 Issue で議論。

### 配備時のチェックリスト

- [ ] `PHOTON_CHECKPOINT_ROOT` を環境変数に設定 (.envrc / docker-compose.yml / k8s ConfigMap)
- [ ] `$PHOTON_CHECKPOINT_ROOT/photon_institutional_retrain_20260428/step_003000/` に `weights.npz`, `state.json`, `integrity.json` の 3 ファイル存在
- [ ] `state.json` の `best_val_loss` が `0.47774723892817733` 付近 (整合性確認)
- [ ] `python -m baseline_reporag.cli --config configs/institutional_docs_photon.yaml` で smoke test、PHOTON 起動 log で `Loaded PHOTON checkpoint from ...` を確認

---

## Monitoring

### Key metrics

| Metric | Target | Description |
|--------|--------|-------------|
| No-citation rate | < 17.5% | Answers without `[C:N]` citations |
| Wrong-citation rate | 0% | Citations pointing to irrelevant chunks |
| Latency P50 | < 20s | End-to-end response time (median) |

### Running evaluation

```bash
python -m scripts.run_baseline_eval --config configs/baseline.yaml --max-questions 120
```

Results are saved to the `reports/` directory.

### Logs

- Location: `logs/` directory (configured via `paths.log_root`)
- Format: JSONL (one JSON object per line)
- Includes: raw prompts, raw answers, retrieval debug info, latency breakdown, memory metrics, citations

### Log rotation

Logs accumulate in `logs/`. Recommended rotation: weekly.

Example using `newsyslog` on macOS (add to `/etc/newsyslog.d/reporag.conf`):

```
# logfilename          [owner:group]  mode  count  size  when  flags
/path/to/photon-mlx/logs/*.jsonl       644   4      *     $W0   J
```

Or use a simple cron job:

```bash
# Weekly log rotation (Sunday 00:00)
0 0 * * 0 cd /path/to/photon-mlx && mkdir -p logs/archive && mv logs/*.jsonl logs/archive/ 2>/dev/null
```

---

## PHOTON Environment Variables (Issue #148)

When running the PHOTON provider (`model.provider: photon`), the following
environment variables control checkpoint loading behaviour.

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOTON_CHECKPOINT_ROOT` | `checkpoints/` (repo-relative) | Root directory under which `cfg.model.checkpoint_path` must reside. Set to an absolute path when checkpoints live outside the repository. Example: `export PHOTON_CHECKPOINT_ROOT=/data/photon_checkpoints` |
| `PHOTON_ALLOW_RANDOM_INIT` | `0` (fail-fast) | Set to `1` to continue with random-init weights when checkpoint loading fails. A WARNING is logged. **Restricted to unit/CI negative-path tests only.** Do **not** set this in Phase A evaluation or production — the random-init model produces garbage answers and reproduces the S7-001 defect. For Phase A eval, place a valid checkpoint before starting. |

### PHOTON checkpoint setup

1. Place the checkpoint directory (containing `weights.npz` and `state.json`) under the allowed root:
   ```bash
   export PHOTON_CHECKPOINT_ROOT=/data/photon_checkpoints
   mkdir -p /data/photon_checkpoints/mulmoclaude_step600
   cp weights.npz state.json /data/photon_checkpoints/mulmoclaude_step600/
   ```

2. Set `model.checkpoint_path` in the PHOTON YAML config:
   ```yaml
   model:
     checkpoint_path: "mulmoclaude_step600"  # relative to PHOTON_CHECKPOINT_ROOT
   ```

3. The checkpoint path must remain within `PHOTON_CHECKPOINT_ROOT`. Symlinks
   that escape the root are rejected with a `RuntimeError`.

---

## Daemon Setup (launchd)

To run the server as a macOS daemon, create a launchd plist file.

### 1. Create the plist

Save as `~/Library/LaunchAgents/com.photon.reporag.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.photon.reporag</string>

    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python</string>
        <string>-m</string>
        <string>baseline_reporag.server</string>
        <string>--config</string>
        <string>configs/baseline.yaml</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/photon-mlx</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/path/to/photon-mlx/logs/server-stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/path/to/photon-mlx/logs/server-stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

> Replace `/path/to/python` with the full path to your Python interpreter
> (e.g., output of `which python`), and `/path/to/photon-mlx` with the
> actual repository path.

### 2. Load the daemon

```bash
launchctl load ~/Library/LaunchAgents/com.photon.reporag.plist
```

### 3. Manage the daemon

```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.photon.reporag.plist

# Restart
launchctl unload ~/Library/LaunchAgents/com.photon.reporag.plist
launchctl load ~/Library/LaunchAgents/com.photon.reporag.plist

# Check status
launchctl list | grep reporag
```

---

## Re-indexing After Repository Updates

When the target repository is updated, re-run the ingestion and indexing pipeline:

```bash
python scripts/ingest_repo.py --config configs/baseline.yaml
python scripts/build_indexes.py --config configs/baseline.yaml
python scripts/build_symbol_graph.py --config configs/baseline.yaml
```

Restart the server after re-indexing to pick up the new data.

---

## Streamlit App: PHOTON Wave 2-4 UI (Issue #82)

The management app (`app/photon_app.py`) exposes the Wave 2-4 PHOTON features through a GUI instead of requiring YAML edits.

### PHOTON プロジェクト作成ウィザード

`page_projects()` の「Create new project」フォーム内に **PHOTON settings** expander を配置。`use_photon=True` で作成するプロジェクトは以下をフォームで設定可能:

- **Config template**: `photon_small` / `photon_tiny` / `photon_long_context`
- **RecGen (PHOTON generation)**: 有効/無効トグル + Fallback policy (`qwen` / `abort`)
- **2-pass search**: 有効/無効トグル + pass1/pass2 top_k
- **Working memory**: enable, max_turns, aggregation (`weighted`/`attention`/`last`), storage_mode (`full`/`top_level_only`), past_turn_pinning

「**Apply best-practice**」チェックボックスを有効にして保存すると、以下 5 キーがテンプレートに merge される:

- `safe_recgen.enabled: true`
- `generation.evidence_pruning_enabled: true`
- `session_memory.working_memory.enabled: true`
- `inference.photon_generation_enabled: false`（RecGen 非推奨・MT eval で +6.1pp NC）
- `retrieval.two_pass_search.enabled: false`（Static 悪化）

プロファイル `photon_tiny_recgen` / `photon_tiny` / `photon_600m_paper` に対しては意図的な設定の上書きについて警告が表示される（生成された YAML は `projects/<project_name>/photon.yaml` に保存）。

### Drift metrics panel（チャット画面）

PHOTON プロジェクトでのチャット応答ごとに以下 4 指標が表示される:

- `token_level`（`latent_cosine_drift_token`）
- `mid_level`（`latent_cosine_drift_mid`）
- `top_level`（`latent_cosine_drift_top`）— 閾値は `cfg.safe_recgen.thresholds.latent_cosine_drift`
- `topic_shift`（`topic_shift_score`）— 閾値は `cfg.safe_recgen.thresholds.topic_shift_score`

閾値超過時は `⚠` バッジ付きで視覚的に区別。baseline プロジェクトや初回ターンは `N/A (baseline_rag or first turn)` 表示。

### Working memory panel（turn_history）

PHOTON + `session_memory.working_memory.enabled=true` のプロジェクトでは最新 `max_turns` 件の履歴を表示:

- Turn ID, question_text, timestamp
- cited_chunk_ids（`SessionManager` の `Turn.cited_chunk_ids` と `turn_id` で join）

baseline プロジェクトは `N/A (baseline_rag)`、working_memory OFF は `N/A (working_memory disabled)` 表示。

### Eval runner（training 詳細）

学習ジョブの展開パネル内から eval を起動可能:

- `[Run Static Eval]`: `scripts.run_baseline_eval` を `-m` で起動
- `[Run Multi-Turn Eval]`: `scripts.run_multi_turn_eval` を `-m` で起動
- 同時実行数は 1（`MAX_CONCURRENT_EVAL=1`）、wall-clock timeout 3600s

Eval ジョブの状態は `.cache/photon_app_state.json` の `eval_jobs` dict に永続化される。起動時の subprocess は `shell=False`、結果 JSON は `reports/eval_runs/<job_id>.json`、ログは `logs/eval/<job_id>.log`（両方 gitignore 済み）。進捗は `PROGRESS done=N total=M p50_ms=X nc=Y` 形式のログ行から自動抽出。正常終了時に `reports/eval_runs/<job_id>.done` マーカーファイルが作成される。

---

## Breaking changes / Migration (Issue #109)

### Markdown chunker の導入

Issue #109 で `.md` ファイルは見出し（H1-H3）・条文（`第N条`/`第N節`）・コードフェンス（バッククォートのみ）を尊重する専用 chunker (`_chunk_markdown`) で分割されるようになった。これにより:

- `Chunk.section_header` が空文字列から `"H1 > H3"` 形式に変わる（評価値に影響する可能性あり、Gate 2 v4 Static NC ±1pp 以内を許容帯としてモニタ）
- `chunk_id` は `{repo_id}::{rel_path}::{start_line}-{end_line}` のまま維持されるが、境界位置が変わるため旧 chunk_id との互換性はなし

**移行手順（既存 index を使っている repo 向け）**:

```bash
rm -rf data/indexes/<repo_id>/
python scripts/ingest_repo.py --repo <path> --repo-id <repo_id> --config configs/<profile>.yaml
python scripts/build_indexes.py --repo-id <repo_id> --config configs/<profile>.yaml
# Python repo のみ（非 Python は enabled=false で skip される）
python scripts/build_symbol_graph.py --repo-id <repo_id> --config configs/<profile>.yaml
```

### `indexing.symbol_graph.enabled: false` の新運用パターン

制度文書（法令・規程など）のように Python シンボルが存在しないリポジトリでは symbol graph が無価値なため、以下の設定で build/load を完全に skip できる:

```yaml
indexing:
  symbol_graph:
    enabled: false
```

Skip される箇所:

- `scripts/build_symbol_graph.py` は `Skipped: indexing.symbol_graph.enabled=false` を stdout に出して早期 return（`symbol_graph.json` は生成されない）
- `baseline_reporag/pipeline_factory.py` は `SymbolGraph.load` を呼ばず `graph=None` を pipeline に渡す
- `expand_with_graph` は `graph=None` の場合 graph-neighbors の展開を skip し、file-neighbors（`store.get_neighbors`）のみを返す

この経路で pipeline を組み立てても retrieval / generation のインターフェースに変化はなく、既存 CLI / server も設定を変えるだけで動く。

### `indexing.heading_graph.enabled` の運用パターン (Issue #180)

Markdown 見出し階層グラフ（親/兄弟チャンク展開）を有効化するには:

```yaml
indexing:
  heading_graph:
    enabled: true
```

**build コマンド**:

```bash
python -m scripts.build_heading_graph --repo-id <repo_id> --config configs/<profile>.yaml
```

Skip 時: `Skipped: indexing.heading_graph.enabled=false (or not set)` を stdout に出して早期 return。
`heading_graph` は `symbol_graph` より優先される (`load_active_graph` の heading > symbol 優先順位)。
