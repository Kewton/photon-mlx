# Deployment Guide - Baseline RepoRAG

## System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| **OS** | macOS with Apple Silicon (M1/M2/M3) | M3 Ultra |
| **RAM** | 32 GB | 64 GB |
| **Python** | 3.12+ | 3.12+ |
| **Storage** | ~10 GB (model + indexes) | 20 GB+ |

> The Qwen2.5-Coder-14B-Instruct-4bit model loads approximately 8 GB into memory.
> Sentence-transformers embedding model and cross-encoder reranker require additional ~1 GB.

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
| `model.model_id` | LLM model | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | Auto-downloaded on first run |
| `retrieval.reranker.model_id` | Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Auto-downloaded on first run |
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
