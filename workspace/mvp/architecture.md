# MVP アーキテクチャ

**更新日**: 2026-04-24

## 現在の稼働構成

```
リポジトリ クローン → Streamlit app (port 3012) または CLI で利用
  │
  ├── app/photon_app.py               # Streamlit UI
  │   ├── components/drift_panel.py        # Wave 3 drift 可視化
  │   ├── components/turn_history_panel.py # Wave 3 会話履歴
  │   ├── components/eval_panel.py         # Wave 4 eval runner
  │   └── components/wizard.py             # Wave 2-4 toggle wizard
  │
  ├── baseline_reporag/               # ← main code path
  │   ├── ingestion/                       # chunker + SQLite store
  │   ├── indexing/                        # BM25 + embedding + symbol graph
  │   ├── retrieval/                       # hybrid + graph expansion + reranker
  │   ├── memory/session.py                # SessionManager (baseline 側 turn history)
  │   ├── generation/                      # evidence_pack + prompt + mlx_lm generator
  │   ├── pipeline.py                      # baseline Qwen-only pipeline
  │   ├── pipeline_factory.py              # provider 分岐 (baseline / photon)
  │   ├── photon_pipeline.py               # PHOTON 拡張 pipeline
  │   ├── server.py                        # FastAPI
  │   └── cli.py                           # CLI エントリ
  │
  ├── photon_mlx/                     # ← PHOTON モデル
  │   ├── model.py / blocks.py             # 階層 encoder + RoPE
  │   ├── session.py                       # PhotonSessionState + DriftMetrics
  │   ├── inference.py                     # hierarchical prefill + prune
  │   ├── safe_recgen.py                   # drift ベース fallback
  │   ├── trainer.py / data.py             # 学習ループ
  │   └── tests/                           # 517 tests
  │
  ├── configs/                        # プロファイル一覧
  │   ├── baseline.yaml                    # Qwen only
  │   ├── photon_small.yaml                # PHOTON Small（推奨 default）
  │   ├── photon_tiny.yaml                 # Tiny モデル
  │   ├── photon_long_context.yaml         # 長コンテキスト用
  │   └── photon_600m_paper.yaml           # 研究参考
  │
  └── scripts/                        # オペレーション
      ├── run_baseline_eval.py             # Static NC eval (120Q)
      ├── run_multi_turn_eval.py           # MT NC eval (180Q)
      ├── retrieval_grid_search.py         # #88 grid search harness
      └── train_photon.py                  # PHOTON 学習 (resume 対応)
```

### MVP 配布目標（Phase 3 完了時の想定）

```
pip install photon-rag        # ← 未着手 (Phase 3)
  │
  ├── HuggingFace から自動 DL:
  │   ├── kewton/photon-python-small    # 未公開
  │   ├── Qwen2.5-Coder-14B-4bit        # 公開済
  │   ├── all-MiniLM-L6-v2              # 公開済
  │   └── ms-marco-MiniLM-L-6-v2        # 公開済
  │
  └── CLI / FastAPI / Streamlit で利用
```

---

## モデル構成（ローカル配置）

| モデル | サイズ | 役割 | 入手 |
|--------|--------|------|------|
| **Qwen2.5-Coder-14B-4bit** | ~8 GB | 回答生成（LLM）| HF 自動 DL |
| **PHOTON Small** | ~1.5 GB | session state + drift + prune | 学習で生成 |
| **all-MiniLM-L6-v2** | ~100 MB | Embedding 検索 | HF 自動 DL |
| **ms-marco-MiniLM-L-6-v2** | ~100 MB | cross-encoder 再ランク | HF 自動 DL |
| （オプション）multilingual-e5-small | ~120 MB | 日本語対応 Embedding | HF 自動 DL |

---

## データフロー（現行実装）

### 初回セットアップ

```
リポジトリ → ingest (ast-chunker) → chunks.db (SQLite)
          → BM25 (rank-bm25) + embedding (MiniLM) → インデックス
          → symbol graph (import/call) → symbol_graph.json
```

### Turn 1（初回質問）

```
質問 → query expansion → BM25 + embedding (hybrid)
     → graph expansion → cross-encoder reranker → top 16 chunks
     → PHOTON hierarchical prefill → coarse state 保存
     → evidence pack → LLM (Qwen 14B) → 回答 [C:N]
     → turn_history に記録
```

### Turn 2+（follow-up）

```
質問 → query expansion → BM25 + embedding (reranker skip 可)
     → PHOTON prune_evidence (coarse state 経由 cosine で絞り込み)
     → 8 chunks（default）
     → PHOTON KV cache 活用で prefill skip
     → LLM (Qwen 14B) → 回答 [C:N]
     → drift 検知 (3 階層 + topic shift) → Safe RecGen 判定
```

---

## PHOTON セッション状態管理（本プロダクトの核心）

### Working Memory（`photon_mlx/session.py`）

```python
PhotonSessionState
  ├── turn_history: list[TurnState]          # Turn 1..N の記録
  │   └── TurnState                           # question_text, coarse_states, timestamp, turn_id
  ├── current_state: LevelStates             # 現在の階層状態 (token, mid, top)
  ├── drift_history: list[DriftMetrics]      # 各 turn の drift 値
  └── aggregation: weighted / attention / last / dynamic
```

- **decay_factor=0.5** で過去ターン重み減衰
- **max_turns=8** 超過時は自動圧縮（storage_mode: full / top_level_only）
- **aggregation 選択肢**: weighted（推奨、default）/ attention / last / dynamic (turn_position / drift_based / hybrid)

### Drift 検知（`DriftMetrics`）

| 階層 | 検知対象 | 用途 |
|------|---------|------|
| `latent_cosine_drift_token` | token-level | 細かい表現変化 |
| `latent_cosine_drift_mid` | mid-level | 意味的変化 |
| `latent_cosine_drift_top` | top-level | 話題転換 |
| `topic_shift_score` | 統合スコア | Safe RecGen トリガ |

閾値超過時は **Safe RecGen controller** が fallback action を選択（`re_retrieve` / `reprefill_hierarchy` / `fallback_to_baseline_path`）。

---

## システム要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| OS | macOS 14+ (Apple Silicon) | macOS 15 |
| RAM | 32 GB | 64 GB |
| Storage | 15 GB | 30 GB |
| Python | 3.12+ | 3.12+ |
| GPU | Apple Silicon M1+ | M2 Ultra / M3 Ultra |

---

## 現行パフォーマンス（2026-04-23 実測）

| シナリオ | baseline | PHOTON（weighted default） | 効果 |
|---------|---------|----------------------------|------|
| Static NC (120Q) | 16.7% | 17.5% | ±ノイズ |
| MT NC (180Q) | 9.4% | **7-8%** | **-2〜4pp** |
| Turn 5-6 NC | ~6.7% | **0.0%** | **working memory が完全有効** |
| First-turn latency | 22.3s | 22.2s | ±同等 |
| Follow-up latency (P50) | 20.1s | **13-14s** | **-34%（KV cache #54）**|

詳細は `metrics.md`。
