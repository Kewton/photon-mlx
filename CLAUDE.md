# CLAUDE.md

このドキュメントはClaude Code向けのプロジェクトガイドラインです。

---

## プロジェクト概要

### 基本情報
- **プロジェクト名**: PHOTON-RepoRAG
- **説明**: PHOTON系の階層working memoryを使った巨大repo向けmulti-turn RepoRAGの高速化・省メモリ化
- **リポジトリ**: https://github.com/Kewton/photon-mlx

### 技術スタック
| カテゴリ | 技術 |
|---------|------|
| **言語** | Python 3.12+ |
| **ML基盤** | MLX (Apple Silicon), PyTorch (reference) |
| **LLMバックエンド** | mlx-lm (Qwen2.5-Coder-14B-Instruct-4bit) |
| **検索** | BM25 (rank-bm25), sentence-transformers |
| **サーバ** | FastAPI, uvicorn |
| **テスト** | pytest |
| **リント** | ruff |

---

## ブランチ構成

### ブランチ戦略
```
main (本番) <- PRマージのみ
  |
develop (受け入れ・動作確認)
  |
feature/*, fix/*, hotfix/* (作業ブランチ)
```

### 命名規則
| ブランチ種類 | パターン | 例 |
|-------------|----------|-----|
| 機能追加 | `feature/<issue-number>-<description>` | `feature/7-fix-retrieval-quality` |
| バグ修正 | `fix/<issue-number>-<description>` | `fix/1-no-citation-rate` |
| 緊急修正 | `hotfix/<description>` | `hotfix/critical-security-fix` |
| ドキュメント | `docs/<description>` | `docs/update-readme` |

---

## 標準マージフロー

### 通常フロー
```
feature/* --PR--> develop --PR--> main
fix/*     --PR--> develop --PR--> main
hotfix/*  --PR--> main (緊急時のみ)
```

### PRルール
1. **PRタイトル**: `<type>: <description>` 形式
   - 例: `feat: add retrieval doc filter`
   - 例: `fix: resolve no-citation rate`
2. **PRラベル**: 種類に応じたラベルを付与
   - `feature`, `bug`, `documentation`, `refactor`
3. **レビュー**: 1名以上の承認必須（main向けPR）
4. **CI/CD**: 全チェックパス必須

### コミットメッセージ規約
```
<type>(<scope>): <subject>

<body>

<footer>
```

| type | 説明 |
|------|------|
| `feat` | 新機能 |
| `fix` | バグ修正 |
| `docs` | ドキュメント |
| `style` | フォーマット（機能変更なし） |
| `refactor` | リファクタリング |
| `test` | テスト追加・修正 |
| `chore` | ビルド・設定変更 |
| `ci` | CI/CD設定 |
| `perf` | パフォーマンス改善 |

---

## コーディング規約

### Python
- `ruff check` で警告ゼロを維持
- `ruff format --check` でフォーマット差分なしを維持
- `pytest` で全テスト通過を維持
- 型ヒントを推奨（`from __future__ import annotations`）

### Code Review Checklist

PR レビュー時に **production code path** に scaffolding 命名 (`_Stub`, `_Mock`, `_Dummy`, `_Placeholder` 等) が残っていないか確認する。詳細は [`docs/code_review_checklist.md`](docs/code_review_checklist.md) を参照 (Issue #140 / S7-001 follow-up)。

### モジュール構成
```
baseline_reporag/       # Baseline RepoRAG (プロダクト線)
├── ingestion/          # ファイル抽出・chunking・SQLite store
├── indexing/           # BM25・embedding・symbol graph
├── retrieval/          # hybrid retrieval・graph expansion
├── memory/             # session memory
├── generation/         # evidence pack・prompt・mlx_lm generator
├── eval/               # 評価データ基盤
│   └── institutional/  # 制度文書 eval set 生成 (LLMClient / ChunkLookup / 2-layer citation grader)
├── contracts.py        # MLX-free 共有型 (QueryResult)
├── pipeline.py         # 共通クエリパイプライン (Qwen)
├── pipeline_factory.py # provider 分岐 factory (lazy MLX import)
├── photon_pipeline.py  # PHOTON-enhanced pipeline (opt-in PHOTON 生成)
├── profiler.py         # latency + memory profiling
├── citation.py         # [C:N] 解析
├── server.py           # FastAPI server
└── cli.py              # CLI

photon_mlx/             # PHOTON 階層デコーダ (研究開発線)
├── blocks.py           # TransformerBlock + RoPE (MLX)
├── model.py            # PhotonModel (bottom-up + top-down)
├── inference.py        # session inference + drift tracking
├── session.py          # PhotonSessionState + DriftMetrics + TurnState + WorkingMemoryConfig
├── safe_recgen.py      # Safe RecGen controller
├── loss.py             # next-token + recursive loss
├── trainer.py          # training loop + checkpoint
├── data.py             # JSONL → pack → batch
├── optimize.py         # Mac optimization utilities
└── tests/              # 168 tests

torch_ref/              # PyTorch reference LM (正しさ確認用)
├── model.py            # MinimalLM (LLaMA-style)
├── config.py           # PhotonConfig (共有)
└── tests/              # 11 tests

scripts/                # ユーティリティスクリプト
bench/                  # Benchmark ハーネス
evals/                  # 評価スキーマ + grader
demo/                   # デモシナリオ
configs/                # YAML 設定ファイル
reports/                # ベンチマークレポート
```

---

## 品質チェック

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全テストパス (約 507/509、残り 2 件は `tests/test_generate_training_corpus.py` の既知の pre-existing failure) |
| リント | `ruff check .` | 警告0件 |
| フォーマット | `ruff format --check .` | 差分なし |
| Baseline疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり |

---

## プロダクトライン

- **プロダクトライン**: baseline_rag + PHOTON institutional retrain (#135 採用済)
- **現在のメトリクス** (Gate 2 v6 = #135 Phase 8 採用、出典: `reports/institutional_photon_mt_eval_v2_3k.md`):
  - **採用 PHOTON checkpoint**: `photon_institutional_retrain_20260428/step_003000` (val_loss=0.4777)
  - **Turn 5-6 no-citation rate (refusal-aware)**: **0.00%** (MVP < 6% / 理想 < 3% を共に達成)
  - **Follow-up latency p50**: 12,092 ms (-37.7% vs baseline 19,426 ms)
  - **Val_loss**: 0.4777 (-70.6% vs mulmoclaude 600-step 1.6238、perplexity 5.07 → 1.61)
  - 訓練データ: institutional_documents 4,228 docs + mulmoclaude train_multi (JP:0.7/EN:0.3)
  - エビデンス: `reports/institutional_photon_mt_eval_v2_3k_bug_check.md` (refusal-aware 検証)
  - 計測 bug: `scripts/run_multi_turn_eval.py` の `is_refusal` 出力欠落は Issue #156 で別途対応
- **過去メトリクス参考**:
  - Gate 2 v4 (#113 / mulmoclaude 600-step): Turn 5-6 NC 10.83%, follow-up p50 10,707 ms
  - Gate 2 v5 (#148 Phase A): institutional baseline 7.78% NC、PHOTON random-init bug 修正
- **運用ドキュメント**: `docs/deployment.md`, `docs/troubleshooting.md`

---

## スラッシュコマンド（Claude Code用）

| コマンド | 説明 |
|----------|------|
| `/multi-stage-issue-review` | Issue記載内容の多段階レビュー（通常→影響範囲）×2回と指摘対応を自動実行（Codex 担当 Stage 5-8 必須） |
| `/multi-stage-design-review` | 設計書の4段階レビュー（通常→整合性→影響分析→セキュリティ）と指摘対応を自動実行（Codex 担当 Stage 3-4 必須） |
| `/pm-auto-issue2dev` | Issueレビューから実装完了まで完全自動化（Issueレビュー→設計→設計レビュー→作業計画→TDD実装） |
| `/pm-auto-design2dev` | 設計レビューから実装完了まで完全自動化（設計→設計レビュー→作業計画→TDD実装） |
| `/work-plan` | Issue単位の作業計画立案 |
| `/tdd-impl` | テスト駆動開発で実装 |
| `/pm-auto-dev` | Issue開発を完全自動化（TDD→テスト→報告） |
| `/bug-fix` | バグ調査・修正を自動化 |
| `/create-pr` | PR自動作成（タイトル・説明自動生成） |
| `/worktree-setup` | Issue用Git Worktree環境構築 |
| `/worktree-cleanup` | Worktree環境のクリーンアップ |
| `/progress-report` | 開発進捗レポート作成 |
| `/refactoring` | コード品質改善 |
| `/acceptance-test` | 受入テスト検証 |
| `/orchestrate` | 複数Issue並列開発オーケストレーション |
| `/uat` | ユーザー受入テスト（HTMLレポート生成） |

---

## サブエージェント

| エージェント | モデル | 役割 |
|-------------|--------|------|
| tdd-impl-agent | opus | TDD実装スペシャリスト |
| acceptance-test-agent | opus | 受入テスト検証 |
| refactoring-agent | opus | コード品質改善 |
| progress-report-agent | sonnet | 進捗レポート作成 |
| investigation-agent | opus | バグ原因調査 |

---

## 禁止事項

- `main` への直接プッシュ禁止
- `force push` 禁止（自分のブランチを除く）
- テストなしのマージ禁止
- ruff警告の放置禁止
- `.env` や credentials の commit 禁止
