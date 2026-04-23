# PHOTON 実用化ロードマップ

- **更新日**: 2026-04-24
- **目標**: PHOTON を MVP (Minimum Viable Product) として公開可能にする
- **MVP 定義**: pip install → 学習済みモデル自動 DL → 自分のリポジトリで使える
- **当初想定**: 5 週間（Phase 1-3）
- **現在の進捗**: Phase 1 完了、Phase 2 未着手（Phase 3 へのボトルネック）

---

## 現在地（2026-04-24）

### ✅ 達成済み

| 項目 | 状態 | 備考 |
|------|------|------|
| PHOTON アーキテクチャ（Small 377M）| ✅ 実装完了 | val_loss 0.4525、600 step 学習済 |
| Hierarchical working memory | ✅ Wave 1-5 で本線統合 | 3 階層 + drift + aggregation 選択 |
| Evidence pruning | ✅ follow-up 16→8 chunks | |
| KV cache（#54）| ✅ merged | follow-up -34% 達成 |
| Safe RecGen controller | ✅ merged | 3 階層 drift トリガ |
| Multi-turn 完走率 | ✅ 180/180 全 variant | |
| Follow-up latency -30% | ✅ 達成（-34%） | spec 超過 |
| MT NC 改善 | ✅ baseline 9.4% → PHOTON 7.8% | -2〜4pp |
| Turn 5-6 NC 0% | ✅ 確定 | PHOTON 独自価値 |
| Gate 2 Go 判定 | ✅ | Gate 2 v4 最終 |
| Gate 3 判定 | ✅ Conditional Go | log 修正完了 |
| **Streamlit アプリ（#82）** | ✅ merged 2026-04-23 | drift panel / eval runner / wizard |
| **Wave 6 再着手サイクル** | ✅ 完結 2026-04-23 | single-PR + eval-gate で安定化 |

### ❌ 未達（MVP 実現に必要）

| 項目 | ブロッカー |
|------|----------|
| **Phase 2: 汎用化**（FastAPI 以外の検証）| 最大ボトルネック |
| **Phase 3: 配布**（pip install + HuggingFace 公開）| Phase 2 の blocker |
| 未知 repo での eval セット自動生成 | 未着手 |
| query_expansion / noise patterns の汎用化 | 未着手 |

---

## Phase 1: 品質保証 — ✅ 完了

### 達成項目

- Safe RecGen ログ修正 ✅
- fallback recall 計測スクリプト化 ✅
- Gate 3 判定 Conditional Go ✅
- テスト 832/834 pass（pre-existing 2 failure のみ、CLAUDE.md 既知）✅
- 異常入力 guardrail（Wave 2 で `_safe_id`, YAML safe_load 等）✅

### Wave 6 教訓の反映

- **batch merge 禁止**: single-PR + eval-gate 必須
- **2-run MT eval**: LLM 非決定性 ±4-5pp を平滑化
- **empirical 根拠なき default 変更禁止**: #89/#103/#104 いずれも abort / opt-in 維持

---

## Phase 2: 汎用化 — ❌ 未着手（MVP の主要タスク）

### 目的

**FastAPI 以外でも動くことを証明する**。現状は FastAPI に最適化されており、他 repo での有効性が未検証。

### タスク

| # | タスク | 工数 | 成果物 | 現状 |
|---|-------|------|--------|------|
| 2-1 | Django repo で eval | 3 日 | eval 結果レポート | 未着手 |
| 2-2 | Pydantic repo で eval | 2 日 | eval 結果レポート | 未着手 |
| 2-3 | query_expansion 汎用化 | 2 日 | FastAPI 固有マッピング分離 | 未着手 |
| 2-4 | noise patterns 設定化 | 1 日 | `_NOISE_PATTERNS` を config に | 未着手 |
| 2-5 | eval set 自動生成スクリプト | 3 日 | `scripts/generate_eval_set.py` | 未着手 |

### 成功条件

- [ ] Django / Pydantic で MT NC < 15%
- [ ] Django / Pydantic で follow-up latency baseline -20% 以上
- [ ] repo 切り替えが config 変更のみで可能

### 判定基準

FastAPI で得られた改善の **70% 以上** が他リポジトリでも再現すること。  
再現しなければ PHOTON の汎用性に問題があり、**Phase 3（配布）に進まない**。

---

## Phase 3: 学習済みモデル配布 — ❌ 未着手

### 目的

ユーザが学習なしで PHOTON を使える状態にする。

### タスク

| # | タスク | 工数 | 成果物 | 現状 |
|---|-------|------|--------|------|
| 3-1 | 汎用コーパス拡張（20+ repos）| 3 日 | `data/processed/train_universal.jsonl` | 未着手 |
| 3-2 | 汎用モデル学習 | 2 日 | `checkpoints/universal/` | 未着手 |
| 3-3 | 未知 repo での eval | 2 日 | 汎化性能レポート | 未着手 |
| 3-4 | HuggingFace 公開 | 1 日 | `kewton/photon-python-small` | 未着手 |
| 3-5 | 自動 DL 機能実装 | 2 日 | config の `photon_model_id` で自動取得 | 未着手 |
| 3-6 | pip パッケージ化 | 2 日 | `pip install photon-rag` | 未着手 |

### 成功条件

- [ ] HuggingFace に weights 公開済み
- [ ] 学習データに含まれない repo で eval → NC < 15%, latency -20%+
- [ ] `pip install photon-rag` で動作

### ユーザ体験（Phase 3 完了時）

```bash
pip install photon-rag
photon-rag ingest --repo /path/to/my/project
photon-rag index --repo /path/to/my/project
photon-rag serve --config my_config.yaml
photon-rag ask "認証処理の入口はどこ？"
```

---

## Phase 4: ユーザ体験向上（MVP 後、オプション）

| # | タスク | 工数 | 現状 |
|---|-------|------|------|
| 4-1 | ワンコマンドセットアップ `photon-rag init --repo .` | 3 日 | 未着手 |
| 4-2 | Streaming 応答（SSE）| 2 日 | 未着手 |
| 4-3 | Web UI | — | **部分達成**（Streamlit #82 merged）|
| 4-4 | VS Code Extension | 5 日 | 未着手 |
| 4-5 | Quick Start ドキュメント（5 ステップ）| 1 日 | **達成**（`app_guide.md` 330 行）|

---

## Phase 5: プロダクション運用（MVP 後、オプション）

| # | タスク | 工数 |
|---|-------|------|
| 5-1 | Index 差分更新（`git pull` 後の増分再 index）| 3 日 |
| 5-2 | Multi-user 対応（セッション分離, 認証）| 2 日 |
| 5-3 | 監視ダッシュボード（NC/latency 時系列）| 2 日 |
| 5-4 | ユーザ feedback（👍/👎）| 2 日 |
| 5-5 | Docker イメージ | 2 日 |

---

## 更新後タイムライン

```
[完了済み]  Phase 1 品質保証 → Gate 3 Conditional Go
            Wave 6 再着手サイクル → single-PR + eval-gate 定着
            #82 Streamlit app → Wave 2-4 機能 UI 反映

[現在]      2026-04-24 時点
            Phase 2 未着手（MVP 最大の残作業）

[次スプリント（2 週間）]
Week 1:     Phase 2-3 query_expansion / noise patterns 汎用化（4 日）
            Phase 2-5 eval set 自動生成スクリプト（3 日）
Week 2:     Phase 2-1 Django eval 実測（3 日）
            Phase 2-2 Pydantic eval 実測（2 日）
            Phase 2 判定（70%+ 再現？）

[MVP スプリント（2 週間）Phase 3]
Week 3:     Phase 3-1 汎用コーパス拡張（3 日）
            Phase 3-2 汎用モデル学習（2 日）
Week 4:     Phase 3-3 未知 repo eval（2 日）
            Phase 3-4/5/6 HuggingFace 公開 + 自動 DL + pip 化（5 日）
            ════════════ MVP リリース ════════════

[MVP 後]    Phase 4 UX（CLI 改善、VS Code Extension）
            Phase 5 運用（差分更新、Docker）
```

---

## マイルストーン別の「使える度」

| Phase | 完了後の状態 | 誰が使えるか | 2026-04-24 時点 |
|-------|------------|------------|----------------|
| Wave 1-6 + #82 | 開発者本人 + Streamlit UI。手動 ingest + プロジェクト登録 | 本人・チームメンバー（手動セットアップ）| ✅ **現状** |
| Phase 2 | FastAPI 以外でも動作確認済み | Python 開発者（手動セットアップ）| ❌ 未達 |
| **Phase 3（MVP）** | **pip install + config だけで使える** | **Python 開発者全般** | ❌ 未達 |
| Phase 4 | ブラウザで質問するだけ | 非 Python 開発者も | 部分達成（Streamlit UI 有）|
| Phase 5 | チーム全体で常時運用 | 全員 | ❌ 未達 |

**現状は「本人 + チームメンバー（Streamlit UI 経由）」まで到達**。MVP としては **Phase 2 と 3 が残り 4 週間分の作業**。

---

## リスクと対策（更新版）

| リスク | 影響 | 対策 | 2026-04-24 時点 |
|--------|------|------|----------------|
| Phase 2 で他 repo に汎化しない | MVP 遅延 | 汎用コーパスで再学習、repo-specific fine-tune を fallback に | 未検証、最大リスク |
| 汎用モデルの品質が低い | ユーザ体験悪化 | fine-tune オプションを残す | Phase 3 判定時に評価 |
| Apple Silicon 以外で動かない | ユーザ限定 | PyTorch backend 整備（`torch_ref` 活用）| 優先度低 |
| pip パッケージ化で依存衝突 | インストール失敗 | optional dependencies で MLX/PyTorch 分離 | Phase 3 に内包 |
| **Static NC 改善の停滞** | ユーザ体験 | **#81 retrieval tuning + #88 harness の grid 実行** | 新規リスク |
| **PHOTON 追加学習の ROI** | 汎用性 | まず現行 Small の 10K step 学習、その後 Medium 検討 | #49 と関連 |

---

## 予算・リソース

| 項目 | 必要量 | 現状 |
|------|--------|------|
| 開発工数 | Phase 2-3: ~4 人週（残作業）| 計画待ち |
| ハードウェア | Mac Studio M2/M3 Ultra | ✅ 利用中 |
| 外部費用 | なし（全てローカル + OSS）| ✅ |
| HuggingFace | 無料（public repo）| アカウント既設 |
| PyPI | 無料 | 未登録 |

---

## Next Action（MVP リリースに向けて）

### 即時実行可能（orchestrate / pm-auto-issue2dev 経由）

1. **#81 retrieval tuning Epic 着手**（#88 harness で grid search 実行）
2. **Phase 2-3 query_expansion 汎用化** — 新規 Issue 化して着手
3. **Phase 2-5 eval set 自動生成** — 新規 Issue 化

### 要計画（ユーザ判断）

- **Phase 2-1 Django / 2-2 Pydantic eval** — repo 選定と期間計画
- **Phase 3 汎用モデル学習** — データセット選定と 10K step 学習の実施可否

### オプション

- #49 PHOTON Medium 1B スケールアップ（MVP 後）
- find_relevant_past_turn の閾値チューニング（#103 follow-up）
