# PHOTON 実用化ロードマップ

- **更新日**: 2026-04-24
- **目標**: PHOTON を MVP (Minimum Viable Product) として公開可能にする
- **MVP 定義**: pip install → 学習済みモデル自動 DL → 自分のリポジトリ / ドキュメントで使える
- **当初想定**: 5 週間（Phase 1-3）
- **現在の進捗**: Phase 1 完了、**Phase 2 完了（2026-04-28、Issue #135 採用）**、Phase 3 着手可能

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

| 項目 | 担当 Epic / Issue |
|------|------------------|
| ~~**Phase 2: 制度文書ドメイン精度検証**~~ | ~~Epic #117~~ ✅ **完了 2026-04-28 (#135 採用)** |
| **Phase 3: 配布**（pip install + HuggingFace 公開）| 未起票（Phase 2 判定後）|

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

## Phase 2: 制度文書ドメイン検証 — ✅ 完了（Epic #117、2026-04-28 #135 採用）

### Phase 2 完了サマリ (2026-04-28)

Issue #135 (本格再学習) で **Turn 5-6 NC < 6% MVP 達成** (refusal-aware では 0.00% で 3% 理想閾値も達成)。Phase 7 institutional MT eval の採用判定により Epic #117 close 条件を満たした。

| 指標 | Phase 2 受入条件 | 採用 retrain (#135) | 判定 |
|------|----------------|--------------------|------|
| 制度文書 Turn 5-6 NC | < 6% (MVP) | **0.00%** (refusal-aware) | ✅ |
| 制度文書 Turn 5-6 NC | < 3% (理想) | **0.00%** | ✅ |
| 制度文書 follow-up latency | baseline -30%+ | -37.7% | ✅ |
| 訓練品質 (val_loss) | (参考) | -70.6% (1.6238 → 0.4777) | ✅ |

**採用 checkpoint**: `photon_institutional_retrain_20260428/step_003000`
**エビデンス**:
- `reports/institutional_photon_mt_eval_v2_3k.md` (採用判定)
- `reports/institutional_photon_mt_eval_v2_3k_bug_check.md` (refusal-aware 検証)

**follow-up Issue**: #156 (`run_multi_turn_eval.py` の `is_refusal` 出力欠落の計測 bug)

### 方針の変更（2026-04-24）

当初計画では Django / Pydantic での code ドメイン汎用化を Phase 2 としていたが、**code フレームワーク間の変動は FastAPI での実証で十分と判断**。よりインパクトある **ドメイン越えの精度確認**として **制度文書（日本語、法律・条文、4,228 ファイル）** を対象に pivot。

### 対象コーパス

- **パス**: `/Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents`
- **規模**: 4,228 ファイル / 757,429 行
- **内容**: 日本語制度文書（法律、条文、報告書、docling PDF 変換済）
- metadata.json に source_url、第○条等の構造情報あり

### Epic と Sub-Issue

**Epic #117**: [Epic] Phase 2: 制度文書ドメインでの PHOTON RepoRAG 精度検証

| Group | Issue | タイトル | 依存 | 工数 |
|-------|-------|---------|------|------|
| **G1（並列着手可）** | #109 | feat(ingestion): markdown chunker + symbol graph conditional skip | なし | 3 日 |
| G1 | #110 | feat(eval): 制度文書 eval set 自動生成 | なし | 4 日 |
| G1 | #111 | feat(retrieval): query_expansion / noise_patterns 汎用化 | なし | 3 日 |
| **G2（G1 後）** | #112 | feat(configs): 制度文書プロファイル + index + baseline Static eval | G1 | 3 日 |
| **G3（クリティカル）** | **#113** | **measure: 現行 PHOTON で MT eval 実測（学習要否の判定根拠）** | #112 | 2 日 |
| G3（並列） | #114 | feat(retrieval): 多言語 embedding / reranker A/B | #112 | 4 日 |
| **G4（仕上げ）** | #115 | feat(app): wizard domain template + 日本語 prompt | #112, #114 | 2 日 |
| G4 | #116 | docs: Phase 2 完了レポート（再現率比較）| 全 Issue | 1 日 |

### PHOTON 再学習の判定（#113 で決定 → 仮説 B 確定 → #135 で本格再学習実施 → 採用）

#113（現行 PHOTON で制度文書 MT eval）の **Turn 5-6 NC** で以下を判定：

| Turn 5-6 NC | 判定 | 次アクション | 実績 |
|-------------|------|-------------|------|
| < 3% | 仮説 A 勝ち | 再学習不要、Phase 2 完了へ | — |
| 3-6% | 中間 | 軽量 fine-tune（5K step、resume）| — |
| **> 6%** | **仮説 B 勝ち** | **本格再学習（10-20K step、JP 50%+）** | ✅ #113 で 10.83% を実測、**#135 で実施 → 0.00% (refusal-aware) で完了** |

**推定仮説 A: 60% / 仮説 B: 40%** (Tokenizer 共有 × encoding 一貫性の同時不確実性) → 実測で **仮説 B 確定**、#135 で本格再学習実施し受入条件達成。

### 成功条件（Phase 2 完了 = Phase 3 着手可能）— ✅ 2026-04-28 達成

| 指標 | 基準 | 実測値 | 判定 |
|------|------|--------|------|
| 制度文書 Static NC | < 20% | 11.21% (#112) | ✅ |
| 制度文書 MT NC | < 15% | 8.33% raw / 0.00% refusal-aware (#135 step_003000) | ✅ |
| 制度文書 follow-up latency | baseline -20%+ | -37.7% (12,092 ms vs 19,426 ms、#135) | ✅ |
| **FastAPI 改善の再現率** | **70%+** | Indicator A 100% / B 110%（`reports/phase2_cross_domain_validation.md` §3） | ✅ |
| PHOTON Turn 5-6 NC | < 3% | 0.00% (refusal-aware) / 6.67% (raw、計測 bug 由来 #156) | ✅ |

### タイムライン

```
Week 1:
  並列 → #109 (chunker) + #110 (eval set) + #111 (query/noise)
        完了
        ↓
  直列 → #112 (config + baseline eval)

Week 2:
  並列 → #113 (PHOTON MT eval) + #114 (embedding/reranker A/B)
        完了
        ↓
  判定 → 再学習要否（conditional issue 新規登録 or skip）

Week 3:
  並列 → #115 (wizard) + (必要なら) 学習 Issue
        ↓
  直列 → #116 (Phase 2 完了レポート)
        ↓
  判定 → MVP Phase 3 進行可否
```

---

## Phase 3: 学習済みモデル配布 — ❌ 未着手（Phase 2 判定後）

### 目的

ユーザが学習なしで PHOTON を使える状態にする。

### タスク（Phase 2 完了後に Issue 化）

| # | タスク | 工数 |
|---|-------|------|
| 3-1 | 汎用コーパス拡張（20+ repos 相当の多様コーパス、制度文書含む）| 3 日 |
| 3-2 | 汎用モデル学習 | 2 日 |
| 3-3 | 未知 repo/doc での eval | 2 日 |
| 3-4 | HuggingFace 公開 | 1 日 |
| 3-5 | 自動 DL 機能実装 | 2 日 |
| 3-6 | pip パッケージ化 | 2 日 |

### 成功条件

- [ ] HuggingFace に weights 公開済み
- [ ] 学習データに含まれない repo/doc で eval → NC < 15%, latency -20%+
- [ ] `pip install photon-rag` で動作

### ユーザ体験（Phase 3 完了時）

```bash
pip install photon-rag
photon-rag ingest --source /path/to/project   # code / markdown 自動検知
photon-rag index --source /path/to/project
photon-rag serve --config my_config.yaml
photon-rag ask "..."
```

---

## Phase 4: ユーザ体験向上（MVP 後、オプション）

| # | タスク | 工数 | 現状 |
|---|-------|------|------|
| 4-1 | ワンコマンドセットアップ `photon-rag init --source .` | 3 日 | 未着手 |
| 4-2 | Streaming 応答（SSE）| 2 日 | 未着手 |
| 4-3 | Web UI | — | **達成**（Streamlit #82 merged）|
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
            Phase 2 Epic #117 登録、8 サブ Issue 登録済、並列着手準備完了

[Phase 2 スプリント（3 週間）]
Week 1:     G1 並列（#109 + #110 + #111）→ #112 統合
Week 2:     G3 並列（#113 + #114）→ 再学習要否判定
Week 3:     G4（#115 + 判定次第で 学習）→ #116 完了レポート
            ════════════ Phase 2 判定 ════════════

[Phase 3 スプリント（MVP リリース、2 週間）]
Week 4:     Phase 3-1/2 汎用コーパス + 汎用モデル学習
Week 5:     Phase 3-3/4/5/6 HuggingFace 公開 + pip 化
            ════════════ MVP リリース ════════════

[MVP 後]    Phase 4 UX（CLI 改善、VS Code Extension）
            Phase 5 運用（差分更新、Docker）
```

---

## マイルストーン別の「使える度」

| Phase | 完了後の状態 | 誰が使えるか | 2026-04-24 時点 |
|-------|------------|------------|----------------|
| Wave 1-6 + #82 | 開発者本人 + Streamlit UI。手動 ingest + プロジェクト登録 | 本人・チームメンバー（手動セットアップ）| ✅ **現状** |
| **Phase 2** | **コード repo + 制度文書どちらでも動作確認済み** | **社内 staff（Streamlit UI 経由）** | 🏃 Epic #117 進行中 |
| **Phase 3（MVP）** | **pip install + config だけで使える** | **Python 開発者全般 + 文書 RAG ユーザ** | ❌ 未達 |
| Phase 4 | ブラウザで質問するだけ | 非 Python 開発者も | 部分達成（Streamlit UI 有）|
| Phase 5 | チーム全体で常時運用 | 全員 | ❌ 未達 |

**現状は「本人 + チームメンバー（Streamlit UI 経由、code repo 限定）」まで到達**。Phase 2 完了で **制度文書 RAG としても使える状態** になる。

---

## リスクと対策（更新版）

| リスク | 影響 | 対策 | 2026-04-24 時点 |
|--------|------|------|----------------|
| **PHOTON が日本語で encoding 崩れる** | MVP 遅延 | **#113 で早期判定、fallback として fine-tune** | 最大リスク |
| 制度文書の構造特殊（附則、改正履歴）| chunk 品質低下 | #109 markdown_chunker で header 優先分割 | #109 で対処 |
| eval set の人手検証コスト大 | 品質 vs 速度 | #110 LLM 自動生成 + 20% サンプル検証 | #110 で対処 |
| 日本語 reranker のメモリ不足 | ハード制約 | #114 で軽量モデル併用検討 | #114 で対処 |
| 汎用モデルの品質が低い（Phase 3）| ユーザ体験悪化 | fine-tune オプションを残す | Phase 3 判定時 |
| Apple Silicon 以外で動かない | ユーザ限定 | PyTorch backend（`torch_ref`）| 優先度低 |
| pip パッケージ化で依存衝突 | インストール失敗 | optional dependencies で MLX/PyTorch 分離 | Phase 3 |
| Static NC 改善の停滞 | ユーザ体験 | #81 retrieval tuning + #88 harness | 継続課題 |

---

## 予算・リソース

| 項目 | 必要量 | 現状 |
|------|--------|------|
| 開発工数 | Phase 2: ~18 日 / Phase 3: ~10 日（計 ~6 週間）| 計画待ち |
| ハードウェア | Mac Studio M2/M3 Ultra | ✅ 利用中 |
| 外部費用 | なし（全てローカル + OSS）| ✅ |
| HuggingFace | 無料（public repo）| アカウント既設 |
| PyPI | 無料 | 未登録 |
| Phase 2 追加学習時間（条件付）| 最大 36h（Mac Studio）| 必要時のみ |

---

## Next Action（Phase 2 着手）

### 即時実行可能（/orchestrate 並列 3 Issue）

```
/orchestrate 109 110 111
```

Group 1（完全独立）を 3 worktree で並列開発。完了後 #112 に進む。

### 要計画（ユーザ判断）

- #113 の 2-run MT eval 実測タイミング（~3h の CPU/GPU 占有）
- PHOTON 再学習の実施可否（#113 結果次第）

### オプション（Phase 2 と独立）

- #81 retrieval tuning Epic（#88 harness で grid search 実行、制度文書にも転用可能）
- #49 PHOTON Medium 1B スケールアップ（MVP 後）

---

## 関連 Epic / Issue

### Open

- **Epic #117**: Phase 2 制度文書ドメイン検証（メイン）
- #109-#116: Phase 2 サブ Issue（8 件）
- Epic #81: Static NC < 15% 達成 retrieval tuning（Phase 2 と並走可能）
- #49: PHOTON Medium 1B（MVP 後）

### Closed（本日クローズ）

- #82: Streamlit UI Wave 1-4 機能反映（PR #107/#108 で main 反映）
- #93: Wave 6 Epic（single-PR + eval-gate プロトコル確立）
- #89, #103, #104: Wave 6 個別 Issue（empirical 検証完了）
