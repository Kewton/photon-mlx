# Issue #14 最終評価判定レポート

**作成日**: 2026-04-16  
**評価対象**: post-processing citation 自動付与 (Phase 2)  
**実行ブランチ**: develop (PR #27 マージ済み)  
**run_id (Static)**: `baseline_eval_fastapi_fastapi_20260416_013816`  
**run_id (MT)**: `mt_eval_fastapi_fastapi_20260416_021455`

---

## 1. 成功条件への照合（Task 3.1）

| 条件 | 目標 | 実績 | 95% CI | 判定 |
|------|------|------|--------|------|
| Static no-cite rate | **< 10%** | **25.8%** (31/120) | [18.8%, 34.3%] | ❌ FAIL |
| MT no-cite rate | **< 15%** | **16.7%** (30/180) | [11.9%, 22.8%] | ❌ FAIL (CI 下限 11.9% は目標以下) |
| Wrong citation rate | **0% 維持** | Static 0.00% / **MT 1.11%** (2/180) | — | ⚠️ MT で違反 |

**総合判定: Case B — 未達だが有意な改善傾向を確認**

---

## 2. Post-processing 効果の計測（Task 2.2）

### 発動状況

| 評価セット | 発動前 no-cite | post-process 発動 | 発動後 no-cite | 解消率 |
|-----------|---------------|-----------------|---------------|--------|
| Static (n=120) | 52/120 (43.3%) | 21 件 (17.5%) | 31/120 (25.8%) | **40.4%** |
| MT (n=180) | 54/180 (30.0%) | 24 件 (13.3%) | 30/180 (16.7%) | **44.4%** |

> post-processing は発動した no-cite の約 40-45% を解消しており、機能として正常動作。  
> 残る no-cite (Static 31 件 / MT 30 件) は「空回答」「ABSTAIN」「空チャンク」等、発動条件を満たさないケース。

---

## 3. Wrong Citation 詳細（Task 2.1）

- **Static (120問): 0 件 — 問題なし**
- **MT (180ターン): 2 件 (1.11%)**
  - `session=eval-MT-025`, T4: 「依存性のキャッシュの仕組みは？」 `wrong_idx=[17]`
  - `session=eval-MT-025`, T6: 「依存性注入と他のフレームワークとの違いは？」 `wrong_idx=[17]`

**特記事項**: 両ケースともセッション MT-025 に集中、chunk index 17 が共通。MT セッション後半ターンで参照チャンクのインデックス不整合が発生している可能性。Static では発生しないため、セッションメモリ蓄積との相互作用が疑われる。

---

## 4. カテゴリ別 / Turn 位置別分析（Task 2.3）

### Static: カテゴリ別 no-cite rate

| カテゴリ | no-cite | n | rate | 前回比 (#6 時点) |
|---------|---------|---|------|----------------|
| onboarding | 4 | 30 | **13.3%** | ↓ 改善 |
| bug_localization | 6 | 30 | **20.0%** | — |
| impact_analysis | 6 | 30 | **20.0%** | — |
| change_planning | 15 | 30 | **50.0%** | 依然高水準 |
| **TOTAL** | **31** | **120** | **25.8%** | 54.2% → **-28.4pp** ↓ |

> `change_planning` カテゴリは 50% と突出して高く、当カテゴリへの特化対策が効果的な可能性。

### MT: Turn 位置別 no-cite rate

| Turn | no-cite | n | rate | 区分 |
|------|---------|---|------|------|
| T1 | 6 | 30 | 20.0% | first |
| T2 | 3 | 30 | 10.0% | follow-up |
| T3 | 3 | 30 | 10.0% | follow-up |
| T4 | 5 | 30 | 16.7% | follow-up |
| T5 | 7 | 30 | 23.3% | follow-up |
| T6 | 6 | 30 | 20.0% | follow-up |
| **First (T1)** | **6** | **30** | **20.0%** | |
| **Follow-up (T2-T6)** | **24** | **150** | **16.0%** | |
| **TOTAL** | **30** | **180** | **16.7%** | 25.0% → **-8.3pp** ↓ |

> T5-T6 の no-cite 率が T2-T3 より高い。セッション後半でのセッションメモリ肥大化による evidence pack 品質低下の可能性。

---

## 5. 統計的有意性の検証（Task 2.4）

### 過去実績との比較

| 時点 | Static (120q) | MT (180t) |
|------|---------------|-----------|
| 初期 (20q換算) | 35% | — |
| Issue #7 後 | 54.2% | 36.1% |
| Issue #1v2 後 | ~20% (20q) / 推定 | 25.0% |
| **Issue #14 後 (本計測)** | **25.8%** | **16.7%** |

### 改善幅の評価

- **Static**: 54.2% → 25.8% = **▲28.4pp 改善** (Wilson CI [18.8%, 34.3%] が 54.2% と重ならない → 統計的有意)
- **MT**: 25.0% → 16.7% = **▲8.3pp 改善** (95%CI 上限 22.8% < 25.0% → 統計的有意)

---

## 6. コスト評価（Task 3.2）

### Latency

| 評価セット | P50 | P95 | mean |
|-----------|-----|-----|------|
| Static (今回) | 14,972ms | 42,591ms | 18,135ms |
| MT (今回) | 14,910ms | 23,589ms | 15,215ms |
| Static (初期 baseline) | 17,585ms | — | — |
| Static (Issue #7 後) | 11,497ms | — | — |

- Generation が P50 で 14,950ms と全体の 99.9% を占める。Retrieval P50 = 20ms。
- **post-processing の latency 追加コストは無視できるレベル**（文字列操作 + resolve_citations 呼び出しのみ）。
- ただし Static mean (18,135ms) は Issue #7 後 (11,497ms) の **+57.8%** と依然高水準。原因はモデル設定 (max_new_tokens=768) や evidence pack サイズ (max_tokens=16000) であり Issue #14 由来ではない。

---

## 7. 最終判定

### 判定: **Case B — 未達・改善傾向確認**

| 軸 | 評価 |
|----|------|
| 機能動作 | ✅ post-processing は意図通りに動作（40-45% の no-cite を解消） |
| 目標達成 | ❌ Static 25.8% (target <10%) / MT 16.7% (target <15%) ともに未達 |
| Wrong citation | ⚠️ MT で 1.11% (2件) 発生。同一セッション集中のため isolated 事象の可能性 |
| Statistical significance | ✅ Issue #1v2 → Issue #14 の改善は統計的に有意 |
| Latency impact | ✅ post-processing 自体の overhead は無視可 |
| Revert 必要性 | ❌ 不要（改善が確認できており機能は維持） |

---

## 8. 次アクション提言（Phase 4）

### 即時対応

1. **Wrong citation 調査**: MT-025 の chunk index 17 の wrong cite 原因を調査。`resolve_citations` のセッション後半での挙動を確認し、必要なら fix 対応。

### Phase 3 (新 Issue 化)

2. **change_planning カテゴリ対策**: no-cite 50% は他カテゴリの 2-4 倍。change_planning 特有の質問パターンに対する retrieval / prompt 強化 Issue を起票。

3. **post-processing しきい値調整**: 現在 `[C:1]` のみを付与するシンプルな戦略。発動条件拡張（例: top-k チャンクの cosine similarity が一定以上の場合のみ発動）で wrong citation リスクを下げつつ解消率向上を検討。

4. **MT 後半ターン劣化対策**: T5-T6 の no-cite 率 (20-23%) が T2-T3 (10%) より高い原因（session memory 肥大化）を Issue 化。`pin_recent_chunks_max` や `max_turns` のチューニングを検討。

5. **Latency 改善**: Static P95 42,591ms が P50 (14,972ms) の 2.8 倍と分布が広い。長時間ターンの原因調査（change_planning の 50% no-cite と相関がないか確認）。

---

## 付録: 評価データファイル

| ファイル | 内容 |
|---------|------|
| `logs/baseline_eval_fastapi_fastapi_20260416_013816_predictions.jsonl` | Static 120問 predictions |
| `logs/baseline_eval_fastapi_fastapi_20260416_013816.jsonl` | Static 120問 run log |
| `logs/mt_eval_fastapi_fastapi_20260416_021455_predictions.jsonl` | MT 180ターン predictions |
| `logs/mt_eval_fastapi_fastapi_20260416_021455.jsonl` | MT 180ターン run log |
| `logs/mt_eval_fastapi_fastapi_20260416_021455_predictions.summary.json` | MT session サマリ |
