# Failure Cases

このファイルは benchmark 実行中に発見された失敗ケースを記録する。
テンプレートは `reports/failure_case_template.md` を参照。

**最終更新**: 2026-04-13 (Full eval 120q static + 30 MT sessions)

---

## 統計サマリー

### 全体

| 指標 | Static 120問 | Multi-turn 180ターン | 合計 |
|---|---|---|---|
| no-citation | 65/120 (54.2%) | 77/180 (42.8%) | 142/300 (47.3%) |
| wrong citation | 0/120 (0%) | 2/180 (1.1%) | 2/300 (0.7%) |

### No-citation の根本原因分類

| 原因 | Static | Multi-turn | 合計 | 比率 |
|---|---|---|---|---|
| **generation_failure** | 51 | 53 | **104** | **73.2%** |
| **retrieval_low_quality** | 14 | 24 | **38** | **26.8%** |

> 分類基準: evidence pack 内の `.py` ファイル比率が 20% 未満 → retrieval_low_quality、20% 以上 → generation_failure

### Static: カテゴリ別 no-citation 内訳

| Category | Total | No-cite | Rate | generation | retrieval |
|---|---|---|---|---|---|
| onboarding | 30 | 9 | 30.0% | 3 | 6 |
| bug_localization | 30 | 12 | 40.0% | 8 | 4 |
| impact_analysis | 30 | 19 | 63.3% | 16 | 3 |
| change_planning | 30 | 25 | 83.3% | 24 | 1 |

### Multi-turn: Turn 位置別 no-citation 内訳

| Turn | No-cite | Rate | generation | retrieval |
|---|---|---|---|---|
| T1 (first) | 9/30 | 30.0% | 9 | 0 |
| T2 | 13/30 | 43.3% | 11 | 2 |
| T3 | 16/30 | 53.3% | 16 | 0 |
| T4 | 9/30 | 30.0% | 3 | 6 |
| T5 | 15/30 | 50.0% | 8 | 7 |
| T6 | 15/30 | 50.0% | 6 | 9 |

> 後半ターン (T4-T6) で retrieval_low_quality が増加。session memory に蓄積された cited chunks の重複やトピック遷移が retrieval 品質を低下させている可能性。

---

## 改善優先度 Top 5

### Priority 1: change_planning の generation_failure (24件)

- **影響**: 全 no-citation の 16.9% (24/142)
- **特徴**: 「設計案」「変更計画」等の提案型質問。evidence pack には .py ファイルが豊富に含まれるが、モデルが設計提案に集中し citation を省略
- **対策案**: 
  - change_planning 専用の prompt 指示（「設計案でも根拠となるコードを [C:N] で引用すること」）
  - post-processing: citation なし回答に evidence pack の最関連 chunk を自動付与

### Priority 2: impact_analysis の generation_failure (16件)

- **影響**: 全 no-citation の 11.3% (16/142)
- **特徴**: 「変更した場合の影響は？」形式の質問。モデルは影響範囲を列挙するが、具体的なコード箇所の citation が不足
- **対策案**: impact_analysis 向けに「影響先の各モジュールを [C:N] で引用すること」を指示

### Priority 3: Multi-turn 後半 (T4-T6) の retrieval_low_quality (22件)

- **影響**: MT no-citation の 28.6% (22/77)
- **特徴**: session 後半でトピック遷移が起きた際に retrieval が追従できず、.py 以外のファイルが evidence pack に入る
- **対策案**:
  - retrieval の session-aware weighting（直近ターンのトピックに重み付け）
  - graph expansion のスコープ拡大

### Priority 4: onboarding の retrieval_low_quality (6件)

- **影響**: Static で 6 件
- **特徴**: 「テスト構成」「ミドルウェア」「静的ファイル配信」等、fastapi の非コアモジュールに関する質問で .py が取れていない
- **対策案**: 
  - retrieval の file-type boost（.py ファイルにスコア補正）
  - include パターンの見直し

### Priority 5: Multi-turn T1 の generation_failure (9件)

- **影響**: MT first turn で 9/30 が no-citation
- **特徴**: Static の onboarding T1 と同等の問題。evidence pack は適切だが citation を省略
- **対策案**: Issue #1 で試みた few-shot examples アプローチを改良（evidence header の毎ターン挿入を避ける設計）

---

## 個別 Failure Cases

### FC-001: 依存性注入の説明で citation なし

| 項目 | 値 |
|---|---|
| run_id | baseline_eval_fastapi_fastapi_20260412_232239 |
| eval_id | SE-ONB-003 |
| カテゴリ | no-citation assertion |
| 根本原因 | generation_failure |
| ステータス | 未対処 |

### FC-002: change_planning カテゴリの系統的 citation 省略

| 項目 | 値 |
|---|---|
| run_id | baseline_eval_fastapi_fastapi_20260413_135703 |
| eval_ids | SE-CHA-001〜030 のうち 25件 |
| カテゴリ | no-citation assertion (系統的) |
| 根本原因 | generation_failure |
| 特徴 | 設計提案型の回答で citation を省略。evidence pack の .py 比率は 44〜100% と十分 |
| ステータス | 未対処 |

### FC-003: Multi-turn 後半の retrieval 品質低下

| 項目 | 値 |
|---|---|
| run_id | mt_eval_fastapi_fastapi_20260413_135708 |
| 発生位置 | T4-T6 (22件) |
| カテゴリ | retrieval_low_quality |
| 根本原因 | session memory 蓄積に伴う retrieval のトピック追従不足 |
| ステータス | 未対処 |

### FC-004: wrong citation (MT で 2件)

| 項目 | 値 |
|---|---|
| run_id | mt_eval_fastapi_fastapi_20260413_135708 |
| カテゴリ | wrong_citation |
| 根本原因 | 調査中 |
| 件数 | 2/180 (1.1%) |
| ステータス | 未対処 |

---

## Run ID Reference

| Run | ID | Scope |
|-----|-----|-------|
| Static 120q | `baseline_eval_fastapi_fastapi_20260413_135703` | full 120q |
| Multi-turn 30sess | `mt_eval_fastapi_fastapi_20260413_135708` | full 30 sessions |
