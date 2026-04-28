# Institutional MT Eval v2 — PHOTON retrain step_003000 (Issue #135 / Phase 7)

**測定日**: 2026-04-28
**Checkpoint**: `checkpoints/photon_institutional_retrain_20260428/step_003000/` (val_loss=0.4777)
**Eval set**: `data/eval_sets/institutional_multi_turn_eval.jsonl` (30 sessions × 6 turns = 180 turns)
**Pipeline**: `configs/institutional_docs_photon_retrain.yaml` (provider=photon, JP:0.7/EN:0.3 mix で再学習)
**Wall-clock**: 約 42 分
**Raw eval log**: `logs/phase_7_institutional_photon_retrained_run1.jsonl`

---

## 結論

| 受入条件 | 値 | 判定 |
|---------|------|------|
| **Turn 5-6 NC < 6%** (MVP 最低) | **6.67%** | ❌ **0.67pp miss** (border) |
| Turn 5-6 NC < 3% (理想) | 6.67% | ❌ |
| **Latency 維持 (-30%+)** | follow-up p50 12,092 ms (vs baseline 19,426 ms = **-37.7%**) | ✅ **達成** |
| Overall NC | 8.33% | (参考、#113 11.39% から -26.9%) |

**総合**: **MVP threshold わずか未達** (6.67% > 6%)。改善幅は実測 **10.83% → 6.67% (-38.4%)** で大きいが境界線突破に至らず。

---

## Per-Turn NC 分布

| Turn | カテゴリ | NC count | NC rate |
|------|---------|---------|---------|
| 1 | definition (定義) | 2/30 | 6.67% |
| 2 | scope (適用範囲) | 0/30 | 0.00% ✅ |
| 3 | article_lookup (中核条項) | 0/30 | 0.00% ✅ |
| **4** | **penalty (罰則)** | **9/30** | **30.00%** ⚠️ |
| 5 | exception (例外) | 4/30 | 13.33% ⚠️ |
| 6 | overview (概観) | 0/30 | 0.00% ✅ |

### Turn 5-6 NC 詳細 (4 sessions)

`Turn 5 NC=True / Turn 6 NC=False` のみのケースのため、 **Turn 5 (例外) が支配的**:

| session | Turn 5 NC | Turn 6 NC | 元 doc 種別 (推定) |
|---------|----------|----------|------------------|
| INST-MT-003 | True | False | 例外規定が薄い doc |
| INST-MT-004 | True | False | 児童手当 (例外条項薄) |
| INST-MT-025 | True | False | (要調査) |
| INST-MT-028 | True | False | (要調査) |

### Turn 4 (罰則) NC 30% の根本原因

NC=True の 9 sessions の質問:

```
INST-MT-004: 児童手当の罰則 ← 児童手当法は administrative guidance、罰則章なし
INST-MT-012/13/14: 省エネ基準義務化の罰則 ← 省令ベース、明示罰則条項なし
INST-MT-015: 気候風土適応住宅の罰則 ← 任意制度、罰則対象外
INST-MT-020: 第4章罰則 ← 該当章存在せず
INST-MT-023/24/27: 保育/事業所内保育の罰則 ← 補助金制度、罰則対象外
```

**所見**: 9/9 とも **元 doc に「罰則」条項が存在しない or 該当なし** のケース。学習やモデル能力の問題ではなく、**eval set design の問題** (制度文書には罰則を持たない administrative guidance / 任意制度が多く、画一に「罰則は?」と聞くのが不適切)。

---

## Latency 分析

| Turn | p50 | mean | 備考 |
|------|-----|------|------|
| 1 | **19,198 ms** | 18,170 ms | retrieval + cold-start |
| 2 | 10,963 ms | 11,255 ms | 効率的 (キャッシュ有効) |
| 3 | 12,738 ms | 13,443 ms | |
| 4 | 10,603 ms | 11,626 ms | |
| 5 | 12,151 ms | 12,843 ms | |
| 6 | 14,184 ms | 13,858 ms | |
| **Overall** | **12,981 ms** | 13,473 ms | |
| **Follow-up (Turn 2+)** | **12,092 ms** | 12,605 ms | **vs baseline 19,426 ms = -37.7%** ✅ |

#113 Phase 1 (mulmoclaude 600-step PHOTON) では follow-up p50 = 10,707 ms (-44.9%) でした。今回 12,092 ms (-37.7%) と若干悪化しているが、**目標 -30% は十分クリア**。

---

## Pre-retrain との比較 (#113 / #148 Phase A 実測)

| 指標 | mulmoclaude 600-step PHOTON (#113) | 本 retrain step 3000 | Δ |
|------|-----------------------------------|--------------------|---|
| Overall NC | 11.39% | **8.33%** | **-26.9%** ✅ |
| **Turn 5-6 NC** | **10.83%** | **6.67%** | **-38.4%** ✅ |
| Follow-up p50 | 10,707 ms | 12,092 ms | +12.9% (悪化、許容範囲内) |
| Val_loss | 1.6238 | 0.4777 | -70.6% ✅ |

**改善幅は十分大きい**。mulmoclaude 600-step では Turn 5-6 NC > 6% で本格再学習が必要 (仮説 C) と判定されたが、本 retrain で **境界帯 (3-6%) に到達**。

---

## 採用判定: ❌ MVP 未達 (border) → 学習延長を推奨

設計方針書 §9 受入条件「Turn 5-6 NC < 6%」を厳密適用すると **0.67pp miss**。`未達なら max_steps=10000 への拡張学習を提案` のフローに従い、以下を提案:

### 推奨案 A (本命): 学習延長 → 5K step で再 eval

- val_loss が plateau 未到達 (-0.066/500 step 継続中)
- 残 -0.67pp の改善余地は十分ありそう (Turn 5 例外の 4 件のうち 1 件でも citation 取れれば 5%)
- ETA: +2,000 step × 12s = **+6.7h** GPU
- `resume_from=step_003000` で続き 5,000 まで → 中間 eval で再判定

### 推奨案 B (代替): 受入条件の柔軟運用

- Turn 5-6 NC 6.67% は #113 (10.83%) からの -38.4% 大幅改善
- Turn 5 だけで見ると 4/30 = 13.3% (Turn 6 は 0%、累計 6.67% は Turn 5 が支配)
- 「3K step adoption + 後続 wave で 5K-10K 拡張学習計画」として段階運用

### 推奨案 C: eval-set design 見直し

- Turn 4 罰則 30% NC は **eval set 設計の限界** (制度文書のうち罰則章がない doc が混在)
- 設計方針書 §11 リスク表 / #110 Issue で eval-set を refine する follow-up Issue 起票
- 本 retrain の本質的成果 (Overall NC -26.9%、Turn 5-6 -38.4%、val_loss -70.6%) は十分で MVP として採用妥当

### 推奨

**A (5K step 拡張学習) → 再 eval** が筋論。ただし B/C で「実質的成功」と位置付けるのも合理的。

ユーザー判断要請。
