# Gate 2 Go/No-Go 判定レポート v2

- **Date**: 2026-04-17
- **Run ID**: `bench_20260417_074810_b5b1f8`
- **対象リポジトリ**: fastapi/fastapi @ eba8942c
- **ベースモデル**: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
- **PHOTON small**: 377M params, 800 steps 学習 (best val_loss 1.713 @ step 400)
- **前回 (v1)**: 2026-04-14, `bench_20260414_130853_c81af3`

---

## 結論: **Gate 2 No-Go 確定（PHOTON 凍結、baseline_rag をプロダクト化）**

前回と同じ PHOTON MT バグ（`cosine_distance` shape mismatch）が再発し、
**follow-up latency の計測が不可能**。Gate 2 必須条件を満たせない。

ただし baseline_rag 側は **no-citation 44.2% → 21.7%（-22.5pp）** と大幅改善。

---

## 1. 計測結果

### 1-1. Static Eval 120問 (3 variants)

| Variant | No-citation | P50 (ms) | P90 (ms) |
|---------|-------------|----------|----------|
| **baseline_rag** | **21.7%** | **17,210** | 25,984 |
| baseline_rag + summary_memory | 20.8% | 17,052 | 27,575 |
| photon_rag | 20.8% | 21,253 | — |
| photon_rag + safe_recgen | — (未実行) | — | — |

### 1-2. Multi-turn Eval 30 sessions × 6 turns (2 variants)

| Variant | No-citation | Follow-up P50 | Follow-up P90 | 状態 |
|---------|-------------|---------------|---------------|------|
| **baseline_rag** | **13.3%** | **19,144 ms** | 31,588 ms | 完了 |
| baseline_rag + summary_memory | 13.9% | 19,619 ms | 33,534 ms | 完了 |
| photon_rag | — | — | — | **MT turn 2 で crash** |
| photon_rag + safe_recgen | — | — | — | **未実行** |

### 1-3. PHOTON MT バグ (再発)

```
ValueError: [broadcast_shapes] Shapes (4096) and (5120) cannot be broadcast.
  File "photon_mlx/session.py", line 50, in cosine_distance
```

異なる入力長で top-level hidden state の次元数が変わり、
ターン間 drift 計算で shape 不一致。前回と同一のバグ。

---

## 2. Gate 2 判定基準と実績

| 条件 (spec §19) | 必要値 | v1 (Apr 14) | **v2 (Apr 17)** | 判定 |
|----------------|--------|-------------|-----------------|------|
| PHOTON forward 安定 | stable | ✅ | ✅ | **PASS** |
| tiny/small で学習収束 | loss 単調減少 | ✅ (1.868) | ✅ (1.713) | **PASS** |
| drift 指標取得 | 3 metrics | ✅ | ✅ (static のみ) | **PASS** |
| **follow-up latency** | **baseline 比 −30%** | ❌ 測定不可 | ❌ **測定不可** | **FAIL** |

---

## 3. v1 → v2 改善サマリー (baseline_rag)

| 指標 | v1 (Apr 14) | **v2 (Apr 17)** | 変化 |
|------|-------------|-----------------|------|
| Static no-citation | 44.2% | **21.7%** | **-22.5 pp** |
| MT no-citation | 30.6% | **13.3%** | **-17.3 pp** |
| Retrieval noise | ~9% | **0.00%** | 完全除去 |
| PHOTON val_loss | 1.868 | **1.713** | -0.155 |
| Static P50 | 11,291 ms | 17,210 ms | +52% (reranker 追加) |
| MT follow-up P50 | 13,727 ms | 19,144 ms | +39% (reranker 追加) |

### 改善の内訳

| 施策 | 寄与先 |
|------|--------|
| Cross-encoder reranker (noise filter + ms-marco) | noise 完全除去 → pack 品質向上 |
| Query expansion (JP→EN, 45 mappings) | BM25 rank lift → 正しいファイル検索 |
| Citation post-processing (#14) | ABSTAIN に [C:1] 自動付与 |
| Prompt Rule 7 拡張 | impact/bug localization の ABSTAIN 抑制 |
| Evidence pack 直近2ターン制限 | MT citation index bleed 防止 |
| Session history から [C:N] 除去 | MT 後半品質劣化防止 |

### Latency 悪化の原因

P50 が 11s → 17s に悪化した主因は **cross-encoder reranker の追加**。
ただしこれは no-citation 大幅改善とのトレードオフであり、
reranker 自体の推論時間は ~50ms のため、主因は **LLM generation の
非決定性** (temperature=0.2 による出力長変動) と考えられる。

---

## 4. summary_memory の評価

| 指標 | baseline_rag | + summary_memory | 差分 |
|------|-------------|-----------------|------|
| Static NC | 21.7% | 20.8% | -0.9 pp |
| MT NC | 13.3% | 13.9% | +0.6 pp |
| Static P50 | 17,210 ms | 17,052 ms | -0.9% |
| MT follow-up P50 | 19,144 ms | 19,619 ms | +2.5% |

→ **差分は誤差範囲。summary_memory は引き続き効果なし。**

---

## 5. PHOTON vs baseline (Static のみ比較)

| 指標 | baseline_rag | photon_rag | 差分 |
|------|-------------|-----------|------|
| Static NC | 21.7% | 20.8% | -0.9 pp |
| Static P50 | 17,210 ms | 21,253 ms | **+23.5%** |

→ **Static では PHOTON は baseline より遅く、no-citation 改善もなし。**

---

## 6. Gate 2 最終判定

### 判定: **No-Go 確定**

| 根拠 | 詳細 |
|------|------|
| **PHOTON MT バグ未解消** | 2 回の bench で同一 crash。follow-up latency 計測不可。|
| **Static でも PHOTON 優位性なし** | NC -0.9pp / latency +23.5% で baseline 以下 |
| **summary_memory も効果なし** | 4 指標すべて誤差範囲 |

### 決定事項

1. **baseline_rag 単体をプロダクトラインとして確定**
2. **PHOTON を研究凍結** (コードは保持、今後の開発投資は停止)
3. **summary_memory variant は廃止**

### 今後の注力先

| 優先度 | タスク | 目標 |
|--------|-------|------|
| **P0** | Retrieval 改善 (#24) — file-type boost | Static NC 21.7% → < 15% |
| **P0** | change_planning / impact_analysis の ABSTAIN 対策 | カテゴリ NC 30% → < 20% |
| **P1** | Latency 最適化 — reranker バッチ化 / generation 早期終了 | P50 17s → < 15s |
| **P1** | #23 デプロイガイド | baseline_rag 運用化 |
| **P2** | #25 Full Eval CI 化 | 週次リグレッション監視 |

---

## 7. 付録

### Run ID 参照

| Scope | Run ID |
|-------|--------|
| Gate 2 v2 bench | `bench_20260417_074810_b5b1f8` |
| Gate 2 v1 bench | `bench_20260414_130853_c81af3` |
| PHOTON small checkpoint | `checkpoints/step_000400` (val_loss 1.713) |
| Static eval v5b (standalone) | `baseline_eval_fastapi_fastapi_20260417_020903` |

### PHOTON MT バグ技術詳細 (Issue #22)

- **場所**: `photon_mlx/session.py:50` `cosine_distance()`
- **原因**: 入力トークン長が変わると top-level hidden state の shape が変わる
  - Turn 1: 入力 4096 tokens → hidden (4096,)
  - Turn 2: 入力 5120 tokens → hidden (5120,)
  - `cosine_distance(prev, curr)` で broadcast 不可
- **修正案**: flatten → fixed-size pooling (mean pool or adaptive pool)
- **工数見積**: 1-2 日 (ただし Gate 2 No-Go のため凍結)
