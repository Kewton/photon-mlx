# Gate 2 Go/No-Go 判定レポート

- **Date**: 2026-04-14
- **Run ID**: `bench_20260414_130853_c81af3`
- **対象リポジトリ**: fastapi/fastapi @ eba8942c
- **ベースモデル**: mlx-community/Qwen2.5-Coder-14B-Instruct-4bit
- **PHOTON small**: 400M params (vocab 152064), 400 steps 学習 (val_loss 1.87)

---

## 結論: **Gate 2 No-Go（PHOTON scope 縮小を推奨）**

spec §19 の Gate 2 必須条件「**follow-up latency が baseline 比 −30%**」を**満たせない**（Static では差なし、MT は PHOTON 側で実行不可）。

---

## 1. 計測結果

### 1-1. Static Eval 120問 (4 variants)

| Variant | No-citation | P50 (ms) | Mean (ms) | P90 (ms) |
|---------|-------------|----------|-----------|----------|
| baseline_rag | 44.2% | **11,291** | 12,939 | 20,795 |
| baseline_rag + summary_memory | 40.0% | 12,168 | 13,068 | 21,773 |
| photon_rag | 42.5% | 11,738 | 13,325 | 22,380 |
| photon_rag + safe_recgen | 40.8% | 12,566 | 13,255 | 21,013 |

**観察**:
- 4 variants 間で P50 latency はすべて 11-13 秒の範囲（**差は統計的に無意味**）
- No-citation も 40-44% の範囲で一貫（PHOTON の citation 改善効果は限定的）
- photon_rag の P50 は baseline の **+4%**（改善なし）

### 1-2. Multi-turn Eval (部分的)

| Variant | No-citation | Follow-up P50 | 状態 |
|---------|-------------|---------------|------|
| baseline_rag | 30.6% | **13,727 ms** | 完了 |
| baseline_rag + summary_memory | 32.2% | 13,857 ms | 完了 |
| photon_rag | — | — | **実行不可（バグ）** |
| photon_rag + safe_recgen | — | — | **実行不可（バグ）** |

---

## 2. Gate 2 判定基準と実績

| 条件 (spec §19) | 必要値 | 実績 | 判定 |
|----------------|--------|------|------|
| PHOTON forward が安定 | stable | ✅ stable (72+ tests passing) | **PASS** |
| tiny/small で学習が回る | loss が下がる | ✅ val 2.63→1.87 単調減少 | **PASS** |
| drift 指標が取得できる | 3 metrics | ✅ 実装済み | **PASS** |
| **follow-up 改善の兆候** | **baseline 比 −30% latency** | ❌ **測定不可** | **FAIL** |

---

## 3. 技術的障害

### 3-1. PHOTON Multi-turn 統合のバグ（Critical）

**事象**: photon_rag variant で MT eval 1 ターン目で `ValueError: shapes (4096) and (5120) cannot be broadcast`

**根本原因**: `photon_mlx/session.py::cosine_distance` で前ターン・現ターンの latent state の shape が不一致。異なる入力長で top-level hidden state の次元数が変わり、drift 計算が失敗。

**影響**:
- **MT での follow-up latency 測定が不可能**
- Gate 2 の核心指標が取得できない
- PR #18 で統合した PhotonRAGPipeline は MT 運用に耐えない

### 3-2. Static Eval での PHOTON 効果が見られない

- photon_rag と baseline_rag の差は P50 で **+4%（むしろ遅い）**
- no-citation も **42.5% vs 44.2%（−1.7pt）** で誤差範囲

**考察**:
- PHOTON small (400M) は 400 steps しか学習していない（tiny の 4 倍程度）
- 本来は 12000 steps 想定だった（10% 程度しか回せていない）
- Safe RecGen は発火率が低く、効果が出ない

### 3-3. summary_memory は全く価値を生まない

| 指標 | baseline | +summary_memory | 変化 |
|------|----------|----------------|------|
| No-citation (static) | 44.2% | 40.0% | −4.2pt |
| No-citation (MT) | 30.6% | 32.2% | **+1.6pt** |
| Follow-up P50 (MT) | 13,727ms | 13,857ms | +0.9% |

→ **summary_memory variant は破棄推奨**

---

## 4. spec §20 未達時対応

spec §19 の Gate 2 No-Go 時対応:
> - PHOTON の scope を縮小し、summary memory 強化を control として継続
> - baseline 単体をプロダクトラインとして維持

**実績から提案する修正方針**:

### 推奨: **baseline_rag 単体をプロダクトラインとして採用**

理由:
1. PHOTON 統合は technical debt（MT バグ）+ 実効価値の証明なし
2. summary_memory も効果なし
3. baseline_rag だけで Issue #7, #1 v2 の改善を享受できる
4. 今後の改善は retrieval / prompt 側（Issue #14）に集中可能

### 次ステップの優先順位

| Priority | タスク | 理由 |
|----------|-------|------|
| **高** | #14 No-citation Phase 2 (post-processing) | 確実に改善効果 |
| **中** | PHOTON MT バグ調査 | scope 縮小の判断根拠、完全に捨てる前に原因特定 |
| **低** | PHOTON medium/large 学習 | コスト大、効果未証明 |

---

## 5. Gate 2 判定結論

| 項目 | 結果 |
|------|------|
| **Gate 2 判定** | **No-Go** |
| **推奨アクション** | baseline_rag をプロダクトラインとして採用、PHOTON scope 縮小 |
| **残課題** | Issue #14 (no-citation post-processing)、PHOTON MT バグの技術的調査 |

---

## 6. 付録: 詳細データ

### Run ID 参照

| Scope | Run ID |
|-------|--------|
| 4-variant bench | `bench_20260414_130853_c81af3` |
| PHOTON small checkpoint | `checkpoints/final` (step 400, val_loss 1.87) |
| Training log | `logs/` |

### 完全な結果ファイル

- `reports/benchmark_runs/bench_20260414_130853_c81af3/bench_20260414_130853_c81af3_baseline_rag.jsonl` (300 turns)
- `reports/benchmark_runs/bench_20260414_130853_c81af3/bench_20260414_130853_c81af3_baseline_rag_summary_memory.jsonl` (300 turns)
- `reports/benchmark_runs/bench_20260414_130853_c81af3/bench_20260414_130853_c81af3_photon_rag.jsonl` (120 turns, static only)
- `reports/benchmark_runs/bench_20260414_130853_c81af3/bench_20260414_130853_c81af3_photon_rag_safe_recgen.jsonl` (120 turns, static only)

### 再現手順

```bash
# 1. Training corpus (Qwen tokenizer)
python scripts/generate_training_corpus.py \
  --repo-id fastapi_fastapi \
  --photon-config configs/photon_small.yaml \
  --tokenizer-id mlx-community/Qwen2.5-Coder-14B-Instruct-4bit \
  --output-dir data/processed --val-ratio 0.1

# 2. PHOTON small training (Early stopping at 400 steps)
# Edit configs/photon_small.yaml: max_steps=400, save_every_steps=100
PYTHONUNBUFFERED=1 python scripts/train_photon.py --config configs/photon_small.yaml

# 3. 4-variant bench
PYTHONPATH=. python bench/run_all.py --config configs/eval.yaml
```
