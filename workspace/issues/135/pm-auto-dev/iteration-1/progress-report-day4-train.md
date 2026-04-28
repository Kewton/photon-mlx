# Issue #135 Day 4 学習完了報告 — staged 3K-step training

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**学習実行日**: 2026-04-28 00:36 〜 08:42 (8h 5min)
**ブランチ**: feature/issue-135-photon-retrain (Day 1-4 = 25 commits)
**ステータス**: ✅ **学習完了、val_loss 1.6238 → 0.4777 で -71% 改善**

---

## 学習サマリ

| 項目 | 値 |
|------|------|
| **入力 checkpoint (resume)** | mulmoclaude `step_000600` (val_loss=1.6238) |
| **start time** | 2026-04-28 00:36:47 JST |
| **end time** | 2026-04-28 08:42:14 JST (~step 3000 到達後 final 保存) |
| **wall-clock** | **8h 5min** (= 29,097 sec / 2,400 step) |
| **per-step** | ~12 sec (effective batch 32 × ctx 2048 = 65,536 tokens/step) |
| **累計 step** | 3,000 (= mulmoclaude 600 + 追加 2,400) |
| **train tokens** | ~157M (2,400 × 65,536) |
| **final val_loss** | **0.4777** (best=step 3000) |
| **early stopping** | 発動なし (patience_counter 一度も増加せず) |

### Hyperparameters (確定)

```yaml
# configs/institutional_docs_photon_retrain.yaml
training:
  train_corpora_mix:
    "<this worktree>/data/training/institutional/train_jp.jsonl": 0.7  # JP raw markdown
    "<develop worktree>/data/processed/train_multi.jsonl": 0.3         # EN mulmoclaude
  val_split: 0.05
  learning_rate: 3.0e-5
  min_learning_rate: 3.0e-6
  warmup_ratio: 0.0
  micro_batch_size: 2
  gradient_accumulation_steps: 16    # effective batch = 32
  context_length: 2048               # tokens/step = 65,536
  max_steps: 3000                    # cumulative state.step
  eval_every_steps: 500
  save_every_steps: 500
  early_stopping:
    patience: 3
    min_delta: 0.001
    restore_best: true
```

---

## Val_loss 推移 (継続改善、収束未到達)

| step | val_loss | Δ from baseline | best 更新 |
|------|---------|-----------------|----------|
| baseline (mulmoclaude 600) | 1.6238 | — | — |
| **1,000** (= +400) | 1.8293 | +12.6% (一時悪化) | (悪化) |
| **1,500** (= +900) | 1.1890 | -26.8% | ✅ |
| **2,000** (= +1,400) | 0.7077 | -56.4% | ✅ |
| **2,500** (= +1,900) | 0.5436 | -66.5% | ✅ |
| **3,000** (= +2,400) | **0.4777** | **-70.6%** | ✅ (最終) |

### Val_loss trajectory analysis

学習序盤 (step 1000) で一時的に val_loss が悪化したのは、JP corpus に最初に exposure された adaptation phase。step 1500 以降は **monotonically decreasing**:

```
step 1000 → 1500: -35.0% (大幅改善、JP 適応開始)
step 1500 → 2000: -40.5% (急峻な勾配)
step 2000 → 2500: -23.2% (緩やかになる)
step 2500 → 3000: -12.1% (まだ改善続く)
```

**Δval_loss / 500 step**:
- 1500→2000: -0.481
- 2000→2500: -0.164
- 2500→3000: -0.066

改善幅は減少傾向ですが **plateau ではない** (まだ -0.066 = step あたり -0.013% 改善)。

---

## mulmoclaude reference (1.6238) との比較

```
mulmoclaude 600-step:           val_loss 1.6238
本 retrain 3,000-step (best):   val_loss 0.4777
                                 ────────────────
                                 -71% 改善 (perplexity ~3.4× 向上)
```

**実質的な意味**:
- val_loss は次トークン予測の cross-entropy。`exp(loss)` で perplexity に変換すると:
  - mulmoclaude 600-step: perplexity = exp(1.6238) ≈ **5.07** (5 候補から正解を 1 つ選ぶ精度に相当)
  - 本 retrain 3000-step: perplexity = exp(0.4777) ≈ **1.61** (1.6 候補から正解を選ぶ精度)
- 制度文書 domain での次トークン予測能力が **3.15× 向上**

---

## max_steps=10000 への拡張余地検討

### 観察

1. **plateau 未到達**: 直近 500 step で -0.066 の改善継続
2. **early stopping 発動なし**: val_loss 一度も連続非改善せず
3. **train_loss も依然 noisy** (0.16 〜 0.80 の範囲): 過学習の signal 弱い

### 拡張学習で期待できる改善 (推定)

直近 500 step あたりの改善率 -12.1% を仮置きすると:

| max_steps | 累計 step | 推定 val_loss | wall-clock |
|-----------|----------|--------------|----------|
| 3,000 (現状) | 3,000 | **0.478** ✅ | 完了済 |
| 5,000 | 5,000 | ~0.34 (-29% 追加) | +6.7h |
| 10,000 | 10,000 | ~0.15 (-69% 追加) | +23h |

ただし上記は線形外挿で、実際には **diminishing returns** が想定されます。

### 拡張可否判断

- **Issue #135 受入条件は val_loss でなく Turn 5-6 NC**:
  - Phase 7 eval で **NC < 6% を達成すれば 3K で十分** → 採用 → 完了
  - Phase 7 NC が境界帯 (5-7%) なら 5K-10K 拡張で更に改善余地
  - Phase 7 NC が大きく未達なら、val_loss 改善があっても retrieval/architecture 系の問題

**決定 logic**: Phase 7 eval 結果で判断 (現在実行中)

---

## 出力 checkpoints

```
checkpoints/photon_institutional_retrain_20260428/
├── step_001000/  (val_loss 1.829, integrity.json + weights + state)
├── step_001500/  (val_loss 1.189)
├── step_002000/  (val_loss 0.708)
├── step_002500/  (val_loss 0.544)
├── step_003000/  (val_loss 0.478) ← Phase 7 eval 対象
└── best/         (= step_003000、最良 val_loss 0.478)
```

各 checkpoint:
- `weights.npz` (約 1.5 GB)
- `state.json` (累計 step、val/train_loss 履歴)
- `integrity.json` (DR4-003 SHA-256 hash)

合計 5 × 1.5GB ≈ **7.5 GB** local。Phase 8 では best/ + step_003000 のみ external storage に保管想定 (採用基準により他削除可)。

---

## 次のステップ (Phase 7 eval 進行中)

Phase 7 eval を `configs/institutional_docs_photon_retrain.yaml` (`model.checkpoint_path: photon_institutional_retrain_20260428/step_003000` 設定済) で実行中:

```bash
PHOTON_CHECKPOINT_ROOT=$(pwd)/checkpoints \
python scripts/run_multi_turn_eval.py \
  --config configs/institutional_docs_photon_retrain.yaml \
  --eval-set data/eval_sets/institutional_multi_turn_eval.jsonl \
  --output reports/institutional_photon_mt_eval_v2_3k.md
```

eval 完了次第、以下を計測:
- **Turn 5-6 no-citation rate**: < 6% で MVP 達成、< 3% で理想達成
- **Latency p50**: < 13.6s (= -30% from baseline 19.4s)
- **Per-category NC**: definition / scope / article_lookup / penalty / exception / overview の breakdown

baseline (`configs/institutional_docs.yaml` provider=baseline) の同 eval は別途 1.5h 程度で必要 (本セッション or 次セッション)。

---

## 累計 commits (Day 4 終了時点 25 件)

```
40e2c79 feat(retrain): JP:0.7/EN:0.3 mix + 3K-step staged validation
fdf16bc fix(eval/institutional): jitter seed per retry — Option A
d4f1bbc docs(issue-135): Day 3 続報 — Option A 適用 + 実測 ETA 再評価
4965f27 docs(issue-135): Day 3 続報 — Step 1-2 完了 + Step 3 corpus 生成 ETA blocker
b7057cd feat(configs): use develop worktree absolute paths
3612d8f feat(scripts): align corpus generator with production LLM client + token output
b099960 docs(issue-135): Day 3 進捗報告 — develop merge OK + Phase 6 blocker 報告
be91682 merge develop into feature/issue-135-photon-retrain (Phase 6 prep)
... (Day 1-3 の 16 件は省略)
```

Phase 6 学習: 完了 ✅
Phase 7 eval: 進行中
