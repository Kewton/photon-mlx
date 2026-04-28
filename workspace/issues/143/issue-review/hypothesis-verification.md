# Issue #143 仮説検証レポート

実施日: 2026-04-28
対象 Issue: #143 fix(eval): institutional eval reproducibility — Qwen 14B nondeterminism causes ±1.7pt baseline drift between runs
検証ブランチ: `feature/issue-143-eval-reproducibility`

---

## 検証結果サマリー

| # | 仮説 | 判定 | 重要度 |
|---|------|------|-------|
| H1 | 既存スクリプトの存在と現状 (`scripts/run_baseline_eval.py` に seed/--runs なし、`scripts/aggregate_institutional_baseline.py` に multi-run 集計なし、`baseline_reporag/generation/generator.py` に deterministic 強制経路なし) | Confirmed | 高 |
| H2 | 現状の generation 設定 (`temperature=0.2, do_sample=False` で「greedy decoding 想定」だが、`make_sampler(temp=0.2)` 経路で完全な greedy にならない可能性) | Partially Confirmed | 高 |
| H3 | nondeterminism の発生源 (institutional `llm_client.py` には `mx.random.seed()` あるが、baseline eval には seed 固定なし — 経路が不統一) | Confirmed | 高 |
| H4 | V0 baseline drift の事実確認 (`reports/institutional_baseline_static.md` 11.21%、`reports/institutional_retrieval_ab.md` 12.93%、+1.72pt 乖離) | Confirmed | 中 |
| H5 | 関連 eval パイプライン (`scripts/run_multi_turn_eval.py` も同一 `build_pipeline` 経由で同種の nondeterminism 影響下) | Confirmed | 中 |
| H6 | 関連 Issue 連携 (Issue #135 seed perturb 実装、#138 tokenizer fix、#156 refusal-aware bug 用 `is_refusal_answer` 既存) | Confirmed | 低 |

---

## 各仮説の詳細検証

### H1: 既存スクリプトの存在と現状 — **Confirmed**

#### H1-1: `scripts/run_baseline_eval.py`
- ファイル存在確認: あり
- 引数: `--config`, `--eval-set`, `--max-questions`, `--output`, `--repo-id`, `--marker-file`
- **`--runs` 引数なし**
- **`import random`, `numpy`, `mlx.core` なし → seed 固定コード無し**

#### H1-2: `scripts/aggregate_institutional_baseline.py`
- ファイル存在確認: あり
- 引数: `--predictions (nargs="+")`, `--output`, `--section`, `--in-place`
- **multi-run aggregation 機能なし** (`load_predictions()` は単純に複数 JSONL を結合するだけ)

#### H1-3: `baseline_reporag/generation/generator.py`
- ファイル存在確認: あり (Issue 本文の `mlx_lm.py` ではなく `generator.py` が実体)
- L38-41: `make_sampler(temp=self._temperature, top_p=self._top_p)` のみ
- L50-56: `mlx_lm.generate(...)` 呼び出しに **seed パラメータ未渡し**
- **deterministic path 強制機構なし**

### H2: 現状の generation 設定 — **Partially Confirmed**

#### H2-1: `configs/institutional_docs.yaml`
- L191-196 generation: `temperature: 0.2`, `top_p: 0.9`, `repetition_penalty: 1.05`, `do_sample: false`
- L267-269 inference: `temperature: 0.2`, `do_sample: false` (重複設定あり)

#### H2-2: `configs/baseline.yaml`
- L253-259: `temperature: 0.2`, `do_sample: false` (一貫)

#### 重要な発見 — **「greedy decoding 想定」が実装上は不完全**
- `do_sample=false` フラグは設定されているが、`baseline_reporag/generation/generator.py:38-41` で `make_sampler(temp=0.2)` が呼ばれており、`mlx_lm.generate()` には **0.2 を取った sampler** が渡される
- 厳密な argmax greedy 経路への切替は実装されていない
- `temperature=0.0` への切替によって本当に greedy になるかは mlx-lm 内部の `make_sampler` 実装次第 (要確認: 現状コードベースでは temperature=0 への明示的 greedy 強制はない)

### H3: nondeterminism の発生源 — **Confirmed**

#### H3-1: `mx.random.seed()` 呼び出し箇所
- `baseline_reporag/eval/institutional/llm_client.py:105-111`:
  ```python
  if seed is not None:
      try:
          import mlx.core as mx
          mx.random.seed(seed)
      except Exception:
          pass
  ```
- → institutional eval LLMClient (Issue #135 Day 3) では seed 固定が実装済み

#### H3-2: baseline eval script での seed 設定
- `scripts/run_baseline_eval.py` 全体スキャン:
  - `import random`, `numpy`, `mlx.core` なし
  - `os.environ['PYTHONHASHSEED']` 設定なし
  - `random.seed()`, `np.random.seed()`, `mx.random.seed()` 呼び出しなし
- → **baseline eval pipeline には seed 固定機構が無い**

#### H3-3: 設計 gap
- institutional LLM generation: seed 固定済 (Issue #135 Day 3)
- baseline pipeline (mlx_lm.generate() を直接): seed 固定なし
- 両 eval script 間で seed handling が **不統一**

### H4: V0 baseline drift の事実確認 — **Confirmed**

#### H4-1: `reports/institutional_baseline_static.md`
- 実測日: 2026-04-25
- ブランチ/commit: `feature/issue-112-institutional-config` @ `644561c`
- L54-61: 全質問数 116, NC 件数 13, **NC rate = 11.21%**

#### H4-2: `reports/institutional_retrieval_ab.md`
- 実施日: 2026-04-26
- L14-15: V0 baseline 再計測 = **12.93%**
- L20-21: 「乖離 1.72pt は LLM (Qwen2.5-Coder-14B-Instruct-4bit) の生成揺らぎ範囲内 (`do_sample=False, temperature=0.2` でも nondeterminism は残る)」と明記

#### H4-3: コミット時期確認
- 両者 post-#125 / post-#126 (chunker AB 実施済) で 1 日間隔
- 同 V0 baseline の +1.72pt drift は **LLM nondeterminism 起因と確定**

### H5: 関連 eval パイプライン — **Confirmed**

#### H5-1: multi-turn eval の generation 経路
- `scripts/run_multi_turn_eval.py:55`: `pipeline = build_pipeline(cfg)`
- L96-100: `pipeline.query()` は generation を internal で呼び出す
- baseline eval と **同じ build_pipeline 経由** → 同一 nondeterminism 影響下
- seed 固定なし、--runs 引数なし

#### H5-2: evals/ 配下の再現性関連テスト
- `evals/tests/` ディレクトリは存在するが空
- 再現性 (determinism) regression test 未導入

### H6: 関連 Issue 連携 — **Confirmed**

#### H6-1: Issue #135 / #138 への参照
- `baseline_reporag/eval/institutional/generator.py:115-118`: 「Issue #135 Day 3: QwenMLXAdapter.generate(prompt, seed=N) is deterministic for a given prompt + seed pair...」
- `baseline_reporag/eval/institutional/multi_turn.py:160`: 「Issue #135 Day 3...」
- `baseline_reporag/photon_pipeline.py` L298, L354, L719, L733, L740, L771, L813: Issue #138 (tokenizer) 参照 6 件
- `baseline_reporag/tests/test_photon_pipeline_lazy_import.py`: 「Regression test for Issue #135 / DR1-002」

#### H6-2: Issue #156 (refusal-aware bug) との関係
- `baseline_reporag/citation.py:26-32`: `REFUSAL_PATTERNS` と `is_refusal_answer()` 実装済み
- `scripts/aggregate_institutional_baseline.py:62-65`: `is_no_citation` で `is_refusal_answer()` を組込済
- Issue #143 (nondeterminism) と Issue #156 (refusal detection) は **独立な実装**で並行対応可

---

## 重要な発見 (Stage 1 レビュー入力)

### Rejected には至らないが「主張の修正」が必要な点

1. **Issue 本文「影響ファイル」の `baseline_reporag/generation/mlx_lm.py` は誤り**
   - 実体は `baseline_reporag/generation/generator.py`
   - `mlx_lm.py` というファイルは存在しない (mlx-lm はライブラリ名)
   - Stage 2 で Issue 本文を修正すべき

2. **Issue 本文「mlx-lm の greedy 経路でも 4-bit 量子化 kernel-level non-determinism」**
   - これは Issue 著者の仮説で、コードベース上では検証不可 (mlx-lm 外部ライブラリの内部挙動)
   - 確実に言えるのは「seed 固定が無い + sampler が temperature=0.2 を消費している」こと
   - `Unverifiable` 部分があり、実機 ablation 実験 (Task 4 noise floor) でしか確認できない

3. **「temperature=0.2, do_sample=False」が「greedy decoding 想定」と本文に記載**
   - 設定値はそうだが、`generator.py:38-41` の `make_sampler(temp=0.2)` 経路では実質的に softmax-with-T=0.2 が使われており、厳密な argmax greedy ではない可能性
   - Stage 2 で「想定」表現を「実装は make_sampler 経由で sampler-based」に修正検討

### 受入条件への適合性評価

| 受入条件 | 現状 | ギャップ |
|---------|------|--------|
| Task 1: seed 固定追加 | 未実装 | `scripts/run_baseline_eval.py` `scripts/run_multi_turn_eval.py` に os/random/np/mx seed 設定が必要 |
| Task 2: temperature=0 採用または検証 | 設定値 0.2、global seed なし | 設定 + 実装の双方で対応必要 |
| Task 3: --runs N 引数追加 | 未実装 | scripts と aggregator に multi-run 対応 |
| Task 4: 10-run noise floor 計測 | 未実装 | noise floor 出力スクリプト不在 |
| 既存 eval scripts test 全パス | - | 実装後再検証 |

---

## Stage 1 レビューへの申し送り

### 優先度 A (Issue 成立に不可欠)

1. **影響ファイルの誤記訂正**: `baseline_reporag/generation/mlx_lm.py` → `baseline_reporag/generation/generator.py` に修正
2. **seed 固定機構の設計判断**: env-var 直接設定 vs config 駆動 (BaseConfig.run.seed) のいずれか
3. **mlx_lm.generate() への seed 受け渡し可否**: API 確認結果次第で代替戦略 (`mx.random.seed()` を generate 直前に毎回呼ぶ等) が必要
4. **temperature=0.2 据え置き vs 0.0 切替**: institutional corpus でのテスト実施計画を Issue に明記すべき

### 優先度 B (品質向上)

5. **institutional eval (`llm_client.py`) と baseline eval の seed handling 統一**: 両者 pattern を `mx.random.seed(seed)` ベースで統一
6. **evals/tests/ への determinism regression test 追加**: 同一 config + prompt 2-run で output 一致をアサート
7. **multi-turn eval 側にも `--runs N` を導入するかの判断**: Issue 範囲を baseline + multi-turn 双方とするか、baseline のみとするか
8. **noise floor 計測 (Task 4) の優先度**: optional / supplementary か必須要件か明確化 (~10h 計算は重い)

---

## 最終結論

Issue #143 の仮説は **概ね Confirmed**。Issue 提案の方向 (seed 固定 + multi-run + noise floor) は妥当だが、以下の追加対応が必要:

- 影響ファイル名の訂正 (`mlx_lm.py` → `generator.py`)
- 「greedy decoding 想定」の正確な表現への修正 (sampler-based 実装である事実の明示)
- mlx_lm.generate() の seed 引数対応可否の事前調査
- institutional eval と baseline eval の seed handling 統一方針
- determinism regression test の追加

これらは Stage 1 通常レビューで提起する。
