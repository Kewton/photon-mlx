## 背景

S7-001 (PHOTON eval が random-init weight で動作) + #138 (tokenizer mismatch) の 2 つの critical bug を 2026-04-26 に解消した (#141, #146, #147 merged)。これにより **過去の PHOTON eval 結果はすべて無効化** され、改めて「真の PHOTON のベースライン」を確立する必要がある。

加えて、本機会に LLM backbone も最新モデルに更新し、現行 `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` と新候補 2 件を同条件で比較評価する。

## ⚠️ 影響を受けた過去結果 (信頼できなくなった範囲)

| 過去結果 | 信頼度 | 理由 |
|---|---|---|
| Gate 2 v1-v4 PHOTON 数値 (NC 6.7%, val_loss 等) | ❌ 無効 | PhotonModel が random-init + StubTokenizer で動作 |
| #113 制度文書 PHOTON NC 11.39% | ❌ 無効 | 同上 |
| Wave 4-6 PHOTON eval | ❌ 無効 | 同上 |
| PHOTON Drift metrics の数値 | ❌ 無効 | 乱数空間で計算 |
| Safe RecGen trigger 発火率 | ❌ 無効 | 乱数 drift に対する反応 |

(Latency 系数値は推論経路最適化由来のため weight 非依存、信頼できる。Baseline 単独 eval も photon_pipeline を経由しないため信頼できる。#137 V4 retrieval A/B も同様に信頼できる。)

## 🚫 ブロック対象

- **#135 Phase 6-8 (本格再学習)**: 「真の PHOTON ベースライン」が未確定なため、再学習前後の Apple-to-Apple 比較が成立しない。本 Issue 解消が **#135 GPU 着手前の必須前提**。

## ゴール

3 つの LLM × 2 つの代表 eval set で **真の PHOTON ベースラインを再確立** し、#135 再学習の比較基準と将来の LLM 戦略決定基盤を作る。

## アプローチ

### Phase A: Sanity Check (最優先、blocker for #135) — Qwen 2.5 + 修正後 PHOTON

修正後の PHOTON pipeline が正常に動作することを最小工数で確認する。

| 対象 | LLM | eval set | 想定 compute |
|---|---|---|---|
| Gate 2 v5 | mlx-community/Qwen2.5-Coder-14B-Instruct-4bit (現行) | FastAPI MT eval (5 repos × 8 questions × 6 turns) | ~1.5h |
| Issue #113 v2 | 同上 | Institutional MT eval (30 sessions × 6 turns = 180 turns) | ~1.5h |

成果物:
- `reports/gate2_judgment_v5_post_s7001.md` (新規) — Gate 2 v4 と同形式、ただし PHOTON 真の数値で更新
- `reports/institutional_photon_mt_eval_v2.md` (新規) — #113 と同形式、ただし PHOTON 真の数値で更新
- `configs/institutional_docs_photon.yaml`: `model.checkpoint_path` に既存 mulmoclaude 600-step ckpt を明示設定
  - **checkpoint 所在: TBD — 担当者確認待ち (Phase A 着手前必須)。val_loss 0.4525 達成時の保存先 (リポ内 path / 共有サーバ / HF hub URL) を Phase A 最初の subtask として特定・記録すること。**

判定:
- もし真の PHOTON が baseline 比 NC 改善 → S7-001 仮説 (random-init で過小評価されていた) 裏付け
- もし真の PHOTON が baseline 比 NC 悪化 (#113 と同程度の +4.44pp) → アーキ自体に課題、#135 再学習で改善が期待される
- いずれの結果でも #135 着手判断材料として確定

### Phase B: 新 LLM への拡張 (Phase A 完了後に着手、strategic)

> **注意**: Phase B は Phase A 完了後に着手すること (GPU 競合回避)。

> **前提確認 (Phase B 着手前必須)**: 各 model_id を `huggingface-cli repo info <id>` で存在確認し、不在時は最も近い既存 mlx-community model に切替えて本 Issue を更新すること。下記 model_id は **仮置き** であり、Phase A 開始前に `huggingface-cli repo info` で正式 slug を確認する。不在時は近い alternative に切替。
> - `mlx-community/Qwen3.5-9B-MLX-8bit`: Qwen3 系の正式 mlx-community 命名は `Qwen3-x` 系であり `Qwen3.5` は非標準呼称の可能性あり。正式 slug (例: `mlx-community/Qwen3-8B-MLX-8bit` 等) を確認・修正すること。
> - `mlx-community/gemma-4-26b-a4b-4bit`: Google 公式 release 名 (例: `mlx-community/gemma-3-27b-it-4bit` または `gemma-4-...` の正規 slug) を確認・修正すること。

| 対象 LLM | provider | quant | params | 想定 compute (eval 1 dataset) |
|---|---|---|---|---|
| **mlx-community/Qwen3.5-9B-MLX-8bit** ※仮 | qwen3.5 | 8bit | 9B (smaller than current 14B) | ~30-45 min/dataset |
| **mlx-community/gemma-4-26b-a4b-4bit** ※仮 | gemma4 | 4bit MoE | 26B total / 4B active | ~45-60 min/dataset |

各 LLM × 2 dataset (FastAPI MT + Institutional MT) × baseline + PHOTON × 2 runs = **8 eval runs/LLM** = ~16 runs total。

**Phase B 最初の subtask**: mlx-lm の現バージョンが Qwen3 / Gemma4 の loader を提供するか確認する。提供しない場合は `baseline_reporag/generation/mlx_lm.py` の修正が本 Issue 範囲外となり、別 Issue として切り出すこと。

成果物:
- `reports/llm_baseline_comparison_2026q2.md` (新規) — 3 LLM × 2 dataset の包括比較
- `configs/baseline_qwen35.yaml` + `configs/baseline_gemma4.yaml` (新規) — 新 LLM 用 base config
- `configs/institutional_docs_qwen35.yaml` + `configs/institutional_docs_gemma4.yaml` (新規)

### Phase C: 採用判定と CLAUDE.md 更新

3 LLM の比較結果をもとに、**#135 再学習の baseline LLM を確定** する。

判定基準:
- NC rate (overall, Turn 5-6) で baseline と PHOTON の差
- latency (p50, p95) — Qwen 3.5 9B は smaller のため latency 改善期待
- 推論時のメモリピーク (gemma 4 MoE は activated parameters のみだが KV cache が大きい)
- 採用 LLM の training tokenizer 互換性 (#138 fix が依存)

成果物:
- `docs/llm_choice_decision_2026q2.md` (新規) — 採用 LLM 選定理由 + 移行ロードマップ
- CLAUDE.md の「LLMバックエンド」行を採用 LLM に更新
- 既存 `configs/baseline.yaml` の global default を採用 LLM に更新

### Phase D: #135 への引継ぎ

採用 LLM を確定後、#135 (本格再学習) の Phase 6-8 で:
- corpus 生成は採用 LLM の tokenizer で実施
- PhotonModel の vocab_size は採用 LLM の vocab に合わせる (`AutoTokenizer.from_pretrained(...).vocab_size` 取得 → yaml 反映)
- vocab_size 不一致時の embedding 行列 reshape 方針 (zero-pad or 再 init) は #135 着手時に決定
- training data tokenizer は採用 LLM 系列で統一
- **本 Issue 範囲**: 採用 LLM の `tokenizer_id` を確定するまで。vocab_size 整合性の完全な実装 (embedding 行列 reshape 等) は #135 Phase 6-8 範囲。

## 影響ファイル

### Phase A
- `configs/institutional_docs_photon.yaml`: `model.checkpoint_path` 明示 (※コメント lines も #148 Phase A 時点の provider 設定に合わせて更新する)
- `reports/gate2_judgment_v5_post_s7001.md` (新規)
- `reports/institutional_photon_mt_eval_v2.md` (新規)

### Phase B
- `configs/baseline_qwen35.yaml` (新規)
- `configs/baseline_gemma4.yaml` (新規)
- `configs/institutional_docs_qwen35.yaml` (新規)
- `configs/institutional_docs_gemma4.yaml` (新規)
- `baseline_reporag/generation/mlx_lm.py`: mlx-lm が新 model_id の loader を提供しない場合は本 Issue 範囲外 (別 Issue 切り出し)。提供する場合のみ新 model_id 対応を実施。
- `reports/llm_baseline_comparison_2026q2.md` (新規)

### Phase C
- `docs/llm_choice_decision_2026q2.md` (新規)
- CLAUDE.md (LLMバックエンド行更新)
- `configs/baseline.yaml` (採用 LLM に変更)
- `tests/test_pipeline_factory_yaml_invariants.py`: 既存 invariant (reranker.model_id / embedding.model_id) は変更しない。新規に LLM model_id 用 invariant test (`test_baseline_yaml_generation_model_id_unchanged` 等) を追加するか、LLM 系は invariant 化方針外とするかを Phase C 着手時に明文化・決定すること。

## 受入条件

### Phase A (blocker for #135)
- [ ] **前提**: mulmoclaude 600-step ckpt の所在特定 (val_loss 0.4525 達成 ckpt の絶対 path または HF URL を `configs/institutional_docs_photon.yaml` に設定済み)
- [ ] Qwen 2.5 + 修正後 PHOTON で FastAPI MT 完走 (run 2 回)
- [ ] Qwen 2.5 + 修正後 PHOTON で Institutional MT 完走 (run 2 回)
- [ ] `reports/gate2_judgment_v5_post_s7001.md` 出力、Gate 2 v4 と数値比較
- [ ] `reports/institutional_photon_mt_eval_v2.md` 出力、#113 と数値比較
- [ ] 各 report に「修正前 (random-init) vs 修正後 (loaded checkpoint)」の delta を記載。**比較基準値**: Gate 2 v4 final (Static NC PHOTON 20.0%, MT NC 6.7%、出典: `reports/gate2_judgment_v4_final.md`) および #113 institutional eval (NC 11.39%、出典: `reports/institutional_photon_mt_eval.md`) を参照点として固定
- [ ] #143 が未解消の場合、各 run の variance を report に併記し、信頼区間 (NC ± std) を明示

### Phase B (strategic)
- [ ] **前提**: 各 model_id を `huggingface-cli repo info <id>` で存在確認し、正式 slug を確定 (不在時は alternative に切替えて Issue 更新)
- [ ] **前提**: mlx-lm が Qwen3 / Gemma4 loader を提供するか確認完了
- [ ] Qwen 3.5-9B-8bit (正式 slug 確定後) を baseline_rag で smoke test 合格: `python -m baseline_reporag.cli --config configs/baseline_qwen35.yaml --repo-id fastapi_fastapi --question 'test'` が non-empty 応答を返す、かつ PHOTON pipeline でも tokenizer mismatch エラーが発生しない
- [ ] Gemma 4-26B-a4b-4bit (正式 slug 確定後) を baseline_rag で smoke test 合格: 同条件 (config は `configs/baseline_gemma4.yaml`) で non-empty 応答を返す、かつ PHOTON pipeline でも tokenizer mismatch エラーが発生しない
- [ ] 各 LLM × 2 dataset × 2 run = 8 runs 完走
- [ ] PHOTON pipeline でも各 LLM 動作確認 (S7-001 fix が新 tokenizer に対応するか検証)
- [ ] `reports/llm_baseline_comparison_2026q2.md` で 3 LLM 比較表 (NC, latency, memory, throughput)

### Phase C (decision)
- [ ] 採用 LLM 確定 + 理由文書化
- [ ] CLAUDE.md / `configs/baseline.yaml` 整合更新
- [ ] LLM model_id invariant test: 追加方針 (新規 `test_baseline_yaml_generation_model_id_unchanged`) または invariant 化方針外の明文化、いずれかを実施
- [ ] Phase A 完了で PR #1 (Phase A 成果物)、Phase B 完了で PR #2 (Phase B 成果物)、Phase C 完了で PR #3 (最終採用判定 + integration) の 3 段 PR で develop マージ。各 PR が #135 のどの段階を解禁するかは PR 説明に明記

### Phase D (#135 引継ぎ)
- [ ] 採用 LLM の tokenizer_id 確定 (本 Issue 範囲)
- [ ] 本 Issue 完了確認後、#135 Phase 6-8 着手解禁

## 想定 compute コスト

| Phase | GPU 時間 |
|---|---|
| A (Qwen 2.5 sanity) | ~3h |
| B pre-flight (HF download: 新 LLM 2 件 計 ~25GB) | ~0.5-1h (100 Mbps 環境の試算) |
| B (Qwen 3.5 + Gemma 4 全 eval) | ~6-8h |
| C (採用判定 + integration) | ~1h |
| **合計** | **~11-13h (HF download 含)** |

#137 (V4 build に 2.5h を要した実績) を踏まえ、メモリスワップ防止に各 LLM の vocab/embedding サイズ事前確認推奨。

## リスク

| リスク | 影響 | 緩和策 |
|---|---|---|
| Qwen 3.5 / Gemma 4 が mlx-community に存在しない | Phase B 中断 | Phase B 着手前に `huggingface-cli repo info` で存在確認、不在時は近い alternative に切替 (受入条件前提チェック) |
| model_id の正式 slug が Issue 記載と異なる | Phase B config 作成ミス | Phase B 着手前に slug 確認・Issue 更新 (上記前提チェックと同一) |
| 新 LLM の tokenizer が PHOTON architecture と不整合 | Phase D で blocker | Phase B の smoke test で事前検証、不整合なら現行 Qwen 2.5 維持 |
| Gemma 4 MoE の memory pressure が #137 V4 並みに重い | wall-clock 大幅延伸 | activated parameters (4B) のみで KV cache を試算、VSZ > 256GB なら 8bit→4bit 切替 |
| 採用 LLM 変更で #137 V4 retrieval 採用が無効化 | retrieval 再評価必要 | Phase C で V4 (bge-m3 + bge-reranker-v2-m3) が新 LLM でも有効か再検証 |
| #143 (Qwen nondeterminism) 未解消で 2 runs 平均の信頼区間が不明 | NC 数値の再現性が低下 | #143 未解消のまま進める場合は各 run の variance を report に併記・信頼区間を明示 (受入条件参照) |

## 並列性

- Phase A は #115 (wizard) と並列可 (#115 は CPU-only、本 Issue Phase A は GPU)
- Phase B は Phase A 完了後に着手 (GPU 競合回避)。ただし Phase B の HF download のみは Phase A eval 完走待ち中に並列実行可
- Phase C は Phase A + B 完了後

## 関連

- 元: S7-001 (#135 commit `2dbf458`) + #138 (tokenizer mismatch、PR #141)
- 完了済 follow-up: #139 (PR #147)、#140 (PR #146)
- ブロック対象: **#135 Phase 6-8 (本格再学習)** — 本 Issue Phase A 完了後に解禁
- 関連 follow-up: #143 (eval reproducibility — Qwen nondeterminism) — 未解消の場合 Phase A/B の variance 記載が必要 (受入条件参照)、#144 (V2 ruri build)
- 派生候補: 採用 LLM 変更により #137 V4 retrieval 採用 (bge-m3) が新 LLM でも最適か再検証する follow-up

## PR 運用方針

- **Phase A 完了**: PR #1 (Phase A 成果物 — reports/gate2_judgment_v5*, reports/institutional_photon_mt_eval_v2*, yaml checkpoint_path 設定) → develop マージ → #135 Phase 6-8 着手解禁
- **Phase B 完了**: PR #2 (Phase B 成果物 — 新 LLM configs + 比較 report) → develop マージ
- **Phase C 完了**: PR #3 (最終採用判定 + CLAUDE.md / baseline.yaml / invariant test 整合更新) → develop マージ → main マージ

## 備考

本 Issue は **S7-001 + #138 が「PHOTON の真の数値を測れていなかった」事実を踏まえ、Phase 2 全体のベースラインを統合的に再構築する重要 milestone** である。Phase A だけでも #135 着手前の必須条件を満たすが、Phase B-C を含めることで「次世代 PHOTON 基盤」の選定まで一括完了させる。
