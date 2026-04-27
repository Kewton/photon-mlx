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

- **#135 Phase 6-8 (本格再学習)**: 「真の PHOTON ベースライン」が未確定なため、再学習前後の Apple-to-Apple 比較が成立しない。**#135 GPU 着手の unblock 条件は本 Issue 全体の完了ではなく Phase A0+A 完了** (Qwen2.5 + loaded checkpoint の true PHOTON baseline 確定) とする。Phase B-C は LLM 戦略判断の追加作業であり、#135 の再学習開始をブロックし続けない。

## ゴール

Qwen2.5 + loaded checkpoint で **真の PHOTON ベースラインを再確立** し、#135 再学習の比較基準を作る。あわせて 3 つの LLM × 2 つの代表 eval set の **baseline-only** 比較で将来の LLM 戦略決定基盤を作る。新 LLM + PHOTON の本格 eval は Phase D/#135 で tokenizer/vocab 整合後に実施する。

## アプローチ

### Phase A0: checkpoint loading 経路の検証・実装 (Phase A 着手前必須)

> **S3-001 対応**: `baseline_reporag/photon_pipeline.py:_build_photon_deps` は現行コード (PR #146/#147 時点) で yaml の `model.checkpoint_path` を読まず、`PhotonModel(photon_cfg)` で **random-init するだけ** である。S7-001 fix の完全性を確認し、実装ギャップが残っている場合は本 Phase で修正する。

subtask:
1. `baseline_reporag/photon_pipeline.py:_build_photon_deps` の実装を確認し、`checkpoint_path` を読んで `photon_mlx.trainer.load_checkpoint(model, path)` を呼ぶ経路が存在するか確認する。
2. 経路が存在しない場合 (実装ギャップあり) は、`_build_photon_deps` に以下を追加する:
   - yaml の `model.checkpoint_path` または top-level `checkpoint_path` を読む
   - `photon_mlx/trainer.py:load_checkpoint(model, path)` を呼んで weight を load する
   - load 成功/失敗を WARNING ログに記録する
3. `photon_mlx/inference.py` は random-init 検出の WARNING を出す診断層であり、checkpoint load の実装先にはしない。必要に応じて warning 文言を Phase A0 の実装方針に合わせて更新する。
4. 上記修正が必要な場合は本 Issue 範囲を **Phase A0** として拡大する。不要な場合 (既存実装で weight load 済み) は Phase A に直接進む。

⚠️ **yaml に `checkpoint_path` を書くだけでは PhotonModel が random-init のまま動作するリスクがある**。Phase A 開始前にこのリスクを解消し、「真の loaded checkpoint」で評価することを保証すること。

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

> **S7-005 影響範囲補足**: 背景で無効化した PHOTON Drift metrics / Safe RecGen trigger 発火率は、Phase A reports 内で「今回再測定する / 本 Issue では再測定せず follow-up 化する」のどちらかを明記する。NC/latency だけを再測定して Issue を閉じる場合、これらの旧指標は invalid のまま残るため、後続 Issue で誤参照されないよう follow-up 番号または明示的な out-of-scope 判定を残す。

> **S3-001 補足**: Phase A0 で checkpoint loading 経路の実装が完了していることを Phase A 着手条件とする。`configs/institutional_docs_photon.yaml` に `checkpoint_path` を設定した後、`_build_photon_deps` が実際に weight を load することをログで確認してから eval を実行すること。

判定:
- もし真の PHOTON が baseline 比 NC 改善 → S7-001 仮説 (random-init で過小評価されていた) 裏付け
- もし真の PHOTON が baseline 比 NC 悪化 (#113 と同程度の +4.44pp) → アーキ自体に課題、#135 再学習で改善が期待される
- いずれの結果でも #135 着手判断材料として確定

### Phase B: 新 LLM への拡張 (Phase A 完了後に着手、strategic)

> **注意**: Phase B は Phase A 完了後に着手すること (GPU 競合回避)。

> **前提確認 (Phase B 着手前必須)**: 各 model_id を `huggingface-cli repo info <id>` で存在確認し、不在時は最も近い既存 mlx-community model に切替えて本 Issue を更新すること。下記 model_id は **仮置き** であり、Phase B 着手前に `huggingface-cli repo info` で正式 slug を確認する。不在時は近い alternative に切替。
> - `mlx-community/Qwen3.5-9B-MLX-8bit`: Qwen3 系の正式 mlx-community 命名は `Qwen3-x` 系であり `Qwen3.5` は非標準呼称の可能性あり。正式 slug (例: `mlx-community/Qwen3-8B-MLX-8bit` 等) を確認・修正すること。
> - `mlx-community/gemma-4-26b-a4b-4bit`: Google 公式 release 名 (例: `mlx-community/gemma-3-27b-it-4bit` または `gemma-4-...` の正規 slug) を確認・修正すること。

> **S3-008 実行環境前提**: Phase B は **Mac Studio M3 Ultra (>=128GB unified memory) を必須要件** とする。Gemma4 26B MoE は mlx_lm.load 時に weight 展開で peak memory が 64GB マシンで OOM を引き起こす可能性がある。Phase B 開始前に `mlx_lm.load(gemma-4-26b-a4b-4bit)` で peak RSS を測定し、peak RSS が unified memory の 80% 未満に収まることを確認すること。Mac mini (~64GB) では failure する可能性があるため、実機 spec を確認してから着手すること。

| 対象 LLM | family label | quant | params | 想定 compute (eval 1 dataset) |
|---|---|---|---|---|
| **mlx-community/Qwen3.5-9B-MLX-8bit** ※仮 | qwen3.5 | 8bit | 9B (smaller than current 14B) | ~30-45 min/dataset |
| **mlx-community/gemma-4-26b-a4b-4bit** ※仮 | gemma4 | 4bit MoE | 26B total / 4B active | ~45-60 min/dataset |

> **S7-002 config provider 注意**: 上表の `family label` は説明用メタデータであり、YAML の `model.provider` には使わない。`configs/baseline_qwen35.yaml` / `configs/baseline_gemma4.yaml` は `model.provider: "mlx_lm"` を維持し、差し替えるのは `model.model_id` のみとする。`model.provider: "qwen3.5"` や `"gemma4"` を書くと、現行 `pipeline_factory.build_pipeline()` では `provider != "photon"` として baseline path に流れるため動作はしてしまうが、provider 意味論が壊れて silent misconfiguration になる。

Phase B の新 LLM 評価は **baseline-only** とする。新 LLM + PHOTON の本格 eval は、Phase D (#135 範囲) で tokenizer_id / vocab_size 同期と embedding reshape 方針が確定してから実施する。したがって Phase B は各新規 LLM × 2 dataset (FastAPI MT + Institutional MT) × baseline × 2 runs = **4 eval runs/LLM** = ~8 runs total。

> **S3-007 nondeterminism 検証**: Phase B 受入条件に各 LLM の nondeterminism 検証を追加する (同質問 × 5 runs で variance 測定)。特に Gemma4 MoE は expert routing に確率性があり、2-runs 平均が信頼区間として不十分な可能性がある。variance が高い場合は Phase B を 3 runs に増やす方針を明文化すること。

> **S3-009 grader バイアス検証**: Phase B で baseline LLM を Qwen3.5 に切替えた際、grader (`configs/eval.yaml` の `llm_judge_model_id: qwen3.5:27b`) と被評価 LLM が同系列 (Qwen3.5 vs Qwen3.5-9B-MLX) となることで **self-preference bias** が発生する可能性がある。Qwen3.5 系 LLM を評価する際は openai/gpt-4o-mini 等の別系列 grader で cross-check し、bias 有無を Phase B 受入条件に追加すること。

**Phase B 最初の subtask**: mlx-lm の現バージョンが Qwen3 / Gemma4 の loader を提供するか確認する。提供しない場合は `baseline_reporag/generation/mlx_lm.py` の修正が本 Issue 範囲外となり、別 Issue として切り出すこと。

> **S3-011 tokenizer_id 事前検証**: Phase B 着手前に採用予定 model_id の正式 slug を `_TOKENIZER_ID_PATTERN = [A-Za-z0-9._-]+/[A-Za-z0-9._-]+` で fullmatch 検証すること (1 行 Python: `import re; assert re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', model_id)`)。変則 slug (underscore-org、MLX 版 suffix 等) の場合はパターン変更が必要になる可能性があるため事前確認を徹底すること。

成果物:
- `reports/llm_baseline_comparison_2026q2.md` (新規) — 3 LLM × 2 dataset の包括比較。Qwen2.5 は Phase A の true PHOTON 結果も併記し、新 LLM 2 件は baseline-only の結果として記録する (新 LLM + PHOTON は Phase D 以降)。
- `configs/baseline_qwen35.yaml` + `configs/baseline_gemma4.yaml` (新規) — 新 LLM 用 base config
- `configs/institutional_docs_qwen35.yaml` + `configs/institutional_docs_gemma4.yaml` (新規)

> **S3-002 / S3-010 yaml 命名規則**: 新規 yaml の命名規則は `configs/baseline_<llm>.yaml` (baseline 用) / `configs/institutional_docs_<llm>.yaml` (institutional 用) で固定し、**`photon_` prefix は絶対に使用しない**こと。`photon_` prefix は `tests/test_pipeline_factory_yaml_invariants.py:_is_photon_profile_yaml` の検出対象となるため、非 PHOTON な baseline yaml が `photon_` prefix で命名されると invariant check が誤検知する。PHOTON 拡張版 yaml は Phase D 範囲とし、命名予約のみ行う (例: `configs/institutional_docs_photon_<llm>.yaml`)。

### Phase C: 採用判定と CLAUDE.md 更新

3 LLM の比較結果をもとに、**#135 再学習の baseline LLM を確定** する。Phase C の判定は、Qwen2.5 の true PHOTON 結果 (Phase A) と 3 LLM の baseline 結果 (Phase A/B) を組み合わせて行い、新 LLM + PHOTON の本格評価は Phase D (#135 範囲) に引き継ぐ。

判定基準:
- NC rate (overall, Turn 5-6): Qwen2.5 では baseline と true PHOTON の差、新 LLM 2 件では baseline 単独の絶対値と Qwen2.5 baseline との差
- latency (p50, p95) — Qwen 3.5 9B は smaller のため latency 改善期待
- 推論時のメモリピーク (gemma 4 MoE は activated parameters のみだが KV cache が大きい)
- 採用 LLM の training tokenizer 互換性 (#138 fix が依存)

> **S3-002 vocab_size 同期方針**: Phase C 完了時点で `configs/baseline.yaml` の `model_id` が新 LLM に更新されるが、以下の photon profile yaml は **Qwen2.5 系の vocab_size: 152064 のまま維持** することを原則とする:
> - `configs/photon_small.yaml`
> - `configs/photon_long_context.yaml`
> - `configs/institutional_docs.yaml`
> - `configs/institutional_docs_photon.yaml`
>
> vocab_size 不整合が発生すると `_load_hf_tokenizer` の #138 invariant で全 PHOTON pipeline が ValueError になる。photon profile の vocab_size/tokenizer_id 更新は **Phase D (#135 範囲)** で実施する。Phase B-C 期間中は photon profile yaml に以下のコメントを追記して意図を明示すること:
> ```yaml
> # NOTE: tokenizer_id and vocab_size are intentionally kept as Qwen2.5 until
> # Phase D (#135) completes vocab reshape. Do not change during Phase B-C.
> ```

> **S3-004 weekly_eval.yml 対応**: `configs/baseline.yaml` の global default を新 LLM に更新する PR (Phase C) と同時に以下を確認・対応すること:
> - `.github/workflows/weekly_eval.yml` の `timeout-minutes: 180` が新 LLM (Gemma4 26B MoE) で不足する可能性を事前評価し、必要に応じて延長する
> - `ci_eval_check.py` の NC rate threshold が新 LLM で false alarm を引き起こさないか確認する
> - Phase C 受入条件に「`workflow_dispatch` でドライランを実行し weekly_eval.yml が正常完了することを確認」を追加する
> - 新 LLM の HF model weight 初回 download (~25GB) による初回 timeout リスクを考慮し、cache warm-up の事前実行を推奨する

> **S7-003 configs/eval.yaml 対応**: `configs/eval.yaml` の `baseline_rag` / `baseline_rag_summary_memory` variant は `config_path: "./configs/baseline.yaml"` を参照している。Phase C で `configs/baseline.yaml` を新 LLM に更新すると、benchmark runner の baseline variant も silent migration する一方、`photon_rag` は `configs/photon_small.yaml` の Qwen2.5 PHOTON のまま残り、同一 benchmark 内で baseline と PHOTON の LLM backbone がズレる。Phase C では `configs/eval.yaml` を更新するか、Qwen2.5 固定用の legacy baseline config を追加して benchmark 比較軸を明示する。

> **S7-004 本番運用影響**: `baseline_reporag/server.py` / `baseline_reporag/cli.py` は既定で `configs/baseline.yaml` を読む。Phase C merge 後は本番 server/CLI の既定 LLM が新 LLM に silent migration し、初回 query 時の `mlx_lm.load()` で大容量 download / OOM / 起動遅延が発生しうる。Phase C では server/CLI の cold-start smoke test、HF cache warm-up、rollback 用 config (`configs/baseline_qwen25.yaml` 等) の有無を確認する。

> **S3-006 V4 retrieval 再検証**: Phase C の compute コスト試算 (~1h) に加えて、以下の追加確認を実施すること:
> - `configs/institutional_docs_<llm>.yaml` で #137 V4 retrieval chain (bge-m3 + bge-reranker-v2-m3) が新 LLM でも正常に機能するか A/B 比較 (追加 ~2-3h)
> - 多言語 reranker 出力に対する新 LLM の cite generation 整合性を確認する

> **S3-005 memory footprint 文書更新**: 採用 LLM 確定後、以下のファイルの memory footprint / model_id 記述を更新すること (Phase C 影響ファイル追加):
> - `docs/deployment.md`
> - `docs/troubleshooting.md`
> - `docs/tutorial.md`
> - `workspace/mvp/architecture.md`
> - `workspace/mvp/app_guide.md`
> - `workspace/mvp/metrics.md`
> - `README.md`
>
> 確認方法: `grep -rn 'Qwen2.5-Coder' docs/ workspace/mvp/ README.md` で全箇所を列挙し、Phase C 受入条件のチェックボックスとして展開する。

> **S3-012 CLAUDE.md 品質チェック表更新**: Phase C 完了時に CLAUDE.md の「LLM バックエンド」行を採用 LLM に更新することに加えて、品質チェック表にも新 LLM の smoke test コマンド行を追加すること (例: `python -m baseline_reporag.cli --config configs/baseline_<llm>.yaml --repo-id fastapi_fastapi --question "test"`)。

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
- **Phase D で実施**: `configs/photon_small.yaml` / `configs/photon_long_context.yaml` / `configs/institutional_docs.yaml` / `configs/institutional_docs_photon.yaml` の `tokenizer_id` + `vocab_size` を採用 LLM に同期更新
- **本 Issue 範囲**: 採用 LLM の `tokenizer_id` を確定するまで。vocab_size 整合性の完全な実装 (embedding 行列 reshape 等) は #135 Phase 6-8 範囲。

## 影響ファイル

### Phase A0 (新規)
- `baseline_reporag/photon_pipeline.py`: `_build_photon_deps` に checkpoint loading 経路を追加 (実装ギャップが確認された場合のみ)
- `photon_mlx/inference.py`: random-init warning 文言の確認・必要時修正のみ (checkpoint load は `_build_photon_deps` に実装)

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
- CLAUDE.md (LLMバックエンド行更新 + 品質チェック表に新 LLM smoke test コマンド追加)
- `configs/baseline.yaml` (採用 LLM に変更)
- `configs/eval.yaml` (baseline variant の silent migration / PHOTON variant との LLM backbone 差分を明示)
- `tests/test_pipeline_factory_yaml_invariants.py`: 既存 invariant (reranker.model_id / embedding.model_id) は変更しない。新規に LLM model_id 用 invariant test (`test_baseline_yaml_generation_model_id_unchanged` 等) を追加するか、LLM 系は invariant 化方針外とするかを Phase C 着手時に明文化・決定すること。
- `baseline_reporag/eval/institutional/llm_client.py`: line 75 の `QwenMLXAdapter` の default `model: str = 'mlx-community/Qwen2.5-Coder-14B-Instruct-4bit'` を採用 LLM に更新するか、yaml-driven に変更するかの方針を Phase C 着手時に明文化・決定すること。Issue #113 v2 (institutional eval set 再生成) を実行する際、`generate_eval_set` が古い LLM で grader を呼ぶリスクを解消するため、hardcode は避けることを推奨する。
- `baseline_reporag/server.py` / `baseline_reporag/cli.py` (既定 `configs/baseline.yaml` 利用者として cold-start / rollback / docs の影響確認。コード変更が不要な場合も Phase C 受入条件で確認する)
- `docs/deployment.md` (memory footprint 更新)
- `docs/troubleshooting.md` (HF cache path, memory footprint 更新)
- `docs/tutorial.md` (model_id, ~8GB 記述更新)
- `workspace/mvp/architecture.md` (memory footprint 更新)
- `workspace/mvp/app_guide.md` (memory footprint 更新)
- `workspace/mvp/metrics.md` (Qwen temp/top_p variance 記述更新)
- `README.md` (model_id 行更新)
- `.github/workflows/weekly_eval.yml` (timeout-minutes 見直し、threshold 確認)

### Phase D (#135 範囲)
- `configs/photon_small.yaml` (tokenizer_id + vocab_size を採用 LLM に更新)
- `configs/photon_long_context.yaml` (同上)
- `configs/institutional_docs.yaml` (tokenizer_id + vocab_size 更新)
- `configs/institutional_docs_photon.yaml` (tokenizer_id + vocab_size 更新)

## 受入条件

### Phase A0 (新規)
- [ ] `baseline_reporag/photon_pipeline.py:_build_photon_deps` が `checkpoint_path` を読んで weight を load する経路を持つことを確認、または実装
- [ ] `photon_mlx/inference.py` は checkpoint load を実行せず、random-init warning の診断責務に留まっていることを確認 (load 実行は `_build_photon_deps` 側で確認)
- [ ] checkpoint load 成否がログに記録されることを確認

### Phase A (blocker for #135)
- [ ] **前提**: Phase A0 完了 (checkpoint loading 経路の実装確認・修正完了)
- [ ] **前提**: mulmoclaude 600-step ckpt の所在特定 (val_loss 0.4525 達成 ckpt の絶対 path または HF URL を `configs/institutional_docs_photon.yaml` に設定済み)
- [ ] Qwen 2.5 + 修正後 PHOTON で FastAPI MT 完走 (run 2 回)
- [ ] Qwen 2.5 + 修正後 PHOTON で Institutional MT 完走 (run 2 回)
- [ ] `reports/gate2_judgment_v5_post_s7001.md` 出力、Gate 2 v4 と数値比較
- [ ] `reports/institutional_photon_mt_eval_v2.md` 出力、#113 と数値比較
- [ ] 各 report に「修正前 (random-init) vs 修正後 (loaded checkpoint)」の delta を記載。**比較基準値**: Gate 2 v4 final (Static NC PHOTON 20.0%, MT NC 6.7%、出典: `reports/gate2_judgment_v4_final.md`) および #113 institutional eval (NC 11.39%、出典: `reports/institutional_photon_mt_eval.md`) を参照点として固定
- [ ] PHOTON Drift metrics / Safe RecGen trigger 発火率について、Phase A reports 内で再測定結果または follow-up / out-of-scope 判定を明記
- [ ] #143 が未解消の場合、各 run の variance を report に併記し、信頼区間 (NC ± std) を明示

### Phase B (strategic)
- [ ] **前提**: 各 model_id を `huggingface-cli repo info <id>` で存在確認し、正式 slug を確定 (不在時は alternative に切替えて Issue 更新)
- [ ] **前提**: mlx-lm が Qwen3 / Gemma4 loader を提供するか確認完了
- [ ] **前提**: Phase B 実行環境が Mac Studio M3 Ultra (>=128GB unified memory) であることを確認。`mlx_lm.load(gemma-4-26b-a4b-4bit)` で peak RSS を測定し、unified memory の 80% 未満に収まることを確認。
- [ ] **前提**: 採用予定 model_id の正式 slug を `_TOKENIZER_ID_PATTERN` で fullmatch 検証 (`import re; assert re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', model_id)`)
- [ ] 新規 baseline yaml 4 件はすべて `model.provider: "mlx_lm"` を維持し、`model.model_id` のみ正式 slug に差し替えることを確認
- [ ] Qwen 3.5-9B-8bit (正式 slug 確定後) を baseline_rag で smoke test 合格: `python -m baseline_reporag.cli --config configs/baseline_qwen35.yaml --repo-id fastapi_fastapi --question 'test'` が non-empty 応答を返す
- [ ] Gemma 4-26B-a4b-4bit (正式 slug 確定後) を baseline_rag で smoke test 合格: 同条件 (config は `configs/baseline_gemma4.yaml`) で non-empty 応答を返す
- [ ] 新規 2 LLM は各 LLM × 2 dataset × baseline × 2 run = 4 runs/LLM、計 8 runs 完走
- [ ] 新 LLM + PHOTON の本格 eval は Phase D (#135 範囲) に延期することを `reports/llm_baseline_comparison_2026q2.md` に明記
- [ ] `reports/llm_baseline_comparison_2026q2.md` で 3 LLM 比較表 (NC, latency, memory, throughput) を作成し、Qwen2.5 の true PHOTON 結果と新 LLM baseline-only 結果を区別して表示
- [ ] nondeterminism 検証: 各 LLM × 同質問 × 5 runs で variance 測定。Gemma4 MoE で variance が高い場合は eval runs を 3 に増やす方針を決定・明文化
- [ ] grader バイアス検証: Qwen3.5 系 LLM 評価時に `qwen3.5:27b` grader との self-preference bias を検証。bias 検出時は openai/gpt-4o-mini 等で cross-check

### Phase C (decision)
- [ ] 採用 LLM 確定 + 理由文書化
- [ ] CLAUDE.md / `configs/baseline.yaml` 整合更新 (品質チェック表に新 LLM smoke test コマンド追加を含む)
- [ ] `baseline_reporag/eval/institutional/llm_client.py` の `QwenMLXAdapter` default model 更新方針を決定・実施
- [ ] LLM model_id invariant test: 追加方針 (新規 `test_baseline_yaml_generation_model_id_unchanged`) または invariant 化方針外の明文化、いずれかを実施
- [ ] `configs/eval.yaml` の baseline variants が採用 LLM へ移行するか、Qwen2.5 固定 config を参照するかを明文化し、PHOTON variants との LLM backbone 差分が report に出るように更新
- [ ] `.github/workflows/weekly_eval.yml` のドライラン (`workflow_dispatch`) を実行し正常完了を確認
- [ ] `baseline_reporag.server` / `baseline_reporag.cli` の既定 `configs/baseline.yaml` cold-start smoke test を実施し、初回 download / memory / rollback 手順を docs/deployment.md または troubleshooting.md に記載
- [ ] `grep -rn 'Qwen2.5-Coder' docs/ workspace/mvp/ README.md` で列挙された全箇所を採用 LLM に更新
- [ ] Phase C 影響ファイル節に列挙した文書 7 件 (`docs/deployment.md`, `docs/troubleshooting.md`, `docs/tutorial.md`, `workspace/mvp/architecture.md`, `workspace/mvp/app_guide.md`, `workspace/mvp/metrics.md`, `README.md`) の memory footprint / model_id を更新
- [ ] #137 V4 retrieval chain (bge-m3 + bge-reranker-v2-m3) が新 LLM でも正常に機能することを A/B 比較で確認 (追加 ~2-3h)
- [ ] Phase A 完了で PR #1 (Phase A 成果物)、Phase B 完了で PR #2 (Phase B 成果物)、Phase C 完了で PR #3 (最終採用判定 + integration) の 3 段 PR で develop マージ。各 PR が #135 のどの段階を解禁するかは PR 説明に明記

### Phase D (#135 引継ぎ)
- [ ] 採用 LLM の tokenizer_id 確定 (本 Issue 範囲)
- [ ] 本 Issue 完了確認後、#135 Phase 6-8 着手解禁

## 想定 compute コスト

| Phase | GPU 時間 |
|---|---|
| A0 (checkpoint loading 経路検証・実装) | ~0.5h |
| A (Qwen 2.5 sanity) | ~3h |
| B pre-flight (HF download: 新 LLM 2 件 計 ~25GB、実回線によっては 1-2h) | ~0.5-1h (100 Mbps 環境の試算) |
| B (Qwen 3.5 + Gemma 4 baseline-only eval) | ~6-8h |
| B nondeterminism 検証 (各 LLM × 5 runs) | ~2-3h 追加 |
| C (採用判定 + integration) | ~1h |
| C V4 retrieval 再検証 (bge-m3 chain × 新 LLM A/B) | ~2-3h 追加 |
| **合計** | **~15-20h (HF download + nondeterminism + V4 再検証 含)** |

> **S3-006 補足**: 上記 compute には #137 V4 retrieval 再検証分と nondeterminism 測定分を追加した。元の試算 ~11-13h から約 4-7h の追加が見込まれる。#137 V4 build に 2.5h を要した実績を踏まえ、メモリスワップ防止に各 LLM の vocab/embedding サイズ事前確認を推奨。

## リスク

| リスク | 影響 | 緩和策 |
|---|---|---|
| Qwen 3.5 / Gemma 4 が mlx-community に存在しない | Phase B 中断 | Phase B 着手前に `huggingface-cli repo info` で存在確認、不在時は近い alternative に切替 (受入条件前提チェック) |
| model_id の正式 slug が Issue 記載と異なる | Phase B config 作成ミス | Phase B 着手前に slug 確認・Issue 更新 (上記前提チェックと同一) |
| 新 LLM の tokenizer が PHOTON architecture と不整合 | Phase D で blocker | Phase B の smoke test で事前検証、不整合なら現行 Qwen 2.5 維持 |
| Gemma 4 MoE の memory pressure が #137 V4 並みに重い | wall-clock 大幅延伸 | activated parameters (4B) のみで KV cache を試算、VSZ > 256GB なら 8bit→4bit 切替 |
| Gemma 4 26B MoE の初回 download (~13-15GB) + mlx_lm.load 時の peak memory が OOM を引き起こす | Phase B 実行不能 (64GB マシン) | Mac Studio M3 Ultra (>=128GB) を必須環境とする。Phase B 開始前に peak RSS 測定を実施 |
| 採用 LLM 変更で #137 V4 retrieval 採用が無効化 | retrieval 再評価必要 | Phase C で V4 (bge-m3 + bge-reranker-v2-m3) が新 LLM でも有効か再検証 (compute ~2-3h 追加) |
| #143 (Qwen nondeterminism) 未解消で 2 runs 平均の信頼区間が不明 | NC 数値の再現性が低下 | #143 未解消のまま進める場合は各 run の variance を report に併記・信頼区間を明示 (受入条件参照) |
| Gemma4 MoE expert routing 確率性で 2-runs 平均が信頼区間として不十分 | NC 数値の再現性が低下 | 同上 + Phase B nondeterminism 検証 (5 runs) で確認、必要なら 3 runs に増加 |
| configs/baseline.yaml 更新で .github/workflows/weekly_eval.yml が silent migration → CI OOM / threshold 誤検知 | CI 安定性低下 | Phase C 受入条件に weekly_eval.yml のドライラン必須化、timeout/threshold 事前評価 |
| configs/eval.yaml の baseline variant が silent migration し、PHOTON variant と LLM backbone がズレる | benchmark 比較の解釈を誤る | Phase C で configs/eval.yaml の config_path 方針を明文化し、report に backbone 差分を出す |
| server.py / cli.py の既定 baseline.yaml が新 LLM に切替わり、初回 request で download / OOM / 起動遅延 | 本番運用の不意な停止・遅延 | Phase C で cold-start smoke、HF cache warm-up、rollback config を確認 |
| photon profile yaml (photon_small.yaml 等) の vocab_size が新 LLM と不整合 → #138 invariant で ValueError | 全 PHOTON pipeline 起動不能 | Phase B-C 期間中は photon profile yaml を Qwen2.5 系のまま維持 (yaml コメントで明示)。更新は Phase D (#135 範囲) で実施 |
| grader (qwen3.5:27b) と被評価 LLM (Qwen3 系) が同系列で self-preference bias 発生 | eval 信頼性低下 | Phase B 受入条件に cross-check grader (openai/gpt-4o-mini 等) による bias 検証を追加 |
| baseline_reporag/eval/institutional/llm_client.py の QwenMLXAdapter hardcode で institutional eval set が旧 LLM で再生成される | eval set 信頼性低下 | Phase C で llm_client.py の方針を確定・更新 (受入条件参照) |

## 並列性

- Phase A0 は即時着手可
- Phase A は Phase A0 完了後に着手 (#135 の blocker)
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

- **Phase A0+A 完了**: PR #1 (Phase A0+A 成果物 — checkpoint loading 経路実装 + reports/gate2_judgment_v5*, reports/institutional_photon_mt_eval_v2*, yaml checkpoint_path 設定) → develop マージ → #135 Phase 6-8 着手解禁
- **Phase B 完了**: PR #2 (Phase B 成果物 — 新 LLM configs + 比較 report) → develop マージ
- **Phase C 完了**: PR #3 (最終採用判定 + CLAUDE.md / baseline.yaml / llm_client.py / invariant test 整合更新) → develop マージ → main マージ

> **S3-002 Phase D 方針**: Phase D (#135 範囲) 着手まで photon profile yaml (photon_small.yaml, photon_long_context.yaml, institutional_docs.yaml, institutional_docs_photon.yaml) の `tokenizer_id` / `vocab_size` は Qwen2.5 系のまま維持する。新 LLM を PHOTON pipeline で使用するには vocab_size 整合 (embedding 行列 reshape 等) が #135 Phase 6-8 で完了していることが前提。Phase B-C 期間中は新 LLM + PHOTON pipeline の本格 eval を実施せず、baseline-only 比較と Phase D 引継ぎ条件の整理に留める。

## 備考

本 Issue は **S7-001 + #138 が「PHOTON の真の数値を測れていなかった」事実を踏まえ、Phase 2 全体のベースラインを統合的に再構築する重要 milestone** である。Phase A だけでも #135 着手前の必須条件を満たすが、Phase B-C を含めることで「次世代 PHOTON 基盤」の選定まで一括完了させる。

---

> **TODO / future work** (本 Issue 範囲外、将来 Issue 化):
> - `configs/eval.yaml` の `llm_judge_model_id` を yaml-configurable にする (現状 qwen3.5:27b がハードコード、Phase B での grader bias 検証で必要性が高まった場合)
> - `baseline_reporag/eval/institutional/llm_client.py` の `QwenMLXAdapter` を yaml-driven model_id 設定に変更する (Phase C 方針決定で「yaml-driven 変更」を選んだ場合は本 Issue 内で実施、「hardcode 更新のみ」ならば将来 Issue に据え置き)
> - photon profile yaml (photon_small.yaml 等) の vocab_size を新 LLM に合わせる embedding 行列 reshape (#135 Phase 6-8 範囲)

