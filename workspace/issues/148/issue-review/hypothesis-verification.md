# Issue #148 仮説検証レポート (Phase 0.5)

**対象**: Issue #148 — `test(eval): re-establish true baseline — fixed PHOTON pipeline + new LLM upgrade (Qwen3.5-9B / Gemma4-26B)`
**実施日**: 2026-04-27
**検証者**: Claude (Explore agent)

## 仮説サマリー

| # | 仮説 | 判定 |
|---|------|------|
| H1 | S7-001 + #138 critical bug は 2026-04-26 解消済 (#141, #146, #147 merged) | **Confirmed** |
| H2 | PhotonModel は以前 random-init + StubTokenizer で動作していた | **Confirmed** |
| H3 | 現行 baseline LLM は `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | **Confirmed** |
| H4 | configs/institutional_docs_photon.yaml に checkpoint_path 未明示 | **Confirmed** |
| H5 | mulmoclaude 600-step ckpt が存在する | **Partially Confirmed** |
| H6 | Latency は推論経路最適化由来、weight 非依存 | **Confirmed** |
| H7 | Baseline 単独 eval は photon_pipeline 非経由で信頼可能 | **Confirmed** |
| H8 | Gate 2 v4: baseline NC 21.7% / PHOTON NC 20.0% | **Confirmed** |
| H9 | Qwen3.5-9B-MLX-8bit / gemma-4-26b-a4b-4bit が HF 上に存在 | **Unverifiable** |
| H10 | `test_pipeline_factory_yaml_invariants.py` にハードコード invariant | **Confirmed** (LLM ではなく reranker model_id) |

---

## 詳細

### H1: S7-001 + #138 critical bug は 2026-04-26 に解消済 (Confirmed)

**根拠**:
- `8e677ca` (2026-04-26 22:04) PR #141 merge: "fix(photon): tokenizer mismatch — load real HF tokenizer in _build_photon_deps (#138)"
- `cc9301c` (2026-04-27 02:51) PR #147 merge: "test(photon): scaffolding pattern audit + tokenizer_id required validation (#139)" — `_StubTokenizer` 削除明記
- PR #146 (62ede02) merged
- `baseline_reporag/photon_pipeline.py:333-340` で tokenizer_id 必須チェック実装済 (ValueError raise)
- `photon_mlx/inference.py` に random-init 検出 (embedding norm check) 追加済

**申し送り**: 両 fix とも本線にマージ済。Issue #148 の前提として確定。

### H2: PhotonModel は以前 random-init + StubTokenizer で動作していた (Confirmed)

**根拠**:
- `baseline_reporag/photon_pipeline.py:323-340` のコメント: 「Issue #139: tokenizer_id is now required for provider=='photon'. The legacy byte-mod stub-tokenizer fallback was deleted to remove a structural path where production code could silently fall back onto a test fixture (the same class of bug as S7-001 random-init weights).」
- 過去コード (b19e8db) で warning + `_get_stub_tokenizer()` fallback が存在
- cc9301c で `_StubTokenizer` / `_get_stub_tokenizer` を削除

**申し送り**: S7-001 根本原因は「テスト用 stub を本番で実行」パターン。Issue #139 で構造的に削除済。

### H3: 現行 baseline LLM は `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` (Confirmed)

**根拠**:
- `configs/baseline.yaml:247`: `model_id: "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"`
- CLAUDE.md:19: 「LLMバックエンド: mlx-lm (Qwen2.5-Coder-14B-Instruct-4bit)」

### H4: configs/institutional_docs_photon.yaml に checkpoint_path 未明示 (Confirmed)

**根拠**:
- 全文確認: `model.checkpoint_path` キーは未存在
- `provider: "photon"` (line 177) は設定済だが checkpoint load 設定なし
- コメント (line 180) に「#113 PHOTON 測定時はこの行を "photon" に切替える」

**申し送り**: Phase A で checkpoint_path 追加が必要。現行は photon_pipeline._build_photon_deps 側に隠蔽されている。

### H5: mulmoclaude 600-step ckpt が存在 (Partially Confirmed)

**根拠**:
- `workspace/mvp/roadmap.md:17`: 「PHOTON アーキテクチャ（Small 377M）✅ 実装完了 | val_loss 0.4525、600 step 学習済」
- `reports/institutional_photon_mt_eval.md` に「既存 mulmoclaude (英語コード) と混合」記載
- リポジトリ artifact としては未コミット (外部 checkpoint server or local path)

**申し送り**: Phase A で checkpoint_path 明示時に local path / HF hub URL を要決定。Issue 内に保存場所が記載されていないため曖昧性残存。

### H6: Latency は推論経路最適化由来、weight 非依存 (Confirmed)

**根拠**:
- `baseline_reporag/profiler.py:90-109`: `TurnProfiler.finish()` は wall-clock `time.perf_counter()` 計測
- weight 値依存なし (推論 flow 効率のみ反映)
- gate2_judgment_v4_final.md:41-50 に「pruning コスト削減」「prompt 最適化」「max_new_tokens 削減」「reranker skip」と記載

### H7: Baseline 単独 eval は photon_pipeline 非経由 (Confirmed)

**根拠**:
- `baseline_reporag/pipeline_factory.py:52-80` で provider == "photon" / "baseline" 分岐
- baseline 系は `_build_baseline_deps_no_mlx()` を呼び出し、MLX import / photon_pipeline 非経由
- `baseline_reporag/photon_pipeline.py:183-192` で baseline path wrapper も `_build_baseline_deps_no_mlx` に委譲

**申し送り**: 設計的に責務分離が確保されている。Phase A 実行時の信頼性根拠として妥当。

### H8: Gate 2 v4 数値 (Confirmed)

**根拠**:
- CLAUDE.md:162-164: 「Static no-citation: baseline 21.7% / PHOTON 20.0%」「MT no-citation: 6.7%」「Retrieval noise: 0%」
- `reports/gate2_judgment_v4_final.md:34`: 「Static NC | 21.7% | 20.0% | -1.7pp」

### H9: Qwen3.5-9B / Gemma4-26B が HF に存在 (Unverifiable)

**根拠**:
- コードベース内に Qwen3.5-9B / Gemma4-26b への参照なし (全 grep 0 hit)
- HF availability はコードからは確認不可

**申し送り**: Stage 1 レビューで HF 上の model_id 存在確認を **Phase A 開始前の前提条件** として明記すべき。`huggingface-cli download` で事前確認、不在時は近い alternative に切替必要。

### H10: `tests/test_pipeline_factory_yaml_invariants.py` のハードコード invariant (Confirmed)

**根拠**:
- `tests/test_pipeline_factory_yaml_invariants.py:32`: `GLOBAL_DEFAULT_RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
- 同 37-40: `INSTITUTIONAL_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"` (#137 V4 採用)
- 同 54-55: invariant assert
- Issue #139 で追加された `test_photon_yaml_has_required_tokenizer_fields` も tokenizer_id / vocab_size の invariant 検査

**申し送り**: Issue 本文の H10 記述「LLM 名のハードコード」は **不正確**。実際は reranker model_id がハードコードされている。Phase C で baseline.yaml の model_id 変更時、当該 test の更新が必要だが Issue 受入条件 (Phase C) には reranker invariant の言及がないため、追加要修正点として Stage 1 で取り上げる。

---

## Stage 1 レビュー申し送りポイント

1. **S7-001 / #138 解消状況の前提が成立**: PR #141, #146, #147 が 2026-04-26 にマージ済。本 Issue の前提 (修正後 PHOTON で測定) が有効。

2. **Baseline vs PHOTON 数値の参照点**: Gate 2 v4 (Static NC baseline 21.7% / PHOTON 20.0%, MT NC 6.7%) が rebaseline 比較基準として妥当。

3. **新 LLM の HuggingFace availability 未確認**: Qwen3.5-9B-MLX-8bit / gemma-4-26b-a4b-4bit のリポ存在を事前検証する手順 (Phase A 開始前の `huggingface-cli download` smoke test) を受入条件に追加すべき。リスクとして Issue 内に記載はあるが、checklist 化されていない。

4. **mulmoclaude 600-step ckpt の所在不明**: Issue 内に checkpoint の保存場所 (local / HF / 共有 server) が記載されていない。Phase A で `model.checkpoint_path` を明示する作業の前提として、参照パス特定タスクが必要。

5. **invariant test 範囲の誤解**: Issue 影響ファイル節に `tests/test_pipeline_factory_yaml_invariants.py` (#132 invariant) を記載しているが、現行 invariant は **reranker model_id** であり LLM model_id ではない。Phase C で adoption LLM を変更する場合、新たに LLM model_id invariant を追加するか、既存 reranker invariant のみ更新するか方針確定が必要。

6. **Phase B の compute コスト見積もりに pre-flight smoke test 時間が含まれているか不明**: Qwen3.5-9B / Gemma4-26B の初回 download (HF cache miss) 時間が ~6-8h 見積もりに含まれているか曖昧。

7. **#143 (Qwen nondeterminism) との相互作用**: 「2 runs/dataset」で平均を取る設計だが、#143 が未解消の場合、平均値の再現性が低下するリスク。本 Issue 完了が #143 の前提となるか/独立か明示が必要。

8. **Phase D の引継ぎ条件曖昧性**: 「採用 LLM の training tokenizer 互換性 (#138 fix が依存)」と書かれているが、新 LLM の vocab_size と PhotonModel の vocab_size 整合性チェックの具体的手順が未定義。

9. **PR レビュー対象の絞り込み**: 影響ファイルが Phase A-C にまたがり 10 ファイル以上。本 Issue を 1 PR で扱うか Phase 分割するかが受入条件に未記載。
