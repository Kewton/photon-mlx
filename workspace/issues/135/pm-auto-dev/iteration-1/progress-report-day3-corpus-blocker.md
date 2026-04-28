# Issue #135 Day 3 進捗報告 (続) — Step 1-2 完了 + Step 3 ETA blocker

**Issue**: feat(training): PHOTON 本格再学習 — 制度文書ドメイン対応 JP corpus 50%+ 混合
**実行日**: 2026-04-27 (Day 3 継続)
**ブランチ**: feature/issue-135-photon-retrain (Day 1-3 = **20 commits**)
**ステータス**: Step 1 (script refactor) + Step 2 (yaml 絶対パス) 完了。**Step 3 (corpus 生成本番) は ETA 想定の 5-10 倍に膨らむため停止 + 判断待ち**

---

## ✅ Step 1 完了: script refactor (commit `3612d8f`)

`scripts/generate_institutional_training_corpus.py` を production LLM client + token output に対応:

- **Drop ad-hoc Protocol**: `LLMClient.generate_turns(...)` を捨て、production `LLMClient.generate(prompt) -> str` に統一
- **Delegate to existing helper**: `build_sessions` は `baseline_reporag.eval.institutional.multi_turn.generate_session` を呼ぶ薄い wrapper に。prompt 構築 / JSON parse / retry / extraction-on-fence をすべて既存実装に委譲
- **Doc layout 対応**: `build_doc_index(corpus_dir)` 経由で `<doc>/document.md` 構造を扱う (smoke で institutional_documents 4228 docs を読み取り確認)
- **Token output**: 新 `tokenize_sessions(...)` で `{"tokens": [int, ...]}` JSONL を出力 (existing `data/processed/train_multi.jsonl` と同 schema)
- **AutoTokenizer 統合**: `main()` が `transformers.AutoTokenizer.from_pretrained(args.tokenizer_id)` で Qwen 152064 vocab 対応
- **Tests**: 22/22 pass (新規 `test_calls_real_protocol_generate` + `TestTokenizeSessions`)

## ✅ Step 2 完了: yaml 絶対パス (commit `b7057cd`)

`configs/institutional_docs_photon_retrain.yaml`:

```yaml
train_corpora_mix:
  "/Users/maenokota/share/work/github_kewton/photon-mlx-feature-issue-135-photon-retrain/data/training/institutional/train_jp.jsonl": 0.5
  "/Users/maenokota/share/work/github_kewton/photon-mlx-develop/data/processed/train_multi.jsonl": 0.5
val_split: 0.05
```

`PHOTON_CHECKPOINT_ROOT=/Users/maenokota/share/work/github_kewton/photon-mlx-develop/checkpoints` を env で渡し、`resume_from=step_000600` (val_loss=1.6238) を train() に渡す運用。

`test_institutional_retrain_yaml_loads_with_expected_hyperparams` も絶対パス対応に更新。

---

## 🛑 Step 3 中断: corpus 生成 ETA blocker

### Smoke run (seed=99, 5 sessions)

✅ **5/5 success**, ~4 分:
- `n_sessions_succeeded`: 5
- `eval_overlap`: 0
- `jp_sequence_ratio`: 1.0
- `total_tokens_train`: 1628 (4 sessions) + `total_tokens_val`: 249 (1 session) = 1877 tokens
- scenario distribution: drill_down 0.4 / cross_reference 0.4 / real_scenario 0.2

**Throughput**: ~48 秒/session。pipeline 自体は健全。

### Production run (seed=42, 2000 sessions) — 中断

開始 ~10 分後に **0 sessions 成功 / 2 sessions 失敗 (各 3 retry exhaust)** を観測して停止 (task `bcfyp2aun`):

```
PDF-163 / cross_reference → JSONDecodeError "Expecting ',' delimiter: line 30 column 6"
  attempt 1, 2, 3 — 同じ error 文字列 (deterministic)
児童手当法施行規則... / drill_down → JSONDecodeError "Expecting ',' delimiter: line 37 column 6"
  attempt 1, 2, 3 — 同じ error 文字列 (deterministic)
```

### 根本原因

1. **Qwen 14B 4-bit + 同一 seed の retry は無意味**: `QwenMLXAdapter.generate(prompt, seed=42)` は seed 固定。同じ prompt → 同じ output → 同じ JSON parse failure。`multi_turn.generate_session` の `for attempt in range(max_retries)` は実質 3 倍の wall-clock を消費するだけで成功率に寄与しない。
2. **Doc 内容で deterministic に fail/succeed が決まる**: smoke (seed=99) の 5 docs は 100% 成功、production (seed=42) の最初の 2 docs は 100% 失敗。corpus 4228 docs のうち何割が fail するかは未知だが、smoke での pure success と production での pure failure の対比から **doc-content 依存の分裂**は確実。
3. **Per-call latency 50 秒の本質的限界**: 単発 mlx_lm generation で 1 session ≈ 50 秒。仮に 100% 成功でも 2000 × 50 = **28 時間**。失敗込みなら 1.5-2 倍 = **40-55 時間**。

### ETA 再計算

| シナリオ | 想定 success rate | 1 session 期待時間 | 2000 sessions ETA |
|---------|------------------|------------------|------------------|
| Smoke 同等 (seed=99 範囲) | 100% | 48s | **~27h** |
| Production (seed=42 観測) | 50%? | 96s (3 retries 平均) | **~53h** |
| 楽観 | 95% | 53s | ~30h |

ユーザーの想定 "数時間" (3-6h) と **5-10 倍乖離**。

---

## 🎯 判断選択肢 + 推奨

### Option A: retry 機構を修正 + 続行 (推奨)
- `baseline_reporag/eval/institutional/multi_turn.generate_session` で retry ごとに `seed=seed+attempt` を渡す改修 (1 ファイル、~5 行、CPU、TDD で 30 分)
- 1 retry で多様なサンプリングが効き、success rate ~80% 期待
- ETA: 2000 × 50 / 0.8 ≈ **35 hours**, 改善後でも 3-6h を超過
- 下流影響: develop の eval set 生成にも効果あり (将来再生成時)

### Option B: sessions 数を縮小 (3-6h 想定維持)
- `--sessions 300` (or 500) で運用、JP token 比率も縮小 → mix 比率を再調整 (例: JP 30% / EN 70%)
- ETA: 300 × 50 ≈ **4-7 hours** ← ユーザー想定と整合
- Trade-off: JP corpus 規模が設計方針書 (≥ 2000) を下回る、retrain 効果に上限

### Option C: OpenAI provider に切替 (DR3-002 例外承認要)
- gpt-4o-mini 等を使用、JSON 安定性が高く高速 (5-10 sec/call)
- ETA: 2000 × 8 = **~5 hours** ← ユーザー想定と整合
- Trade-off: institutional 文書 (制度文書) を OpenAI に送信、設計方針書 §11 リスク表「外部 LLM provider 送信制限」(**DR3-002**) の例外承認が必要

### Option D: 並列 / batched mlx_lm
- mlx_lm の batched generation API を使い 4-8 並列化
- ETA: 28h / 4 ≈ **7 hours**
- Trade-off: 実装工数 4-6 時間 (CPU、TDD)、OOM リスク (Qwen 14B 4-bit + batch 4 で peak RAM ~30GB)

### 推奨: **Option A + Option B の組合せ**

1. retry 機構修正 (Option A) を CPU 30 分で実装
2. `--sessions 300-500` で運用 (Option B) — 設計方針書を pragmatic に緩める
3. 結果を見て不足なら 2nd batch (Option D) で並列化検討

**ETA**: A 修正 30 分 + 500 sessions × 60s = **約 9-10 時間**、ユーザー想定の "数時間" には届かないが現実的に達成可能。

---

## ✅ 不変事実

- Step 1 + Step 2 commit 済 (3 commits、累計 20 commits): ALL ruff / pytest pass (1247 + 22 corpus tests)
- Smoke 5-session 成功で **pipeline は健全**
- Token output schema は EN corpus (`train_multi.jsonl`) と完全互換 (= mix 学習可能)
- 環境変数 `PHOTON_CHECKPOINT_ROOT` 経由の checkpoint 参照経路は #148 Phase A0 で確立済 → resume_from へ即時利用可

---

## ❓ ユーザー判断が必要

1. **Option A / B / C / D / 組合せ どれで進めるか**
2. (Option C 採用なら) **DR3-002 例外承認** (institutional 文書を OpenAI 送信)
3. (Option B / Option A で sessions 縮小なら) **mix 比率再調整**: JP 30% / EN 70% 等
4. **次セッションへの引継ぎ**: ユーザー判断後、本セッションは idle で待機 OR 別セッションで Option 実装着手

ユーザー指示があるまで実行保留。本セッションは **idle 状態で待機** します。

---

## 参考: Day 3 累計コミット (Day 1-3 通算 20 件)

```
b7057cd feat(configs): use develop worktree absolute paths (#135 / Phase 6 prep)
3612d8f feat(scripts): align corpus generator with production LLM client + token output
b099960 docs(issue-135): Day 3 進捗報告 — develop merge OK + Phase 6 blocker 報告
be91682 merge develop into feature/issue-135-photon-retrain (Phase 6 prep)
994ba29 docs(issue-135): Day 2 EOD 進捗報告
344caae feat(scripts): DR4-001 CLI hardening for training corpus generator
34a833a feat(photon_mlx/trainer): dispatch to iterate_mixed_batches when mix set
d0277e3 docs(issue-135): Day 2 PM 進捗報告
f25022c feat(scripts): training corpus generator scaffolding
87802fb feat(configs): institutional_docs_photon_retrain.yaml
a2a9d5b feat(photon_mlx/checkpoint): integrity.json SHA-256 verification
587930c feat(torch_ref/config): add train_corpora_mix + val_split
397b0bb feat(photon_mlx/data): add iterate_mixed_batches
ecd1c2a docs(issue-135): Day 2 AM 進捗報告
8f1672d chore: add pre-commit config with detect-secrets
f17c3a6 test(photon_pipeline): pin DR1-002 boundary in subprocess
1c920ae test(photon_pipeline): add Day 1 checkpoint load smoke test (削除済)
2dbf458 fix(photon_pipeline): load checkpoint in _build_photon_deps
57d7742 docs(issue-135): pm-auto-issue2dev 完了報告
ea2fa57 refactor(photon_mlx): extract checkpoint I/O into checkpoint.py
```
