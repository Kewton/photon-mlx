# Issue #148 設計方針書 — true baseline 再確立 + 新 LLM 評価基盤構築

> **対象 Issue**: [#148 test(eval): re-establish true baseline — fixed PHOTON pipeline + new LLM upgrade (Qwen3.5-9B / Gemma4-26B)](https://github.com/Kewton/photon-mlx/issues/148)
>
> **レビュー反映済み**: 累計 56 findings (Must Fix 13 / Should Fix 27 / Nice to Have 16) — Stage 1 (14 findings: Must Fix 1 / Should Fix 7 / Nice to Have 6) + Stage 2 (8 findings: Must Fix 1 / Should Fix 4 / Nice to Have 3) 含む
>
> **作成日**: 2026-04-27

---

## 1. ゴール (Issue 本文と一致)

S7-001 (PHOTON random-init) + #138 (tokenizer mismatch) の 2 つの critical bug を修正した PR (#141, #146, #147) を踏まえ、**「真の PHOTON ベースラインを再確立」** し、#135 再学習の比較基準を作る。

| Phase | 内容 | #135 への影響 |
|-------|------|-------------|
| **A0** | checkpoint loading 経路の検証・実装 (実装ギャップ対策) | Phase A 着手前の必須ゲート |
| **A** | Qwen2.5 + loaded checkpoint で Sanity Check eval (FastAPI MT + Institutional MT) | **完了で #135 Phase 6-8 unblock** |
| **B** | 新 LLM 2 件 (Qwen3.x / Gemma4) の baseline-only eval | #135 をブロックしない (並列可) |
| **C** | 3 LLM 比較・採用判定・CLAUDE.md / baseline.yaml 整合更新 | 採用 LLM 確定 |
| **D** | #135 への引継ぎ (tokenizer_id / vocab_size 同期) | #135 Phase 6-8 範囲 |

> **S7-001 対応**: #135 GPU 着手の unblock 条件は **Phase A0+A 完了**。Phase B-C は LLM 戦略判断の追加作業であり、#135 再学習開始をブロックし続けない。

---

## 2. 技術スコープ

| 領域 | 変更種別 | 影響度 |
|------|---------|-------|
| `baseline_reporag/photon_pipeline.py` (`_build_photon_deps`) | checkpoint loading 経路追加 (Phase A0) | **高** (PHOTON pipeline の動作正確性) |
| `photon_mlx/inference.py` | WARNING 文言確認・必要時更新のみ | 低 |
| `photon_mlx/trainer.py` (`load_checkpoint`) | 呼び出し対象 API (変更なし、利用のみ) | 中 |
| `configs/institutional_docs_photon.yaml` | `model.checkpoint_path` 追記 + yaml コメント更新 | 高 (Phase A の正確な eval 前提) |
| `configs/baseline_qwen35.yaml` / `configs/baseline_gemma4.yaml` (新規) | 新 LLM 用 baseline config | 中 (Phase B) |
| `configs/institutional_docs_qwen35.yaml` / `configs/institutional_docs_gemma4.yaml` (新規) | 新 LLM 用 institutional config | 中 (Phase B) |
| `configs/baseline.yaml` | 採用 LLM への model_id 更新 (Phase C) | **高** (本番・CI 全体に波及) |
| `configs/eval.yaml` | baseline variant の silent migration 対策 | 高 (Phase C) |
| `baseline_reporag/eval/institutional/llm_client.py` (`QwenMLXAdapter`) | default model 更新または yaml-driven 化 (Phase C) | 中 |
| `tests/test_pipeline_factory_yaml_invariants.py` | LLM model_id invariant 追加方針の決定 (Phase C) | 中 |
| `.github/workflows/weekly_eval.yml` | timeout / threshold 見直し (Phase C) | 中 |
| `docs/deployment.md` / `docs/troubleshooting.md` / `docs/tutorial.md` | memory footprint / model_id 更新 (Phase C) | 低 |
| `workspace/mvp/architecture.md` / `app_guide.md` / `metrics.md` / `README.md` | 同上 (Phase C) | 低 |
| `reports/gate2_judgment_v5_post_s7001.md` (新規) | Phase A eval 結果 | - |
| `reports/institutional_photon_mt_eval_v2.md` (新規) | Phase A eval 結果 | - |
| `reports/llm_baseline_comparison_2026q2.md` (新規) | Phase B eval 結果 | - |
| `docs/llm_choice_decision_2026q2.md` (新規) | Phase C 採用判定文書 | - |

---

## 3. アーキテクチャ判断

### 設計判断 #1: Phase A0 の checkpoint loading 実装場所 (S5-001 対応)

**背景 (S5-001 silent bug)**:
Stage 4 反映で Phase A0 受入条件に「`photon_mlx/inference.py:137` が load を実行することを確認」と記載されたが、実際の `inference.py:137` は WARNING 文言のみであり checkpoint load を実行しない。load API は `photon_mlx/trainer.py:load_checkpoint` にある。この誤指定を修正する。

また現行 `_build_photon_deps`（`baseline_reporag/photon_pipeline.py:254-`）は `PhotonModel(photon_cfg)` で random-init するのみであり、`checkpoint_path` を読んで weight を load する経路が存在しない（仮説検証 H4 Confirmed）。

**選択肢**:

| 案 | 実装場所 | 概要 |
|----|---------|------|
| (a) | `_build_photon_deps` | yaml の `checkpoint_path` を読み `photon_mlx.trainer.load_checkpoint(model, path)` を呼ぶ |
| (b) | `PhotonInference.__init__` | PhotonInference 内で checkpoint path を受け取り load する |
| (c) | factory レイヤ (`pipeline_factory.py`) | `build_pipeline` に wrapper を追加 |

**決定**: **(a) `_build_photon_deps` 内で実装**

**理由**:
- `_build_photon_deps` は PHOTON 依存関係を一括構築する Single Source of Truth であり、weight load もここで完結させるのが責務上自然
- `PhotonInference` は推論を担当する layer であり、checkpoint IO を持たせると SRP 違反
- `pipeline_factory.py` は MLX-free を維持するためのラッパーであり、実装詳細を持ち込まない
- `photon_mlx/inference.py` は `_check_weight_initialization` で random-init を **診断・警告** する責務のみを持つ（Issue #140 S7-001 follow-up の設計判断 #2）

**import スタイルの選択**:

`photon_pipeline.py` は MLX-aware 層であるため、`load_checkpoint` は関数内 lazy import ではなく **module top-level import** を採用する (SRP/DIP 上自然であり、かつ `§10.1` の unit test で `mocker.patch("baseline_reporag.photon_pipeline.load_checkpoint")` が正常に動作するため)。`pipeline_factory.py` は MLX-free を維持し `load_checkpoint` を import しない。

> **モジュール境界**:
> - `pipeline_factory.py`: MLX-free。`cfg.model.provider` を見て分岐するのみ。MLX import なし。
> - `photon_pipeline.py`: MLX-aware。`PhotonModel` / `load_checkpoint` 等の MLX 依存を保持。
> - `baseline_reporag/` 内の他モジュール: `photon_pipeline` への直接依存は `pipeline_factory` 経由に限定。

**実装方針 (fail-fast 原則)**:

checkpoint load 失敗は「真の baseline 再確立」というゴールと直接矛盾するため、**fail-fast を原則**とする。`checkpoint_path` が yaml に設定されている場合、load 失敗時は `RuntimeError` を raise して eval 全体を停止する。例外として、CI / test 環境で `PHOTON_ALLOW_RANDOM_INIT=1` 環境変数が設定されている場合のみ fail-soft (WARNING ログのみ) で続行できる。

```python
# baseline_reporag/photon_pipeline.py (module top-level import)
import os
from pathlib import Path
from photon_mlx.trainer import load_checkpoint

_CHECKPOINT_REQUIRED_FILES = ("weights.npz", "state.json")

def _allowed_checkpoint_roots() -> list[Path]:
    roots = [Path("checkpoints").resolve()]
    if os.environ.get("PHOTON_CHECKPOINT_ROOT"):
        roots.append(Path(os.environ["PHOTON_CHECKPOINT_ROOT"]).expanduser().resolve())
    return roots

def _validate_checkpoint_dir(raw_path: str) -> Path:
    try:
        ckpt_dir = Path(raw_path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("PhotonModel: checkpoint_path does not exist.") from exc
    if not any(ckpt_dir.is_relative_to(root) for root in _allowed_checkpoint_roots()):
        raise RuntimeError(
            "PhotonModel: checkpoint_path is outside approved checkpoint roots. "
            "Use repo-local checkpoints/ or set PHOTON_CHECKPOINT_ROOT."
        )
    missing = [
        name for name in _CHECKPOINT_REQUIRED_FILES
        if not (ckpt_dir / name).is_file()
    ]
    if not ckpt_dir.is_dir() or missing:
        raise RuntimeError(
            "PhotonModel: checkpoint_path must point to a photon_mlx checkpoint "
            "directory containing weights.npz and state.json."
        )
    return ckpt_dir

# _build_photon_deps 内
model = PhotonModel(photon_cfg)

# Phase A0: checkpoint loading 経路 (fail-fast 原則)
# [DR2-002] Config オブジェクトは attribute / dict-style 両 API をサポートする。
# 既存 _build_photon_deps では cfg.model.get('architecture', ...) (dict-style) と
# getattr(cfg.model, 'head_dim', 64) (attribute-style) が混在している。
# 本設計書では getattr を採用するが、実装時に既存パターン
# (cfg.model.get('checkpoint_path', None)) に統一しても可。
checkpoint_path = getattr(cfg.model, "checkpoint_path", None)
if checkpoint_path:
    ckpt_dir = _validate_checkpoint_dir(checkpoint_path)
    try:
        load_checkpoint(model, ckpt_dir)
        _logger.info("PhotonModel: checkpoint loaded from %s", ckpt_dir.name)
    except Exception as exc:
        _allow_random_init = os.environ.get("PHOTON_ALLOW_RANDOM_INIT", "0") == "1"
        if _allow_random_init:
            _logger.warning(
                "PhotonModel: checkpoint load failed (path=%s, reason=%s). "
                "Continuing with random-init weights (PHOTON_ALLOW_RANDOM_INIT=1).",
                checkpoint_path, type(exc).__name__,
            )
        else:
            raise RuntimeError(
                f"PhotonModel: checkpoint load failed (path={checkpoint_path}, "
                f"reason={type(exc).__name__}: {exc}). "
                "Set PHOTON_ALLOW_RANDOM_INIT=1 to bypass (test/CI only)."
            ) from exc
else:
    _logger.warning(
        "PhotonModel: checkpoint_path not set. "
        "Model will run with random-init weights. "
        "Set model.checkpoint_path in yaml to load trained weights."
    )
```

**checkpoint_path 形式の制約 (DR3-001)**:
`photon_mlx.trainer.load_checkpoint(model, path)` は `weights.npz` と `state.json` を含む checkpoint **directory** を読む API であり、`.safetensors` 単体ファイルや HF Hub URL を直接受け取る API ではない。Phase A0 の実装範囲では `model.checkpoint_path` を local directory path に限定し、HF URL / safetensors の resolver や converter は別タスクとする。実装時は `load_checkpoint` 呼び出し前に directory 形状を検証し、不正な場合は `RuntimeError` で fail-fast する。

**checkpoint_path のセキュリティ制約 (DR4-001 / DR4-004)**:
`checkpoint_path` は repo-local `checkpoints/` または `PHOTON_CHECKPOINT_ROOT` 配下に閉じ込め、`resolve(strict=True)` 後の実パスが許可 root 配下であることを確認する。`../`、絶対パス、symlink で許可 root 外へ抜ける path は拒否する。ログと例外メッセージには raw absolute path を出さず、checkpoint directory の basename または許可 root からの相対 path のみを出力する。

**Phase A 受入条件への追記**: eval 起動ログに `PhotonModel: checkpoint loaded from <path>` の INFO 行が必ず出現することを report に貼付して確認する。欠如時は eval 結果を invalid 扱いとする。

**Phase A0+A 完了の追加条件 (DR2-005)**: PR #1 merge 前に、Phase A reports 内で PHOTON Drift metrics / Safe RecGen 指標について「再測定する (follow-up Issue 番号記載)」または「out-of-scope 判定 + 理由」のいずれかを明記していること。

**Phase A0 完了の追加条件 (DR2-007)**: `PHOTON_ALLOW_RANDOM_INIT=1` の test/CI 用途限定の意図と運用方針を `docs/troubleshooting.md` または `docs/deployment.md` に記載済みであること。

**トレードオフ**:
- メリット: load 失敗を即座に検知、S7-001 再発パターン (silent random-init eval) を構造的に防止、既存 `inference.py` の診断層と責務が分離
- デメリット: `_build_photon_deps` がさらに多機能化するが、他の依存関係構築と同種の処理であり許容範囲
- fail-soft 経路: `PHOTON_ALLOW_RANDOM_INIT=1` 環境変数でのみ有効化 (test/CI 用途限定)

**`photon_mlx/inference.py` との関係**:
`_check_weight_initialization` (line 115-) は load 後の weight 状態を **診断・警告** する役割に留まる。load 実行は `_build_photon_deps` が担うため、WARNING が出た場合はログ確認を促す文言に更新する。既存 WARNING 文言「`Check model.checkpoint_path and load result`」はこの分担を示す有効な案内であり、文言は維持または明確化する方針とする。

---

### 設計判断 #2: 新 LLM 採用時の baseline.yaml 移行戦略 (S7-003, S7-004 対応)

**背景**: Phase C で `configs/baseline.yaml` を新 LLM に更新すると、以下が **silent migration** する:
- `.github/workflows/weekly_eval.yml` → CI の LLM が変わる
- `configs/eval.yaml` の `baseline_rag` / `baseline_rag_summary_memory` variant (`config_path: "./configs/baseline.yaml"` 参照)
- `baseline_reporag/server.py` / `baseline_reporag/cli.py` の既定 config
- 初回 query 時の `mlx_lm.load()` で ~25GB download / OOM / 起動遅延リスク

**選択肢**:

| 案 | 概要 | 影響 |
|----|------|------|
| (a) | `configs/baseline.yaml` を直接書き換え | silent migration 全発生、旧 LLM 戻し困難 |
| (b) | `configs/baseline_qwen25.yaml` を新設し旧 LLM 保存、`configs/baseline.yaml` を新 LLM 化 | rollback 経路確保、eval.yaml / CI 影響あり |
| (c) | CLI flag `--provider` で override | 実装コスト高、今回範囲外 |

**決定**: **(b) 旧 LLM 用 `configs/baseline_qwen25.yaml` を新設して移行**

**移行手順 (Phase C)**:
1. `configs/baseline_qwen25.yaml` を `configs/baseline.yaml` からコピーし旧 LLM 設定を保存（rollback 用）
2. `configs/baseline.yaml` の `model_id` を採用 LLM に更新
3. `configs/eval.yaml` の `photon_rag` variant は `configs/photon_small.yaml` を参照するため LLM backbone が Qwen2.5 のまま残り、Phase D まで baseline と PHOTON の backbone がズレる → **report 内に LLM backbone 差分を明記することで対応**
4. `.github/workflows/weekly_eval.yml` の timeout / threshold を事前評価し必要なら更新
5. `baseline_reporag/server.py` / `baseline_reporag/cli.py` の cold-start smoke test を実施
6. HF cache warm-up を事前実行して初回 download 時の timeout リスクを軽減

**`configs/eval.yaml` の方針**（S7-003）:
- `baseline_rag` variant が新 LLM に移行することを `configs/eval.yaml` 内コメントで明示
- `photon_rag` variant は Qwen2.5 のまま残り、**同一 benchmark 内で backbone 差分が発生する**
- report テンプレートに「baseline LLM: <採用 LLM> / PHOTON LLM: Qwen2.5 (Phase D まで)」のメタ情報欄を追加

**backbone 差分の自動出力 (DR2-001 / DR3-003 対応)**:
`bench/run_all.py` と `bench/tests/test_run_all.py`、および `configs/eval.yaml` に以下を追加し、backbone 差分を人手確認に依存させない:
- `configs/eval.yaml` に `report.show_llm_backbone: true` フラグを追加 (Phase C)
- report 生成時に各 variant の `model_id` を必ず出力する (例: `baseline_rag: <new_llm_id>` / `photon_rag: Qwen2.5-Coder-14B-Instruct-4bit`)
- weekly_eval の CI artifact にも backbone 情報を含める

> **photon_rag variant の Qwen2.5 維持の正当化**:
> Phase B-D 期間中、`photon_rag` variant が Qwen2.5 を維持し続けるのは以下の理由による:
> (1) PHOTON の PhotonModel は `vocab_size: 152064` (Qwen2.5 系) に対して closed であり、新 LLM 対応は #135 の embedding reshape (Phase D) を経由する必要がある;
> (2) Phase B-C での backbone 変更は `test_photon_yaml_has_required_tokenizer_fields` の誤検知を招く;
> (3) 公平な NC 比較は「同一 eval set × 異なる LLM」で baseline のみ新 LLM に更新することに主眼があり、PHOTON 側の backbone 更新は #135 完了後に改めて実施することで研究上の独立変数制御が維持される。
> backbone 差分は report に必ず明記し、解釈ミスを防ぐ。

---

### 設計判断 #3: PHOTON pipeline と新 LLM の関係 (S5-002 対応)

**背景**: S5-002 が指摘した Phase B/C/D の矛盾。

**決定**: **Phase B は新 LLM 2 件の baseline-only eval に限定。新 LLM + PHOTON の本格 eval は Phase D (#135 範囲) に延期**

**理由**:
- 現行 `configs/photon_*.yaml` 5 件のうち Qwen2.5 系は `photon_small.yaml` / `photon_long_context.yaml` の 2 件のみ (`vocab_size: 152064`)。残り 3 件 (`photon_600m_paper.yaml` / `photon_tiny.yaml` / `photon_tiny_recgen.yaml`) は LLaMA-2 系 (`vocab_size: 32000`)。
- 新 LLM の vocab_size が Qwen2.5 と異なる場合、`_load_hf_tokenizer` の #138 invariant (`ValueError`) が発動し PHOTON pipeline 全体が起動不能になる
- vocab_size 整合 (embedding 行列 reshape 等) は #135 Phase 6-8 の本格再学習前後で扱うべき問題
- Phase B-C 期間中に photon profile yaml を変更すると `tests/test_pipeline_factory_yaml_invariants.py` の `test_photon_yaml_has_required_tokenizer_fields` が誤検知するリスクがある

**OCP 観点の補足 (DR3-001)**:
PhotonModel の vocab/embedding 層は現行で Qwen2.5 `vocab_size: 152064` に対して **closed** であり、新 LLM 対応は #135 の embedding reshape (extension) を経由する。これは OCP に従い「vocab に対して closed, embedding 層拡張で open」を維持する設計判断である。Phase D 以降に新 LLM が追加された場合も、同じ extension パターン (vocab_size の reshape) を経由することで `PhotonModel` 本体への修正を最小化する。

**Phase B の eval スコープ**:

```
3 LLM × eval set × pipeline 種別:

| LLM | FastAPI MT | Institutional MT | PHOTON |
|-----|-----------|-----------------|--------|
| Qwen2.5 (現行) | Phase A (2 runs) | Phase A (2 runs) | Phase A (true baseline) |
| Qwen3.x (新規) | Phase B (2 runs) | Phase B (2 runs) | Phase D (#135 範囲) |
| Gemma4 (新規) | Phase B (2 runs) | Phase B (2 runs) | Phase D (#135 範囲) |
```

**photon profile yaml の保護コメント** (Phase B-C 期間中に必須追記):
```yaml
# NOTE: tokenizer_id and vocab_size are intentionally kept as Qwen2.5 until
# Phase D (#135) completes vocab reshape. Do not change during Phase B-C.
```

---

### 設計判断 #4: HF model_id の正規 slug 確認手順 (S1-001 対応)

**背景 (H9 Unverifiable)**: `mlx-community/Qwen3.5-9B-MLX-8bit` / `mlx-community/gemma-4-26b-a4b-4bit` はコードベース内に参照なし。Issue 記載 slug は **仮置き** であり正式 slug 未確定。

**確認手順 (Phase B 着手前必須)**:

```bash
# Step 1: HF repo 存在確認
huggingface-cli repo info mlx-community/Qwen3.5-9B-MLX-8bit
huggingface-cli repo info mlx-community/gemma-4-26b-a4b-4bit

# Step 2: tokenizer_id パターン検証
python3 -c "
import re
for slug in ['mlx-community/Qwen3.5-9B-MLX-8bit', 'mlx-community/gemma-4-26b-a4b-4bit']:
    assert re.fullmatch(r'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+', slug), f'Invalid slug: {slug}'
    print(f'OK: {slug}')
"
```

**不在時の代替策**:
- Qwen3.x 系: `mlx-community/Qwen3-8B-MLX-8bit` 等の近い slug を `huggingface-cli search` で特定し、Issue 本文を更新してから Phase B を開始する
- Gemma4 系: `mlx-community/gemma-3-27b-it-4bit` 等の近い slug を確認し同様に更新
- **代替 slug 選定基準 (定量化)**:
  1. **param 数の許容範囲**: ±20% 以内 (Qwen3.x 対象: 7B-11B、Gemma4 対象: 21B-32B)
  2. **vocab 系統**: 採用予定 LLM と同系統の vocab (Qwen3.x は Qwen tokenizer 系、Gemma4 は Gemma tokenizer 系) — 異系統の場合は Phase B スコープ外とし別 Issue に切り出す
  3. **量子化形式**: `4bit` または `8bit` (mlx-community 標準 quant 形式に限定)
  4. **最新安定版**: HF Hub 上で `mlx-community` org の verified author が公開し、かつ公開後 30 日以内に 100 downloads 以上の実績があること
  5. **Phase B 受入条件への追記**: 代替 slug を採用した場合は選定根拠 (downloads 数、updated_at、param 数差、vocab 系統) を `reports/llm_baseline_comparison_2026q2.md` の「slug 選定根拠」欄に必ず記載する

**注意 (S7-002 対応)**: Phase B 候補表の `family label` 列（説明用メタデータ）を YAML の `model.provider` に誤用しない。新規 yaml 4 件はすべて `model.provider: "mlx_lm"` を維持し、差し替えるのは `model.model_id` のみとする。

---

### 設計判断 #5: 受入条件「smoke test」の定義 (S1-005 対応)

**各 LLM の smoke test 合格条件** (Phase B 受入条件として明文化):

```bash
# Qwen3.x
python -m baseline_reporag.cli \
  --config configs/baseline_qwen35.yaml \
  --repo-id fastapi_fastapi \
  --question 'test'
# 合格条件:
# 1. 応答テキストが non-empty (空文字列・エラーメッセージでない)
# 2. tokenizer mismatch エラー (ValueError) が出ない
# 3. latency が <180s 以内 (初回 download 除く)
# 4. peak RSS 測定を実施し report に記録 (全 LLM 共通)

# Gemma4
python -m baseline_reporag.cli \
  --config configs/baseline_gemma4.yaml \
  --repo-id fastapi_fastapi \
  --question 'test'
# 合格条件: 同上 + peak RSS が unified memory の 80% 未満 (Gemma4 は閾値チェック必須)

# mlx_lm.load 時 peak memory 測定 (全 LLM 共通手順 — 閾値は LLM ごとに調整可)
python3 -c "
import mlx_lm, resource, sys
model, tok = mlx_lm.load('<正式 slug に置換>')
raw_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# macOS reports bytes; Linux reports KiB.
peak_gb = raw_maxrss / (1024**3) if sys.platform == 'darwin' else raw_maxrss / (1024**2)
print(f'peak_rss={peak_gb:.1f}GB')
"
# Qwen3.x: 測定値を report に記録 (64GB マシン動作確認を Phase B pre-flight subtask に追加)
# Gemma4: peak_rss < unified_memory_gb * 0.80 を必須チェック (Mac Studio 128GB 前提)
```

> **peak RSS 測定の統一方針 (DR5-001)**: 全 LLM について peak RSS 測定を実施し report に記録することを必須とする。閾値 (80%) の適用は LLM ごとに調整可だが、測定手順は 3 LLM 共通。Qwen3.x については「64GB マシンでの動作可能性確認」を Phase B pre-flight subtask として追加し、OOM リスクを事前に評価する。

---

### 設計判断 #6: Phase 分割と PR 運用方針 (S1-011, S7-001 対応)

**決定**: **3 段 PR (PR #1 内で commit 単位分離)**

| PR | Phase | 成果物 | #135 への効果 |
|----|-------|--------|-------------|
| PR #1 | Phase A0+A | (commit A0) checkpoint loading 実装 + unit test / (commit A) yaml checkpoint_path 設定 + reports/gate2_judgment_v5* + reports/institutional_photon_mt_eval_v2* | **#135 Phase 6-8 着手解禁** |
| PR #2 | Phase B | 新 LLM configs 4 件 + reports/llm_baseline_comparison_2026q2.md | #135 をブロックしない |
| PR #3 | Phase C | 採用判定文書 + CLAUDE.md / baseline.yaml / llm_client.py / eval.yaml / weekly_eval.yml / docs 7 件更新 + invariant test 方針 | main マージ (LLM 採用確定) |

**PR #1 の commit 分離方針 (PHASE-001 対応)**:

PR #1 は Phase A0 (code change) と Phase A (data/config change) の責務が異なるため、**単一 PR 内で commit 単位を分離**する:
- **commit A0**: `baseline_reporag/photon_pipeline.py` 変更 + unit test 追加 (code change のみ)
- **commit A**: `configs/institutional_docs_photon.yaml` の `checkpoint_path` 設定 + eval 実行 + reports 2 件出力 (yaml + data change)

この分離により:
1. Phase A0 の code review reject 時に Phase A の eval 再実行コストが最小化される
2. `git revert` で Phase A0 / Phase A を独立して revert 可能
3. PR #1 レビュー時に code 変更と eval 結果を commit 単位で追跡できる

#135 unblock 解禁条件は「PR #1 (commit A0+A 両方) merge 後」と定義する。

**#135 unblock 条件の明示**:
- Phase A0+A 完了 = checkpoint loading 経路実装確認 + Qwen2.5 true PHOTON eval 2 run 完走 + reports 2 件出力
- Phase B-C は #135 再学習開始をブロックし続けない

> **[DR2-005] PHOTON Drift metrics / Safe RecGen 指標の follow-up 化判定**:
> PR #1 merge (= #135 unblock 解禁) の **前** に、Phase A reports 内で以下のいずれかを確定する必要がある:
> (1) Drift metrics / Safe RecGen 指標を「再測定する」と明記し follow-up Issue 番号を確保する、または
> (2) 「本 Issue 範囲外 (out-of-scope)」と判定し判定理由を記録する。
> follow-up 番号が未確定のまま PR #1 を merge すると、#135 着手解禁後に無効な数値が誤参照されるリスクがある。
> Phase A0+A 完了の追加条件として「Drift metrics / Safe RecGen 指標の follow-up 化判定が reports に記載済みであること」を Phase A 受入条件に加える (§10.1 参照)。

---

## 4. データフロー / 処理フロー

### Phase A0 の checkpoint load 経路

```
configs/institutional_docs_photon.yaml
  model.checkpoint_path: "/path/to/mulmoclaude_600step/checkpoint_dir"
    # directory containing weights.npz and state.json
  model.provider: "photon"
         |
         v
pipeline_factory.build_pipeline(cfg)
  provider == "photon" → 分岐
         |
         v
_build_photon_deps(cfg)                  [baseline_reporag/photon_pipeline.py]
  1. PhotonConfig 構築 (ModelConfig / HierarchyConfig / TokenizerConfig)
  2. tokenizer_id 検証 + _load_hf_tokenizer() 呼び出し
  3. PhotonModel(photon_cfg)  ← random-init
  4. checkpoint_path = cfg.model.checkpoint_path   ← Phase A0 追加
  5. checkpoint_path directory 形状を検証 (weights.npz / state.json 必須)
  6. load_checkpoint(model, checkpoint_path)        ← Phase A0 追加 (photon_mlx/trainer.py)
     ├── 成功: _logger.info("checkpoint loaded")
     └── 失敗: RuntimeError で停止 (PHOTON_ALLOW_RANDOM_INIT=1 の test/CI 例外時のみ WARNING 継続)
  7. PhotonInference(model, photon_cfg, tokenizer, ...)
         |
         v
PhotonInference.__init__                 [photon_mlx/inference.py]
  _check_weight_initialization(model, cfg.model.embedding_random_init_threshold)
  ├── σ > threshold → WARNING "high variance — possibly random-init"
  │     ※ Phase A0 の load 成功後は σ が下がり、この WARNING は出ない想定
  └── σ <= threshold → silent (正常 loaded checkpoint)
         |
         v
PhotonRAGPipeline(cfg, baseline_deps, photon_deps)
```

**既存 random-init WARNING との関係**:
`_check_weight_initialization` は checkpoint load 後の state を **診断する** 層。Phase A0 実装後に正しく checkpoint が load されていれば σ が下がり WARNING は出なくなる。もし load 後も WARNING が出る場合は「checkpoint が corrupt か path 誤り」の診断材料として有効に機能する。

---

## 5. 互換性とマイグレーション戦略

### 5.1 photon profile yaml の保護 (Phase B-C 期間)

`configs/photon_*.yaml` 5 件は Phase D (#135 範囲) まで `tokenizer_id` / `vocab_size` を変更しない:

| yaml ファイル | vocab_size | tokenizer_id | Phase D での扱い |
|-------------|-----------|-------------|--------------|
| `configs/photon_small.yaml` | 152064 (Qwen2.5) | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | 採用 LLM に同期更新 |
| `configs/photon_long_context.yaml` | 152064 (Qwen2.5) | `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit` | 採用 LLM に同期更新 |
| `configs/photon_600m_paper.yaml` | **32000 (LLaMA-2 系)** | **`meta-llama/Llama-2-7b-hf`** | **Qwen2.5 系でないため Phase D で #135 と独立して個別評価** |
| `configs/photon_tiny.yaml` | **32000 (LLaMA-2 系)** | **`meta-llama/Llama-2-7b-hf`** | **同上** |
| `configs/photon_tiny_recgen.yaml` | **32000 (LLaMA-2 系)** | **`meta-llama/Llama-2-7b-hf`** | **同上** |

> **[DR2-001 修正]**: `photon_600m_paper.yaml` の実値は `vocab_size: 32000` / `tokenizer_id: meta-llama/Llama-2-7b-hf` (LLaMA-2 系) であり、Qwen2.5 系ではない (実コード `configs/photon_600m_paper.yaml:129-130` で確認済み)。保護コメント追記対象 (Qwen2.5 固定の理由) は **Qwen2.5 系 2 件 (`photon_small.yaml` / `photon_long_context.yaml`) のみ** が正しい。
>
> `photon_tiny.yaml` / `photon_tiny_recgen.yaml` / `photon_600m_paper.yaml` の 3 件は LLaMA-2 系 (vocab_size: 32000) であり、保護コメント対象外。代わりに「LLaMA-2 系 vocab のため Phase B-C の変更影響なし、Phase D で #135 と独立して対応方針を確認」を明記する。

Phase B-C 期間中は Qwen2.5 系 **2 件** (`photon_small.yaml` / `photon_long_context.yaml`) に以下コメントを追記 (保護コメントの Single Source of Truth は §3 設計判断 #3 の記述とし、§5.1 は参照のみ):

> 保護コメント仕様は §3 設計判断 #3 を参照。`photon_tiny.yaml` / `photon_tiny_recgen.yaml` / `photon_600m_paper.yaml` は LLaMA-2 系 (vocab_size: 32000) のため保護コメント対象外。

### 5.2 invariant test への影響

`tests/test_pipeline_factory_yaml_invariants.py` の現行 invariant:
- `GLOBAL_DEFAULT_RERANKER_MODEL_ID` (reranker model_id 固定) — Phase B-C で変更なし
- `INSTITUTIONAL_RERANKER_MODEL_ID` (bge-reranker-v2-m3) — 変更なし
- `test_photon_yaml_has_required_tokenizer_fields` (tokenizer_id / vocab_size 存在確認) — photon yaml を保護するため変更なし

**LLM model_id invariant の追加方針 (YAGNI-001: 設計方針書段階で決定)**:

| 案 | 内容 | メリット | デメリット |
|----|------|---------|----------|
| (A) | `test_baseline_yaml_generation_model_id_unchanged` を追加 | 誤変更防止 | LLM 変更時に毎回 test 更新が必要 |
| **(B)** | **LLM model_id は invariant 化しない** | **運用コスト低** | silent migration 防止は運用プロセスで管理 |

**決定: (B) LLM model_id は invariant 化しない**

**理由**:
- LLM 採用変更は Phase C の意識的な人間判断 (採用判定文書 + PR #3 review) で実施されるため、invariant による防護は YAGNI
- 新 LLM 採用毎に `test_baseline_yaml_generation_model_id_unchanged` を更新するコストが、invariant 化による silent migration 防止メリットを上回る
- backbone 差分は `configs/eval.yaml` のコメントと report の自動出力 (DR2-001 対応) で二重に明示するため、invariant 化なしでも運用上のリスクは許容範囲

Phase C 実施時にこの判断を覆す場合は、Phase C PR 説明に理由を明記し別途記録する。

### 5.3 新規 yaml の命名規則 (S3-002 対応)

新規 yaml 4 件の命名:
- `configs/baseline_qwen35.yaml` (baseline 用)
- `configs/institutional_docs_qwen35.yaml` (institutional 用)
- `configs/baseline_gemma4.yaml` (baseline 用)
- `configs/institutional_docs_gemma4.yaml` (institutional 用)

**`photon_` prefix を絶対に使わない**: `tests/test_pipeline_factory_yaml_invariants.py:_is_photon_profile_yaml` の検出対象となるため、非 PHOTON な baseline yaml に `photon_` prefix が付くと invariant check が誤検知する。PHOTON 拡張版 yaml は Phase D 範囲とし、命名予約のみ (`configs/institutional_docs_photon_<llm>.yaml` 等)。

---

## 6. セキュリティ / 運用設計

### 6.1 HF cache の disk 圧迫

| モデル | 推定サイズ | 対策 |
|--------|----------|------|
| Qwen3.x-9B-8bit | ~9GB | Phase B 前に `huggingface-cli download` で事前 warm-up |
| Gemma4-26B-4bit | ~13-15GB | 同上 + Mac Studio M3 Ultra (>=128GB) を必須環境とする |
| 合計追加 | ~22-24GB | HF cache path (`~/.cache/huggingface`) の空き容量確認必須 |

### 6.2 server.py cold-start での大容量 download 対策 (S7-004)

Phase C で `configs/baseline.yaml` を新 LLM に更新すると、`baseline_reporag/server.py` / `baseline_reporag/cli.py` の初回起動時に `mlx_lm.load()` が大容量 download を実行し、OOM / 起動遅延が発生しうる。

**対策**:
1. Phase C merge 前に採用 LLM の HF cache を実行環境に事前 download
2. Phase C 受入条件に cold-start smoke test を必須化: `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question 'test'` が 180s 以内に non-empty 応答を返すこと
3. `docs/deployment.md` に HF cache warm-up 手順を追記

### 6.3 rollback (旧 LLM 戻し) 経路

```
旧 LLM 戻しが必要な場合:
  configs/baseline_qwen25.yaml (Phase C で新設) を configs/baseline.yaml に上書き
  → server / CLI の再起動で旧 LLM に戻る (HF cache 残存であれば download 不要)
```

### 6.4 secrets 取り扱い

- HF token: Phase B の private model download が必要な場合は `HUGGING_FACE_HUB_TOKEN` 環境変数経由 (コードに埋め込まない)
- `configs/*.yaml` に token 値を書かない
- `workspace/` / `reports/` に PII を含む eval 入力を置かない
- `model_id` は原則 public slug として扱う。private HF org / R&D codename を含む model_id を使う場合、public artifact に出す report では org/name を redaction し、内部 report にのみ full slug を残す
- checkpoint load / HF download / config validation のログは raw config dict、HF token、API key、absolute local path を出力しない

### 6.5 Gemma4 MoE のメモリ要件

```
Phase B 実行前の必須確認:
  実行環境: Mac Studio M3 Ultra (>=128GB unified memory) を必須
  peak RSS 測定: mlx_lm.load(gemma4-slug) で peak RSS が unified memory の 80% 未満であること
  Mac mini (~64GB): failure 可能性あり → Phase B Gemma4 は Mac Studio 限定
```

### 6.6 コマンド実行の安全性 (DR4-003)

Phase A0/A/B の smoke / warm-up / slug 確認をスクリプト化する場合、外部入力 (`model_id`, `repo_id`, `question`, `checkpoint_path`, `--variants`) を shell 文字列へ連結しない。

- Python から実行する場合は `subprocess.run([...], shell=False, check=True)` の argv list 形式のみを許可する
- `os.system`, `subprocess.run(..., shell=True)`, f-string で組み立てた shell command は禁止する
- `python -c` に slug を文字列補間する自動化は禁止し、検証済み `model_id` を Python 関数引数として渡す小スクリプトまたは module entry point を使う
- 設計書内の shell コマンド例は operator 手順の説明であり、実装時の subprocess 呼び出し仕様ではない

### 6.7 config 入力検証 (DR4-001 / DR4-002)

- YAML は必ず `yaml.safe_load` で読む。`yaml.load`、`eval()`、`exec()` による config 解釈は禁止する
- `model.model_id` / `tokenizer.tokenizer_id` は HF repo-id allowlist (`<org>/<name>`, ASCII `[A-Za-z0-9._-]`, slash 1 個のみ) を通し、URL、local path、`..`、先頭 dot、`~`、改行・制御文字を拒否する
- Phase B/C の新 LLM config は `model.provider: "mlx_lm"` を維持し、`model_id` 以外の provider / loader 指定で任意コード実行面を広げない
- `checkpoint_path` は §3 の通り repo-local `checkpoints/` または `PHOTON_CHECKPOINT_ROOT` 配下に限定し、symlink escape を `resolve(strict=True)` 後の root containment で拒否する

### 6.8 HF artifact / dependency supply chain (DR4-005)

新 LLM 追加では Python package dependency を増やさないことを原則とし、HF model artifact は以下を満たすものだけを Phase B 候補にする:

- `mlx-community` または明示承認された org の repo に限定し、Phase B report に `repo_id`, `revision` (commit SHA), `updated_at`, `downloads`, quantization を記録する
- 可能な場合は config に `revision` を明記し、少なくとも report には eval 時点の commit SHA を残す
- `trust_remote_code=True` を必要とする model は本 Issue 範囲外とし、採用する場合は別 Issue で security review を実施する
- dependency 追加が必要になった場合は `requirements*.txt` / lock 相当ファイルの diff と供給元を PR 説明に明記する

### 6.9 権限・環境変数の安全性

- `PHOTON_ALLOW_RANDOM_INIT=1` は unit test / CI の negative-path 検証専用。Phase A eval、Phase B/C smoke、本番 server 起動では未設定であることを受入条件に含める
- Phase A report には `PHOTON_ALLOW_RANDOM_INIT` が未設定または `0` であることを記録する
- `HUGGING_FACE_HUB_TOKEN`, `OPENAI_API_KEY` 等は環境変数または GitHub Actions secrets のみから読む。CLI option、YAML、report への転記は禁止する

### 6.10 rollback config の安全性

`configs/baseline_qwen25.yaml` は rollback 用に残置されるため、作成時に以下を確認する:

- token / api_key / Authorization header / local absolute path を含まない
- `model_id` は public slug または redaction 方針が明記された approved private slug のみ
- baseline.yaml へ戻す手順では secret を含む local override file を commit しない

---

## 7. 影響範囲

| モジュール / ファイル | 変更種別 | Phase | 影響度 |
|--------------------|---------|-------|-------|
| `baseline_reporag/photon_pipeline.py:_build_photon_deps` | checkpoint loading 経路追加 | A0 | **高** (PHOTON pipeline の動作正確性 / silent random-init 再発防止) |
| `photon_mlx/inference.py` | WARNING 文言確認・必要時更新のみ | A0 | 低 |
| `docs/troubleshooting.md` / `docs/deployment.md` | `PHOTON_ALLOW_RANDOM_INIT` 環境変数の test/CI 用途限定の意図を documenting (DR2-007) | A0 | 低 |
| `configs/institutional_docs_photon.yaml` | `model.checkpoint_path` 追記、yaml コメント更新 | A | 高 |
| `reports/gate2_judgment_v5_post_s7001.md` (新規) | eval 結果 | A | - |
| `reports/institutional_photon_mt_eval_v2.md` (新規) | eval 結果 | A | - |
| `configs/baseline_qwen35.yaml` (新規) | `model.provider: mlx_lm` + 正式 slug | B | 中 |
| `configs/baseline_gemma4.yaml` (新規) | 同上 | B | 中 |
| `configs/institutional_docs_qwen35.yaml` (新規) | institutional 用 config | B | 中 |
| `configs/institutional_docs_gemma4.yaml` (新規) | 同上 | B | 中 |
| `reports/llm_baseline_comparison_2026q2.md` (新規) | 3 LLM 比較 eval 結果 | B | - |
| `configs/photon_*.yaml` (5 件) | 保護コメント追記のみ | B-C | 低 |
| `configs/baseline.yaml` | `model_id` を採用 LLM に更新 | C | **高** |
| `configs/baseline_qwen25.yaml` (新規) | rollback 用 旧 LLM config | C | 中 |
| `configs/eval.yaml` | baseline variant の LLM backbone 差分明記 + `report.show_llm_backbone: true` フラグ追加 (DR-2 backbone 自動出力) | C | 高 |
| `bench/run_all.py` | `configs/eval.yaml` の `report.show_llm_backbone` を読み、variant ごとの `model_id` を report / CI artifact に必ず出力する | C | 中 |
| `bench/tests/test_run_all.py` | backbone metadata 出力の regression test を追加し、人手 report 確認への依存をなくす | C | 中 |
| `baseline_reporag/eval/institutional/llm_client.py` | `QwenMLXAdapter` default model 方針決定・実施 | C | 中 |
| `tests/test_pipeline_factory_yaml_invariants.py` | LLM invariant 追加方針決定 | C | 中 |
| `.github/workflows/weekly_eval.yml` | timeout / threshold 見直し | C | 中 |
| `baseline_reporag/server.py` / `cli.py` | cold-start smoke test (コード変更なし可) | C | 中 |
| `docs/deployment.md` | memory footprint / model_id / warm-up 手順更新 | C | 低 |
| `docs/troubleshooting.md` | HF cache path / memory footprint 更新 | C | 低 |
| `docs/tutorial.md` | model_id / ~8GB 記述更新 | C | 低 |
| `workspace/mvp/architecture.md` | memory footprint 更新 | C | 低 |
| `workspace/mvp/app_guide.md` | memory footprint 更新 | C | 低 |
| `workspace/mvp/metrics.md` | Qwen temp/top_p variance 記述更新 | C | 低 |
| `README.md` | model_id 行更新 | C | 低 |
| `docs/llm_choice_decision_2026q2.md` (新規) | 採用 LLM 選定理由 | C | - |
| `CLAUDE.md` | LLMバックエンド行 + 品質チェック表更新 | C | 低 |
| `configs/photon_*.yaml` (5 件) | tokenizer_id / vocab_size 採用 LLM に同期 | D (#135) | **高** |

---

## 8. 品質基準

| チェック | コマンド | 基準 |
|---------|---------|------|
| テスト | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全パス (既知 pre-existing failure 2 件: `tests/test_generate_training_corpus.py` 除く) |
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| Baseline 疎通 | `python -m baseline_reporag.cli --config configs/baseline.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり |
| Phase A smoke | `python -m baseline_reporag.cli --config configs/institutional_docs_photon.yaml --repo-id fastapi_fastapi --question "test"` | 応答あり + checkpoint load ログ確認 |
| Phase B smoke (Qwen3.x) | `python -m baseline_reporag.cli --config configs/baseline_qwen35.yaml --repo-id fastapi_fastapi --question "test"` | non-empty 応答、tokenizer mismatch なし、<180s |
| Phase B smoke (Gemma4) | `python -m baseline_reporag.cli --config configs/baseline_gemma4.yaml --repo-id fastapi_fastapi --question "test"` | 同上 + peak RSS < unified memory × 80% |
| Security static check | `rg -n "shell=True|os\\.system|yaml\\.load\\(|(^|[^.])\\b(eval|exec)\\(" baseline_reporag photon_mlx bench scripts tests` | 意図しない shell / unsafe yaml / Python code eval なし (`mx.eval` は対象外) |
| Secret scan (configs/reports) | `rg -n "HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|api[_-]?key|token:|Authorization|Bearer " configs reports workspace/mvp docs README.md` | secret 値の commit / artifact 混入なし |

---

## 9. リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| **S5-001 silent bug 再発**: Phase A0 実装で checkpoint load 経路が誤実装される | PHOTON が random-init のまま eval | fail-fast 設計: `checkpoint_path` 設定時の load 失敗は `RuntimeError` を raise して即座に停止。Phase A eval 前に起動ログの `checkpoint loaded from <path>` INFO 行を report に貼付して確認する (欠如時は eval 結果を invalid 扱い) |
| **mulmoclaude 600-step ckpt の所在不明** (H5 Partially Confirmed) | Phase A 開始不能 | Phase A0 最初の subtask として `weights.npz` / `state.json` を含む local checkpoint directory を特定し `configs/institutional_docs_photon.yaml` に設定する。HF Hub URL / safetensors しかない場合は本 Issue 内で直接指定せず、resolver / converter を別途切り出す。未特定なら Phase A を開始しない |
| **Qwen3.x / Gemma4 slug が HF 上に存在しない** (H9 Unverifiable) | Phase B 中断 | Phase B 着手前に `huggingface-cli repo info` で存在確認、不在時は近い alternative に切替えて Issue 更新 |
| **mlx-lm が新 LLM の loader を未提供** | Phase B scope 外化 | Phase B 最初の subtask として mlx-lm の現バージョンで Qwen3 / Gemma4 loader 提供を確認。未提供なら別 Issue 切り出し |
| **Gemma4 26B MoE の初回 download + peak memory が OOM** | Phase B 実行不能 | Mac Studio M3 Ultra (>=128GB) 必須、Phase B 開始前に peak RSS 測定、80% 超なら Phase B Gemma4 を別機に移管 |
| **photon profile yaml (photon_*.yaml) の vocab_size が新 LLM と不整合** | #138 invariant で ValueError → PHOTON pipeline 全停止 | Phase B-C 期間中は photon profile yaml を Qwen2.5 系のまま維持 (保護コメント明記) |
| **configs/baseline.yaml 更新で weekly_eval.yml が silent migration → CI timeout / threshold 誤検知** | CI 安定性低下 | Phase C 受入条件に `workflow_dispatch` ドライラン必須化、timeout / threshold 事前評価 |
| **configs/eval.yaml の baseline variant が silent migration し PHOTON variant と LLM backbone がズレる** | benchmark 比較の解釈を誤る | Phase C で eval.yaml のコメントを更新し、report に backbone 差分を明記 |
| **grader (qwen3.5:27b) と被評価 LLM が同系列で self-preference bias** | eval 信頼性低下 | Phase B 受入条件に cross-check grader (openai/gpt-4o-mini 等) による bias 検証を追加 |
| **Gemma4 MoE expert routing 確率性で 2-runs 平均が信頼区間として不十分** | NC 数値の再現性低下 | Phase B で各 LLM × 同質問 × 5 runs の nondeterminism 検証を実施、variance 高い場合は 3 runs に増加 |
| **server.py / cli.py の既定 baseline.yaml が新 LLM に切替わり、初回 request で download / OOM / 起動遅延** | 本番運用の不意な停止 | Phase C で cold-start smoke、HF cache warm-up、rollback config (`configs/baseline_qwen25.yaml`) の確認 |
| **llm_client.py の QwenMLXAdapter hardcode で institutional eval set が旧 LLM で再生成される** | eval set 信頼性低下 | Phase C で方針を確定。yaml-driven 変更を選んだ場合は本 Issue 内で実施 |
| **#143 (Qwen nondeterminism) 未解消で 2 runs 平均の信頼区間が不明** | NC 数値の再現性低下 | 未解消のまま進める場合は各 run の variance を report に併記し、NC ± std の信頼区間を明示 |
| **PHOTON Drift metrics / Safe RecGen 指標が invalid のまま残る** | 後続 Issue で無効な数値の誤参照 | Phase A reports 内で「再測定する / out-of-scope 判定 + follow-up 番号」のいずれかを明記 |
| **checkpoint_path が許可 root 外へ抜ける** | 任意 local path の checkpoint 読み込み、absolute path のログ露出 | repo-local `checkpoints/` または `PHOTON_CHECKPOINT_ROOT` 配下に限定し、`resolve(strict=True)` 後の root containment と secret-free logging を必須化 |
| **model_id が local path / URL / traversal 形状になる** | 意図しない local artifact / 未承認 HF repo の load、supply chain リスク増加 | `model.model_id` に HF repo-id allowlist を適用し、URL / path / dot segment / control char を拒否。HF revision を report に記録 |
| **smoke / warm-up 自動化で shell injection が混入する** | model_id / question / repo_id 経由の任意コマンド実行 | `subprocess.run([...], shell=False)` の argv list のみ許可し、`os.system` / `shell=True` / string-built shell を禁止 |
| **backbone 自動出力で private model_id / local path が artifact に漏れる** | private org 名、R&D codename、local username/path の漏洩 | model_id は public slug 前提。private slug は redaction 方針を明記し、checkpoint path は basename / relative path のみログ出力 |

---

## 10. テスト戦略

### 10.1 Phase A0: checkpoint load 検証 (新規 unit test)

`baseline_reporag/tests/test_photon_pipeline.py` に以下を追加:

```python
# [DR2-003] _make_photon_cfg ヘルパは必ず有効な tokenizer 設定を注入すること。
# _build_photon_deps は _load_hf_tokenizer 呼び出し前に tokenizer_id 必須チェック
# (Issue #139 invariant: _validate_tokenizer_id) を実行するため、tokenizer_id が
# 未設定または不正な場合は ValueError が先行して raise され、load_checkpoint の
# 呼び出し検証に到達しない。
#
# 対策: (推奨) _validate_tokenizer_id を mock で bypass する。
#   mocker.patch("baseline_reporag.photon_pipeline._validate_tokenizer_id",
#                return_value="test-org/test-model")
# または _make_photon_cfg が返す cfg に以下を必ず含める:
#   cfg.tokenizer.tokenizer_id = "test-org/test-model"
#   cfg.tokenizer.vocab_size = 32000

def test_build_photon_deps_loads_checkpoint_when_path_set(tmp_path, monkeypatch, mocker):
    """_build_photon_deps が checkpoint_path 設定時に load_checkpoint を呼ぶことを確認"""
    mock_load = mocker.patch("baseline_reporag.photon_pipeline.load_checkpoint")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "weights.npz").write_bytes(b"test")
    (ckpt / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    cfg = _make_photon_cfg(checkpoint_path=str(ckpt))
    # tokenizer_id 検証を bypass (Issue #139 invariant が先行 raise するため必須)
    mocker.patch(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        return_value="test-org/test-model",
    )
    # PhotonModel / HF tokenizer は mock
    mocker.patch("baseline_reporag.photon_pipeline.PhotonModel")
    mocker.patch("baseline_reporag.photon_pipeline._load_hf_tokenizer")
    _build_photon_deps(cfg)
    mock_load.assert_called_once()

def test_build_photon_deps_rejects_invalid_checkpoint_shape(tmp_path, mocker):
    """weights.npz / state.json がない path を fail-fast で拒否することを確認"""
    mock_load = mocker.patch("baseline_reporag.photon_pipeline.load_checkpoint")
    mocker.patch("baseline_reporag.photon_pipeline.PhotonModel")
    mocker.patch("baseline_reporag.photon_pipeline._load_hf_tokenizer")
    mocker.patch(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        return_value="test-org/test-model",
    )
    cfg = _make_photon_cfg(checkpoint_path=str(tmp_path / "missing_ckpt"))
    with pytest.raises(RuntimeError, match="weights.npz.*state.json|checkpoint_path"):
        _build_photon_deps(cfg)
    mock_load.assert_not_called()

def test_build_photon_deps_rejects_checkpoint_outside_allowed_root(tmp_path, monkeypatch, mocker):
    """checkpoint_path が許可 root 外へ抜ける場合は load_checkpoint 前に拒否する"""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (outside / "weights.npz").write_bytes(b"test")
    (outside / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(allowed))
    mock_load = mocker.patch("baseline_reporag.photon_pipeline.load_checkpoint")
    mocker.patch("baseline_reporag.photon_pipeline.PhotonModel")
    mocker.patch("baseline_reporag.photon_pipeline._load_hf_tokenizer")
    mocker.patch(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        return_value="test-org/test-model",
    )
    cfg = _make_photon_cfg(checkpoint_path=str(outside))
    with pytest.raises(RuntimeError, match="approved checkpoint roots"):
        _build_photon_deps(cfg)
    mock_load.assert_not_called()

def test_build_photon_deps_warns_when_no_checkpoint(caplog, mocker):
    """checkpoint_path 未設定時に WARNING ログが出ることを確認"""
    mocker.patch("baseline_reporag.photon_pipeline.PhotonModel")
    mocker.patch("baseline_reporag.photon_pipeline._load_hf_tokenizer")
    mocker.patch(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        return_value="test-org/test-model",
    )
    cfg = _make_photon_cfg(checkpoint_path=None)
    with caplog.at_level(logging.WARNING, logger="baseline_reporag.photon_pipeline"):
        _build_photon_deps(cfg)
    assert any("random-init" in r.message or "checkpoint_path not set" in r.message
               for r in caplog.records)

def test_build_photon_deps_raises_on_load_failure_by_default(tmp_path, monkeypatch, mocker):
    """checkpoint load 失敗時はデフォルトで RuntimeError を raise することを確認"""
    mocker.patch(
        "baseline_reporag.photon_pipeline.load_checkpoint",
        side_effect=FileNotFoundError("not found"),
    )
    mocker.patch("baseline_reporag.photon_pipeline.PhotonModel")
    mocker.patch("baseline_reporag.photon_pipeline._load_hf_tokenizer")
    mocker.patch(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        return_value="test-org/test-model",
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "weights.npz").write_bytes(b"test")
    (ckpt / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    cfg = _make_photon_cfg(checkpoint_path=str(ckpt))
    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        _build_photon_deps(cfg)

def test_build_photon_deps_allows_random_init_only_with_env(tmp_path, caplog, monkeypatch, mocker):
    """PHOTON_ALLOW_RANDOM_INIT=1 の test/CI 例外時のみ WARNING 継続することを確認"""
    mocker.patch(
        "baseline_reporag.photon_pipeline.load_checkpoint",
        side_effect=FileNotFoundError("not found"),
    )
    mocker.patch("baseline_reporag.photon_pipeline.PhotonModel")
    mocker.patch("baseline_reporag.photon_pipeline._load_hf_tokenizer")
    mocker.patch(
        "baseline_reporag.photon_pipeline._validate_tokenizer_id",
        return_value="test-org/test-model",
    )
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "weights.npz").write_bytes(b"test")
    (ckpt / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PHOTON_CHECKPOINT_ROOT", str(tmp_path))
    cfg = _make_photon_cfg(checkpoint_path=str(ckpt))
    monkeypatch.setenv("PHOTON_ALLOW_RANDOM_INIT", "1")
    with caplog.at_level(logging.WARNING, logger="baseline_reporag.photon_pipeline"):
        _build_photon_deps(cfg)
    assert any("checkpoint load failed" in r.message for r in caplog.records)
```

### 10.2 新 LLM smoke test の実装方針 (Phase B)

- `configs/baseline_qwen35.yaml` / `configs/baseline_gemma4.yaml` に正式 slug 設定後、CLI smoke test をスクリプト化
- nondeterminism 検証: 同一質問 × 5 runs → NC variance を計算し、variance > 閾値 (TBD) で Phase B eval runs を 3 に増やす
- grader bias 検証: Qwen3.x 評価時に qwen3.5:27b grader の self-preference bias を openai/gpt-4o-mini でクロスチェック

### 10.3 invariant test の追加方針 (Phase C)

**決定済み (YAGNI-001 対応): (B) LLM model_id は invariant 化しない**

LLM model_id invariant 方針の詳細は **§5.2 を SSOT** として参照。実装観点では §10.4 の現行 invariant (`INSTITUTIONAL_RERANKER_MODEL_ID` / `test_photon_yaml_has_required_tokenizer_fields`) に変更なし。

> **[DR2-008]**: §5.2 を SSOT として整理済み。本節では判断の参照と実装への注意のみを記載し、決定理由の重複記述は §5.2 に集約した。Phase C 実施時にこの判断を覆す場合は Phase C PR 説明に理由を明記すること。

### 10.4 セキュリティ regression test (Phase A0-C)

- `model.model_id` / `tokenizer.tokenizer_id` の unsafe 形状 (`http://...`, `../../x`, `.cache/x`, `org/..`, 改行入り) を拒否する table-driven test を追加または既存 tokenizer_id test と同等に拡張する
- checkpoint path は `PHOTON_CHECKPOINT_ROOT` 配下の valid directory のみ通し、root 外 directory / symlink escape / missing required file を `load_checkpoint` 前に拒否する
- smoke / warm-up の automation helper を追加する場合は、`shell=True` / `os.system` が使われていないことを AST または `rg` ベースの regression test で固定する
- report / log 出力に `HUGGING_FACE_HUB_TOKEN`, `OPENAI_API_KEY`, `Bearer `, raw absolute checkpoint path が混入しないことを snapshot または string assertion で確認する

### 10.5 既存テストへの影響

- `tests/test_pipeline_factory_yaml_invariants.py` の reranker invariant: Phase A0-B で reranker config を変更しないため影響なし
- `tests/test_pipeline_factory_yaml_invariants.py:test_photon_yaml_has_required_tokenizer_fields`: photon profile yaml を保護するため影響なし
- `photon_mlx/tests/`: checkpoint load path の追加は `_build_photon_deps` 内であり、`PhotonInference` の unit test には影響しない
- `baseline_reporag/tests/test_photon_pipeline.py`: `_build_photon_deps` の mock pattern に checkpoint load mock を追加する必要がある場合は既存 test を確認・更新

---

---

## Future Work (Nice to Have — 本 Issue 範囲外)

以下は Stage 1 レビューで「Nice to Have」と分類された項目のうち、本 Issue の実装範囲では対応しないが、将来の保守・拡張時に参照すべき方針を記録する。

### FW-1: `_build_photon_deps` の責務分割ポリシー (SRP-001)

現状 (Phase A0 実装後) の `_build_photon_deps` は PhotonConfig 構築 + tokenizer 検証 + PhotonModel 初期化 + checkpoint load + 診断ログという多目的関数になる。現行では許容範囲だが、将来の拡張時に備えた責務分割ポリシー:

> `_build_photon_deps` の責務が **200 行以上** / または **3 つ以上の provider 分岐** に達した時点で、`_load_photon_checkpoint(model, cfg)` 等のヘルパー関数に抽出する。この分割は OCP/SRP の長期維持のため、Phase D 以降の provider 追加時に評価すること。

### FW-2: `pipeline_factory` の interface 契約明文化 (ISP-001)

`pipeline_factory.py` が公開すべき interface 契約 (Phase D 以降の新 provider 追加時に参照):

- **input contract**: `cfg.model.provider` が `"photon"` / `"baseline"` / `"mlx_lm_only"` (将来) 等の文字列
- **output contract**: `QueryResult` を返す pipeline オブジェクト
- **隠蔽する実装詳細**: MLX import / PhotonModel 構築 / checkpoint load の全詳細
- Phase D 以降に provider が 3 種以上に増える場合は、`pipeline_factory.py` 内に上記契約を docstring / type hint で明文化する。

### FW-3: §6 と §9 の重複整理 (DRY-002)

§6 (セキュリティ / 運用設計) と §9 (リスク表) の一部項目 (cold-start 大容量 download / Gemma4 MoE OOM) は内容が重複している。将来の設計書リファクタリング時:

- §6 は「対策の方針 (設計上の決定事項)」に絞る
- §9 は「リスク × 緩和策の cross-reference 表」に絞り、§6 への参照リンクを追加する
- 現時点では可読性維持を優先し重複を許容する。

---

## 参考資料

- Issue #148: https://github.com/Kewton/photon-mlx/issues/148
- Issue 本文 (最新): `workspace/issues/148/design/latest-issue-body.md`
- 仮説検証: `workspace/issues/148/issue-review/hypothesis-verification.md`
- Stage 5 レビュー (Codex): `workspace/issues/148/issue-review/stage5-review-result.json`
- Stage 7 レビュー (Codex): `workspace/issues/148/issue-review/stage7-review-result.json`
- マルチステージレビュー完了報告: `workspace/issues/148/issue-review/summary-report.md`
- 比較基準 (Gate 2 v4): `reports/gate2_judgment_v4_final.md` (Static NC baseline 21.7% / PHOTON 20.0%, MT NC 6.7%)
- 関連 Issue: #135 (本格再学習), #138 (tokenizer mismatch — CLOSED), #139, #140, #143 (eval reproducibility), #144
- 参考設計方針書 (#140): `workspace/design/issue-140-review-process-design-policy.md`
