## 背景

#137 5-variant A/B (PR #142 = MERGED 済 / 2026-04-26) で V2 (`cl-nagoya/ruri-small-v2`) が **build 失敗**となり計測不能のまま V4 採用が確定した。本 Issue は V2 のみを補完して 5-variant 計測を完成させるフォローアップ作業である。

worker の調査で "sentence-transformers 5.4.0 + ruri モデル config の互換性問題" と分類されたが、本セッションでの追加調査で**実際の原因は HF cache に `spiece.model` がダウンロードされていないこと**と判明。

## 推定真因 (実測ベースの仮説)

> 以下は再現マシン (M3 Ultra, sentence-transformers 5.4.0) での実測ベースの仮説。Option A 適用後に load が成功した場合に限り、(1) cache 漏れ仮説が真因として確定する。失敗した場合は (2)/(3) の互換性問題に該当するため、後述「実行順序とフォールバック分岐」に従って判別する。

1. ruri-small-v2 リポジトリ (https://huggingface.co/cl-nagoya/ruri-small-v2/tree/main) には:
   - `model.safetensors` (272 MB) ← cache あり
   - `tokenizer_config.json` (1.7 KB) ← cache あり
   - `special_tokens_map.json` (970 B) ← cache あり
   - **`spiece.model` (439 KB)** ← **cache 漏れ**
2. sentence-transformers 5.4.0 + transformers 5.5.3 の組合せでは、DistilBert architecture の SentencePiece tokenizer (`spiece.model`) を自動 fetch できていない (推定)
3. AutoTokenizer fallback も失敗 → AutoProcessor fallback も failed → "Unrecognized processing class" として最終的に raise (実測 traceback ベース)

## 検証

```bash
# spiece.model 不在を確認
$ find ~/.cache/huggingface/hub/models--cl-nagoya--ruri-small-v2 -name "*spiece*"
(空)

# HF Hub には存在
$ curl -s https://huggingface.co/api/models/cl-nagoya/ruri-small-v2/tree/main | grep spiece
  spiece.model (439391 bytes)

# AutoTokenizer 単独でも fail
$ python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('cl-nagoya/ruri-small-v2')"
ValueError: Couldn't instantiate the backend tokenizer from one of: ...

# ruri-base (同じ vendor) も同じ問題 (本 Issue のスコープ外、参考情報)
$ python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cl-nagoya/ruri-base')"
ValueError: Unrecognized processing class in cl-nagoya/ruri-base
```

## ワークアラウンド (簡単な fix)

### Option A: 手動 spiece.model 取得 (推奨)

```bash
# huggingface-cli で個別ファイルを取得
huggingface-cli download cl-nagoya/ruri-small-v2 spiece.model

# または python から
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('cl-nagoya/ruri-small-v2', 'spiece.model')"
```

### Option B: ライブラリバージョン pin (Option A 失敗時の fallback)

```bash
# ⚠ V0/V1/V3/V4 は既に sentence-transformers 5.4.0 で計測完了済みのため、
#   既存環境を破壊しない専用 venv で隔離すること。
#   現 repo には pyproject.toml / setup.py がないため `pip install -e .` は使わず、
#   repo root から requirements.txt + version constraint を install して実行する。
#   Option B を採用する場合は `.gitignore` に `.venv-ruri-*/` を追加してから作成する。
python -m venv .venv-ruri-v2
source .venv-ruri-v2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt "transformers<5.0" "sentence-transformers<5.0"
```

### Option C: 上流に fix を提案 (恒久対応)

sentence-transformers (or huggingface_hub) の側で DistilBert + SentencePiece の自動取得に対応するよう upstream PR / issue 報告。

### 実行順序とフォールバック分岐

1. **Step 1**: Option A 実行 (`scripts/_setup_ruri_models.py`)
2. **Step 2**: load 検証 (`python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('cl-nagoya/ruri-small-v2')"`)
3. **Step 3a**: 成功 → 真因 (1) cache 漏れ確定。Task 2 (build/eval) へ進む
4. **Step 3b**: 失敗かつ traceback が `Couldn't instantiate the backend tokenizer` 系 → 真因 (2)/(3) の sentence-transformers 互換性問題に該当 → Option B (専用 venv で旧版 install) に分岐
5. **Step 3c**: 失敗かつ別系統の error → Option C (上流報告) を選択し本 Issue を blocked にエスカレーション

## ゴール

V2 (ruri-small-v2) を fix し、5-variant 比較を完成させる。本格的な V2 vs V4 比較が成立すれば、日本語特化 small embedding (ruri) と多言語 large embedding (bge-m3) のトレードオフが定量化できる。

## 変更内容 (案)

### Task 1: ruri が要求するファイルを取得する補助スクリプト

> **依存契約**: 本 script は `huggingface_hub` を直接 import するため、`requirements.txt` に明示宣言を追加する (現状 `transformers` / `sentence-transformers` の transitive 依存に依存している)。これにより transitive バージョン range の変動による break-glass 余地を確保する。

```python
# scripts/_setup_ruri_models.py (新規)
"""
ruri 系モデル (spiece.model) を明示的に取得する。
sentence-transformers / transformers の version 違いに依存しない。

スコープ: 本 Issue では ruri-small-v2 のみを対象とする (受入条件参照)。
将来 ruri-base 等を追加する場合は RURI_MODELS list を拡張する。
#135 (PHOTON 再学習) で ruri を tokenizer 候補にする場合も本 list 拡張で再利用可能。
"""
from __future__ import annotations

import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError as e:
    raise ImportError(
        "huggingface_hub が必要です。`pip install huggingface_hub` または requirements.txt 経由で install してください。"
    ) from e

RURI_MODELS = ['cl-nagoya/ruri-small-v2']
# 将来追加候補: 'cl-nagoya/ruri-base' (本 Issue 対象外、必要時 list 拡張)
RURI_FILES = ['spiece.model']


def fetch_ruri_files() -> None:
    for model_id in RURI_MODELS:
        for filename in RURI_FILES:
            try:
                path = hf_hub_download(model_id, filename)
                print(f"Fetched {filename} for {model_id} -> {path}")
            except (HfHubHTTPError, OSError) as e:
                print(f"ERROR: failed to fetch {filename} for {model_id}: {e}", file=sys.stderr)
                raise


def verify_load() -> None:
    """Setup 成功を fail-fast で検知する load 検証。"""
    from sentence_transformers import SentenceTransformer
    for model_id in RURI_MODELS:
        SentenceTransformer(model_id)
        print(f"OK: SentenceTransformer({model_id}) loaded successfully")


if __name__ == "__main__":
    fetch_ruri_files()
    verify_load()
```

#### Task 1 の test 方針

`scripts/` 配下は `tests/test_no_scaffolding_in_prod.py` の `PROD_ROOTS` 対象外 (Issue #139 設計判断)、かつ本 script は network/HF Hub 依存の運用 helper のため **unit test 対象外** とする。代わりに `verify_load()` の手動実行 log を Issue にコメント記録して受入根拠とする。

### Task 2: V2 build を再試行 + eval

V2 だけ単独で build_indexes + run_baseline_eval を実行し、報告書に追加。既に V0/V1/V3/V4 のデータは確定しているので、V2 だけを補完すれば 5-variant 完走となる。

#### V2 設定の固定パラメータ (V4 設定との比較整合性確保)

> ⚠ V2 比較が `reports/institutional_retrieval_ab.md` の他 variant と直接比較可能であるためには、以下の固定値を維持すること。誤って `configs/institutional_docs.yaml` (V4 = bge-m3, max_input_chars=8192, bge-reranker-v2-m3) ベースで build すると **V2 ruri × V4 component のキメラ比較** となり報告に使えない。

| パラメータ | V2 値 | 出典 |
|----------|-------|------|
| `embedding.model_id` | `cl-nagoya/ruri-small-v2` | 本 Issue |
| `embedding.max_input_chars` | `2048` | reports/institutional_retrieval_ab.md:17 (V2 行) |
| `embedding.batch_size` | `32` (要確認) | V0-V3 と同等想定 |
| `embedding.normalize` | `true` | 既存 baseline |
| `chunking.max_chars` | `800` | #137 V0-V3 共通 |
| `chunking.overlap_chars` | `100` | #137 V0-V3 共通 |
| `reranker.model_id` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reports/institutional_retrieval_ab.md:17 |
| `repo.repo_id` | `institutional_documents_V2` | #137 work-plan.md:49 |

#### 具体実行手順

`configs/institutional_docs.yaml` は `tests/test_pipeline_factory_yaml_invariants.py:81-87` で `BAAI/bge-m3` に pin されている (invariant test) ため、**直接書き換え禁止**。#137 と同じく `configs/_experiments/` 配下の variant config (gitignored: `.gitignore:configs/_experiments/`) を再生成する運用とする。

```bash
# 1. 専用 variant config を作成 (full copy + override)
mkdir -p configs/_experiments
cp configs/institutional_docs.yaml configs/_experiments/institutional_V2.yaml
# Edit configs/_experiments/institutional_V2.yaml で上表の固定パラメータを上書き。
# 併せて repo.repo_path / repo.repo_commit が下記 ingest 引数と同値で残っていることを確認。

# 2. ingest 再実行 (repo_id=institutional_documents_V2 で SQLite chunk store を再構築)
python scripts/ingest_repo.py \
  --repo /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \
  --commit 9e500539f29555364217b773368305e7f59aa026 \
  --config configs/_experiments/institutional_V2.yaml \
  --repo-id institutional_documents_V2

# 3. build_indexes (BM25 + embedding + symbol graph)
python scripts/build_indexes.py \
  --config configs/_experiments/institutional_V2.yaml \
  --repo-id institutional_documents_V2

# 4. run_baseline_eval (predictions JSONL を logs/ に保存)
python scripts/run_baseline_eval.py \
  --config configs/_experiments/institutional_V2.yaml \
  --repo-id institutional_documents_V2 \
  --eval-set data/eval_sets/institutional_static_eval.jsonl \
  --output logs/institutional_V2_$(date +%Y%m%d_%H%M%S).jsonl
```

### Task 3: institutional_retrieval_ab.md の補正

V2 計測結果を取り込んで report 更新。V4 採用判定自体は変わらないが、V2 vs V4 比較が判定の根拠の一つとして加わる。

加えて、`configs/institutional_docs.yaml:84-87` のコメント `# #137 Phase B: 5-variant A/B (V0 e5-small, V1 e5-base, V2 ruri 計測不能, ...)` を V2 計測値で更新する (例: 「V2 ruri 計測不能」→ 「V2 ruri-small-v2 NC X.XX%」)。

> **PR 起点**: PR #142 は既に MERGED (2026-04-26) のため、本 Task 3 の `reports/institutional_retrieval_ab.md` 更新は **本 Issue 用 feature branch (`feature/issue-144-ruri-v2-build`) からの新規 PR** で実施する。

## 受入条件

- [ ] Task 0 (前提): `requirements.txt` に `huggingface_hub` を明示追加 (Task 1 が直接 import するため)。Option B を採用する場合は `.gitignore` に `.venv-ruri-*/` も追加
- [ ] Task 1: `scripts/_setup_ruri_models.py` 作成、ruri が動作する environment セットアップを再現可能化 (load 検証 step 込み)。unit test は不要 (network/HF Hub 依存の運用 helper)、`verify_load()` 手動実行 log を Issue コメントに記録する
- [ ] Task 2: V2 build_indexes + run_baseline_eval 完走、predictions JSONL が `logs/institutional_V2_*.jsonl` に保存
- [ ] Task 3: `reports/institutional_retrieval_ab.md` に V2 結果反映 (NC rate、category breakdown、latency)
- [ ] Task 3 (追加): `configs/institutional_docs.yaml:84-87` のコメントを V2 計測値で更新
- [ ] V4 採用判定は不変 (V2 が V4 を超えていない場合)
- [ ] V2 が V4 を上回った場合 (低確率): #137 reopen + 新 PR で `configs/institutional_docs.yaml` 採用 variant、`tests/test_pipeline_factory_yaml_invariants.py` の institutional pin 定数、`docs/deployment.md` の institutional memory note を更新 (採用判定再評価)
- [ ] Option A 失敗時は「実行順序とフォールバック分岐」に従い Option B/C を選択し、判別経過を Issue にコメント記録

## 影響ファイル

- `scripts/_setup_ruri_models.py` (新規、commit 対象)
- `requirements.txt` (1 行追加: `huggingface_hub`)
- `configs/_experiments/institutional_V2.yaml` (新規 — gitignored、commit 対象外)
- `data/indexes/institutional_documents_V2/embedding/` (新規 — build 結果、gitignored)
- `logs/institutional_V2_*.jsonl` (新規 — eval 結果、gitignored)
- `reports/institutional_retrieval_ab.md` (更新、PR で commit)
- `configs/institutional_docs.yaml` (コメント更新のみ、L84-87、PR で commit)
- `.gitignore` (Option B 採用時のみ: `.venv-ruri-*/` を追加)
- `tests/test_pipeline_factory_yaml_invariants.py` (V2 採用時のみ: institutional pin 定数を更新)
- `docs/deployment.md` (V2 採用時のみ: institutional memory note を更新)

> **ドキュメント波及確認**: `docs/` 配下に ruri / spiece / V2 関連の記載なし、`reports/institutional_retrieval_ab.md` のみが更新対象 (Stage 3 影響範囲レビュー S3-006 確認済)。
> **CI/CD 影響**: `.github/workflows/weekly_eval.yml` は `configs/baseline.yaml` のみ参照、`configs/_experiments/` 不参照のため CI 影響なし (Stage 3 S3-005 確認済)。
> **既存 indexes 影響**: `repo.repo_id=institutional_documents_V2` で V0/V1/V3/V4 と subdir 分離されるため、命名衝突なし (Stage 3 S3-004 確認済)。

## 想定 compute

- spiece.model 取得: ~1 min (439 KB)
- V2 ingest 再実行: ~5-10 min (institutional_documents repo を再 chunk)
- V2 build_indexes: ~30-60 min (ruri-small-v2 で 4228 docs を re-embed)
- V2 run_baseline_eval: ~30 min
- Report + config コメント更新: ~15 min
- **合計: ~1.5-2h**

## 並列性

#138 / #139 / #140 と独立、いつでも実施可。GPU 占有時間は短いため #135 学習との競合も限定的。

## 優先度

🟢 **低** — V4 採用判定は既に成立、V2 結果は将来検討用 (ruri 系列を今後使うかの判断材料)。Phase 2 完了 (#116) までに完了すれば良い。

## 関連

- 元: #137 (5-variant A/B、V2 のみ計測不能)
- 関連: #135 (本格再学習、ruri を training tokenizer 候補にする可能性)
