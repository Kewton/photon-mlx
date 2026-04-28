# Issue #144 設計方針書 — ruri-small-v2 build failure follow-up

**Issue**: #144 (test(retrieval): ruri-small-v2 (V2) build failure follow-up — manual spiece.model download workaround)
**作成日**: 2026-04-28
**作成者**: Claude (PM Auto Issue2Dev Phase 2)
**ブランチ**: feature/issue-144-ruri-v2-build

---

## 1. 目的とスコープ

### 1.1 目的

#137 5-variant A/B (PR #142 = MERGED 済) で **build 失敗 / 計測不能** だった V2 (`cl-nagoya/ruri-small-v2`) を補完計測し、5-variant 比較を完成させる。本格的な V2 vs V4 (BAAI/bge-m3) 比較を成立させ、日本語特化 small embedding と多言語 large embedding のトレードオフを定量化する。

### 1.2 スコープ (本 Issue で扱う)

- **In Scope**:
  - `scripts/_setup_ruri_models.py` 新規作成 (huggingface_hub 経由で `spiece.model` を明示的に download)
  - `requirements.txt` に `huggingface_hub` を 1 行追加 (依存契約の明示化)
  - `configs/_experiments/institutional_V2.yaml` 再生成 (gitignored)
  - V2 ingest + build_indexes + run_baseline_eval 完走
  - `reports/institutional_retrieval_ab.md` に V2 結果反映
  - `configs/institutional_docs.yaml` コメント更新 (L84-87)

- **Out of Scope** (本 Issue では扱わない):
  - sentence-transformers / transformers のバージョン pin (Option B fallback として記載するが、Option A 成功時は不要)
  - 上流ライブラリへの fix PR (Option C、Option A/B 失敗時のみ別 Issue 化)
  - ruri-base 等の追加対応 (将来 #135 等で必要時に list 拡張)
  - V4 採用判定の変更 (V2 が V4 を上回らない限り不変)

### 1.3 採用判定の前提

- **本 Issue の primary outcome**: 5-variant A/B 表 (`reports/institutional_retrieval_ab.md`) を完成させること
- **採用判定**: V2 が V4 を上回る確率は低 (small=67M params vs bge-m3=568M params)。上回った場合のみ #137 reopen して採用 variant 切替を検討する低確率分岐を備える

---

## 2. アーキテクチャ位置づけ

### 2.1 PHOTON-RepoRAG モジュールマップ上の位置

本 Issue が触れるレイヤー:

```
┌─────────────────────────────────────────────────────────┐
│  ユーティリティスクリプト層 (新規変更)                     │
│  ├─ scripts/_setup_ruri_models.py (新規)                 │
│  └─ scripts/build_indexes.py / run_baseline_eval.py /    │
│      ingest_repo.py (既存、変更なし、引数で V2 切替)      │
├─────────────────────────────────────────────────────────┤
│  Config 層                                                │
│  ├─ configs/_experiments/institutional_V2.yaml (新規・   │
│  │    gitignored、variant override)                      │
│  └─ configs/institutional_docs.yaml (コメント L84-87 のみ│
│      更新、embedding.model_id pin は不変 = bge-m3)       │
├─────────────────────────────────────────────────────────┤
│  依存関係層                                                │
│  └─ requirements.txt (huggingface_hub 1 行追加)          │
├─────────────────────────────────────────────────────────┤
│  Indexing 層 (既存、変更なし)                              │
│  └─ baseline_reporag/indexing/embedding.py               │
│     (SentenceTransformer 直接 instantiate、               │
│      variant config から model_id を受け取る)             │
├─────────────────────────────────────────────────────────┤
│  レポート層                                                │
│  └─ reports/institutional_retrieval_ab.md (V2 行更新)    │
└─────────────────────────────────────────────────────────┘
```

### 2.2 既存コード/Test 不変条件 (本 Issue が破ってはいけないもの)

| 不変条件 | 出典 |
|---------|------|
| `configs/institutional_docs.yaml.embedding.model_id == "BAAI/bge-m3"` | `tests/test_pipeline_factory_yaml_invariants.py:81-87` |
| `configs/institutional_docs.yaml.reranker.model_id == "BAAI/bge-reranker-v2-m3"` | 同上 |
| `configs/institutional_docs.yaml.embedding.max_input_chars == 8192` | 同上 |
| `scripts/` 配下は `tests/test_no_scaffolding_in_prod.py:PROD_ROOTS` 対象外 | Issue #139 設計判断 |
| `configs/_experiments/` は `.gitignore` 対象 | `.gitignore:configs/_experiments/` |
| `data/indexes/` / `logs/` は `.gitignore` 対象 | 既存運用 |
| 既存 V0/V1/V3/V4 の `data/indexes/institutional_documents_V<N>/` を破壊しない | repo_id 分離による構造的保証 |

---

## 3. レイヤー別設計

### 3.1 scripts/_setup_ruri_models.py (新規)

#### 責務

ruri 系モデルが必要とするファイル (`spiece.model`) を sentence-transformers / transformers のバージョン違いに依存せず明示的に取得する setup helper。

#### 関数構成

```python
# scripts/_setup_ruri_models.py
"""
ruri 系モデル (spiece.model) を明示的に取得する setup helper。

スコープ: ruri-small-v2 のみ (本 Issue 受入条件参照)
将来拡張: #135 で ruri を tokenizer 候補にする場合は RURI_MODELS list を拡張
"""
from __future__ import annotations
from pathlib import Path
import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError as e:
    raise ImportError(
        "huggingface_hub が必要です。`pip install huggingface_hub` または "
        "requirements.txt 経由で install してください。"
    ) from e

RURI_MODELS = ['cl-nagoya/ruri-small-v2']
RURI_FILES = ['spiece.model']

def fetch_ruri_files() -> None: ...
def verify_load() -> None: ...
```

#### 関数の責務 (DR1-001 反映)

**`fetch_ruri_files() -> None`**:
- 責務: `RURI_MODELS` × `RURI_FILES` の cross product を `hf_hub_download(repo_id, filename)` で取得 (cache hit 時は no-op)。
- 成功条件: 各 `(model_id, file)` ペアの local cache path が返る。
- 失敗時: `HfHubHTTPError` / `OSError` を catch せず raise (fail-fast)。
- 出力: `Path(path).name` と file size のみを出す。local HF cache の full path は home directory / user name を含むため Issue コメントには貼らない (例: `print(f"[OK] fetch: {model_id}:{file} ({Path(path).name}, {Path(path).stat().st_size} bytes)")`)。

**`verify_load() -> None`**:
- 責務: 各 `model_id` について `SentenceTransformer(model_id)` を try で instantiate し、tokenizer / model load が例外なく完了することを検証。
- 成功条件: SentenceTransformer インスタンス生成が完了。tokenizer の round-trip (encode/decode) は本 Issue では検証不要 (build_indexes 側で実装済 path を使うため)。
- 失敗時: `ValueError` (`Couldn't instantiate the backend tokenizer`) / `OSError` を catch せず raise (fail-fast)。上位 (operator/script 呼び出し元) は §3.4 失敗時フォールバック表に基づき Option B/C に分岐する。
- 出力: 成功時 `print(f"[OK] verify_load: {model_id}")`。失敗時の traceback は local での判別根拠に留め、Issue コメントには exception type / sanitized summary のみを記録する (HF token / PAT / full cache path / raw traceback を貼らない)。

#### 設計判断

| 項目 | 判断 |
|------|------|
| 配置 | `scripts/` (運用 helper、CI 自動実行対象外) |
| プレフィックス | `_` 付き (Issue #139 の scaffolding 規約と区別、内部 helper を意図、既存 `_corpus_core.py` / `_grid_search_core.py` と命名整合) |
| Test | unit test なし (network 依存、運用 helper)、`verify_load()` 手動実行 log を Issue コメント記録 |
| `fetch_ruri_files()` エラーハンドリング | `ImportError` (huggingface_hub 不在) と `HfHubHTTPError`/`OSError` (download 失敗) を fail-fast で raise |
| `verify_load()` エラーハンドリング (DR1-002 反映) | SentenceTransformer の `ValueError` (`Couldn't instantiate the backend tokenizer`) / `OSError` を catch せず raise (fail-fast)、上位が §3.4 フォールバック表で Option B/C を判断 |
| `fetch_ruri_files()` 冪等性 (DR1-007 反映) | `hf_hub_download` の cache hit 動作に委譲、cache 一致時は network 不要、再実行可能 |
| `verify_load()` 冪等性 (DR1-007 反映) | SentenceTransformer の HF cache 再利用に委譲、cache 一致時は network 不要 (transformers の AutoConfig/Tokenizer も同様) |

### 3.2 requirements.txt の依存追加

#### 設計判断

```diff
 # Embeddings
 sentence-transformers
 
 # Tokenizer
 transformers
+
+# HF Hub direct file access (Issue #144: spiece.model 個別 download 用)
+huggingface_hub
```

| 項目 | 判断 |
|------|------|
| 追加位置 | `transformers` 直下 (関連グルーピング) |
| バージョン pin | なし (transitive 依存と range 衝突を避ける) |
| 削除条件 | 本 Issue 完了後も保持 (Task 1 script が直接 import) |

### 3.3 configs/_experiments/institutional_V2.yaml (新規・gitignored)

#### 派生方針

`configs/institutional_docs.yaml` を full copy し、以下を override:

| パラメータ | V4 (base) | V2 (override) | 出典 |
|----------|----------|--------------|------|
| `embedding.model_id` | `BAAI/bge-m3` | `cl-nagoya/ruri-small-v2` | 本 Issue |
| `embedding.max_input_chars` | `8192` | `2048` | reports/institutional_retrieval_ab.md:17 |
| `embedding.batch_size` | `32` | `32` | 既存 |
| `embedding.normalize` | `true` | `true` | 既存 baseline (override 不要、cp で継承) — DR2-001 反映 |
| `chunking.max_chars` | `800` | `800` | #137 V0-V3 共通 (configs/institutional_docs.yaml:65) — DR2-003 反映 |
| `chunking.overlap_chars` | `100` | `100` | #137 V0-V3 共通 (configs/institutional_docs.yaml:66) — DR2-003 反映 |
| `reranker.model_id` | `BAAI/bge-reranker-v2-m3` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reports/institutional_retrieval_ab.md:17 |
| `repo.repo_id` | `institutional_documents` | `institutional_documents_V2` | #137 work-plan |
| `repo.repo_path` | (sentinel) | (sentinel と同値、変更なし) | sentinel commit 整合性 |
| `repo.repo_commit` | `9e500539...` | `9e500539...` (同値) | sentinel commit 整合性 |

#### 設計判断

| 項目 | 判断 |
|------|------|
| Copy 元 | `configs/institutional_docs.yaml` (base config) |
| 配置 | `configs/_experiments/` (gitignored、worktree 内 local override) |
| 作成タイミング | Task 2 の最初 (`mkdir -p configs/_experiments` → `cp`) |
| invariant test 影響 | `glob('*.yaml')` は `configs/` 直下のみで `_experiments/` 配下は対象外 |

### 3.4 V2 build/eval 実行手順

#### 実行順序

```bash
# Step 0: 依存追加 (Task 0)
# (前提: 既存 venv が activate 済み。Option B 採用時は §3.4 Option B fallback
#  節の手順で .venv-ruri-v2 を activate してから本 step に戻る。— DR1-005 反映)
# requirements.txt に huggingface_hub を追記
pip install -r requirements.txt

# Step 1: spiece.model 取得 (Task 1)
python scripts/_setup_ruri_models.py
# → fetch_ruri_files() + verify_load() 自動実行

# Step 2: variant config 作成 (Task 2 a)
mkdir -p configs/_experiments
cp configs/institutional_docs.yaml configs/_experiments/institutional_V2.yaml
# Step 2 (続き): 以下のフィールドを override する (DR1-003 反映、§3.3 表参照)。
# yq が利用可能な場合 (推奨):
yq -i '.indexing.embedding.model_id = "cl-nagoya/ruri-small-v2"' \
  configs/_experiments/institutional_V2.yaml
yq -i '.indexing.embedding.max_input_chars = 2048' \
  configs/_experiments/institutional_V2.yaml
yq -i '.retrieval.reranker.model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"' \
  configs/_experiments/institutional_V2.yaml
yq -i '.repo.repo_id = "institutional_documents_V2"' \
  configs/_experiments/institutional_V2.yaml
# yq 不在時は python -c での代替:
#   python -c "import yaml; p='configs/_experiments/institutional_V2.yaml'; \
#     d=yaml.safe_load(open(p)); \
#     d['indexing']['embedding']['model_id']='cl-nagoya/ruri-small-v2'; \
#     d['indexing']['embedding']['max_input_chars']=2048; \
#     d['retrieval']['reranker']['model_id']='cross-encoder/ms-marco-MiniLM-L-6-v2'; \
#     d['repo']['repo_id']='institutional_documents_V2'; \
#     yaml.safe_dump(d, open(p,'w'), allow_unicode=True, sort_keys=False)"
# 手編集の場合は §3.3 表の 4 行 (embedding.model_id, max_input_chars,
# reranker.model_id, repo.repo_id) を漏れなく書き換える (chunking パラメータと
# repo.repo_path / repo.repo_commit は base から不変で OK)。

# Step 2.5: variant config security validation (Task 2 gate)
# gitignored config の typo / tampering / 過大値を build/eval 前に fail-fast で止める。
python - <<'PY'
from baseline_reporag.config import load_config, validate_repo_id

cfg = load_config("configs/_experiments/institutional_V2.yaml")
assert validate_repo_id(cfg.repo.repo_id) == "institutional_documents_V2"
assert cfg.repo.repo_path == "/Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents"
assert cfg.repo.repo_commit == "9e500539f29555364217b773368305e7f59aa026"
assert cfg.indexing.embedding.model_id == "cl-nagoya/ruri-small-v2"
assert cfg.indexing.embedding.max_input_chars == 2048
assert cfg.indexing.embedding.batch_size == 32
assert cfg.indexing.embedding.normalize is True
assert cfg.ingestion.chunking.max_chars == 800
assert cfg.ingestion.chunking.overlap_chars == 100
assert cfg.retrieval.reranker.model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
PY

# Step 3: ingest 再実行 (Task 2 b)
python scripts/ingest_repo.py \
  --repo /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \
  --commit 9e500539f29555364217b773368305e7f59aa026 \
  --config configs/_experiments/institutional_V2.yaml \
  --repo-id institutional_documents_V2

# Step 4: build_indexes (Task 2 c)
python scripts/build_indexes.py \
  --config configs/_experiments/institutional_V2.yaml \
  --repo-id institutional_documents_V2

# Step 5: run_baseline_eval (Task 2 d)
python scripts/run_baseline_eval.py \
  --config configs/_experiments/institutional_V2.yaml \
  --repo-id institutional_documents_V2 \
  --eval-set data/eval_sets/institutional_static_eval.jsonl \
  --output logs/institutional_V2_$(date +%Y%m%d_%H%M%S).jsonl

# Step 6: report + config コメント更新 (Task 3)
# reports/institutional_retrieval_ab.md の V2 行を更新
# configs/institutional_docs.yaml:84-87 コメントを「V2 ruri-small-v2 NC X.XX%」に更新
# (注: 上記更新は YAML コメント (# 行) のみで embedding.model_id 値は変更しない。
#  tests/test_pipeline_factory_yaml_invariants.py:84-107 は
#  cfg.indexing.embedding.model_id 値のみを assert するため、コメント更新は
#  invariant test を破らない。§2.2 不変条件参照。 — DR1-004 反映)
```

#### 失敗時のフォールバック分岐

| 段階 | 失敗パターン | 分岐 |
|------|------------|------|
| Step 1 | `verify_load()` 成功 | → Step 2 へ進む |
| Step 1 | `Couldn't instantiate the backend tokenizer` | → Option B (専用 venv で旧版 install) |
| Step 1 | 上記以外の error | → Option C (上流 issue 報告) で blocked エスカレーション |

#### Option B fallback (低確率分岐)

```bash
python -m venv .venv-ruri-v2
source .venv-ruri-v2/bin/activate
python -m pip install -r requirements.txt "transformers<5.0" "sentence-transformers<5.0"
# .venv-ruri-v2 採用時は .gitignore に `.venv-ruri-*/` を追加 (worktree 衛生)
```

---

## 4. 設計判断とトレードオフ

### 設計判断 #1: 真因の確定方法

**選択肢**:
- A: コード調査 + Web 検索で真因 (cache 漏れ vs 互換性) を確定してから修正
- B: ワークアラウンド優先で適用し、適用後の挙動から真因を逆引き

**決定**: 選択肢 B

**理由**:
- 真因 (Hypothesis H1/H2/H3) はいずれもコードベース照合では Unverifiable
- 上流ライブラリ内部を調査して確定するコストが高い
- Option A (`spiece.model` 単独 download) は両仮説で動作するため、適用後の挙動で判別可能
- 本 Issue は Phase 2 完了 (#116) までの低優先度タスクで、研究投資より工数効率を優先

**トレードオフ**:
- メリット: 短い ETA (~1.5-2h)、ワークアラウンド成功なら真因確定不要
- デメリット: 真因が互換性問題だった場合は Option B fallback コストが上乗せされる
- リスク: Option B も失敗した場合は Option C (上流 PR) に escalation する必要

### 設計判断 #2: variant config の管理方式

**選択肢**:
- A: `configs/institutional_docs.yaml` を直接書き換えて V2 build → 戻す
- B: `configs/_experiments/institutional_V2.yaml` (gitignored) を新規作成して切替
- C: CLI 引数 (`--embedding-model-id`) を `build_indexes.py` 等に追加

**決定**: 選択肢 B (#137 work-plan と同方式)

**理由**:
- 選択肢 A は `tests/test_pipeline_factory_yaml_invariants.py:81-87` の bge-m3 pin invariant test に違反する
- 選択肢 C は scripts CLI の API 拡張になり、既存スクリプトの引数規約変更コストが大きい
- 選択肢 B は #137 5-variant A/B で実証済みの運用慣例で、追加コストは `mkdir -p` のみ

**トレードオフ**:
- メリット: invariant test 不変、既存スクリプト不変、#137 と同じ操作モデル
- デメリット: variant config が gitignored で commit 対象外のため、再現性は手順書 (本書) に依存
- リスク: 手順書を見ないオペレーターが V4 設定で V2 を build してしまう (S1-004 で対応済: 固定パラメータ表を Issue 本文に記載)

### 設計判断 #3: huggingface_hub の依存宣言

**選択肢**:
- A: requirements.txt に明示宣言 (`huggingface_hub`)
- B: transitive 依存に依存 (`sentence-transformers` / `transformers` 経由)
- C: import 時に try/except で fallback ImportError raise のみ

**決定**: 選択肢 A + 選択肢 C のハイブリッド (依存契約の明示 + 親切なエラー)

**理由**:
- 選択肢 B は transitive バージョン range が将来の break-glass で問題化する余地がある
- 選択肢 A 単独だと requirements.txt 管理外環境でハマるリスクあり
- 両方を組み合わせて契約の明示と運用の保険を両立

**トレードオフ**:
- メリット: 依存契約が明示、environment ミスマッチ時に親切な ImportError
- デメリット: 1 行の追加、テスト変更なし、コスト極小
- リスク: なし (huggingface_hub は安定広範に利用される library)

### 設計判断 #4: scripts/_setup_ruri_models.py の test 戦略

**選択肢**:
- A: monkeypatch で `hf_hub_download` を mock した unit test を追加
- B: unit test なし、`verify_load()` 手動実行 log を Issue コメント記録
- C: integration test として weekly_eval CI で実行

**決定**: 選択肢 B

**理由**:
- `scripts/` 配下は `PROD_ROOTS` 対象外で test 必須対象外 (Issue #139 設計判断)
- 本 script は network/HF Hub 依存の運用 helper で、mock test は network エラー時の挙動確認に限定的価値
- weekly_eval CI に組み込むと self-hosted runner の HF cache 状態に依存する fragile な test になる
- `verify_load()` の手動実行 log を Issue コメントで残せば受入根拠として十分

**トレードオフ**:
- メリット: 工数 0、CI フラ生成なし、既存 test 慣例と整合
- デメリット: 自動テストカバレッジなし、運用 helper として手動実行に依存
- リスク: なし (低頻度 helper、手動運用前提)

### 設計判断 #5: V2 採用低確率分岐の影響範囲

**選択肢**:
- A: `configs/institutional_docs.yaml` のみ更新
- B: 上記 + `tests/test_pipeline_factory_yaml_invariants.py` の pin 定数 + `docs/deployment.md` の memory note も更新
- C: 採用判定が出てから別 Issue で対応

**決定**: 選択肢 B (Stage 7 Codex 指摘 S7-003 反映)

**理由**:
- 選択肢 A だけだと invariant test が `BAAI/bge-m3` で固定されているため CI が失敗する
- 選択肢 C は分岐想起の機会が失われ、採用判定後の作業が分散する
- 選択肢 B は受入条件として明記し、採用判定が出た時点で漏れなく対応できる

**トレードオフ**:
- メリット: 低確率分岐の網羅性、CI 失敗回避、deployment 文書の整合性維持
- デメリット: 受入条件文の冗長化
- リスク: なし (条件付き対応で、V2 採用が出ない限り発火しない)

---

## 5. 影響範囲

### 5.1 変更対象ファイル

| ファイル | 変更種別 | 影響度 | 備考 |
|---------|---------|--------|------|
| `scripts/_setup_ruri_models.py` | 新規 | 低 | 運用 helper、scaffolding 規約対象外 |
| `requirements.txt` | 更新 (1 行追加) | 低 | 依存契約の明示 |
| `configs/_experiments/institutional_V2.yaml` | 新規 (gitignored) | 中 | V2 計測に必須、commit 対象外 |
| `data/indexes/institutional_documents_V2/` | 新規 (gitignored) | 中 | ingest/build 結果 (`chunks.db`, `lexical.pkl`, `embedding/{embeddings.npy,chunk_ids.json,model_id.txt,max_input_chars.txt}`)、commit 対象外 |
| `logs/institutional_V2_*.jsonl` | 新規 (gitignored) | 中 | eval predictions、commit 対象外 |
| `logs/bench_variant_*.jsonl` / `logs/sessions/session_eval-*.json` | 新規 (gitignored) | 中 | `run_baseline_eval.py` 経由の pipeline run log / session memory log、commit 対象外 |
| `reports/institutional_retrieval_ab.md` | 更新 | 中 | V2 行の NC rate / latency 追記 |
| `configs/institutional_docs.yaml` | 更新 (コメントのみ L84-87) | 低 | invariant test 不変 |
| `.gitignore` | 条件付き更新 | 低 | Option B 採用時のみ `.venv-ruri-*/` を追加 |

### 5.2 V2 採用低確率分岐 (条件付き、V2 が V4 を上回った場合のみ)

| ファイル | 条件付き変更 | 影響度 |
|---------|------------|--------|
| `configs/institutional_docs.yaml` | embedding.model_id / max_input_chars / reranker.model_id 更新 | 高 |
| `tests/test_pipeline_factory_yaml_invariants.py:43-47` | institutional pin 定数更新 | 高 |
| `docs/deployment.md:13-16` | institutional memory note 更新 | 中 |

### 5.3 不変対象 (本 Issue で変更しない)

- `baseline_reporag/indexing/embedding.py` (SentenceTransformer 直接 instantiate のロジックは V2 でも変更不要)
- `scripts/build_indexes.py` / `run_baseline_eval.py` / `ingest_repo.py` (CLI 引数は既存のまま)
- `.github/workflows/weekly_eval.yml` (workflow file / eval 対象は不変。`pip install -r requirements.txt` は `huggingface_hub` 追加後の依存解決を通るため、CI 影響は install step の直接依存 1 件追加に限定)
- `docs/deployment.md` / `docs/troubleshooting.md` / `docs/tutorial.md` (V2 採用判定が出ない限り不変)

---

## 6. セキュリティ設計

| 脅威 | 対策 | 優先度 |
|------|------|--------|
| HF Hub への悪意ある model_id 指定 | `RURI_MODELS` を hard-coded で固定、CLI 引数化しない | 中 |
| Cache pollution / path traversal | `huggingface_hub` 標準の cache 配置に委譲 | 低 |
| API キー漏洩 (HF private model) | 本 Issue は public model のみ、認証不要 | - |
| Log / Issue コメント経由の secret・local path 漏洩 | `_setup_ruri_models.py` の成功 log は filename + size のみ。失敗時も Issue コメントには sanitized summary のみを貼り、HF token / PAT / full cache path / raw traceback を貼らない | 中 |
| gitignored variant config の typo / tampering / 過大値 | §3.4 Step 2.5 で `load_config` + `validate_repo_id` + 固定値 assert を実行し、`max_input_chars` 等の危険値を build/eval 前に fail-fast で拒否 | 中 |
| 大量 download での disk 圧迫 | `spiece.model` (439 KB) のみで影響軽微 | 低 |
| 依存追加による supply chain 攻撃 | `huggingface_hub` は HuggingFace 公式 package。pin は transitive conflict 回避のため行わないが、`pip check` と resolved version 記録で resolver 結果を監査可能にする | 低 |

---

## 7. 品質基準

| チェック項目 | コマンド | 基準 |
|-------------|----------|------|
| Lint | `ruff check .` | 警告 0 件 |
| Format | `ruff format --check .` | 差分なし |
| Test | `python -m pytest torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v` | 全テストパス (本 Issue で test 追加なし) |
| invariant test | `python -m pytest tests/test_pipeline_factory_yaml_invariants.py -v` | bge-m3 pin 不変、PASS |
| scaffolding test | `python -m pytest tests/test_no_scaffolding_in_prod.py -v` | scripts/_setup_ruri_models.py は対象外、PASS |
| dependency resolver | `python -m pip check && python -m pip show huggingface_hub transformers sentence-transformers` | dependency conflict なし、resolved version を Issue/PR に記録 |
| 手動受入 (Task 1) | `python scripts/_setup_ruri_models.py` | `verify_load()` 成功 log を Issue コメント記録 |
| variant config validation | §3.4 Step 2.5 の Python validation | V2 固定値 / repo_id / repo_commit が期待値と一致し、build/eval 前に PASS |
| 手動受入 (Task 2) | V2 build + eval 完走 | `data/indexes/institutional_documents_V2/` と predictions JSONL (`logs/institutional_V2_*.jsonl`) が生成され、run/session logs (`logs/bench_variant_*.jsonl`, `logs/sessions/session_eval-*.json`) が gitignored のまま残る |
| 手動受入 (Task 3) | `reports/institutional_retrieval_ab.md` | V2 行に NC rate / category breakdown / latency 記載 |

---

## 8. リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|-------|
| Option A 失敗 (互換性問題が真因) | 工数増 (~+1h) | Option B (専用 venv) に分岐、判別フローを Issue に明記済 |
| Option A/B 失敗 | 本 Issue 完了不能 | Option C (上流 PR) に escalation、別 Issue で長期対応 |
| V4 設定で V2 build (キメラ比較) | 報告書に使えないデータ | 固定パラメータ表を Issue 本文に明示済 (Stage 1 S1-004 反映) |
| gitignored V2 config の typo / tampering | DoS または不正確な比較 | §3.4 Step 2.5 の validation gate で固定値を assert してから ingest/build/eval に進む |
| raw traceback / HF cache path の Issue コメント貼付 | local path / secret 漏洩 | §3.1 / §6 に従い sanitized summary のみ記録 |
| `huggingface_hub` 追加時の resolver drift | CI/install 差分、transitive conflict | `pip check` と resolved version 記録を Task 0 の受入条件に含める |
| ingest 再実行で repo_commit ずれ | build/eval 不一致 | `--commit 9e500539...` sentinel 明記済 (Stage 5 S5-001 反映) |
| `.venv-ruri-v2` の誤 commit | worktree 汚染 | `.gitignore` に `.venv-ruri-*/` 追加方針を Option B 採用条件に明記 (S7-002 反映) |
| V2 採用で invariant test 失敗 | CI failure | 採用低確率分岐に invariant test 更新を含めた (S7-003 反映) |

---

## 9. 完了条件

(各 Task の影響範囲詳細は §5.1 / 設計詳細は §3 を相互参照 — DR1-006 反映)

- [ ] Task 0: `requirements.txt` に `huggingface_hub` を追加。Option B 採用時のみ `.gitignore` に `.venv-ruri-*/` を追加。`python -m pip check` と resolved version 記録を完了 (§5.1 行 2・9 / §3.2 / §3.4 Option B / §7)
- [ ] Task 1: `scripts/_setup_ruri_models.py` 作成 + 手動 `verify_load()` 成功 log を Issue コメント記録。Issue コメントには sanitized summary のみを貼り、HF token / PAT / full cache path / raw traceback を含めない (§5.1 行 1 / §3.1 / §6)
- [ ] Task 2: §3.4 Step 2.5 の variant config validation を PASS させた後、V2 build + eval 完走、`data/indexes/institutional_documents_V2/` と `logs/institutional_V2_*.jsonl` (+ run/session logs) を生成 (§5.1 行 3-6 / §3.3-§3.4)
- [ ] Task 3: `reports/institutional_retrieval_ab.md` に V2 結果反映 (§5.1 行 7 / §3.4 Step 6)
- [ ] Task 3 (追加): `configs/institutional_docs.yaml:84-87` コメント更新 (§5.1 行 8 / §3.4 Step 6 注記)
- [ ] Lint / Format / Test 全パス (§7)
- [ ] Option A 失敗時 (該当時のみ): §3.4 失敗時フォールバック表に従い Option B/C を選択し、判別経過 (sanitized traceback summary / 採用 Option / 後続 step の判断) を Issue にコメント記録 (§3.4 / §8 — DR2-002 反映)
- [ ] V4 採用判定不変の確認: V2 結果の overall NC が V4 NC 6.03% を上回らないことを `reports/institutional_retrieval_ab.md` で明示 (§1.3 / §5.2 — DR2-002 反映)
- [ ] V2 採用低確率分岐 (条件付き、V2 NC < V4 NC の場合のみ): invariant test + deployment docs 更新 (§5.2)

---

## 10. 関連 Issue / PR / Document

- 元 Issue: #137 (5-variant A/B、V2 のみ計測不能)
- マージ済 PR: #142 (V4 採用、本 Issue は別 PR で V2 補完)
- 関連 Issue: #135 (PHOTON 再学習、ruri を tokenizer 候補にする可能性)
- レビュー成果物: `workspace/issues/144/issue-review/summary-report.md`
- 仮説検証: `workspace/issues/144/issue-review/hypothesis-verification.md`
- Issue 本文 (最新): `workspace/issues/144/design/latest-issue-body.md`
