## 背景

#135 (S7-001 解消の commit `2dbf458`) で発見した「random-init weight が production で動いていた」事象は **構造的な test ギャップ** から発生した:

- 既存 `baseline_reporag/tests/test_photon_pipeline.py` (約 4,500 行) が `MagicMock` 中心で、**実 PhotonModel + 実 weight** を通る経路が一度も走っていなかった
- `_StubTokenizer` のような scaffolding 命名のシンボルが production code path に残存
- 「設計 Must Fix が実装に反映されているか」を CI で固定する仕組みがなかった

これは S7-001 単体ではなく、**同型の silent bug が他にも潜む可能性** を示唆。本 Issue で **systematic に audit + test 補強** を行う。

> **⚠ 前提状況 (2026-04-26 時点)**: #135 (S7-001 fix の commit `2dbf458`) は `feature/issue-135-photon-retrain` 上のみで **main にはマージされていない**。現行 main の `baseline_reporag/photon_pipeline.py` には `checkpoint_path` 設定も `load_checkpoint` 呼び出しも未導入。
> このため、real-weight integration test (旧 Task 2) は **#135 マージ後に着手** とし、本 Issue からは **切り出して別 Issue 化** する (#145)。

## ゴール

1. **scaffolding pattern を production path から排除** — `_Stub*`, `_Mock*`, `_Dummy*`, `_Placeholder*` 等の命名を持つシンボルが production code path (`baseline_reporag/`, `photon_mlx/`, `torch_ref/`) に到達しないことを CI で固定
2. **設定値が実際に読まれているか** の audit — `getattr(cfg, "X", default)` パターンで「常に default が選ばれている」箇所を grep で抽出 + production 必須フィールドを invariant test で固定

> 旧 Goal 2 (real-weight integration test) は #135 依存のため #145 に切り出し済み (S1-001 / S1-008 反映)。

## スコープ

| Task | 概要 | 状態 | 備考 |
|------|------|------|------|
| Task 1 | Scaffolding 命名 audit + 境界 test | **本 Issue** | #138 マージ済みのため即着手可 |
| Task 3 | `getattr default` audit + invariant test 拡張 (Phase A: 2件のみ) | **本 Issue** | 同上 |
| Task 2 | real-weight integration test | **#145 へ切出し** | #135 マージ後に着手 |

> 対象 dir: `baseline_reporag/`, `photon_mlx/`, `torch_ref/`。
> `bench/`, `scripts/`, `demo/` は研究 harness / CLI utility で **production runtime path 外** のため対象外 (S3-008 反映)。

## 変更内容

### Task 1: Scaffolding 命名 audit + 境界 test

**Step 1: codebase audit**

```bash
grep -rEn '_Stub|_Mock|_Dummy|_Placeholder' \
  baseline_reporag/ photon_mlx/ torch_ref/ --include='*.py' \
  | grep -v '/tests/' | grep -v "# noqa"
```

(`-E` で alternation を ERE 化、`/tests/` パスで test 配下を除外。)

**Step 2: 統一方針 (S1-005 反映)**

`_StubTokenizer` / `_get_stub_tokenizer` 等の test/dev only scaffolding を production module から **完全削除** し、test fixture へ移設する。production 側は以下の不変条件を守る:

- `cfg.tokenizer.tokenizer_id` を **必須化** (`provider == "photon"` のとき)
- `_build_photon_deps` で tokenizer_id 未設定または load 失敗時は **`ValueError` を raise** (現状の warning + fallback を撤去)
- tokenizer load 失敗 (`AutoTokenizer.from_pretrained` 例外 / vocab mismatch 等) は `_build_photon_deps` 境界で `ValueError` として観測できるようにし、対象 tokenizer_id を含む message に正規化する (S5-002 反映)
- これにより production code path から `_Stub*` シンボルへの到達経路を消す

> 「rename して残す」案は採用しない: production 流路に test fixture が到達することそのものを構造的に塞ぐ方針を採る。

**Step 3: 境界 test 追加 (S3-003 / S3-007 反映で実装サンプル修正済み)**

新規ファイル `tests/test_no_scaffolding_in_prod.py`:

```python
# tests/test_no_scaffolding_in_prod.py (新規)
import re
from pathlib import Path

# 意図: `_Stub`, `_Mock`, `_Dummy`, `_Placeholder` で始まる識別子全般を捕捉する。
# 単一の `\b_Stub` だと `_Stub` 単体しかマッチせず `_StubTokenizer` を見逃すため、
# `\w*` で識別子末尾までを許容する。
FORBIDDEN_PATTERN = re.compile(r'\b_(?:Stub|Mock|Dummy|Placeholder)\w*')

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_ROOTS = [
    REPO_ROOT / 'baseline_reporag',
    REPO_ROOT / 'photon_mlx',
    REPO_ROOT / 'torch_ref',
]

# tuple membership で完全一致除外する。リポジトリの test dir は **複数形** `tests/` なので
# `'test' in f.parts` (単数) ではなく `'tests' in f.parts` (複数) を使う必要がある。
EXCLUDED_DIR_PARTS = {'tests', '__pycache__'}


def _is_excluded(path: Path) -> bool:
    return bool(EXCLUDED_DIR_PARTS.intersection(path.parts))


def test_no_scaffolding_naming_in_production():
    violations: list[tuple[str, str]] = []
    for root in PROD_ROOTS:
        assert root.exists(), f'production root not found: {root}'
        for f in root.rglob('*.py'):
            if _is_excluded(f):
                continue
            content = f.read_text(encoding='utf-8')
            for match in FORBIDDEN_PATTERN.finditer(content):
                violations.append((str(f), match.group(0)))
    assert not violations, f"Scaffolding naming in production: {violations}"
```

> S1-004 / S3-007 反映: regex は `\b_(?:Stub|Mock|Dummy|Placeholder)\w*` を採用。`r'_Stub\b'` だと `_StubTokenizer` を捕捉できないため誤り (検証済み: `re.search(r'_Stub\b', '_StubTokenizer')` → None)。
> S3-003 反映: path 除外は `'tests' in f.parts` (複数形 + tuple membership)。`'test' in f.parts` は `('baseline_reporag', 'tests', ...)` に対し False となり、test files まで scan して即 FAIL するため誤り。
> S7-001 反映: `Path('baseline_reporag')` のような cwd 依存 path は使わず、`Path(__file__).resolve().parents[1]` から repo root を解決する。root が見つからない場合は skip せず fail させ、対象 0 件での偽 pass を防ぐ。

### Task 3: "getattr default" audit + invariant test 拡張

**Step 1: 抽出 (現状 8 件確認済み)**

`baseline_reporag/` / `photon_mlx/` 配下 (test 除く) の `getattr(cfg.*, "...", default)` 8 件 (`workspace/issues/139/issue-review/hypothesis-verification.md` Claim 5 参照):

1. `baseline_reporag/photon_pipeline.py:242` — `getattr(cfg, "session_memory", None)`
2. `baseline_reporag/photon_pipeline.py:283` — `head_dim=getattr(cfg.model, "head_dim", 64)`
3. `baseline_reporag/photon_pipeline.py:284` — `max_position_embeddings=getattr(cfg.model, "max_position_embeddings", 2048)`
4. `baseline_reporag/photon_pipeline.py:285` — `rope_theta=getattr(cfg.model, "rope_theta", 1_000_000.0)`
5. `baseline_reporag/photon_pipeline.py:349` — `safe_recgen_enabled = getattr(cfg.get("inference"), "safe_recgen_enabled", True)`
6. `baseline_reporag/pipeline_factory.py:52` — `provider = getattr(cfg.model, "provider", None) or "baseline"`
7. `baseline_reporag/pipeline.py:212` — `answering_cfg = getattr(cfg, "answering", None)`
8. `baseline_reporag/photon_pipeline.py:1109` — `answering_cfg = getattr(cfg, "answering", None)` (重複)

**Step 2: 必須化方針 (S1-006 / S3-001 / S3-002 反映 / Phase A 最小スコープ)**

本 Issue では以下 **2 件のみ** を invariant test で必須化対象とする:

- `tokenizer.vocab_size` (canonical key。`baseline_reporag/photon_pipeline.py:295-314` で「`tokenizer.vocab_size` is the canonical source」と明記済み。`model.vocab_size` は legacy fallback で全 yaml で未設定)
- `tokenizer.tokenizer_id` (Task 1 で必須化される値を yaml 側でも保証)

**対象 yaml の絞り込み (S3-002 / S5-001 反映)**:
- `cfg.model.provider == "photon"` の yaml に加え、PHOTON training/generation profile として使われる `configs/photon_*.yaml` と `configs/institutional_docs_photon.yaml` を対象とする
- 現行 `configs/photon_tiny.yaml` / `configs/photon_600m_paper.yaml` は `model.provider` 未設定だが PHOTON profile であるため、provider 判定だけだと invariant test から漏れる
- `configs/baseline.yaml` (provider=mlx_lm) と `configs/eval.yaml` (benchmark runner config、`model:` block 自体なし) は **対象外**
- 全件 glob ではなく、yaml load → `provider == "photon"` または profile filename 判定 → PHOTON profile のみ assert する形に変更

残り 6 件 (`head_dim`, `max_position_embeddings`, `rope_theta`, `safe_recgen_enabled`, `provider`, `session_memory`/`answering` 系) は default の意図確認のみ行い、必須化は **別 Issue 化 (Phase B)** とする。

> S1-007 反映: 旧サンプルの `model.checkpoint_path` は #135 マージ前のため required から外す。#135 マージ後の Phase B 同タイミングで invariant 化を再検討する。

**Step 3: invariant test 拡張 (実装サンプル修正済み)**

```python
# tests/test_pipeline_factory_yaml_invariants.py に追加
from pathlib import Path
import yaml


def _load_yaml(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _get_nested(cfg: dict, dotted_key: str):
    cur = cfg
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _is_photon_profile_yaml(path: Path, cfg: dict) -> bool:
    return (
        _get_nested(cfg, 'model.provider') == 'photon'
        or path.name.startswith('photon_')
        or path.name == 'institutional_docs_photon.yaml'
    )


def test_photon_yaml_has_required_tokenizer_fields():
    """PHOTON profile yaml は tokenizer.vocab_size / tokenizer.tokenizer_id を必須とする。"""
    required_keys = ['tokenizer.vocab_size', 'tokenizer.tokenizer_id']
    failures: list[tuple[str, str]] = []
    # 既存 file の CONFIGS_DIR を使う。Path('configs') だと pytest cwd が repo root
    # 以外の場合に対象 0 件で pass し得るため、絶対 path 基準にする。
    for yaml_path in sorted(CONFIGS_DIR.glob('*.yaml')):
        cfg = _load_yaml(yaml_path)
        if not _is_photon_profile_yaml(yaml_path, cfg):
            continue
        for key in required_keys:
            value = _get_nested(cfg, key)
            if value in (None, ''):
                failures.append((str(yaml_path), key))
    assert not failures, f"photon yaml missing required tokenizer fields: {failures}"
```

> 既存 `tests/test_pipeline_factory_yaml_invariants.py` は `reranker.model_id` の不変宣言で先行例があるため、それに揃える。

## 受入条件

- [ ] Task 1: production 配下 (`baseline_reporag/`, `photon_mlx/`, `torch_ref/`) に `_Stub*`/`_Mock*`/`_Dummy*`/`_Placeholder*` 命名のシンボルが存在しない (`tests/test_no_scaffolding_in_prod.py` pass、`'tests'` 複数形での除外と word boundary 仕様は本 Issue 記載通り)
- [ ] Task 1: `tests/test_no_scaffolding_in_prod.py` は repo root を `Path(__file__).resolve().parents[1]` で解決し、pytest cwd が repo root 以外でも対象 0 件で偽 pass しない
- [ ] Task 1: `_build_photon_deps` で tokenizer_id 未設定時 `ValueError` を raise する (新規 test で検証)
- [ ] Task 1: tokenizer_id 設定済みでも tokenizer load 失敗時は `_build_photon_deps` 境界で `ValueError` を raise する (e.g. `AutoTokenizer.from_pretrained` を `OSError` に patch した新規 test で検証)
- [ ] Task 1 (S3-004 反映): `baseline_reporag/tests/test_photon_pipeline.py` 既存 test の migration 完了:
  - line 521 周辺 `from baseline_reporag.photon_pipeline import _StubTokenizer` を削除
  - line 563 周辺 `not isinstance(deps["tokenizer"], _StubTokenizer)` 等を削除または等価な assertion に書換 (e.g. `not isinstance(..., MagicMock)`)
  - `test_falls_back_to_stub_when_tokenizer_id_missing` (line 604-636) を削除し、`test_raises_when_tokenizer_id_missing` で置換
- [ ] Task 3: `tests/test_pipeline_factory_yaml_invariants.py` に `tokenizer.vocab_size` / `tokenizer.tokenizer_id` の存在チェックを追加し、**PHOTON profile yaml** (`provider == "photon"` または `configs/photon_*.yaml` / `configs/institutional_docs_photon.yaml`) のみを対象として全件 pass
- [ ] 既存テスト + 新規テスト全パス (`python -m pytest` 全 collected test pass、CLAUDE.md 既知の pre-existing failure 2 件は除外可)
- [ ] `ruff check .` 警告 0 件、`ruff format --check .` 差分なし
- [ ] (out-of-scope) Task 2 (real-weight integration test) は #145 で追跡、本 Issue の受入条件には含めない

## PR 戦略 (S1-008 / S3-005 反映)

- 本 Issue 単一 PR で Task 1 + Task 3 をまとめてマージ可能 (両者 #138 マージ済みの上で独立)。
- **#135 との merge 順序**: 本 Issue (#139) を **先に main へ merge** することを推奨 (より小さい変更で本格再学習 #135 を阻害しない)。`feature/issue-135-photon-retrain` は本 Issue マージ後に rebase が必要 (両者は `_build_photon_deps` の近接行を編集する S3-005 参照)。
- **#135 現時点の衝突状況 (S7-003 反映)**: ローカル `feature/issue-135-photon-retrain` には `2dbf458` 後の commit (`994ba29` まで) が積まれており、`git merge-tree $(git merge-base HEAD feature/issue-135-photon-retrain) HEAD feature/issue-135-photon-retrain` で `baseline_reporag/photon_pipeline.py` / `baseline_reporag/tests/test_photon_pipeline.py` に conflict marker が出る。#135 rebase 時は #139 側の real tokenizer / stub removal を維持し、その後に #135 側の checkpoint load (`photon_mlx.checkpoint.load_checkpoint`) を `PhotonModel` 構築後へ再適用する。
- Task 2 (#145) は #135 マージを待ち、独立 PR として進行。

## 並列性

#135 (本格再学習) と並列で実装可能。本 Issue は CPU only でメモリ要件も小さい。

## 影響ファイル (S3-004 / S3-006 反映で migration 対象を明示)

- `baseline_reporag/photon_pipeline.py`
  - `_StubTokenizer` クラス本体を削除
  - `_get_stub_tokenizer` 関数を削除
  - `_build_photon_deps` を改修: `cfg.tokenizer.tokenizer_id` 未設定 / load 失敗時に `ValueError` を raise (現状の warning + fallback 経路を撤去)
  - tokenizer load 失敗の exception type を `ValueError` に正規化し、test で固定
- `baseline_reporag/tests/test_photon_pipeline.py`
  - `_StubTokenizer` import / isinstance 参照を削除 or 等価書換 (line 521, 563 周辺)
  - `test_falls_back_to_stub_when_tokenizer_id_missing` (line 604-636) を削除
  - `test_raises_when_tokenizer_id_missing` を新設
- `tests/test_no_scaffolding_in_prod.py` (新規)
- `tests/test_pipeline_factory_yaml_invariants.py` (拡張: `tokenizer.vocab_size` + `tokenizer.tokenizer_id` 存在チェック、PHOTON profile yaml に絞る)
- `docs/troubleshooting.md` (S3-006 / S7-002 反映: `cfg.model.provider == 'photon'` の起動失敗 checklist (line 149 周辺) に「`tokenizer.tokenizer_id` 未設定 → `ValueError`」と「tokenizer load 失敗 (HF Hub 障害 / gated model / 未 cache) → server/CLI 起動失敗。事前 cache、HF login、ネットワーク疎通、対象 tokenizer_id の確認」を追記)
- 必要なら `configs/*.yaml` (`provider == "photon"` のもの) に `tokenizer.vocab_size` / `tokenizer.tokenizer_id` を補完

> 備考 (S3-010 反映): `.github/workflows/weekly_eval.yml` および CLAUDE.md 疎通基準コマンドは `configs/baseline.yaml` (provider=mlx_lm) を使用するため、`_build_photon_deps` の raise 化は runtime に影響しない。
> CI コスト (S3-009): 新規 test 2 件は yaml 9 件 + py 約 100 file の walk で <1 秒、PR ごとの追加コストは無視可。

## 関連

- 元: S7-001 (#135 commit `2dbf458`) — random-init bug
- 緊急: #138 (tokenizer mismatch、本 Issue より先に解消) — マージ済み (#141)
- 並列可: #135 (本格再学習) — #138 解消後に並列実施。**merge 順は #139 → #135** (#135 側が rebase する)
- **切り出し先**: #145 (real-weight integration test、#135 マージ後に着手)

## レビュー履歴

- Stage 1 (通常レビュー / 2026-04-26): Must Fix 3 / Should Fix 5 / Nice to Have 2 → 反映済み
  - workspace/issues/139/issue-review/stage1-review-result.json
- Stage 3 (影響範囲レビュー / 2026-04-26): Must Fix 4 / Should Fix 3 / Nice to Have 3 → 反映済み
  - workspace/issues/139/issue-review/stage3-review-result.json
- Stage 5 (通常レビュー 2回目 / Codex / 2026-04-26): Must Fix 0 / Should Fix 2 / Nice to Have 0 → 反映済み
  - workspace/issues/139/issue-review/stage5-review-result.json
- Stage 7 (影響範囲レビュー 2回目 / Codex / 2026-04-26): Must Fix 0 / Should Fix 3 / Nice to Have 0 → 反映済み
  - workspace/issues/139/issue-review/stage7-review-result.json
- 仮説検証レポート: workspace/issues/139/issue-review/hypothesis-verification.md
