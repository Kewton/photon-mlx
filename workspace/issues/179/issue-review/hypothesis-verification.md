# Issue #179 仮説検証レポート

## 対象 Issue
- Issue #179: feat(ui): parallel baseline/PHOTON comparison mode in chat (G7)

## 仮説・前提条件の一覧

### H1: `scripts/compare_baseline_photon.py::run_variant` が共通化可能
- **主張**: PR #168 で追加した `run_variant` 関数があり、モジュール化すれば UI から呼べる

**検証結果**: **Confirmed**
- `scripts/compare_baseline_photon.py` に `run_variant(variant_id, config_path, question, repo_id, session_id)` が実装済み (L53-84)
- `VariantResult` dataclass も定義済み (L43-52)
- `build_pipeline` / `override_repo_for_pipeline` を使っており、そのまま `baseline_reporag/comparison.py` に移植可能

### H2: baseline pipeline と PHOTON pipeline をそれぞれ `st.session_state` に cache できる
- **主張**: 重複構築を回避するため、各 pipeline を session_state にキャッシュする

**検証結果**: **Confirmed**
- `app/photon_app.py` に既に `_pipeline_cache_key(project_name, config_path)` (L1747) と pipeline cache ロジック (L1800-1827) が存在する
- 比較モードでは baseline/photon それぞれのキーで同様にキャッシュ可能

### H3: `app/photon_app.py:page_chat` に比較モードトグルを追加できる
- **主張**: 既存の `page_chat` に `st.toggle("⚖ 比較モード")` を追加し 2 column 表示に切り替え

**検証結果**: **Confirmed**
- `page_chat()` は L1550 から始まり、input/send ロジック・history 管理が明確に分離されている
- `st.columns(2)` / `st.columns([5,1])` は既存箇所で多数使用されており、2 column 追加は容易

### H4: session memory は variant 毎に分離する必要がある
- **主張**: baseline/PHOTON それぞれで session_key を分けることで前ターンの文脈が混ざらない

**検証結果**: **Confirmed**
- `session_key = f"chat_{project_name}"` がベースとなっており (L1555)、
  `f"chat_{project_name}_baseline"` / `f"chat_{project_name}_photon"` のように suffix で分離可能
- `_run_query` に `session_key` が渡され pipeline.query の session_id になる (L1832)

### H5: drift パネルは既存と同じロジックを PHOTON 列に再利用できる
- **主張**: 既存の `_drift_panel.format_drift_panel(...)` 呼び出しをそのまま PHOTON 列に適用

**検証結果**: **Confirmed**
- `_drift_panel.format_drift_panel` は `dm_dict` (dict or None) を受け取るシンプルなインタフェース (L1638-1662)
- metadata から `drift_metrics` を取り出すロジックをそのまま PHOTON 列に移植可能

## 申し送り事項（Rejected / Partially Confirmed 仮説）

なし（全仮説 Confirmed）。

## 追加確認事項（Issue に記載なし、コードベース照合で判明）

1. **`baseline_reporag/comparison.py` は未存在**: 新規作成が必要
2. **`tests/test_comparison.py` は未存在**: 新規作成が必要
3. **Issue が想定する `configs/photon_small.yaml`**: `configs/` 配下に存在するか別途確認が必要
4. **parallel 実行**: Issue では「並列実行」を謳っているが `asyncio`/`threading` の選択は未決定。Streamlit の `st.spinner` は blocking なので `threading.Thread` または `concurrent.futures.ThreadPoolExecutor` を検討すべき
